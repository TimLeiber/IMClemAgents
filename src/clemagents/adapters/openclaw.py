from .traces.openclaw import parse_openclaw_agent_trace
from . import model_connection as connections
from .utils import warn_model_generation_config
from .tls import configure_tls, require_resolved_tls
import json
import os
import re
import secrets
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from clemagents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemagents.adapters.openai_compatible_proxy import proxy_for_model_connection
from clemagents.adapters.utils import (GAME_MCP_SERVER_NAME, deep_merge_dicts, load_model_connection, mcp_environment,
                                       model_connection_environment, new_game_completion_path, read_game_completion,
                                       redact_sensitive, resolve_runtime_model, run_process_until_game_complete,
                                       temporary_environment, write_text_artifact)

SESSION_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
DUCKDUCKGO_PLUGIN_PATH = Path("/opt/openclaw-plugins/duckduckgo")
GAME_TOOL_MARKERS = ("mcp__game__", "mcp_game_", "game.start_game", "game.submit_response", "mcp__clem-game__",
                     "mcp_clem_game_", "clem_game.start_game", "clem_game.submit_response", '"start_game"',
                     '"submit_response"',
                     )


def _validate_openclaw_model_connection(connection: dict[str, Any] | None) -> None:
    """Validate the native OpenRouter model and credential expected by OpenClaw."""
    if not connection or connection.get("backend") != "openrouter":
        return

    runtime_model = connection.get("model")
    if not isinstance(runtime_model, str) or not runtime_model.startswith("openrouter/"):
        raise ValueError("OpenClaw OpenRouter connections require a canonical "
                         f"openrouter/<provider>/<model> reference, got {runtime_model!r}.")

    environment = connection.get("env")
    if not isinstance(environment, dict) or not environment.get("OPENROUTER_API_KEY"):
        raise ValueError("OpenClaw OpenRouter connections require OPENROUTER_API_KEY.")


def _error_detail(result: subprocess.CompletedProcess[str]) -> str | None:
    """Return a short, redacted error from a failed OpenClaw command."""
    try:
        payload = json.loads(result.stdout)
        detail = payload.get("error", {}).get("message")
        if isinstance(detail, str) and detail.strip():
            return redact_sensitive(detail)
    except (ValueError, AttributeError):
        pass
    lines = [line.strip() for output in (result.stderr, result.stdout) for line in output.splitlines() if line.strip()]
    return redact_sensitive(lines[-1]) if lines else None


@contextmanager
def _openclaw_gateway(base_command: list[str], log_path: Path, startup_timeout: float = 45):
    """Keep the native Gateway alive for the episode and stop it before export."""
    command = base_command + ["gateway", "run"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"OpenClaw Gateway exited during startup ({process.returncode}); "
                                       f"see {log_path.name}")
                try:
                    with urlopen("http://127.0.0.1:18789/startupz", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError(f"OpenClaw Gateway was not ready after {startup_timeout}s; "
                                   f"see {log_path.name}")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class OpenClawHarness(ExternalAgentHarness):
    """Run the native OpenClaw CLI with an isolated profile and Clem MCP tools."""

    @classmethod
    def resolve_model_connection(cls, model_spec: dict[str, Any], agent_config: dict[str, Any]) -> dict[str, Any]:
        warn_model_generation_config(model_spec)
        return configure_tls(_resolve_model_connection(model_spec), agent_config)

    def __init__(self, model: str | None = None, clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp", thinking: str | None = None,
                 verbose: bool = False, profile: str = "game-agent", yolo: bool = True, debug: bool = False,
                 reasoning_effort: str | None = None, model_connection_path: str | None = None,
                 trace_model_io: bool = True, temperature: float | None = None, ca_bundle: str | None = None,
                 verify_tls: bool | None = None):
        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.thinking = reasoning_effort if reasoning_effort is not None else thinking
        self.reasoning_effort = self.thinking
        self.temperature = temperature
        self.verbose = verbose
        self.profile = profile
        self.yolo = yolo
        self.debug = debug
        self.trace_model_io = trace_model_io
        self._model_connection = load_model_connection("openclaw", model_connection_path)
        require_resolved_tls(self._model_connection, ca_bundle, verify_tls)
        _validate_openclaw_model_connection(self._model_connection)

    @classmethod
    def parse_agent_trace(cls, episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Delegate OpenClaw-specific trace parsing to adapter utilities."""

        return parse_openclaw_agent_trace(episode_dir=episode_dir, metadata=metadata)

    def run_episode(self, instruction: str, output_dir: Path | str | None = None) -> AgentRunResult:
        completion_path = new_game_completion_path()
        proxy = proxy_for_model_connection(self._model_connection, completion_path, include_openrouter=True,
                                           trace_responses=self.trace_model_io, trace_requests=self.trace_model_io)

        if proxy is not None:
            with proxy:
                result = self._run_episode(instruction, output_dir, completion_path, proxy.base_url)
                proxy_trace = redact_sensitive(proxy.captured_trace())
                adapter_messages = result.artifacts.get("adapter_messages")

                if proxy_trace and adapter_messages is not None:
                    trace_path = Path(adapter_messages)
                    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n" + proxy_trace, encoding="utf-8")

                return result

        return self._run_episode(instruction, output_dir, completion_path, None)

    def _run_episode(self, instruction: str, output_dir: Path | str | None, completion_path: Path,
                     proxied_base_url: str | None) -> AgentRunResult:
        artifacts: dict[str, Path | str | int | float | bool | None] = {}
        metadata: dict[str, Any] = {"adapter": "openclaw",
                                    "reasoning_effort": self.reasoning_effort,
                                    "temperature": self.temperature,
                                    "resolved_backend": (self._model_connection or {}).get("backend"),
                                    "gateway_base_url": (self._model_connection or {}).get("base_url"),
                                    "compatibility_proxy_base_url": proxied_base_url,
                                    "tool_choice": (self._model_connection or {}).get("tool_choice"),
                                    "verify_tls": (self._model_connection or {}).get("verify_tls"),
                                    "trace_model_io": self.trace_model_io,
                                    "success": False,
                                    "returncode": None,
                                    "runtime_error": None,
                                    "game_completed": False,
                                    "terminated_after_game": False}

        try:
            runtime_model = resolve_runtime_model(model_connection=self._model_connection, model=self.model,
                                                  harness_name="OpenClawHarness")
            runtime_model = ((self._model_connection or {}).get("runtime_model") or runtime_model)
        except Exception as error:
            metadata["runtime_error"] = str(error)
            return AgentRunResult(False, artifacts, metadata)

        run_dir = (Path(output_dir) if output_dir is not None else Path("/tmp") / f"openclaw-episode-{os.getpid()}")
        run_dir.mkdir(parents=True, exist_ok=True)
        home_dir = run_dir / "openclaw_home"
        home_dir.mkdir(parents=True, exist_ok=True)
        instruction_path = run_dir / "openclaw_instruction.txt"
        instruction_path.write_text(instruction, encoding="utf-8")

        session_parts = ("game-session", os.environ.get("GAME_EXPERIMENT",
                                                        "experiment"), os.environ.get("GAME_INSTANCE_ID",
                                                                                      "game"), str(os.getpid()),
                         )
        session_key = "agent:main:" + "-".join(SESSION_PART_PATTERN.sub("-", part).strip("-") or "x" for part in session_parts)

        runtime_environment = model_connection_environment(self._model_connection)
        runtime_environment["HOME"] = str(home_dir)
        runtime_environment["OPENCLAW_GATEWAY_TOKEN"] = secrets.token_urlsafe(32)
        base_command = ["openclaw", "--no-color", "--profile", self.profile]
        gateway_log_path = run_dir / "openclaw_gateway.txt"
        plugin_path = DUCKDUCKGO_PLUGIN_PATH
        if not (plugin_path / "openclaw.plugin.json").is_file():
            metadata["runtime_error"] = ("OpenClaw's native DuckDuckGo plugin is missing; rebuild "
                                         "clem-agent-sandbox:dev using the updated Dockerfile")
            return AgentRunResult(False, artifacts, metadata)

        bridge_environment = mcp_environment(self.mcp_url)
        bridge_environment["GAME_COMPLETION_PATH"] = str(completion_path)
        mcp_command = base_command + ["mcp", "add", GAME_MCP_SERVER_NAME, "--command", "python", "--arg", "-m", "--arg",
                                      "clemagents.mcp.bridge",
                                      "--cwd", "/opt/clemagents/src", "--include", "start_game,submit_response", "--no-probe"]
        for name, value in sorted(bridge_environment.items()):
            mcp_command.extend(["--env", f"{name}={value}"])

        config = (self._model_connection or {}).get("openclaw_config_patch", {})
        config = dict(config) if isinstance(config, dict) else {}

        if self.temperature is not None:
            model_params = {runtime_model: {"params": {"temperature": self.temperature}}}
            config = deep_merge_dicts(config, {"agents": {"defaults": {"models": model_params}}})

        if proxied_base_url is not None:
            provider_name = (self._model_connection or {}).get("openclaw_provider")

            if provider_name:
                config = deep_merge_dicts(config, {"models": {"providers": {str(provider_name): {"baseUrl": proxied_base_url}}}})
        # retain native tools and the openrouter model catalog in yolo mode
        config = deep_merge_dicts(config, {"plugins": {"enabled": True,
                                                       "allow": ["openrouter", "duckduckgo", "browser"],
                                                       # search uses the installed duckduckgo provider
                                                       # an openrouter key otherwise triggers an unrelated npm install
                                                       "entries": {"perplexity": {"enabled": False}}},
                                           "browser": {"enabled": True,
                                                       "headless": True,
                                                       "noSandbox": True,
                                                       "executablePath": "/usr/bin/chromium",
                                                       "defaultProfile": "openclaw"},
                                           "tools": {"web": {"search": {"enabled": True,
                                                                        "provider": "duckduckgo"}}},
                                           "gateway": {"mode": "local",
                                                       "bind": "loopback",
                                                       "port": 18789,
                                                       "auth": {"mode": "token",
                                                                "token": "${OPENCLAW_GATEWAY_TOKEN}"}},
                                           # initialize the gateway catalog with the selected model too
                                           # disable independent background model turns
                                           "agents": {"defaults": {"model": {"primary": runtime_model},
                                                                    "heartbeat": {"every": "0m"}}}})
        if self.yolo:
            config = deep_merge_dicts(config, {"tools": {"exec": {"security": "full", "ask": "off"}}})

        agent_command = base_command + [
            # the gateway owns this profile's state and runs the agent. an
            # embedded --local agent would conflict with its exclusive lock
            "agent", "--session-key", session_key, "--model", runtime_model, "--message-file",
            str(instruction_path), "--timeout", "0", "--verbose", "on" if self.debug else "off"]
        if self.trace_model_io:
            agent_command.append("--json")
        if self.thinking is not None:
            agent_command.extend(["--thinking", self.thinking])

        commands = (
            # register the image's pinned official package in this isolated profile
            # --force acknowledges local-source installation; no download or patch
            ("plugin install",
             base_command + ["plugins", "install", "--link", "--force", "--accept-capabilities",
                             str(plugin_path)], None, 60),
            ("mcp add", mcp_command, None, 60), ("config patch", base_command + ["config", "patch", "--stdin"],
                                                 json.dumps(config), 60), ("agent", agent_command, None, None),
        )
        results: dict[str, subprocess.CompletedProcess[str]] = {}
        terminated_after_game = False

        try:
            with temporary_environment(runtime_environment):
                for name, command, input_text, command_timeout in commands:
                    if name == "agent":
                        with _openclaw_gateway(base_command, gateway_log_path):
                            result, terminated_after_game = run_process_until_game_complete(command,
                                                                                            completion_path=completion_path, timeout=command_timeout)
                    else:
                        result = subprocess.run(command, input=input_text, text=True, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, timeout=command_timeout, check=False)
                    results[name] = result
                    completion = read_game_completion(completion_path)
                    game_completed = bool(completion and completion.get("done") is True
                                          and completion.get("control_failure") is not True)
                    if result.returncode != 0 and not (name == "agent" and (terminated_after_game or game_completed)):
                        detail = _error_detail(result)
                        metadata["runtime_error"] = f"OpenClaw {name} failed"
                        if detail:
                            metadata["runtime_error"] += f": {detail}"
                        break
        except subprocess.TimeoutExpired as error:
            metadata["runtime_error"] = f"OpenClaw timed out after {error.timeout}s"
        except (OSError, subprocess.SubprocessError) as error:
            metadata["runtime_error"] = f"OpenClaw could not run: {error}"
        except RuntimeError as error:
            metadata["runtime_error"] = str(error)

        if gateway_log_path.exists():
            gateway_log_path.write_text(redact_sensitive(gateway_log_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
            artifacts["openclaw_gateway"] = gateway_log_path

        agent_result = results.get("agent")
        if agent_result is not None:
            metadata["returncode"] = agent_result.returncode
        metadata["terminated_after_game"] = terminated_after_game

        # the agent output plus native session jsonl is sufficient for scoring and
        # debugging. setup-command chatter is only included when setup fails
        trace_parts: list[str] = ["trace_settings:", json.dumps({"model_io": self.trace_model_io})]
        if agent_result is not None:
            trace_parts.extend(["openclaw_agent_stdout:",
                                redact_sensitive(agent_result.stdout), "openclaw_agent_stderr:",
                                redact_sensitive(agent_result.stderr)])
        elif results:
            failed_result = list(results.values())[-1]
            trace_parts.extend(["openclaw_setup_stdout:",
                                redact_sensitive(failed_result.stdout), "openclaw_setup_stderr:",
                                redact_sensitive(failed_result.stderr)])

        for session_path in sorted(home_dir.glob(".openclaw*/agents/**/*.jsonl")):
            trace_parts.extend([f"openclaw_session: {session_path}",
                                redact_sensitive(session_path.read_text(encoding="utf-8", errors="replace"))])
        combined_trace = "\n".join(trace_parts)

        tool_call_count = sum(combined_trace.count(marker) for marker in GAME_TOOL_MARKERS)
        metadata["tool_call_count_hint"] = tool_call_count
        completion = read_game_completion(completion_path)
        metadata["game_completed"] = bool(completion and completion.get("done") is True
                                          and completion.get("control_failure") is not True)
        if metadata["runtime_error"] is None and agent_result is not None:
            if metadata["game_completed"]:
                metadata["success"] = True
            else:
                metadata["runtime_error"] = ("OpenClaw ended before clem_game reported done=true")

        trace_path = write_text_artifact(output_dir=output_dir, filename="adapter_messages.txt", content=combined_trace)
        if trace_path is not None:
            artifacts["adapter_messages"] = trace_path

        if self.debug:
            print(combined_trace)
        if metadata["runtime_error"]:
            print(metadata["runtime_error"])

        return AgentRunResult(bool(metadata["success"]), artifacts, metadata)


def _resolve_model_connection(model_spec: dict[str, Any]) -> dict[str, Any]:
    clem_model = model_spec["model_name"]
    backend = model_spec.get("backend")
    model_id = model_spec.get("model_id") or model_spec.get("model_name")

    if backend == "openrouter":
        key_config = connections._openrouter_key_config()

        return {"harness": "openclaw",
                "clem_model": model_spec["model_name"],
                "backend": "openrouter",
                "model": f"openrouter/{model_id}",
                "display_model": model_spec["model_name"],
                "base_url": connections._openrouter_openai_base_url(key_config),
                "openclaw_provider": "openrouter",
                "env": {"OPENROUTER_API_KEY": connections._openrouter_api_key(key_config)}}

    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        provider_name = "openai_compatible"
        api_key_env = "OPENAI_COMPATIBLE_API_KEY"
        runtime_model = f"{provider_name}/{model_id}"
        common = connections._openai_compatible_common(model_spec, key_config)

        model_definition: dict[str, Any] = {"id": model_id,
                                            "name": model_spec["model_name"],
                                            "api": "openai-completions",
                                            "reasoning": connections._model_reasoning_enabled(model_spec),
                                            "input": _openclaw_input_modalities(model_spec)}

        model_config = model_spec.get("model_config") or {}
        extra_body = {}

        if isinstance(model_config, dict):
            extra_body = model_config.get("extra_body") or {}

        chat_template_kwargs = {}

        if isinstance(extra_body, dict):
            raw_chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}

            if isinstance(raw_chat_template_kwargs, dict):
                chat_template_kwargs = raw_chat_template_kwargs

        if "enable_thinking" in chat_template_kwargs:
            # declare the native protocol capability without fixing its on/off value
            # the agent's thinking option controls each run
            model_definition["reasoning"] = True
            model_definition["compat"] = {"thinkingFormat": "qwen-chat-template"}

        openclaw_model_config: dict[str, Any] = {"alias": model_spec["model_name"]}

        context_window = connections._model_context_window(model_spec)

        if context_window is not None:
            model_definition["contextWindow"] = context_window

        model_definition["maxTokens"] = 8192

        return {**common, "harness": "openclaw",
                "runtime_model": runtime_model,
                "openclaw_provider": provider_name,
                "env": {api_key_env: connections._openai_compatible_api_key(key_config)},
                "openclaw_config_patch": {"models": {"mode": "merge",
                                                     "providers": {provider_name: {"baseUrl": common["base_url"],
                                                                                   "apiKey": f"${{{api_key_env}}}",
                                                                                   "api": "openai-completions",
                                                                                   "models": [model_definition]}}},
                                          "agents": {"defaults": {"models": {runtime_model: openclaw_model_config}}}}}

    raise NotImplementedError("MVP limitation: OpenClaw registry-model support currently only "
                              f"supports clembench models with backend='openrouter' or backend='openai_compatible'. "
                              f"Model {clem_model!r} has backend={backend!r}.")


def _openclaw_input_modalities(model_spec: dict[str, Any]) -> list[str]:
    """Translate registry input capabilities into OpenClaw's model schema."""
    model_config = model_spec.get("model_config") or {}

    if not isinstance(model_config, dict):
        return ["text"]

    configured = model_config.get("input_modalities")

    if not isinstance(configured, list):
        return ["text"]

    modalities = ["text"]

    for modality in configured:
        if not isinstance(modality, str):
            continue

        normalized = modality.strip().lower()

        if normalized == "image" and normalized not in modalities:
            modalities.append(normalized)

    return modalities
