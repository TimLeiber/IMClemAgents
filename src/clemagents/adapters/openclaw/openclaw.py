import json
import os
import re
import secrets
import subprocess
from pathlib import Path
from typing import Any

from ..base import AgentRunResult, ExternalAgentHarness
from ..utils import (GAME_MCP_SERVER_NAME, configure_tls, deep_merge_dicts, load_model_connection, mcp_environment,
                     model_connection_environment, new_game_completion_path, read_game_completion,
                     redact_sensitive, require_resolved_tls, resolve_runtime_model, run_process_until_game_complete,
                     temporary_environment, warn_model_generation_config, write_text_artifact)
from ..utils.openai_compatible_proxy import proxy_for_model_connection
from .parse import parse_openclaw_agent_trace
from .utils import _error_detail, _openclaw_gateway, _resolve_model_connection, _validate_openclaw_model_connection


SESSION_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
DUCKDUCKGO_PLUGIN_PATH = Path("/opt/openclaw-plugins/duckduckgo")
GAME_TOOL_MARKERS = ("mcp__game__", "mcp_game_", "game.start_game", "game.submit_response", "mcp__clem-game__",
                     "mcp_clem_game_", "clem_game.start_game", "clem_game.submit_response", '"start_game"',
                     '"submit_response"',
                     )


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
