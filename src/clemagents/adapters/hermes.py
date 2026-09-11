"""Native Hermes configuration, execution and artifact collection."""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import hermes_observer, model_connection as connections
from .base import AgentRunResult, ExternalAgentHarness
from .traces.hermes import parse_hermes_agent_trace
from .utils import (GAME_MCP_SERVER_NAME, load_model_connection, mcp_environment, model_connection_environment,
                    new_game_completion_path, read_game_completion, redact_sensitive, resolve_runtime_model,
                    run_process_until_game_complete, temporary_environment, warn_model_generation_config,
                    write_text_artifact)


class HermesHarness(ExternalAgentHarness):
    """Run the native Hermes CLI with an isolated home and its default tools."""

    @classmethod
    def resolve_model_connection(cls, model_spec: dict[str, Any], agent_config: dict[str, Any]) -> dict[str, Any]:
        warn_model_generation_config(model_spec)
        return _resolve_model_connection(model_spec)

    def __init__(self, model: str | None = None, clem_model: str | None = None, provider: str = "openrouter",
                 mcp_url: str = "http://host.docker.internal:8001/mcp", max_turns: int = 20, yolo: bool = True,
                 reasoning_effort: str | None = None, model_connection_path: str | None = None,
                 trace_model_io: bool = True):
        """Configure Hermes through its native controls.

        Args:
            model: native model identifier
            clem_model: registry model resolved by the host
            provider: native provider for direct model configurations
            mcp_url: endpoint forwarded to the container-side MCP bridge
            max_turns: native maximum agent turns
            yolo: disable native approval gates
            reasoning_effort: native effort, rejected if a complete snapshot proves its API control absent
            model_connection_path: resolved model-connection file
            trace_model_io: record native request dumps and observation events
        """
        self.model = model or clem_model
        self.clem_model = clem_model
        self.provider = provider
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.yolo = yolo
        self.reasoning_effort = reasoning_effort
        self.trace_model_io = trace_model_io
        self._model_connection = load_model_connection("hermes", model_connection_path)
        connection = self._model_connection or {}
        if any(connection.get(key) for key in ("request_body_overrides", "generation_overrides", "upstream_model")):
            raise ValueError("Hermes does not accept request overrides; configure its native agent controls instead")
        if connection.get("verify_tls") is False:
            raise ValueError("Hermes uses native TLS verification; disabling it is not supported by this adapter")

    @classmethod
    def parse_agent_trace(cls, episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return parse_hermes_agent_trace(episode_dir=episode_dir, metadata=metadata)

    def run_episode(self, instruction: str, output_dir: Path | str | None = None) -> AgentRunResult:
        completion_path = new_game_completion_path()
        run_dir = Path(output_dir).resolve() if output_dir is not None else Path(tempfile.mkdtemp(prefix="hermes-episode-"))
        run_dir.mkdir(parents=True, exist_ok=True)
        observer_path = run_dir / "hermes_observer.jsonl"
        home = run_dir / "hermes_home"
        connection = self._model_connection or {}
        provider = connection.get("provider") or self.provider
        metadata = {"adapter": "hermes", "model": self.model, "clem_model": self.clem_model,
                    "resolved_backend": connection.get("backend"), "gateway_base_url": connection.get("base_url"),
                    "provider": provider, "max_turns": self.max_turns, "reasoning_effort": self.reasoning_effort,
                    "trace_model_io": self.trace_model_io, "capture_method": "native_observer_and_request_dumps",
                    "success": False, "returncode": None, "runtime_error": None, "game_completed": False,
                    "terminated_after_game": False, "hermes_session_id": None, "tool_call_count_hint": 0}
        artifacts, trace_parts = {}, []
        try:
            runtime_model = resolve_runtime_model(model_connection=self._model_connection, model=self.model,
                                                  harness_name="HermesHarness")
            metadata["runtime_model"] = runtime_model
            environment = model_connection_environment(self._model_connection)
            endpoint_variable = "OPENROUTER_BASE_URL" if provider == "openrouter" else "OPENAI_BASE_URL"
            if connection.get("base_url"):
                environment[endpoint_variable] = connection["base_url"]
            bridge_environment = mcp_environment(self.mcp_url, include_pythonpath=True)
            bridge_environment["GAME_COMPLETION_PATH"] = str(completion_path)
            config = {"model": {"default": runtime_model, "provider": provider},
                      "display": {"show_reasoning": self.trace_model_io, "streaming": True, "tool_progress": "verbose"},
                      "mcp_servers": {GAME_MCP_SERVER_NAME: {"command": "python",
                                                            "args": ["-m", "clemagents.mcp.bridge"],
                                                            "env": bridge_environment}}}
            if environment.get(endpoint_variable):
                config["model"]["base_url"] = environment[endpoint_variable]
            if self.reasoning_effort is not None:
                config["agent"] = {"reasoning_effort": self.reasoning_effort}
            environment.update(hermes_observer.configure(home, config, observer_path, completion_path,
                                                          self.reasoning_effort, self.trace_model_io))
            with temporary_environment(environment):
                self._check_setup(trace_parts)
                command = ["hermes", "chat", "--provider", provider, "--model", runtime_model,
                           "--max-turns", str(self.max_turns), "--ignore-rules"]
                if self.yolo:
                    command.append("--yolo")
                command.extend(["-q", instruction])
                trace_parts.extend(["hermes_chat_command:", " ".join(command[:-1] + ["<instruction>"])])
                chat, terminated = run_process_until_game_complete(command, completion_path=completion_path)
                metadata.update(returncode=chat.returncode, terminated_after_game=terminated)
                trace_parts.extend(["hermes_chat_stdout:", chat.stdout, "hermes_chat_stderr:", chat.stderr])
                session = self._session_id(observer_path, chat.stdout + "\n" + chat.stderr)
                metadata["hermes_session_id"] = session
                if session:
                    self._export_session(session, run_dir, artifacts, trace_parts)
        except subprocess.TimeoutExpired as error:
            metadata["runtime_error"] = f"Hermes timed out after {error.timeout}s"
            for value in (error.stdout, error.stderr):
                trace_parts.append(value.decode(errors="replace") if isinstance(value, bytes) else value or "")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            metadata["runtime_error"] = f"Hermes setup or execution failed: {error}"

        completion = read_game_completion(completion_path)
        metadata["game_completed"] = bool(completion and completion.get("done") is True
                                          and completion.get("control_failure") is not True)
        model_event_count = 0
        if observer_path.exists():
            artifacts["hermes_observer"] = observer_path
            for line in observer_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("hook") == "unsupported_control":
                    metadata["runtime_error"] = event["content"]
                if event.get("hook") == "post_tool_call":
                    metadata["tool_call_count_hint"] += 1
                if event.get("hook") in {"pre_api_request", "post_api_request"}:
                    model_event_count += 1
        if self.trace_model_io and not model_event_count and metadata["returncode"] is not None:
            metadata["capture_warning"] = "Hermes produced no native model request/response observations"
        if metadata["runtime_error"] is None and not metadata["game_completed"]:
            metadata["runtime_error"] = "Hermes ended before clem_game reported done=true"
        metadata["success"] = metadata["game_completed"] and metadata["runtime_error"] is None
        if metadata["runtime_error"]:
            trace_parts.extend(["agent_runtime_error:", metadata["runtime_error"]])
        trace = redact_sensitive("\n".join(trace_parts))
        artifacts["adapter_messages"] = write_text_artifact(output_dir=run_dir, filename="adapter_messages.txt", content=trace)
        print(trace)
        return AgentRunResult(metadata["success"], artifacts, metadata)

    @staticmethod
    def _check_setup(trace_parts):
        # mcp test prints errors with exit code zero, so require the actual inventory too
        commands = [["hermes", "mcp", "test", GAME_MCP_SERVER_NAME],
                    ["python", "-m", "clemagents.adapters.hermes_observer"]]
        for command in commands:
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            trace_parts.extend([" ".join(command), result.stdout, result.stderr])
            if result.returncode:
                detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
                raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}): {redact_sensitive(detail)}")
            if command[1:3] == ["mcp", "test"]:
                inventory = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
                if not all(re.search(rf"(?m)^\s+(?:mcp__game__)?{name}\s", inventory)
                           for name in ("start_game", "submit_response")):
                    raise RuntimeError("Hermes MCP test did not discover start_game and submit_response; chat not started")

    @staticmethod
    def _session_id(observer_path, text):
        if observer_path.exists():
            for line in observer_path.read_text(encoding="utf-8").splitlines():
                try:
                    session = json.loads(line).get("session_id")
                    if session:
                        return session
                except json.JSONDecodeError:
                    pass
        match = re.search(r"(?:^Session:\s*|hermes --resume\s+|\bsession=)([A-Za-z0-9_.:-]+)", text, re.MULTILINE)
        return match.group(1) if match else None

    @staticmethod
    def _export_session(session, run_dir, artifacts, trace_parts):
        path = run_dir / "hermes_session_export.jsonl"
        result = subprocess.run(["hermes", "sessions", "export", "--session-id", session, str(path)],
                                text=True, capture_output=True, timeout=60)
        trace_parts.extend(["hermes_sessions_export:", result.stdout, result.stderr])
        if path.exists():
            artifacts["hermes_session_export"] = path


def _resolve_model_connection(model_spec: dict[str, Any]) -> dict[str, Any]:
    backend = model_spec.get("backend")
    model_id = model_spec.get("model_id") or model_spec["model_name"]
    if backend == "openrouter":
        key_config = connections._openrouter_key_config()
        return {"harness": "hermes", "clem_model": model_spec["model_name"], "backend": backend,
                "provider": "openrouter", "model": model_id, "display_model": model_spec["model_name"],
                "base_url": connections._openrouter_openai_base_url(key_config),
                "env": {"OPENROUTER_API_KEY": connections._openrouter_api_key(key_config)}}
    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        connection = connections._openai_compatible_common(model_spec, key_config)
        connection.update({"harness": "hermes", "provider": "openai-api",
                           "env": {"OPENAI_BASE_URL": connection["base_url"],
                                   "OPENAI_API_KEY": connections._openai_compatible_api_key(key_config)}})
        return connection
    raise NotImplementedError(f"Hermes does not support registry backend {backend!r}")
