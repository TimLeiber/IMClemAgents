import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..base import AgentRunResult, ExternalAgentHarness
from ..utils import (GAME_MCP_SERVER_NAME, configure_tls, load_model_connection, mcp_environment,
                     model_connection_environment, new_game_completion_path, read_game_completion,
                     redact_sensitive, require_resolved_tls, resolve_runtime_model, run_process_until_game_complete,
                     temporary_environment, warn_model_generation_config, write_text_artifact)
from ..utils import model_connection as connections
from ..utils.openai_compatible_proxy import proxy_for_model_connection
from .parse import parse_codex_agent_trace
from .utils import _positive_token_limit, _resolve_model_connection



class CodexHarness(ExternalAgentHarness):
    """Run Codex through the Codex CLI.

    The harness writes an isolated Codex configuration that registers the
    container-side MCP bridge, then runs one non-interactive `codex exec`.
    """

    @classmethod
    def resolve_model_connection(cls, model_spec: dict[str, Any], agent_config: dict[str, Any]) -> dict[str, Any]:
        warn_model_generation_config(model_spec)
        connection = _resolve_model_connection(model_spec)
        context_window = connections._model_context_window(model_spec)
        if context_window is not None:
            connection["model_context_window"] = _positive_token_limit(context_window, "context_size")
        elif model_spec.get("context_size") is not None:
            raise ValueError("Codex cannot interpret model registry context_size")
        return configure_tls(connection, agent_config)

    def __init__(self, model: str | None = None, clem_model: str | None = None,
                 mcp_url: str = "http://host.docker.internal:8001/mcp", sandbox: str = "full_access",
                 reasoning_effort: str | None = None, model_connection_path: str | None = None,
                 trace_model_io: bool = True, model_context_window: int | None = None,
                 model_auto_compact_token_limit: int | None = None, model_catalog: dict[str, Any] | None = None,
                 ca_bundle: str | None = None, verify_tls: bool | None = None):
        """Configure the Codex harness.

        Args:
            model: model identifier passed directly to Codex
            clem_model: clembench model resolved by the outer pipeline
            mcp_url: URL forwarded to the container-side MCP bridge
            sandbox: Codex sandbox policy
            reasoning_effort: model reasoning effort passed to Codex
            model_connection_path: optional resolved model-connection file
            trace_model_io: whether to record model requests and responses
            model_context_window: native context limit in tokens, overriding registry context_size
            model_auto_compact_token_limit: optional native compaction threshold in tokens
            model_catalog: optional explicit native catalog otherwise leaving Codex defaults intact
            ca_bundle: optional host PEM file resolved through clem_model before startup
            verify_tls: optional provider certificate verification policy resolved through clem_model
        """

        self.model = model or clem_model or "gpt-5.4"
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.sandbox = sandbox
        self.reasoning_effort = reasoning_effort
        self.trace_model_io = trace_model_io
        self._model_connection = load_model_connection("codex", model_connection_path)
        require_resolved_tls(self._model_connection, ca_bundle, verify_tls)
        registry_window = (self._model_connection or {}).get("model_context_window")
        self.model_context_window = _positive_token_limit(model_context_window if model_context_window is not None
                                                          else registry_window, "model_context_window")
        self.model_auto_compact_token_limit = _positive_token_limit(model_auto_compact_token_limit,
                                                                    "model_auto_compact_token_limit")
        if (self.model_context_window is not None and self.model_auto_compact_token_limit is not None
                and self.model_auto_compact_token_limit > self.model_context_window):
            raise ValueError("model_auto_compact_token_limit must not exceed model_context_window")
        self.model_catalog = model_catalog

    @classmethod
    def parse_agent_trace(cls, episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Delegate Codex-specific trace parsing to adapter utilities."""

        return parse_codex_agent_trace(episode_dir=episode_dir, metadata=metadata)

    def run_episode(self, instruction: str, output_dir: Path | str | None = None) -> AgentRunResult:
        completion_path = new_game_completion_path()
        resolved_backend = (self._model_connection or {}).get("backend")
        proxy = proxy_for_model_connection(self._model_connection, completion_path, include_openrouter=True,
                                           trace_responses=self.trace_model_io, trace_requests=self.trace_model_io)

        if proxy is not None:
            with proxy:
                result = self._run_episode(instruction, output_dir, completion_path, proxy.base_url)
                # finalized artifacts must retain the same api records as the live log
                proxy_trace = redact_sensitive(proxy.captured_trace())
                adapter_messages = result.artifacts.get("adapter_messages")
                if proxy_trace and adapter_messages is not None:
                    trace_path = Path(adapter_messages)
                    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "\n" + proxy_trace, encoding="utf-8")
                return result

        return self._run_episode(instruction, output_dir, completion_path, None)

    def _run_episode(self, instruction: str, output_dir: Path | str | None, completion_path: Path,
                     proxied_base_url: str | None) -> AgentRunResult:
        """Run one Codex episode.

        Args:
            instruction: task instruction passed to Codex
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized Codex run result
        """

        # step 1
        # resolve the model, provider environment, and sandbox
        runtime_model = resolve_runtime_model(model_connection=self._model_connection, model=self.model,
                                              harness_name="CodexHarness")
        resolved_backend = (self._model_connection or {}).get("backend")
        codex_model = runtime_model
        runtime_environment = model_connection_environment(self._model_connection)
        runtime_base_url = proxied_base_url

        if runtime_base_url is None and self._model_connection and self._model_connection.get("base_url"):
            runtime_base_url = str(self._model_connection["base_url"])

        sandbox_modes = {"read_only": "read-only",
                         "workspace_write": "workspace-write",
                         "full_access": "danger-full-access"}

        if self.sandbox not in sandbox_modes:
            raise ValueError(f"Unknown Codex sandbox '{self.sandbox}'. "
                             f"Expected one of: {sorted(sandbox_modes)}")

        # configure the native provider and mcp bridge for this episode
        config_dir = Path.home() / ".codex"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.toml"
        bridge_environment = mcp_environment(self.mcp_url, include_pythonpath=True)
        bridge_environment["GAME_COMPLETION_PATH"] = str(completion_path)
        environment_lines = "\n".join(f"{key} = {json.dumps(value)}"
                                      for key, value in sorted(bridge_environment.items()))

        # allow tool use without approval prompts
        config_lines = [f"model = {json.dumps(codex_model)}", 'approval_policy = "never"',
                        f"sandbox_mode = {json.dumps(sandbox_modes[self.sandbox])}", 'web_search = "live"']

        if self.reasoning_effort is not None:
            config_lines.append(f"model_reasoning_effort = {json.dumps(self.reasoning_effort)}")

        # pass context controls to codex itself without changing provider requests or model profiles
        for key in ("model_context_window", "model_auto_compact_token_limit"):
            value = getattr(self, key)
            if value is not None:
                config_lines.append(f"{key} = {value}")
        catalog = self.model_catalog
        saved_catalog_path = None
        if catalog is not None:
            models = catalog.get("models") if isinstance(catalog, dict) else None
            if not isinstance(models, list) or not any(isinstance(model, dict) and model.get("slug") == codex_model
                                                     for model in models):
                raise ValueError("model_catalog must contain a models list with the selected runtime model slug")
            catalog_path = config_dir / "model_catalog.json"
            catalog_text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
            catalog_path.write_text(catalog_text, encoding="utf-8")
            config_lines.append(f"model_catalog_json = {json.dumps(str(catalog_path))}")
            saved_catalog_path = write_text_artifact(output_dir=output_dir, filename="codex_model_catalog.json",
                                                     content=catalog_text)

        if runtime_base_url:
            if resolved_backend == "openrouter" and proxied_base_url is not None:
                config_lines.extend(['model_provider = "openrouter_proxy"', "", "[model_providers.openrouter_proxy]",
                                     'name = "openrouter_proxy"', f"base_url = {json.dumps(runtime_base_url)}",
                                     'env_key = "OPENROUTER_API_KEY"', 'wire_api = "responses"'])
            elif "openrouter.ai" in runtime_base_url:
                config_lines.extend(['model_provider = "openrouter"', "", "[model_providers.openrouter]",
                                     'name = "openrouter"',
                                     f"base_url = {json.dumps(runtime_base_url)}", 'env_key = "OPENROUTER_API_KEY"'])
            else:
                config_lines.extend(['model_provider = "openai_api"', "", "[model_providers.openai_api]",
                                     f"name = {json.dumps('OpenAI API')}", f"base_url = {json.dumps(runtime_base_url)}",
                                     f"env_key = {json.dumps('OPENAI_API_KEY')}", 'wire_api = "responses"'])

        config_lines.extend(["", f"[mcp_servers.{GAME_MCP_SERVER_NAME}]", 'command = "python"',
                             'args = ["-m", "clemagents.mcp.bridge"]',
                             "startup_timeout_sec = 20", "tool_timeout_sec = 120", "enabled = true", "required = true",
                             'enabled_tools = ["start_game", "submit_response"]', 'default_tools_approval_mode = "approve"', "",
                             f"[mcp_servers.{GAME_MCP_SERVER_NAME}.env]", environment_lines, ""])
        config_path.write_text("\n".join(config_lines), encoding="utf-8")
        # preserve native configuration even when the host terminates the episode before finalization
        write_text_artifact(output_dir=output_dir, filename="codex_config.toml",
                            content=config_path.read_text(encoding="utf-8"))

        # step 3
        # build the codex instruction and command
        last_message_path = config_dir / f"last_message_{os.getpid()}.txt"

        if last_message_path.exists():
            last_message_path.unlink()

        command = ["codex", "exec", "--strict-config", "--json", "--cd", "/workspace", "--skip-git-repo-check",
                   "--ephemeral",
                   "--output-last-message",
                   str(last_message_path), "-"]

        # step 4
        # verify that codex can read the generated mcp configuration
        mcp_list_stdout = ""
        mcp_list_stderr = ""
        mcp_list_returncode = None

        try:
            with temporary_environment(runtime_environment):
                mcp_list = subprocess.run(["codex", "mcp", "list"], text=True, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, cwd="/workspace", check=False)

            mcp_list_stdout = mcp_list.stdout
            mcp_list_stderr = mcp_list.stderr
            mcp_list_returncode = mcp_list.returncode
        except Exception as error:
            mcp_list_stderr = str(error)

        # step 5
        # run codex and capture its jsonl trace
        metadata = {"adapter": "codex",
                    "model": self.model,
                    "clem_model": self.clem_model,
                    "runtime_model": runtime_model,
                    "codex_model": codex_model,
                    "resolved_backend": resolved_backend,
                    "gateway_base_url": ((self._model_connection or {}).get("base_url") or runtime_base_url),
                    "compatibility_proxy_base_url": proxied_base_url,
                    "tool_choice": (self._model_connection or {}).get("tool_choice"),
                    "verify_tls": (self._model_connection or {}).get("verify_tls"),
                    "mcp_url": self.mcp_url,
                    "codex_config": str(config_path),
                    "sandbox": self.sandbox,
                    "reasoning_effort": self.reasoning_effort,
                    "model_context_window": self.model_context_window,
                    "model_auto_compact_token_limit": self.model_auto_compact_token_limit,
                    "custom_model_catalog": self.model_catalog is not None,
                    "trace_model_io": self.trace_model_io,
                    "success": False,
                    "runtime_error": None,
                    "returncode": None,
                    "final_response": None,
                    "game_completed": False,
                    "terminated_after_game": False}
        transcript = [f"codex_config: {config_path}", "codex_config_toml:",
                      config_path.read_text(encoding="utf-8"), f"model: {self.model}", f"runtime_model: {runtime_model}",
                      f"codex_model: {codex_model}", f"gateway_base_url: {metadata['gateway_base_url']}",
                      f"compatibility_proxy_base_url: {proxied_base_url}", f"sandbox: {self.sandbox}", "trace_settings:",
                      json.dumps({"model_io": self.trace_model_io}), "codex_mcp_list_returncode:",
                      str(mcp_list_returncode), "codex_mcp_list_stdout:", mcp_list_stdout, "codex_mcp_list_stderr:",
                      mcp_list_stderr, "codex_command:", " ".join(command)]

        try:
            with temporary_environment(runtime_environment):
                completed, terminated_after_game = run_process_until_game_complete(command,
                                                                                   completion_path=completion_path,
                                                                                   input_text=instruction,
                                                                                   cwd="/workspace")

            metadata["returncode"] = completed.returncode
            metadata["terminated_after_game"] = terminated_after_game
            transcript.extend(["codex_stdout_jsonl:", completed.stdout, "codex_stderr:", completed.stderr])

            if last_message_path.exists():
                final_response = last_message_path.read_text(encoding="utf-8")
                metadata["final_response"] = final_response
                transcript.extend(["final_response:", final_response])
        except Exception as error:
            metadata["runtime_error"] = str(error)
            transcript.append(f"agent_runtime_error: {error}")

        completion = read_game_completion(completion_path)
        metadata["game_completed"] = bool(completion and completion.get("done") is True
                                          and completion.get("control_failure") is not True)
        metadata["success"] = metadata["game_completed"]

        if metadata["runtime_error"] is None and not metadata["game_completed"]:
            metadata["runtime_error"] = "Codex ended before clem_game reported done=true"

        # step 6
        # write the trace and generated configuration
        trace_text = "\n".join(transcript)
        print(trace_text)
        artifacts = {}
        messages_path = write_text_artifact(output_dir=output_dir, filename="adapter_messages.txt", content=trace_text)
        saved_config_path = write_text_artifact(output_dir=output_dir, filename="codex_config.toml",
                                                content=config_path.read_text(encoding="utf-8"))

        if messages_path is not None:
            artifacts["adapter_messages"] = messages_path

        if saved_config_path is not None:
            artifacts["codex_config"] = saved_config_path
        if saved_catalog_path is not None:
            artifacts["codex_model_catalog"] = saved_catalog_path

        return AgentRunResult(success=bool(metadata["success"]), artifacts=artifacts, metadata=metadata)
