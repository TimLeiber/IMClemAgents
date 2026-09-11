from .traces.claude_code import parse_claude_code_agent_trace
from . import model_connection as connections
import asyncio
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from clemagents.adapters.base import AgentRunResult, ExternalAgentHarness
from clemagents.adapters.openai_compatible_proxy import proxy_for_model_connection
from clemagents.adapters.utils import (
    GAME_MCP_SERVER_NAME,
    load_model_connection,
    model_connection_environment,
    new_game_completion_path,
    read_game_completion,
    resolve_runtime_model,
    temporary_environment,
    write_text_artifact,
)


def _anthropic_proxy_base_url(proxy_base_url: str) -> str:
    """Let Anthropic clients append /v1/messages exactly once."""
    return proxy_base_url[:-3] if proxy_base_url.endswith("/v1") else proxy_base_url


class ClaudeCodeHarness(ExternalAgentHarness):
    """Run Claude Code through the Claude Agent SDK.

    The harness configures the container-side MCP bridge as a Claude Code
    server and collects every SDK message emitted during the episode.
    """


    @classmethod
    def resolve_model_connection(cls, clem_model: str) -> dict[str, Any]:
        return resolve_clem_model_for_claude_code(clem_model)

    def __init__(self,
                 model: str | None = None,
                 clem_model: str | None = None,
                 mcp_url: str = "http://localhost:8001/mcp",
                 max_turns: int | None = None,
                 allowed_tools: list[str] | None = None,
                 permission_mode: str = "bypassPermissions",
                 timeout: float | None = None,
                 reasoning_effort: str | None = None,
                 model_connection_path: str | None = None,
                 trace_model_io: bool = True):
        """Configure the Claude Code harness.

        Args:
            model: model identifier passed directly to Claude Code
            clem_model: clembench model resolved by the outer pipeline
            mcp_url: URL forwarded to the container-side MCP bridge
            max_turns: optional maximum number of Claude Code turns
            allowed_tools: tool patterns Claude Code pre-approves; this does
                not replace its built-in tool inventory
            permission_mode: Claude Code tool-permission policy
            timeout: optional Claude Code episode deadline in seconds
            reasoning_effort: model reasoning effort passed to Claude Code;
                ``none`` disables thinking explicitly
            model_connection_path: optional resolved model-connection file
            trace_model_io: whether to record model requests and responses
        """

        self.model = model or clem_model
        self.clem_model = clem_model
        self.mcp_url = mcp_url
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or [
            f"mcp__{GAME_MCP_SERVER_NAME}__start_game",
            f"mcp__{GAME_MCP_SERVER_NAME}__submit_response",
        ]
        self.permission_mode = permission_mode
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.trace_model_io = trace_model_io
        self._model_connection = load_model_connection("claude_code", model_connection_path)

    @classmethod
    def parse_agent_trace(cls,
                          episode_dir: Path,
                          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Delegate Claude Code trace parsing to adapter utilities."""

        return parse_claude_code_agent_trace(episode_dir=episode_dir, metadata=metadata)

    async def run_episode_async(self,
                                instruction: str,
                                runtime_model: str | None,
                                runtime_environment: dict[str, str | None],
                                completion_path: Path) -> tuple[list[Any], str | None, bool]:
        """Collect the asynchronous Claude SDK message stream.

        This method is the asynchronous boundary required by the Claude Agent
        SDK while the shared harness interface remains synchronous.

        Args:
            instruction: task instruction passed to Claude Code
            runtime_model: resolved model identifier
            runtime_environment: temporary model-provider environment
            completion_path: bridge-to-harness completion marker

        Returns:
            the collected SDK messages, an optional runtime error, and whether
            the game completed
        """

        options_kwargs = {
            "mcp_servers": {
                GAME_MCP_SERVER_NAME: {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "clemagents.mcp.bridge"],
                },
            },
            "allowed_tools": self.allowed_tools,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
            "include_partial_messages": self.trace_model_io,
            "include_hook_events": self.trace_model_io,
            # support large mcp image payloads from multimodal games
            "max_buffer_size": 16 * 1024 * 1024,
        }

        if runtime_model:
            options_kwargs["model"] = runtime_model

        if self.reasoning_effort == "none":
            options_kwargs["thinking"] = {"type": "disabled"}
        elif self.reasoning_effort is not None:
            options_kwargs["effort"] = self.reasoning_effort

        options = ClaudeAgentOptions(**options_kwargs)
        messages = []
        runtime_error = None
        game_completed = False

        with temporary_environment(runtime_environment):
            try:
                async for message in query(prompt=instruction, options=options):
                    messages.append(message)
                    print(message)

                    completion = read_game_completion(completion_path)

                    if completion and completion.get("done") is True:
                        game_completed = completion.get("control_failure") is not True
                        break

                    tool_use_result = getattr(message, "tool_use_result", None)
                    structured_content = (tool_use_result.get("structuredContent")
                                          if isinstance(tool_use_result, dict) else None)

                    if isinstance(structured_content, dict) and structured_content.get("done") is True:
                        game_completed = True
                        break
            except Exception as error:
                runtime_error = str(error)
                print(f"agent_runtime_error: {error}")

        return messages, runtime_error, game_completed

    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        completion_path = new_game_completion_path()
        proxy = proxy_for_model_connection(self._model_connection,
                                           completion_path,
                                           include_openrouter=True,
                                           trace_responses=self.trace_model_io,
                                           trace_requests=self.trace_model_io)

        if proxy is not None:
            with proxy:
                return self._run_episode(
                    instruction,
                    output_dir,
                    completion_path,
                    _anthropic_proxy_base_url(proxy.base_url),
                )

        return self._run_episode(instruction, output_dir, completion_path, None)

    def _run_episode(self,
                     instruction: str,
                     output_dir: Path | str | None,
                     completion_path: Path,
                     proxied_base_url: str | None) -> AgentRunResult:
        """Run one Claude Code episode.

        Args:
            instruction: task instruction passed to Claude Code
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized Claude Code run result
        """

        # ----- step 1 -----
        # resolve the model and provider environment
        runtime_model = resolve_runtime_model(model_connection=self._model_connection,
                                              model=self.model,
                                              harness_name="ClaudeCodeHarness",
                                              required=False)
        runtime_model = (
            (self._model_connection or {}).get("runtime_model")
            or runtime_model
        )
        runtime_environment = model_connection_environment(self._model_connection)
        runtime_environment["GAME_COMPLETION_PATH"] = str(completion_path)

        if proxied_base_url is not None:
            runtime_environment["ANTHROPIC_BASE_URL"] = proxied_base_url

        # ----- step 2 -----
        # run Claude Code and collect the SDK message stream
        coroutine = self.run_episode_async(instruction=instruction,
                                           runtime_model=runtime_model,
                                           runtime_environment=runtime_environment,
                                           completion_path=completion_path)

        if self.timeout is not None:
            coroutine = asyncio.wait_for(coroutine, timeout=self.timeout)

        try:
            messages, runtime_error, game_completed = asyncio.run(coroutine)
        except asyncio.TimeoutError:
            messages = []
            runtime_error = f"Claude Code timed out after {self.timeout}s"
            game_completed = False
            print(f"agent_runtime_error: {runtime_error}")

        # ----- step 3 -----
        # extract standardized metadata from the SDK messages
        completion = read_game_completion(completion_path)
        game_completed = bool(
            game_completed
            or (
                completion
                and completion.get("done") is True
                and completion.get("control_failure") is not True
            )
        )
        metadata = {
            "adapter": "claude_code",
            "model": self.model,
            "clem_model": self.clem_model,
            "runtime_model": runtime_model,
            "resolved_backend": (self._model_connection or {}).get("backend"),
            "gateway_base_url": (
                (self._model_connection or {}).get("base_url")
                or runtime_environment.get("ANTHROPIC_BASE_URL")
            ),
            "compatibility_proxy_base_url": proxied_base_url,
            "tool_choice": (self._model_connection or {}).get("tool_choice"),
            "registry_request_body_overrides": (
                (self._model_connection or {}).get("request_body_overrides") or {}
            ),
            # OpenAI extra_body fields are not injected into Anthropic
            # Messages requests, so this accurately records the wire behavior.
            "request_body_overrides": {},
            "verify_tls": (self._model_connection or {}).get("verify_tls"),
            "timeout": self.timeout,
            "reasoning_effort": self.reasoning_effort,
            "trace_model_io": self.trace_model_io,
            "success": game_completed,
            "session_id": None,
            "duration_ms": None,
            "total_cost_usd": None,
            "num_turns": None,
            "stop_reason": None,
            "runtime_error": runtime_error,
            "game_completed": game_completed,
        }

        for message in messages:
            session_id = getattr(message,
                                 "session_id",
                                 None)

            if session_id is not None:
                metadata["session_id"] = session_id

            for field_name in ("duration_ms", "total_cost_usd", "num_turns", "stop_reason"):
                value = getattr(message,
                                field_name,
                                None)

                if value is not None:
                    metadata[field_name] = value

        # ----- step 4 -----
        # write the raw SDK messages when artifact output is enabled
        artifacts = {}
        messages_path = write_text_artifact(output_dir=output_dir,
                                            filename="adapter_messages.txt",
                                            content="\n".join(repr(message) for message in messages))

        if messages_path is not None:
            artifacts["adapter_messages"] = messages_path

        return AgentRunResult(success=bool(metadata["success"]),
                              artifacts=artifacts,
                              metadata=metadata)


def resolve_clem_model_for_claude_code(clem_model: str) -> dict[str, Any]:
    model_spec = connections._find_model_spec(clem_model)
    backend = model_spec.get("backend")

    if backend == "openai_compatible":
        key_config = connections._openai_compatible_key_config()
        connection = connections._openai_compatible_common(model_spec, key_config)
        connection.update({
            "harness": "claude_code",
            "runtime_model": "claude-sonnet-4-5",
            "env": {
                "ANTHROPIC_BASE_URL": connection["base_url"],
                "ANTHROPIC_AUTH_TOKEN": connections._openai_compatible_api_key(key_config),
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
        })
        return connection

    if backend != "openrouter":
        raise NotImplementedError(
            "MVP limitation: Claude Code registry-model support currently only "
            f"supports clembench models with backend='openrouter' or backend='openai_compatible'. "
            f"Model {clem_model!r} has backend={backend!r}."
        )

    model_id = model_spec.get("model_id") or model_spec.get("model_name")
    key_config = connections._openrouter_key_config()

    return {
        "harness": "claude_code",
        "clem_model": model_spec["model_name"],
        "backend": "openrouter",
        "model": model_id,
        "display_model": model_spec["model_name"],
        "base_url": _openrouter_anthropic_base_url(key_config),
        "request_body_overrides": connections._model_request_body_overrides(model_spec),
        "env": {
            "ANTHROPIC_BASE_URL": _openrouter_anthropic_base_url(key_config),
            "ANTHROPIC_AUTH_TOKEN": connections._openrouter_api_key(key_config),
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    }



def _openrouter_anthropic_base_url(config: dict[str, Any]) -> str:
    base_url = str(config.get("base_url") or "https://openrouter.ai/api").rstrip("/")

    # clemcore/OpenRouter configs often use the OpenAI-compatible /api/v1
    # endpoint. Claude Code needs OpenRouter's Anthropic-compatible endpoint.
    if base_url.endswith("/api/v1"):
        return base_url[:-3]

    return base_url

