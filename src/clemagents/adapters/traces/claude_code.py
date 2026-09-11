import ast
import json
import re
from pathlib import Path
from typing import Any
from .schema import missing_agent_trace
from .common import _marked_trace_section, _provider_trace_records


def parse_claude_code_agent_trace(episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Claude Code SDK trace into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return missing_agent_trace("Native agent_trace.log is missing", "claude_code")

    if metadata is None:
        metadata_path = episode_dir / "agent_trace_meta.json"

        if metadata_path.exists():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                metadata = None

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    records = _claude_code_trace_records(trace_text)
    events = []
    tasks, tasks_by_parent_call = _claude_code_task_contexts(records)
    agent_loop_instruction = _marked_trace_section(trace_text, "agent_loop_instruction_start",
                                                   "agent_loop_instruction_end")

    if agent_loop_instruction is not None:
        events.append({"type": "instruction",
                       "kind": "agent_loop",
                       "source": "container",
                       "content": agent_loop_instruction})

    message_ids_with_tools = {record.get("message_id") for record in records
                              if record.get("__type__") == "AssistantMessage" and record.get("message_id") is not None and any(block.get("__type__") == "ToolUseBlock" for block in record.get("content", []) if isinstance(block, dict))}
    message_turns = {}
    call_turns = {}
    runtime = {}
    result = {}
    thinking_token_updates = 0
    thinking_token_total = 0
    previous_thinking_tokens = 0
    parse_errors = 0

    for record in records:
        record_type = record.get("__type__")

        if record_type == "parse_error":
            parse_errors += 1
            events.append({"type": "trace_warning",
                           "kind": "parser",
                           "source": "sdk_trace",
                           "content": record.get("content", "Could not parse SDK trace line")})
            continue

        if record_type == "SystemMessage":
            subtype = record.get("subtype")
            data = record.get("data", {})

            if subtype == "init" and isinstance(data, dict):
                runtime = {key: data.get(key)
                           for key in ("cwd", "session_id", "model", "permissionMode", "mcp_servers", "claude_code_version",
                                       "output_style", "agents", "skills", "plugins", "capabilities",
                                       ) if data.get(key) is not None}
                tools = data.get("tools", [])

                if tools:
                    events.append({"type": "tool_definitions",
                                   "kind": "sdk_inventory",
                                   "source": "sdk_trace",
                                   "tools": [{"name": name} for name in tools]})

            elif subtype == "thinking_tokens" and isinstance(data, dict):
                estimated_tokens = data.get("estimated_tokens")

                if isinstance(estimated_tokens, int):
                    thinking_token_updates += 1

                    if estimated_tokens < previous_thinking_tokens:
                        thinking_token_total += previous_thinking_tokens

                    previous_thinking_tokens = estimated_tokens

            continue

        if record_type == "TaskStartedMessage":
            task = _claude_code_task_context(record, tasks, tasks_by_parent_call)

            if task is not None:
                event = {"type": "message",
                         "kind": ("agent_started" if task.get("scope") == "subagent" else "background_task_started"),
                         "role": "runtime",
                         "source": "sdk_trace",
                         "content": {"description": task.get("description"),
                                     "prompt": task.get("prompt")},
                         "payload": record}
                event.update(_claude_code_agent_event_fields(task))
                events.append(event)

            continue

        if record_type == "TaskProgressMessage":
            task = _claude_code_task_context(record, tasks, tasks_by_parent_call)

            if task is not None:
                data = record.get("data") if isinstance(record.get("data"), dict) else {}
                event = {"type": "message",
                         "kind": ("agent_progress" if task.get("scope") == "subagent" else "background_task_progress"),
                         "role": "runtime",
                         "source": "sdk_trace",
                         "content": {"description": _first_not_none(record.get("description"), data.get("description")),
                                     "usage": _first_not_none(record.get("usage"), data.get("usage")),
                                     "last_tool_name": _first_not_none(record.get("last_tool_name"), data.get("last_tool_name"))},
                         "payload": record}
                event.update(_claude_code_agent_event_fields(task))
                events.append(event)

            continue

        if record_type == "AssistantMessage":
            message_id = record.get("message_id")
            task = _claude_code_task_context(record, tasks, tasks_by_parent_call)

            if message_id not in message_turns:
                message_turns[message_id] = len(message_turns) + 1

            turn = message_turns[message_id]
            agent_event_fields = _claude_code_agent_event_fields(task)

            if record.get("model") is not None:
                agent_event_fields["model"] = record.get("model")

            for block in record.get("content", []):
                if not isinstance(block, dict):
                    continue

                block_type = block.get("__type__")

                if block_type == "ThinkingBlock":
                    event = {"type": "reasoning",
                             "source": "sdk_trace",
                             "turn": turn,
                             "content": block.get("thinking", ""),
                             "payload": block}
                    event.update(agent_event_fields)
                    events.append(event)
                elif block_type == "ToolUseBlock":
                    call_id = block.get("id")
                    call_turns[call_id] = turn
                    event = {"type": "tool_call",
                             "source": "sdk_trace",
                             "turn": turn,
                             "call_id": call_id,
                             "name": block.get("name"),
                             "arguments": block.get("input", {}),
                             "payload": block}
                    event.update(agent_event_fields)
                    events.append(event)
                elif block_type == "TextBlock":
                    event = {"type": ("tool_preamble" if message_id in message_ids_with_tools else "assistant_text"),
                             "source": "sdk_trace",
                             "turn": turn,
                             "content": block.get("text", ""),
                             "payload": block}
                    event.update(agent_event_fields)
                    events.append(event)

            continue

        if record_type == "UserMessage":
            task = _claude_code_task_context(record, tasks, tasks_by_parent_call)
            agent_event_fields = _claude_code_agent_event_fields(task)

            for block in record.get("content", []):
                if not isinstance(block, dict):
                    continue

                block_type = block.get("__type__")

                if block_type == "ToolResultBlock":
                    call_id = block.get("tool_use_id")
                    content = block.get("content", "")
                    clem_result = _decode_clem_game_tool_result(content)
                    event = {"type": "tool_result",
                             "source": "sdk_trace",
                             "turn": call_turns.get(call_id),
                             "call_id": call_id,
                             "content": _claude_code_tool_result_text(content),
                             "payload": block}

                    if clem_result is not None:
                        context = clem_result["context"]
                        event["content"] = context["content"]
                        event["role"] = context.get("role")
                        event["reward"] = clem_result.get("reward")
                        event["done"] = clem_result.get("done")
                        event["metadata"] = clem_result.get("metadata")

                    event.update(agent_event_fields)
                    events.append(event)
                elif block_type == "TextBlock":
                    event = {"type": "message",
                             "source": "sdk_trace",
                             "role": "user",
                             "content": block.get("text", ""),
                             "payload": block}
                    event.update(agent_event_fields)
                    events.append(event)

            continue

        if record_type == "TaskNotificationMessage":
            task = _claude_code_task_context(record, tasks, tasks_by_parent_call)
            data = record.get("data") if isinstance(record.get("data"), dict) else {}
            event = {"type":
                     "message",
                     "kind": ("agent_completed"
                              if task is not None and task.get("scope") == "subagent" else "background_task_completed"),
                     "role":
                     "runtime",
                     "source":
                     "sdk_trace",
                     "content": {"status": _first_not_none(record.get("status"), data.get("status")),
                                 "summary": _first_not_none(record.get("summary"), data.get("summary")),
                                 "output_file": _first_not_none(record.get("output_file"), data.get("output_file")),
                                 "usage": _first_not_none(record.get("usage"), data.get("usage"))},
                     "payload":
                     record}
            event.update(_claude_code_agent_event_fields(task))
            events.append(event)
            continue

        if record_type == "ResultMessage":
            task = _claude_code_task_context(record, tasks, tasks_by_parent_call)

            if task is not None:
                event = {"type": "message",
                         "kind": ("agent_result" if task.get("scope") == "subagent" else "background_task_result"),
                         "role": "runtime",
                         "source": "sdk_trace",
                         "content": {key: record.get(key)
                                     for key in ("subtype", "duration_ms", "duration_api_ms", "is_error", "num_turns", "stop_reason",
                                                 "result", "usage", "model_usage", "errors", "terminal_reason",
                                                 ) if record.get(key) is not None},
                         "payload": record}
                event.update(_claude_code_agent_event_fields(task))
                events.append(event)
                continue

            result = {key: record.get(key) for key in ("subtype", "duration_ms", "duration_api_ms", "is_error", "num_turns",
                                                       "session_id", "stop_reason", "total_cost_usd", "usage", "model_usage",
                                                       "permission_denials", "errors", "api_error_status", "terminal_reason",
                                                       ) if record.get(key) is not None}

    thinking_token_total += previous_thinking_tokens
    protocol_events, protocol_capture = _claude_code_provider_events(trace_text)
    if any(event["type"] == "tool_definitions" for event in protocol_events):
        events = [event for event in events if event["type"] != "tool_definitions"]
    events.extend(protocol_events)
    # keep native settings readable even when the html truncates large request previews
    request_settings = []
    for event in protocol_events:
        body = event.get("payload")
        if event["type"] == "model_request" and isinstance(body, dict):
            settings = {key: body[key] for key in ("model", "thinking", "output_config", "max_tokens",
                                                  "temperature", "top_p", "top_k", "stream") if key in body}
            if settings and settings not in request_settings:
                request_settings.append(settings)
    if request_settings:
        runtime["observed_request_settings"] = request_settings
    partial = bool(not result or parse_errors or (metadata or {}).get("episode_timed_out")
                   or (metadata or {}).get("artifact_capture_status") == "partial"
                   or any(event["type"] == "trace_warning" for event in protocol_events))
    for capture in protocol_capture.values():
        capture["status"] = ("partial" if partial else "complete") if capture["count"] else "unavailable"

    for line in trace_text.splitlines():
        if line.startswith("agent_runtime_error:"):
            events.append({"type": "error",
                           "source": "runtime",
                           "content": line.removeprefix("agent_runtime_error:").strip()})

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    return {"schema_version": 1,
            "backend": "claude_code",
            "capture": {"agent_loop_instruction": {"status": "complete" if agent_loop_instruction is not None else "unavailable",
                                                   "source": "container" if agent_loop_instruction is not None else None},
                        **protocol_capture,
                        "sdk_messages": {"status": ("partial" if partial else "complete") if records else "unavailable",
                                         "source": "claude_agent_sdk",
                                         "count": len(records),
                                         "parse_errors": parse_errors},
                        "reasoning": {"status": (("partial" if partial else "complete") if any(event.get("type") == "reasoning" for event in events) else "unavailable"),
                                      "source": "ThinkingBlock"},
                        "thinking_tokens": {"status": "estimated" if thinking_token_updates else "unavailable",
                                            "estimated_total": thinking_token_total,
                                            "updates": thinking_token_updates}},
            "runtime": runtime,
            "result": result,
            "metadata": metadata or {},
            "events": events}


def _claude_code_provider_events(trace_text: str) -> tuple[list[dict], dict]:
    """Retain recorded transport separately from the SDK dialogue timeline."""
    records = [record for record in _provider_trace_records(trace_text)
               if record["type"] in {"model_request", "model_response"}]
    events, tool_sets, systems = [], [], []
    for record in records:
        events.append({**record, "source": "raw_upstream"})
        if record.get("parse_error"):
            events.append({"type": "trace_warning", "content": f"Could not parse recorded request JSON: {record['parse_error']}"})
        body = record.get("payload")
        if not isinstance(body, dict):
            continue
        tools, system = body.get("tools"), body.get("system")
        if isinstance(tools, list) and tools and tools not in tool_sets:
            tool_sets.append(tools)
            events.append({"type": "tool_definitions", "source": "raw_upstream", "kind": "native_schema", "tools": tools})
        if isinstance(system, (str, list)) and system and system not in systems:
            systems.append(system)
            events.append({"type": "instruction", "source": "raw_upstream", "kind": "native_harness", "content": system})
    requests = sum(record["type"] == "model_request" for record in records)
    responses = len(records) - requests
    starts = len(re.findall(r"(?m)^raw_upstream_(?:request|response)_\d+_start$", trace_text))
    if starts > len(records):
        events.append({"type": "trace_warning", "content": "An unfinished transport block remains in agent_trace.log; it could not be parsed as a complete API record."})
    if requests > responses:
        events.append({"type": "trace_warning", "content": f"Captured {requests} API requests and {responses} response records. An in-flight response may be missing; request/response pairing is not inferred."})
    return events, {"model_requests": {"count": requests, "source": "raw_upstream_request"},
                    "model_responses": {"count": responses, "source": "raw_upstream_response"},
                    "native_harness_instruction": {"count": len(systems), "source": "raw_upstream_request"}}


def _claude_code_tool_result_text(content: Any) -> Any:
    """Return readable text from a Claude Code tool-result content value."""

    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            return content

        if (isinstance(decoded, dict) and set(decoded) == {"result"} and isinstance(decoded["result"], str)):
            return decoded["result"]

        return content

    if not isinstance(content, list):
        return content

    text_parts = []

    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text_parts.append(item["text"])

    return "\n\n".join(text_parts) if text_parts else content


def _decode_clem_game_tool_result(content: Any) -> dict[str, Any] | None:
    """Decode an MCP game result returned to a Claude Code tool call."""

    if not isinstance(content, str):
        return None

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(result, dict):
        return None

    context = result.get("context")

    if not isinstance(context, dict) or not isinstance(context.get("content"), str):
        return None

    return result


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not ``None``."""

    return next((value for value in values if value is not None), None)


def _claude_code_task_contexts(records: list[dict[str,
                                                  Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index Claude SDK background tasks and subagents before parsing events."""

    tasks: dict[str, dict[str, Any]] = {}
    tasks_by_parent_call: dict[str, dict[str, Any]] = {}

    for record in records:
        if record.get("__type__") not in {"TaskStartedMessage", "TaskProgressMessage", "TaskNotificationMessage"}:
            continue

        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        task_id = _first_not_none(record.get("task_id"), data.get("task_id"))

        if not isinstance(task_id, str) or not task_id:
            continue

        parent_call_id = _first_not_none(record.get("tool_use_id"), data.get("tool_use_id"))
        task_type = _first_not_none(record.get("task_type"), data.get("task_type"))
        subagent_type = _first_not_none(record.get("subagent_type"), data.get("subagent_type"))
        is_subagent = task_type == "local_agent" or subagent_type is not None
        existing = tasks.get(task_id, {})
        task = {"task_id":
                task_id,
                "scope":
                "subagent" if is_subagent else existing.get("scope", "background_task"),
                "parent_call_id":
                _first_not_none(parent_call_id, existing.get("parent_call_id")),
                "description":
                _first_not_none(record.get("description"), data.get("description"), existing.get("description")),
                "task_type":
                _first_not_none(task_type, existing.get("task_type")),
                "subagent_type":
                _first_not_none(subagent_type, existing.get("subagent_type")),
                "prompt":
                _first_not_none(record.get("prompt"), data.get("prompt"), existing.get("prompt")),
                "status":
                _first_not_none(record.get("status"), data.get("status"), existing.get("status")),
                "summary":
                _first_not_none(record.get("summary"), data.get("summary"), existing.get("summary")),
                "output_file":
                _first_not_none(record.get("output_file"), data.get("output_file"), existing.get("output_file")),
                "usage":
                _first_not_none(record.get("usage"), data.get("usage"), existing.get("usage")),
                "last_tool_name":
                _first_not_none(record.get("last_tool_name"), data.get("last_tool_name"), existing.get("last_tool_name"))}
        tasks[task_id] = {key: value for key, value in task.items() if value is not None}

    for task in tasks.values():
        parent_call_id = task.get("parent_call_id")

        if isinstance(parent_call_id, str) and parent_call_id:
            tasks_by_parent_call[parent_call_id] = task

    return tasks, tasks_by_parent_call


def _claude_code_task_context(record: dict[str, Any], tasks: dict[str, dict[str, Any]],
                              tasks_by_parent_call: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve the task or subagent responsible for one SDK record."""

    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    task_id = _first_not_none(record.get("task_id"), data.get("task_id"))

    if isinstance(task_id, str) and task_id in tasks:
        return tasks[task_id]

    parent_call_id = record.get("parent_tool_use_id")

    if isinstance(parent_call_id, str):
        return tasks_by_parent_call.get(parent_call_id)

    lifecycle_call_id = _first_not_none(record.get("tool_use_id"), data.get("tool_use_id"))

    if isinstance(lifecycle_call_id, str):
        return tasks_by_parent_call.get(lifecycle_call_id)

    return None


def _claude_code_agent_event_fields(task: dict[str, Any] | None) -> dict[str, Any]:
    """Identify the delegated agent responsible for an SDK trace event."""

    if task is None or task.get("scope") != "subagent":
        return {}

    task_id = task.get("task_id")
    description = task.get("description")

    if description and task_id:
        return {"agent": f"{description} [{task_id}]"}

    return {"agent": description or task_id}


def _claude_code_trace_records(trace_text: str) -> list[dict[str, Any]]:
    """Safely parse Claude Agent SDK constructor representations."""

    records = []
    supported_types = {"AssistantMessage", "ResultMessage", "SystemMessage", "TaskNotificationMessage",
                       "TaskProgressMessage",
                       "TaskStartedMessage", "UserMessage"}

    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        constructor_name = line.split("(", 1)[0]

        if constructor_name not in supported_types:
            continue

        try:
            expression = ast.parse(line, mode="eval").body
            record = _constructor_ast_value(expression)
        except (SyntaxError, ValueError) as error:
            records.append({"__type__": "parse_error",
                            "line": line_number,
                            "content": f"Could not parse {constructor_name} on line {line_number}: {error}"})
            continue

        if isinstance(record, dict):
            records.append(record)

    return records


def _constructor_ast_value(node: ast.AST) -> Any:
    """Convert a constructor repr AST into JSON-serializable values."""

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_constructor_ast_value(item) for item in node.elts]

    if isinstance(node, ast.Tuple):
        return [_constructor_ast_value(item) for item in node.elts]

    if isinstance(node, ast.Dict):
        return {_constructor_ast_value(key): _constructor_ast_value(value) for key,
                value in zip(node.keys, node.values)}

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_constructor_ast_value(node.operand)

    if isinstance(node, ast.Name) and node.id in {"None", "True", "False"}:
        return {"None": None, "True": True, "False": False}[node.id]

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        value = {"__type__": node.func.id}

        if node.args:
            value["__args__"] = [_constructor_ast_value(item) for item in node.args]

        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Constructor repr contains unsupported keyword expansion")

            value[keyword.arg] = _constructor_ast_value(keyword.value)

        return value

    raise ValueError(f"Unsupported trace expression: {type(node).__name__}")
