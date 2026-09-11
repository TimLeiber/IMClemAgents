import json
import re
from pathlib import Path
from typing import Any
from .schema import missing_agent_trace
from .common import _provider_trace_records


def parse_codex_agent_trace(episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Codex trace into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return missing_agent_trace("Native agent_trace.log is missing", "codex")

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    records = _provider_trace_records(trace_text)
    events = []
    seen_input_items = set()
    seen_output_item_ids = set()
    model_request_count = 0
    model_response_count = 0
    has_agent_loop_instruction = False
    has_native_instruction = False

    for record in records:
        record_type = record["type"]

        if record_type == "agent_loop_instruction":
            has_agent_loop_instruction = True
            events.append({"type": "instruction",
                           "kind": "agent_loop",
                           "source": "container",
                           "content": record["content"]})
            continue

        if record_type == "model_request":
            model_request_count += 1
            request = record["payload"]
            turn = model_request_count
            events.append({"type": "model_request",
                           "source": "wire_request",
                           "turn": turn,
                           "path": record["path"],
                           "raw": record["raw"],
                           "payload": request})

            if record["parse_error"] is not None:
                events.append({"type": "error",
                               "source": "wire_request",
                               "turn": turn,
                               "content": f"Could not parse request JSON: {record['parse_error']}"})

            native_instruction = request.get("instructions")

            if native_instruction and not has_native_instruction:
                has_native_instruction = True
                events.append({"type": "instruction",
                               "kind": "native_harness",
                               "source": "wire_request",
                               "turn": turn,
                               "content": native_instruction})

            if request.get("tools") and turn == 1:
                events.append({"type": "tool_definitions",
                               "source": "wire_request",
                               "turn": turn,
                               "tools": request["tools"]})

            request_events, found_agent_loop_instruction = _codex_request_events(request=request, turn=turn,
                                                                                 seen_input_items=seen_input_items,
                                                                                 seen_output_item_ids=seen_output_item_ids)
            has_agent_loop_instruction = (has_agent_loop_instruction or found_agent_loop_instruction)
            events.extend(request_events)
            continue

        if record_type == "model_response":
            model_response_count += 1
            turn = model_response_count
            events.append({"type": "model_response",
                           "source": "wire_response",
                           "turn": turn,
                           "path": record["path"],
                           "status": record["status"],
                           "content_type": record["content_type"],
                           "content_encoding": record["content_encoding"],
                           "raw": record["raw"]})
            events.extend(_codex_response_events(record=record, turn=turn, seen_output_item_ids=seen_output_item_ids))

            if record["status"] >= 400:
                events.append({"type": "error",
                               "source": "wire_response",
                               "turn": turn,
                               "content": f"Provider response returned HTTP {record['status']}"})

    events.extend(_codex_runtime_errors(trace_text, final_turn=model_response_count or None))

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    return {"schema_version": 1,
            "backend": "codex",
            "capture": {"agent_loop_instruction": {"status": "complete" if has_agent_loop_instruction else "unavailable",
                                                   "source": "container" if "agent_loop_instruction_start" in trace_text else "wire_request"},
                        "native_harness_instruction": {"status": "complete" if has_native_instruction else "unavailable",
                                                       "source": "wire_request"},
                        "model_requests": {"status": "complete" if model_request_count else "unavailable",
                                           "source": "raw_upstream_request",
                                           "count": model_request_count},
                        "model_responses": {"status": "complete" if model_response_count else "unavailable",
                                            "source": "raw_upstream_response",
                                            "count": model_response_count}},
            "events": events}


def _codex_request_events(request: dict[str, Any], turn: int, seen_input_items: set[str],
                          seen_output_item_ids: set[str]) -> tuple[list[dict[str, Any]], bool]:
    """Extract previously unseen input items from one Responses request."""

    events = []
    found_agent_loop_instruction = False

    for item in request.get("input", []):
        item_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        item_id = item.get("id") if isinstance(item, dict) else None

        if item_key in seen_input_items or item_id in seen_output_item_ids:
            continue

        seen_input_items.add(item_key)
        item_type = item.get("type") if isinstance(item, dict) else None

        if item_type == "function_call_output":
            events.append({"type": "tool_result",
                           "source": "wire_request",
                           "turn": turn,
                           "call_id": item.get("call_id"),
                           "content": _codex_tool_result_text(item.get("output")),
                           "payload": item})
            continue

        if item_type == "function_call":
            continue

        if item_type == "reasoning":
            events.append({"type": "reasoning",
                           "source": "wire_request",
                           "turn": turn,
                           "content": _codex_reasoning_text(item),
                           "payload": item})
            continue

        if item_type != "message":
            events.append({"type": "message", "source": "wire_request", "turn": turn, "payload": item})
            continue

        role = item.get("role")
        text = _codex_message_text(item.get("content"))
        event_type = "message"
        kind = None

        if role == "developer":
            event_type = "instruction"
            kind = "runtime"
        elif _is_agent_loop_instruction(text):
            event_type = "instruction"
            kind = "agent_loop"
            found_agent_loop_instruction = True
        elif text.startswith("<environment_context>"):
            event_type = "instruction"
            kind = "environment_context"

        event = {"type": event_type,
                 "source": "wire_request",
                 "turn": turn,
                 "role": role,
                 "content": text,
                 "payload": item}

        if kind is not None:
            event["kind"] = kind

        events.append(event)

    return events, found_agent_loop_instruction


def _codex_response_events(record: dict[str, Any], turn: int, seen_output_item_ids: set[str]) -> list[dict[str, Any]]:
    """Extract agent-loop events from one Responses API stream."""

    events = []
    completed_items = []
    started_tool_items = {}
    terminal_failure = None
    tool_item_types = {"function_call", "openrouter:web_search", "web_search_call"}

    for line in record["raw"].splitlines():
        if not line.startswith("data: "):
            continue

        try:
            event = json.loads(line.removeprefix("data: "))
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "response.output_item.added":
            item = event.get("item", {})

            if item.get("type") in tool_item_types and item.get("id"):
                started_tool_items[item["id"]] = item

        elif event_type == "response.output_item.done":
            completed_items.append(event.get("item", {}))

        elif event_type in {"response.failed", "response.incomplete"}:
            response = event.get("response", {})
            terminal_failure = (response.get("error") or response.get("incomplete_details") or {"message": "Provider response did not complete"})
        elif event_type == "response.completed":
            response = event.get("response", {})

            if response.get("status") in {"failed", "incomplete"}:
                terminal_failure = (response.get("error") or response.get("incomplete_details") or {"message": "Provider response did not complete"})

    # some compatible providers serialize terminal function calls before the
    # reasoning and preamble that caused them, while codex executes them last
    function_calls = [item for item in completed_items if item.get("type") == "function_call"]
    completed_items = [item for item in completed_items if item.get("type") != "function_call"] + function_calls

    for item_index, item in enumerate(completed_items):
        item_type = item.get("type")
        item_id = item.get("id")

        if item_id:
            seen_output_item_ids.add(item_id)

        if item_type in {"openrouter:web_search", "web_search_call"}:
            action = item.get("action", {})
            query = action.get("query") if isinstance(action, dict) else None
            sources = action.get("sources", []) if isinstance(action, dict) else []
            source_urls = [source.get("url") for source in sources if isinstance(source, dict) and source.get("url")]
            events.append({"type": "tool_call",
                           "kind": "hosted_web_search",
                           "source": "wire_response",
                           "turn": turn,
                           "call_id": item_id,
                           "name": "web_search",
                           "arguments": {"query": query},
                           "payload": item})
            events.append({"type":
                           "tool_result",
                           "kind":
                           "hosted_web_search",
                           "source":
                           "wire_response",
                           "turn":
                           turn,
                           "call_id":
                           item_id,
                           "name":
                           "web_search",
                           "content": ("Search-result context was supplied to the model internally.\n"
                                       "The retrieved text is not exposed in this trace.\n\n"
                                       "Captured source metadata:\n" + "\n".join(source_urls)),
                           "payload": {"status": item.get("status"),
                                       "sources": sources}})
            continue

        if item_type == "function_call":
            events.append({"type": "tool_call",
                           "source": "wire_response",
                           "turn": turn,
                           "call_id": item.get("call_id"),
                           "name": item.get("name"),
                           "arguments": _codex_tool_arguments(item.get("arguments")),
                           "payload": item})
            continue

        if item_type == "message":
            has_following_tool_call = any(later_item.get("type") in tool_item_types for later_item in completed_items[item_index + 1:])

            for content in item.get("content", []):
                content_type = content.get("type")

                if content_type == "output_text":
                    events.append({"type": "tool_preamble" if has_following_tool_call else "assistant_text",
                                   "source": "wire_response",
                                   "turn": turn,
                                   "content": content.get("text", ""),
                                   "payload": content})
                elif content_type == "reasoning":
                    events.append({"type": "reasoning",
                                   "source": "wire_response",
                                   "turn": turn,
                                   "content": _codex_reasoning_text(content),
                                   "payload": content})

            continue

        if item_type == "reasoning":
            events.append({"type": "reasoning",
                           "source": "wire_response",
                           "turn": turn,
                           "content": _codex_reasoning_text(item),
                           "payload": item})
            continue

        events.append({"type": "assistant_output", "source": "wire_response", "turn": turn, "payload": item})

    completed_item_ids = {item.get("id") for item in completed_items if item.get("id")}
    incomplete_tool_items = [item for item_id, item in started_tool_items.items() if item_id not in completed_item_ids]

    for item in incomplete_tool_items:
        item_type = item.get("type")
        item_id = item.get("id")
        action = item.get("action", {})
        query = action.get("query") if isinstance(action, dict) else None
        tool_name = item.get("name")
        arguments = _codex_tool_arguments(item.get("arguments"))

        if item_type in {"openrouter:web_search", "web_search_call"}:
            tool_name = "web_search"
            arguments = {"query": query}

        events.append({"type": "tool_call",
                       "kind": "provider_hosted",
                       "source": "wire_response",
                       "turn": turn,
                       "call_id": item.get("call_id") or item_id,
                       "name": tool_name or item_type,
                       "arguments": arguments,
                       "status": "failed" if terminal_failure is not None else "in_progress",
                       "payload": item})

        if terminal_failure is not None:
            error_code = terminal_failure.get("code")
            error_message = terminal_failure.get("message", "Tool execution failed")
            content = ": ".join(value for value in (error_code, error_message) if value)
            events.append({"type": "tool_result",
                           "kind": "provider_hosted",
                           "source": "wire_response",
                           "turn": turn,
                           "call_id": item.get("call_id") or item_id,
                           "name": tool_name or item_type,
                           "status": "failed",
                           "content": content or "Tool execution failed",
                           "payload": terminal_failure})

    if terminal_failure is not None and not incomplete_tool_items:
        error_code = terminal_failure.get("code")
        error_message = terminal_failure.get("message", "Provider response failed")
        events.append({"type": "error",
                       "kind": "provider_response",
                       "source": "wire_response",
                       "turn": turn,
                       "status": "failed",
                       "content": ": ".join(value for value in (error_code, error_message) if value),
                       "payload": terminal_failure})

    return events


def _codex_runtime_errors(trace_text: str, final_turn: int | None = None) -> list[dict[str, Any]]:
    """Extract adapter-level runtime errors from the native trace."""

    errors = []
    seen_errors = set()

    for match in re.finditer(r"agent_runtime_error: (?P<content>.*)", trace_text):
        content = match.group("content")
        seen_errors.add(content)
        errors.append({"type": "error", "source": "native_cli", "content": content})

    for line in trace_text.splitlines():
        if not line.startswith("{"):
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if record.get("type") == "turn.failed":
            error = record.get("error", {})
            content = error.get("message", "Harness turn failed")

            if content in seen_errors:
                continue

            seen_errors.add(content)
            errors.append({"type": "error",
                           "kind": "turn_failed",
                           "source": "native_cli",
                           "turn": final_turn,
                           "status": "failed",
                           "content": content,
                           "payload": record})
            continue

        item = record.get("item", {})

        if (record.get("type") == "item.completed" and isinstance(item, dict) and item.get("type") == "error"):
            content = item.get("message", "Harness warning")

            if content in seen_errors:
                continue

            seen_errors.add(content)
            errors.append({"type": "trace_warning",
                           "kind": "harness",
                           "source": "native_cli",
                           "content": content,
                           "payload": record})

    return errors


def _codex_message_text(content: Any) -> str:
    """Return the text content from one Responses API message item."""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    return "\n".join(str(part.get("text") or part.get("input_text") or "") for part in content if isinstance(part, dict))


def _codex_reasoning_text(item: Any) -> str:
    """Return provider-exposed reasoning text from one Responses item."""

    if isinstance(item, str):
        return item

    if not isinstance(item, dict):
        return ""

    text_parts = []

    for field_name in ("content", "summary"):
        content = item.get(field_name, [])

        if isinstance(content, str):
            text_parts.append(content)
            continue

        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue

            text = part.get("text") or part.get("reasoning_text")

            if text:
                text_parts.append(str(text))

    return "\n\n".join(text_parts)


def _codex_tool_arguments(arguments: Any) -> Any:
    """Parse function arguments when they were encoded as JSON text."""

    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _codex_tool_result_text(output: Any) -> str:
    """Extract the readable game message from a Codex tool result."""

    if not isinstance(output, str):
        return str(output)

    prefix, separator, possible_json = output.partition("\nOutput:\n")

    if not separator:
        return output

    try:
        result = json.loads(possible_json)
    except json.JSONDecodeError:
        return output

    if not isinstance(result, dict):
        return output

    tool_result = result.get("result")

    if set(result) == {"result"} and isinstance(tool_result, str):
        return tool_result

    context = result.get("context")

    if not isinstance(context, dict):
        return output

    if set(result) == {"context"} and isinstance(context.get("content"), str):
        return context["content"]

    sections = []

    if prefix:
        sections.append(prefix)

    if "done" in result:
        sections.append(f"done: {result['done']}")

    if result.get("reward") is not None:
        sections.append(f"reward: {result['reward']}")

    role = context.get("role")
    content = context.get("content")

    if role:
        sections.append(f"{role}: {content}")
    elif content is not None:
        sections.append(str(content))

    return "\n\n".join(sections)


def _is_agent_loop_instruction(text: str) -> bool:
    """Return whether a user message is the universal game-loop instruction."""

    return text.startswith("You are connected to a game environment through MCP tools.")
