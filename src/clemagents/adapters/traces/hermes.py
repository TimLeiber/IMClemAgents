import json
import re
from pathlib import Path
from typing import Any
from .schema import missing_agent_trace
from .common import _provider_trace_records, _marked_trace_section


def parse_hermes_agent_trace(episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one Hermes session into the common agent-loop schema."""

    if (episode_dir / "hermes_observer.jsonl").exists():
        return _observer_trace(episode_dir, metadata)

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return missing_agent_trace("Native agent_trace.log is missing", "hermes")

    if metadata is None:
        metadata_path = episode_dir / "agent_trace_meta.json"

        if metadata_path.exists():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                metadata = None

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    session, session_parse_errors = _hermes_session_export(episode_dir)
    wire_records = [record for record in _provider_trace_records(trace_text)
                    if record["type"] in {"model_request", "model_response"}]
    native_instructions, wire_tools = _hermes_wire_configuration(wire_records)
    events = []

    for instruction in native_instructions:
        events.append({"type": "instruction",
                       "kind": "native_harness",
                       "source": "wire_request",
                       "content": instruction})

    agent_loop_instruction = _marked_trace_section(trace_text, "agent_loop_instruction_start",
                                                   "agent_loop_instruction_end")

    if agent_loop_instruction is None:
        agent_loop_instruction = _hermes_query_instruction(trace_text)

    if agent_loop_instruction is not None:
        events.append({"type": "instruction",
                       "kind": "agent_loop",
                       "source": "container",
                       "content": agent_loop_instruction})

    cli_tools = _hermes_tool_inventory(trace_text)
    tools = wire_tools or cli_tools

    if tools:
        events.append({"type": "tool_definitions",
                       "kind": "wire_schema" if wire_tools else "cli_inventory",
                       "source": "wire_request" if wire_tools else "hermes_cli",
                       "tools": tools})

    runtime = _hermes_runtime(trace_text, session)
    result = _hermes_result(trace_text, session)
    native_instruction_found = bool(native_instructions)

    if session is not None:
        session_events, session_has_native_instruction = _hermes_session_events(session=session,
                                                                                agent_loop_instruction=agent_loop_instruction)
        events.extend(session_events)
        native_instruction_found = (native_instruction_found or session_has_native_instruction)
        semantic_source = "hermes_session_export"
    else:
        # cli output may contain reasoning but omit calls without --verbose
        # prefer recorded api events when no native session export is available
        wire_events = _chat_completion_wire_events(wire_records)
        if wire_events:
            events.extend(wire_events)
            semantic_source = "chat_completions_wire"
        else:
            events.extend(_hermes_cli_events(trace_text))
            semantic_source = "hermes_cli"

    model_request_count = 0
    model_response_count = 0

    for record in wire_records:
        if record["type"] == "model_request":
            model_request_count += 1
            events.append({"type": "model_request",
                           "source": "wire_request",
                           "turn": model_request_count,
                           "path": record["path"],
                           "raw": record["raw"],
                           "payload": record["payload"]})
        else:
            model_response_count += 1
            events.append({"type": "model_response",
                           "source": "wire_response",
                           "turn": model_response_count,
                           "path": record["path"],
                           "status": record["status"],
                           "content_type": record["content_type"],
                           "content_encoding": record["content_encoding"],
                           "raw": record["raw"]})

    for parse_error in session_parse_errors:
        events.append({"type": "error", "source": "hermes_session_export", "content": parse_error})

    for match in re.finditer(r"agent_runtime_error:\s*(?P<content>.*)", trace_text):
        events.append({"type": "error", "source": "runtime", "content": match.group("content").strip()})

    deduplicated_events = []
    seen_instructions = set()

    for event in events:
        if event.get("type") == "instruction":
            instruction_key = (event.get("kind"), event.get("content"))

            if instruction_key in seen_instructions:
                continue

            seen_instructions.add(instruction_key)

        deduplicated_events.append(event)

    events = deduplicated_events

    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence

    reasoning_count = sum(event.get("type") == "reasoning" for event in events)
    tool_call_count = sum(event.get("type") == "tool_call" for event in events)
    tool_result_count = sum(event.get("type") == "tool_result" for event in events)

    return {"schema_version": 1,
            "backend": "hermes",
            "capture": {"agent_loop_instruction": {"status": "complete" if agent_loop_instruction is not None else "unavailable",
                                                   "source": "container" if agent_loop_instruction is not None else None},
                        "native_harness_instruction": {"status": "complete" if native_instruction_found else "unavailable",
                                                       "source": "wire_request" if native_instruction_found else "not_exposed"},
                        "session_export": {"status": "complete" if session is not None else "unavailable",
                                           "source": "hermes_sessions_export",
                                           "parse_errors": len(session_parse_errors)},
                        "semantic_events": {"status": "complete" if reasoning_count or tool_call_count else "unavailable",
                                            "source": semantic_source,
                                            "reasoning": reasoning_count,
                                            "tool_calls": tool_call_count,
                                            "tool_results": tool_result_count},
                        "tool_definitions": {"status": "complete" if wire_tools else ("partial" if cli_tools else "unavailable"),
                                             "source": "wire_request" if wire_tools else ("hermes_cli" if cli_tools else None),
                                             "count": len(tools)},
                        "model_requests": {"status": "complete" if model_request_count else "unavailable",
                                           "source": "raw_upstream_request",
                                           "count": model_request_count},
                        "model_responses": {"status": "complete" if model_response_count else "unavailable",
                                            "source": "raw_upstream_response",
                                            "count": model_response_count}},
            "runtime": runtime,
            "result": result,
            "metadata": metadata or {},
            "events": events}


def _observer_trace(episode_dir: Path, metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize native observations without pretending they are raw HTTP bytes."""
    events, requests, responses, pending = [], 0, 0, set()
    path = episode_dir / "hermes_observer.jsonl"
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("event is not an object")
        except (ValueError, json.JSONDecodeError) as error:
            events.append({"type": "trace_warning", "content": f"Incomplete native event at line {line_number}: {error}"})
            continue
        hook = record.get("hook")
        identity = (record.get("session_id"), record.get("api_request_id"))
        common = {"source": "hermes_observer", "agent_id": record.get("task_id"),
                  "session_id": record.get("session_id"), "turn": record.get("turn_id"),
                  "request_id": record.get("api_request_id"), "timestamp": record.get("timestamp")}
        if hook == "pre_api_request":
            requests += 1
            pending.add(identity)
            body = (record.get("request") or {}).get("body", {})
            events.append({**common, "type": "model_request", "payload": body, "raw": record})
            if record.get("capture_warning"):
                events.append({**common, "type": "trace_warning", "content": record["capture_warning"]})
            if requests == 1:
                instructions, tools = _hermes_wire_configuration([{"type": "model_request", "payload": body}])
                events.extend({**common, "type": "instruction", "kind": "native_harness", "content": text}
                              for text in instructions)
                if tools:
                    events.append({**common, "type": "tool_definitions", "tools": tools})
        elif hook == "post_api_request":
            responses += 1
            pending.discard(identity)
            events.append({**common, "type": "model_response", "payload": record.get("response"), "raw": record})
            message = record.get("assistant_message") or (record.get("response") or {}).get("assistant_message") or {}
            provider_data = message.get("provider_data")
            provider_data = provider_data if isinstance(provider_data, dict) else {}
            reasoning = (message.get("reasoning_content") or message.get("reasoning")
                         or provider_data.get("reasoning_content") or provider_data.get("reasoning"))
            if reasoning:
                events.append({**common, "type": "reasoning", "content": reasoning})
            if message.get("content"):
                events.append({**common, "type": "tool_preamble" if message.get("tool_calls") else "assistant_text",
                               "content": message["content"]})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or call
                events.append({**common, "type": "tool_call", "call_id": call.get("id"),
                               "name": function.get("name"), "arguments": _hermes_json_value(function.get("arguments", ""))})
        elif hook == "post_tool_call":
            events.append({**common, "type": "tool_result", "name": record.get("tool_name"),
                           "call_id": record.get("tool_call_id"), "content": record.get("result"),
                           "is_error": record.get("status") not in {None, "ok", "success"}, "raw": record})
        elif hook in {"api_request_error", "unsupported_control"}:
            pending.discard(identity)
            events.append({**common, "type": "error", "content": record.get("content") or record.get("error"),
                           "raw": record})
        elif hook in {"subagent_start", "subagent_stop", "episode_closed"}:
            events.append({**common, "type": hook, "raw": record})

    # full native dumps retain schemas and kwargs before sdk serialization, not wire bytes
    dumps = sorted((episode_dir / "hermes_home").rglob("request_dump_*.json"))
    native_tool_sets = []
    for dump in dumps:
        try:
            payload = json.loads(dump.read_text(encoding="utf-8"))
            events.append({"type": "native_request_dump", "source": "hermes_request_dump", "payload": payload})
            request = payload.get("request") if isinstance(payload, dict) else None
            body = request.get("body") if isinstance(request, dict) else None
            tools = body.get("tools") if isinstance(body, dict) else None
            if isinstance(tools, list) and tools and tools not in native_tool_sets:
                native_tool_sets.append(tools)
                events.append({"type": "tool_definitions", "source": "hermes_request_dump", "kind": "native_schema",
                               "artifact": dump.name, "tools": tools})
        except (OSError, ValueError) as error:
            events.append({"type": "trace_warning", "content": f"Unreadable Hermes request dump: {error}"})
    if native_tool_sets:
        # prefer intact schemas over the observer's potentially truncated first snapshot
        events = [event for event in events if not (event["type"] == "tool_definitions"
                                                   and event.get("source") == "hermes_observer")]
    note = ("Native observations preserve completed normalized responses, including exposed reasoning. "
            "They are not raw HTTP/SSE capture; interrupted streams, SDK-internal retries and auxiliary API calls "
            "outside the conversation hooks (such as session-title generation) may be unavailable. "
            "Observer snapshots can truncate schemas; full native request dumps are retained separately.")
    if pending:
        note += f" {len(pending)} observed request(s) have no terminal event."
    events.append({"type": "trace_warning", "content": note})
    for sequence, event in enumerate(events, 1):
        event["sequence"] = sequence
    return {"schema_version": 1, "backend": "hermes", "metadata": metadata or {}, "events": events,
            "capture": {"model_requests": {"status": "partial" if requests else "unavailable", "count": requests,
                                           "source": "hermes_observer", "full_native_dumps": len(dumps)},
                        "model_responses": {"status": "partial" if responses else "unavailable", "count": responses,
                                            "source": "hermes_observer", "pending_requests": len(pending)},
                        "tool_definitions": {"status": "partial" if any(e["type"] == "tool_definitions" for e in events)
                                                         else "unavailable",
                                             "source": "hermes_request_dump" if native_tool_sets else "hermes_observer"}}}


def _hermes_session_export(episode_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the single Hermes session exported for one episode."""

    export_path = episode_dir / "hermes_session_export.jsonl"

    if not export_path.exists():
        return None, []

    sessions = []
    errors = []

    for line_number, line in enumerate(export_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"Could not parse session-export line {line_number}: {error}")
            continue

        if isinstance(value, dict):
            sessions.append(value)
        else:
            errors.append(f"Session-export line {line_number} is not an object")

    if not sessions:
        return None, errors

    return sessions[-1], errors


def _hermes_wire_configuration(records: list[dict[str, Any]]) -> tuple[list[str], list[Any]]:
    """Extract native instructions and tool schemas from Hermes requests."""

    instructions = []
    seen_instructions = set()
    tools = []

    for record in records:
        if record.get("type") != "model_request":
            continue

        request = record.get("payload", {})

        if not isinstance(request, dict):
            continue

        if not tools and isinstance(request.get("tools"), list):
            tools = request["tools"]

        for message in request.get("messages", []):
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
                continue

            content = _hermes_content_text(message.get("content"))

            if content and content not in seen_instructions:
                seen_instructions.add(content)
                instructions.append(content)

    return instructions, tools


def _hermes_query_instruction(trace_text: str) -> str | None:
    """Recover the user query printed by Hermes before initialization."""

    stdout = _hermes_chat_stdout(trace_text)
    match = re.search(r"(?:^|\n)Query: (?P<content>.*?)(?:\nInitializing agent\.\.\.)", stdout, flags=re.DOTALL)
    return match.group("content").strip() if match is not None else None


def _hermes_tool_inventory(trace_text: str) -> list[dict[str, str]]:
    """Return the names Hermes reports in its final enabled tool set."""

    stdout = _hermes_chat_stdout(trace_text)
    match = re.search(r"Final tool selection \(\d+ tools\): (?P<tools>[^\n]+)", stdout)

    if match is None:
        return []

    return [{"name": name.strip()} for name in match.group("tools").split(",") if name.strip()]


def _hermes_runtime(trace_text: str, session: dict[str, Any] | None) -> dict[str, Any]:
    """Collect concise Hermes runtime details from trace and session data."""

    stdout = _hermes_chat_stdout(trace_text)
    runtime = {}
    model_match = re.search(r"AI Agent initialized with model: (?P<model>[^\n]+)", stdout)
    context_match = re.search(r"Context limit: (?P<context>[\d,]+) tokens \(compress at (?P<percent>\d+)% = (?P<threshold>[\d,]+)\)", stdout)
    command_match = re.search(r"hermes_chat_command:\nhermes chat --provider (?P<provider>\S+) --model (?P<model>\S+).*?(?=\n)", trace_text)

    if model_match is not None:
        runtime["model"] = model_match.group("model").strip()

    if command_match is not None:
        runtime["provider"] = command_match.group("provider")
        runtime.setdefault("model", command_match.group("model"))

    if context_match is not None:
        runtime["context_limit"] = int(context_match.group("context").replace(",", ""))
        runtime["compression_percent"] = int(context_match.group("percent"))
        runtime["compression_threshold"] = int(context_match.group("threshold").replace(",", ""))

    if "--yolo" in trace_text:
        runtime["permission_mode"] = "yolo"

    if session is not None:
        for source_key, target_key in (("id", "session_id"), ("model", "session_model"),
                                       ("provider", "session_provider"), ("started_at", "started_at"),
                                       ("ended_at", "ended_at"), ("end_reason", "end_reason"),
                                       ):
            if session.get(source_key) is not None:
                runtime[target_key] = session[source_key]

    return runtime


def _hermes_result(trace_text: str, session: dict[str, Any] | None) -> dict[str, Any]:
    """Collect the Hermes completion status without treating cleanup as failure."""

    result = {}
    success_matches = list(re.finditer(r"(?:^|\n)success:\s*(True|False)", trace_text))
    timeout_match = re.search(r"hermes_timeout:\s*(?P<timeout>[^\n]+)", trace_text)

    if success_matches:
        result["success"] = success_matches[-1].group(1) == "True"

    if timeout_match is not None:
        result["terminal_reason"] = "timeout"
        result["timeout"] = timeout_match.group("timeout")
    elif "Interrupted during API call" in trace_text:
        result["terminal_reason"] = "terminated_after_game"

    if session is not None:
        for key in ("end_reason", "finish_reason", "message_count", "tool_call_count", "input_tokens", "output_tokens",
                    "total_cost",
                    ):
            if session.get(key) is not None:
                result[key] = session[key]

    return result


def _hermes_session_events(session: dict[str, Any],
                           agent_loop_instruction: str | None) -> tuple[list[dict[str, Any]], bool]:
    """Convert Hermes session messages into ordered semantic events."""

    events = []
    call_turns = {}
    current_turn = 0
    synthetic_call_number = 0
    has_native_instruction = False

    for message in session.get("messages", []):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = _hermes_content_text(message.get("content"))

        if role in {"system", "developer"}:
            if content:
                has_native_instruction = True
                events.append({"type": "instruction",
                               "kind": "native_harness",
                               "source": "hermes_session_export",
                               "role": role,
                               "content": content,
                               "payload": message})
            continue

        if role == "assistant":
            current_turn += 1
            reasoning = _hermes_content_text(message.get("reasoning") or message.get("reasoning_content"))
            tool_calls = message.get("tool_calls") or []

            if reasoning:
                events.append({"type": "reasoning",
                               "source": "hermes_session_export",
                               "turn": current_turn,
                               "content": reasoning,
                               "payload": {"reasoning_details": message.get("reasoning_details")}})

            if content:
                events.append({"type": "tool_preamble" if tool_calls else "assistant_text",
                               "source": "hermes_session_export",
                               "turn": current_turn,
                               "content": content,
                               "payload": message})

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                function = tool_call.get("function", {})
                function = function if isinstance(function, dict) else {}
                synthetic_call_number += 1
                call_id = tool_call.get("id") or f"hermes-call-{synthetic_call_number}"
                call_turns[call_id] = current_turn
                events.append({"type": "tool_call",
                               "source": "hermes_session_export",
                               "turn": current_turn,
                               "call_id": call_id,
                               "name": function.get("name") or tool_call.get("name"),
                               "arguments": _hermes_json_value(function.get("arguments", tool_call.get("arguments", {}))),
                               "payload": tool_call})
            continue

        if role == "tool":
            call_id = message.get("tool_call_id")
            events.append({"type": "tool_result",
                           "source": "hermes_session_export",
                           "turn": call_turns.get(call_id),
                           "call_id": call_id,
                           "name": message.get("tool_name"),
                           "content": _hermes_readable_tool_result(_hermes_json_value(message.get("content"))),
                           "payload": message})
            continue

        if role == "user" and content == agent_loop_instruction:
            continue

        if content:
            events.append({"type": "message",
                           "source": "hermes_session_export",
                           "role": role,
                           "content": content,
                           "payload": message})

    return events, has_native_instruction


def _chat_completion_wire_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover semantic events from OpenAI Chat Completions wire traffic.

    This is a protocol parser, not a harness parser.  It is used when a
    harness-native transcript was not flushed before the completed game was
    torn down.  Assistant output is reconstructed from streamed response
    deltas; tool results are recovered from the next request's accumulated
    message history.
    """

    events = []
    call_turns: dict[str, int] = {}
    call_names: dict[str, str | None] = {}
    seen_tool_results = set()
    current_turn = 0

    for record in records:
        if not str(record.get("path", "")).split("?", 1)[0].rstrip("/").endswith("/chat/completions"):
            continue

        if record.get("type") == "model_request":
            request = record.get("payload", {})
            messages = request.get("messages", []) if isinstance(request, dict) else []

            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue

                call_id = message.get("tool_call_id")
                content = message.get("content")
                result_key = (call_id, json.dumps(content, sort_keys=True, ensure_ascii=False, default=str),)

                if result_key in seen_tool_results:
                    continue

                seen_tool_results.add(result_key)
                events.append({"type": "tool_result",
                               "source": "wire_request",
                               "turn": call_turns.get(call_id),
                               "call_id": call_id,
                               "name": message.get("name") or call_names.get(call_id),
                               "content": _chat_completion_readable_tool_result(content),
                               "payload": message})

            continue

        if record.get("type") != "model_response":
            continue

        current_turn += 1
        response_events = _chat_completion_response_events(record, current_turn)

        for event in response_events:
            if event.get("type") != "tool_call":
                continue

            call_id = event.get("call_id")

            if call_id:
                call_turns[call_id] = current_turn
                call_names[call_id] = event.get("name")

        events.extend(response_events)

    return events


def _chat_completion_response_events(record: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    """Aggregate one streamed Chat Completions response into dialogue events."""

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}

    for line in str(record.get("raw", "")).splitlines():
        if not line.startswith("data: "):
            continue

        data = line.removeprefix("data: ").strip()

        if not data or data == "[DONE]":
            continue

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", []) if isinstance(chunk, dict) else []

        if not choices or not isinstance(choices[0], dict):
            continue

        delta = choices[0].get("delta", {})

        if not isinstance(delta, dict):
            continue

        reasoning = delta.get("reasoning")

        if reasoning is None:
            reasoning = delta.get("reasoning_content")

        if reasoning:
            reasoning_parts.append(str(reasoning))

        content = delta.get("content")

        if isinstance(content, str) and content:
            content_parts.append(content)

        for fragment in delta.get("tool_calls") or []:
            if not isinstance(fragment, dict):
                continue

            index = fragment.get("index", 0)
            call = tool_calls.setdefault(index, {"id": None,
                                                 "name": None,
                                                 "arguments": "",
                                                 "type": fragment.get("type", "function")})
            function = fragment.get("function", {})
            function = function if isinstance(function, dict) else {}

            if fragment.get("id"):
                call["id"] = fragment["id"]

            if function.get("name"):
                call["name"] = function["name"]

            if function.get("arguments") is not None:
                call["arguments"] += str(function["arguments"])

    events = []
    reasoning = "".join(reasoning_parts).strip()
    content = "".join(content_parts).strip()

    if reasoning:
        events.append({"type": "reasoning", "source": "wire_response", "turn": turn, "content": reasoning})

    if content:
        events.append({"type": "tool_preamble" if tool_calls else "assistant_text",
                       "source": "wire_response",
                       "turn": turn,
                       "content": content})

    for index in sorted(tool_calls):
        call = tool_calls[index]
        events.append({"type": "tool_call",
                       "source": "wire_response",
                       "turn": turn,
                       "call_id": call.get("id") or f"chat-wire-call-{turn}-{index}",
                       "name": call.get("name"),
                       "arguments": _hermes_json_value(call.get("arguments", "")),
                       "payload": call})

    return events


def _chat_completion_readable_tool_result(content: Any) -> Any:
    """Return textual tool content while keeping media as typed placeholders."""

    if not isinstance(content, list):
        return _hermes_readable_tool_result(_hermes_json_value(content))

    parts = []

    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue

        block_type = block.get("type", "unknown")

        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type in {"image", "image_url"}:
            parts.append("[image content supplied to the model]")
        elif block_type == "audio":
            parts.append("[audio content supplied to the model]")
        else:
            parts.append(f"[{block_type} content supplied to the model]")

    return "\n\n".join(part for part in parts if part)


def _hermes_cli_events(trace_text: str) -> list[dict[str, Any]]:
    """Parse reasoning and tool exchanges from Hermes verbose output."""

    records = _hermes_cli_records(_hermes_chat_stdout(trace_text))
    events = []
    pending_calls = []
    current_turn = 0
    turn_open = False
    call_number = 0

    for record in records:
        record_type = record["type"]

        if record_type == "reasoning":
            if not turn_open:
                current_turn += 1
                turn_open = True

            events.append({"type": "reasoning",
                           "source": "hermes_cli",
                           "turn": current_turn,
                           "content": _hermes_unwrap_prose(record["content"])})
            continue

        if record_type == "tool_call":
            if not turn_open:
                current_turn += 1
                turn_open = True

            call_number += 1
            call_id = f"hermes-call-{call_number}"
            pending_calls.append((call_id, current_turn, record.get("name")))
            events.append({"type": "tool_call",
                           "source": "hermes_cli",
                           "turn": current_turn,
                           "call_id": call_id,
                           "name": record.get("name"),
                           "arguments": _hermes_json_value(record.get("arguments", "")),
                           "payload": {"display": record.get("arguments", "")}})
            continue

        if pending_calls:
            call_id, turn, name = pending_calls.pop(0)
        else:
            call_id, turn, name = None, current_turn or None, None

        parsed_result = _hermes_json_value(record.get("content", ""))
        events.append({"type": "tool_result",
                       "source": "hermes_cli",
                       "turn": turn,
                       "call_id": call_id,
                       "name": name,
                       "content": _hermes_readable_tool_result(parsed_result),
                       "payload": parsed_result})

        if not pending_calls:
            turn_open = False

    return events


def _hermes_cli_records(stdout: str) -> list[dict[str, Any]]:
    """Return Hermes reasoning, tool-call, and tool-result display blocks."""

    records = []
    reasoning_pattern = re.compile(r"^┌─ Reasoning [^\n]*\n(?P<content>.*?)\n└[─]+┘", flags=re.MULTILINE | re.DOTALL)
    tool_call_pattern = re.compile(r"^[ \t]*📞 Tool \d+: (?P<name>[^\s(]+)\([^\n]*\)\n"
                                   r"[ \t]*Args: (?P<arguments>.*?)(?=\n[ \t]*(?:┊|✅|📞|⚡))", flags=re.MULTILINE | re.DOTALL)
    tool_result_pattern = re.compile(r"^[ \t]*✅ Tool \d+ completed[^\n]*\n"
                                     r"[ \t]*Result: (?P<content>.*?)"
                                     r"(?=\n(?:[ \t]*\n)?(?:┌─ Reasoning|[ \t]*┊|[ \t]*📞 Tool|⚡ Interrupt)|\Z)", flags=re.MULTILINE | re.DOTALL)

    for match in reasoning_pattern.finditer(stdout):
        records.append({"position": match.start(), "type": "reasoning", "content": match.group("content").strip()})

    for match in tool_call_pattern.finditer(stdout):
        records.append({"position": match.start(),
                        "type": "tool_call",
                        "name": match.group("name"),
                        "arguments": match.group("arguments").strip()})

    for match in tool_result_pattern.finditer(stdout):
        records.append({"position": match.start(), "type": "tool_result", "content": match.group("content").strip()})

    return sorted(records, key=lambda record: record["position"])


def _hermes_chat_stdout(trace_text: str) -> str:
    """Return only Hermes chat stdout from the combined adapter trace."""

    match = re.search(r"hermes_chat_stdout:\n(?P<content>.*?)(?:\nhermes_chat_stderr:|\Z)", trace_text, flags=re.DOTALL)
    return match.group("content") if match is not None else trace_text


def _hermes_content_text(content: Any) -> str:
    """Return readable text from Hermes/OpenAI message content."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")

                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))

        return "\n".join(parts)

    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)

    return str(content)


def _hermes_json_value(value: Any) -> Any:
    """Decode JSON strings, including Hermes terminal-wrapped JSON."""

    if not isinstance(value, str):
        return value

    stripped = value.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        unwrapped = re.sub(r"\n[ \t]+", " ", stripped)

        try:
            return json.loads(unwrapped)
        except json.JSONDecodeError:
            return stripped


def _hermes_readable_tool_result(result: Any) -> Any:
    """Prefer the meaningful result while retaining the full payload separately."""

    if not isinstance(result, dict):
        return result

    structured = result.get("structuredContent")

    if isinstance(structured, dict):
        context = structured.get("context")

        if isinstance(context, dict):
            sections = []

            if "done" in structured:
                sections.append(f"done: {structured['done']}")

            if structured.get("reward") is not None:
                sections.append(f"reward: {structured['reward']}")

            role = context.get("role")
            content = context.get("content")

            if role:
                sections.append(f"{role}: {content}")
            elif content is not None:
                sections.append(str(content))

            if sections:
                return "\n\n".join(sections)

    if "output" in result:
        sections = [str(result["output"])]

        if result.get("exit_code") is not None:
            sections.append(f"exit_code: {result['exit_code']}")

        if result.get("error"):
            sections.append(f"error: {result['error']}")

        return "\n\n".join(sections)

    if result.get("success") is True and "data" in result:
        return result["data"]

    return result


def _hermes_unwrap_prose(content: str) -> str:
    """Remove Rich terminal wrapping while preserving paragraph breaks."""

    paragraphs = re.split(r"\n\s*\n", content.strip())
    return "\n\n".join(" ".join(line.strip() for line in paragraph.splitlines()) for paragraph in paragraphs)
