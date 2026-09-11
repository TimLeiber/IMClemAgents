import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from .schema import missing_agent_trace
from .common import _provider_trace_records, _marked_trace_section


def parse_openclaw_agent_trace(episode_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one OpenClaw session into the common agent-loop schema."""

    trace_path = episode_dir / "agent_trace.log"

    if not trace_path.exists():
        return missing_agent_trace("Native agent_trace.log is missing", "openclaw")

    if metadata is None:
        metadata_path = episode_dir / "agent_trace_meta.json"

        if metadata_path.exists():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = loaded_metadata if isinstance(loaded_metadata, dict) else None
            except (json.JSONDecodeError, OSError):
                metadata = None

    trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
    session, trajectory, session_parse_errors = _openclaw_trace_exports(trace_text)
    stdout_payload, stdout_parse_error = _openclaw_stdout_payload(trace_text)
    wire_records = [record for record in _provider_trace_records(trace_text)
                    if record["type"] in {"model_request", "model_response"}]
    wire_instructions, wire_tools = _openclaw_wire_configuration(wire_records)
    trajectory_instruction, trajectory_tools = _openclaw_trajectory_configuration(trajectory)
    events = []
    native_instructions = wire_instructions

    if not native_instructions and trajectory_instruction is not None:
        native_instructions = [trajectory_instruction]

    for instruction in native_instructions:
        events.append({"type": "instruction",
                       "kind": "native_harness",
                       "source": "wire_request" if wire_instructions else "openclaw_trajectory",
                       "content": instruction})

    agent_loop_instruction = _marked_trace_section(trace_text, "agent_loop_instruction_start",
                                                   "agent_loop_instruction_end")

    if agent_loop_instruction is None:
        agent_loop_instruction = _openclaw_agent_loop_instruction(session, trajectory)

    if agent_loop_instruction is not None:
        events.append({"type": "instruction",
                       "kind": "agent_loop",
                       "source": "container" if "agent_loop_instruction_start" in trace_text else "openclaw_session",
                       "content": agent_loop_instruction})

    tools = wire_tools or trajectory_tools

    if tools:
        events.append({"type": "tool_definitions",
                       "kind": "wire_schema" if wire_tools else "trajectory_inventory",
                       "source": "wire_request" if wire_tools else "openclaw_trajectory",
                       "tools": tools})

    session_events = _openclaw_session_events(session=session, agent_loop_instruction=agent_loop_instruction)
    wire_events = [] if session_events else _openclaw_wire_events(records=wire_records,
                                                                  agent_loop_instruction=agent_loop_instruction)
    events.extend(session_events or wire_events)

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

            if record["parse_error"] is not None:
                events.append({"type": "error",
                               "source": "wire_request",
                               "turn": model_request_count,
                               "content": f"Could not parse request JSON: {record['parse_error']}"})
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

            if record["status"] >= 400:
                events.append({"type": "error",
                               "source": "wire_response",
                               "turn": model_response_count,
                               "content": f"Provider response returned HTTP {record['status']}"})

    for parse_error in session_parse_errors:
        events.append({"type": "error", "source": "openclaw_session", "content": parse_error})

    if stdout_parse_error is not None:
        events.append({"type": "error", "source": "openclaw_stdout", "content": stdout_parse_error})

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
    system_prompt_report = _openclaw_system_prompt_report(stdout_payload)

    return {"schema_version": 1,
            "backend": "openclaw",
            "capture": {"agent_loop_instruction": {"status": "complete" if agent_loop_instruction is not None else "unavailable",
                                                   "source": "container" if "agent_loop_instruction_start" in trace_text else "openclaw_session"},
                        "native_harness_instruction": {"status":
                                                       "complete" if native_instructions else "unavailable",
                                                       "source": ("wire_request" if wire_instructions else
                                                                  ("openclaw_trajectory" if trajectory_instruction is not None else "not_exposed")),
                                                       **system_prompt_report},
                        "session_export": {"status": "complete" if session else "unavailable",
                                           "source": "openclaw_session_jsonl",
                                           "parse_errors": len(session_parse_errors)},
                        "trajectory_export": {"status": "complete" if trajectory else "unavailable",
                                              "source": "openclaw_trajectory_jsonl"},
                        "semantic_events": {"status": "complete" if session_events else ("partial" if wire_events else "unavailable"),
                                            "source": "openclaw_session_jsonl" if session_events else ("wire_request" if wire_events else None),
                                            "reasoning": reasoning_count,
                                            "tool_calls": tool_call_count,
                                            "tool_results": tool_result_count},
                        "tool_definitions": {"status": "complete" if tools else "unavailable",
                                             "source": "wire_request" if wire_tools else ("openclaw_trajectory" if trajectory_tools else None),
                                             "count": len(tools)},
                        "model_requests": {"status": "complete" if model_request_count else "unavailable",
                                           "source": "raw_upstream_request",
                                           "count": model_request_count},
                        "model_responses": {"status": "complete" if model_response_count else "unavailable",
                                            "source": "raw_upstream_response",
                                            "count": model_response_count}},
            "runtime": _openclaw_runtime(session, trajectory, stdout_payload),
            "result": _openclaw_result(session, trajectory, stdout_payload, metadata),
            "metadata": metadata or {},
            "events": events}


def _openclaw_trace_exports(trace_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Recover OpenClaw session and trajectory JSONL embedded in a trace."""

    blocks = []
    errors = []
    current_path = None
    current_records = []

    def finish_block() -> None:
        nonlocal current_path, current_records

        if current_path is not None:
            blocks.append((current_path, current_records))

        current_path = None
        current_records = []

    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        if line.startswith("openclaw_session: "):
            finish_block()
            current_path = line.removeprefix("openclaw_session: ").strip()
            continue

        if current_path is None or not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            if line.lstrip().startswith("{"):
                errors.append(f"Could not parse OpenClaw JSONL line {line_number}: {error}")

            finish_block()
            continue

        if isinstance(value, dict):
            current_records.append(value)
        else:
            errors.append(f"OpenClaw JSONL line {line_number} is not an object")

    finish_block()
    sessions = [records for path, records in blocks if not path.endswith(".trajectory.jsonl")]
    trajectories = [records for path, records in blocks if path.endswith(".trajectory.jsonl")]
    session = max(sessions, key=lambda records: sum(record.get("type") == "message" for record in records), default=[])
    trajectory = max(trajectories, key=len, default=[])
    return session, trajectory, errors


def _openclaw_stdout_payload(trace_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Decode OpenClaw's JSON command result from the combined trace."""

    match = re.search(r"openclaw_agent_stdout:\n(?P<content>.*?)(?:\nopenclaw_agent_stderr:|\Z)", trace_text,
                      flags=re.DOTALL)

    if match is None or not match.group("content").strip():
        return None, None

    try:
        payload = json.loads(match.group("content"))
    except json.JSONDecodeError as error:
        return None, f"Could not parse OpenClaw JSON output: {error}"

    if not isinstance(payload, dict):
        return None, "OpenClaw JSON output is not an object"

    return payload, None


def _openclaw_wire_configuration(records: list[dict[str, Any]]) -> tuple[list[str], list[Any]]:
    """Extract native instructions and tool schemas from OpenClaw requests."""

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

            content = _openclaw_content_text(message.get("content"))

            if content and content not in seen_instructions:
                seen_instructions.add(content)
                instructions.append(content)

    return instructions, tools


def _openclaw_trajectory_configuration(trajectory: list[dict[str, Any]]) -> tuple[str | None, list[Any]]:
    """Recover prompt and tool configuration exposed by OpenClaw trajectory data."""

    for record in trajectory:
        if record.get("type") != "context.compiled":
            continue

        data = record.get("data", {})

        if not isinstance(data, dict):
            return None, []

        system_prompt = data.get("systemPrompt")
        tools = data.get("tools") if isinstance(data.get("tools"), list) else []
        return system_prompt if isinstance(system_prompt, str) else None, tools

    return None, []


def _openclaw_agent_loop_instruction(session: list[dict[str, Any]], trajectory: list[dict[str, Any]]) -> str | None:
    """Recover the agent-loop instruction from native OpenClaw artifacts."""

    for record in trajectory:
        if record.get("type") not in {"prompt.submitted", "context.compiled"}:
            continue

        data = record.get("data", {})
        prompt = data.get("prompt") if isinstance(data, dict) else None

        if isinstance(prompt, str) and prompt:
            return prompt

    for record in session:
        if record.get("type") != "message":
            continue

        message = record.get("message", {})

        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        content = _openclaw_content_text(message.get("content"))

        if content:
            return content

    return None


def _openclaw_session_events(session: list[dict[str, Any]], agent_loop_instruction: str | None) -> list[dict[str, Any]]:
    """Convert OpenClaw session messages into ordered semantic events."""

    events = []
    call_turns = {}
    current_turn = 0
    synthetic_call_number = 0
    game_completed = False
    pending_prompt_error = None

    def append_prompt_error(record: dict[str, Any]) -> None:
        error = record.get("data", {})

        if game_completed:
            events.append({"type":
                           "termination",
                           "kind":
                           "after_game_completion",
                           "source":
                           "openclaw_session",
                           "content": ("OpenClaw stopped after clem_game reported done=true. "
                                       f"Native termination detail: {error.get('error', error)}"),
                           "payload":
                           record})
        else:
            events.append({"type": "error", "source": "openclaw_session", "content": error, "payload": record})

    for record in session:
        if record.get("type") == "custom" and record.get("customType") == "openclaw:prompt-error":
            pending_prompt_error = record
            continue

        if record.get("type") != "message":
            continue

        message = record.get("message", {})

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        blocks = message.get("content", [])
        blocks = blocks if isinstance(blocks, list) else []

        if role == "assistant":
            current_turn += 1
            has_tool_call = any(isinstance(block, dict) and block.get("type") == "toolCall" for block in blocks)

            for block in blocks:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type == "thinking":
                    content = block.get("thinking") or block.get("text") or ""

                    if content:
                        events.append({"type": "reasoning",
                                       "source": "openclaw_session",
                                       "turn": current_turn,
                                       "content": content,
                                       "payload": block})
                elif block_type == "text":
                    content = block.get("text", "")

                    if content.strip():
                        events.append({"type": "tool_preamble" if has_tool_call else "assistant_text",
                                       "source": "openclaw_session",
                                       "turn": current_turn,
                                       "content": content,
                                       "payload": block})
                elif block_type == "toolCall":
                    synthetic_call_number += 1
                    call_id = block.get("id") or f"openclaw-call-{synthetic_call_number}"
                    arguments = block.get("arguments")

                    if arguments is None:
                        arguments = _openclaw_json_value(block.get("partialArgs", {}))

                    call_turns[call_id] = current_turn
                    events.append({"type": "tool_call",
                                   "source": "openclaw_session",
                                   "turn": current_turn,
                                   "call_id": call_id,
                                   "name": block.get("name"),
                                   "arguments": arguments,
                                   "payload": block})

            if pending_prompt_error is not None:
                append_prompt_error(pending_prompt_error)
                pending_prompt_error = None

            continue

        content = _openclaw_content_text(message.get("content"))

        if role == "toolResult":
            call_id = message.get("toolCallId")
            events.append({"type": "tool_result",
                           "source": "openclaw_session",
                           "turn": call_turns.get(call_id),
                           "call_id": call_id,
                           "name": message.get("toolName"),
                           "content": content or message.get("details", {}),
                           "payload": message})
            game_completed = game_completed or _openclaw_tool_result_done(message)
            continue

        if role == "user" and content == agent_loop_instruction:
            continue

        if content:
            events.append({"type": "message",
                           "source": "openclaw_session",
                           "role": role,
                           "content": content,
                           "payload": message})

    if pending_prompt_error is not None:
        append_prompt_error(pending_prompt_error)

    return events


def _openclaw_wire_events(records: list[dict[str, Any]], agent_loop_instruction: str | None) -> list[dict[str, Any]]:
    """Recover semantic events from accumulated OpenClaw wire requests."""

    events = []
    seen_message_counts = Counter()
    call_turns = {}
    call_names = {}
    current_turn = 0
    synthetic_call_number = 0

    for record in records:
        if record.get("type") != "model_request":
            continue

        request = record.get("payload", {})

        if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
            continue

        request_counts = Counter()

        for message in request["messages"]:
            if not isinstance(message, dict):
                continue

            message_key = json.dumps(message, sort_keys=True, ensure_ascii=False)
            request_counts[message_key] += 1

            if request_counts[message_key] <= seen_message_counts[message_key]:
                continue

            role = message.get("role")

            if role in {"system", "developer"}:
                continue

            if role == "assistant":
                current_turn += 1
                tool_calls = message.get("tool_calls", [])
                tool_calls = tool_calls if isinstance(tool_calls, list) else []
                reasoning = message.get("reasoning_content")

                if reasoning is None:
                    reasoning = message.get("reasoning")

                reasoning_text = _openclaw_content_text(reasoning)

                if reasoning_text:
                    events.append({"type": "reasoning",
                                   "source": "wire_request",
                                   "turn": current_turn,
                                   "content": reasoning_text,
                                   "payload": reasoning})

                content = _openclaw_content_text(message.get("content"))

                if content.strip():
                    events.append({"type": "tool_preamble" if tool_calls else "assistant_text",
                                   "source": "wire_request",
                                   "turn": current_turn,
                                   "content": content,
                                   "payload": message.get("content")})

                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue

                    function = tool_call.get("function", {})
                    function = function if isinstance(function, dict) else {}
                    synthetic_call_number += 1
                    call_id = tool_call.get("id") or f"openclaw-wire-call-{synthetic_call_number}"
                    name = function.get("name") or tool_call.get("name")
                    arguments = function.get("arguments", tool_call.get("arguments", {}))
                    call_turns[call_id] = current_turn
                    call_names[call_id] = name
                    events.append({"type": "tool_call",
                                   "source": "wire_request",
                                   "turn": current_turn,
                                   "call_id": call_id,
                                   "name": name,
                                   "arguments": _openclaw_json_value(arguments),
                                   "payload": tool_call})

                continue

            if role == "tool":
                call_id = message.get("tool_call_id")
                content = message.get("content")
                events.append({"type": "tool_result",
                               "source": "wire_request",
                               "turn": call_turns.get(call_id),
                               "call_id": call_id,
                               "name": message.get("name") or call_names.get(call_id),
                               "content": _openclaw_json_value(content),
                               "payload": message})
                continue

            content = _openclaw_content_text(message.get("content"))

            if role == "user" and content == agent_loop_instruction:
                continue

            if content:
                events.append({"type": "message",
                               "source": "wire_request",
                               "role": role,
                               "content": content,
                               "payload": message})

        for message_key, count in request_counts.items():
            seen_message_counts[message_key] = max(seen_message_counts[message_key], count)

    return events


def _openclaw_tool_result_done(message: dict[str, Any]) -> bool:
    """Return whether an OpenClaw tool result completed the game."""

    details = message.get("details", {})
    structured = details.get("structuredContent") if isinstance(details, dict) else None

    if isinstance(structured, dict):
        return structured.get("done") is True

    content = _openclaw_content_text(message.get("content"))
    marker = "structuredContent:"

    if marker not in content:
        return False

    try:
        structured = json.loads(content.split(marker, 1)[1].strip())
    except json.JSONDecodeError:
        return False

    return isinstance(structured, dict) and structured.get("done") is True


def _openclaw_runtime(session: list[dict[str, Any]], trajectory: list[dict[str, Any]],
                      stdout_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Collect concise OpenClaw runtime details from native artifacts."""

    runtime = {}

    for record in session:
        record_type = record.get("type")

        if record_type == "session":
            runtime.update({key: record[key] for key in ("id", "timestamp", "cwd") if record.get(key) is not None})
        elif record_type == "model_change":
            runtime.update({key: record[key] for key in ("provider", "modelId") if record.get(key) is not None})
        elif record_type == "thinking_level_change" and record.get("thinkingLevel") is not None:
            runtime["thinking_level"] = record["thinkingLevel"]

    for record in trajectory:
        if record.get("type") != "trace.metadata":
            continue

        data = record.get("data", {})
        harness = data.get("harness", {}) if isinstance(data, dict) else {}
        model = data.get("model", {}) if isinstance(data, dict) else {}

        if isinstance(harness, dict):
            runtime["harness_version"] = harness.get("version")

        if isinstance(model, dict):
            runtime["model_api"] = model.get("api")
            runtime["reasoning_level"] = model.get("reasoningLevel")

        break

    stdout_meta = stdout_payload.get("meta", {}) if isinstance(stdout_payload, dict) else {}
    agent_meta = stdout_meta.get("agentMeta", {}) if isinstance(stdout_meta, dict) else {}

    if isinstance(stdout_meta, dict) and stdout_meta.get("durationMs") is not None:
        runtime["duration_ms"] = stdout_meta["durationMs"]

    if isinstance(agent_meta, dict):
        for source_key, target_key in (("sessionId", "session_id"), ("provider", "provider"), ("model", "model"),
                                       ("contextTokens", "context_tokens"), ("agentHarnessId", "agent_harness_id"),
                                       ("promptTokens", "prompt_tokens"),
                                       ):
            if agent_meta.get(source_key) is not None:
                runtime[target_key] = agent_meta[source_key]

    return {key: value for key, value in runtime.items() if value is not None}


def _openclaw_result(session: list[dict[str, Any]], trajectory: list[dict[str, Any]],
                     stdout_payload: dict[str, Any] | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Collect OpenClaw completion status and usage details."""

    result = {}
    stdout_meta = stdout_payload.get("meta", {}) if isinstance(stdout_payload, dict) else {}
    agent_meta = stdout_meta.get("agentMeta", {}) if isinstance(stdout_meta, dict) else {}
    payloads = stdout_payload.get("payloads", []) if isinstance(stdout_payload, dict) else []

    if isinstance(stdout_meta, dict):
        for source_key, target_key in (("durationMs", "duration_ms"), ("aborted", "aborted"),):
            if stdout_meta.get(source_key) is not None:
                result[target_key] = stdout_meta[source_key]

    if isinstance(agent_meta, dict):
        for source_key, target_key in (("usage", "usage"), ("lastCallUsage", "last_call_usage"),
                                       ("contextBudgetStatus", "context_budget_status"),
                                       ):
            if agent_meta.get(source_key) is not None:
                result[target_key] = agent_meta[source_key]

    final_text = [payload.get("text") for payload in payloads if isinstance(payload, dict) and payload.get("text")]

    if final_text:
        result["final_text"] = "\n".join(final_text)

    for record in trajectory:
        if record.get("type") not in {"trace.artifacts", "session.ended"}:
            continue

        data = record.get("data", {})

        if not isinstance(data, dict):
            continue

        for key in ("status", "finalStatus", "aborted", "externalAbort", "timedOut", "idleTimedOut",
                    "timedOutDuringCompaction", "timedOutDuringToolExecution", "promptError", "promptErrorSource",
                    "usage", "compactionCount",
                    ):
            if data.get(key) is not None:
                result[key] = data[key]

    if isinstance(metadata, dict):
        for source_key, target_key in (("success", "success"), ("runtime_error", "runtime_error"), ("game_completed",
                                                                                                    "game_completed"),
                                       ("terminated_after_game", "terminated_after_game"), ("returncode", "returncode"),
                                       ):
            if metadata.get(source_key) is not None:
                result[target_key] = metadata[source_key]

    game_completed = any(record.get("type") == "message" and isinstance(record.get("message"), dict)
                         and record["message"].get("role") == "toolResult" and _openclaw_tool_result_done(record["message"])
                         for record in session)

    if game_completed:
        result["game_completed"] = True
        result["success"] = True

    if game_completed and result.get("externalAbort") is True:
        cleanup = {}

        for key in ("status", "finalStatus", "aborted", "externalAbort", "promptError", "promptErrorSource",):
            if key in result:
                cleanup[key] = result.pop(key)

        result["status"] = "completed"
        result["terminal_reason"] = "terminated_after_game_completion"
        result["cleanup"] = cleanup

    return result


def _openclaw_system_prompt_report(stdout_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return non-content metadata OpenClaw reports for its system prompt."""

    meta = stdout_payload.get("meta", {}) if isinstance(stdout_payload, dict) else {}
    report = meta.get("systemPromptReport", {}) if isinstance(meta, dict) else {}
    prompt = report.get("systemPrompt", {}) if isinstance(report, dict) else {}

    if not isinstance(prompt, dict):
        return {}

    return {target_key: prompt[source_key]
            for source_key, target_key in (("chars", "reported_chars"), ("hash", "reported_hash"),
                                           ("projectContextChars", "reported_project_context_chars"),
                                           ("nonProjectContextChars", "reported_non_project_context_chars"),
                                           ) if prompt.get(source_key) is not None}


def _openclaw_content_text(content: Any) -> str:
    """Return readable text from OpenClaw message content blocks."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")

                if text is not None:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)

        return "\n".join(parts)

    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)

    return str(content)


def _openclaw_json_value(value: Any) -> Any:
    """Decode an OpenClaw JSON argument string when possible."""

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
