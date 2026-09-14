"""Small JSON contract shared by adapter parsers and the transcript renderer."""

import json
import re
from typing import Any


def missing_agent_trace(reason: str, backend: str = "unknown") -> dict[str, Any]:
    """Describe unavailable capture without inventing dialogue.

    Args:
        reason: explanation displayed in the transcript
        backend: adapter identifier when known

    Returns:
        a versioned trace containing a visible warning
    """
    return {"schema_version": 1, "backend": backend,
            "capture": {"events": {"status": "unavailable", "reason": reason}},
            "events": [{"sequence": 1, "type": "trace_warning", "content": reason}]}


def normalize_agent_trace(trace: dict[str, Any], backend: str = "unknown") -> dict[str, Any]:
    """Validate the common envelope and fill optional version and sequence fields.

    Args:
        trace: adapter output containing an ordered events list
        backend: adapter identifier used when the trace omits it

    Returns:
        a new trace with version, backend and event sequence numbers

    Raises:
        ValueError: malformed envelope, event or unsupported schema version
    """
    if not isinstance(trace, dict):
        raise ValueError("Agent-trace parsers must return a dictionary containing events")
    if not trace:
        return missing_agent_trace("The adapter returned no trace; native capture is unavailable", backend)
    version = trace.get("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported agent-trace schema_version: {version!r}")
    backend = trace.get("backend", backend)
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("Agent-trace backend must be a non-empty string")
    if not isinstance(trace.get("events"), list):
        raise ValueError("Agent-trace events must be an ordered list")
    if "capture" in trace and not isinstance(trace["capture"], dict):
        raise ValueError("Agent-trace capture must be a dictionary when provided")

    events = []
    previous = 0
    for index, event in enumerate(trace["events"]):
        if not isinstance(event, dict) or not isinstance(event.get("type"), str) or not event["type"].strip():
            raise ValueError(f"events[{index}] must be a dictionary with a non-empty type")
        sequence = event.get("sequence", previous + 1)
        if type(sequence) is not int or sequence <= previous:
            raise ValueError(f"events[{index}].sequence must be an increasing positive integer")
        events.append({**event, "sequence": sequence})
        previous = sequence

    # retain older parser errors as visible warnings when rendering saved traces
    reason = trace.get("parser_error")
    warning = f"Trace parsing failed: {reason}"
    if reason and not any(event["type"] == "trace_warning" and event.get("content") == warning for event in events):
        events.append({"sequence": previous + 1, "type": "trace_warning", "content": warning})
    elif not events:
        events = missing_agent_trace("No agent-loop events were captured", backend)["events"]
    capture = dict(trace.get("capture", {}))
    metadata = trace.get("metadata") or {}
    artifact_status = metadata.get("artifact_capture_status") if isinstance(metadata, dict) else None
    if artifact_status in {"partial", "unavailable"}:
        reason = ("Native artifact finalization did not complete; already-written files were recovered. "
                  "Capture is partial: the last request, response or tool result and session export may be missing."
                  if artifact_status == "partial" else
                  "No native artifact directory was recovered; available console output does not establish complete capture.")
        capture["artifacts"] = {"status": artifact_status, "reason": reason}
        if not any(event["type"] == "trace_warning" and event.get("content") == reason for event in events):
            events.append({"sequence": events[-1]["sequence"] + 1, "type": "trace_warning", "content": reason})
    return {**trace, "schema_version": version, "backend": backend, "events": events, "capture": capture}



def _marked_trace_section(trace_text: str, start_marker: str, end_marker: str) -> str | None:
    """Return the first text section enclosed by exact trace markers."""

    pattern = re.compile(rf"{re.escape(start_marker)}\n(?P<content>.*?)\n{re.escape(end_marker)}", re.DOTALL)
    match = pattern.search(trace_text)
    return match.group("content") if match is not None else None


def _provider_trace_records(trace_text: str) -> list[dict[str, Any]]:
    """Return recorded provider trace blocks in their original order."""

    records = []
    request_pattern = re.compile(r"raw_upstream_request_(?P<id>\d+)_start\n"
                                 r"path: (?P<path>[^\n]*)\n"
                                 r"(?P<body>.*?)\n"
                                 r"raw_upstream_request_(?P=id)_end", re.DOTALL)
    response_pattern = re.compile(r"raw_upstream_response_(?P<id>\d+)_start\n"
                                  r"path: (?P<path>[^\n]*)\n"
                                  r"status: (?P<status>\d+)\n"
                                  r"content_type: (?P<content_type>[^\n]*)\n"
                                  r"content_encoding: (?P<content_encoding>[^\n]*)\n"
                                  r"(?P<body>.*?)\n"
                                  r"raw_upstream_response_(?P=id)_end", re.DOTALL)
    instruction_pattern = re.compile(r"agent_loop_instruction_start\n"
                                     r"(?P<content>.*?)\n"
                                     r"agent_loop_instruction_end", re.DOTALL)

    for match in request_pattern.finditer(trace_text):
        raw = match.group("body")

        try:
            payload = json.loads(raw)
            parse_error = None
        except json.JSONDecodeError as error:
            payload = {}
            parse_error = str(error)

        records.append({"position": match.start(),
                        "type": "model_request",
                        "path": match.group("path"),
                        "raw": raw,
                        "payload": payload,
                        "parse_error": parse_error})

    for match in response_pattern.finditer(trace_text):
        records.append({"position": match.start(),
                        "type": "model_response",
                        "path": match.group("path"),
                        "status": int(match.group("status")),
                        "content_type": match.group("content_type"),
                        "content_encoding": match.group("content_encoding"),
                        "raw": match.group("body")})

    for match in instruction_pattern.finditer(trace_text):
        records.append({"position": match.start(), "type": "agent_loop_instruction", "content": match.group("content")})

    return sorted(records, key=lambda record: record["position"])
