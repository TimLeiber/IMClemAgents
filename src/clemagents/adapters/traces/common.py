import json
import re
from typing import Any


def _marked_trace_section(trace_text: str,
                          start_marker: str,
                          end_marker: str) -> str | None:
    """Return the first text section enclosed by exact trace markers."""

    pattern = re.compile(
        rf"{re.escape(start_marker)}\n(?P<content>.*?)\n{re.escape(end_marker)}",
        re.DOTALL,
    )
    match = pattern.search(trace_text)
    return match.group("content") if match is not None else None


def _provider_trace_records(trace_text: str) -> list[dict[str, Any]]:
    """Return recorded provider trace blocks in their original order."""

    records = []
    request_pattern = re.compile(
        r"raw_upstream_request_(?P<id>\d+)_start\n"
        r"path: (?P<path>[^\n]*)\n"
        r"(?P<body>.*?)\n"
        r"raw_upstream_request_(?P=id)_end",
        re.DOTALL
    )
    response_pattern = re.compile(
        r"raw_upstream_response_(?P<id>\d+)_start\n"
        r"path: (?P<path>[^\n]*)\n"
        r"status: (?P<status>\d+)\n"
        r"content_type: (?P<content_type>[^\n]*)\n"
        r"content_encoding: (?P<content_encoding>[^\n]*)\n"
        r"(?P<body>.*?)\n"
        r"raw_upstream_response_(?P=id)_end",
        re.DOTALL
    )
    instruction_pattern = re.compile(
        r"agent_loop_instruction_start\n"
        r"(?P<content>.*?)\n"
        r"agent_loop_instruction_end",
        re.DOTALL
    )

    for match in request_pattern.finditer(trace_text):
        raw = match.group("body")

        try:
            payload = json.loads(raw)
            parse_error = None
        except json.JSONDecodeError as error:
            payload = {}
            parse_error = str(error)

        records.append({
            "position": match.start(),
            "type": "model_request",
            "path": match.group("path"),
            "raw": raw,
            "payload": payload,
            "parse_error": parse_error
        })

    for match in response_pattern.finditer(trace_text):
        records.append({
            "position": match.start(),
            "type": "model_response",
            "path": match.group("path"),
            "status": int(match.group("status")),
            "content_type": match.group("content_type"),
            "content_encoding": match.group("content_encoding"),
            "raw": match.group("body")
        })

    for match in instruction_pattern.finditer(trace_text):
        records.append({
            "position": match.start(),
            "type": "agent_loop_instruction",
            "content": match.group("content")
        })

    return sorted(records, key=lambda record: record["position"])

