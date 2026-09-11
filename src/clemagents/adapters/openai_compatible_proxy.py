import json
import threading
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import urllib3


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# The host pipeline owns the episode deadline. The compatibility proxy only
# bounds connection establishment; model inference may legitimately take
# longer than any fixed adapter-level read timeout.
UPSTREAM_REQUEST_TIMEOUT = (30, None)
INFERENCE_PATH_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/messages",
    "/responses",
)


def _is_inference_request(method: str, path: str) -> bool:
    """Return whether a provider request can start new model inference."""

    request_path = urlsplit(path).path.rstrip("/")
    return method == "POST" and request_path.endswith(INFERENCE_PATH_SUFFIXES)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def _prepare_request_body(
    body: bytes,
    path: str,
    completion_path: Path,
    request_body_overrides: dict[str, Any],
    upstream_model: str | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> tuple[bytes, Any]:
    """Apply transport options without changing harness tool definitions."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, {}

    if not isinstance(payload, dict):
        return body, {}

    _write_completion_marker_from_tool_results(payload, completion_path)

    request_path = urlsplit(path).path.rstrip("/")
    is_anthropic_messages = request_path.endswith("/messages")

    if upstream_model is not None:
        payload["model"] = upstream_model

    # Generation controls are protocol-neutral.  Apply them to both
    # OpenAI-compatible and Anthropic-shaped requests before forwarding to the
    # selected model provider. Provider-specific request overrides remain
    # restricted to OpenAI-compatible requests below.
    if generation_overrides:
        payload = _deep_merge(payload, generation_overrides)

    # Registry extra_body fields follow the OpenAI-compatible wire format.
    # Do not add those provider extensions to Anthropic Messages requests.
    if request_body_overrides and not is_anthropic_messages:
        payload = _deep_merge(payload, request_body_overrides)

        reasoning_override = request_body_overrides.get("reasoning")
        if (
            isinstance(reasoning_override, dict)
            and reasoning_override.get("enabled") is False
        ):
            payload["reasoning"] = {"enabled": False}

    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), None


def _write_completion_marker_from_tool_results(
    payload: dict[str, Any],
    completion_path: Path,
) -> None:
    """Record game completion observed in a Responses API tool result."""

    if completion_path.exists():
        return

    tool_results = []
    input_items = payload.get("input")

    if isinstance(input_items, list):
        tool_results.extend(item.get("output")
                            for item in input_items
                            if isinstance(item, dict)
                            and item.get("type") == "function_call_output")

    messages = payload.get("messages")

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue

            content = message.get("content")

            if message.get("role") == "tool":
                tool_results.append(content)

            if isinstance(content, list):
                tool_results.extend(block.get("content")
                                    for block in content
                                    if isinstance(block, dict)
                                    and block.get("type") == "tool_result")

    def reports_completion(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("done") is True or any(reports_completion(item)
                                                     for item in value.values())

        if isinstance(value, list):
            return any(reports_completion(item) for item in value)

        if isinstance(value, str):
            return '"done":true' in "".join(value.split())

        return False

    for tool_result in tool_results:
        if not reports_completion(tool_result):
            continue

        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion_path.write_text(
            json.dumps({"done": True, "source": "model_proxy"}),
            encoding="utf-8",
        )
        return


def _normalize_sse_output_order(events: list[dict[str, Any]]) -> None:
    """Make completed Responses output follow the order items were emitted.

    Some OpenAI-compatible servers stream a reasoning item before a function
    call but assign both items output index zero and later serialize the final
    response as ``function_call, reasoning``. Stateless clients then replay
    that inverted history on the next request, placing the reasoning between
    the call and its result. Preserve the observable generation order instead.
    """

    emitted_items = []

    for event in events:
        if event.get("type") != "response.output_item.added":
            continue

        item = event.get("item")
        item_id = item.get("id") if isinstance(item, dict) else None

        if (
            isinstance(item_id, str)
            and all(existing_id != item_id
                    for existing_id, _ in emitted_items)
        ):
            emitted_items.append((item_id, event.get("output_index")))

    if len(emitted_items) < 2:
        return

    declared_indexes = [output_index
                        for _, output_index in emitted_items]

    if (
        all(isinstance(output_index, int)
            for output_index in declared_indexes)
        and sorted(declared_indexes) == list(range(len(emitted_items)))
    ):
        # A conforming provider's declared indexes are authoritative even if
        # independent items happen to be observed out of order on the wire.
        item_ids = [
            item_id
            for item_id, _ in sorted(
                emitted_items,
                key=lambda emitted_item: emitted_item[1],
            )
        ]
    else:
        # Duplicate or contradictory indexes cannot define a valid response
        # order. Fall back to the order in which generation first exposed the
        # items, which preserves reasoning that preceded a function call.
        item_ids = [item_id for item_id, _ in emitted_items]

    output_indexes = {
        item_id: output_index
        for output_index, item_id in enumerate(item_ids)
    }

    for event in events:
        item = event.get("item")
        item_id = item.get("id") if isinstance(item, dict) else None

        if not isinstance(item_id, str):
            item_id = event.get("item_id")

        if isinstance(item_id, str) and item_id in output_indexes:
            event["output_index"] = output_indexes[item_id]

        if event.get("type") != "response.completed":
            continue

        response = event.get("response")
        output = response.get("output") if isinstance(response, dict) else None

        if not isinstance(output, list):
            continue

        original_positions = {
            id(item): position
            for position, item in enumerate(output)
        }
        output.sort(
            key=lambda item: output_indexes.get(
                item.get("id") if isinstance(item, dict) else None,
                len(output_indexes) + original_positions[id(item)],
            )
        )


def _normalize_response_output_order(body: bytes) -> bytes:
    """Repair contradictory Responses SSE indexes without changing tools."""

    decoded_body = body.decode("utf-8", errors="replace")
    response_lines = decoded_body.splitlines(keepends=True)
    has_sse_data = any(
        line.rstrip("\r\n").startswith("data: ")
        for line in response_lines
    )

    if not has_sse_data:
        return body

    parsed_events = []

    for line in response_lines:
        stripped = line.rstrip("\r\n")

        if not stripped.startswith("data: "):
            continue

        try:
            event = json.loads(stripped[6:])
        except json.JSONDecodeError:
            continue

        if isinstance(event, dict):
            parsed_events.append(event)

    _normalize_sse_output_order(parsed_events)
    event_iterator = iter(parsed_events)
    rewritten_lines = []

    for line in response_lines:
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]

        if not stripped.startswith("data: "):
            rewritten_lines.append(line)
            continue

        try:
            json.loads(stripped[6:])
        except json.JSONDecodeError:
            rewritten_lines.append(line)
            continue

        event = next(event_iterator)
        rewritten_lines.append(
            f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}{newline}"
        )

    return "".join(rewritten_lines).encode("utf-8")


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self,
                 target_origin: str,
                 completion_path: Path,
                 request_body_overrides: dict[str, Any],
                 generation_overrides: dict[str, Any],
                 verify_tls: bool,
                 upstream_model: str | None,
                 trace_responses: bool,
                 trace_requests: bool):
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.target_origin = target_origin
        self.completion_path = completion_path
        self.request_body_overrides = request_body_overrides
        self.generation_overrides = generation_overrides
        self.verify_tls = verify_tls
        self.upstream_model = upstream_model
        self.trace_responses = trace_responses
        self.trace_requests = trace_requests
        self.response_trace_lock = threading.Lock()
        self.response_trace_count = 0
        self.trace_records: list[str] = []

    def trace_request(self,
                      path: str,
                      body: bytes) -> None:
        if not self.trace_requests:
            return

        request_text = body.decode("utf-8", errors="replace")

        with self.response_trace_lock:
            self.response_trace_count += 1
            trace_id = self.response_trace_count
            trace_record = (
                f"\nraw_upstream_request_{trace_id}_start\n"
                f"path: {path}\n"
                f"{request_text}\n"
                f"raw_upstream_request_{trace_id}_end"
            )
            self.trace_records.append(trace_record)
            print(trace_record, flush=True)

    def trace_response(self,
                       path: str,
                       status_code: int,
                       content_type: str,
                       content_encoding: str,
                       body: bytes) -> None:
        if not self.trace_responses:
            return

        response_text = body.decode("utf-8", errors="replace")

        with self.response_trace_lock:
            self.response_trace_count += 1
            trace_id = self.response_trace_count
            trace_record = (
                f"\nraw_upstream_response_{trace_id}_start\n"
                f"path: {path}\n"
                f"status: {status_code}\n"
                f"content_type: {content_type}\n"
                f"content_encoding: {content_encoding}\n"
                f"{response_text}\n"
                f"raw_upstream_response_{trace_id}_end"
            )
            self.trace_records.append(trace_record)
            print(trace_record, flush=True)

    def captured_trace(self) -> str:
        """Return request and response records in their observed order."""

        with self.response_trace_lock:
            return "\n".join(self.trace_records)


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _proxy(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b""

        body, _ = _prepare_request_body(
            body,
            self.path,
            self.server.completion_path,
            self.server.request_body_overrides,
            self.server.upstream_model,
            self.server.generation_overrides,
        )

        # A completed game result is immutable. Refuse any new inference at
        # this generic provider boundary while the adapter finalizes its native
        # artifacts; do not let a harness begin a post-game model turn. This
        # check follows request inspection so a completion first visible in a
        # submitted tool-result payload also gates that same request.
        if (self.server.completion_path.exists()
                and _is_inference_request(self.command, self.path)):
            payload = json.dumps({
                "error": {
                    "message": "The game is complete; no further model inference was forwarded.",
                    "type": "game_completed",
                }
            }).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return

        self.server.trace_request(self.path, body)
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
        }
        target_url = f"{self.server.target_origin}{self.path}"

        try:
            upstream = requests.request(
                self.command,
                target_url,
                headers=headers,
                data=body or None,
                stream=True,
                timeout=UPSTREAM_REQUEST_TIMEOUT,
                verify=self.server.verify_tls,
            )
        except requests.RequestException as error:
            payload = json.dumps({"error": f"OpenAI-compatible proxy failed: {error}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return

        self.send_response(upstream.status_code)

        for name, value in upstream.headers.items():
            if name.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(name, value)

        self.send_header("Connection", "close")
        self.end_headers()

        response_chunks = []

        try:
            upstream.raw.decode_content = False

            while True:
                chunk = upstream.raw.read(65536)

                if not chunk:
                    break

                response_chunks.append(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.trace_response(
                self.path,
                upstream.status_code,
                upstream.headers.get("Content-Type", ""),
                upstream.headers.get("Content-Encoding", ""),
                b"".join(response_chunks),
            )
            upstream.close()
            self.close_connection = True

    do_DELETE = _proxy
    do_GET = _proxy
    do_PATCH = _proxy
    do_POST = _proxy
    do_PUT = _proxy

    def log_message(self, format: str, *args: Any) -> None:
        return


class OpenAICompatibleProxy(AbstractContextManager["OpenAICompatibleProxy"]):
    """Local protocol-preserving proxy for OpenAI-compatible model servers."""

    def __init__(self,
                 target_base_url: str,
                 completion_path: Path,
                 request_body_overrides: dict[str, Any] | None = None,
                 generation_overrides: dict[str, Any] | None = None,
                 verify_tls: bool = True,
                 upstream_model: str | None = None,
                 trace_responses: bool = True,
                 trace_requests: bool = False):
        parsed = urlsplit(target_base_url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid OpenAI-compatible base URL: {target_base_url!r}")

        target_origin = f"{parsed.scheme}://{parsed.netloc}"
        self._base_path = parsed.path.rstrip("/")
        self._server = _ProxyServer(
            target_origin=target_origin,
            completion_path=completion_path,
            request_body_overrides=request_body_overrides or {},
            generation_overrides=generation_overrides or {},
            verify_tls=verify_tls,
            upstream_model=upstream_model,
            trace_responses=trace_responses,
            trace_requests=trace_requests,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="clem-openai-compatible-proxy",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}{self._base_path}"

    def captured_trace(self) -> str:
        """Return request and response records captured by the proxy server."""

        return self._server.captured_trace()

    def __enter__(self) -> "OpenAICompatibleProxy":
        if not self._server.verify_tls:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def proxy_for_model_connection(connection: dict[str, Any] | None,
                               completion_path: Path,
                               include_openrouter: bool = False,
                               trace_responses: bool = True,
                               trace_requests: bool = False,
                               ) -> OpenAICompatibleProxy | None:
    if not connection:
        return None

    backend = connection.get("backend")

    if backend != "openai_compatible" and not (include_openrouter and backend == "openrouter"):
        return None

    base_url = connection.get("base_url")

    if not isinstance(base_url, str) or not base_url:
        raise ValueError("OpenAI-compatible model connections require base_url.")

    overrides = connection.get("request_body_overrides")

    if not isinstance(overrides, dict):
        overrides = {}

    generation_overrides = connection.get("generation_overrides")

    if not isinstance(generation_overrides, dict):
        generation_overrides = {}

    return OpenAICompatibleProxy(
        target_base_url=base_url,
        completion_path=completion_path,
        request_body_overrides=overrides,
        generation_overrides=generation_overrides,
        verify_tls=bool(connection.get("verify_tls", True)),
        upstream_model=connection.get("upstream_model", connection.get("model")),
        trace_responses=trace_responses,
        trace_requests=trace_requests,
    )
