import json
import threading
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import urllib3

from . import write_ca_bundle

HOP_BY_HOP_HEADERS = {"connection", "content-length", "host", "keep-alive", "proxy-authenticate", "proxy-authorization",
                      "te", "trailer",
                      "transfer-encoding", "upgrade"}

# the host pipeline owns the episode deadline. the compatibility proxy only
# bounds connection establishment; model inference may legitimately take
# longer than any fixed adapter-level read timeout
UPSTREAM_REQUEST_TIMEOUT = (30, None)
INFERENCE_PATH_SUFFIXES = ("/chat/completions", "/completions", "/messages", "/responses",)


def _is_inference_request(method: str, path: str) -> bool:
    """Return whether a provider request can start new model inference."""

    request_path = urlsplit(path).path.rstrip("/")
    return method == "POST" and request_path.endswith(INFERENCE_PATH_SUFFIXES)


def _observe_request_body(body: bytes, completion_path: Path) -> None:
    """Inspect a copy for game completion without changing the forwarded bytes."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    if not isinstance(payload, dict):
        return

    _write_completion_marker_from_tool_results(payload, completion_path)


def _write_completion_marker_from_tool_results(payload: dict[str, Any], completion_path: Path) -> None:
    """Record game completion observed in a Responses API tool result."""

    if completion_path.exists():
        return

    tool_results = []
    input_items = payload.get("input")

    if isinstance(input_items, list):
        tool_results.extend(item.get("output") for item in input_items
                            if isinstance(item, dict) and item.get("type") == "function_call_output")

    messages = payload.get("messages")

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue

            content = message.get("content")

            if message.get("role") == "tool":
                tool_results.append(content)

            if isinstance(content, list):
                tool_results.extend(block.get("content") for block in content
                                    if isinstance(block, dict) and block.get("type") == "tool_result")

    def reports_completion(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("done") is True or any(reports_completion(item) for item in value.values())

        if isinstance(value, list):
            return any(reports_completion(item) for item in value)

        if isinstance(value, str):
            return '"done":true' in "".join(value.split())

        return False

    for tool_result in tool_results:
        if not reports_completion(tool_result):
            continue

        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion_path.write_text(json.dumps({"done": True, "source": "model_proxy"}), encoding="utf-8")
        return


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, target_origin: str, completion_path: Path, verify_tls: bool | str,
                 trace_responses: bool, trace_requests: bool):
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.target_origin = target_origin
        self.completion_path = completion_path
        self.verify_tls = verify_tls
        self.trace_responses = trace_responses
        self.trace_requests = trace_requests
        self.response_trace_lock = threading.Lock()
        self.response_trace_count = 0
        self.trace_records: list[str] = []

    def trace_request(self, path: str, body: bytes) -> None:
        if not self.trace_requests:
            return

        request_text = body.decode("utf-8", errors="replace")

        with self.response_trace_lock:
            self.response_trace_count += 1
            trace_id = self.response_trace_count
            trace_record = (f"\nraw_upstream_request_{trace_id}_start\n"
                            f"path: {path}\n"
                            f"{request_text}\n"
                            f"raw_upstream_request_{trace_id}_end")
            self.trace_records.append(trace_record)
            print(trace_record, flush=True)

    def trace_response(self, path: str, status_code: int, content_type: str, content_encoding: str,
                       body: bytes) -> None:
        if not self.trace_responses:
            return

        response_text = body.decode("utf-8", errors="replace")

        with self.response_trace_lock:
            self.response_trace_count += 1
            trace_id = self.response_trace_count
            trace_record = (f"\nraw_upstream_response_{trace_id}_start\n"
                            f"path: {path}\n"
                            f"status: {status_code}\n"
                            f"content_type: {content_type}\n"
                            f"content_encoding: {content_encoding}\n"
                            f"{response_text}\n"
                            f"raw_upstream_response_{trace_id}_end")
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

        _observe_request_body(body, self.server.completion_path)

        # a completed game result is immutable. refuse any new inference at
        # this generic provider boundary while the adapter finalizes its native
        # artifacts; do not let a harness begin a post-game model turn. this
        # check follows request inspection so a completion first visible in a
        # submitted tool-result payload also gates that same request
        if (self.server.completion_path.exists() and _is_inference_request(self.command, self.path)):
            payload = json.dumps({"error": {"message": "The game is complete; no further model inference was forwarded.",
                                            "type": "game_completed"}}).encode("utf-8")
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return

        self.server.trace_request(self.path, body)
        headers = {name: value for name, value in self.headers.items() if name.lower() not in HOP_BY_HOP_HEADERS}
        target_url = f"{self.server.target_origin}{self.path}"

        try:
            upstream = requests.request(self.command, target_url, headers=headers, data=body or None, stream=True,
                                        timeout=UPSTREAM_REQUEST_TIMEOUT, verify=self.server.verify_tls)
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
            self.server.trace_response(self.path, upstream.status_code, upstream.headers.get("Content-Type", ""),
                                       upstream.headers.get("Content-Encoding", ""), b"".join(response_chunks))
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

    def __init__(self, target_base_url: str, completion_path: Path, verify_tls: bool = True,
                 trace_responses: bool = True, trace_requests: bool = False, ca_certificates: str | None = None):
        parsed = urlsplit(target_base_url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid OpenAI-compatible base URL: {target_base_url!r}")

        target_origin = f"{parsed.scheme}://{parsed.netloc}"
        self._base_path = parsed.path.rstrip("/")
        self._ca_bundle_path = None
        if ca_certificates is not None:
            if not verify_tls:
                raise ValueError("ca_certificates requires TLS verification to remain enabled")
            self._ca_bundle_path = completion_path.with_suffix(".ca.pem")
            verify_tls = write_ca_bundle(ca_certificates, self._ca_bundle_path)
        self._server = _ProxyServer(target_origin=target_origin, completion_path=completion_path,
                                    verify_tls=verify_tls, trace_responses=trace_responses,
                                    trace_requests=trace_requests)
        self._thread = threading.Thread(target=self._server.serve_forever, name="clem-openai-compatible-proxy",
                                        daemon=True)

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
        if self._ca_bundle_path is not None:
            self._ca_bundle_path.unlink(missing_ok=True)


def proxy_for_model_connection(connection: dict[str, Any] | None, completion_path: Path,
                               include_openrouter: bool = False, trace_responses: bool = True,
                               trace_requests: bool = False) -> OpenAICompatibleProxy | None:
    if not connection:
        return None

    # reject stale connections instead of silently discarding requested controls
    for key in ("request_body_overrides", "generation_overrides", "upstream_model"):
        if connection.get(key):
            raise ValueError(f"{key} is no longer supported; configure the harness through its native interface")

    backend = connection.get("backend")

    if backend != "openai_compatible" and not (include_openrouter and backend == "openrouter"):
        return None

    base_url = connection.get("base_url")

    if not isinstance(base_url, str) or not base_url:
        raise ValueError("OpenAI-compatible model connections require base_url.")

    return OpenAICompatibleProxy(target_base_url=base_url, completion_path=completion_path,
                                 verify_tls=bool(connection.get("verify_tls", True)),
                                 trace_responses=trace_responses, trace_requests=trace_requests,
                                 ca_certificates=connection.get("ca_certificates"))
