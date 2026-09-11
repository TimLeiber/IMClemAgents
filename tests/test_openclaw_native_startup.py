"""Opt-in native CLI regression using a scripted API, never model inference.

Run inside clemagents-sandbox:dev with --network none, the repository mounted
read-only at /opt/clemagents/src, and RUN_OPENCLAW_NATIVE_TEST=1. No keys are required.
"""

import json
import io
import os
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import redirect_stdout
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from clemagents.adapters.openclaw import OpenClawHarness


@unittest.skipUnless(os.environ.get("RUN_OPENCLAW_NATIVE_TEST") == "1",
                     "requires the isolated agent-sandbox image")
class TestOpenClawNativeStartup(unittest.TestCase):
    def test_gateway_agent_reaches_recorded_api_and_preserves_artifacts(self):
        requests = []

        class ScriptedAPI(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                chunk = {
                    "id": "offline-fixture", "object": "chat.completion.chunk",
                    "created": 1, "model": "fixture",
                    "choices": [{"index": 0, "delta": {
                        "role": "assistant", "content": "OFFLINE_FIXTURE_COMPLETE",
                    }, "finish_reason": None}],
                }
                terminal = {**chunk, "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                ]}
                payload = (
                    f"data: {json.dumps(chunk)}\n\n"
                    f"data: {json.dumps(terminal)}\n\ndata: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedAPI)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        connection = {
            "backend": "openai_compatible", "model": "fixture",
            "runtime_model": "offline/fixture", "upstream_model": "fixture",
            "base_url": base_url, "openclaw_provider": "offline", "env": {},
            "openclaw_config_patch": {
                "models": {"providers": {"offline": {
                    "baseUrl": base_url, "apiKey": "offline-test-not-a-secret",
                    "api": "openai-completions", "models": [{
                        "id": "fixture", "name": "Offline fixture",
                        "reasoning": False, "input": ["text"],
                        "contextWindow": 65536, "maxTokens": 4096,
                    }],
                }}},
                "agents": {"defaults": {"model": {"primary": "offline/fixture"}}},
            },
        }
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                "clemagents.adapters.openclaw.load_model_connection",
                return_value=connection,
            ):
                with redirect_stdout(io.StringIO()):
                    result = OpenClawHarness(
                        model_connection_path="unused", thinking="off",
                    ).run_episode("Offline startup fixture", output_dir=directory)
                trace = Path(result.artifacts["adapter_messages"]).read_text()
                self.assertEqual(result.metadata["returncode"], 0, trace[-6000:])
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["model"], "fixture")
                self.assertTrue(any(
                    "start_game" in tool.get("function", {}).get("name", "")
                    for tool in requests[0].get("tools", [])
                ), "the native MCP inventory must reach the API")
                self.assertIn("OFFLINE_FIXTURE_COMPLETE", trace)
                self.assertTrue("raw_upstream_request_" in trace)
                self.assertTrue("raw_upstream_response_" in trace)
                self.assertNotIn("GatewayLockError", trace)
                # Current OpenClaw stores native sessions in SQLite; older
                # versions used JSONL. Both live in the retained artifact home.
                native_files = list(Path(directory).glob("openclaw_home/.openclaw*/agents/**/*"))
                self.assertTrue(any(path.suffix in {".jsonl", ".sqlite"}
                                    for path in native_files))
                # A text-only fixture intentionally does not complete a game.
                self.assertFalse(result.metadata["game_completed"])
                self.assertIn("ended before", result.metadata["runtime_error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
