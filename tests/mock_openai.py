from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockOpenAIServer(ThreadingHTTPServer):
    daemon_threads = True
    reject_stream_options = False
    token_delay_s = 0.002
    token_count = 3


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return None

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "not found"}})
            return
        server = self.server
        assert isinstance(server, MockOpenAIServer)
        include_usage = "stream_options" in body
        if include_usage and server.reject_stream_options:
            self._write_json(400, {"error": {"message": "unknown field stream_options"}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def event(payload: dict[str, Any]) -> None:
            data = f"data: {json.dumps(payload)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()

        event({"choices": [{"delta": {"role": "assistant"}}]})
        for index in range(server.token_count):
            time.sleep(server.token_delay_s)
            event({"choices": [{"delta": {"content": f"t{index}"}}]})
        event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        if include_usage:
            event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": server.token_count,
                    },
                }
            )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
