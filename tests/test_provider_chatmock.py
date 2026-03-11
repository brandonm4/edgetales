import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from provider import create_gateway


class _CaptureHandler(BaseHTTPRequestHandler):
    requests = []
    response_body = {"choices": [{"message": {"content": "ok"}}]}

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        self.__class__.requests.append({
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "payload": payload,
        })
        body = json.dumps(self.__class__.response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ChatMockProviderTests(unittest.TestCase):
    def setUp(self):
        _CaptureHandler.requests = []
        _CaptureHandler.response_body = {"choices": [{"message": {"content": "ok"}}]}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_chatmock_mode_uses_chat_completions_and_placeholder_auth(self):
        gateway = create_gateway(
            api_key="",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            mode="chatmock",
        )
        response = gateway.create_response(
            model="claude-sonnet-4-5-20250929",
            instructions="Be terse.",
            input=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(response.output_text, "ok")
        self.assertEqual(len(_CaptureHandler.requests), 1)
        req = _CaptureHandler.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], "Bearer local-proxy")
        self.assertEqual(req["payload"]["model"], "gpt-5")
        self.assertEqual(req["payload"]["messages"][0]["role"], "system")
        self.assertEqual(req["payload"]["messages"][-1]["content"], "hello")

    def test_chatmock_mode_extracts_schema_json_from_wrapped_text(self):
        _CaptureHandler.response_body = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"ok\": true, \"items\": [1]}\n```"
                    }
                }
            ]
        }
        gateway = create_gateway(
            api_key="",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            mode="chatmock",
        )
        response = gateway.create_response(
            model="claude-haiku-4-5-20251001",
            input=[{"role": "user", "content": "return json"}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "demo",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                }
            },
        )
        self.assertEqual(json.loads(response.output_text)["ok"], True)
        req = _CaptureHandler.requests[0]
        self.assertEqual(req["payload"]["model"], "gpt-5.1-codex-mini")
        self.assertIn("Return only valid JSON", req["payload"]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
