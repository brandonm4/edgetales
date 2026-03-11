import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

from provider import create_gateway
from proxy_server import ChatMockBackend, MockBackend, ProxyConfig, start_server


class _ChatMockHandler(BaseHTTPRequestHandler):
    responses = []
    requests = []

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        self.__class__.requests.append(payload)
        response_body = self.__class__.responses.pop(0)
        body = json.dumps(response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProxyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.server = start_server("127.0.0.1", 0, MockBackend())
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_health_endpoint(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health") as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(data["ok"])
        self.assertEqual(data["service"], "edgetales-proxy")

    def test_openai_compatible_text_response(self):
        gateway = create_gateway(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            mode="proxy",
        )
        response = gateway.create_response(
            model="mock-model",
            instructions="Be terse.",
            input=[{"role": "user", "content": "hello world"}],
            max_output_tokens=64,
        )
        self.assertEqual(response.output_text, "mock:hello world")

    def test_openai_compatible_json_schema_response(self):
        gateway = create_gateway(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            mode="proxy",
        )
        response = gateway.create_response(
            model="mock-model",
            input=[{"role": "user", "content": "return json"}],
            max_output_tokens=64,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "demo",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean"},
                            "name": {"type": "string"},
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["ok", "name", "items"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        data = json.loads(response.output_text)
        self.assertEqual(data["ok"], True)
        self.assertEqual(data["name"], "mock-value")
        self.assertEqual(data["items"], [1])

    def test_proxy_rejects_invalid_bearer_token(self):
        secured_server = start_server("127.0.0.1", 0, MockBackend(client_api_key="expected-key"))
        secured_port = secured_server.server_address[1]
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{secured_port}/v1/responses",
                data=json.dumps({"model": "mock-model", "input": []}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer wrong-key",
                },
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 401)
            payload = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertEqual(payload["error"]["type"], "authentication_error")
        finally:
            secured_server.shutdown()
            secured_server.server_close()

    def test_chatmock_backend_retries_until_schema_valid(self):
        _ChatMockHandler.requests = []
        _ChatMockHandler.responses = [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": "{\"ok\": true, \"items\": [1]}"}}]},
        ]
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ChatMockHandler)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        upstream_port = upstream.server_address[1]
        try:
            backend = ChatMockBackend(ProxyConfig(
                backend="chatmock",
                upstream_base_url=f"http://127.0.0.1:{upstream_port}/v1",
            ))
            payload = {
                "model": "claude-haiku-4-5-20251001",
                "input": [{"role": "user", "content": "return json"}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "demo",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "ok": {"type": "boolean"},
                                "items": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["ok", "items"],
                            "additionalProperties": False,
                        },
                    }
                },
            }
            response = backend.create_response(payload)
            self.assertEqual(json.loads(response["output_text"])["ok"], True)
            self.assertEqual(len(_ChatMockHandler.requests), 2)
            self.assertEqual(_ChatMockHandler.requests[0]["model"], "gpt-5.1-codex-mini")
            self.assertIn("Previous attempt failed validation", _ChatMockHandler.requests[1]["messages"][0]["content"])
        finally:
            upstream.shutdown()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
