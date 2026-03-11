#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local OpenAI-compatible proxy for EdgeTales.

Endpoints:
- GET  /health
- POST /v1/responses

Backends:
- openai: passthrough to an upstream OpenAI-compatible Responses API
- chatmock: adapter to a ChatMock `chat/completions` server with schema validation/retries
- mock: deterministic backend for tests and local development
"""

import argparse
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib import error as urlerror
from urllib import request

try:
    import openai
except ImportError:
    openai = None


_LOGGER = None


def set_logger(logger_func) -> None:
    global _LOGGER
    _LOGGER = logger_func


def _log(msg: str, level: str = "info") -> None:
    if _LOGGER is not None:
        _LOGGER(f"[Proxy] {msg}", level=level)
        return
    print(f"[Proxy] {msg}")


def _extract_input_text(payload: dict) -> str:
    parts = []
    for item in payload.get("input", []) or []:
        content = item.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "input_text":
                    parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def _schema_example(schema: dict):
    if not isinstance(schema, dict):
        return None

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema_type = non_null[0] if non_null else "null"

    if schema_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        return {key: _schema_example(props[key]) for key in required}
    if schema_type == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_schema_example(item_schema)]
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None
    return "mock-value"


def _extract_schema(payload: dict) -> Optional[dict]:
    text_cfg = payload.get("text") if isinstance(payload.get("text"), dict) else None
    fmt = text_cfg.get("format") if isinstance(text_cfg, dict) else None
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = fmt.get("schema")
        if isinstance(schema, dict):
            return schema
    return None


def _extract_json_text(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()
    return text.strip()


def _normalize_schema_type(schema: dict):
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        if not non_null:
            return "null"
        return non_null[0]
    return schema_type


def _validate_json_schema(value, schema: dict, path: str = "$") -> None:
    if not isinstance(schema, dict):
        return

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: expected one of {schema['enum']}, got {value!r}")

    schema_type = _normalize_schema_type(schema)
    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key}: missing required property")
        if schema.get("additionalProperties") is False:
            extras = [key for key in value.keys() if key not in properties]
            if extras:
                raise ValueError(f"{path}: unexpected properties {extras}")
        for key, subschema in properties.items():
            if key in value:
                _validate_json_schema(value[key], subschema, f"{path}.{key}")
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, f"{path}[{index}]")
        return

    if schema_type == "string" and not isinstance(value, str):
        raise ValueError(f"{path}: expected string")
    elif schema_type == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        raise ValueError(f"{path}: expected integer")
    elif schema_type == "number" and not ((isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)):
        raise ValueError(f"{path}: expected number")
    elif schema_type == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{path}: expected boolean")
    elif schema_type == "null" and value is not None:
        raise ValueError(f"{path}: expected null")


def _chatmock_model_name(model: str) -> str:
    lowered = (model or "").lower()
    if "haiku" in lowered:
        return "gpt-5.1-codex-mini"
    if "sonnet" in lowered or "opus" in lowered:
        return "gpt-5"
    return model or "gpt-5"


def _chatmock_messages(payload: dict, schema: Optional[dict], prior_error: str = "", prior_content: str = "") -> list[dict]:
    messages = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    if schema:
        schema_msg = (
            "Return only valid JSON matching this schema exactly. "
            "Do not include markdown fences, commentary, or any extra text.\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        if prior_error:
            schema_msg += f"\nPrevious attempt failed validation: {prior_error}"
            if prior_content:
                schema_msg += f"\nPrevious invalid output:\n{prior_content[:1200]}"
        messages.append({"role": "system", "content": schema_msg})
    for item in payload.get("input", []) or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content", "")
        if isinstance(content, str) and content:
            messages.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("input_text", "output_text"):
                    parts.append(block.get("text", ""))
            if parts:
                messages.append({"role": role, "content": "\n".join(parts)})
    return messages


def _chatmock_tools(schema: Optional[dict]):
    if not schema:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_result",
                "description": "Submit the JSON result",
                "parameters": schema,
            },
        }
    ]


def _extract_chatmock_schema_result(data: dict) -> str:
    message = (((data.get("choices") or [{}])[0]).get("message") or {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        fn = (tool_calls[0].get("function") or {}) if isinstance(tool_calls[0], dict) else {}
        arguments = fn.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments.strip()
    return _extract_json_text(message.get("content") or "")


def _build_response_payload(output_text: str, model: str, *, incomplete_reason: Optional[str] = None) -> dict:
    payload = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed" if not incomplete_reason else "incomplete",
        "model": model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": output_text,
    }
    if incomplete_reason:
        payload["incomplete_details"] = {"reason": incomplete_reason}
    return payload


@dataclass
class ProxyConfig:
    backend: str = "openai"
    client_api_key: str = ""
    upstream_api_key: str = ""
    upstream_base_url: str = ""


class ProxyBackend:
    client_api_key: str = ""

    def create_response(self, payload: dict, request_id: str = "") -> dict:
        raise NotImplementedError


class MockBackend(ProxyBackend):
    def __init__(self, client_api_key: str = ""):
        self.client_api_key = client_api_key

    def create_response(self, payload: dict, request_id: str = "") -> dict:
        model = payload.get("model", "mock-model")
        text_cfg = payload.get("text") or {}
        fmt = text_cfg.get("format") if isinstance(text_cfg, dict) else None
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            schema = fmt.get("schema", {})
            obj = _schema_example(schema)
            return _build_response_payload(json.dumps(obj), model)

        input_text = _extract_input_text(payload) or "mock-text"
        return _build_response_payload(f"mock:{input_text[:120]}", model)


class OpenAIBackend(ProxyBackend):
    def __init__(self, cfg: ProxyConfig):
        self.client_api_key = cfg.client_api_key
        kwargs = {"api_key": cfg.upstream_api_key or os.environ.get("OPENAI_API_KEY", "")}
        if cfg.upstream_base_url:
            kwargs["base_url"] = cfg.upstream_base_url.rstrip("/")
        if openai is None:
            raise RuntimeError("openai package is required for the upstream openai backend")
        self._client = openai.OpenAI(**kwargs)

    def create_response(self, payload: dict, request_id: str = "") -> dict:
        _log(f"{request_id} upstream=openai create_response model={payload.get('model')!r}")
        response = self._client.responses.create(**payload)
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response)


class ChatMockBackend(ProxyBackend):
    def __init__(self, cfg: ProxyConfig, max_retries: int = 2):
        self.client_api_key = cfg.client_api_key
        self.base_url = (cfg.upstream_base_url or "http://127.0.0.1:8000/v1").rstrip("/")
        self.max_retries = max_retries

    def _post_chat_completion(self, payload: dict) -> dict:
        req = request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-proxy",
            },
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ChatMock HTTP {exc.code}: {body[:1200]}")

    def create_response(self, payload: dict, request_id: str = "") -> dict:
        requested_model = payload.get("model", "gpt-5")
        schema = _extract_schema(payload)
        last_error = ""
        last_content = ""
        for attempt in range(self.max_retries + 1):
            _log(
                f"{request_id} upstream=chatmock attempt={attempt + 1}/{self.max_retries + 1} "
                f"model={requested_model!r} schema={'yes' if schema else 'no'}",
                level="info",
            )
            upstream_payload = {
                "model": _chatmock_model_name(requested_model),
                "messages": _chatmock_messages(payload, schema, last_error, last_content),
            }
            tools = _chatmock_tools(schema)
            if tools:
                upstream_payload["tools"] = tools
                upstream_payload["tool_choice"] = "auto"
            data = self._post_chat_completion(upstream_payload)
            message = (((data.get("choices") or [{}])[0]).get("message") or {})
            content = message.get("content") or ""
            if not schema:
                return _build_response_payload(content, requested_model)
            last_content = content
            try:
                json_text = _extract_chatmock_schema_result(data)
                parsed = json.loads(json_text)
                _validate_json_schema(parsed, schema)
                _log(f"{request_id} upstream=chatmock schema_validation=ok attempt={attempt + 1}")
                return _build_response_payload(json.dumps(parsed, ensure_ascii=False), requested_model)
            except Exception as exc:
                last_error = str(exc)
                _log(
                    f"{request_id} upstream=chatmock schema_validation=failed "
                    f"attempt={attempt + 1} error={last_error} content={content[:300]!r}",
                    level="warning",
                )
                if attempt == self.max_retries:
                    raise RuntimeError(f"ChatMock schema validation failed after {self.max_retries + 1} attempts: {last_error}")
        raise RuntimeError("ChatMock schema validation failed")


def build_backend(cfg: ProxyConfig) -> ProxyBackend:
    if cfg.backend == "mock":
        return MockBackend(client_api_key=cfg.client_api_key)
    if cfg.backend == "chatmock":
        return ChatMockBackend(cfg)
    if cfg.backend == "openai":
        return OpenAIBackend(cfg)
    raise ValueError(f"Unsupported proxy backend: {cfg.backend}")


class ProxyHandler(BaseHTTPRequestHandler):
    backend: ProxyBackend = MockBackend()
    expected_api_key: str = ""

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            self._write_json(200, {"ok": True, "service": "edgetales-proxy"})
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/v1/responses":
            self._write_json(404, {"error": "not_found"})
            return

        request_id = f"req_{uuid.uuid4().hex[:8]}"
        try:
            if self.expected_api_key:
                auth_header = self.headers.get("Authorization", "")
                expected_header = f"Bearer {self.expected_api_key}"
                if auth_header != expected_header:
                    _log(f"{request_id} auth=failed remote={self.client_address[0]} path={self.path}", level="warning")
                    self._write_json(401, {"error": {"message": "Invalid proxy API key", "type": "authentication_error"}})
                    return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            _log(
                f"{request_id} inbound path={self.path} backend={type(self.backend).__name__} "
                f"model={payload.get('model')!r} schema={'yes' if _extract_schema(payload) else 'no'} "
                f"input_chars={len(_extract_input_text(payload))}",
            )
            response = self.backend.create_response(payload, request_id=request_id)
            _log(f"{request_id} completed status=200")
            self._write_json(200, response)
        except Exception as exc:
            if openai is not None and isinstance(exc, openai.AuthenticationError):
                _log(f"{request_id} upstream_auth_error={exc}", level="warning")
                self._write_json(401, {"error": {"message": str(exc), "type": "authentication_error"}})
                return
            _log(f"{request_id} failed error={type(exc).__name__}: {exc}", level="warning")
            self._write_json(500, {"error": {"message": str(exc), "type": "proxy_error"}})


def start_server(host: str, port: int, backend: ProxyBackend) -> ThreadingHTTPServer:
    expected_api_key = getattr(backend, "client_api_key", "") if backend is not None else ""
    handler = type(
        "EdgeTalesProxyHandler",
        (ProxyHandler,),
        {"backend": backend, "expected_api_key": expected_api_key},
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    parser = argparse.ArgumentParser(description="Run the EdgeTales local proxy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=4000, type=int)
    parser.add_argument("--backend", choices=("openai", "chatmock", "mock"), default="openai")
    parser.add_argument("--client-api-key", default=os.environ.get("EDGETALES_PROXY_API_KEY", os.environ.get("PROXY_API_KEY", "")))
    parser.add_argument("--upstream-base-url", default=os.environ.get("UPSTREAM_OPENAI_BASE_URL", ""))
    parser.add_argument("--upstream-api-key", default=os.environ.get("UPSTREAM_OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    args = parser.parse_args()

    server = start_server(
        args.host,
        args.port,
        build_backend(ProxyConfig(
            backend=args.backend,
            client_api_key=args.client_api_key,
            upstream_api_key=args.upstream_api_key,
            upstream_base_url=args.upstream_base_url,
        )),
    )
    try:
        print(f"EdgeTales proxy listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
