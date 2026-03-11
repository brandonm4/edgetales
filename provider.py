#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provider abstraction for model access.

Keeps the rest of the app independent from whether requests go directly to
OpenAI or to a local OpenAI-compatible proxy.
"""

from dataclasses import dataclass
import json
import re
from types import SimpleNamespace
from typing import Optional
from urllib import request

try:
    import openai
except ImportError:
    openai = None


@dataclass
class ProviderConfig:
    api_key: str = ""
    base_url: str = ""
    mode: str = "openai"


class ModelGateway:
    """Thin wrapper around the OpenAI Responses API client."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None
        if openai is not None and config.mode != "chatmock":
            api_key = config.api_key or "local-proxy"
            kwargs = {"api_key": api_key}
            if config.base_url:
                kwargs["base_url"] = config.base_url.rstrip("/")
            self._client = openai.OpenAI(**kwargs)

    def _chatmock_model_name(self, model: str) -> str:
        lowered = (model or "").lower()
        if "haiku" in lowered:
            return "gpt-5.1-codex-mini"
        if "sonnet" in lowered:
            return "gpt-5"
        if "opus" in lowered:
            return "gpt-5"
        return model

    def _chatmock_messages(self, instructions: str, input_items, schema: Optional[dict]) -> list[dict]:
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if schema:
            messages.append({
                "role": "system",
                "content": (
                    "Return only valid JSON matching this schema exactly. "
                    "Do not include markdown fences, commentary, or any extra text.\n"
                    + json.dumps(schema, ensure_ascii=False)
                ),
            })
        for item in input_items or []:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type in ("input_text", "output_text"):
                        parts.append(block.get("text", ""))
                messages.append({"role": role, "content": "\n".join(p for p in parts if p)})
        return messages

    def _extract_json_text(self, text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                return text[start:end + 1].strip()
        return text.strip()

    def _create_response_via_chatmock(self, **kwargs):
        if not self.config.base_url:
            raise RuntimeError("ChatMock provider mode requires provider_base_url")
        text_cfg = kwargs.get("text") if isinstance(kwargs.get("text"), dict) else None
        fmt = text_cfg.get("format") if isinstance(text_cfg, dict) else None
        schema = fmt.get("schema") if isinstance(fmt, dict) and fmt.get("type") == "json_schema" else None
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": self._chatmock_model_name(kwargs.get("model", "")),
            "messages": self._chatmock_messages(kwargs.get("instructions", ""), kwargs.get("input", []), schema),
        }).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key or 'local-proxy'}",
            },
            method="POST",
        )
        with request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = (((data.get("choices") or [{}])[0]).get("message") or {})
        content = message.get("content") or ""
        if schema:
            content = self._extract_json_text(content)
        return SimpleNamespace(
            output_text=content,
            output=[SimpleNamespace(
                content=[SimpleNamespace(type="output_text", text=content)]
            )],
            incomplete_details=None,
            raw=data,
        )

    def _create_response_via_http(self, **kwargs):
        if not self.config.base_url:
            raise RuntimeError("Direct provider mode requires the openai package")
        url = self.config.base_url.rstrip("/") + "/responses"
        payload = json.dumps(kwargs).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key or 'local-proxy'}",
            },
            method="POST",
        )
        with request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return SimpleNamespace(
            output_text=data.get("output_text", ""),
            output=[SimpleNamespace(
                content=[SimpleNamespace(type="output_text", text=data.get("output_text", ""))]
            )],
            incomplete_details=SimpleNamespace(**data["incomplete_details"]) if data.get("incomplete_details") else None,
            raw=data,
        )

    def create_response(self, **kwargs):
        if self.config.mode == "chatmock":
            return self._create_response_via_chatmock(**kwargs)
        if self._client is not None:
            return self._client.responses.create(**kwargs)
        return self._create_response_via_http(**kwargs)


def create_gateway(api_key: str = "", base_url: str = "", mode: str = "openai") -> ModelGateway:
    return ModelGateway(ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        mode=mode,
    ))


def create_gateway_from_config(cfg: dict, api_key: Optional[str] = None) -> ModelGateway:
    return create_gateway(
        api_key=api_key if api_key is not None else cfg.get("api_key", ""),
        base_url=cfg.get("provider_base_url", ""),
        mode=cfg.get("provider_mode", "openai"),
    )
