"""Provider interface and the first provider: an OpenAI-compatible chat-completions endpoint (LM Studio by default), over
the existing HTTP client (httpx). Tests inject an httpx transport; no live model in tests."""
from __future__ import annotations

import json
import time
from typing import Protocol

import httpx


class ProviderError(RuntimeError):
    """The model could not be reached or returned no usable message."""


class Provider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[dict], temperature: float = 0.2) -> tuple[str, float]: ...


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, timeout_s: float, api_key: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.base_url, self.model = base_url.rstrip("/"), model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout_s, headers=headers, transport=transport)

    def complete(self, messages: list[dict], temperature: float = 0.2, schema: dict | None = None) -> tuple[str, float]:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "stream": False}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "explanation", "strict": True, "schema": schema}}
        t0 = time.perf_counter()
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body)
            if r.status_code == 400 and schema is not None:  # a server without structured output: plain JSON instructions only
                body.pop("response_format")
                r = self.client.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("empty message")
        return content, (time.perf_counter() - t0) * 1000

    def chat(self, messages: list[dict], tools: list[dict], temperature: float = 0.2) -> dict:
        """One chat-completions turn offering `tools` (Stage App-7). Returns {"content": str, "tool_calls": [{"id", "name",
        "arguments" (dict, or None if not valid JSON), "raw_arguments"}]}. Any reasoning field the server adds is dropped
        here and never reaches the caller (CLAUDE.md §11 rule 4)."""
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": temperature, "stream": False}
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                args = args if isinstance(args, dict) else None
            except ValueError:
                args = None
            calls.append({"id": str(c.get("id") or f"call_{i}"), "name": str(fn.get("name") or ""), "arguments": args,
                          "raw_arguments": raw if isinstance(raw, str) else json.dumps(raw)})
        content = msg.get("content") if isinstance(msg.get("content"), str) else ""
        if not calls and not content.strip():
            raise ProviderError("empty message")
        return {"content": content, "tool_calls": calls}
