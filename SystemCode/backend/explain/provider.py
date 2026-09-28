"""Provider interface and the first provider: an OpenAI-compatible chat-completions endpoint (LM Studio by default), over
the existing HTTP client (httpx). Tests inject an httpx transport; no live model in tests."""
from __future__ import annotations

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
