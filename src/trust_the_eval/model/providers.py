"""Real provider clients (INFRA-1). Fully implemented; require an API key at
runtime. The SDK import is lazy so the package works without these installed.
"""
from __future__ import annotations
import os
from typing import Any

from ..probe import ModelClient


class AnthropicClient(ModelClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str | None = None, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self._key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            import anthropic  # lazy
            self._client = anthropic.Anthropic(api_key=self._key)
        return self._client

    def complete(self, prompt: str, *, temperature: float = 0.0, **kw: Any) -> str:
        client = self._ensure()
        msg = client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAIClient(ModelClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self._key:
                raise RuntimeError("OPENAI_API_KEY not set")
            import openai  # lazy
            self._client = openai.OpenAI(api_key=self._key)
        return self._client

    def complete(self, prompt: str, *, temperature: float = 0.0, **kw: Any) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class LocalServerClient(ModelClient):
    """Talk to ANY OpenAI-compatible chat endpoint over stdlib urllib.

    Works with local servers (Ollama `http://localhost:11434/v1`, LM Studio
    `http://localhost:1234/v1`, vLLM, llama.cpp --api) and remote ones. No SDK,
    no extra dependency. An API key is optional (most local servers ignore it);
    when provided it is sent as a Bearer header for that request only.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 max_tokens: int = 1024, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str, *, temperature: float = 0.0, **kw: Any) -> str:
        import json as _json
        import urllib.request as _rq
        body = _json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = _rq.Request(self.base_url + "/chat/completions", data=body,
                          headers=headers, method="POST")
        with _rq.urlopen(req, timeout=self.timeout) as r:
            data = _json.loads(r.read().decode("utf-8"))
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            # some servers return {"message": {"content": ...}} or {"content": ...}
            if isinstance(data, dict):
                msg = data.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"] or ""
                if "content" in data:
                    return data["content"] or ""
            return ""
