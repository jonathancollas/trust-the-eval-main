"""Cost accounting + caching wrapper (INFRA-6).

Wrap any ModelClient in CachingClient: identical calls are served from cache and
every call's (approx) token/$ cost is metered, so model-in-the-loop probes report
cost and are replayable.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any

from .probe import ModelClient


def _est_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


@dataclass
class CostMeter:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd_per_1k_prompt: float = 0.0
    usd_per_1k_completion: float = 0.0

    def record(self, prompt: str, completion: str) -> None:
        self.calls += 1
        self.prompt_tokens += _est_tokens(prompt)
        self.completion_tokens += _est_tokens(completion)

    @property
    def est_usd(self) -> float:
        return (self.prompt_tokens / 1000 * self.usd_per_1k_prompt
                + self.completion_tokens / 1000 * self.usd_per_1k_completion)

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "est_usd": round(self.est_usd, 6)}


class CachingClient(ModelClient):
    def __init__(self, base: ModelClient, meter: CostMeter | None = None):
        self.base = base
        self.meter = meter or CostMeter()
        self._cache: dict[str, str] = {}

    @staticmethod
    def _key(prompt: str, temperature: float, n: int, idx: int) -> str:
        h = hashlib.sha256(f"{temperature}|{n}|{idx}|{prompt}".encode("utf-8"))
        return h.hexdigest()

    def complete(self, prompt: str, *, temperature: float = 0.0, **kw: Any) -> str:
        key = self._key(prompt, temperature, 1, 0)
        if key in self._cache:
            self.meter.cache_hits += 1
            return self._cache[key]
        out = self.base.complete(prompt, temperature=temperature, **kw)
        self.meter.record(prompt, out)
        self._cache[key] = out
        return out

    def sample(self, prompt: str, n: int, *, temperature: float = 1.0, **kw: Any) -> list[str]:
        outs: list[str] = []
        for i in range(n):
            key = self._key(prompt, temperature, n, i)
            if key in self._cache:
                self.meter.cache_hits += 1
                outs.append(self._cache[key])
                continue
            out = self.base.complete(prompt, temperature=temperature, _variant=i, **kw)
            self.meter.record(prompt, out)
            self._cache[key] = out
            outs.append(out)
        return outs
