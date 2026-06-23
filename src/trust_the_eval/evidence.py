"""Evidence capture (INFRA-5): probes attach the OFFENDING ITEMS, not just a number."""
from __future__ import annotations
from typing import Any


def examples(rows: list[dict[str, Any]], cap: int = 5) -> list[dict[str, Any]]:
    """Trim a list of evidence rows to a compact, capped sample."""
    return rows[:cap]


def trim(text: str, n: int = 200) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"
