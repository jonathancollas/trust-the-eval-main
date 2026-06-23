"""Deterministic sampling (INFRA-2)."""
from __future__ import annotations
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def subsample(items: Sequence[T], k: int, seed: int = 0) -> list[tuple[int, T]]:
    """Return up to k (index, item) pairs, deterministically for a given seed."""
    n = len(items)
    if k <= 0 or k >= n:
        return list(enumerate(items))
    idx = sorted(random.Random(seed).sample(range(n), k))
    return [(i, items[i]) for i in idx]
