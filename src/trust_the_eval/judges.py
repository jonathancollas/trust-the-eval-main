"""Judge-call harness (INFRA-7): pointwise correctness judging with order-swap,
cross-family judging, and verbosity perturbation. Used by judge_swap.
"""
from __future__ import annotations
from typing import Optional

from .probe import ModelClient


def judge_correct(judge: ModelClient, candidate: str, gold: str,
                  first: bool = False, author: str = "") -> int:
    pos = "[first] " if first else ""
    auth = f" AUTHOR: {author}" if author else ""
    prompt = (f"{pos}Decide if the candidate answer is correct.\n"
              f"CANDIDATE: {candidate}\nGOLD: {gold}{auth}\n"
              f"Reply 1 if correct else 0.")
    out = judge.complete(prompt, temperature=0.0)
    return 1 if "1" in out[:3] else 0
