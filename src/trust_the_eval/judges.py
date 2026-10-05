"""Judge-call harness (INFRA-7): pointwise correctness judging with order-swap,
cross-family judging, and verbosity perturbation. Used by judge_swap.
"""
from __future__ import annotations
from typing import Optional

from .probe import ModelClient


def judge_correct(judge: ModelClient, candidate: str, gold: str,
                  first: bool = False, author: str = "") -> int:
    auth = f" AUTHOR: {author}" if author else ""
    if first:
        answers = f"ANSWER 1 (CANDIDATE): {candidate}\nANSWER 2 (REFERENCE): {gold}"
        position = 1
    else:
        answers = f"ANSWER 1 (REFERENCE): {gold}\nANSWER 2 (CANDIDATE): {candidate}"
        position = 2
    prompt = ("Decide if the candidate answer is correct.\n"
              f"{answers}\nCANDIDATE POSITION: {position}{auth}\n"
              f"Reply 1 if correct else 0.")
    out = judge.complete(prompt, temperature=0.0)
    return 1 if "1" in out[:3] else 0
