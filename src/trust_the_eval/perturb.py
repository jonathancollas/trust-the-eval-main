"""Meaning-preserving perturbations + template reformatting (INFRA-3)."""
from __future__ import annotations
import random
import re

# Synonym swaps that preserve the answer to a question.
_SUBS = [
    (r"\bwhat is\b", "compute"),
    (r"\bhow many\b", "what number of"),
    (r"\bcalculate\b", "work out"),
    (r"\bfind\b", "determine"),
    (r"\btotal\b", "sum"),
    (r"\beach\b", "per unit"),
]


def paraphrase(text: str, seed: int = 0) -> str:
    """Reword while preserving meaning (and thus the correct answer)."""
    rng = random.Random(seed)
    out = text.strip()
    for pat, repl in _SUBS:
        if rng.random() < 0.85:
            out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    prefixes = ["Please solve: ", "Consider the following. ", "Here is a problem. ", ""]
    out = rng.choice(prefixes) + out
    out = re.sub(r"\s+", " ", out).strip()
    return out


def reformat_templates(question: str) -> list[tuple[str, str]]:
    """Equivalent presentations of the same question (for format sensitivity)."""
    q = question.strip()
    return [
        ("plain", q),
        ("qa", f"Question: {q}\nAnswer:"),
        ("instructed", f"Answer the question concisely.\n{q}"),
        ("boxed", f"{q}\nGive only the final answer."),
    ]


def numeric_renumber(question: str, gold: str, seed: int = 0):
    """For simple 'A op B' arithmetic: change operands and recompute gold.

    Returns (new_question, new_gold) or None if the item isn't recognised.
    """
    m = re.search(r"(-?\d+)\s*([\+\-\*x×/])\s*(-?\d+)", question)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    rng = random.Random(seed)
    na, nb = a + rng.randint(1, 9), b + rng.randint(1, 9)
    op_n = {"x": "*", "×": "*"}.get(op, op)
    try:
        if op_n == "+":
            res = na + nb
        elif op_n == "-":
            res = na - nb
        elif op_n == "*":
            res = na * nb
        elif op_n == "/":
            if nb == 0 or na % nb != 0:
                return None
            res = na // nb
        else:
            return None
    except Exception:
        return None
    new_q = question[:m.start()] + f"{na} {op} {nb}" + question[m.end():]
    return (new_q, str(res))


def permute_options(options: list[str], seed: int = 0) -> tuple[list[str], list[int]]:
    """Return shuffled options and the permutation (new[i] = old[perm[i]])."""
    perm = list(range(len(options)))
    random.Random(seed).shuffle(perm)
    return ([options[i] for i in perm], perm)
