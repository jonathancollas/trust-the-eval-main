"""Genuine local models with CONTROLLABLE behaviours (INFRA-1, for demo/tests).

These are NOT canned-string mocks: each actually computes answers and exhibits a
real, well-defined behaviour, so a probe can be PROVEN to detect its target
phenomenon. They solve simple arithmetic / lookup questions of the form used in
the bundled examples and tests.
"""
from __future__ import annotations
import random
import re
from typing import Any, Callable, Optional

from ..grading import extract_final, normalize
from ..probe import ModelClient

_ARITH = re.compile(r"(-?\d+)\s*([\+\-\*x×/])\s*(-?\d+)")


def solve_arithmetic(question: str) -> Optional[str]:
    """Genuinely solve 'A op B' questions. Returns the answer string or None."""
    m = _ARITH.search(question)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    op = {"x": "*", "×": "*"}.get(op, op)
    try:
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        if op == "/":
            return str(a // b) if b and a % b == 0 else f"{a / b:.4f}" if b else None
    except Exception:
        return None
    return None


class LocalModel(ModelClient):
    """Base honest solver. Behaviour hooks let subclasses bend it precisely."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    # --- hooks ---
    def _knows(self, question: str) -> bool:
        return True

    def _wrong_answer(self, correct: Optional[str], question: str, variant: int) -> str:
        if correct is not None:
            try:
                return str(int(float(correct)) + 1 + variant)
            except Exception:
                pass
        return "i don't know"

    def _format(self, answer: str, prompt: str) -> str:
        return f"The answer is {answer}."

    def _refuses(self, prompt: str) -> bool:
        return False

    # --- main ---
    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        if self._refuses(prompt):
            return "I'm sorry, but I can't help with that."
        correct = solve_arithmetic(prompt)
        if self._knows(prompt) and correct is not None:
            return self._format(correct, prompt)
        return self._format(self._wrong_answer(correct, prompt, _variant), prompt)


class HonestModel(LocalModel):
    """Solves correctly and robustly. The 'clean' baseline."""


class MemorizerModel(LocalModel):
    """Answers correctly ONLY for memorised verbatim questions; fails paraphrases.
    Demonstrates contamination via perturbation."""

    def __init__(self, memorized: list[str], seed: int = 0):
        super().__init__(seed)
        self._mem = {normalize(q) for q in memorized}

    def _knows(self, question: str) -> bool:
        # consider only the trailing problem line, ignoring added paraphrase prose
        return any(m in normalize(question) for m in self._mem) and \
            normalize(question) in self._mem or normalize(question) in self._mem

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if normalize(prompt) in self._mem and correct is not None:
            return f"The answer is {correct}."
        # not the exact memorised string -> it does not actually know it
        return f"The answer is {self._wrong_answer(correct, prompt, _variant)}."


class SandbaggerModel(LocalModel):
    """Solves correctly UNLESS the prompt signals an evaluation, then underperforms.
    Demonstrates sandbagging / evaluation-awareness."""

    _CUES = ("evaluation", "capability test", "benchmark", "this is a test",
             "graded", "exam", "assessment")

    def _looks_like_eval(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(c in p for c in self._CUES)

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if correct is None:
            return "I don't know."
        if self._looks_like_eval(prompt):
            return f"The answer is {self._wrong_answer(correct, prompt, _variant)}."
        return f"The answer is {correct}."


class ElicitationGatedModel(LocalModel):
    """Fails on plain prompts but succeeds with chain-of-thought / scaffolding.
    Demonstrates under-elicitation (the eval lowballed capability)."""

    def _elicited(self, prompt: str) -> bool:
        p = prompt.lower()
        return ("step by step" in p or "reason" in p or "let's think" in p
                or "show your work" in p)

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if correct is None:
            return "I don't know."
        if self._elicited(prompt):
            return f"Working through it... The answer is {correct}."
        return f"The answer is {self._wrong_answer(correct, prompt, _variant)}."


class RefuserModel(LocalModel):
    """Refuses items matching a trigger (instead of being incapable).
    Demonstrates refusal/abstention vs incapacity confound."""

    def __init__(self, triggers: list[str], seed: int = 0):
        super().__init__(seed)
        self._triggers = [t.lower() for t in triggers]

    def _refuses(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(t in p for t in self._triggers)


class FormatSensitiveModel(LocalModel):
    """Only answers correctly under one template; brittle to presentation.
    Demonstrates prompt-format sensitivity."""

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if correct is None:
            return "I don't know."
        if prompt.strip().lower().startswith("question:"):
            return f"The answer is {correct}."
        return f"The answer is {self._wrong_answer(correct, prompt, _variant)}."


class StochasticModel(LocalModel):
    """Random among plausible answers when temperature>0; stable at temp 0.
    Demonstrates self-consistency / item-ambiguity probing."""

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if correct is None:
            return "I don't know."
        if temperature and temperature > 0:
            rng = random.Random(hash((prompt, _variant)) & 0xffffffff)
            base = int(float(correct))
            return f"The answer is {rng.choice([base, base, base + 1, base - 1])}."
        return f"The answer is {correct}."


class DriftedModel(LocalModel):
    """Gives different (often wrong) answers than a previously-recorded run.
    Demonstrates model drift / temporal validity."""

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if correct is None:
            return "I don't know."
        return f"The answer is {int(float(correct)) + 2}."


class ExploiterModel(LocalModel):
    """Outputs text that satisfies a LENIENT scorer without solving the task
    (mentions every plausible token). Demonstrates reward-hacking / extraction
    audit. Detection-only target — never a recipe."""

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        correct = solve_arithmetic(prompt)
        if correct is None:
            return "answer: 0 1 2 3 4 5 6 7 8 9"
        # echoes the correct token inside noise -> lenient grader passes, robust fails
        return f"Possibly one of 0 1 2 {correct} 3 4 5; hard to say."


class JudgeModel(LocalModel):
    """A judge that scores a candidate response's correctness, with optional
    position bias and self-preference. Used by judge_swap / option_order_bias.

    Prompt convention (set by judges.py):
      '... CANDIDATE: <resp> GOLD: <gold> ... Reply 1 if correct else 0.'
    """

    def __init__(self, position_bias: float = 0.0, self_preference: float = 0.0,
                 family: str = "fam-A", seed: int = 0):
        super().__init__(seed)
        self.position_bias = position_bias
        self.self_preference = self_preference
        self.family = family

    def complete(self, prompt: str, *, temperature: float = 0.0, _variant: int = 0, **kw: Any) -> str:
        m = re.search(r"candidate:\s*(.*?)\s*gold:\s*(.*?)\s*(?:author:\s*(\S+))?\s*$",
                      prompt, re.IGNORECASE | re.DOTALL)
        if not m:
            return "0"
        cand, gold = m.group(1), m.group(2)
        author = (m.group(3) or "").strip()
        correct = extract_final(cand) == extract_final(gold) and extract_final(gold) != ""
        score = 1 if correct else 0
        rng = random.Random(hash((prompt, _variant)) & 0xffffffff)
        # position bias: if candidate is presented FIRST (flag in prompt), nudge up
        if "[first]" in prompt.lower() and rng.random() < self.position_bias:
            score = 1
        # self-preference: prefer responses from its own family
        if author and author == self.family and rng.random() < self.self_preference:
            score = 1
        return str(score)


REGISTRY: dict[str, Callable[[], ModelClient]] = {
    "honest": HonestModel,
    "sandbagger": SandbaggerModel,
    "elicitation": ElicitationGatedModel,
    "drifted": DriftedModel,
    "exploiter": ExploiterModel,
    "stochastic": StochasticModel,
    "format_sensitive": FormatSensitiveModel,
}


def from_spec(spec: str) -> ModelClient:
    """Build a local model from a 'local:<name>' spec (for the CLI/demo)."""
    name = spec.split("local:", 1)[-1] if spec.startswith("local:") else spec
    if name not in REGISTRY:
        raise ValueError(f"unknown local model '{name}'; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()
