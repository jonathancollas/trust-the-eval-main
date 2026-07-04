"""The correction policy — declared, visible, and the single source of truth.

When an evaluation ships human corrections (e.g. MMLU-Redux), each item carries an
``error_type`` and, sometimes, a ``correct_answer``. Turning that into a scoring key
is a *policy*, not a fact — and until now it lived, unstated, inside a one-off
generator. This module makes the policy explicit: one declared table mapping each
``error_type`` to an action, every action carrying its scientific/operational
source, and one function that applies it. The import adapter and the lineage tracer
both call ``apply_policy`` — they never re-implement the rule.

Actions
-------
* ``keep``   — the answer key is unchanged (no label defect, or a clarity/ambiguity
               defect that is surfaced by the ``item_ambiguity`` probe, not by re-keying).
* ``change`` — re-key to the human-provided correction (``wrong_groundtruth`` with a
               parseable ``correct_answer``).
* ``union``  — the gold becomes the *set* of all correct options; a prediction matching
               ANY of them is correct (``multiple_correct_answers``).
* ``drop``   — the item is excluded from scoring: there is no valid key to score
               against (``no_correct_answer``, or a known-wrong key with no correction).
               Scoring against a known-invalid key would fabricate signal.

The honesty rule that governs all of this: **a correction is a candidate, not a
truth.** The policy re-scores under a candidate key and measures the *consequence*
(via ``result_sensitivity``); it never asserts that a given model's answer is right
or wrong in fact. Corrections here are single-pass (no inter-annotator agreement).

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from .calibration.realworld import (AMBIGUITY_ERROR_TYPES, DEFECT_ERROR_TYPES,
                                    canonical_error_type)

LETTERS = "ABCD"

CORRECTION_IS_CANDIDATE = (
    "A correction is a candidate from single-pass annotation (no inter-annotator "
    "agreement). The policy re-scores under it and measures the consequence; it never "
    "asserts a model's answer is right or wrong in fact."
)

# why each action is the honest choice — attached to every decision and every table row
_ACTION_SOURCE = {
    "keep": ("No answer-key change. A clarity/ambiguity defect does not change the key; "
             "it is surfaced by the item_ambiguity probe, not by re-keying "
             "(clarity issues are not label errors)."),
    "change": ("Re-key to the human-provided correction. Label errors bias scores and "
               "can be corrected from provenance — Northcutt et al. (2021); "
               "MMLU-Redux, Gema et al. (2024)."),
    "union": ("More than one option is correct; the gold becomes the set and a prediction "
              "matching ANY correct option scores (set-membership scoring) — "
              "MMLU-Redux multiple-correct items."),
    "drop": ("Excluded from scoring: there is no valid key to score against (no correct "
             "option, or a known-wrong key with no correction). Scoring against a "
             "known-invalid key would fabricate signal."),
}

# the declared policy: canonical error_type -> action (+ optional fallback / note).
# `change` falls back to `drop` when no parseable correction is supplied.
DEFAULT_POLICY: Dict[str, Dict[str, Any]] = {
    "ok": {"action": "keep", "note": "no defect"},
    "wrong_groundtruth": {"action": "change", "fallback": "drop",
                          "note": "re-key to correct_answer; drop if none is provided"},
    "multiple_correct_answers": {"action": "union",
                                 "note": "gold = {original} ∪ {other correct options}"},
    "no_correct_answer": {"action": "drop",
                          "note": "no option is correct → not scorable"},
    "bad_question_clarity": {"action": "keep", "note": "ambiguity → item_ambiguity probe"},
    "bad_options_clarity": {"action": "keep", "note": "ambiguity → item_ambiguity probe"},
    "expert": {"action": "keep", "note": "flagged for expert review; no key change provided"},
}
DEFAULT_ACTION = "keep"   # unknown / unannotated error types: conservative, no re-key


@dataclass
class CorrectionDecision:
    raw_error_type: Optional[str]
    error_type: Optional[str]          # canonicalised
    action: str                        # keep | change | union | drop
    original_gold: List[str]
    corrected_gold: Optional[List[str]]  # None when dropped (not scorable)
    scored: bool
    changed: bool                      # corrected_gold differs from original_gold
    source: str
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_error_type": self.raw_error_type, "error_type": self.error_type,
            "action": self.action, "original_gold": self.original_gold,
            "corrected_gold": self.corrected_gold, "scored": self.scored,
            "changed": self.changed, "source": self.source, "note": self.note,
        }


def parse_correct_answer(raw: Any, choices: Iterable[str], base: int = 0) -> Optional[Set[str]]:
    """Parse a ``correct_answer`` field into a set of option letters.

    Handles a numeric index (``base``-relative), an exact option-text match, a unique
    substring match, or bare A–D letters. Returns ``None`` when nothing parses. Faithful
    port of the generator's parser, kept here so parsing is part of the declared policy.
    """
    s = "" if raw is None else str(raw).strip()
    if not s:
        return None
    choices = list(choices)
    if re.fullmatch(r"[0-9]+", s):
        i = int(s) - base
        return {LETTERS[i]} if 0 <= i < 4 else None
    low = s.lower()
    for i, c in enumerate(choices):
        if c and str(c).strip().lower() == low:
            return {LETTERS[i]}
    hit = [i for i, c in enumerate(choices) if c and str(c).strip().lower() in low]
    if len(hit) == 1:
        return {LETTERS[hit[0]]}
    letters = set(re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", s.upper()))
    return letters or None


def _as_set(gold: Any) -> Set[str]:
    if gold is None:
        return set()
    if isinstance(gold, str):
        return {gold}
    return {str(x) for x in gold}


def apply_policy(error_type: Any, original_gold: Any, correct_answer: Any = None,
                 choices: Optional[Iterable[str]] = None,
                 *, policy: Optional[Dict[str, Dict[str, Any]]] = None,
                 base: int = 0) -> CorrectionDecision:
    """Apply the declared correction policy to one item.

    Returns a ``CorrectionDecision`` carrying the action, the resulting gold (or
    ``None`` when dropped), whether it is scored, whether the key changed, and the
    source for the action. This is the single executable form of the policy.
    """
    policy = policy or DEFAULT_POLICY
    choices = list(choices or [])
    canon = canonical_error_type(error_type)
    orig = _as_set(original_gold)
    rule = policy.get(canon or "ok", {"action": DEFAULT_ACTION})
    action = rule.get("action", DEFAULT_ACTION)

    parsed = (parse_correct_answer(correct_answer, choices, base=base)
              if action in ("change", "union") else None)

    if action == "change":
        if parsed:
            corrected: Optional[Set[str]] = parsed
            scored = True
        else:                                   # no correction supplied -> fallback
            action = rule.get("fallback", "drop")
            corrected, scored = (None, False) if action == "drop" else (orig, True)
    elif action == "union":
        corrected, scored = (set(orig) | (parsed or set())), True
    elif action == "drop":
        corrected, scored = None, False
    else:  # keep
        action, corrected, scored = "keep", set(orig), True

    changed = bool(scored and corrected is not None and set(corrected) != orig)
    return CorrectionDecision(
        raw_error_type=(None if error_type is None else str(error_type)),
        error_type=canon,
        action=action,
        original_gold=sorted(orig),
        corrected_gold=(None if corrected is None else sorted(corrected)),
        scored=scored,
        changed=changed,
        source=_ACTION_SOURCE[action],
        note=str(rule.get("note", "")),
    )


def policy_table(policy: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The declared policy as a flat, displayable table — every row carries its source."""
    policy = policy or DEFAULT_POLICY
    order = ["ok", "wrong_groundtruth", "multiple_correct_answers", "no_correct_answer",
             "bad_question_clarity", "bad_options_clarity", "expert"]
    rows = []
    for et in order:
        if et not in policy:
            continue
        r = policy[et]
        act = r["action"]
        rows.append({
            "error_type": et,
            "action": act,
            "fallback": r.get("fallback"),
            "scored": act not in ("drop",),
            "effect": r.get("note", ""),
            "is_defect": et in DEFECT_ERROR_TYPES,
            "is_ambiguity": et in AMBIGUITY_ERROR_TYPES,
            "source": _ACTION_SOURCE[act],
        })
    rows.append({
        "error_type": "(any other / unannotated)", "action": DEFAULT_ACTION,
        "fallback": None, "scored": True, "effect": "conservative default, no re-key",
        "is_defect": False, "is_ambiguity": False, "source": _ACTION_SOURCE[DEFAULT_ACTION],
    })
    return rows


def render_policy_table(policy: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """Plain-text rendering for the CLI."""
    rows = policy_table(policy)
    out = ["Correction policy (error_type -> action) — declared, single source of truth",
           "%-26s %-7s %-7s %s" % ("error_type", "action", "scored", "effect"),
           "-" * 78]
    for r in rows:
        out.append("%-26s %-7s %-7s %s" % (r["error_type"], r["action"],
                                           "yes" if r["scored"] else "NO", r["effect"]))
    out.append("")
    out.append("Sources:")
    for act in ("keep", "change", "union", "drop"):
        out.append("  %-7s %s" % (act, _ACTION_SOURCE[act]))
    out.append("")
    out.append(CORRECTION_IS_CANDIDATE)
    return "\n".join(out)
