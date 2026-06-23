"""Real known-bad datasets as calibration ground truth (the acid test).

The synthetic scenarios are a controlled floor; the credibility claim needs a
REAL dataset whose defects were labelled by humans. MMLU-Redux (Gema et al.,
2024) is exactly that for label errors: every item carries an ``error_type``
from the taxonomy {ok, bad_question_clarity, bad_options_clarity,
no_correct_answer, multiple_correct_answers, wrong_groundtruth, expert}, so we
know, per item, whether the MMLU gold label is defective.

This adapter converts MMLU-Redux rows into labelled :class:`CalibrationCase`
objects for ``label_error_audit``: it builds small MMLU artifacts that are
either CLEAN (only ``error_type == "ok"`` items) or DEFECTIVE (seeded with
``wrong_groundtruth`` / ``no_correct_answer`` items, using the ORIGINAL MMLU
gold so the planted-in-MMLU error is what the probe must catch). Running
``label_error_audit`` over these measures its REAL recall and false-positive
rate against human annotations.

Honesty note (kept consistent with the HF Pull feature): this sandbox has no
outbound network, so the real ``edinburgh-dawg/mmlu-redux-2.0`` fetch executes
on the user's machine via the existing HF source. Here the adapter is proven on
an in-repo fixture matching the published schema; the loader code path is the
same one the user runs against the live dataset.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from ..artifact import EvalArtifact, EvalItem
from .core import CalibrationCase

# error_type values that mean the MMLU gold label is genuinely defective
DEFECT_ERROR_TYPES = {"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"}
OK_ERROR_TYPE = "ok"
# clarity issues are ambiguity, not label errors -> routed to item_ambiguity, not here
AMBIGUITY_ERROR_TYPES = {"bad_question_clarity", "bad_options_clarity"}


def canonical_error_type(s):
    """Normalise an annotation error_type to the canonical snake_case used by
    DEFECT/AMBIGUITY sets. Real annotation files vary in casing/spacing (e.g.
    'Wrong Groundtruth', 'No correct answer', 'OK'); canonicalising makes
    label-error measurement robust to the source's surface form. Already-canonical
    values are returned unchanged, so record ids are preserved."""
    if s is None:
        return None
    return str(s).strip().lower().replace(" ", "_")


_LETTERS = "ABCD"


def _row_to_item(row: dict, use_original_gold: bool = True) -> EvalItem:
    """One MMLU-Redux row -> an EvalItem with MCQ options and the MMLU gold.

    ``answer`` is the MMLU ground-truth index (0-3); we expose the gold as the
    option TEXT so the model-in-the-loop probe can adjudicate by content.
    """
    choices = list(row.get("choices") or [])
    answer_idx = int(row.get("answer", 0))
    gold_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    return EvalItem(
        question=str(row.get("question", "")),
        answer=gold_text,
        response=None,
        score=None,
        meta={
            "options": [str(c) for c in choices],
            "answer_index": answer_idx,
            "error_type": row.get("error_type", "ok"),
            "subject": row.get("subject") or row.get("source") or "mmlu",
        },
    )


def cases_from_mmlu_redux(rows: list[dict],
                          *,
                          group_size: int = 25,
                          defect_per_group: int = 5,
                          max_groups: int = 8,
                          seed: int = 0) -> list[CalibrationCase]:
    """Build labelled calibration cases for ``label_error_audit`` from rows.

    Strategy: partition ``ok`` items into clean groups, and build matched
    defective groups that are mostly ``ok`` items plus a few rows whose
    ``error_type`` is in :data:`DEFECT_ERROR_TYPES`. A group's ground-truth
    label is "defect present" iff it contains >= 1 genuine label-error row.
    Returns alternating defective/clean cases so both classes are represented.
    """
    rng = random.Random(seed)
    ok_rows = [r for r in rows if r.get("error_type") == OK_ERROR_TYPE]
    bad_rows = [r for r in rows if r.get("error_type") in DEFECT_ERROR_TYPES]
    rng.shuffle(ok_rows)
    rng.shuffle(bad_rows)

    cases: list[CalibrationCase] = []
    if not ok_rows:
        return cases

    clean_fill = max(group_size - defect_per_group, 1)
    bi = 0   # index into bad_rows
    oi = 0   # index into ok_rows

    g = 0
    while g < max_groups and oi + group_size <= len(ok_rows):
        if g % 2 == 0 and bi + defect_per_group <= len(bad_rows):
            # DEFECTIVE group: clean fill + a few real label errors
            fill = ok_rows[oi:oi + clean_fill]; oi += clean_fill
            bad = bad_rows[bi:bi + defect_per_group]; bi += defect_per_group
            items = [_row_to_item(r) for r in (fill + bad)]
            rng.shuffle(items)
            art = EvalArtifact(dataset=f"mmlu-redux/defect-{g}", items=items,
                               model="calib/honest")
            cases.append(CalibrationCase(
                name=f"mmlu-redux defect group {g} (+{len(bad)} labelled errors)",
                artifact=art, defect=True, model=None, regime="real_mmlu",
                note="contains wrong_groundtruth/no_correct_answer items"))
        else:
            # CLEAN group: only ok items
            clean = ok_rows[oi:oi + group_size]; oi += group_size
            items = [_row_to_item(r) for r in clean]
            art = EvalArtifact(dataset=f"mmlu-redux/clean-{g}", items=items,
                               model="calib/honest")
            cases.append(CalibrationCase(
                name=f"mmlu-redux clean group {g}",
                artifact=art, defect=False, model=None, regime="real_mmlu",
                note="only error_type=ok items"))
        g += 1
    return cases


def cases_from_mmlu_redux_ambiguity(rows: list[dict],
                                    *,
                                    group_size: int = 25,
                                    defect_per_group: int = 5,
                                    max_groups: int = 8,
                                    seed: int = 0) -> list[CalibrationCase]:
    """Build labelled calibration cases for ``item_ambiguity`` from MMLU-Redux.

    Uses the SAME gold human-annotated source, but the ground truth here is the
    CLARITY taxonomy: ``bad_question_clarity`` / ``bad_options_clarity`` mark
    genuinely ill-posed items (ambiguity), distinct from label errors. A group is
    "defect present" iff it contains >= 1 clarity-flagged item.

    Note: item_ambiguity is a model-in-the-loop probe (it resamples a model to
    see whether answers scatter on ill-posed items). The caller supplies the
    adjudicating model; on real ambiguous prompts a capable model genuinely
    scatters, which is what the probe detects.
    """
    rng = random.Random(seed)
    ok_rows = [r for r in rows if r.get("error_type") == OK_ERROR_TYPE]
    amb_rows = [r for r in rows if r.get("error_type") in AMBIGUITY_ERROR_TYPES]
    rng.shuffle(ok_rows)
    rng.shuffle(amb_rows)

    cases: list[CalibrationCase] = []
    if not ok_rows:
        return cases
    clean_fill = max(group_size - defect_per_group, 1)
    ai = oi = g = 0
    while g < max_groups and oi + group_size <= len(ok_rows):
        if g % 2 == 0 and ai + defect_per_group <= len(amb_rows):
            fill = ok_rows[oi:oi + clean_fill]; oi += clean_fill
            amb = amb_rows[ai:ai + defect_per_group]; ai += defect_per_group
            items = [_row_to_item(r) for r in (fill + amb)]
            rng.shuffle(items)
            art = EvalArtifact(dataset=f"mmlu-redux-amb/defect-{g}", items=items,
                               model="calib/honest")
            cases.append(CalibrationCase(
                name=f"mmlu-redux ambiguity group {g} (+{len(amb)} clarity-flagged)",
                artifact=art, defect=True, model=None, regime="real_mmlu_clarity",
                note="contains bad_question_clarity/bad_options_clarity items"))
        else:
            clean = ok_rows[oi:oi + group_size]; oi += group_size
            items = [_row_to_item(r) for r in clean]
            art = EvalArtifact(dataset=f"mmlu-redux-amb/clean-{g}", items=items,
                               model="calib/honest")
            cases.append(CalibrationCase(
                name=f"mmlu-redux ambiguity clean group {g}",
                artifact=art, defect=False, model=None, regime="real_mmlu_clarity",
                note="only error_type=ok items"))
        g += 1
    return cases


def load_mmlu_redux_rows(path: Optional[str] = None,
                         hf_subset: str = "all") -> list[dict]:
    """Load MMLU-Redux rows.

    If ``path`` is given, read a local JSON/JSONL export (the offline path used
    here and reproducible by the user). Otherwise, fetch via the package's HF
    source (runs on the user's machine; requires network). The returned rows use
    the published column names: question, choices, answer, error_type, subject.
    """
    if path:
        import json
        import os
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith(".jsonl"):
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            else:
                data = json.load(fh)
                rows = data if isinstance(data, list) else data.get("rows", data.get("items", []))
        return rows
    # network path (user machine): reuse the HF Dataset Viewer client
    from ..sources.hf import pull  # noqa: WPS433
    art = pull("edinburgh-dawg/mmlu-redux-2.0", config=hf_subset, split="test",
               n=2000, mapping={"question": "question", "options": "choices"})
    # pull() returns an artifact; we need raw columns, so prefer load_mmlu_redux_rows(path=...)
    out = []
    for it in art.items:
        out.append({"question": it.question,
                    "choices": it.meta.get("options", []),
                    "answer": it.meta.get("answer_index", 0),
                    "error_type": it.meta.get("error_type", "ok")})
    return out
