"""Tests for the per-datum lineage tracer.

The lineage must *assemble* the canonical functions, never re-implement them — so
its numbers must equal a direct call to ``sensitivity`` and ``per_item_records``
(the recompute guarantee, extended to the lineage). Dropped items stay visible, and
the verdict never scores a model (the bright line).
"""
import pytest

from trust_the_eval.lineage import format_lineage, lineage
from trust_the_eval.result_sensitivity import sensitivity
from trust_the_eval.transparency import _pog, per_item_records

BENCH = "TEST::demo"
PRED = [
    {"item": "TEST::demo::what is x::1", "subject": "demo",
     "original_gold": ["B"], "corrected_gold": ["A"],
     "preds": {"m1": "A", "m2": "B", "m3": "A", "m4": "B"}},
    {"item": "TEST::demo::what is y::0", "subject": "demo",
     "original_gold": ["A"], "corrected_gold": ["A"],
     "preds": {"m1": "A", "m2": "C", "m3": "A", "m4": "A"}},
    {"item": "TEST::demo::what is z::2", "subject": "demo",
     "original_gold": ["C"], "corrected_gold": ["C"],
     "preds": {"m1": "C", "m2": "C", "m3": "B", "m4": "A"}},
]
INTR = [
    {"question": "What is X", "choices": ["o0", "o1", "o2", "o3"], "answer": 1,
     "error_type": "Wrong groundtruth", "subject": "demo"},
    {"question": "What is Y", "choices": ["o0", "o1", "o2", "o3"], "answer": 0,
     "error_type": "ok", "subject": "demo"},
    {"question": "What is Z", "choices": ["o0", "o1", "o2", "o3"], "answer": 2,
     "error_type": "ok", "subject": "demo"},
    {"question": "no answer here", "choices": ["o0", "o1", "o2", "o3"], "answer": 0,
     "error_type": "no_correct_answer", "subject": "demo"},
]
ITEM = "TEST::demo::what is x::1"


def test_thread_has_the_expected_ordered_steps():
    th = lineage(BENCH, ITEM, PRED, INTR, iters=200, seed=0)
    assert th.scored is True
    assert [s.n for s in th.steps] == [1, 2, 3, 4, 5, 6]
    names = " ".join(s.name.lower() for s in th.steps)
    for kw in ("normalised", "correction policy", "correctness", "accuracy", "ranking", "verdict"):
        assert kw in names


def test_every_step_carries_input_rule_output_source():
    th = lineage(BENCH, ITEM, PRED, INTR, iters=200, seed=0)
    for s in th.steps:
        assert s.input and s.rule and s.output and s.source
    by_n = {s.n: s for s in th.steps}
    assert by_n[3].formula and "pred" in by_n[3].formula        # correctness formula present
    assert by_n[5].formula and "τ" in by_n[5].formula           # kendall formula present


def test_lineage_numbers_match_the_canonical_recompute():
    th = lineage(BENCH, ITEM, PRED, INTR, iters=200, seed=0)
    by_n = {s.n: s for s in th.steps}

    # ranking step == result_sensitivity (deterministic parts)
    p, o, c, _ = _pog(PRED)
    res = sensitivity(p, o, c, iters=200, seed=0)
    assert by_n[5].data["kendall_tau"] == res["kendall_tau"]
    assert by_n[5].data["C"] == res["n_pairs"] - res["inversions"]
    assert by_n[5].data["D"] == res["inversions"]
    assert by_n[5].data["n_pairs"] == res["n_pairs"]
    assert 0.0 <= (by_n[5].data["p_top1_change"] or 0.0) <= 1.0

    # correctness step == per_item_records for this item
    models, recs = per_item_records(BENCH, PRED, INTR)
    rec = next(r for r in recs if r["item"] == ITEM)
    assert by_n[3].data["per_model"] == {m: rec["models"][m] for m in models}


def test_correction_step_reflects_the_declared_policy():
    th = lineage(BENCH, ITEM, PRED, INTR, iters=200, seed=0)
    step2 = next(s for s in th.steps if s.n == 2)
    assert step2.data["action"] == "change"
    assert step2.data["corrected_gold"] == ["A"]
    assert step2.data["changed"] is True
    assert "candidate, not a truth" in step2.source.lower()


def test_dropped_item_stays_visible_and_stops():
    th = lineage(BENCH, "no answer here", PRED, INTR, iters=200, seed=0)
    assert th.scored is False
    assert len(th.steps) == 1
    assert th.steps[0].data["action"] == "drop"
    assert th.steps[0].data["scored"] is False
    assert "dropped" in format_lineage(th).lower()


def test_verdict_scores_no_model_bright_line():
    th = lineage(BENCH, ITEM, PRED, INTR, iters=200, seed=0)
    verdict = next(s for s in th.steps if s.n == 6)
    assert "no model is scored" in verdict.source.lower()
    # the verdict carries only the conditional rank flag — no per-model rating
    assert set(verdict.data.keys()) == {"rank_fragile"}
    assert isinstance(verdict.data["rank_fragile"], bool)
    # no step invents an aggregate "trust score"
    for s in th.steps:
        assert "trust score" not in (s.output + s.name).lower()


def test_format_lineage_renders():
    th = lineage(BENCH, ITEM, PRED, INTR, iters=200, seed=0)
    txt = format_lineage(th)
    assert "lineage" in txt and "scored: yes" in txt
    assert "src :" in txt and "f   :" in txt
