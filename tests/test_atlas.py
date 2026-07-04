"""Tests for the fragility atlas.

The atlas must read every per-cell quantity from the canonical ``sensitivity`` (no
re-derivation), test the conditional law honestly (reporting it holds or not), attach
bootstrap CIs, and never emit an aggregate score.
"""

from trust_the_eval.atlas import (cell_metrics, conditional_law_test,
                                  fragility_atlas, format_atlas)
from trust_the_eval.result_sensitivity import sensitivity
from trust_the_eval.transparency import _pog

CELL = "TEST::c"
PRED = [
    {"item": "TEST::c::q1::1", "subject": "c", "original_gold": ["B"], "corrected_gold": ["A"],
     "preds": {"m1": "A", "m2": "B", "m3": "A", "m4": "B"}},
    {"item": "TEST::c::q2::0", "subject": "c", "original_gold": ["A"], "corrected_gold": ["A"],
     "preds": {"m1": "A", "m2": "C", "m3": "A", "m4": "A"}},
    {"item": "TEST::c::q3::2", "subject": "c", "original_gold": ["C"], "corrected_gold": ["C"],
     "preds": {"m1": "C", "m2": "C", "m3": "B", "m4": "A"}},
]
INTR = [{"question": "q%d" % i, "choices": ["w", "x", "y", "z"], "answer": a, "error_type": e,
         "subject": "c"} for i, a, e in [(1, 1, "wrong_groundtruth"), (2, 0, "ok"), (3, 2, "ok")]]


def test_cell_metrics_reads_from_canonical_sensitivity():
    m = cell_metrics(CELL, PRED, INTR, iters=300, seed=0)
    p, o, c, _ = _pog(PRED)
    r = sensitivity(p, o, c, iters=300, seed=0)
    assert m["p_top1_change"] == r["p_top1_change"]
    assert m["kendall_tau"] == r["kendall_tau"]
    assert m["skill_discrimination"] == r["skill_discrimination"]
    assert m["n_changed"] == r["n_changed_items"]
    # closeness = the #1-#2 accuracy gap under the original key
    accs = sorted((v["acc_orig"] for v in r["per_model"].values()), reverse=True)
    assert abs(m["model_gap_12"] - (accs[0] - accs[1])) < 1e-9


def _row(gap, disc, ptop, spread=0.01):
    return {"n_changed": 5, "p_top1_change": ptop, "model_gap_12": gap,
            "skill_discrimination": disc, "delta_spread": spread}


def test_law_holds_when_close_and_discriminating_are_fragile():
    rows = ([_row(0.0, 0.6, 0.60) for _ in range(4)]      # close + discriminating -> fragile
            + [_row(0.2, -0.1, 0.05) for _ in range(4)])  # far + flat -> stable
    law = conditional_law_test(rows, boot_iters=200, seed=0)
    assert law["law_holds"] is True
    assert law["corr_p_top1_vs_discrimination"] > 0
    assert law["corr_p_top1_vs_model_gap"] < 0
    assert law["quadrant_mean_p_top1"]["close_disc"] >= law["quadrant_mean_p_top1"]["far_nondisc"]


def test_law_fails_when_discrimination_is_protective():
    # mirrors the real finding: among close models, flat corrections are MORE fragile
    rows = ([_row(0.0, 0.6, 0.10) for _ in range(4)]      # close + discriminating -> stable
            + [_row(0.0, -0.2, 0.40) for _ in range(4)])  # close + flat -> fragile
    law = conditional_law_test(rows, boot_iters=200, seed=0)
    assert law["law_holds"] is False
    assert law["corr_p_top1_vs_discrimination"] < 0       # protective, wrong sign for the law


def test_law_reports_bootstrap_cis():
    rows = [_row(0.0, 0.6, 0.60), _row(0.1, 0.2, 0.3), _row(0.2, -0.1, 0.05),
            _row(0.05, 0.4, 0.4), _row(0.15, 0.0, 0.1), _row(0.02, 0.5, 0.5)]
    law = conditional_law_test(rows, boot_iters=300, seed=0)
    assert law["ci_discrimination"][0] is not None and len(law["ci_discrimination"]) == 2
    assert law["ci_model_gap"][0] is not None


def test_law_insufficient_data():
    law = conditional_law_test([_row(0.0, 0.5, 0.4), _row(0.1, 0.2, 0.2)], boot_iters=50)
    assert law.get("insufficient") is True


def test_atlas_has_no_aggregate_score_and_renders():
    cells = {CELL: (PRED, INTR), "TEST::d": (PRED, INTR)}
    atlas = fragility_atlas(cells, iters=200, seed=0, boot_iters=200)
    assert "rows" in atlas and "law" in atlas and atlas["n_cells"] == 2
    # bright-line discipline: no single aggregate number rates the set
    for k in ("score", "overall", "trust_score", "grade", "rating"):
        assert k not in atlas
    txt = format_atlas(atlas)
    assert "fragility atlas" in txt and "conditional law" in txt and "P(top1)" in txt
