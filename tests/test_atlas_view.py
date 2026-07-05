"""Tests for the fragility-atlas observatory view."""
from trust_the_eval.atlas_view import (_cells_table, _law_block, _scatter,
                                       render_atlas_html)

LAW = {
    "n": 25, "insufficient": False,
    "corr_p_top1_vs_discrimination": -0.565, "ci_discrimination": (-0.73, -0.23),
    "corr_p_top1_vs_model_gap": -0.515, "ci_model_gap": (-0.68, -0.18),
    "corr_p_top1_vs_delta_spread": 0.883, "ci_delta_spread": (0.68, 0.96),
    "median_discrimination": 0.2, "median_gap": 0.01,
    "quadrant_mean_p_top1": {"close_disc": 0.13, "close_nondisc": 0.30,
                             "far_disc": 0.01, "far_nondisc": 0.15},
    "quadrant_counts": {"close_disc": 6, "close_nondisc": 7, "far_disc": 7, "far_nondisc": 5},
    "law_holds": False,
}
ROWS = [
    {"cell": "MMLU::virology", "n_changed": 37, "kendall_tau": 0.29, "tau_lo": 0.1, "tau_hi": 0.5,
     "p_top1_change": 0.76, "skill_discrimination": 0.16, "model_gap_12": 0.0, "severity": "high"},
    {"cell": "MMLU::college_chemistry", "n_changed": 21, "kendall_tau": 0.56, "tau_lo": 0.3,
     "tau_hi": 0.8, "p_top1_change": 0.64, "skill_discrimination": 0.49, "model_gap_12": 0.0,
     "severity": "high"},
    {"cell": "MMLU::abstract_algebra", "n_changed": 5, "kendall_tau": 0.82, "tau_lo": 0.6,
     "tau_hi": 1.0, "p_top1_change": 0.16, "skill_discrimination": -0.23, "model_gap_12": 0.011,
     "severity": "high"},
]


def test_law_block_shows_correlations_cis_and_verdict():
    h = _law_block(LAW)
    assert "corr( P(top1) , discrimination )" in h
    assert "-0.565" in h and "[-0.73, -0.23]" in h
    assert "does NOT cleanly hold" in h          # law_holds False
    assert "discriminating" in h and "flat" in h  # quadrant labels


def test_law_block_insufficient():
    h = _law_block({"insufficient": True, "note": "need >= 4 active cells", "n": 2})
    assert "active cells" in h and "n=2" in h


def test_scatter_renders_svg_with_a_dot_per_cell():
    svg = _scatter(ROWS)
    assert "<svg" in svg and svg.count("<circle") == 3
    assert "skill discrimination" in svg


def test_cells_table_lists_active_cells_with_severity():
    h = _cells_table(ROWS)
    assert "MMLU::virology" in h and "sev-high" in h
    assert "P(top-1 change)" in h


def test_render_atlas_html_structure():
    pred = {"TEST::c": [
        {"item": "TEST::c::q1::1", "subject": "c", "original_gold": ["B"], "corrected_gold": ["A"],
         "preds": {"m1": "A", "m2": "B", "m3": "A"}},
        {"item": "TEST::c::q2::0", "subject": "c", "original_gold": ["A"], "corrected_gold": ["A"],
         "preds": {"m1": "A", "m2": "C", "m3": "A"}},
    ]}
    intr = {"TEST::c": [{"question": "q1", "choices": ["w", "x", "y", "z"], "answer": 1,
                         "error_type": "wrong_groundtruth", "subject": "c"}]}
    html = render_atlas_html(pred, intr, iters=200, boot_iters=200)
    assert "fragility atlas" in html.lower()
    assert "<script" not in html and "fetch(" not in html
    assert "no model is scored" in html.lower()
    for w in ("Trust Score", "trust score", "overall grade"):
        assert w not in html
