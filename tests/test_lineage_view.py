"""Tests for the observatory lineage drill-down view.

The page must render real numbers (the τ it shows equals ``sensitivity``), attach each
step's source, use a no-JS drill-down, surface dropped items, and never score a model.
"""

from trust_the_eval.lineage_view import render_lineage_html
from trust_the_eval.result_sensitivity import sensitivity
from trust_the_eval.transparency import _pog

BENCH = "TEST::demo"
PRED = {BENCH: [
    {"item": "TEST::demo::q1::1", "subject": "demo",
     "original_gold": ["B"], "corrected_gold": ["A"],
     "preds": {"m1": "A", "m2": "B", "m3": "A", "m4": "B"}},
    {"item": "TEST::demo::q2::0", "subject": "demo",
     "original_gold": ["A"], "corrected_gold": ["A"],
     "preds": {"m1": "A", "m2": "C", "m3": "A", "m4": "A"}},
]}
INTR = {BENCH: [
    {"question": "q1", "choices": ["w", "x", "y", "z"], "answer": 1,
     "error_type": "wrong_groundtruth", "subject": "demo"},
    {"question": "q2", "choices": ["w", "x", "y", "z"], "answer": 0,
     "error_type": "ok", "subject": "demo"},
    {"question": "no valid option", "choices": ["w", "x", "y", "z"], "answer": 0,
     "error_type": "no_correct_answer", "subject": "demo"},
]}


def _html():
    return render_lineage_html(PRED, INTR, iters=200, seed=0)


def test_renders_the_real_tau():
    p, o, c, _ = _pog(PRED[BENCH])
    res = sensitivity(p, o, c, iters=200, seed=0)
    assert ("τ = <b>%.3f</b>" % res["kendall_tau"]) in _html()


def test_is_a_no_js_drilldown():
    html = _html()
    assert "<details>" in html and "<summary>" in html
    assert "<script" not in html and "fetch(" not in html


def test_each_step_carries_its_source():
    html = _html()
    assert "Kendall" in html              # ranking step source
    assert "exact-match" in html          # correctness step source
    assert "candidate" in html.lower()    # correction step / footer


def test_shows_the_gold_change_and_flip():
    html = _html()
    assert "key B → A" in html
    assert "@orig" in html and "@corr" in html   # the per-model flip table


def test_dropped_items_are_surfaced():
    html = _html()
    assert "no_correct_answer" in html
    assert "dropped by policy" in html.lower()


def test_bright_line_no_model_scored():
    html = _html().lower()
    assert "no model is scored" in html
    assert "trust score" not in html


def test_empty_inputs_render_gracefully():
    html = render_lineage_html({}, {}, iters=50)
    assert "No predictions with corrections" in html
    assert "<script" not in html
