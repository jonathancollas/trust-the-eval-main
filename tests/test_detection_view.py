"""Tests for the detection-confrontation observatory view."""
from trust_the_eval.detection_view import render_detection_html

PRED, INTR = [], []
for i in range(8):   # ok items
    PRED.append({"item": "S::ok%d::0" % i, "subject": "S", "original_gold": ["A"],
                 "corrected_gold": ["A"], "preds": {"m1": "A", "m2": "A", "m3": "A"}})
    INTR.append({"question": "ok%d" % i, "choices": ["a", "b", "c", "d"], "answer": 0,
                 "error_type": "ok", "subject": "S"})
for i in range(4):   # wrong_groundtruth items
    PRED.append({"item": "S::bad%d::0" % i, "subject": "S", "original_gold": ["A"],
                 "corrected_gold": ["B"], "preds": {"m1": "B", "m2": "B", "m3": "C"}})
    INTR.append({"question": "bad%d" % i, "choices": ["a", "b", "c", "d"], "answer": 0,
                 "error_type": "wrong_groundtruth", "subject": "S"})


def _html():
    return render_detection_html({"S": PRED}, {"S": INTR}, iters=200, seed=0)


def test_renders_auc_table_and_detectors():
    h = _html()
    assert "disagreement" in h and "ability_weighted" in h and "answer_entropy" in h
    assert "AUC" in h and "P@50" in h


def test_circularity_caveat_is_front_and_centre():
    h = _html().lower()
    assert "circular" in h and "upper bound" in h
    assert "no model is scored" in h


def test_has_back_link_and_no_script():
    h = _html()
    assert 'href="index.html"' in h                      # navigation is not a dead-end
    assert "<script" not in h and "fetch(" not in h


def test_no_aggregate_score():
    h = _html()
    for w in ("Trust Score", "trust score", "overall grade"):
        assert w not in h
