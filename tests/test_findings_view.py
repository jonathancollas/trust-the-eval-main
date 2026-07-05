"""Tests for the Findings hub view."""
from trust_the_eval.detection import confront
from trust_the_eval.findings_view import render_findings_html

PRED, INTR = [], []
for i in range(8):
    PRED.append({"item": "S::ok%d::0" % i, "subject": "S", "original_gold": ["A"],
                 "corrected_gold": ["A"], "preds": {"m1": "A", "m2": "A", "m3": "A"}})
    INTR.append({"question": "ok%d" % i, "choices": ["a", "b", "c", "d"], "answer": 0,
                 "error_type": "ok", "subject": "S"})
for i in range(4):
    PRED.append({"item": "S::bad%d::0" % i, "subject": "S", "original_gold": ["A"],
                 "corrected_gold": ["B"], "preds": {"m1": "B", "m2": "B", "m3": "C"}})
    INTR.append({"question": "bad%d" % i, "choices": ["a", "b", "c", "d"], "answer": 0,
                 "error_type": "wrong_groundtruth", "subject": "S"})


def _html():
    return render_findings_html({"S": PRED}, {"S": INTR}, iters=200, boot_iters=200)


def test_states_two_findings_and_links_to_their_pages():
    h = _html()
    assert h.lower().count("self-correction") >= 2      # both findings framed as self-corrections
    for p in ("atlas.html", "detection.html", "lineage.html", "index.html"):
        assert 'href="%s"' % p in h


def test_detection_finding_uses_recomputed_number():
    res = confront({"S": (PRED, INTR)}, iters=200, seed=0, ks=(50, 100))
    auc = res["methods"]["disagreement"]["auc"]
    assert ("disagreement detector AUC = <b>%.3f</b>" % auc) in _html()


def test_no_script_no_aggregate():
    h = _html()
    assert "<script" not in h and "fetch(" not in h
    for w in ("Trust Score", "trust score", "overall grade"):
        assert w not in h
