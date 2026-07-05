"""Tests for the detection confrontation."""
from trust_the_eval.detection import (confront, format_confront, precision_at_k,
                                      roc_auc)


def test_roc_auc_perfect_reversed_and_ties():
    # perfect separation (positives score higher)
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    # perfectly reversed
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    # all-ties -> 0.5 (no separation)
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # single class -> undefined
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_precision_at_k():
    labels = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    assert precision_at_k(labels, scores, 2) == 0.5      # top-2: one positive
    assert precision_at_k(labels, scores, 3) == pytest_approx(2 / 3)


def pytest_approx(x):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < 1e-9
    return _A()


def _cell_with_planted_signal():
    # defect items: models disagree with the (wrong) gold; ok items: models agree.
    pred, intr = [], []
    for i in range(8):  # ok items — everyone right
        pred.append({"item": "S::ok%d::0" % i, "subject": "S",
                     "original_gold": ["A"], "corrected_gold": ["A"],
                     "preds": {"m1": "A", "m2": "A", "m3": "A", "m4": "A"}})
        intr.append({"question": "ok%d" % i, "choices": ["a", "b", "c", "d"], "answer": 0,
                     "error_type": "ok", "subject": "S"})
    for i in range(4):  # wrong_groundtruth — everyone disagrees with the stated key
        pred.append({"item": "S::bad%d::0" % i, "subject": "S",
                     "original_gold": ["A"], "corrected_gold": ["B"],
                     "preds": {"m1": "B", "m2": "B", "m3": "B", "m4": "C"}})
        intr.append({"question": "bad%d" % i, "choices": ["a", "b", "c", "d"], "answer": 0,
                     "error_type": "wrong_groundtruth", "subject": "S"})
    return {"S": (pred, intr)}


def test_confront_structure_and_planted_signal():
    res = confront(_cell_with_planted_signal(), iters=300, seed=0, ks=(4,))
    assert res["n_pos"] == 4 and res["n_neg"] == 8
    assert set(res["methods"]) == {"disagreement", "ability_weighted",
                                   "consensus_disagree", "answer_entropy"}
    # the planted signal is exactly disagreement -> its AUC is perfect
    assert res["methods"]["disagreement"]["auc"] == 1.0
    assert res["methods"]["disagreement"]["prec_at_k"][4] == 1.0
    # ranked is sorted by AUC desc
    aucs = [res["methods"][m]["auc"] or 0 for m in res["ranked"]]
    assert aucs == sorted(aucs, reverse=True)


def test_ability_weighted_is_always_defined():
    # even when all models agree (ok) or all disagree (bad), no crash / no None from the score
    res = confront(_cell_with_planted_signal(), iters=100, seed=1, ks=(2,))
    assert res["methods"]["ability_weighted"]["auc"] is not None


def test_no_aggregate_score_and_caveat_present():
    res = confront(_cell_with_planted_signal(), iters=100, seed=0)
    for k in ("score", "overall", "trust_score", "grade", "rating"):
        assert k not in res
    txt = format_confront(res)
    assert "circularity" in txt.lower()                  # the honesty caveat is surfaced
    assert "AUC" in txt and "chance" in txt
