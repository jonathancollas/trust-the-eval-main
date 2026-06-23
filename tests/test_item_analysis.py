"""Tests for item_analysis (CTT): deterministic, known-answer cases."""
from trust_the_eval.item_analysis import (analyze, concordance,
                                          correctness_from_predictions,
                                          cronbach_alpha, gold_disagreement,
                                          reliability_summary, reliability_audit)


def _matrix(rows):
    # rows: dict model -> list of 0/1 over items i0..i{n-1}
    return {m: {f"i{k}": v for k, v in enumerate(vals)} for m, vals in rows.items()}


def test_discrimination_sign():
    # 3 models of decreasing skill; item i0 tracks skill (good), i1 is reversed.
    cm = _matrix({
        "strong": [1, 0, 1, 1],   # ability 3/4
        "mid":    [1, 0, 1, 0],   # ability 2/4
        "weak":   [0, 1, 1, 0],   # ability 2/4 (tie ok)
    })
    a = analyze(cm)
    pi = a["per_item"]
    assert pi["i0"]["discrimination"] is not None and pi["i0"]["discrimination"] > 0
    assert pi["i1"]["discrimination"] is not None and pi["i1"]["discrimination"] < 0
    assert "i1" in a["suspect_items"]          # reversed item flagged suspect
    assert pi["i2"]["discrimination"] is None  # constant item (all right) -> undefined


def test_cronbach_alpha_reasonable():
    # consistent test: same rank pattern across items -> high alpha
    cm = _matrix({
        "a": [1, 1, 1, 1],
        "b": [1, 1, 1, 0],
        "c": [1, 1, 0, 0],
        "d": [0, 0, 0, 0],
    })
    al = cronbach_alpha(cm, [f"i{k}" for k in range(4)])
    assert al is not None and al > 0.7


def test_concordance_recovers_flags():
    # i3 is a reversed (suspect) item; human flags i3 as broken.
    cm = _matrix({
        "strong": [1, 1, 1, 0],
        "mid":    [1, 1, 0, 0],
        "weak":   [0, 0, 0, 1],
    })
    a = analyze(cm)
    human = {"i0": False, "i1": False, "i2": False, "i3": True}
    c = concordance(a, human)
    assert c["n"] == 4 and c["n_human_flagged"] == 1
    assert c["recall"] == 1.0          # the suspect item is recovered
    assert c["roc_auc"] is not None and c["roc_auc"] >= 0.5


def test_correctness_helper():
    preds = {"m1": {"i0": "A", "i1": "B"}, "m2": {"i0": "B", "i1": "B"}}
    gold = {"i0": ["A"], "i1": ["A"]}
    cm = correctness_from_predictions(preds, gold)
    assert cm["m1"]["i0"] == 1 and cm["m1"]["i1"] == 0
    assert cm["m2"]["i0"] == 0 and cm["m2"]["i1"] == 0


def test_gold_disagreement_is_one_minus_difficulty():
    cm = _matrix({"a": [1, 0, 1], "b": [1, 0, 0], "c": [0, 1, 0]})
    a = analyze(cm)
    gd = gold_disagreement(a)
    # i0: 2/3 correct -> disagreement 1/3 ; i1: 1/3 correct -> 2/3 ; i2: 1/3 -> 2/3
    assert abs(gd["i0"] - (1 - 2/3)) < 1e-9
    assert abs(gd["i1"] - (1 - 1/3)) < 1e-9
    # highest-disagreement item is a label-error candidate
    worst = max(gd, key=gd.get)
    assert worst in ("i1", "i2")


def test_reliability_summary_pooled_and_per_subject():
    cm = _matrix({"a": [1, 1, 1, 1], "b": [1, 1, 1, 0], "c": [1, 1, 0, 0], "d": [0, 0, 0, 0]})
    r = reliability_summary(cm)
    assert r["pooled_alpha"] is not None and r["n_models"] == 4
    assert r["per_subject"] is False
    # with subjects: i0,i1 -> S1 ; i2,i3 -> S2
    subs = {"i0": "S1", "i1": "S1", "i2": "S2", "i3": "S2"}
    r2 = reliability_summary(cm, subs)
    assert r2["per_subject"] is True and r2["n_subjects"] >= 1
    assert "median_alpha" in r2 and "inflated" in r2["note"]
    a = reliability_audit(cm, subs)
    assert a.probe_id == "test_reliability" and a.severity in ("low", "medium", "info")
