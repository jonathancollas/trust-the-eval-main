"""Comparison engine tests: Newcombe math, pairing, findings diff, ranking verdict."""
from __future__ import annotations

import math

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.artifact import EvalArtifact, EvalItem
from trust_the_eval.compare import (accuracy_block, compare, findings_diff,
                                    newcombe_diff, paired_block, ranking_verdict)
from trust_the_eval.stats import wilson_ci


def art(correct, qs=None, dataset="t"):
    qs = qs or [f"q{i}" for i in range(len(correct))]
    return EvalArtifact(dataset=dataset, items=[
        EvalItem(question=q, answer="a", score=1.0 if c else 0.0)
        for q, c in zip(qs, correct)])


def F(pid, sev):
    return {"probe_id": pid, "severity": sev, "summary": "s"}


# ----------------------------- math -----------------------------

def test_newcombe_matches_hand_computation():
    a_lo, a_hi = wilson_ci(80, 100)
    b_lo, b_hi = wilson_ci(70, 100)
    a = {"acc": 0.8, "lo": a_lo, "hi": a_hi}
    b = {"acc": 0.7, "lo": b_lo, "hi": b_hi}
    d = newcombe_diff(a, b)
    assert abs(d["d"] - 0.10) < 1e-9
    exp_lo = 0.10 - math.sqrt((0.8 - a_lo) ** 2 + (b_hi - 0.7) ** 2)
    exp_hi = 0.10 + math.sqrt((a_hi - 0.8) ** 2 + (0.7 - b_lo) ** 2)
    assert abs(d["lo"] - exp_lo) < 1e-12 and abs(d["hi"] - exp_hi) < 1e-12
    assert d["exceeds_noise"] is False  # CI crosses 0 at n=100


def test_newcombe_detects_a_large_real_gap():
    a_lo, a_hi = wilson_ci(90, 100)
    b_lo, b_hi = wilson_ci(60, 100)
    d = newcombe_diff({"acc": .9, "lo": a_lo, "hi": a_hi},
                      {"acc": .6, "lo": b_lo, "hi": b_hi})
    assert d["exceeds_noise"] is True and d["lo"] > 0


def test_paired_sign_analysis_beats_unpaired_on_correlated_data():
    # 62% vs 52% on 100 shared items, discordants 12 vs 2: the unpaired MOVER
    # interval crosses 0 (10-pt gap at n=100 is noise), but the paired sign
    # analysis on the 14 discordant items detects the gap at 95%.
    base = [True] * 50 + [False] * 36
    ca = base + [True] * 12 + [False] * 2
    cb = base + [False] * 12 + [True] * 2
    qs = [f"q{i}" for i in range(100)]
    res = compare(art(ca, qs), [], art(cb, qs), [])
    p = res["score"]["paired"]
    assert p and p["a_wins"] == 12 and p["b_wins"] == 2
    assert p["exceeds_noise"] is True
    assert res["score"]["gap"]["exceeds_noise"] is False  # the unpaired bound alone calls it noise
    assert res["ranking"]["verdict"] in ("sup", "fra")    # paired analysis supersedes the unpaired bound


def test_pairing_requires_overlap():
    a = art([True] * 30, qs=[f"a{i}" for i in range(30)])
    b = art([True] * 30, qs=[f"b{i}" for i in range(30)])
    assert paired_block({(it.question, it.answer): True for it in a.items},
                        {(it.question, it.answer): True for it in b.items}) is None
    res = compare(a, [], b, [])
    assert res["comparability"]["paired"] is False


def test_identical_artifacts_flagged():
    a = art([True, False] * 20)
    res = compare(a, [], a, [])
    assert res["comparability"]["same_content_hash"] is True
    assert any("zero by construction" in n for n in res["comparability"]["notes"])


def test_accuracy_block_none_without_scores():
    a = EvalArtifact(dataset="t", items=[EvalItem(question="q", answer="a")])
    assert accuracy_block(a) is None


# ----------------------------- findings diff -----------------------------

def test_findings_diff_classification():
    fa = [F("judge_swap", "high"), F("model_drift", "medium"), F("dataset_hygiene", "low")]
    fb = [F("judge_swap", "low"), F("model_drift", "high"), F("statistical_power", "medium")]
    rows = {r["probe_id"]: r for r in findings_diff(fa, fb)}
    assert rows["judge_swap"]["change"] == "better"
    assert rows["model_drift"]["change"] == "worse"
    assert rows["dataset_hygiene"]["change"] == "a_only"
    assert rows["statistical_power"]["change"] == "b_only"


# ----------------------------- ranking verdict -----------------------------

GAP_REAL = {"d": .2, "lo": .1, "hi": .3, "exceeds_noise": True}
GAP_NOISE = {"d": .02, "lo": -.05, "hi": .09, "exceeds_noise": False}


def test_ranking_uns_when_one_run_invalid():
    out = ranking_verdict([F("statistical_power", "high")], [], GAP_REAL, None)
    assert out["verdict"] == "uns" and "not supportable" in out["reasons"][0]


def test_ranking_uns_when_gap_is_noise():
    out = ranking_verdict([], [], GAP_NOISE, None)
    assert out["verdict"] == "uns" and "noise" in out["reasons"][0]


def test_ranking_uns_without_scores():
    out = ranking_verdict([], [], None, None)
    assert out["verdict"] == "uns"


def test_ranking_fragile_when_gap_real_but_threats_remain():
    out = ranking_verdict([F("option_order_bias", "medium")], [], GAP_REAL, None)
    assert out["verdict"] == "fra"


def test_ranking_supportable_when_clean_and_gap_real():
    out = ranking_verdict([], [F("dataset_hygiene", "low")], GAP_REAL, None)
    assert out["verdict"] == "sup"


def test_paired_supersedes_unpaired_in_verdict():
    paired_noise = {"discordant": 6, "a_wins": 4, "b_wins": 2,
                    "sign_lo": .3, "sign_hi": .9, "exceeds_noise": False}
    out = ranking_verdict([], [], GAP_REAL, paired_noise)
    assert out["verdict"] == "uns" and "paired" in out["reasons"][0]
