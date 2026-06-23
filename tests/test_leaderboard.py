"""Tests for the validity leaderboard (src/trust_the_eval/leaderboard.py)."""
from trust_the_eval.api import export
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import audit_corpus
from trust_the_eval.leaderboard import NEAR_BAND, classify, leaderboard
from trust_the_eval.record import ProbeAudit, ValidityRecord
from trust_the_eval.store import RecordStore


# ---- classify(): the verdict logic, pure and exhaustive ----

def test_classify_over_ceiling():
    v = classify(0.95, 0.60, verifiable=False)
    assert v["key"] == "over_ceiling"
    # arithmetic verdict holds even for a score-only claim
    v2 = classify(0.601, 0.60, verifiable=True)
    assert v2["key"] == "over_ceiling"


def test_classify_at_ceiling_band():
    # exactly NEAR_BAND below the ceiling is still "at ceiling"
    v = classify(0.60 - NEAR_BAND, 0.60, verifiable=True)
    assert v["key"] == "near_ceiling"
    # at the ceiling itself
    assert classify(0.60, 0.60, verifiable=False)["key"] == "near_ceiling"


def test_classify_unverifiable_vs_ok_below_band():
    below = 0.60 - NEAR_BAND - 0.05
    assert classify(below, 0.60, verifiable=False)["key"] == "unverifiable"
    assert classify(below, 0.60, verifiable=True)["key"] == "ok"


def test_classify_no_ceiling_is_unverifiable():
    assert classify(0.80, None, verifiable=True)["key"] == "unverifiable"
    assert classify(None, 0.60, verifiable=True)["key"] == "unverifiable"


# ---- integration over a real store ----

def _intrinsic(benchmark, ceiling, dv="v1"):
    rate = round(1.0 - ceiling, 4)
    return ValidityRecord.literature(
        benchmark=benchmark, kind="intrinsic", dataset_version=dv,
        audits=[ProbeAudit("label_error_audit", "high",
                            "Annotation error rate sets a hard ceiling.",
                            measured={"effective_accuracy_ceiling": ceiling,
                                      "rate": rate, "k": int(rate * 100),
                                      "n_annotated": 100,
                                      "derived": "human_annotation_single_pass"})])


def _claim(benchmark, model, score, date, verifiable=False):
    audits = ([ProbeAudit("statistical_power", "low", "n is adequate.",
                          measured={"n": 1000})] if verifiable else None)
    return ValidityRecord.literature(
        benchmark=benchmark, kind="claim", model=model, reported_score=score,
        metric="accuracy", date=date, audits=audits)


def _store(tmp_path):
    st = RecordStore(tmp_path / "lb")
    st.put(_intrinsic("BENCH", 0.60))
    st.put(_claim("BENCH", "Over", 0.95, "2024-05"))               # over_ceiling
    st.put(_claim("BENCH", "AtCeil", 0.60 - NEAR_BAND, "2024-04"))  # near_ceiling
    st.put(_claim("BENCH", "ScoreOnly", 0.40, "2024-03"))           # unverifiable
    st.put(_claim("BENCH", "Audited", 0.40, "2024-06", verifiable=True))  # ok
    return st


def test_leaderboard_verdicts_and_order(tmp_path):
    lb = leaderboard(_store(tmp_path))
    res = lb["results"]
    assert len(res) == 4
    by_model = {r["model"]: r for r in res}
    assert by_model["Over"]["verdict"] == "over_ceiling"
    assert by_model["AtCeil"]["verdict"] == "near_ceiling"
    assert by_model["ScoreOnly"]["verdict"] == "unverifiable"
    assert by_model["Audited"]["verdict"] == "ok"
    # most concerning first: over_ceiling leads, ok trails
    assert res[0]["model"] == "Over"
    assert res[-1]["model"] == "Audited"
    # headroom sign: over is negative, ok positive
    assert by_model["Over"]["headroom"] < 0 < by_model["Audited"]["headroom"]
    # tally is consistent
    assert sum(lb["tally"].values()) == lb["n_results"] == 4
    assert lb["tally"]["over_ceiling"] == 1


def test_score_only_marked_unverifiable_with_no_instruments(tmp_path):
    lb = leaderboard(_store(tmp_path))
    so = next(r for r in lb["results"] if r["model"] == "ScoreOnly")
    assert so["verifiable"] is False
    assert so["instruments"] == []
    aud = next(r for r in lb["results"] if r["model"] == "Audited")
    assert aud["verifiable"] is True and aud["instruments"]


def test_benchmarks_profile_ordered_by_label_error(tmp_path):
    st = _store(tmp_path)
    st.put(_intrinsic("CLEAN", 0.95, dv="v2"))  # lower label-error
    benches = leaderboard(st)["benchmarks"]
    names = [b["name"] for b in benches]
    assert names == ["BENCH", "CLEAN"]  # 40% label-error before 5%
    assert "trust" not in benches[0]  # the trust scalar is gone
    assert benches[0]["label_error"]["k"] == 40 and benches[0]["label_error"]["n"] == 100
    assert benches[0]["sensitivity"] is None  # no result_sensitivity audit present


def test_leaderboard_surfaces_result_sensitivity_audit(tmp_path):
    measured = {"derived": "label_correction_sensitivity", "n_models": 10,
                "n_changed_items": 109, "kendall_tau": 1.0, "ranking_stable": True,
                "delta_min_pts": 0.69, "delta_max_pts": 0.88, "skill_discrimination": 0.49}
    rec = ValidityRecord.literature(
        benchmark="MMLU", kind="intrinsic", dataset_version="v1",
        audits=[ProbeAudit("label_error_audit", "medium", "lit.",
                           measured={"effective_accuracy_ceiling": 0.935, "error_rate": 0.065}),
                ProbeAudit("result_sensitivity", "low", "rank-stable",
                           measured=measured)])
    st = RecordStore(tmp_path / "s")
    st.put(rec)
    b = leaderboard(st)["benchmarks"][0]
    assert b["sensitivity"]["kendall_tau"] == 1.0
    assert b["sensitivity"]["ranking_stable"] is True
    assert b["sensitivity"]["n_changed_items"] == 109


def test_api_export_includes_leaderboard(tmp_path):
    st = RecordStore(tmp_path / "s")
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    doc = export(st)
    assert "leaderboard" in doc
    assert set(doc["leaderboard"]) >= {"results", "benchmarks", "tally", "n_results"}
