"""Tests for trust_the_eval.claims — the per-claim feed (P3)."""
import os

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.claims import (
    RESULT_PROBES, calibrate_result_probes, claim_record, claims_from_hf_leaderboard,
    ingest_claims, load_leaderboard_rows, normalize_row, synthetic_claims,
)
from trust_the_eval.observatory import build_site_html
from trust_the_eval.store import RecordStore

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "leaderboard_sample.jsonl")


def test_normalize_row_accepts_aliases_and_0_100():
    c = normalize_row({"model_name": "X", "acc": 86.4, "release": "2023-03"}, benchmark="MMLU")
    assert c["model"] == "X" and abs(c["score"] - 0.864) < 1e-9
    assert c["benchmark"] == "MMLU"


def test_synthetic_claims_become_claim_records():
    recs = ingest_claims(synthetic_claims("MMLU"), benchmark="MMLU")
    assert recs and all(r.subject.kind == "claim" for r in recs)
    r0 = recs[1]  # GPT-4
    assert r0.subject.model and r0.subject.reported_score is not None
    assert r0.verify()


def test_claim_with_scores_audits_the_result():
    rec = claim_record(synthetic_claims("MMLU")[1])  # GPT-4, has per-sample
    pids = {a.probe_id for a in rec.audits}
    assert "statistical_power" in pids
    sp = [a for a in rec.audits if a.probe_id == "statistical_power"][0]
    assert sp.measured and sp.reliability is not None


def test_calibrations_attach_metrics_to_claim_audits():
    cals = calibrate_result_probes(seed=0)
    rec = claim_record(synthetic_claims("MMLU")[2], calibrations=cals)
    audits = [a for a in rec.audits if a.probe_id in RESULT_PROBES]
    assert audits and any(a.reliability and a.reliability.recall is not None for a in audits)


def test_score_only_claim_has_no_fabricated_audits():
    rec = claim_record({"benchmark": "MMLU", "model": "GPT-3.5", "date": "2022-11", "acc": 70.0})
    assert rec.audits == []
    assert "Score-only" in (rec.provenance.note or "")
    assert rec.subject.reported_score is not None and rec.verify()


def test_leaderboard_fixture_loads_and_ingests(tmp_path):
    rows, sources = claims_from_hf_leaderboard(path=FIX, benchmark="MMLU")
    assert len(rows) == 3
    st = RecordStore(tmp_path)
    recs = ingest_claims(rows, store=st, sources=sources)
    assert len(recs) == 3 and len(st.list(kind="claim")) == 3
    hist = st.history("MMLU", kind="claim")
    dates = [r["date"] for r in hist]
    assert dates == sorted(dates)


def test_observatory_trajectory_is_data_driven(tmp_path):
    st = RecordStore(tmp_path)
    ingest_claims(synthetic_claims("MMLU"), store=st)
    html = build_site_html(st)
    assert "GPT-4o" in html and "2024-05" in html
