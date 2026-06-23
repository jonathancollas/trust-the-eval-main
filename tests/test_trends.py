"""Tests for trust_the_eval.trends — longitudinal signals from the store (P5)."""
import os

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.claims import calibrate_result_probes, ingest_claims, synthetic_claims
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import audit_corpus, calibrate_dataset_probes
from trust_the_eval.store import RecordStore
from trust_the_eval.trends import compute_trends

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mmlu_redux_sample.jsonl")


def _store(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.from_mmlu_redux(path=FIX),
                 calibrations=calibrate_dataset_probes(), store=st)
    ingest_claims(synthetic_claims("MMLU"), store=st, calibrations=calibrate_result_probes())
    return st


def test_trends_headroom_and_over_ceiling(tmp_path):
    t = compute_trends(_store(tmp_path))["MMLU"]
    # fixture label-error rate => ceiling 0.70; synthetic claims rise to ~0.90
    assert t["n_claims"] == 4 and t["ceiling"] is not None
    assert t["over_ceiling"] is True and t["headroom"] < 0
    assert t["saturated"] is True and t["saturation_reasons"]


def test_trends_gain_vs_noise_uses_recorded_cis(tmp_path):
    t = compute_trends(_store(tmp_path))["MMLU"]
    assert t["last_gain"] is not None
    assert t["gain_within_noise"] is True            # n=200 CIs dwarf the small gain
    assert "CI half-width" in t["noise_basis"]


def test_trends_handles_no_claims(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)   # intrinsic only
    t = compute_trends(st)["ToyMMLU"]
    assert t["n_claims"] == 0 and t["saturated"] is False
