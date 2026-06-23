"""Tests for trust_the_eval.intrinsic — the model-free intrinsic audit job (P1)."""
import os

import trust_the_eval.probes  # noqa: F401  (register probes)
from trust_the_eval.calibration.realworld import DEFECT_ERROR_TYPES
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import (
    DATASET_PROBES, audit_corpus, calibrate_dataset_probes,
)
from trust_the_eval.store import RecordStore

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mmlu_redux_sample.jsonl")


def test_audit_corpus_emits_intrinsic_record():
    c = AnnotationCorpus.synthetic_seed()
    rec = audit_corpus(c)
    assert rec.subject.kind == "intrinsic" and rec.subject.benchmark == "ToyMMLU"
    assert rec.subject.dataset_version == "synthetic-seed-1"
    assert rec.provenance.method == "intrinsic_audit"
    assert rec.provenance.inputs_hash == c.content_hash()
    assert rec.verify()
    pids = {a.probe_id for a in rec.audits}
    assert "label_error_audit" in pids and "item_ambiguity" in pids
    assert "coverage_distribution" in pids or "dataset_hygiene" in pids


def test_label_error_audit_is_annotation_derived_with_ceiling():
    rec = audit_corpus(AnnotationCorpus.synthetic_seed())
    le = [a for a in rec.audits if a.probe_id == "label_error_audit"][0]
    assert le.measured["derived"] == "human_annotation_single_pass"
    assert le.measured["inter_annotator_agreement"] is None
    assert le.measured["rate"] > 0
    assert 0 < le.measured["effective_accuracy_ceiling"] < 1
    assert le.reliability.tier == "real_labeled"


def test_calibrations_attach_real_metrics_to_dataset_probes():
    cals = calibrate_dataset_probes(seed=0)
    rec = audit_corpus(AnnotationCorpus.synthetic_seed(), calibrations=cals)
    probe_audits = [a for a in rec.audits if a.probe_id in DATASET_PROBES]
    assert probe_audits
    assert any(a.reliability and a.reliability.recall is not None for a in probe_audits)


def test_store_integration_and_integrity(tmp_path):
    st = RecordStore(tmp_path)
    rec = audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    assert st.has(rec.record_id)
    assert len(st.list(kind="intrinsic")) == 1
    assert st.verify_all() == []


def test_fixture_corpus_label_rate_matches_annotations():
    c = AnnotationCorpus.from_mmlu_redux(path=FIX)
    rec = audit_corpus(c)
    le = [a for a in rec.audits if a.probe_id == "label_error_audit"][0]
    exp = sum(1 for r in c.rows if r["error_type"] in DEFECT_ERROR_TYPES) / len(c.rows)
    assert abs(le.measured["rate"] - exp) < 1e-9
