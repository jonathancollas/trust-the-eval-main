"""Tests for trust_the_eval.integrity — corpus digest + end-to-end verify (P5)."""
import trust_the_eval.probes  # noqa: F401
from trust_the_eval import integrity
from trust_the_eval.admission import ProbeRegistry, evaluate_probe
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import audit_corpus
from trust_the_eval.record import ValidityRecord
from trust_the_eval.store import RecordStore


def test_manifest_digest_is_stable_and_changes_on_add(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    d1 = integrity.manifest(st)["corpus_digest"]
    assert d1 == integrity.manifest(st)["corpus_digest"]      # stable
    st.put(ValidityRecord.literature(benchmark="GPQA", kind="claim", model="m",
                                     date="2024", reported_score=0.5, metric="accuracy"))
    assert integrity.manifest(st)["corpus_digest"] != d1      # changed


def test_verify_reports_clean_store(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    rep = integrity.verify(st)
    assert rep["verified"] is True and rep["records_verified"] is True
    assert rep["n_records"] >= 1 and "intrinsic" in rep["by_kind"]
    assert rep["corpus_digest"].startswith("sha256:")


def test_verify_includes_registry(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    reg = ProbeRegistry(tmp_path / "reg")
    reg.admit("statistical_power")
    rep = integrity.verify(st, reg)
    assert rep["registry_verified"] is True and rep["admitted_probes"] == 1
    assert rep["verified"] is True
