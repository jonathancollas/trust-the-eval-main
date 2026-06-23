"""Tests for trust_the_eval.api — machine-readable export + read helpers (P5)."""
import json
import os

import trust_the_eval.probes  # noqa: F401
from trust_the_eval import api
from trust_the_eval.admission import ProbeRegistry
from trust_the_eval.claims import ingest_claims, synthetic_claims
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import audit_corpus
from trust_the_eval.store import RecordStore

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mmlu_redux_sample.jsonl")


def _store(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.from_mmlu_redux(path=FIX), store=st)
    ingest_claims(synthetic_claims("MMLU"), store=st)
    return st


def test_export_structure(tmp_path):
    doc = api.export(_store(tmp_path))
    assert doc["schema"] == api.SCHEMA
    assert "integrity" in doc and doc["integrity"]["verified"] is True
    b = [x for x in doc["benchmarks"] if x["name"] == "MMLU"][0]
    assert b["trajectory"] and b["trends"]["n_claims"] == 4
    assert b["intrinsic"]["method"] == "intrinsic_audit"


def test_export_includes_admitted_when_registry_given(tmp_path):
    st = _store(tmp_path)
    reg = ProbeRegistry(tmp_path / "reg")
    reg.admit("statistical_power")
    doc = api.export(st, reg)
    assert doc["method"]["admitted"] and doc["method"]["admitted"][0]["probe_id"] == "statistical_power"
    assert doc["method"]["admitted"][0]["recall"] is not None


def test_write_api_and_read_helpers(tmp_path):
    st = _store(tmp_path)
    out = api.write_api(st, tmp_path / "site")
    doc = json.loads(open(out["api"], encoding="utf-8").read())
    assert doc["schema"] == api.SCHEMA
    assert "MMLU" in api.benchmark_names(st)
    assert api.trajectory(st, "MMLU") and api.trends_for(st, "MMLU")["n_claims"] == 4
