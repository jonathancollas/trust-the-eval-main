"""Tests for trust_the_eval.observatory — static site generated from the store (P2)."""
import json
import os

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import audit_corpus
from trust_the_eval.observatory import build_site, build_site_html
from trust_the_eval.record import ProbeAudit, ReliabilitySnapshot, ValidityRecord
from trust_the_eval.store import RecordStore

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mmlu_redux_sample.jsonl")


def _store(tmp_path):
    st = RecordStore(tmp_path / "store")
    # a GENERATED intrinsic record (real P1 output) for MMLU from the fixture
    audit_corpus(AnnotationCorpus.from_mmlu_redux(path=FIX), store=st)
    # two dated per-claim records to form a trajectory
    for m, d, s in [("GPT-4", "2023-03", 0.864), ("GPT-4.1", "2025-02", 0.902)]:
        st.put(ValidityRecord.literature(
            benchmark="MMLU", kind="claim", model=m, date=d, reported_score=s, metric="accuracy",
            audits=[ProbeAudit(probe_id="discrimination_saturation", severity="high",
                               summary="reported accuracy = %.3f" % s, score=s,
                               reliability=ReliabilitySnapshot.from_evidence("discrimination_saturation"))]))
    return st


def test_build_site_writes_index_and_data(tmp_path):
    st = _store(tmp_path)
    out = build_site(st, tmp_path / "site", title="Meridian")
    assert os.path.exists(out["index"]) and os.path.exists(out["data"])
    html = open(out["index"], encoding="utf-8").read()
    assert "MMLU" in html
    assert "intrinsic_audit" in html          # provenance method surfaced
    assert "sha256:" in html                  # record id / permalink embedded
    assert "Calibration is the bar" in html   # method page content present
    recs = json.loads(open(out["data"], encoding="utf-8").read())
    assert isinstance(recs, list) and len(recs) == len(st.all_ids())


def test_site_html_is_data_driven(tmp_path):
    st = _store(tmp_path)
    html = build_site_html(st, title="Meridian")
    assert "const SITE =" in html
    assert '"benchmark": "MMLU"' in html or '"benchmark":"MMLU"' in html


def test_status_reflects_severity(tmp_path):
    st = _store(tmp_path)
    html = build_site_html(st)
    # the high-severity saturation claim should make MMLU read as degraded
    assert "degraded" in html
