"""Smoke tests for the record explorer view + observatory phase banner."""
import trust_the_eval.probes  # noqa: F401
from trust_the_eval.corpus import AnnotationCorpus
from trust_the_eval.intrinsic import audit_corpus
from trust_the_eval.explorer import build_explorer_html
from trust_the_eval.observatory import build_site_html
from trust_the_eval.store import RecordStore


def test_explorer_renders_records(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    html = build_explorer_html(st, phase=("P1", "Model-free intrinsic audit"))
    assert "sha256:" in html and "ToyMMLU" in html
    assert "P1" in html and "intrinsic_audit" in html
    assert "const DATA =" in html and "__PHASE__" not in html


def test_observatory_phase_banner(tmp_path):
    st = RecordStore(tmp_path)
    audit_corpus(AnnotationCorpus.synthetic_seed(), store=st)
    html = build_site_html(st, phase=("P2", "Public surface"))
    assert "phasebar" in html and "P2" in html and "__PHASE__" not in html
