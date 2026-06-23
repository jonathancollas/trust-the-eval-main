"""Run Trust Report renderer tests: self-contained, provenance-stamped, honest."""
from __future__ import annotations

import json
import re

import trust_the_eval.probes  # noqa: F401
from trust_the_eval import datagen
from trust_the_eval.adapters import generic_json
from trust_the_eval.emit.trust_report import calibration_index, render_trust_report
from trust_the_eval.model import build_model
from trust_the_eval.runner import run_battery


def _artifact(tmp_path, n=30, seed=3):
    p = tmp_path / "art.json"
    p.write_text(json.dumps(datagen.generate(n=n, seed=seed)), encoding="utf-8")
    return generic_json.load(str(p))


def test_trust_report_is_self_contained_and_honest(tmp_path):
    art = _artifact(tmp_path)
    rep = run_battery(art, model=build_model(None))
    html = render_trust_report(rep, art, model_spec="none")

    assert html.lstrip().lower().startswith("<!doctype")
    assert "<script" not in html
    # zero external resources (offline, archivable)
    assert re.findall(r'(?:src|href)="https?://', html) == []
    # provenance: the artifact's own content hash, with honest stamp wording
    assert art.content_hash() in html
    assert "provenance stamp, not a cryptographic signature" in html
    # four per-claim verdict cards, humble framing, remediation present
    assert html.count('class="vb"') == 4
    assert "not undermined by the threats we test" in html
    assert "does not certify the model" in html
    assert "what to do" in html
    # measured reliability wording from calibration
    assert "recall" in html and "specificity" in html and "precision" in html


def test_trust_report_includes_skips_when_no_model():
    # with model=None the model-dependent probes are skipped and listed
    import trust_the_eval.probes  # noqa: F811
    from trust_the_eval.artifact import EvalArtifact, EvalItem
    art = EvalArtifact(dataset="mini", items=[
        EvalItem(question=f"q{i}", answer="a", score=1.0) for i in range(12)])
    rep = run_battery(art, model=build_model(None))
    html = render_trust_report(rep, art, model_spec="none")
    assert "Skipped" in html and "need model access" in html


def test_calibration_index_is_cached():
    idx1 = calibration_index(seed=0)
    idx2 = calibration_index(seed=0)
    assert idx1 is idx2
    assert "statistical_power" in idx1
    e = idx1["statistical_power"]
    assert set(e) >= {"tier", "precision", "recall", "specificity"}
