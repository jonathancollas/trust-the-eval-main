"""The observatory, fed live by evals.

`observe()` ingests eval sources (model predictions + dataset annotations), computes
the validity audit incrementally, and renders the rich navigable observatory as
index.html. Re-running with unchanged evals leaves the site untouched (live but
incremental); feeding new/updated evals recomputes and rebuilds.
"""
import json
from pathlib import Path

import trust_the_eval.pipeline as PL
from trust_the_eval.pipeline import observe
from trust_the_eval.sources.feeds import LocalFileSource

FIX = Path(__file__).parent / "fixtures" / "mmlu_redux_real_mini.json"


def test_feeds_accepts_predictions_eval(tmp_path):
    p = tmp_path / "pred.json"
    p.write_text(json.dumps([{"item": "i1", "subject": "s", "original_gold": ["A"],
                              "corrected_gold": ["A"], "preds": {"prov_m": "A"}}]), encoding="utf-8")
    s = LocalFileSource(path=str(p), id="pred-s", kind="predictions",
                        benchmark="MMLU::s", dataset_version="v0", anchors={"min_rows": 1})
    pay = s.fetch()
    assert pay["kind"] == "predictions" and pay["rows"]


def _eval_sources(real, tmp_path):
    srcs = []
    for subj, d in real.items():
        ip = tmp_path / ("int_%s.json" % subj)
        pp = tmp_path / ("pred_%s.json" % subj)
        ip.write_text(json.dumps(d["intrinsic"]), encoding="utf-8")
        pp.write_text(json.dumps(d["pred"]), encoding="utf-8")
        srcs.append(LocalFileSource(path=str(ip), id="int-%s" % subj, kind="intrinsic",
                                    benchmark="MMLU::%s" % subj, dataset_version="mmlu-redux-2.0",
                                    anchors={"min_rows": 1}))
        srcs.append(LocalFileSource(path=str(pp), id="pred-%s" % subj, kind="predictions",
                                    benchmark="MMLU::%s" % subj, dataset_version="mmlu-redux-2.0",
                                    anchors={"min_rows": 1}))
    return srcs


def test_observe_builds_rich_observatory_from_evals(tmp_path, monkeypatch):
    real = json.loads(FIX.read_text(encoding="utf-8"))
    srcs = _eval_sources(real, tmp_path)
    orig = PL.build_record
    monkeypatch.setattr(PL, "build_record",
                        lambda *a, **k: (k.setdefault("iters", 150), orig(*a, **k))[1])
    out = tmp_path / "site"

    rep = observe(tmp_path / "store", srcs, out, always=True)
    assert rep["changed"] is True

    idx = out / "index.html"
    assert idx.exists()
    html = idx.read_text(encoding="utf-8")
    # the live site is the rich observatory built from the evals
    assert html.startswith("<!DOCTYPE html>")
    assert "Probe library" in html
    assert "MMLU::virology" in html          # a dossier for the fed eval
    assert "show calculation" in html        # per-number computation panels
    # the lightweight SPA is preserved alongside
    assert (out / "classic.html").exists()
    # data artifact emitted too
    assert (out / "observatory-data.json").exists()


def test_observe_is_incremental_when_evals_unchanged(tmp_path, monkeypatch):
    real = json.loads(FIX.read_text(encoding="utf-8"))
    srcs = _eval_sources(real, tmp_path)
    orig = PL.build_record
    monkeypatch.setattr(PL, "build_record",
                        lambda *a, **k: (k.setdefault("iters", 150), orig(*a, **k))[1])
    out = tmp_path / "site"
    observe(tmp_path / "store", srcs, out, always=True)
    again = observe(tmp_path / "store", srcs, out, always=False)
    assert again["changed"] is False         # unchanged evals -> site untouched


def test_manifest_can_declare_prediction_evals(tmp_path):
    """The CLI feeds evals via a manifest (sources/feeds.build_sources); prediction
    evals must be declarable there too."""
    from trust_the_eval.sources.feeds import build_sources, load_sources
    p = tmp_path / "pred.json"
    p.write_text(json.dumps([{"item": "i", "subject": "s", "original_gold": ["A"],
                              "corrected_gold": ["A"], "preds": {"prov_m": "A"}}]), encoding="utf-8")
    spec = [{"type": "local", "kind": "predictions", "path": str(p), "id": "pred-s",
             "benchmark": "MMLU::s", "dataset_version": "v0", "anchors": {"min_rows": 1}}]
    srcs = build_sources(spec)
    assert len(srcs) == 1 and srcs[0].fetch()["kind"] == "predictions"
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(spec), encoding="utf-8")
    assert load_sources(str(man))[0].kind == "predictions"
