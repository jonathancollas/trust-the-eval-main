"""Tests for the autonomy layer: sources, watermarks, fidelity gate, sync, observe."""
import os

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.changelog import diff_exports
from trust_the_eval.pipeline import observe, sync
from trust_the_eval.sources.feeds import LocalFileSource, build_sources, load_sources
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore

HERE = os.path.dirname(__file__)
FIX_C = os.path.join(HERE, "fixtures", "mmlu_redux_sample.jsonl")
FIX_L = os.path.join(HERE, "fixtures", "leaderboard_sample.jsonl")


def _sources(le_band=None):
    intr_anchors = {"min_rows": 5, "require_fields": ["error_type"]}
    if le_band:
        intr_anchors["label_error_between"] = list(le_band)
    return [
        LocalFileSource(path=FIX_C, id="vir", kind="intrinsic",
                        benchmark="MMLU-Redux (Virology)", dataset_version="virology/test",
                        anchors=intr_anchors),
        LocalFileSource(path=FIX_L, id="lb", kind="claim", benchmark="MMLU",
                        anchors={"min_rows": 1, "score_between": [0, 1], "dates_parseable": True}),
    ]


def test_build_sources_from_spec():
    srcs = build_sources([
        {"type": "local", "id": "x", "kind": "claim", "benchmark": "MMLU", "path": FIX_L}])
    assert len(srcs) == 1 and srcs[0].kind == "claim" and srcs[0].fingerprint().startswith("sha256:")


def test_sync_ingests_then_skips(tmp_path):
    store = RecordStore(tmp_path / "s")
    st = SourceState(tmp_path / "s")
    o1 = sync(store, _sources(), st, root=str(tmp_path / "s"))
    assert all(o["status"] == "ingested" for o in o1) and len(store.all_ids()) > 0
    n = len(store.all_ids())
    o2 = sync(store, _sources(), st, root=str(tmp_path / "s"))   # nothing changed
    assert all(o["status"] == "skipped" for o in o2)
    assert len(store.all_ids()) == n                              # idempotent


def test_fidelity_gate_quarantines(tmp_path):
    store = RecordStore(tmp_path / "s")
    st = SourceState(tmp_path / "s")
    out = sync(store, _sources(le_band=(0.99, 1.0)), st, root=str(tmp_path / "s"))
    by = {o["source"]: o for o in out}
    assert by["vir"]["status"] == "quarantined"     # parsed label-error not in [0.99,1]
    assert by["lb"]["status"] == "ingested"
    assert os.path.isdir(str(tmp_path / "s" / "quarantine"))


def test_changelog_detects_changes():
    prev = {"benchmarks": [{"name": "MMLU", "status": "drifting",
                            "trajectory": [{"model": "A", "date": "2023", "score": 0.8}],
                            "trends": {"saturated": False}}]}
    curr = {"benchmarks": [{"name": "MMLU", "status": "degraded",
                            "trajectory": [{"model": "A", "date": "2023", "score": 0.8},
                                           {"model": "B", "date": "2024", "score": 0.9}],
                            "trends": {"saturated": True}}]}
    types = {c["type"] for c in diff_exports(prev, curr)}
    assert {"status_change", "new_claim", "saturation_change"} <= types


def test_observe_one_shot_then_no_change(tmp_path):
    rep = observe(str(tmp_path / "s"), _sources(), str(tmp_path / "out"), always=True)
    assert rep["changed"] and rep["integrity"]["verified"]
    for f in ("index.html", "meridian.json", "feed.json", "records.json"):
        assert os.path.exists(str(tmp_path / "out" / f))
    rep2 = observe(str(tmp_path / "s"), _sources(), str(tmp_path / "out"))
    assert rep2["changed"] is False                  # watermarks -> no rebuild
