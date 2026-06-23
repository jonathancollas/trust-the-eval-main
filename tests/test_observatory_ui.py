"""Tests for trust_the_eval.observatory_ui — the navigable validity observatory
generated from a synced store. Driven by the committed real-data mini fixture.

The headline guarantee is the *faithful-trace invariant*: every figure the UI shows
must be reconstructable from its trace. We check it directly here, for every
benchmark, by recomputing Kendall tau from the trace's concordant/discordant pair
counts and requiring it to equal the headline tau exactly.
"""
import json
from pathlib import Path

import pytest

import trust_the_eval.pipeline as PL
from trust_the_eval.leaderboard import leaderboard
from trust_the_eval.observatory_ui import (assemble_ui_data, build_ui_html,
                                           build_ui_site, _auto_spotlight, SCI)
from trust_the_eval.remediation import PROBE_IDS
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore

FIX = Path(__file__).parent / "fixtures" / "mmlu_redux_real_mini.json"
BOOT = 200


class _Src:
    def __init__(self, sid, payload, anchors=None):
        self.id = sid
        self._p = payload
        self.anchors = anchors or {"min_rows": 1}

    def fingerprint(self):
        return "fp-" + self.id

    def fetch(self):
        return self._p


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    real = json.loads(FIX.read_text(encoding="utf-8"))
    srcs = []
    for subj, d in real.items():
        srcs.append(_Src("int-%s" % subj, {"kind": "intrinsic", "benchmark": "MMLU::%s" % subj,
                                            "dataset_version": "mmlu-redux-2.0", "rows": d["intrinsic"]}))
        srcs.append(_Src("pred-%s" % subj, {"kind": "predictions", "benchmark": "MMLU::%s" % subj,
                                             "dataset_version": "mmlu-redux-2.0", "rows": d["pred"]}))
    orig = PL.build_record

    def fast(*a, **k):
        k.setdefault("iters", BOOT)
        return orig(*a, **k)

    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setattr(PL, "build_record", fast)
    root = tmp_path_factory.mktemp("obsui")
    store = RecordStore(root / "s")
    PL.sync(store, srcs, SourceState(root / "s"), root=root / "s",
            calibrations_dataset={}, calibrations_result={})
    mp.undo()

    pred_rows = {"MMLU::%s" % s: d["pred"] for s, d in real.items()}
    intrinsic_rows = {"MMLU::%s" % s: d["intrinsic"] for s, d in real.items()}
    models = list(next(iter(real.values()))["pred"][0]["preds"].keys())
    spotlight = ["MMLU::%s" % s for s in real]
    data = assemble_ui_data(store, pred_rows, intrinsic_rows, spotlight, models)
    return store, data


def test_portfolio_and_spotlight_shapes(assembled):
    _store, data = assembled
    assert data["n_benchmarks"] == 3
    assert len(data["portfolio"]) == 3
    assert len(data["spotlight"]) == 3
    assert set(data["models"]) >= {"Claude-3-Opus", "GPT-4o"}


def test_faithful_trace_invariant_tau(assembled):
    """(C - D) / npair must equal the headline Kendall tau, for every benchmark."""
    _store, data = assembled
    checked = 0
    for row in data["portfolio"]:
        t = row["trace"]
        tau = row["sens"]["tau"]
        if not t or tau is None or t["tau"]["npair"] == 0:
            continue
        C, D, npair = t["tau"]["C"], t["tau"]["D"], t["tau"]["npair"]
        assert C + D == npair, (row["name"], C, D, npair)
        assert abs((C - D) / npair - tau) < 1e-9, (row["name"], (C - D) / npair, tau)
        checked += 1
    assert checked >= 2  # at least the subjects that actually have a ranking


def test_trace_reconstructs_alpha_and_label(assembled):
    _store, data = assembled
    for row in data["portfolio"]:
        t = row["trace"]
        if not t:
            continue
        if row["alpha"] is not None and t["cronbach"]["alpha"] is not None:
            assert abs(t["cronbach"]["alpha"] - row["alpha"]) < 1e-2, row["name"]
        if row["label_error"]["k"] is not None:
            assert t["label"]["defects"] == row["label_error"]["k"], row["name"]


def test_virology_is_rank_fragile_with_real_numbers(assembled):
    _store, data = assembled
    vir = next(p for p in data["portfolio"] if p["name"] == "MMLU::virology")
    assert vir["label_error"]["k"] == 41 and vir["label_error"]["n"] == 100
    assert vir["sens"]["stable"] is False           # correcting labels reshuffles the ranking
    assert vir["alpha"] < 0                          # items anti-cohere under the wrong key
    sp = data["spotlight"]["MMLU::virology"]
    assert sp["top1_changed"] is True               # the #1 model changes under correction


def test_science_covers_every_probe(assembled):
    # Every remediation probe id has an authored science block (explained + demonstrated).
    missing = set(PROBE_IDS) - set(SCI)
    assert not missing, missing
    for pid, s in SCI.items():
        for field in ("title", "measures", "formula", "method", "demo", "limits"):
            assert s.get(field), (pid, field)


def test_html_renders_and_states_the_bright_line(assembled):
    _store, data = assembled
    html = build_ui_html(data)
    assert html.startswith("<!DOCTYPE html>") and "</html>" in html
    # navigable surfaces present
    assert "Probe library" in html and "Methodology" in html
    # the transparency invariant is stated, and the per-number computation affordance exists
    assert "No number without its computation" in html
    assert "show calculation" in html
    # the bright line: audit the instrument, never rate the model
    assert "Audit the instrument" in html
    assert "never rate a model" in html or "never rate" in html
    # data + science are embedded (data-driven, not hard-coded)
    assert "MMLU::virology" in html
    assert "Result-sensitivity" in html


def test_no_model_is_scored_as_an_endpoint(assembled):
    """The instrument profiles benchmarks, not models: no portfolio row exposes a
    model quality/score/ranking field as a verdict about the model."""
    _store, data = assembled
    banned = {"model_score", "best_model", "winner", "model_rank", "trust_score"}
    for row in data["portfolio"]:
        assert not (banned & set(row)), row["name"]


def _sync_fixture(tmp_path):
    """Sync the mini fixture and return (store, sources) — the source objects, so the
    store+sources observatory entry point can be exercised."""
    real = json.loads(FIX.read_text(encoding="utf-8"))
    srcs = []
    for subj, d in real.items():
        srcs.append(_Src("int-%s" % subj, {"kind": "intrinsic", "benchmark": "MMLU::%s" % subj,
                                            "dataset_version": "mmlu-redux-2.0", "rows": d["intrinsic"]}))
        srcs.append(_Src("pred-%s" % subj, {"kind": "predictions", "benchmark": "MMLU::%s" % subj,
                                             "dataset_version": "mmlu-redux-2.0", "rows": d["pred"]}))
    orig = PL.build_record

    def fast(*a, **k):
        k.setdefault("iters", BOOT)
        return orig(*a, **k)

    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setattr(PL, "build_record", fast)
    store = RecordStore(tmp_path / "s")
    PL.sync(store, srcs, SourceState(tmp_path / "s"), root=tmp_path / "s",
            calibrations_dataset={}, calibrations_result={})
    mp.undo()
    return store, srcs


def test_build_ui_site_from_store_and_sources(tmp_path):
    """The observatory emits from a synced store + the sources that built it, with no
    extra row plumbing and no persisted-trace requirement (no content-hash churn)."""
    store, srcs = _sync_fixture(tmp_path)
    res = build_ui_site(store, srcs, tmp_path / "site", title="Meridian")
    index = Path(res["index"])
    assert index.exists() and index.stat().st_size > 10_000
    html = index.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and "</html>" in html
    # auto-spotlight is data-driven and picks the rank-fragile benchmark for contrast
    assert "MMLU::virology" in html
    assert Path(res["data"]).exists()


def test_auto_spotlight_prefers_rank_fragile(tmp_path):
    from trust_the_eval.leaderboard import leaderboard
    store, srcs = _sync_fixture(tmp_path)
    has_pred = {"MMLU::%s" % s for s in json.loads(FIX.read_text(encoding="utf-8"))}
    pick = _auto_spotlight(leaderboard(store), has_pred)
    assert "MMLU::virology" in pick  # the fragile one is surfaced
    assert len(pick) <= 4
