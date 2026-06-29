"""Tests for trust_the_eval.transparency — total per-datum transparency.

The headline guarantee here is the *recompute guarantee*: for every benchmark and
every displayed headline metric, the value re-derived from the named inputs by the
canonical functions must equal exactly what `leaderboard(store)` shows. If the
pipeline were a black box, this test could not exist; because it passes, every
number on the site is reproducible from its inputs.

We also check that the per-item export reproduces the headline accuracy (single
source of truth), that the CSV/JSON are well-formed, and that the datasheet states
the honest annotation limits.
"""
import csv
import io
import json
from pathlib import Path

import pytest

import trust_the_eval.pipeline as PL
from trust_the_eval.leaderboard import leaderboard
from trust_the_eval.result_sensitivity import sensitivity
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore
from trust_the_eval.transparency import (datasheet, explain, per_item_records,
                                         recompute_headline, records_to_csv,
                                         write_transparency)

FIX = Path(__file__).parent / "fixtures" / "mmlu_redux_real_mini.json"
BOOT = 200


class _Src:
    def __init__(self, sid, payload):
        self.id = sid
        self._p = payload
        self.anchors = {"min_rows": 1}

    def fingerprint(self):
        return "fp-" + self.id

    def fetch(self):
        return self._p


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    real = json.loads(FIX.read_text(encoding="utf-8"))
    srcs, source_meta = [], {}
    pred_rows, intrinsic_rows = {}, {}
    for subj, d in real.items():
        bench = "MMLU::%s" % subj
        srcs.append(_Src("int-%s" % subj, {"kind": "intrinsic", "benchmark": bench,
                                           "dataset_version": "mmlu-redux-2.0", "rows": d["intrinsic"]}))
        srcs.append(_Src("pred-%s" % subj, {"kind": "predictions", "benchmark": bench,
                                            "dataset_version": "mmlu-redux-2.0", "rows": d["pred"]}))
        pred_rows[bench] = d["pred"]
        intrinsic_rows[bench] = d["intrinsic"]
        source_meta[bench] = {"sources": ["int-%s" % subj, "pred-%s" % subj],
                              "dataset_version": "mmlu-redux-2.0"}

    orig = PL.build_record

    def fast(*a, **k):
        k.setdefault("iters", BOOT)
        return orig(*a, **k)

    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setattr(PL, "build_record", fast)
    root = tmp_path_factory.mktemp("transp")
    store = RecordStore(root / "s")
    PL.sync(store, srcs, SourceState(root / "s"), root=root / "s",
            calibrations_dataset={}, calibrations_result={})
    mp.undo()
    return store, pred_rows, intrinsic_rows, source_meta, root


def test_recompute_equals_displayed_for_every_benchmark(built):
    """THE recompute guarantee: re-deriving each headline metric from the inputs
    reproduces exactly what the leaderboard displays. Pipeline, not black box."""
    store, pred_rows, intrinsic_rows, _sm, _root = built
    benches = [b["name"] for b in leaderboard(store)["benchmarks"]]
    checked = 0
    for name in benches:
        for metric in ("tau", "alpha", "label", "ambiguity"):
            e = explain(store, name, metric, pred_rows=pred_rows.get(name, []),
                        intrinsic_rows=intrinsic_rows.get(name, []))
            if e["displayed"] is not None:
                assert e["match"], (name, metric, e["displayed"], e["recomputed"])
                checked += 1
    assert checked >= 4  # at least a few real metrics actually compared


def test_export_reproduces_headline_accuracy(built):
    """The per-item export's correctness, averaged, equals the headline per-model
    accuracy from `sensitivity` — proving export and headline share one source."""
    _store, pred_rows, intrinsic_rows, _sm, _root = built
    name = next(iter(pred_rows))
    preds, og, cg = {}, {}, {}
    for r in pred_rows[name]:
        og[r["item"]] = set(r["original_gold"])
        cg[r["item"]] = set(r["corrected_gold"])
        for m, l in r["preds"].items():
            preds.setdefault(m, {})[r["item"]] = l
    res = sensitivity(preds, og, cg, iters=50, seed=0)
    models, recs = per_item_records(name, pred_rows[name], intrinsic_rows.get(name))
    n = len(recs)
    assert n == res["n_items_scored"]
    for m in models:
        from trust_the_eval.observatory_ui import _short
        acc_o = sum(r["models"][m]["correct_orig"] for r in recs) / n
        acc_c = sum(r["models"][m]["correct_corr"] for r in recs) / n
        assert abs(acc_o - res["per_model"][m]["acc_orig"]) < 1e-9, (name, m)
        assert abs(acc_c - res["per_model"][m]["acc_corr"]) < 1e-9, (name, m)


def test_per_item_export_is_complete_and_well_formed(built):
    """Every scored item carries every model's raw answer and correctness under both
    keys; the CSV has one row per item and three columns per model."""
    _store, pred_rows, intrinsic_rows, _sm, _root = built
    name = next(iter(pred_rows))
    models, recs = per_item_records(name, pred_rows[name], intrinsic_rows.get(name))
    for r in recs:
        assert set(r["models"]) == set(models)
        for m in models:
            mb = r["models"][m]
            assert set(mb) == {"pred", "correct_orig", "correct_corr"}
            assert mb["correct_orig"] in (0, 1) and mb["correct_corr"] in (0, 1)
        assert isinstance(r["changed"], bool)
        assert r["original_gold"] == sorted(r["original_gold"])  # canonicalised
    txt = records_to_csv(models, recs)
    rows = list(csv.reader(io.StringIO(txt)))
    assert len(rows) == len(recs) + 1  # header + one row per item
    assert len(rows[0]) == 8 + 3 * len(models)  # 8 base columns + 3 per model


def test_datasheet_states_honest_limits(built):
    """The datasheet carries provenance + the single-pass annotation caveat (no κ)."""
    _store, pred_rows, intrinsic_rows, source_meta, _root = built
    name = next(iter(pred_rows))
    ds = datasheet(name, source_meta=source_meta[name], pred_rows=pred_rows[name],
                   intrinsic_rows=intrinsic_rows.get(name))
    for key in ("benchmark", "dataset_version", "license", "annotation",
                "composition", "measured", "intended_use", "known_limitations"):
        assert key in ds, key
    assert ds["annotation"]["inter_annotator_agreement"] is None
    assert "single-pass" in ds["annotation"]["process"]
    # the measured label-error in the datasheet equals the displayed one
    le = (ds["measured"]["label_error"] or {})
    rc = recompute_headline(name, pred_rows[name], intrinsic_rows.get(name))
    assert le == rc["label_error"]


def test_write_transparency_emits_files_that_match_headline(built):
    """Writing the export produces parseable JSON/CSV/datasheet per benchmark, and the
    JSON's recomputed headline equals the live leaderboard headline."""
    store, pred_rows, intrinsic_rows, source_meta, root = built
    out = root / "site"
    index = write_transparency(out, store, pred_rows, intrinsic_rows, source_meta)
    assert index, "expected at least one benchmark exported"
    LB = {b["name"]: b for b in leaderboard(store)["benchmarks"]}
    for name, urls in index.items():
        for kind in ("json", "csv", "datasheet"):
            assert (out / urls[kind]).exists(), (name, kind)
        payload = json.loads((out / urls["json"]).read_text(encoding="utf-8"))
        assert payload["n_items"] == urls["n_items"]
        assert len(payload["items"]) == urls["n_items"]
        # the export's recomputed headline matches what the dossier displays
        disp = LB[name]
        rec = payload["headline_recomputed"]
        if (disp.get("sensitivity") or {}).get("kendall_tau") is not None:
            assert abs(rec["kendall_tau"] - disp["sensitivity"]["kendall_tau"]) < 1e-9, name
        if (disp.get("reliability") or {}).get("pooled_alpha") is not None:
            assert rec["pooled_alpha"] == disp["reliability"]["pooled_alpha"], name
        # CSV row count matches item count
        csv_rows = list(csv.reader(io.StringIO((out / urls["csv"]).read_text(encoding="utf-8"))))
        assert len(csv_rows) == urls["n_items"] + 1, name


def test_mismatch_would_be_detected():
    """Guard the guard: explain must FLAG a mismatch when the displayed value is wrong,
    so a green `match` actually means something."""
    # a 1-item, 2-model toy where we hand a deliberately wrong 'displayed' store
    class _FakeStore:
        def all_ids(self):
            return []

    # craft a benchmark dict with a wrong tau via a stub leaderboard
    import trust_the_eval.transparency as T
    pred = [{"item": "i1", "subject": "s", "original_gold": ["A"], "corrected_gold": ["A"],
             "preds": {"m1": "A", "m2": "B"}},
            {"item": "i2", "subject": "s", "original_gold": ["B"], "corrected_gold": ["B"],
             "preds": {"m1": "A", "m2": "B"}}]

    def fake_displayed(store, benchmark):
        return {"name": benchmark, "sensitivity": {"kendall_tau": 0.123456}}  # wrong on purpose

    orig = T._displayed
    T._displayed = fake_displayed
    try:
        e = explain(_FakeStore(), "MMLU::toy", "tau", pred_rows=pred, intrinsic_rows=[])
        assert e["displayed"] == 0.123456
        assert e["match"] is False  # recompute != the wrong displayed value
    finally:
        T._displayed = orig
