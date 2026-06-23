"""End-to-end test on a small slice of REAL MMLU-Redux + HELM data.

A committed fixture (`fixtures/mmlu_redux_real_mini.json`, 3 subjects with their
genuine annotations and 10-model HELM predictions) is driven through the full
pipeline. Unlike the synthetic unit tests, this exercises real-world messiness —
notably error_type values in mixed casing ('Wrong Groundtruth', 'Wrong
groundtruth', 'Bad Question Clarity') — and pins real numbers.
"""
import json
from pathlib import Path

import pytest

import trust_the_eval.pipeline as PL
from trust_the_eval.calibration.realworld import (DEFECT_ERROR_TYPES,
                                                  canonical_error_type)
from trust_the_eval.item_analysis import correctness_from_predictions
from trust_the_eval.leaderboard import leaderboard
from trust_the_eval.result_sensitivity import audit_measured, sensitivity
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore

FIX = Path(__file__).parent / "fixtures" / "mmlu_redux_real_mini.json"
BOOT = 300  # small bootstrap; recomputation below uses the same value


@pytest.fixture(scope="module")
def real():
    return json.loads(FIX.read_text(encoding="utf-8"))


class _Src:
    def __init__(self, sid, payload, anchors=None):
        self.id = sid
        self._p = payload
        self.anchors = anchors or {"min_rows": 1}

    def fingerprint(self):
        return "fp-" + self.id

    def fetch(self):
        return self._p


def _sources(real):
    srcs = []
    for subj, d in real.items():
        srcs.append(_Src("int-%s" % subj, {
            "kind": "intrinsic", "benchmark": "MMLU::%s" % subj,
            "dataset_version": "mmlu-redux-2.0", "rows": d["intrinsic"]}))
        srcs.append(_Src("pred-%s" % subj, {
            "kind": "predictions", "benchmark": "MMLU::%s" % subj,
            "dataset_version": "mmlu-redux-2.0", "rows": d["pred"]}))
    return srcs


@pytest.fixture(scope="module")
def built(real, tmp_path_factory, monkeypatch_module):
    # shrink bootstrap for speed; recomputation uses the same BOOT
    orig = PL.build_record

    def fast(*a, **k):
        k.setdefault("iters", BOOT)
        return orig(*a, **k)
    monkeypatch_module.setattr(PL, "build_record", fast)

    root = tmp_path_factory.mktemp("e2e_real")
    store = RecordStore(root / "s")
    state = SourceState(root / "s")
    out = PL.sync(store, _sources(real), state, root=root / "s",
                  calibrations_dataset={}, calibrations_result={})
    return store, out


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_all_real_sources_ingested(built):
    _store, out = built
    statuses = {o["source"]: o["status"] for o in out}
    assert all(v == "ingested" for v in statuses.values()), statuses
    assert len(statuses) == 6  # 3 intrinsic + 3 predictions


def test_store_integrity(built):
    store, _ = built
    assert store.verify_all() == []
    for rid in store.all_ids():
        assert store.get_record(rid).verify()


def test_label_error_reproduced_despite_mixed_casing(built, real):
    """The headline of the error_type fix: real files use 'Wrong Groundtruth',
    'Wrong groundtruth', 'Bad Question Clarity' — canonicalisation must yield
    the right label-error rate end to end."""
    store, _ = built
    bench = {b["name"]: b for b in leaderboard(store)["benchmarks"]}
    expected = {"MMLU::virology": (41, 100), "MMLU::college_chemistry": (23, 100),
                "MMLU::high_school_geography": (0, 100)}
    for name, (k, n) in expected.items():
        b = bench[name]
        assert b["label_error"]["k"] == k, (name, b["label_error"])
        assert abs(b["label_error"]["rate"] - k / n) < 1e-9
        # independent recount from the RAW rows via canonicalisation
        rows = real[name.split("::")[1]]["intrinsic"]
        recount = sum(1 for r in rows
                      if canonical_error_type(r["error_type"]) in DEFECT_ERROR_TYPES)
        assert recount == k


def test_clean_subject_has_zero_label_error(built):
    store, _ = built
    b = {x["name"]: x for x in leaderboard(store)["benchmarks"]}["MMLU::high_school_geography"]
    assert b["label_error"]["rate"] == 0.0
    assert b["status_word"] == "valid"
    # no corrections to apply -> sensitivity reports no change
    assert b["sensitivity"]["n_changed_items"] == 0


def test_sensitivity_and_reliability_reproduce(built, real):
    store, _ = built
    bench = {b["name"]: b for b in leaderboard(store)["benchmarks"]}
    for subj in real:
        name = "MMLU::%s" % subj
        preds, og, cg = {}, {}, {}
        for r in real[subj]["pred"]:
            og[r["item"]] = set(r["original_gold"])
            cg[r["item"]] = set(r["corrected_gold"])
            for m, l in r["preds"].items():
                preds.setdefault(m, {})[r["item"]] = l
        res = sensitivity(preds, og, cg, iters=BOOT, seed=0)
        want = audit_measured(res)
        got = bench[name]["sensitivity"]
        assert got["n_changed_items"] == want["n_changed_items"]
        assert (got["kendall_tau"] is None) == (want["kendall_tau"] is None)
        if got["kendall_tau"] is not None:
            assert abs(got["kendall_tau"] - want["kendall_tau"]) < 1e-9
        assert abs(got["delta_min_pts"] - want["delta_min_pts"]) < 1e-6
        # reliability present and consistent
        assert bench[name]["reliability"] is not None


def test_no_trust_score_anywhere(built):
    store, _ = built
    for b in leaderboard(store)["benchmarks"]:
        assert "trust" not in b


def test_remediation_relays_label_errors_never_asserts(built):
    """Remediation is exposed per benchmark, label-errors are relay-only with the
    right count, and no action ever asserts a ground-truth correction."""
    store, _ = built
    bench = {b["name"]: b for b in leaderboard(store)["benchmarks"]}
    vir = bench["MMLU::virology"]
    plan = vir["remediation"]
    assert plan is not None and plan["n_actions"] >= 1
    # sensitivity was measured -> verdict impact is known
    assert plan["verdict_impact"]["known"] is True
    le = [a for a in plan["actions"] if a["probe_id"] == "label_error_audit"]
    assert len(le) == 1 and le[0]["tier"] == "relay_only"
    assert le[0]["n_items_affected"] == 41
    assert le[0]["candidates_available"] is True and "verify" in le[0]["action"].lower()
    # invariant across every benchmark: never assert a correction
    for b in bench.values():
        for a in (b["remediation"] or {}).get("actions", []):
            assert a["asserts_correction"] is False
            assert "the answer is" not in a["action"].lower()


def test_clean_subject_needs_minimal_remediation(built):
    store, _ = built
    geo = {b["name"]: b for b in leaderboard(store)["benchmarks"]}["MMLU::high_school_geography"]
    plan = geo["remediation"]
    # a clean (0 label-error) subject has no relay-only label correction
    assert not any(a["probe_id"] == "label_error_audit" for a in plan["actions"])
