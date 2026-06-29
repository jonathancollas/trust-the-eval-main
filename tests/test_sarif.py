"""Tests for the SARIF emitter.

The two load-bearing disciplines get hard assertions: (1) no aggregate "trust
score" result of any kind, and (2) the honesty clamp — a probe whose evidence
strength is `contested`/`exploratory` can never be reported as `error`. The
offline corpus only yields `validated` structural probes, so the clamp is proven
synthetically via `_result` on a contested/exploratory probe id.
"""
import json
from pathlib import Path

import pytest

import trust_the_eval.meridian_cli as CLI
import trust_the_eval.pipeline as PL
from trust_the_eval.emit.sarif import _result, _tier, to_sarif
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore

FIX = Path(__file__).parent / "fixtures" / "mmlu_redux_real_mini.json"
BOOT = 120


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
def store_dir(tmp_path_factory):
    real = json.loads(FIX.read_text(encoding="utf-8"))
    srcs = []
    for subj, d in real.items():
        b = "MMLU::%s" % subj
        srcs.append(_Src("int-%s" % subj, {"kind": "intrinsic", "benchmark": b,
                                           "dataset_version": "mmlu-redux-2.0", "rows": d["intrinsic"]}))
        srcs.append(_Src("pred-%s" % subj, {"kind": "predictions", "benchmark": b,
                                            "dataset_version": "mmlu-redux-2.0", "rows": d["pred"]}))
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    orig = PL.build_record
    mp.setattr(PL, "build_record", lambda *a, **k: orig(*a, **{**k, "iters": k.get("iters", BOOT)}))
    root = tmp_path_factory.mktemp("sarif")
    store = RecordStore(root / "s")
    PL.sync(store, srcs, SourceState(root / "s"), root=root / "s",
            calibrations_dataset={}, calibrations_result={})
    mp.undo()
    return root / "s"


# ----------------------------------------------------------------- structure
def test_sarif_is_well_formed_2_1_0(store_dir):
    log = to_sarif(RecordStore(store_dir))
    assert log["version"] == "2.1.0"
    assert log["$schema"].endswith("sarif-2.1.0.json")
    run = log["runs"][0]
    assert run["tool"]["driver"]["name"] == "trust-the-eval"
    assert run["tool"]["driver"]["rules"]
    for r in run["results"]:
        assert r["ruleId"] and r["message"]["text"]
        assert r["locations"][0]["logicalLocations"][0]["fullyQualifiedName"]
        assert r["partialFingerprints"]["tte/v1"]
        assert r["level"] in ("error", "warning", "note", "none")
        assert r["kind"] in ("fail", "review", "pass", "open", "informational", "notApplicable")
    for rule in run["tool"]["driver"]["rules"]:
        assert rule["id"] and rule["shortDescription"]["text"]
        assert rule["properties"]["evidence_strength"] in ("validated", "contested", "exploratory")


# ----------------------------------------------- discipline 1: no aggregate
def test_no_aggregate_trust_score_anywhere(store_dir):
    run = to_sarif(RecordStore(store_dir))["runs"][0]
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    result_ids = {r["ruleId"] for r in run["results"]}
    assert result_ids <= rule_ids                       # every result maps to a real probe
    banned = {"trust_score", "overall", "score", "composite", "total", "grade"}
    assert not (banned & result_ids)                    # no synthetic composite finding
    for r in run["results"]:                            # no 0-100 score smuggled in properties
        assert "score" not in r["properties"]
        assert "trust_score" not in r["properties"]


# ------------------------------------- discipline 2: the honesty clamp (teeth)
@pytest.mark.parametrize("pid", ["contamination_perturb", "sandbagging_paired",
                                 "model_drift", "reward_hacking_eval", "elicitation_ceiling"])
def test_non_validated_probe_can_never_be_error(pid):
    assert _tier(pid) in ("contested", "exploratory")
    # even at the highest severity, a non-validated finding is clamped
    r = _result(pid, "MMLU::x", {"severity": "high", "summary": "stuff", "measured": {}},
                model=None, record_id="rid", inputs_hash="h")
    assert r["level"] != "error"
    assert r["level"] == "warning"
    assert r["kind"] == "review"
    assert "INVESTIGATE" in r["message"]["text"]


def test_conditional_validated_verdict_is_review_not_fail():
    # result_sensitivity is validated but its verdict is conditional -> review, never a hard error
    r = _result("result_sensitivity", "MMLU::x",
                {"severity": "high", "summary": "rank-fragile", "measured": {"kendall_tau": 0.2}},
                model=None, record_id="rid", inputs_hash="h")
    assert r["kind"] == "review"
    assert r["level"] == "warning"
    assert r["message"]["text"].startswith("[validated]")


def test_hard_validated_defect_can_be_error():
    # a measured label-error defect from real human labels IS a hard instrument fault
    r = _result("label_error_audit", "MMLU::x",
                {"severity": "high", "summary": "41% of the key is wrong", "measured": {"rate": 0.41}},
                model=None, record_id="rid", inputs_hash="h")
    assert r["level"] == "error"
    assert r["kind"] == "fail"


# --------------------------------------------------- recompute pointer + ids
def test_supported_metrics_carry_a_recompute_pointer(store_dir):
    run = to_sarif(RecordStore(store_dir))["runs"][0]
    supported = {"result_sensitivity", "test_reliability", "label_error_audit", "item_ambiguity"}
    sup = [r for r in run["results"] if r["ruleId"] in supported]
    assert sup, "expected some supported-metric findings"
    for r in sup:
        assert r["properties"]["verify"].startswith("meridian explain")


def test_fingerprint_is_identity_not_value(store_dir):
    run = to_sarif(RecordStore(store_dir))["runs"][0]
    for r in run["results"]:
        fq = r["locations"][0]["logicalLocations"][0]["fullyQualifiedName"]
        bench = fq.split(" [")[0]
        assert r["partialFingerprints"]["tte/v1"] == "%s:%s" % (r["ruleId"], bench)


# -------------------------------------------------------------- determinism
def test_emitter_is_deterministic(store_dir):
    a = json.dumps(to_sarif(RecordStore(store_dir)), sort_keys=True)
    b = json.dumps(to_sarif(RecordStore(store_dir)), sort_keys=True)
    assert a == b


# ---------------------------------------------------------------------- CLI
def test_cli_sarif_writes_a_valid_file(store_dir, tmp_path, capsys):
    rc = CLI.main(["sarif", "--store", str(store_dir), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "meridian.sarif" in out
    log = json.loads((tmp_path / "meridian.sarif").read_text(encoding="utf-8"))
    assert log["version"] == "2.1.0"
    assert "no aggregate score" in out
