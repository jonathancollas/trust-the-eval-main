import json
import os
import zipfile

from trust_the_eval.adapters import generic_json
from trust_the_eval.adapters import inspect_log
from trust_the_eval.artifact import SOURCE_RECORD_KEY
from trust_the_eval.cli import _load_artifact
from trust_the_eval.runner import estimate_battery_calls, run_battery
from trust_the_eval import probes as _probes  # noqa: F401

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_eval_result.json")


def test_battery_runs_static_and_skips_model_probes():
    rep = run_battery(generic_json.load(SAMPLE), model=None)
    ids = {f.probe_id for f in rep.findings}
    assert "statistical_power" in ids
    assert "dataset_hygiene" in ids
    # model-in-the-loop probes are SKIPPED, not failed
    assert "sandbagging_paired" in rep.skipped
    assert "contamination_perturb" in rep.skipped
    assert rep.coverage()["requested"] == 20
    assert rep.coverage()["completed"] == 8
    assert rep.coverage()["complete"] is False


def test_battery_stops_at_hard_model_call_budget():
    from trust_the_eval.model.local import HonestModel
    art = generic_json.load(SAMPLE)
    rep = run_battery(art, model=HonestModel(),
                      probe_ids=["contamination_perturb", "provenance_repro"],
                      max_model_calls=1)
    assert rep.cost["calls"] == 1
    assert rep.cost["budget_exhausted"] is True
    assert rep.stopped_reason and "budget exhausted" in rep.stopped_reason
    assert rep.coverage()["complete"] is False


def test_preflight_estimate_applies_constructor_overrides():
    art = generic_json.load(SAMPLE)
    estimate = estimate_battery_calls(
        art, probe_ids=["self_consistency", "contamination_perturb"],
        overrides={"self_consistency": {"sample_size": 2, "votes": 7},
                   "contamination_perturb": {"sample_size": 3}},
    )
    assert estimate["per_probe"] == {
        "self_consistency": 14,
        "contamination_perturb": 6,
    }
    assert estimate["total"] == 20


def test_report_exposes_preflight_estimate():
    art = generic_json.load(SAMPLE)
    rep = run_battery(art, probe_ids=["statistical_power", "provenance_repro"])
    assert rep.call_estimate["per_probe"] == {
        "statistical_power": 0,
        "provenance_repro": min(art.n, 20) * 2,
    }


def test_cli_detects_promptfoo_by_schema_not_filename(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": {"results": [{
        "prompt": "q", "response": {"output": "a"}, "success": True,
        "vars": {"expected": "a"}}]}}))
    art = _load_artifact(str(path))
    assert art.n == 1 and art.items[0].score == 1.0
    assert art.metadata["schema_version"] == "1.0"
    assert art.metadata["source_format"] == "promptfoo"
    assert art.items[0].meta[SOURCE_RECORD_KEY]["success"] is True


def test_inspect_current_plural_scores_are_preserved(tmp_path):
    path = tmp_path / "run.eval"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("header.json", json.dumps({"eval": {"task": "t"}}))
        z.writestr("samples/1.json", json.dumps({
            "input": "q", "target": "a", "output": "a",
            "scores": {"model_graded_qa": {"value": "C"}}}))
    art = inspect_log.load(str(path))
    assert art.items[0].score == 1.0
    assert "scores" in art.items[0].meta
    assert art.metadata["source_format"] == "inspect_eval"
    assert art.items[0].meta[SOURCE_RECORD_KEY]["scores"] == {
        "model_graded_qa": {"value": "C"}}


def test_generic_adapter_has_versioned_lossless_source_record(tmp_path):
    path = tmp_path / "generic.json"
    raw_item = {"question": "q", "answer": "a", "response": "a",
                "score": 1, "rubric": {"name": "exact"}, "trace": ["tool"]}
    path.write_text(json.dumps({"dataset": "d", "items": [raw_item]}))
    art = generic_json.load(str(path))
    assert art.metadata["schema_version"] == "1.0"
    assert art.metadata["source_format"] == "generic_json"
    assert art.items[0].meta[SOURCE_RECORD_KEY] == raw_item
    assert art.manifest()["n_scored"] == 1
