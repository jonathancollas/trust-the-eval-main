import json
import os
import zipfile

from trust_the_eval.adapters import generic_json
from trust_the_eval.adapters import inspect_log
from trust_the_eval.cli import _load_artifact
from trust_the_eval.runner import run_battery
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


def test_cli_detects_promptfoo_by_schema_not_filename(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": {"results": [{
        "prompt": "q", "response": {"output": "a"}, "success": True,
        "vars": {"expected": "a"}}]}}))
    art = _load_artifact(str(path))
    assert art.n == 1 and art.items[0].score == 1.0


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
