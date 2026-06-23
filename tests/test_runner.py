import os

from trust_the_eval.adapters import generic_json
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
