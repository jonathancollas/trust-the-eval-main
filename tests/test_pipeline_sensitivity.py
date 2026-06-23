"""The autonomy pipeline turns a multi-model 'predictions' source into a
result_sensitivity record, surfaced on the validity leaderboard."""
from trust_the_eval.leaderboard import leaderboard
from trust_the_eval.pipeline import sync
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore


class _PredSource:
    id = "pred-test"
    anchors = {"min_rows": 2}

    def fingerprint(self):
        return "fp-1"

    def fetch(self):
        # i1's gold is corrected A->B (m2 benefits); i2 unchanged.
        return {
            "kind": "predictions", "benchmark": "MMLU", "dataset_version": "v1",
            "rows": [
                {"item": "i1", "subject": "MMLU", "original_gold": ["A"], "corrected_gold": ["B"],
                 "preds": {"m1": "A", "m2": "B"}},
                {"item": "i2", "subject": "MMLU", "original_gold": ["A"], "corrected_gold": ["A"],
                 "preds": {"m1": "A", "m2": "A"}},
            ],
        }


def test_predictions_source_becomes_sensitivity_record(tmp_path):
    store = RecordStore(tmp_path / "s")
    state = SourceState(tmp_path / "s")
    out = sync(store, [_PredSource()], state, root=tmp_path / "s",
               calibrations_dataset={}, calibrations_result={})
    assert any(o["status"] == "ingested" for o in out)

    # exactly one result_sensitivity audit landed in the store
    audits = [a for rid in store.all_ids()
              for a in store.get(rid)["audits"] if a["probe_id"] == "result_sensitivity"]
    assert len(audits) == 1
    m = audits[0]["measured"]
    assert m["n_changed_items"] == 1 and m["n_models"] == 2

    # the record is a verifiable battery_run with an inputs_hash
    rec = [store.get_record(rid) for rid in store.all_ids()][0]
    assert rec.verify() and rec.provenance.method == "battery_run"
    assert rec.provenance.inputs_hash

    # and the leaderboard surfaces it on the MMLU profile row
    b = [x for x in leaderboard(store)["benchmarks"] if x["name"] == "MMLU"][0]
    assert b["sensitivity"] is not None
    assert b["sensitivity"]["n_changed_items"] == 1

    # the same record also carries a test_reliability audit, surfaced on the row
    rel = [a for rid in store.all_ids() for a in store.get(rid)["audits"]
           if a["probe_id"] == "test_reliability"]
    assert len(rel) == 1
    assert b["reliability"] is not None

    # re-syncing the same fingerprint is a no-op (idempotent)
    out2 = sync(store, [_PredSource()], state, root=tmp_path / "s",
                calibrations_dataset={}, calibrations_result={})
    assert all(o["status"] == "skipped" for o in out2)
