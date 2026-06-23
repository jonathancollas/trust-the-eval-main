"""Tests for the `meridian` CLI capstone (build / verify)."""
import json
import os

from trust_the_eval.meridian_cli import main


def test_build_demo_then_verify(tmp_path):
    store, out = tmp_path / "s", tmp_path / "o"
    assert main(["build", "--store", str(store), "--out", str(out)]) == 0
    for f in ("index.html", "records.json", "meridian.json"):
        assert (out / f).exists()
    doc = json.loads((out / "meridian.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "meridian-api/v1" and doc["integrity"]["verified"] is True
    assert doc["benchmarks"] and doc["method"]["admitted"]
    assert main(["verify", "--store", str(store)]) == 0     # re-verify same store+registry


def test_build_with_fixtures(tmp_path):
    here = os.path.dirname(__file__)
    corpus = os.path.join(here, "fixtures", "mmlu_redux_sample.jsonl")
    board = os.path.join(here, "fixtures", "leaderboard_sample.jsonl")
    store, out = tmp_path / "s", tmp_path / "o"
    rc = main(["build", "--corpus", corpus, "--leaderboard", board,
               "--benchmark", "MMLU", "--store", str(store), "--out", str(out)])
    assert rc == 0
    doc = json.loads((out / "meridian.json").read_text(encoding="utf-8"))
    assert "MMLU" in [b["name"] for b in doc["benchmarks"]]


def test_build_no_admission_flag(tmp_path):
    store, out = tmp_path / "s", tmp_path / "o"
    assert main(["build", "--no-admission", "--store", str(store), "--out", str(out)]) == 0
    doc = json.loads((out / "meridian.json").read_text(encoding="utf-8"))
    assert "method" not in doc                              # gate skipped
