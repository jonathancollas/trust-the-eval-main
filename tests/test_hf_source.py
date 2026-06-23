"""Offline tests for the Hugging Face Dataset Viewer client.

The HTTP layer (hf._get) is monkeypatched to return canned envelopes shaped
exactly like the documented datasets-server responses, so parsing, mapping,
coercion and pagination are all proven without any network access.
"""
import pytest
from trust_the_eval.sources import hf
from trust_the_eval.runner import run_battery
from trust_the_eval import probes as _probes  # noqa: F401  registers all probes


def test_guess_mapping_exact_and_substring():
    g = hf.guess_mapping(["Question", "Target", "model_output", "exact_match", "Subject", "choices"])
    assert g["question"] == "Question"
    assert g["answer"] == "Target"
    assert g["response"] == "model_output"
    assert g["score"] == "exact_match"
    assert g["category"] == "Subject"
    assert g["options"] == "choices"


def test_parse_rows_response():
    payload = {"features": [{"name": "q"}, {"name": "a"}],
               "rows": [{"row_idx": 0, "row": {"q": "2+2?", "a": "4"}},
                        {"row_idx": 1, "row": {"q": "3+3?", "a": "6"}}]}
    rows = hf._parse_rows_response(payload)
    assert rows == [{"q": "2+2?", "a": "4"}, {"q": "3+3?", "a": "6"}]


def test_coerce_score_variants():
    assert hf._coerce_score(True) == 1.0
    assert hf._coerce_score(0) == 0.0
    assert hf._coerce_score("correct") == 1.0
    assert hf._coerce_score("0.5") == 0.5
    assert hf._coerce_score([1]) == 1.0
    assert hf._coerce_score({"acc": 1.0}) == 1.0
    assert hf._coerce_score("banana") is None
    assert hf._coerce_score(None) is None


def test_coerce_text_nested_and_list():
    assert hf._coerce_text(["hello", "x"]) == "hello"
    assert hf._coerce_text({"text": "hi"}) == "hi"
    assert hf._coerce_text(None) == ""
    assert hf._coerce_text(42) == "42"


def test_rows_to_artifact_benchmark_shape():
    # gsm8k-like: question + answer only (a benchmark, no responses)
    rows = [{"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 3 + 5?", "answer": "8"},
            {"question": "   ", "answer": "x"}]  # blank question -> skipped
    art = hf.rows_to_artifact(rows, {"question": "question", "answer": "answer"}, "demo/gsm")
    assert art.n == 2 and art.items[0].question.startswith("What")
    assert all(it.response is None for it in art.items)  # benchmark: no responses


def test_rows_to_artifact_eval_result_shape_and_probe_runs():
    # details-like: includes model output + metric => a real eval RESULT
    rows = [
        {"doc": "What is 2 + 2?", "target": "4", "filtered_resps": ["The answer is 4."],
         "exact_match": 1, "subject": "addition"},
        {"doc": "What is 10 - 7?", "target": "3", "filtered_resps": ["The answer is 2."],
         "exact_match": 0, "subject": "subtraction"},
    ]
    mapping = hf.guess_mapping(list(rows[0].keys()))
    art = hf.rows_to_artifact(rows, mapping, "org__model-details")
    assert art.n == 2
    assert art.items[0].response == "The answer is 4."
    assert art.items[0].score == 1.0 and art.items[1].score == 0.0
    assert art.items[0].meta["category"] == "addition"
    # a static probe should run on this real-result artifact
    rep = run_battery(art, model=None, probe_ids=["answer_extraction_audit", "statistical_power"])
    assert any(f.probe_id == "answer_extraction_audit" for f in rep.findings)


def test_fetch_rows_paginates(monkeypatch):
    # simulate a 250-row dataset served 100 at a time
    total = 250
    def fake_get(path, params, token=None, timeout=30):
        assert path == "/rows"
        off = int(params["offset"]); length = int(params["length"])
        rows = [{"row_idx": i, "row": {"q": f"q{i}", "a": str(i)}}
                for i in range(off, min(off + length, total))]
        return {"features": [{"name": "q"}, {"name": "a"}], "rows": rows}
    monkeypatch.setattr(hf, "_get", fake_get)
    got = hf.fetch_rows("d", "c", "train", n=230, token=None)
    assert len(got) == 230
    assert got[0]["q"] == "q0" and got[-1]["q"] == "q229"


def test_inspect_dataset(monkeypatch):
    def fake_get(path, params, token=None, timeout=30):
        if path == "/splits":
            return {"splits": [{"config": "main", "split": "train"},
                               {"config": "main", "split": "test"}]}
        if path == "/first-rows":
            assert params["split"] == "test"  # prefers test split
            return {"features": [{"name": "question"}, {"name": "answer"}],
                    "rows": [{"row": {"question": "2+2?", "answer": "4"}}]}
        raise AssertionError(path)
    monkeypatch.setattr(hf, "_get", fake_get)
    info = hf.inspect_dataset("demo/ds")
    assert info["split"] == "test" and "question" in info["columns"]
    assert info["guess"]["question"] == "question"
    assert len(info["sample"]) == 1


def test_gated_without_token_raises(monkeypatch):
    import urllib.error, io
    def fake_get(path, params, token=None, timeout=30):
        raise PermissionError("Hugging Face returned 401: gated (provide an access token)")
    monkeypatch.setattr(hf, "_get", fake_get)
    with pytest.raises(PermissionError):
        hf.list_splits("gated/ds")


def test_search_datasets_filters_relevant_by_default(monkeypatch):
    import urllib.request as _urlreq
    payload = [
        {"id": "openai/gsm8k", "tags": ["benchmark"], "downloads": 10},
        {"id": "imagenet-1k", "tags": ["image-classification"], "downloads": 20},
        {"id": "cais/mmlu", "tags": ["question-answering"], "downloads": 30},
    ]

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            import json
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(_urlreq, "urlopen", lambda *a, **k: FakeResp())
    out = hf.search_datasets(limit=10)
    ids = [d["id"] for d in out]
    assert "openai/gsm8k" in ids
    assert "cais/mmlu" in ids
    assert "imagenet-1k" not in ids


def test_search_datasets_relevant_filter_can_be_disabled(monkeypatch):
    import urllib.request as _urlreq
    payload = [
        {"id": "openai/gsm8k", "tags": ["benchmark"], "downloads": 10},
        {"id": "imagenet-1k", "tags": ["image-classification"], "downloads": 20},
    ]

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            import json
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(_urlreq, "urlopen", lambda *a, **k: FakeResp())
    out = hf.search_datasets(limit=10, relevant_only=False)
    ids = [d["id"] for d in out]
    assert "openai/gsm8k" in ids
    assert "imagenet-1k" in ids
