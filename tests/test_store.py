"""Tests for trust_the_eval.store — append-only content-addressed store (P0)."""
import json

from trust_the_eval.record import ProbeAudit, ValidityRecord
from trust_the_eval.store import RecordStore


def _intrinsic(benchmark="MMLU", dataset_version="mmlu-redux-2.0", summary="6.49%"):
    return ValidityRecord.literature(
        benchmark=benchmark, dataset_version=dataset_version,
        audits=[ProbeAudit(probe_id="label_error_audit", severity="medium",
                           summary=summary, score=0.5)],
        sources=[{"title": "MMLU-Redux", "url": "https://arxiv.org/abs/2406.04127"}])


def _claim(model, date, score):
    return ValidityRecord.literature(
        benchmark="MMLU", kind="claim", model=model, date=date,
        reported_score=score, metric="accuracy",
        audits=[ProbeAudit(probe_id="discrimination_saturation", severity="high",
                           summary="saturated", score=0.9)])


def test_put_get_roundtrip(tmp_path):
    st = RecordStore(tmp_path)
    rec = _intrinsic()
    rid = st.put(rec)
    assert rid == rec.record_id
    assert st.has(rid)
    assert st.get(rid)["record_id"] == rid
    assert st.get_record(rid).verify()


def test_put_is_idempotent(tmp_path):
    st = RecordStore(tmp_path)
    rec = _intrinsic()
    rid1 = st.put(rec)
    rid2 = st.put(rec)                 # same content again
    assert rid1 == rid2
    assert len(st.all_ids()) == 1      # no duplicate file
    assert len(st.index()) == 1        # no duplicate index line


def test_distinct_content_distinct_records(tmp_path):
    st = RecordStore(tmp_path)
    st.put(_intrinsic(summary="6.49% of items erroneous"))
    st.put(_intrinsic(summary="something else"))
    assert len(st.all_ids()) == 2


def test_list_filters_by_kind_and_benchmark(tmp_path):
    st = RecordStore(tmp_path)
    st.put(_intrinsic())
    st.put(_claim("GPT-4", "2023-03", 0.864))
    assert len(st.list(kind="intrinsic")) == 1
    assert len(st.list(kind="claim")) == 1
    assert len(st.list(benchmark="MMLU")) == 2
    assert st.list(benchmark="GSM8K") == []


def test_history_is_ordered_by_date(tmp_path):
    st = RecordStore(tmp_path)
    st.put(_claim("GPT-4.1", "2025-02", 0.902))
    st.put(_claim("GPT-4", "2023-03", 0.864))
    st.put(_claim("GPT-4o", "2024-05", 0.887))
    hist = st.history("MMLU", kind="claim")
    dates = [r["date"] for r in hist]
    assert dates == sorted(dates)
    assert dates[0] == "2023-03" and dates[-1] == "2025-02"


def test_verify_all_clean_then_detects_tampering(tmp_path):
    st = RecordStore(tmp_path)
    rid = st.put(_intrinsic())
    assert st.verify_all() == []                       # pristine
    # tamper with the stored file in place, keeping its (now stale) filename
    path = st.records_dir / (rid.split(":", 1)[-1] + ".json")
    d = json.loads(path.read_text(encoding="utf-8"))
    d["audits"][0]["severity"] = "low"                 # alter content
    path.write_text(json.dumps(d), encoding="utf-8")
    assert st.verify_all() == [rid]                    # content no longer hashes to its id


def test_store_is_append_only(tmp_path):
    st = RecordStore(tmp_path)
    assert not hasattr(st, "delete")
    assert not hasattr(st, "update")
