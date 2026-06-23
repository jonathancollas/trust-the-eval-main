"""Tests for trust_the_eval.record — the content-addressed validity record (P0)."""
import trust_the_eval.probes  # noqa: F401  (register all probes)
from trust_the_eval.artifact import EvalArtifact, EvalItem
from trust_the_eval.calibration import build_cases, calibrate_probe, score_threshold_for
from trust_the_eval.calibration.coverage import (
    TIER_BEHAVIORAL, TIER_REAL, TIER_STRUCTURAL,
)
from trust_the_eval.finding import Finding, Severity
from trust_the_eval.probe import get_probe
from trust_the_eval.record import (
    ProbeAudit, ReliabilitySnapshot, Subject, ValidityRecord, evidence_version,
)
from trust_the_eval.runner import Report, run_battery

_TIERS = {TIER_REAL, TIER_STRUCTURAL, TIER_BEHAVIORAL}


def _artifact(n=12):
    items = []
    for i in range(n):
        items.append(EvalItem(
            question="q{}".format(i), answer="A",
            response="A" if i % 2 else "B", score=float(i % 2),
            meta={"category": "math" if i % 3 else "history"}))
    return EvalArtifact(dataset="toy", items=items, model="m-1")


def _report():
    return run_battery(_artifact(), model=None)  # static probes only


def test_from_report_builds_record_with_id_and_verifies():
    rec = ValidityRecord.from_report(_report(), kind="intrinsic",
                                     benchmark="Toy", dataset_version="v1")
    assert rec.record_id and rec.record_id.startswith("sha256:")
    assert rec.verify()
    assert rec.subject.kind == "intrinsic"
    assert rec.provenance.method == "battery_run"
    assert rec.provenance.inputs_hash and rec.provenance.inputs_hash.startswith("sha256:")
    assert len(rec.audits) >= 1  # at least some static probe fired


def test_every_audit_carries_a_reliability_snapshot_with_known_tier():
    rec = ValidityRecord.from_report(_report(), kind="intrinsic", benchmark="Toy")
    for a in rec.audits:
        assert a.reliability is not None
        assert a.reliability.tier in _TIERS
        assert a.reliability.calibration_version == evidence_version()


def test_content_id_is_deterministic_and_excludes_timestamp():
    rep = _report()
    a = ValidityRecord.from_report(rep, kind="intrinsic", benchmark="Toy")
    b = ValidityRecord.from_report(rep, kind="intrinsic", benchmark="Toy")
    assert a.record_id == b.record_id                 # same content -> same id
    a.provenance.created_utc = "1999-01-01T00:00:00Z"  # timestamp is not identity
    assert a.compute_id() == a.record_id


def test_changing_a_finding_changes_the_id():
    art = _artifact()
    rep1 = Report(artifact=art, findings=[Finding("statistical_power", Severity.HIGH, "underpowered", 0.9)])
    rep2 = Report(artifact=art, findings=[Finding("statistical_power", Severity.LOW, "fine", 0.1)])
    r1 = ValidityRecord.from_report(rep1, kind="intrinsic", benchmark="Toy")
    r2 = ValidityRecord.from_report(rep2, kind="intrinsic", benchmark="Toy")
    assert r1.record_id != r2.record_id


def test_roundtrip_to_dict_from_dict_preserves_id():
    rec = ValidityRecord.from_report(_report(), kind="intrinsic", benchmark="Toy")
    rec2 = ValidityRecord.from_dict(rec.to_dict())
    assert rec2.record_id == rec.record_id
    assert rec2.verify()
    assert rec2.compute_id() == rec.record_id


def test_calibration_snapshot_attaches_full_metrics():
    pid = "statistical_power"
    probe = get_probe(pid)()
    cal = calibrate_probe(probe, build_cases(pid, seed=0),
                          score_threshold=score_threshold_for(pid))
    snap = ReliabilitySnapshot.from_calibration(cal)
    assert snap.tier == TIER_STRUCTURAL
    assert snap.recall is not None and snap.recall_n > 0
    assert snap.specificity is not None and snap.specificity_n > 0
    # and it flows through from_report when calibrations are supplied
    rep = Report(artifact=_artifact(),
                 findings=[Finding(pid, Severity.HIGH, "underpowered", 0.9)])
    rec = ValidityRecord.from_report(rep, kind="intrinsic", benchmark="Toy",
                                     calibrations={pid: cal})
    assert rec.audits[0].reliability.recall is not None
    assert rec.audits[0].reliability.recall_n > 0


def test_literature_record_is_honest_about_provenance():
    rec = ValidityRecord.literature(
        benchmark="MMLU", dataset_version="mmlu-redux-2.0",
        audits=[ProbeAudit(probe_id="label_error_audit", severity="medium",
                           summary="6.49% of items erroneous", score=0.5)],
        sources=[{"title": "Are We Done with MMLU?", "url": "https://arxiv.org/abs/2406.04127"}])
    assert rec.provenance.method == "literature"
    assert rec.provenance.inputs_hash is None
    assert rec.provenance.sources and rec.provenance.sources[0]["url"].startswith("https://")
    assert rec.verify()


def test_claim_subject_key_distinguishes_axis():
    s_claim = Subject(kind="claim", benchmark="MMLU", model="GPT-4")
    s_intr = Subject(kind="intrinsic", benchmark="MMLU", dataset_version="redux-2.0")
    assert s_claim.key().startswith("claim:MMLU:")
    assert s_intr.key().startswith("intrinsic:MMLU@")
