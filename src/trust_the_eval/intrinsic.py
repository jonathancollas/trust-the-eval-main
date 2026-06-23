"""Model-free intrinsic audit job (P1): turn an annotation corpus into a record.

This is the generator that makes intrinsic profiles REAL instead of hand-written.
Given an :class:`AnnotationCorpus`, it produces a :class:`ValidityRecord`
(``kind="intrinsic"``) whose audits come from two honest, model-free sources:

  1. dataset-level probes run on the bare items — ``dataset_hygiene`` and
     ``coverage_distribution`` — each carrying its calibrated reliability;
  2. facts MEASURED DIRECTLY from the corpus's human annotations when present —
     the label-error rate, the item-ambiguity rate, the worst subject, and the
     effective accuracy ceiling (1 − label-error rate). These are exact counts
     over the corpus's SINGLE-PASS human annotations (not model estimates).
     Because the annotation is single-pass, inter-annotator agreement is NOT
     available, so the rates carry unquantified annotation uncertainty on top
     of sampling error, and are labelled as such.

Provenance records ``method="intrinsic_audit"`` and pins ``inputs_hash`` to the
corpus content hash, so the record is reproducible and verifiable.

Note: ``label_error_audit`` and ``item_ambiguity`` are model-in-the-loop probes
(they adjudicate with a panel of models). Here we are model-free, so we report
those quantities from the human annotations directly (a single-pass
re-annotation; no inter-annotator agreement) and say so in each audit's basis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import probes as _probes  # noqa: F401  (ensure probe registry is populated)
from .calibration import (
    build_cases, calibrate_probe, has_scenarios, score_threshold_for,
)
from .calibration.coverage import TIER_LABEL, evidence_for
from .corpus import AnnotationCorpus
from .probe import get_probe
from .record import (
    ProbeAudit, Provenance, ReliabilitySnapshot, Subject, ValidityRecord,
    _sha_of, _short, evidence_version,
)
from .runner import run_battery
from .taxonomy import claims_of

# model-free probes that audit the bare DATASET (no scores / responses needed)
DATASET_PROBES = ["dataset_hygiene", "coverage_distribution"]


def _sev_from_rate(rate: Optional[float], med: float, high: float) -> str:
    if rate is None:
        return "info"
    if rate >= high:
        return "high"
    if rate >= med:
        return "medium"
    if rate > 0:
        return "low"
    return "info"


def calibrate_dataset_probes(seed: int = 0) -> Dict[str, Any]:
    """Calibrate the dataset-level probes so their audits carry recall/spec."""
    cals: Dict[str, Any] = {}
    for pid in DATASET_PROBES:
        if not has_scenarios(pid):
            continue
        probe = get_probe(pid)()
        cals[pid] = calibrate_probe(probe, build_cases(pid, seed=seed),
                                    score_threshold=score_threshold_for(pid))
    return cals


def _annotation_audit(probe_id: str, rate_info: Dict[str, Any], worst,
                      kind_word: str, med: float, high: float,
                      corpus: AnnotationCorpus) -> ProbeAudit:
    rate = rate_info["rate"]
    sev = _sev_from_rate(rate, med, high)
    pct = "n/a" if rate is None else "{:.2f}%".format(rate * 100)
    summ = "{} measured from the corpus's human annotations (single-pass): {} of {} annotated items".format(
        kind_word, pct, rate_info["n"])
    measured: Dict[str, Any] = {"rate": rate, "k": rate_info["k"],
                                "n_annotated": rate_info["n"],
                                "derived": "human_annotation_single_pass",
                                "annotation_passes": 1,
                                "inter_annotator_agreement": None}
    if probe_id == "label_error_audit" and rate is not None:
        measured["effective_accuracy_ceiling"] = round(1.0 - rate, 4)
    if worst:
        measured["worst_subject"] = worst["subject"]
        measured["worst_subject_rate"] = round(worst["rate"], 4)
        summ += " · worst subject {} at {:.0f}%".format(worst["subject"], worst["rate"] * 100)
    tier, source, _note = evidence_for(probe_id)
    rel = ReliabilitySnapshot(
        probe_id=probe_id, tier=tier, tier_label=TIER_LABEL.get(tier, ""),
        source=(corpus.source or {}).get("dataset") or source,
        basis=("Measured directly from the corpus's per-item human annotations "
               "(a single-pass re-annotation; inter-annotator agreement "
               "unavailable) — not estimated by the model-in-the-loop probe."),
        calibration_version=evidence_version())
    return ProbeAudit(probe_id=probe_id, severity=sev, summary=summ, score=rate,
                      claims=claims_of(probe_id), measured=measured, reliability=rel)


def audit_corpus(corpus: AnnotationCorpus, *,
                 calibrations: Optional[Dict[str, Any]] = None,
                 store: Any = None,
                 label_med: float = 0.05, label_high: float = 0.15,
                 amb_med: float = 0.03, amb_high: float = 0.10) -> ValidityRecord:
    """Audit a corpus model-free and emit (and optionally store) a ValidityRecord."""
    artifact = corpus.to_artifact()
    report = run_battery(artifact, model=None, probe_ids=DATASET_PROBES)
    cals = calibrations or {}
    cv = evidence_version()

    audits: List[ProbeAudit] = []
    # (a) dataset-level probe findings (a real, model-free probe run)
    for f in report.findings:
        rel = (ReliabilitySnapshot.from_calibration(cals[f.probe_id], cv)
               if f.probe_id in cals
               else ReliabilitySnapshot.from_evidence(f.probe_id, cv))
        audits.append(ProbeAudit.from_finding(f, rel))
    # (b) annotation-derived facts (exact, only if the corpus is annotated)
    if corpus.has_annotations:
        audits.append(_annotation_audit(
            "label_error_audit", corpus.label_error(), corpus.worst_subject(),
            "Label-error rate", label_med, label_high, corpus))
        audits.append(_annotation_audit(
            "item_ambiguity", corpus.ambiguity(), None,
            "Item-ambiguity rate", amb_med, amb_high, corpus))

    applied = sorted(set(a.probe_id for a in audits)
                     | set(report.errors or {}) | set(report.skipped or []))
    psv = _short(_sha_of(applied))
    from . import __version__ as _v
    sources = [corpus.source] if corpus.source else []
    prov = Provenance(
        method="intrinsic_audit", tool_version=_v, probe_set_version=psv,
        calibration_version=cv, sources=sources, inputs_hash=corpus.content_hash(),
        note=("Model-free intrinsic audit of {}@{}. dataset_hygiene & "
              "coverage_distribution from a probe run on the items; label-error & "
              "item-ambiguity measured directly from the corpus's human annotations."
              ).format(corpus.benchmark, corpus.dataset_version))
    rec = ValidityRecord(
        subject=Subject(kind="intrinsic", benchmark=corpus.benchmark,
                        dataset_version=corpus.dataset_version),
        audits=audits, provenance=prov).finalize()
    if store is not None:
        store.put(rec)
    return rec
