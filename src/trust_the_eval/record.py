"""Validity record — the canonical, content-addressed unit of the observatory (P0).

A ValidityRecord is one dated, provenance-bearing audit of an evaluation's
VALIDITY. Two axes share a single envelope:

  - INTRINSIC : a benchmark at a dataset version (model-free properties:
    label errors, ambiguity, coverage, saturation-by-construction).
  - CLAIM     : a dated model-on-benchmark result (adds model / release / date /
    the reported score the claim actually rests on).

Every probe audit carries a SNAPSHOT of that probe's calibrated reliability as
known when the record was made, so a record never silently changes meaning when
probes are later recalibrated. Provenance is explicit and honest — including
whether the figures came from a live battery run or from cited literature.

Records are addressed by the SHA-256 of their CANONICAL content (everything that
asserts a fact: subject, audits, sources, method, versions, inputs) — never by a
wall-clock timestamp. Identical findings therefore yield an identical id, which
makes the store deduplicating and tamper-evident.

Bright line preserved: a record rates the eval instrument and the claim, never
the model or its safety.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .calibration.coverage import PROBE_EVIDENCE, TIER_LABEL, evidence_for
from .finding import Finding, Severity
from .taxonomy import claims_of

SCHEMA_VERSION = "ttev-record/1"


# --------------------------------------------------------------------- helpers
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha_of(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _short(sha: str, n: int = 12) -> str:
    return sha.split(":", 1)[-1][:n]


def _jsonable(x: Any) -> Any:
    """Best-effort conversion to JSON-safe, deterministic content."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Severity):
        return x.value
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def _fields(cls) -> List[str]:
    return list(cls.__dataclass_fields__)  # type: ignore[attr-defined]


def evidence_version() -> str:
    """Short version tag of the probe-evidence (tier) taxonomy."""
    pairs = sorted((pid, ev[0]) for pid, ev in PROBE_EVIDENCE.items())
    return _short(_sha_of(pairs))


# --------------------------------------------------------- reliability snapshot
@dataclass
class ReliabilitySnapshot:
    """A probe's calibrated reliability, frozen as known at record time."""
    probe_id: str
    tier: str
    tier_label: str = ""
    recall: Optional[float] = None
    recall_lo: Optional[float] = None
    recall_hi: Optional[float] = None
    recall_n: int = 0
    specificity: Optional[float] = None
    specificity_lo: Optional[float] = None
    specificity_hi: Optional[float] = None
    specificity_n: int = 0
    source: Optional[str] = None
    basis: str = ""
    calibration_version: str = ""

    @classmethod
    def from_evidence(cls, probe_id: str,
                      calibration_version: Optional[str] = None) -> "ReliabilitySnapshot":
        """Tier + honest basis only (no fresh metrics) — when no run is supplied."""
        tier, source, note = evidence_for(probe_id)
        return cls(probe_id=probe_id, tier=tier, tier_label=TIER_LABEL.get(tier, ""),
                   source=source, basis=note,
                   calibration_version=calibration_version or evidence_version())

    @classmethod
    def from_calibration(cls, cal: Any,
                         calibration_version: Optional[str] = None) -> "ReliabilitySnapshot":
        """Full metrics from a ProbeCalibration (recall/specificity + Wilson CIs)."""
        tier, source, note = evidence_for(cal.probe_id)
        rec = cal.matrix.recall()
        spec = cal.matrix.specificity()
        return cls(
            probe_id=cal.probe_id, tier=tier, tier_label=TIER_LABEL.get(tier, ""),
            recall=rec.value, recall_lo=rec.lo, recall_hi=rec.hi, recall_n=rec.n,
            specificity=spec.value, specificity_lo=spec.lo, specificity_hi=spec.hi,
            specificity_n=spec.n, source=source, basis=note,
            calibration_version=calibration_version or evidence_version())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReliabilitySnapshot":
        return cls(**{k: d[k] for k in _fields(cls) if k in d})


# --------------------------------------------------------------------- audit
@dataclass
class ProbeAudit:
    """One probe's finding within a record, with its reliability snapshot."""
    probe_id: str
    severity: str
    summary: str
    score: Optional[float] = None
    claims: List[str] = field(default_factory=list)
    measured: Dict[str, Any] = field(default_factory=dict)
    reliability: Optional[ReliabilitySnapshot] = None

    @classmethod
    def from_finding(cls, f: Finding,
                     reliability: Optional[ReliabilitySnapshot] = None) -> "ProbeAudit":
        sev = getattr(f.severity, "value", str(f.severity))
        return cls(probe_id=f.probe_id, severity=sev, summary=f.summary,
                   score=f.score, claims=claims_of(f.probe_id),
                   measured=_jsonable(f.evidence or {}),
                   reliability=reliability or ReliabilitySnapshot.from_evidence(f.probe_id))

    def to_dict(self) -> Dict[str, Any]:
        return {"probe_id": self.probe_id, "severity": self.severity,
                "summary": self.summary, "score": self.score,
                "claims": list(self.claims), "measured": self.measured,
                "reliability": self.reliability.to_dict() if self.reliability else None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProbeAudit":
        rel = d.get("reliability")
        return cls(probe_id=d["probe_id"], severity=d["severity"],
                   summary=d.get("summary", ""), score=d.get("score"),
                   claims=list(d.get("claims", [])), measured=d.get("measured", {}),
                   reliability=ReliabilitySnapshot.from_dict(rel) if rel else None)


# -------------------------------------------------------------------- subject
@dataclass
class Subject:
    """What is audited. `intrinsic` = benchmark@dataset_version (model-free);
    `claim` = a dated model-on-benchmark result."""
    kind: str
    benchmark: str
    dataset_version: Optional[str] = None
    model: Optional[str] = None
    release: Optional[str] = None
    date: Optional[str] = None          # date the claim refers to (YYYY-MM / ISO)
    reported_score: Optional[float] = None
    metric: Optional[str] = None        # e.g. "accuracy"

    def key(self) -> str:
        if self.kind == "claim":
            return "claim:{}:{}".format(self.benchmark, self.model or "?")
        return "intrinsic:{}@{}".format(self.benchmark, self.dataset_version or "?")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Subject":
        return cls(**{k: d[k] for k in _fields(cls) if k in d})


# ------------------------------------------------------------------ provenance
@dataclass
class Provenance:
    """How the record was produced. `created_utc` is metadata, NOT part of the id."""
    method: str                          # "battery_run" | "literature" | "imported"
    tool_version: str
    probe_set_version: str
    calibration_version: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    inputs_hash: Optional[str] = None
    note: Optional[str] = None
    created_utc: str = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "tool_version": self.tool_version,
                "probe_set_version": self.probe_set_version,
                "calibration_version": self.calibration_version,
                "sources": [_jsonable(s) for s in self.sources],
                "inputs_hash": self.inputs_hash, "note": self.note,
                "created_utc": self.created_utc}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        return cls(**{k: d[k] for k in _fields(cls) if k in d})


# ----------------------------------------------------------------- the record
@dataclass
class ValidityRecord:
    subject: Subject
    audits: List[ProbeAudit]
    provenance: Provenance
    schema_version: str = SCHEMA_VERSION
    record_id: Optional[str] = None      # the content hash; set by finalize()

    # ----- canonical content: hash-relevant, excludes wall-clock + record_id --
    def canonical(self) -> Dict[str, Any]:
        audits = sorted((a.to_dict() for a in self.audits),
                        key=lambda a: a["probe_id"])
        prov = self.provenance.to_dict()
        prov.pop("created_utc", None)    # a timestamp is not identity
        return {"schema_version": self.schema_version,
                "subject": self.subject.to_dict(),
                "audits": audits,
                "provenance": prov}

    def compute_id(self) -> str:
        return _sha_of(self.canonical())

    def finalize(self) -> "ValidityRecord":
        self.record_id = self.compute_id()
        return self

    def verify(self) -> bool:
        """True iff the stored id matches a fresh hash of the content."""
        return self.record_id == self.compute_id()

    def to_dict(self) -> Dict[str, Any]:
        return {"record_id": self.record_id or self.compute_id(),
                "schema_version": self.schema_version,
                "subject": self.subject.to_dict(),
                "audits": [a.to_dict() for a in self.audits],
                "provenance": self.provenance.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidityRecord":
        return cls(subject=Subject.from_dict(d["subject"]),
                   audits=[ProbeAudit.from_dict(a) for a in d.get("audits", [])],
                   provenance=Provenance.from_dict(d["provenance"]),
                   schema_version=d.get("schema_version", SCHEMA_VERSION),
                   record_id=d.get("record_id"))

    # ------------------------------------------------------------- builders ---
    @classmethod
    def from_report(cls, report: Any, *, kind: str, benchmark: str,
                    dataset_version: Optional[str] = None,
                    model: Optional[str] = None, release: Optional[str] = None,
                    date: Optional[str] = None, reported_score: Optional[float] = None,
                    metric: Optional[str] = None,
                    calibrations: Optional[Dict[str, Any]] = None,
                    calibration_version: Optional[str] = None,
                    sources: Optional[List[Dict[str, Any]]] = None,
                    method: str = "battery_run",
                    note: Optional[str] = None,
                    tool_version: Optional[str] = None) -> "ValidityRecord":
        """Build a record from a runner.Report.

        `calibrations` is an optional {probe_id: ProbeCalibration}; where present
        the audit snapshots full recall/specificity, else just the evidence tier.
        """
        from . import __version__ as _v
        cv = calibration_version or evidence_version()
        cals = calibrations or {}
        audits: List[ProbeAudit] = []
        for f in report.findings:
            rel = (ReliabilitySnapshot.from_calibration(cals[f.probe_id], cv)
                   if f.probe_id in cals
                   else ReliabilitySnapshot.from_evidence(f.probe_id, cv))
            audits.append(ProbeAudit.from_finding(f, rel))
        applied = sorted(set(a.probe_id for a in audits)
                         | set(getattr(report, "errors", {}) or {})
                         | set(getattr(report, "skipped", []) or []))
        psv = _short(_sha_of(applied)) if applied else _short(_sha_of(["none"]))
        subj = Subject(kind=kind, benchmark=benchmark, dataset_version=dataset_version,
                       model=model, release=release, date=date,
                       reported_score=reported_score, metric=metric)
        inputs_hash = None
        art = getattr(report, "artifact", None)
        if art is not None:
            try:
                inputs_hash = art.content_hash()
            except Exception:
                inputs_hash = None
        prov = Provenance(method=method, tool_version=tool_version or _v,
                          probe_set_version=psv, calibration_version=cv,
                          sources=list(sources or []), inputs_hash=inputs_hash, note=note)
        return cls(subject=subj, audits=audits, provenance=prov).finalize()

    @classmethod
    def literature(cls, *, benchmark: str, kind: str = "intrinsic",
                   dataset_version: Optional[str] = None,
                   audits: Optional[List[ProbeAudit]] = None,
                   sources: Optional[List[Dict[str, Any]]] = None,
                   note: Optional[str] = None,
                   calibration_version: Optional[str] = None,
                   **subject_kw: Any) -> "ValidityRecord":
        """A record whose figures are REPORTED BY CITED STUDIES, not a live run.

        Honest by construction: method='literature' and no inputs_hash, so the
        record can never be mistaken for a reproduced battery measurement.
        """
        from . import __version__ as _v
        cv = calibration_version or evidence_version()
        aud = list(audits or [])
        applied = sorted(set(a.probe_id for a in aud))
        psv = _short(_sha_of(applied)) if applied else _short(_sha_of(["literature"]))
        subj = Subject(kind=kind, benchmark=benchmark,
                       dataset_version=dataset_version, **subject_kw)
        prov = Provenance(
            method="literature", tool_version=_v, probe_set_version=psv,
            calibration_version=cv, sources=list(sources or []), inputs_hash=None,
            note=note or "Figures reported by cited studies; not a live battery run.")
        return cls(subject=subj, audits=aud, provenance=prov).finalize()
