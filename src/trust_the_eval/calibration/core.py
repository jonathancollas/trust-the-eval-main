"""Calibration core: treat each probe as a diagnostic test and *measure* it.

A probe emits a per-artifact verdict (defect present / absent) plus, usually, a
continuous score. Calibration confronts that verdict against a KNOWN ground
truth and reports the diagnostic-test quantities the rest of this package
preaches in ``statistical_power``:

  * a confusion matrix (TP / FP / FN / TN),
  * precision, recall (sensitivity), specificity, F1 and accuracy, each with a
    Wilson 95% interval (``stats.wilson_ci``) so a small calibration set is read
    with honest uncertainty rather than a bare point estimate,
  * a threshold sweep over the probe's continuous score giving ROC and
    precision-recall curves with their areas (ROC-AUC, average precision).

This is the engine behind the published "here is the measured precision/recall
of every probe" report. It is deliberately dependency-free and offline; the
SVG curves and the real-dataset adapters live in sibling modules.

Bright line preserved: calibration measures whether a probe correctly detects a
VALIDITY defect in an eval artifact. It never assesses model safety, and the
"defect" populations are constructed from the package's own controllable
fixtures and planted-defect generators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..probe import ModelClient, Probe
from ..stats import wilson_ci


# --------------------------------------------------------------------------
# Ground-truth cases and the label a probe verdict is scored against
# --------------------------------------------------------------------------
@dataclass
class CalibrationCase:
    """One labelled artifact: does the targeted defect truly exist in it?

    ``defect`` is the ground truth (True = the artifact really has the defect
    this probe is meant to catch). ``regime`` tags *clean-but-tricky* cases so
    specificity can be reported per failure regime (the honest, measured form
    of "when not to trust the probe").
    """
    name: str
    artifact: EvalArtifact
    defect: bool
    model: Optional[ModelClient] = None
    regime: str = "default"
    note: str = ""


def severity_rank(sev: Severity) -> int:
    order = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3}
    return order.get(sev, 0)


def verdict_from_findings(findings: list[Finding],
                          flag_at: Severity = Severity.MEDIUM,
                          score_threshold: Optional[float] = None) -> bool:
    """A probe 'flags a defect' for the confusion matrix.

    Two verdict rules, picked per scenario:
      * **severity rule** (default): the top finding reaches ``flag_at`` and is
        not INFO. Right for probes whose severity is the operating signal.
      * **score rule** (``score_threshold`` given): the probe's numeric score is
        >= the threshold. Right for probes that report a constant severity but a
        discriminating continuous score (e.g. statistical_power always MEDIUM,
        but score = minimum significant gap separates powered from underpowered).
    INFO-only / no-score findings never count as a positive verdict.
    """
    if score_threshold is not None:
        s = score_from_findings(findings)
        only_info = bool(findings) and all(f.severity == Severity.INFO for f in findings)
        return (s is not None) and (s >= score_threshold) and not only_info
    thr = severity_rank(flag_at)
    return any(severity_rank(f.severity) >= thr and f.severity != Severity.INFO
               for f in findings)


def score_from_findings(findings: list[Finding]) -> Optional[float]:
    """The continuous probe score used for the ROC/PR sweep.

    Falls back to a severity-derived score when a probe reports no numeric
    ``score`` (so even purely categorical probes get a usable sweep axis).
    """
    numeric = [f.score for f in findings if f.score is not None]
    if numeric:
        return max(numeric)
    if not findings:
        return 0.0
    top = max(severity_rank(f.severity) for f in findings)
    return {0: 0.0, 1: 0.25, 2: 0.6, 3: 0.9}.get(top, 0.0)


# --------------------------------------------------------------------------
# Confusion matrix + diagnostic-test metrics (with Wilson intervals)
# --------------------------------------------------------------------------
@dataclass
class Metric:
    """A rate with its Wilson 95% interval and the k/n it came from."""
    value: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    k: int
    n: int

    @classmethod
    def of(cls, k: int, n: int) -> "Metric":
        if n <= 0:
            return cls(None, None, None, k, n)
        lo, hi = wilson_ci(k, n)
        return cls(k / n, lo, hi, k, n)

    def to_dict(self) -> dict:
        return {"value": self.value, "lo": self.lo, "hi": self.hi,
                "k": self.k, "n": self.n}


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def positives(self) -> int:      # ground-truth defective
        return self.tp + self.fn

    @property
    def negatives(self) -> int:      # ground-truth clean
        return self.tn + self.fp

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    # diagnostic-test rates, each as a Wilson-interval Metric
    def precision(self) -> Metric:   # of the things flagged, how many were real
        return Metric.of(self.tp, self.tp + self.fp)

    def recall(self) -> Metric:      # sensitivity: of real defects, how many caught
        return Metric.of(self.tp, self.tp + self.fn)

    def specificity(self) -> Metric:  # of clean artifacts, how many correctly cleared
        return Metric.of(self.tn, self.tn + self.fp)

    def accuracy(self) -> Metric:
        return Metric.of(self.tp + self.tn, self.total)

    def f1(self) -> Optional[float]:
        p = self.precision().value
        r = self.recall().value
        if not p or not r or (p + r) == 0:
            return 0.0 if (p is not None and r is not None) else None
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "positives": self.positives, "negatives": self.negatives,
            "total": self.total,
            "precision": self.precision().to_dict(),
            "recall": self.recall().to_dict(),
            "specificity": self.specificity().to_dict(),
            "accuracy": self.accuracy().to_dict(),
            "f1": self.f1(),
        }


# --------------------------------------------------------------------------
# ROC / precision-recall sweep over the probe's continuous score
# --------------------------------------------------------------------------
@dataclass
class CurvePoint:
    threshold: float
    tpr: float            # recall / sensitivity
    fpr: float            # 1 - specificity
    precision: float
    recall: float


@dataclass
class Curves:
    points: list[CurvePoint] = field(default_factory=list)
    roc_auc: Optional[float] = None
    average_precision: Optional[float] = None
    prevalence: Optional[float] = None
    orientation: str = "direct"   # direct | inverted | non_monotonic | degenerate

    def to_dict(self) -> dict:
        return {
            "roc": [{"fpr": p.fpr, "tpr": p.tpr, "threshold": p.threshold}
                    for p in self.points],
            "pr": [{"recall": p.recall, "precision": p.precision,
                    "threshold": p.threshold} for p in self.points],
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "prevalence": self.prevalence,
            "orientation": self.orientation,
        }


def _sweep(labels: list[bool], scores: list[float]) -> Curves:
    """Build ROC and PR curves from labelled continuous scores, orientation-aware.

    A probe's numeric score is not always monotone-increasing in "defect more
    likely". Some scores are INVERTED (e.g. provenance_repro: high determinism =
    clean) and some are NON-MONOTONIC / U-shaped (e.g. discrimination_saturation:
    both ceiling≈1 and floor≈0 are defects). We detect the orientation by
    comparing mean score on defect vs clean, sweep in the direction that makes
    the score a defect indicator, and tag the orientation so a reader interprets
    the AUC correctly rather than seeing a misleadingly low number.

    ROC-AUC is the trapezoid area; average precision is Σ (R_k - R_{k-1})·P_k.
    """
    n = len(labels)
    P = sum(1 for d in labels if d)
    N = n - P
    prevalence = P / n if n else None
    if P == 0 or N == 0:
        return Curves(points=[], roc_auc=None, average_precision=None,
                      prevalence=prevalence, orientation="degenerate")

    mean_def = sum(s for s, d in zip(scores, labels) if d) / P
    mean_clean = sum(s for s, d in zip(scores, labels) if not d) / N
    # detect non-monotonic: defects spread on BOTH sides of the clean mean
    defect_scores = [s for s, d in zip(scores, labels) if d]
    hi = sum(1 for s in defect_scores if s > mean_clean)
    lo = sum(1 for s in defect_scores if s < mean_clean)
    if hi > 0 and lo > 0 and min(hi, lo) / max(hi, lo) >= 0.34:
        orientation = "non_monotonic"
    elif mean_def >= mean_clean:
        orientation = "direct"
    else:
        orientation = "inverted"

    # orient scores so that "larger oriented score" = "more defect-like"
    if orientation == "inverted":
        osc = [-s for s in scores]
    elif orientation == "non_monotonic":
        # fold around the clean mean: distance from clean center indicates defect
        osc = [abs(s - mean_clean) for s in scores]
    else:
        osc = list(scores)

    thresholds = sorted(set(osc), reverse=True)
    grid = [thresholds[0] + 1e-9] + thresholds
    pts: list[CurvePoint] = []
    for t in grid:
        tp = sum(1 for d, s in zip(labels, osc) if s >= t and d)
        fp = sum(1 for d, s in zip(labels, osc) if s >= t and not d)
        fn = P - tp
        tpr = tp / P
        fpr = fp / N
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        pts.append(CurvePoint(t, tpr, fpr, precision, tpr))

    roc_sorted = sorted(pts, key=lambda p: (p.fpr, p.tpr))
    roc_auc = 0.0
    for a, b in zip(roc_sorted, roc_sorted[1:]):
        roc_auc += (b.fpr - a.fpr) * (a.tpr + b.tpr) / 2.0

    pr_sorted = sorted(pts, key=lambda p: p.recall)
    ap = 0.0
    prev_r = 0.0
    for p in pr_sorted:
        if p.recall > prev_r:
            ap += (p.recall - prev_r) * p.precision
            prev_r = p.recall

    return Curves(points=pts, roc_auc=roc_auc, average_precision=ap,
                  prevalence=prevalence, orientation=orientation)


# --------------------------------------------------------------------------
# Running a probe over a set of labelled cases
# --------------------------------------------------------------------------
@dataclass
class CaseResult:
    name: str
    defect: bool          # ground truth
    flagged: bool         # probe verdict
    score: Optional[float]
    severity: str
    regime: str
    summary: str
    outcome: str          # TP / FP / FN / TN

    def to_dict(self) -> dict:
        return {"name": self.name, "defect": self.defect, "flagged": self.flagged,
                "score": self.score, "severity": self.severity,
                "regime": self.regime, "summary": self.summary,
                "outcome": self.outcome}


@dataclass
class ProbeCalibration:
    probe_id: str
    probe_name: str
    requires_model: bool
    flag_at: str
    matrix: ConfusionMatrix
    curves: Curves
    cases: list[CaseResult]
    regime_specificity: dict[str, dict]   # regime -> {specificity Metric dict, n}
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "probe_name": self.probe_name,
            "requires_model": self.requires_model,
            "flag_at": self.flag_at,
            "matrix": self.matrix.to_dict(),
            "curves": self.curves.to_dict(),
            "cases": [c.to_dict() for c in self.cases],
            "regime_specificity": self.regime_specificity,
            "error": self.error,
        }


def _outcome(defect: bool, flagged: bool) -> str:
    if defect and flagged:
        return "TP"
    if not defect and flagged:
        return "FP"
    if defect and not flagged:
        return "FN"
    return "TN"


def calibrate_probe(probe: Probe,
                    cases: list[CalibrationCase],
                    flag_at: Severity = Severity.MEDIUM,
                    score_threshold: Optional[float] = None) -> ProbeCalibration:
    """Run ``probe`` over labelled ``cases`` and measure it as a diagnostic test.

    ``score_threshold`` (optional) switches the verdict from the severity rule to
    a score rule — used for probes whose severity is constant but whose numeric
    score discriminates (see ``verdict_from_findings``).
    """
    matrix = ConfusionMatrix()
    results: list[CaseResult] = []
    labels: list[bool] = []
    scores: list[float] = []
    # per-regime specificity is computed over clean (non-defect) cases only
    regime_tn: dict[str, int] = {}
    regime_n: dict[str, int] = {}

    for case in cases:
        try:
            findings = probe.run(case.artifact, case.model)
        except Exception as exc:  # a probe that crashes is a measurement event, not fatal
            findings = [Finding(probe.id, Severity.INFO, f"probe error: {exc}")]
        flagged = verdict_from_findings(findings, flag_at, score_threshold)
        score = score_from_findings(findings)
        top = max(findings, key=lambda f: severity_rank(f.severity)) if findings else None
        sev = top.severity.value if top else "info"
        summary = top.summary if top else "(no finding)"

        outcome = _outcome(case.defect, flagged)
        setattr(matrix, outcome.lower(), getattr(matrix, outcome.lower()) + 1)
        labels.append(case.defect)
        scores.append(score if score is not None else 0.0)

        if not case.defect:
            regime_n[case.regime] = regime_n.get(case.regime, 0) + 1
            if not flagged:
                regime_tn[case.regime] = regime_tn.get(case.regime, 0) + 1

        results.append(CaseResult(
            name=case.name, defect=case.defect, flagged=flagged, score=score,
            severity=sev, regime=case.regime, summary=summary, outcome=outcome))

    curves = _sweep(labels, scores)
    regime_spec = {
        reg: {"specificity": Metric.of(regime_tn.get(reg, 0), regime_n[reg]).to_dict(),
              "n": regime_n[reg]}
        for reg in sorted(regime_n)
    }
    return ProbeCalibration(
        probe_id=probe.id,
        probe_name=getattr(probe, "name", probe.id),
        requires_model=getattr(probe, "requires_model", False),
        flag_at=flag_at.value,
        matrix=matrix,
        curves=curves,
        cases=results,
        regime_specificity=regime_spec,
    )
