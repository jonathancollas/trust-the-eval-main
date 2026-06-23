"""Calibration report runner: measure every probe and assemble the report.

Runs each probe over its planted-defect scenarios (and, when provided, real
known-bad cases), then assembles a structured report with per-probe diagnostic
metrics, ROC/PR curves, per-regime specificity, and pooled aggregates. This is
the data behind the published "here is the measured precision/recall of every
probe" artifact.

Offline and dependency-free. The SVG curves (``curves_svg``) and the
real-dataset cases (``realworld``) are produced by sibling modules and folded
in here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from ..finding import Severity
from ..probe import all_probes, get_probe
from ..stats import wilson_ci
from .core import ProbeCalibration, calibrate_probe
from .scenarios import all_scenario_ids, build_cases, has_scenarios, score_threshold_for


@dataclass
class CalibrationReport:
    generated_at: str
    seed: int
    probes: list[ProbeCalibration]
    pooled: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": "trust-the-eval",
            "report": "probe-calibration",
            "generated_at": self.generated_at,
            "seed": self.seed,
            "meta": self.meta,
            "pooled": self.pooled,
            "probes": [p.to_dict() for p in self.probes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _pooled(cals: list[ProbeCalibration]) -> dict:
    """Pool confusion counts across all probes for a headline P/R/specificity."""
    tp = sum(c.matrix.tp for c in cals)
    fp = sum(c.matrix.fp for c in cals)
    fn = sum(c.matrix.fn for c in cals)
    tn = sum(c.matrix.tn for c in cals)

    def metric(k, n):
        if n <= 0:
            return {"value": None, "lo": None, "hi": None, "k": k, "n": n}
        lo, hi = wilson_ci(k, n)
        return {"value": k / n, "lo": lo, "hi": hi, "k": k, "n": n}

    aucs = [c.curves.roc_auc for c in cals if c.curves.roc_auc is not None]
    aps = [c.curves.average_precision for c in cals
           if c.curves.average_precision is not None]
    n_perfect = sum(1 for c in cals
                    if (c.matrix.recall().value or 0) >= 0.999
                    and (c.matrix.specificity().value or 0) >= 0.999)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": metric(tp, tp + fp),
        "recall": metric(tp, tp + fn),
        "specificity": metric(tn, tn + fp),
        "accuracy": metric(tp + tn, tp + fp + fn + tn),
        "macro_roc_auc": sum(aucs) / len(aucs) if aucs else None,
        "macro_average_precision": sum(aps) / len(aps) if aps else None,
        "probes_total": len(cals),
        "probes_perfect": n_perfect,
    }


def run_calibration(seed: int = 0,
                    probe_ids: Optional[list[str]] = None,
                    extra_cases: Optional[dict] = None,
                    flag_at: Severity = Severity.MEDIUM) -> CalibrationReport:
    """Calibrate probes that have scenarios and assemble the report.

    ``extra_cases`` maps probe_id -> list[CalibrationCase] to APPEND to the
    planted scenarios (e.g. real MMLU-Redux cases for label_error_audit). This
    lets the same report carry both the synthetic floor and the real acid test.
    """
    extra_cases = extra_cases or {}
    ids = probe_ids or all_scenario_ids()
    cals: list[ProbeCalibration] = []
    errors: dict = {}
    for pid in ids:
        if not has_scenarios(pid) and pid not in extra_cases:
            continue
        try:
            probe = get_probe(pid)()
            cases = build_cases(pid, seed) + list(extra_cases.get(pid, []))
            cal = calibrate_probe(probe, cases, flag_at=flag_at,
                                  score_threshold=score_threshold_for(pid))
            cals.append(cal)
        except KeyError:
            continue
        except Exception as exc:  # one bad probe (cases or calibration) must not sink the report
            errors[pid] = f"{type(exc).__name__}: {exc}"

    cals.sort(key=lambda c: c.probe_id)
    report = CalibrationReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seed=seed,
        probes=cals,
        pooled=_pooled(cals),
        meta={
            "errors": errors,
            "verdict_rule": f"severity >= {flag_at.value} (or per-probe score threshold)",
            "ground_truth": "planted synthetic defects (controllable models + datagen)"
                            + ("; plus real cases" if extra_cases else ""),
            "note": "Synthetic scenarios are the controlled floor; real known-bad "
                    "datasets (e.g. MMLU-Redux) are the acid test when supplied.",
        },
    )
    return report


def summary_table(report: CalibrationReport) -> str:
    """A compact text table of per-probe metrics for console output."""
    lines = []
    head = f"{'probe':30} {'kind':6} {'prec':>6} {'recall':>7} {'spec':>6} {'AUC':>6} {'AP':>6}"
    lines.append(head)
    lines.append("-" * len(head))
    for c in report.probes:
        m = c.matrix
        kind = "model" if c.requires_model else "static"

        def f(metric_or_val):
            v = metric_or_val.value if hasattr(metric_or_val, "value") else metric_or_val
            return f"{v:.2f}" if isinstance(v, (int, float)) else "   - "
        lines.append(
            f"{c.probe_id:30} {kind:6} {f(m.precision()):>6} {f(m.recall()):>7} "
            f"{f(m.specificity()):>6} {f(c.curves.roc_auc):>6} "
            f"{f(c.curves.average_precision):>6}")
    p = report.pooled
    lines.append("-" * len(head))

    def pf(d):
        v = d.get("value")
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
    lines.append(f"POOLED  precision={pf(p['precision'])}  recall={pf(p['recall'])}  "
                 f"specificity={pf(p['specificity'])}  "
                 f"(TP={p['tp']} FP={p['fp']} FN={p['fn']} TN={p['tn']})")
    lines.append(f"probes perfect (R=Spec=1.0): {p['probes_perfect']}/{p['probes_total']}  "
                 f"macro ROC-AUC="
                 f"{p['macro_roc_auc']:.3f}" if p['macro_roc_auc'] else "")
    return "\n".join(lines)
