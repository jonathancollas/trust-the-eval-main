"""trust_the_eval.calibration — measure each probe as a diagnostic test.

Public API:
    run_calibration(seed=, extra_cases=) -> CalibrationReport
    summary_table(report) -> str
    calibrate_probe(probe, cases, ...) -> ProbeCalibration
    build_cases(probe_id, seed) -> [CalibrationCase]          (planted scenarios)
    cases_from_mmlu_redux(rows, ...) -> [CalibrationCase]     (real acid test)
    roc_svg / pr_svg / curves_panel                            (offline SVG)

The synthetic scenarios are the controlled floor (perfect ground truth); real
known-bad datasets (MMLU-Redux) are the acid test. Both measure whether a probe
correctly detects a VALIDITY defect in an eval artifact — never model safety.
"""
from .core import (
    CalibrationCase, CaseResult, ConfusionMatrix, Curves, Metric, ProbeCalibration,
    calibrate_probe, score_from_findings, verdict_from_findings,
)
from .curves_svg import curves_panel, pr_svg, roc_svg
from .coverage import (
    PROBE_EVIDENCE, TIER_BEHAVIORAL, TIER_LABEL, TIER_REAL, TIER_STRUCTURAL,
    coverage_summary, evidence_for,
)
from .realworld import (
    AMBIGUITY_ERROR_TYPES, DEFECT_ERROR_TYPES, cases_from_mmlu_redux,
    cases_from_mmlu_redux_ambiguity, load_mmlu_redux_rows,
)
from .report_html import render_report_html
from .runner import CalibrationReport, run_calibration, summary_table
from .scenarios import (
    all_scenario_ids, build_cases, has_scenarios, score_threshold_for,
)

__all__ = [
    "CalibrationCase", "CaseResult", "ConfusionMatrix", "Curves", "Metric",
    "ProbeCalibration", "calibrate_probe", "score_from_findings",
    "verdict_from_findings", "curves_panel", "pr_svg", "roc_svg",
    "cases_from_mmlu_redux", "cases_from_mmlu_redux_ambiguity",
    "load_mmlu_redux_rows", "DEFECT_ERROR_TYPES", "PROBE_EVIDENCE",
    "TIER_REAL", "TIER_STRUCTURAL", "TIER_BEHAVIORAL", "TIER_LABEL",
    "coverage_summary", "evidence_for",
    "AMBIGUITY_ERROR_TYPES", "CalibrationReport", "run_calibration",
    "summary_table", "all_scenario_ids", "build_cases", "has_scenarios",
    "score_threshold_for", "render_report_html",
]
