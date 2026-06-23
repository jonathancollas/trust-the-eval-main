"""Tests for the calibration engine — the credibility moat must not silently break."""
from __future__ import annotations

import trust_the_eval.probes  # noqa: F401  (self-registers all 20 probes)
from trust_the_eval.calibration import (
    ConfusionMatrix, build_cases, calibrate_probe, cases_from_mmlu_redux,
    curves_panel, pr_svg, roc_svg, run_calibration, score_threshold_for,
    summary_table, all_scenario_ids,
)
from trust_the_eval.calibration.core import score_from_findings, verdict_from_findings
from trust_the_eval.finding import Finding, Severity
from trust_the_eval.model.local import HonestModel
from trust_the_eval.probe import get_probe


# --------------------------------------------------------------------------
# The headline guarantee: every probe discriminates its planted defect.
# --------------------------------------------------------------------------
def test_every_probe_discriminates_its_planted_defect():
    """Recall and specificity must both be perfect on the planted scenarios.

    This is the empirical claim the whole credibility story rests on: each probe
    catches the defect it targets and stays quiet on clean artifacts.
    """
    ids = all_scenario_ids()
    assert len(ids) >= 20
    failures = []
    for pid in ids:
        probe = get_probe(pid)()
        cal = calibrate_probe(probe, build_cases(pid, seed=0),
                              flag_at=Severity.MEDIUM,
                              score_threshold=score_threshold_for(pid))
        rc = cal.matrix.recall().value or 0
        sp = cal.matrix.specificity().value or 0
        if rc < 0.999 or sp < 0.999:
            failures.append((pid, rc, sp,
                             f"{cal.matrix.tp}/{cal.matrix.fp}/{cal.matrix.fn}/{cal.matrix.tn}"))
    assert not failures, f"probes not discriminating perfectly: {failures}"


def test_pooled_metrics_are_perfect_on_planted_floor():
    rep = run_calibration(seed=0)
    p = rep.pooled
    assert p["precision"]["value"] == 1.0
    assert p["recall"]["value"] == 1.0
    assert p["specificity"]["value"] == 1.0
    assert p["probes_perfect"] == p["probes_total"] >= 20
    assert p["fp"] == 0 and p["fn"] == 0


# --------------------------------------------------------------------------
# Confusion-matrix math and Wilson intervals
# --------------------------------------------------------------------------
def test_confusion_matrix_metrics():
    m = ConfusionMatrix(tp=8, fp=2, fn=2, tn=8)
    assert abs(m.precision().value - 0.8) < 1e-9
    assert abs(m.recall().value - 0.8) < 1e-9
    assert abs(m.specificity().value - 0.8) < 1e-9
    assert abs(m.accuracy().value - 0.8) < 1e-9
    assert abs(m.f1() - 0.8) < 1e-9
    # Wilson interval brackets the point estimate and stays in [0,1]
    pr = m.precision()
    assert 0.0 <= pr.lo <= pr.value <= pr.hi <= 1.0


def test_metric_handles_empty_denominator():
    m = ConfusionMatrix(tp=0, fp=0, fn=0, tn=5)
    assert m.precision().value is None      # 0/0 -> undefined, not a crash
    assert m.specificity().value == 1.0


# --------------------------------------------------------------------------
# Verdict rules
# --------------------------------------------------------------------------
def test_severity_verdict_ignores_info():
    info = [Finding("x", Severity.INFO, "nothing to assess")]
    assert verdict_from_findings(info, Severity.MEDIUM) is False
    med = [Finding("x", Severity.MEDIUM, "defect", score=0.4)]
    assert verdict_from_findings(med, Severity.MEDIUM) is True


def test_score_threshold_verdict():
    low = [Finding("x", Severity.MEDIUM, "ok", score=0.04)]
    high = [Finding("x", Severity.MEDIUM, "bad", score=0.4)]
    assert verdict_from_findings(low, score_threshold=0.10) is False
    assert verdict_from_findings(high, score_threshold=0.10) is True
    # info-only never positive even with a score threshold
    info = [Finding("x", Severity.INFO, "n/a")]
    assert verdict_from_findings(info, score_threshold=0.0) is False


def test_score_from_findings_falls_back_to_severity():
    assert score_from_findings([]) == 0.0
    f = [Finding("x", Severity.HIGH, "h")]   # no numeric score
    assert score_from_findings(f) == 0.9


# --------------------------------------------------------------------------
# Curve orientation: inverted and non-monotonic scores are handled honestly
# --------------------------------------------------------------------------
def test_provenance_repro_curve_is_inverted_and_perfect():
    cal = calibrate_probe(get_probe("provenance_repro")(),
                          build_cases("provenance_repro", 0), Severity.MEDIUM,
                          score_threshold_for("provenance_repro"))
    # determinism score is HIGH when clean -> inverted orientation, AUC still 1.0
    assert cal.curves.orientation == "inverted"
    assert cal.curves.roc_auc == 1.0


def test_discrimination_saturation_curve_is_non_monotonic_and_perfect():
    cal = calibrate_probe(get_probe("discrimination_saturation")(),
                          build_cases("discrimination_saturation", 0), Severity.MEDIUM,
                          score_threshold_for("discrimination_saturation"))
    # ceiling(1.0) and floor(0.0) are both defects -> U-shaped score
    assert cal.curves.orientation == "non_monotonic"
    assert cal.curves.roc_auc == 1.0


# --------------------------------------------------------------------------
# Offline SVG rendering
# --------------------------------------------------------------------------
def test_curves_render_offline_svg():
    rep = run_calibration(seed=0)
    c = next(p.curves for p in rep.probes if p.curves.points)
    roc = roc_svg(c)
    pr = pr_svg(c)
    assert roc.startswith("<svg") and "AUC" in roc
    assert pr.startswith("<svg") and "AP" in pr
    assert "<script" not in roc and "<script" not in pr   # no JS, fully offline
    panel = curves_panel(c)
    assert "te-curves" in panel


def test_curves_panel_graceful_when_degenerate():
    from trust_the_eval.calibration.core import Curves
    assert "undefined" in curves_panel(Curves(points=[], orientation="degenerate"))


# --------------------------------------------------------------------------
# Real-world acid test: MMLU-Redux adapter (schema-faithful, offline fixture)
# --------------------------------------------------------------------------
def _mmlu_redux_fixture(n_ok=40, n_bad=12, seed=1):
    import random
    rng = random.Random(seed)
    rows = []
    for _ in range(n_ok):
        a, b = rng.randint(2, 20), rng.randint(2, 9); v = a + b
        opts = [str(v), str(v + 1), str(v + 2), str(v + 3)]; rng.shuffle(opts)
        rows.append({"question": f"What is {a} + {b}?", "choices": opts,
                     "answer": opts.index(str(v)), "error_type": "ok"})
    for _ in range(n_bad):
        a, b = rng.randint(2, 20), rng.randint(2, 9); v = a + b
        opts = [str(v), str(v + 1), str(v + 2), str(v + 3)]; rng.shuffle(opts)
        correct = opts.index(str(v))
        rows.append({"question": f"What is {a} + {b}?", "choices": opts,
                     "answer": (correct + 1) % 4, "error_type": "wrong_groundtruth"})
    return rows


def test_mmlu_redux_adapter_builds_labelled_cases():
    rows = _mmlu_redux_fixture()
    cases = cases_from_mmlu_redux(rows, group_size=20, defect_per_group=5,
                                  max_groups=6, seed=0)
    assert cases, "expected at least one calibration group"
    assert any(c.defect for c in cases) and any(not c.defect for c in cases)
    # defect groups must come from genuine wrong_groundtruth rows
    for c in cases:
        assert c.regime == "real_mmlu"


def test_label_error_audit_catches_real_mmlu_redux_errors():
    rows = _mmlu_redux_fixture(n_ok=80, n_bad=30)
    cases = cases_from_mmlu_redux(rows, group_size=20, defect_per_group=5,
                                  max_groups=8, seed=0)
    for c in cases:
        c.model = HonestModel(seed=7)   # solves arithmetic -> disagrees with wrong golds
    cal = calibrate_probe(get_probe("label_error_audit")(), cases, Severity.MEDIUM,
                          score_threshold_for("label_error_audit"))
    # on real-style labelled errors the probe should catch defects and clear clean groups
    assert (cal.matrix.recall().value or 0) >= 0.5
    assert (cal.matrix.specificity().value or 0) >= 0.5


# --------------------------------------------------------------------------
# Report shape + reproducibility
# --------------------------------------------------------------------------
def test_report_is_serialisable_and_reproducible():
    import json
    r1 = run_calibration(seed=0)
    r2 = run_calibration(seed=0)
    # deterministic metrics across runs (same seed)
    assert r1.pooled["precision"]["value"] == r2.pooled["precision"]["value"]
    blob = json.loads(r1.to_json())
    assert blob["report"] == "probe-calibration"
    assert len(blob["probes"]) >= 20
    assert "summary_table" not in blob  # sanity: structured, not text
    assert summary_table(r1)            # text table renders


# --------------------------------------------------------------------------
# A-completion: coverage taxonomy, ambiguity adapter, published report
# --------------------------------------------------------------------------
from trust_the_eval.calibration import (
    cases_from_mmlu_redux_ambiguity, coverage_summary, evidence_for,
    render_report_html, TIER_REAL, TIER_STRUCTURAL, TIER_BEHAVIORAL,
)


def test_coverage_taxonomy_covers_every_probe():
    cov = coverage_summary()
    total = sum(cov["counts"].values())
    assert total == cov["total"] >= 20
    # every scenario probe has an evidence classification
    for pid in all_scenario_ids():
        tier, source, basis = evidence_for(pid)
        assert tier in (TIER_REAL, TIER_STRUCTURAL, TIER_BEHAVIORAL)
        assert basis  # a non-empty honest basis note
    # the two real-data probes are present and have a source
    assert "label_error_audit" in cov["real_sources"]
    assert "item_ambiguity" in cov["real_sources"]


def test_ambiguity_adapter_catches_real_clarity_labels():
    from trust_the_eval.calibration.scenarios import _ScatterModel
    import random
    rng = random.Random(3)
    rows = []
    for _ in range(60):  # ok / well-posed
        a, b = rng.randint(2, 20), rng.randint(2, 9); v = a + b
        opts = [str(v), str(v + 1), str(v + 2), str(v + 3)]; rng.shuffle(opts)
        rows.append({"question": f"What is {a} + {b}?", "choices": opts,
                     "answer": opts.index(str(v)), "error_type": "ok"})
    for i in range(20):  # ill-posed / clarity-flagged
        rows.append({"question": f"Which is best, item {i}?",
                     "choices": ["red", "blue", "green", "yellow"],
                     "answer": 0, "error_type": "bad_question_clarity"})
    cases = cases_from_mmlu_redux_ambiguity(rows, group_size=20, defect_per_group=6,
                                            max_groups=8, seed=0)
    assert cases and any(c.defect for c in cases) and any(not c.defect for c in cases)
    for c in cases:
        assert c.regime == "real_mmlu_clarity"
        c.model = _ScatterModel(seed=5)
    cal = calibrate_probe(get_probe("item_ambiguity")(), cases, Severity.MEDIUM,
                          score_threshold_for("item_ambiguity"))
    assert (cal.matrix.recall().value or 0) >= 0.5
    assert (cal.matrix.specificity().value or 0) >= 0.5


def test_published_report_is_self_contained_offline():
    import re
    rep = run_calibration(seed=0)
    html = render_report_html(rep, caveats={c.probe_id: "test caveat" for c in rep.probes})
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()                       # no JS
    # no fetched external resources (xmlns namespace URIs are identifiers, allowed)
    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html)
    fetched = [u for u in re.findall(r'https?://[^\s"\'<>]+', html)
               if "www.w3.org" not in u]
    assert not external and not fetched
    assert html.count("<svg") >= 2                             # inline curves
    assert "Real-data coverage" in html
    assert "Limitations" in html
    assert "When not to trust" in html


def test_report_orders_real_tier_first():
    rep = run_calibration(seed=0)
    html = render_report_html(rep)
    # label_error_audit (real) should appear before a behavioral probe in coverage
    assert html.index("label_error_audit") < html.index("contamination_perturb")
