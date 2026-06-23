"""Evidence-tier taxonomy: on what kind of ground truth is each probe validated?

A calibration number is only as honest as the ground truth behind it. Not all
probes can be validated the same way, and pretending otherwise is exactly the
over-confidence this tool exists to catch. We therefore classify every probe
into one of three evidence tiers and state, plainly, the basis and its limits.

  TIER 1 — REAL human-annotated ground truth.
      A public dataset labels, per item, whether the defect is present.
      We measure the probe against those human labels. Strongest evidence.
      Today: label_error_audit and item_ambiguity, both via MMLU-Redux
      (Gema et al., 2024) — its error_type taxonomy labels wrong-groundtruth /
      no-correct-answer (label errors) and bad_question_clarity /
      bad_options_clarity (ambiguity).

  TIER 2 — STRUCTURAL / computable ground truth.
      The defect is a mathematical property of the artifact itself (sample size,
      number of configurations tried, score distribution, duplicates). The
      "truth" is therefore exact by construction, not a proxy: a 12-item eval
      really is underpowered; an eval with one category really is non-diverse.
      No external dataset is needed because the property is exact. Synthetic
      construction here is not a simplification — it is the ground truth.

  TIER 3 — BEHAVIORAL defect, no public per-item ground truth.
      The defect is a property of model BEHAVIOUR (contamination, sandbagging,
      under-elicitation, format brittleness, position bias, refusal-vs-inability,
      stochastic inconsistency, judge bias, drift, non-reproducibility,
      reward-hacking). Establishing per-item ground truth would require knowing
      the model's training corpus or internals, which is generally not public —
      a real labeled contamination set, for instance, does not publicly exist
      (the literature uses indirect signals: perplexity gaps, search logs). We
      therefore validate against controllable reference models that exhibit the
      defect BY CONSTRUCTION (a MemorizerModel is contaminated; a SandbaggerModel
      underperforms under eval cues). This is the honest floor: perfect internal
      validity, with external real-world validation pending datasets that, for
      most of these defects, do not yet exist.

This module is data, not logic; the report and UI read it to label each probe's
evidence honestly. Bright line preserved: every tier concerns the VALIDITY of an
eval result, never model safety.
"""
from __future__ import annotations

TIER_REAL = "real_labeled"
TIER_STRUCTURAL = "structural_exact"
TIER_BEHAVIORAL = "behavioral_synthetic"

TIER_LABEL = {
    TIER_REAL: "Real human-annotated ground truth",
    TIER_STRUCTURAL: "Structural / exact-by-construction ground truth",
    TIER_BEHAVIORAL: "Behavioral defect — synthetic floor (no public per-item ground truth)",
}

TIER_RANK = {TIER_REAL: 1, TIER_STRUCTURAL: 2, TIER_BEHAVIORAL: 3}

# probe_id -> (tier, real_source_or_None, honest_basis_note)
PROBE_EVIDENCE: dict[str, tuple] = {
    # --- TIER 1: real human labels ---
    "label_error_audit": (
        TIER_REAL, "MMLU-Redux (edinburgh-dawg/mmlu-redux-2.0)",
        "Validated against human error_type labels (wrong_groundtruth / "
        "no_correct_answer / multiple_correct_answers) on real MMLU items."),
    "item_ambiguity": (
        TIER_REAL, "MMLU-Redux (edinburgh-dawg/mmlu-redux-2.0)",
        "Validated against human clarity labels (bad_question_clarity / "
        "bad_options_clarity) marking genuinely ill-posed real MMLU items."),

    # --- TIER 2: structural / exact ---
    "statistical_power": (
        TIER_STRUCTURAL, None,
        "Defect = sample size / confidence-interval width; exact from the "
        "artifact, no external label needed."),
    "subgroup_power": (
        TIER_STRUCTURAL, None,
        "Defect = per-subgroup sample sizes; exact from the artifact."),
    "multiplicity_cherrypick": (
        TIER_STRUCTURAL, None,
        "Defect = number of configurations tried and best-vs-mean gap; exact "
        "from the recorded metadata."),
    "coverage_distribution": (
        TIER_STRUCTURAL, None,
        "Defect = category share concentration; exact from the item labels."),
    "discrimination_saturation": (
        TIER_STRUCTURAL, None,
        "Defect = ceiling/floor accuracy and extreme-score share; exact from "
        "the recorded scores."),
    "dataset_hygiene": (
        TIER_STRUCTURAL, None,
        "Defect = exact/near duplicates, empties, canary strings; exact from "
        "the items."),
    "answer_extraction_audit": (
        TIER_STRUCTURAL, None,
        "Defect = disagreement between the recorded score and a robust re-grade "
        "of the SAME stored responses; exact from the artifact."),

    # --- TIER 3: behavioral, synthetic floor ---
    "contamination_perturb": (
        TIER_BEHAVIORAL, None,
        "No public per-item contamination ground truth (would require the "
        "training corpus); validated against a MemorizerModel that is "
        "contaminated by construction. Real-world validation pending."),
    "sandbagging_paired": (
        TIER_BEHAVIORAL, None,
        "No public per-item sandbagging labels; validated against a "
        "SandbaggerModel that underperforms under eval cues by construction."),
    "elicitation_ceiling": (
        TIER_BEHAVIORAL, None,
        "No public elicitation-gap labels; validated against an "
        "ElicitationGatedModel that improves with scaffolding by construction."),
    "prompt_format_sensitivity": (
        TIER_BEHAVIORAL, None,
        "No public per-item format-brittleness labels; validated against a "
        "FormatSensitiveModel brittle to template by construction."),
    "option_order_bias": (
        TIER_BEHAVIORAL, None,
        "No public position-bias labels; validated against a model that always "
        "selects a fixed option slot by construction."),
    "refusal_confound": (
        TIER_BEHAVIORAL, None,
        "No public refusal-vs-inability labels; validated against a "
        "RefuserModel that declines triggered items by construction."),
    "self_consistency": (
        TIER_BEHAVIORAL, None,
        "No public stochastic-inconsistency labels; validated against a "
        "StochasticModel that scatters across resamples by construction."),
    "judge_swap": (
        TIER_BEHAVIORAL, None,
        "No public judge-bias labels; validated against a JudgeModel with "
        "position bias / self-preference by construction."),
    "model_drift": (
        TIER_BEHAVIORAL, None,
        "No public drift labels; validated against a DriftedModel whose answers "
        "differ from the recorded run by construction."),
    "provenance_repro": (
        TIER_BEHAVIORAL, None,
        "No public reproducibility labels; validated against a non-replayable "
        "model whose identical inputs yield different outputs by construction."),
    "reward_hacking_eval": (
        TIER_BEHAVIORAL, None,
        "Detection-only. No public gaming labels; validated against answer-leak "
        "items and an ExploiterModel that passes a lenient grader without "
        "solving, by construction. Never a recipe."),
}


def evidence_for(probe_id: str) -> tuple:
    """Return (tier, real_source, note) for a probe; defaults to behavioral."""
    return PROBE_EVIDENCE.get(
        probe_id,
        (TIER_BEHAVIORAL, None, "Validated against a controllable reference model."))


def coverage_summary() -> dict:
    """Counts per tier and the list of probes with a real-data source."""
    counts = {TIER_REAL: 0, TIER_STRUCTURAL: 0, TIER_BEHAVIORAL: 0}
    real_sources: dict[str, str] = {}
    for pid, (tier, source, _note) in PROBE_EVIDENCE.items():
        counts[tier] = counts.get(tier, 0) + 1
        if tier == TIER_REAL and source:
            real_sources[pid] = source
    return {
        "counts": counts,
        "labels": TIER_LABEL,
        "real_sources": real_sources,
        "total": len(PROBE_EVIDENCE),
    }
