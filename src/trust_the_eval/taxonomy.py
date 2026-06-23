"""Canonical claim taxonomy and remediation advice.

Single source of truth for:
- which CLAIM each probe bears on (capability / ranking / longitudinal /
  judge-scored). Editorial domain judgement, used for filtering and the
  per-claim verdict — never presented as an empirical mapping.
- the one-line, honest remediation shown when a probe fires.

The UI (src/trust_the_eval/ui/index.html) keeps a JS copy of both maps for
offline rendering; tests/test_taxonomy.py asserts the two stay identical.
"""
from __future__ import annotations

CLAIM_TYPES: dict[str, list[str]] = {
    "capability": [
        "contamination_perturb", "sandbagging_paired", "elicitation_ceiling",
        "refusal_confound", "label_error_audit", "item_ambiguity",
        "answer_extraction_audit", "reward_hacking_eval",
        "prompt_format_sensitivity", "dataset_hygiene",
    ],
    "ranking": [
        "discrimination_saturation", "statistical_power", "subgroup_power",
        "multiplicity_cherrypick", "option_order_bias", "coverage_distribution",
        "prompt_format_sensitivity",
    ],
    "longitudinal": ["model_drift", "provenance_repro", "self_consistency"],
    "judge-scored": ["judge_swap", "self_consistency"],
}

CLAIM_LABEL: dict[str, str] = {
    "capability": "capability",
    "ranking": "ranking",
    "longitudinal": "longitudinal",
    "judge-scored": "judge-scored",
}


def claims_of(probe_id: str) -> list[str]:
    return [c for c, ids in CLAIM_TYPES.items() if probe_id in ids]


REMEDIATION: dict[str, str] = {
    "dataset_hygiene": "Deduplicate, drop empty items, and remove any canary/leaked strings before scoring.",
    "contamination_perturb": "Re-test on held-out or freshly-authored items; report paraphrase-robust accuracy.",
    "label_error_audit": "Have the flagged items re-annotated; correct or remove wrong golds before reporting.",
    "item_ambiguity": "Rewrite or drop under-specified items; report on the well-posed subset.",
    "coverage_distribution": "Rebalance categories, or scope the claim to the categories actually covered.",
    "discrimination_saturation": "Move to a harder or dynamic benchmark; don't rank models on a saturated test.",
    "prompt_format_sensitivity": "Report across several equivalent templates, or fix and disclose one canonical format.",
    "option_order_bias": "Randomize option order across items and average; report the order-robust score.",
    "answer_extraction_audit": "Fix the answer extractor / use hybrid scoring, then re-score.",
    "sandbagging_paired": "Compare eval-framed vs neutral-framed performance; investigate the gap before trusting the score.",
    "elicitation_ceiling": "Report the elicited (best-scaffold) score, or disclose the elicitation method used.",
    "refusal_confound": "Separate refusals from genuine failures; report capability on attempted items.",
    "self_consistency": "Report at temperature 0 or aggregate multiple samples; disclose decoding settings.",
    "reward_hacking_eval": "Inspect flagged passes for gaming; tighten the grader so lenient matches don't count.",
    "judge_swap": "Swap/rotate judge and candidate positions and average; report inter-judge agreement (kappa).",
    "statistical_power": "Increase n or widen the reported confidence interval; don't compare scores within the margin.",
    "subgroup_power": "Aggregate thin subgroups or report per-subgroup CIs; avoid per-subgroup claims at low n.",
    "multiplicity_cherrypick": "Pre-register the config, or report all configs / a multiplicity-corrected result — not the best.",
    "model_drift": "Re-run the eval on the current model; date-stamp and version every reported score.",
    "provenance_repro": "Pin model version, decoding and seed; ship a provenance hash with the score.",
}
