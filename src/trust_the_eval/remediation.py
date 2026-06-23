"""Remediation guidance — turn a benchmark's audit findings into a prioritised,
*calibrated* list of fixes.

The discipline is the same one the whole project holds: every output is either
something the probe **reliably measures** (a confident instrument fix), a
**flag for human review**, a **relay** of a single-pass annotation (a candidate
to verify, never asserted), an **investigate** prompt for a weak/defeatable
signal, or an **informational** caveat. We never assert ground truth — in
particular we never tell an author "the correct answer is X" off our own
statistics, because we measured that signal to be near-chance.

Priorisation uses result_sensitivity: a fix is flagged by whether correcting it
would actually move the benchmark's verdict (the model ranking) or only its
absolute scores. Authors get told what matters, honestly.
"""
from typing import Any, Dict, List, Optional

# the 20 probes (audit instruments); kept explicit so the registry is provably
# complete (see tests/test_remediation.py).
PROBE_IDS = frozenset({
    "answer_extraction_audit", "contamination_perturb", "coverage_distribution",
    "dataset_hygiene", "discrimination_saturation", "elicitation_ceiling",
    "item_ambiguity", "judge_swap", "label_error_audit", "model_drift",
    "multiplicity_cherrypick", "option_order_bias", "prompt_format_sensitivity",
    "provenance_repro", "refusal_confound", "reward_hacking_eval",
    "sandbagging_paired", "self_consistency", "statistical_power",
    "subgroup_power",
})

# confidence tiers, from "safe to act on" to "do not over-read"
TIER_CAPTION = {
    "instrument_fix": ("High confidence", "a structural property of the "
                       "instrument the probe measures directly — safe to act on."),
    "review": ("Medium — review", "the probe flags candidates; the resolution "
               "needs human judgement."),
    "relay_only": ("Relay only", "surfaced from a single-pass annotation — a "
                   "candidate to verify with a second annotator, not a "
                   "determination. Do not change the answer key on this alone."),
    "investigate": ("Weak signal", "detection here is near-chance or defeatable; "
                    "treat as a prompt to investigate (ideally via temporal / "
                    "held-out data), not a verdict."),
    "informational": ("Informational", "affects how to interpret or report the "
                      "score, not a defect to fix."),
}

# probe_id -> (tier, action, rationale)
REMEDIATION: Dict[str, Any] = {
    "answer_extraction_audit": ("instrument_fix",
        "Fix answer extraction / scoring: some responses are mis-parsed "
        "(wrong normalisation or an under-specified answer format).",
        "Mis-scoring is deterministic — the responses exist but are graded wrong."),
    "contamination_perturb": ("investigate",
        "Check contamination with TEMPORAL / held-out items before adjusting any "
        "score; do not rescore on this static signal.",
        "Static-benchmark contamination detection barely beats chance "
        "(MIA ROC-AUC < 0.6); only time-split signals are reliable."),
    "coverage_distribution": ("instrument_fix",
        "Rebalance the test: the labelled categories and/or the answer key are "
        "skewed; add items to under-represented cells or rebalance the key.",
        "Category / key balance is measured directly and is structurally fixable."),
    "dataset_hygiene": ("instrument_fix",
        "Clean the dataset: remove exact / near-duplicate items, resolve "
        "contradictory labels, drop empty items, investigate any canary hits.",
        "Duplicates, empties and contradictions are detected exactly and are "
        "unambiguous to fix."),
    "discrimination_saturation": ("informational",
        "Test-quality note: many items are saturated (all-correct) or "
        "non-discriminating; consider harder / replacement items. Do NOT use low "
        "discrimination as a label-error screen — we found that unreliable.",
        "Discrimination measures item informativeness, not correctness; it is "
        "near-chance as a label-error detector."),
    "elicitation_ceiling": ("informational",
        "Scores may be capped by elicitation (prompt / format / length), not by "
        "capability; strengthen elicitation and re-measure before concluding.",
        "A ceiling can reflect extraction effort rather than the construct."),
    "item_ambiguity": ("review",
        "Send the flagged ambiguous items to multi-annotator re-review and "
        "clarify the wording / options.",
        "Ambiguity is a reliable flag-for-review, but the resolution needs a human."),
    "judge_swap": ("instrument_fix",
        "Harden the LLM-judge: randomise option / answer order, use multiple "
        "independent judges, and report inter-judge agreement (kappa).",
        "Position bias / low judge agreement is a measurable methodology defect."),
    "label_error_audit": ("relay_only",
        "Candidate label corrections (from a single-pass human annotation) are "
        "attached for the flagged items — verify each with a second annotator "
        "before changing the answer key. Do not change the key on this alone.",
        "We measured statistical label-error detection to be near-chance and "
        "these corrections are single-pass; asserting them would exceed our "
        "validated reliability."),
    "model_drift": ("informational",
        "Temporal-validity caveat: scores drift across model versions / time; pin "
        "model snapshots and evaluation dates, and avoid comparing across drift.",
        "Drift is about when / which model, not a dataset defect."),
    "multiplicity_cherrypick": ("instrument_fix",
        "Apply a multiplicity correction (or pre-register a single primary "
        "metric); reported gains may be selection artefacts across many "
        "comparisons.",
        "Uncorrected multiple comparisons inflate apparent effects — fixable."),
    "option_order_bias": ("instrument_fix",
        "Permute MCQ option order across runs and average; the score currently "
        "depends on option position.",
        "Order sensitivity is measurable and removable by randomisation."),
    "prompt_format_sensitivity": ("instrument_fix",
        "Pin a prompt-format spec and report sensitivity across formats; the "
        "score swings with formatting.",
        "Format sensitivity is a measurable, controllable instrument property."),
    "provenance_repro": ("instrument_fix",
        "Restore provenance / reproducibility: pin the dataset version, record "
        "sources, and fix seeds so the run reproduces.",
        "Missing provenance is a concrete, fixable gap."),
    "refusal_confound": ("instrument_fix",
        "Separate refusals / abstentions from incorrect answers in scoring; "
        "refusals currently confound the capability estimate.",
        "Conflating refusal with error is a scoring defect that biases the score."),
    "reward_hacking_eval": ("investigate",
        "Inspect responses for metric-gaming (format exploits, shortcut tokens) "
        "and harden the scoring rule.",
        "Possible eval gaming; confirm by inspection before trusting the metric."),
    "sandbagging_paired": ("investigate",
        "Possible strategic underperformance; detection is nascent and "
        "defeatable, so treat as a flag for paired / held-out investigation, not "
        "a verdict.",
        "Sandbagging detection has no settled SOTA and adapts to monitoring."),
    "self_consistency": ("instrument_fix",
        "Stabilise decoding (fix temperature / seed) or report run-to-run "
        "confidence intervals; the score has high stochastic variance.",
        "Run-to-run instability is measurable and reducible."),
    "statistical_power": ("informational",
        "Precision note: too few items for the claimed precision — add items or "
        "widen the reported confidence interval.",
        "Under-powered estimates need more data or honester CIs, not a key change."),
    "subgroup_power": ("informational",
        "Do not report per-subgroup rankings at this sample size; subgroup claims "
        "are under-powered.",
        "Subgroup precision is insufficient; this caps the claims, not the data."),
    # audits that ride in records but are not in the 20-probe set:
    "test_reliability": ("informational",
        "Low internal consistency (Cronbach alpha): the subtest may be "
        "heterogeneous or contain weak items; revisit item selection or split the "
        "construct.",
        "Reliability bounds interpretability; it is a diagnostic, not a per-item fix."),
}

_SEV_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}
# result_sensitivity is the prioritiser, never itself a remediation action
_NON_ACTION = {"result_sensitivity"}


def _n_items(measured: Dict[str, Any]) -> Optional[int]:
    for k in ("k", "n_flagged", "count", "n_items", "n_suspect"):
        v = measured.get(k)
        if isinstance(v, int):
            return v
    return None


def _detail(probe_id: str, measured: Dict[str, Any]) -> str:
    """A short, specific suffix from the probe's own measured dict."""
    if not measured:
        return ""
    if probe_id == "dataset_hygiene":
        bits = [(k, measured.get(k)) for k in
                ("exact_dups", "near_dups", "contradictory_labels", "empty",
                 "canary_hits") if measured.get(k)]
        if bits:
            return " [" + ", ".join("%s=%s" % (k, v) for k, v in bits) + "]"
    if probe_id == "coverage_distribution":
        tc, ts = measured.get("top_category"), measured.get("top_share")
        if tc is not None and isinstance(ts, (int, float)):
            return " [top '%s' = %.0f%%]" % (tc, ts * 100)
    return ""


def _verdict_impact(sensitivity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """From a result_sensitivity audit: would fixing change the ranking?"""
    if not sensitivity:
        return {"known": False, "ranking_stable": None, "models_moved": None,
                "note": "No result-sensitivity measured: cannot say whether "
                        "fixing changes the verdict."}
    stable = sensitivity.get("ranking_stable")
    moved = sensitivity.get("models_moved")
    if stable is None:
        moved_n = sensitivity.get("n_changed_items")
        stable = (moved_n == 0) if isinstance(moved_n, int) else None
    if stable is True:
        note = ("Fixing these will correct absolute scores and may reshuffle "
                "near-tied pairs, but does NOT change the model ranking.")
    elif stable is False:
        note = ("Fixing these CAN change the model ranking — prioritise.")
    else:
        note = "Result-sensitivity inconclusive on ranking impact."
    return {"known": True, "ranking_stable": stable, "models_moved": moved,
            "note": note}


def remediation_plan(audits: List[Dict[str, Any]],
                     sensitivity: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Build a prioritised, calibrated remediation plan from a benchmark's audits.

    Each fired probe (severity above 'info') becomes one action, tagged with a
    confidence tier and an honesty caveat. Sorted by verdict impact, then
    severity, then tier (confident fixes first, weak signals last).
    """
    # one entry per probe; keep the highest-severity instance + its measured
    best: Dict[str, Dict[str, Any]] = {}
    for a in audits:
        pid = a.get("probe_id")
        if pid in _NON_ACTION or pid not in REMEDIATION:
            continue
        if _SEV_RANK.get(a.get("severity", "info"), 0) <= 0:
            continue  # 'info' -> nothing to remediate
        prev = best.get(pid)
        if prev is None or _SEV_RANK.get(a.get("severity"), 0) > _SEV_RANK.get(
                prev.get("severity"), 0):
            best[pid] = a

    impact = _verdict_impact(sensitivity)
    _tier_order = {"instrument_fix": 0, "review": 1, "relay_only": 2,
                   "investigate": 3, "informational": 4}
    actions: List[Dict[str, Any]] = []
    for pid, a in best.items():
        tier, action, rationale = REMEDIATION[pid]
        measured = a.get("measured") or {}
        cap_short, cap_long = TIER_CAPTION[tier]
        actions.append({
            "probe_id": pid,
            "severity": a.get("severity"),
            "tier": tier,
            "confidence": cap_short,
            "action": action + _detail(pid, measured),
            "rationale": rationale,
            "caveat": cap_long,
            "n_items_affected": _n_items(measured),
            "candidates_available": (tier == "relay_only"),
            "asserts_correction": False,  # invariant: we never assert ground truth
        })

    # ranking-relevant fixes first when we know the impact, then severity, then tier
    rank_first = 0 if impact.get("ranking_stable") is False else 1
    actions.sort(key=lambda x: (
        rank_first if x["tier"] in ("instrument_fix", "relay_only", "review") else 1,
        -_SEV_RANK.get(x["severity"], 0),
        _tier_order[x["tier"]],
        x["probe_id"],
    ))

    n_high = sum(1 for x in actions if x["tier"] == "instrument_fix")
    n_relay = sum(1 for x in actions if x["tier"] == "relay_only")
    return {
        "verdict_impact": impact,
        "actions": actions,
        "n_actions": len(actions),
        "n_instrument_fix": n_high,
        "n_relay_only": n_relay,
        "summary": summary_line(actions, impact),
    }


def summary_line(actions: List[Dict[str, Any]], impact: Dict[str, Any]) -> str:
    if not actions:
        return "No remediation needed: no audit fired above 'info'."
    n_high = sum(1 for x in actions if x["tier"] == "instrument_fix")
    n_relay = sum(1 for x in actions if x["tier"] == "relay_only")
    n_inv = sum(1 for x in actions if x["tier"] == "investigate")
    parts = ["%d action(s)" % len(actions)]
    if n_high:
        parts.append("%d high-confidence instrument fix(es)" % n_high)
    if n_relay:
        parts.append("%d relay-only candidate(s) to verify" % n_relay)
    if n_inv:
        parts.append("%d weak-signal investigation(s)" % n_inv)
    rank = impact.get("ranking_stable")
    tail = ("; ranking is stable to these fixes" if rank is True
            else "; these can change the ranking" if rank is False
            else "")
    return ", ".join(parts) + tail


def remediation_for_benchmark(store: Any, benchmark: str) -> Dict[str, Any]:
    """Gather all audits for a benchmark across its records + its
    result_sensitivity, and build the plan."""
    all_audits: List[Dict[str, Any]] = []
    sens: Optional[Dict[str, Any]] = None
    for rid in store.all_ids():
        d = store.get(rid)
        if d["subject"]["benchmark"] != benchmark:
            continue
        for a in d["audits"]:
            all_audits.append(a)
            if a.get("probe_id") == "result_sensitivity" and a.get("measured"):
                sens = a["measured"]
    return remediation_plan(all_audits, sensitivity=sens)
