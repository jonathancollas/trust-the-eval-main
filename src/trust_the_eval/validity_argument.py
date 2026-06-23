"""The validity argument, one row per probe.

This is the document that decides whether the tool does justice to measurement
validity (Messick 1995; Cronbach & Meehl 1955; Kane 2013; Borsboom et al. 2004).
For each probe it states the *complete chain*:

    data examined  ->  the validity threat it bears on (a NAMED Messick facet)
                   ->  the MEASURED reliability of its detection (+ evidence tier)
                   ->  what passing / failing actually licenses you to conclude.

Where that chain is complete and calibrated against real labels, the probe
honours the method. Where the reliability cell reads "synthetic floor", the
detection link is argued against *constructed* defects, not validated in the
field — the tool borrows the method's authority there, and that is the research
agenda, stated openly (Kane's argument-based validity: a validity argument is
judged by the strength of its weakest inference, not hidden).

Editorial note: each probe is assigned ONE dominant Messick facet; probes that
carry a second threat are tagged (dual: ...). The facet assignment is domain
judgement, not an empirical result. The reliability numbers and the evidence
tier are produced by the calibration engine, not asserted here.
"""
from __future__ import annotations

CU = "Construct underrepresentation"      # the test is too narrow / misses the construct
CIV = "Construct-irrelevant variance"     # the test captures systematic non-construct variance
REL = "Reliability (precondition)"        # precision / reproducibility — necessary, not sufficient
INF = "Inferential validity"              # the leap from a finite sample to the claim

FACET_ORDER = [CU, CIV, REL, INF]
FACET_GLOSS = {
    CU: "The score misses part of what it claims to measure.",
    CIV: "The score reflects systematic things that are NOT the construct.",
    REL: "Necessary but not sufficient: an unreliable measurement cannot be valid.",
    INF: "Whether the sample licenses the claim made from it (a model is better, a score is 81%).",
}

# probe_id -> (facet, dual_or_None, data_examined, pass_licenses, fail_signals)
CASES: dict[str, tuple] = {
    # ---- construct underrepresentation ----
    "coverage_distribution": (
        CU, None,
        "Distribution of items across categories/topics vs the breadth the score implies.",
        "Coverage is reasonably balanced; the score does not over-generalise from a narrow slice.",
        "Some categories are under-sampled — the test under-represents the construct; scope the claim to what is covered."),
    "discrimination_saturation": (
        CU, INF,
        "The score / difficulty distribution: ceiling effects and item discrimination.",
        "Headroom remains; the test still separates models on the construct.",
        "The test is saturated — at the ceiling it no longer represents or ranks the construct."),
    "elicitation_ceiling": (
        CU, None,
        "Accuracy under weak vs stronger elicitation / scaffolding.",
        "A stronger scaffold barely lifts the score; you are near the capability, not a floor.",
        "A better scaffold raises the score — you measured a floor; the construct is under-represented."),
    # ---- construct-irrelevant variance ----
    "contamination_perturb": (
        CIV, None,
        "Accuracy on original items vs meaning-preserving perturbations (paraphrase / reformat).",
        "No accuracy collapse under perturbation; little sign the score rides on memorised surface forms.",
        "Accuracy drops on perturbed items — the score partly reflects training-data overlap, not the construct."),
    "label_error_audit": (
        CIV, None,
        "Gold answers flagged as likely wrong (validated against real MMLU-Redux wrong-key labels).",
        "No systematic key errors surfaced; the scoring criterion is not visibly corrupted.",
        "Likely-wrong gold answers found — a corrupted key mis-scores ability regardless of the model."),
    "item_ambiguity": (
        CIV, CU,
        "Items flagged under-specified / ambiguous (validated against real MMLU-Redux ambiguity labels).",
        "Items appear well-posed; little irrelevant variance from ambiguity.",
        "Ambiguous items present — Messick: flawed items elicit construct-irrelevant variance and blur what is measured."),
    "prompt_format_sensitivity": (
        CIV, None,
        "Accuracy across several equivalent prompt templates.",
        "Score is stable across formats; little irrelevant variance from presentation.",
        "Score swings with format — textbook construct-irrelevant variance; report across templates or fix and disclose one."),
    "option_order_bias": (
        CIV, None,
        "Accuracy across permutations of multiple-choice option order.",
        "No order effect; choice position is not driving the score.",
        "Score depends on option order — irrelevant variance; randomise order and average."),
    "answer_extraction_audit": (
        CIV, None,
        "Extracted answers vs raw model responses (parser / scorer mismatches).",
        "Extraction matches responses; the response->score mapping adds little irrelevant variance.",
        "The extractor mis-scores correct/incorrect responses — irrelevant variance injected at scoring."),
    "sandbagging_paired": (
        CIV, CU,
        "Accuracy under eval-framed vs neutral-framed presentation (paired).",
        "No framing gap; little sign of measurement reactivity (observer effect).",
        "Performance shifts when the model 'knows' it is being tested — reactivity contaminates the score."),
    "refusal_confound": (
        CIV, None,
        "Refusals vs genuine attempts among the failing items.",
        "Failures are genuine attempts; refusals are not masquerading as incapacity.",
        "Refusals are scored as failures — irrelevant variance; separate refusal from inability."),
    "reward_hacking_eval": (
        CIV, None,
        "Passing items showing shortcut / leak / gaming patterns (detection only).",
        "No gaming patterns in the passes; passing plausibly reflects the construct.",
        "Some passes show gaming / leak shortcuts — irrelevant variance; tighten the grader. (Flag only — never an exploit.)"),
    "dataset_hygiene": (
        CIV, REL,
        "The item set: exact duplicates, empty items, leaked / canary strings.",
        "No structural artefacts detected; the score is not inflated by duplication or leakage on this axis.",
        "Duplicates / empties / canaries present — they inject non-construct variance and can inflate the score."),
    # ---- reliability (precondition) ----
    "self_consistency": (
        REL, None,
        "Agreement of the model's own answers across repeated samples.",
        "Answers are stable across samples; the measurement is reliable at this temperature.",
        "Answers vary run-to-run — low reliability; report at temperature 0 or aggregate samples."),
    "judge_swap": (
        REL, CIV,
        "Scores under swapped / rotated judge and candidate position.",
        "Verdicts are stable across judge configuration; inter-rater reliability holds.",
        "The judge's verdict flips with position / identity — scorer-induced variance; rotate and report agreement (kappa)."),
    "model_drift": (
        REL, None,
        "A current re-run vs the artifact's recorded scores; version / date stamps.",
        "Current behaviour matches the recorded score; the measured object has not drifted.",
        "The model's behaviour has changed since the artifact — the score is stale; re-run and date-stamp."),
    "provenance_repro": (
        REL, None,
        "Presence of version / decoding / seed metadata and whether a re-run reproduces.",
        "Enough provenance to reproduce; the measurement is repeatable.",
        "Missing provenance / non-reproducible — the score cannot be independently repeated; pin version, decoding, seed."),
    # ---- inferential validity ----
    "statistical_power": (
        INF, REL,
        "n and the score -> Wilson interval and the minimum significant gap.",
        "The estimate is precise enough for the intended inference at this n.",
        "The interval is wide — comparisons within the margin are noise; raise n or report the CI."),
    "subgroup_power": (
        INF, None,
        "Per-subgroup n and the precision of any per-subgroup claim.",
        "Subgroups are large enough to support the per-subgroup statements made.",
        "Thin subgroups — per-subgroup claims are unsupported; aggregate or report per-cell CIs."),
    "multiplicity_cherrypick": (
        INF, None,
        "Number of configs / variants run vs the single result reported (forking paths).",
        "No sign of selection among many configurations; the reported result is not cherry-picked.",
        "Many configs, best one reported — researcher degrees of freedom inflate false discovery; pre-register or correct."),
}

_TIER_TRUTH = {
    "real_labeled": "**Real labels.** Detection validated against human-labelled errors (MMLU-Redux). R/S/P = {r:.2f}/{s:.2f}/{p:.2f} on that real set.",
    "structural_exact": "**Exact.** Detection is a deterministic rule, not a sampled estimate — R/S/P = {r:.2f}/{s:.2f}/{p:.2f} by construction.",
    "behavioral_synthetic": "**Synthetic floor.** R/S/P = {r:.2f}/{s:.2f}/{p:.2f} against *constructed* defects only; real-world precision unproven.",
}
_TIER_RANK = {"real_labeled": 0, "structural_exact": 1, "behavioral_synthetic": 2}


def render_validity_argument_md(seed: int = 0) -> str:
    import trust_the_eval.probes  # noqa: F401  (registers probes)
    from .probe import all_probes
    from .emit.trust_report import calibration_index

    cal = calibration_index(seed=seed)
    names = {p.id: p.name for p in all_probes()}
    requires_model = {p.id: p.requires_model for p in all_probes()}

    tier_counts: dict[str, int] = {}
    for c in cal.values():
        tier_counts[c["tier"]] = tier_counts.get(c["tier"], 0) + 1

    out = []
    out.append("# Trust the Eval — the validity argument, probe by probe\n")
    out.append(
        "Each probe attacks the **validity of an eval result as a measurement** — never the "
        "safety of the model. This table states, for every probe, the full chain a validity "
        "argument requires: *what data is examined → which named validity threat it bears on "
        "(Messick) → the **measured** reliability of detection (with its evidence tier) → what "
        "passing or failing licenses you to conclude*.\n")
    out.append(
        "> **How to read the reliability column (the honest part).** It is the *strength of the "
        "argument*, following Kane (2013): a validity argument is only as strong as its weakest "
        "inference. **Real labels** and **Exact** are strong. **Synthetic floor** means the "
        "detection link is argued against defects we constructed, not validated in the field — "
        f"the tool's research agenda, stated openly. Tiers on this build: "
        f"{tier_counts.get('real_labeled',0)} real-labelled · "
        f"{tier_counts.get('structural_exact',0)} exact · "
        f"{tier_counts.get('behavioral_synthetic',0)} synthetic floor.\n")
    out.append(
        "> **\u201cSupportable\u201d \u2260 \u201cproven.\u201d** A probe that does not fire means *that* "
        "threat was not detected, not that the score is valid. No single trustworthiness grade is "
        "produced — validity is multi-dimensional and non-collapsible (Messick).\n")

    # facet -> rows
    by_facet: dict[str, list[str]] = {f: [] for f in FACET_ORDER}
    for pid, (facet, dual, examined, ok, bad) in CASES.items():
        c = cal.get(pid, {})
        tier = c.get("tier", "?")
        rel = _TIER_TRUTH.get(tier, "not calibrated").format(
            r=c.get("recall", 0), s=c.get("specificity", 0), p=c.get("precision", 0))
        dual_tag = f" <br>_(dual: {dual})_" if dual else ""
        mdep = "needs model" if requires_model.get(pid) else "static"
        name = names.get(pid, pid)
        row = (f"| **{name}**<br>`{pid}` · {mdep}{dual_tag} "
               f"| {examined} "
               f"| {rel} "
               f"| **Pass:** {ok}<br>**Fail:** {bad} |")
        by_facet[facet].append((tier, row))

    for facet in FACET_ORDER:
        rows = by_facet[facet]
        if not rows:
            continue
        rows.sort(key=lambda tr: _TIER_RANK.get(tr[0], 9))
        out.append(f"\n## {facet}\n")
        out.append(f"*{FACET_GLOSS[facet]}*\n")
        out.append("| Probe | Data examined | Detection reliability | What pass / fail licenses |")
        out.append("|---|---|---|---|")
        out.extend(r for _, r in rows)

    out.append("\n---\n")
    out.append(
        "**The reading.** The four sections above are not an arbitrary checklist: they are the "
        "complete threat space of construct validity — the test missing the construct (under-"
        "representation), the test capturing non-construct signal (irrelevant variance), the "
        "measurement being unreliable (precondition), and the inference outrunning the sample "
        "(inferential validity). That the 20 probes *fill* this space, and only this space, is the "
        "sense in which the tool does justice to the method — the taxonomy is reconstructible from "
        "the theory, not decreed.\n")
    out.append(
        "**The limit.** The tool audits validity *given* a construct; it cannot tell you whether the "
        "construct itself is meaningful (Borsboom et al. 2004: does the attribute exist and causally "
        "produce the scores?). Consequential validity — whether using the eval leads to good "
        "decisions — is out of scope by design (validity, not safety). And the detection links marked "
        "*synthetic floor* are argued, not yet field-validated. Those three are the honest ceiling, "
        "and naming them is itself the method (Kane).\n")
    return "\n".join(out)
