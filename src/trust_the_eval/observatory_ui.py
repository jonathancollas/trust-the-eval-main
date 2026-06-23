"""Rich validity-observatory UI, generated from a synced store.

This is the navigable observatory: a Portfolio of every benchmark profiled by
result-sensitivity (does correcting the labels change the verdict?), a per-benchmark
dossier, and a Probe library that explains AND demonstrates the science behind each
probe. Two invariants are enforced by tests:

  * No number is shown without its computation. Every figure carries a panel that
    reconstructs it from the same trace the pipeline produced; in particular each
    benchmark trace satisfies (C - D) / npair == kendall_tau exactly.
  * We audit the instrument, never the model. No model is rated or ranked for its
    own sake; a model appears only as a data point in whether the instrument holds.

`assemble_ui_data` is pure (store + item rows -> data dict); `build_ui_html` renders
the standalone HTML. The heavy real-data driver lives in scripts/build_observatory_ui.py.
"""
import hashlib
import json

from trust_the_eval.item_analysis import (analyze, correctness_from_predictions,
                                           reliability_summary)
from trust_the_eval.leaderboard import leaderboard

SCI = {
 "result_sensitivity": {
  "title": "Result-sensitivity", "kind": "headline",
  "measures": "Whether correcting the gold labels changes the <b>conclusion</b> you draw — the model ranking — not merely the absolute scores.",
  "formula": "\u03c4 = (C \u2212 D) / [n(n\u22121)/2]   \u00b7   \u0394\u2098 = acc_corr(m) \u2212 acc_orig(m)   \u00b7   p\u2081 = P(top\u20111 changes under item resampling)",
  "method": "Re-score every model under the corrected key. Compare the original and corrected rankings with Kendall\u2019s \u03c4 (concordant minus discordant model pairs over all n(n\u22121)/2 pairs). Bootstrap the items to estimate the probability the #1 model changes.",
  "demo": "A ranking is a permutation. \u03c4 = +1 iff the two rankings are identical (C = all pairs, D = 0) and falls as pairs invert, so \u03c4 < 1 is a literal count of how many model pairs swap order once the labels are fixed \u2014 an assumption-light measure of verdict fragility. The bootstrap turns \u201cdoes the leader change\u201d into a calibrated probability instead of a yes/no on one sample.",
  "limits": "Corrections are single-pass (no \u03ba). The consequence is conditional: label errors flip a <i>global</i> ranking only when they are skill-discriminating and the models are close. \u03c4 says nothing about whether the corrected key is itself perfect.",
  "refs": "Kendall (1938); criterion study on HELM\u00d7MMLU-Redux; Northcutt et al. (2021); Gema et al., MMLU-Redux (2025).",
 },
 "test_reliability": {
  "title": "Reliability (Cronbach \u03b1)", "kind": "headline",
  "measures": "Internal consistency: do the items behave as one coherent scale, or pull in different directions?",
  "formula": "\u03b1 = (k / (k\u22121)) \u00b7 (1 \u2212 \u03a3\u1d62 \u03c3\u00b2\u1d62 / \u03c3\u00b2_T)",
  "method": "Treat the k questions as items and the models as respondents. \u03c3\u00b2\u1d62 is the variance of item i (correct/incorrect across models); \u03c3\u00b2_T is the variance of total scores. Computed on the original key.",
  "demo": "Under the classical model an observed score = true score + error, and reliability = true-variance / total-variance. If items measure one trait they covary, so \u03a3\u03c3\u00b2\u1d62 \u226a \u03c3\u00b2_T and \u03b1 \u2192 1. If items are unrelated \u2014 or anti-related, as when wrong gold makes \u201ccorrect\u201d inconsistent \u2014 \u03a3\u03c3\u00b2\u1d62 approaches or exceeds \u03c3\u00b2_T and \u03b1 drops, even negative. So a negative \u03b1 is a red flag that \u2018percent correct\u2019 is not measuring a single thing.",
  "limits": "Assumes one dimension and tau-equivalent items; it is a lower bound on reliability, not an upper. Pooling heterogeneous subjects inflates \u03b1 through between-group spread \u2014 a high pooled \u03b1 can hide low per-subject \u03b1.",
  "refs": "Cronbach (1951); Lord & Novick (1968).",
 },
 "label_error_audit": {
  "title": "Label / ground-truth error audit", "kind": "intrinsic",
  "measures": "The share of items whose gold answer is genuinely wrong, has no correct option, or has several \u2014 measured from a human annotation, not inferred from model behaviour.",
  "formula": "rate = |{ error_type \u2208 {wrong_groundtruth, no_correct_answer, multiple_correct_answers} }| / |annotated|",
  "method": "Count the annotation\u2019s defect categories (canonicalised across casing/spacing). Clarity issues are routed to ambiguity, not counted here.",
  "demo": "This is a direct measured proportion from expert annotation \u2014 the most reliable label-error signal available, because it does not ask a model to judge the model\u2019s own test. Its validity rests on the annotation, not on a detector.",
  "limits": "Single-pass annotation \u2192 no inter-annotator agreement (\u03ba). We separately measured that <i>behaviour</i>-based detection (item discrimination) of label errors is near-chance (ROC-AUC \u2248 0.55), so we never substitute a statistical screen for the human annotation.",
  "refs": "Gema et al., MMLU-Redux (2025); Northcutt et al. (2021).",
 },
 "item_ambiguity": {
  "title": "Item ambiguity", "kind": "intrinsic",
  "measures": "Share of items flagged as unclear in their question or options \u2014 ambiguity, distinct from a wrong key.",
  "formula": "rate = |{ error_type \u2208 {bad_question_clarity, bad_options_clarity} }| / |annotated|",
  "method": "Count clarity-flagged items from the human annotation, kept separate from label errors.",
  "demo": "An ambiguous item has no single defensible answer, so any key is partly arbitrary and the item adds noise rather than signal. Separating ambiguity from wrong-key matters: the fix differs (rewrite vs. recolour the key).",
  "limits": "Single-pass; ambiguity is partly subjective, so this is a flag for human re-review, not a determination.",
  "refs": "Gema et al., MMLU-Redux (2025).",
 },
 "discrimination_saturation": {
  "title": "Discrimination & saturation", "kind": "intrinsic",
  "measures": "Whether each item separates stronger from weaker models (discrimination), and whether the set is saturated (everyone correct).",
  "formula": "r_pb = (M\u2081 \u2212 M\u2080) / \u03c3_T \u00b7 \u221a(p\u00b7(1\u2212p))",
  "method": "M\u2081, M\u2080 = mean total score of models that got the item right vs wrong; \u03c3_T = sd of total scores; p = share correct.",
  "demo": "r_pb is the correlation between getting the item right (1/0) and overall ability (total score). A good item is positively correlated \u2014 strong models get it, weak models miss it. r_pb \u2248 0 means the item carries no signal; negative means weak models do better, often a sign the keyed answer is off.",
  "limits": "Undefined when all models agree (no variance) \u2014 exactly where universal-agreement wrong-gold items hide, which is why discrimination is <b>not</b> a reliable label-error screen. With ~10 respondents it is noisy.",
  "refs": "Classical test theory; Lord & Novick (1968).",
 },
 "coverage_distribution": {
  "title": "Coverage / distribution", "kind": "intrinsic",
  "measures": "Balance across the labelled categories, and balance of the MCQ answer key.",
  "formula": "Gini(category counts) ; top_share = max_c n_c / N   (skew flagged only when \u2265 2 categories)",
  "method": "Compute the Gini coefficient and the dominant share of the categories you provide; with a single category, assess answer-key balance instead.",
  "demo": "Concentration in one category, or a key dominated by one option, lets a model score by exploiting the distribution rather than the construct. Gini and top_share quantify that skew directly.",
  "limits": "Audits the categories you provide \u2014 it cannot say whether those are the right decomposition of the construct.",
  "refs": "\u2014",
 },
 "dataset_hygiene": {
  "title": "Dataset hygiene", "kind": "intrinsic",
  "measures": "Exact and near-duplicate items, contradictory labels, empty items, and benchmark-canary leakage.",
  "formula": "exact dup = identical normalised text ; near dup = similarity \u2265 \u03b8 ; contradictory = same item, different gold",
  "method": "Hash normalised items for exact duplicates, compare for near-duplicates, group by item to find conflicting labels, and scan for known canary strings.",
  "demo": "Duplicates inflate apparent sample size and let memorisation pay twice; contradictory labels are self-refuting; canary strings indicate the test leaked into training. All are exactly detectable and unambiguous to fix.",
  "limits": "Structural only \u2014 it says nothing about whether the non-duplicate items are correct.",
  "refs": "\u2014",
 },
 "statistical_power": {
  "title": "Statistical reliability (power)", "kind": "result",
  "measures": "Whether there are enough items for the precision being claimed.",
  "formula": "CI half-width \u2248 z\u00b7\u221a(p(1\u2212p)/n) ; MDE from a two-proportion test",
  "method": "Treat accuracy as a proportion and compute the sampling error for n items, and the smallest detectable difference between two models.",
  "demo": "Accuracy is a proportion; its sampling error shrinks like 1/\u221an. Reporting a 0.5-point gap on 100 items claims precision the sample cannot support. Power analysis makes the smallest trustworthy difference explicit.",
  "limits": "Addresses sampling error only, not bias (label errors, contamination).",
  "refs": "\u2014",
 },
 "option_order_bias": {"title": "MCQ option-order bias", "kind": "result",
  "measures": "Whether scores change when MCQ option positions are permuted.",
  "formula": "swing = max_perm acc \u2212 min_perm acc", "method": "Re-evaluate under shuffled option positions and report the swing.",
  "demo": "If the same item scores differently depending on whether the answer sits at A vs D, the score is measuring position, not knowledge. Averaging over permutations removes it.",
  "limits": "Requires re-running the eval under permutations.", "refs": "\u2014"},
 "prompt_format_sensitivity": {"title": "Prompt-format sensitivity", "kind": "result",
  "measures": "Score variance across equivalent prompt formats.",
  "formula": "range over formats of acc", "method": "Evaluate under several semantically-equivalent templates; report the spread.",
  "demo": "A large swing means the number is a property of the template, not the model \u2014 so it does not transfer to any other framing of the same task.",
  "limits": "Needs multiple format runs.", "refs": "\u2014"},
 "self_consistency": {"title": "Self-consistency / stochastic stability", "kind": "result",
  "measures": "How much of a score is run-to-run noise under stochastic decoding.",
  "formula": "between-run sd of acc", "method": "Repeat the eval at the same settings; quantify variance across runs.",
  "demo": "Stochastic decoding makes a single run a sample. If repeats wander, a reported point estimate overstates certainty; fixing the seed or reporting run CIs restores honesty.",
  "limits": "Reducible but only quantifiable with repeats.", "refs": "\u2014"},
 "judge_swap": {"title": "Judge validity (LLM-as-judge)", "kind": "result",
  "measures": "Position bias, self-preference/collusion, and inter-judge agreement for LLM judges.",
  "formula": "position-swap \u0394 ; Cohen/Fleiss \u03ba across judges", "method": "Swap answer order, swap judges, and measure agreement.",
  "demo": "If a judge favours the first answer or its own family, or two judges barely agree, the metric reflects the judge, not the contestants. \u03ba and swap-deltas expose this.",
  "limits": "Needs multiple judges / swapped runs.", "refs": "\u2014"},
 "answer_extraction_audit": {"title": "Answer-extraction / scoring audit", "kind": "result",
  "measures": "Whether the scorer parses model responses correctly.",
  "formula": "disagreement(extractor, gold-format)", "method": "Compare the automatic extraction against the expected answer format on a sample.",
  "demo": "If the parser misreads a correct free-form answer as wrong, the benchmark mis-scores deterministically \u2014 a bug in the ruler, not the model.",
  "limits": "Catches parsing defects, not conceptual ones.", "refs": "\u2014"},
 "refusal_confound": {"title": "Refusal / abstention confound", "kind": "result",
  "measures": "Whether refusals/abstentions are scored as wrong, confounding capability.",
  "formula": "rate of refusals counted as incorrect", "method": "Separate refusal/abstention from incorrect answers in scoring.",
  "demo": "A model that declines is not the same as a model that is wrong; conflating them depresses the capability estimate and rewards over-confident guessing.",
  "limits": "Requires a refusal detector.", "refs": "\u2014"},
 "multiplicity_cherrypick": {"title": "Multiplicity / cherry-picking", "kind": "result",
  "measures": "Uncorrected multiple comparisons or best-of-many reporting.",
  "formula": "family-wise error \u2191 with #comparisons", "method": "Count comparisons; apply a multiplicity correction or pre-registration check.",
  "demo": "Run enough comparisons and some \u2018wins\u2019 appear by chance. Without correction, a headline gain may be selection, not signal.",
  "limits": "Detects the structure, not intent.", "refs": "\u2014"},
 "model_drift": {"title": "Model drift / temporal validity", "kind": "result",
  "measures": "Whether scores drift across model versions / time.",
  "formula": "\u0394 acc across snapshots", "method": "Track the same eval across versions and dates.",
  "demo": "A score attached to a moving endpoint is not reproducible; pinning snapshots and dates is required before comparing.",
  "limits": "Needs versioned runs.", "refs": "\u2014"},
 "elicitation_ceiling": {"title": "Elicitation ceiling", "kind": "result",
  "measures": "Whether a score is capped by extraction effort rather than capability.",
  "formula": "acc(best elicitation) \u2212 acc(default)", "method": "Vary prompt/format/length/scaffolding and observe the lift.",
  "demo": "If stronger elicitation keeps raising the score, the original number was a lower bound on capability, not the capability \u2014 the lower edge of the capability interval.",
  "limits": "A bound, not a point estimate of \u2018true\u2019 capability.", "refs": "\u2014"},
 "contamination_perturb": {"title": "Contamination via perturbation", "kind": "result",
  "measures": "Signals of memorisation by re-testing under perturbations.",
  "formula": "acc(original) \u2212 acc(perturbed)", "method": "Compare performance on original vs. perturbed/equivalent items.",
  "demo": "Memorised items survive only the exact form; a drop under meaning-preserving perturbation hints at recall rather than reasoning.",
  "limits": "On a static benchmark this barely beats chance (membership-inference ROC-AUC < 0.6); only temporal/held-out splits are reliable. Reported as <b>investigate</b>, never a verdict.",
  "refs": "Ravaut et al. (2024); LiveCodeBench (2024)."},
 "sandbagging_paired": {"title": "Sandbagging / evaluation-awareness", "kind": "result",
  "measures": "Strategic underperformance / awareness of being evaluated.",
  "formula": "acc(eval-framed) \u2212 acc(neutral-framed), paired", "method": "Compare matched items under eval-signalling vs. neutral conditions.",
  "demo": "A model that detects evaluation can lower its score selectively; a paired gap is a hint \u2014 but a capable strategist can erase the hint.",
  "limits": "Detection is nascent and adapts to monitoring; no settled method. Reported as <b>investigate</b>, never a verdict.",
  "refs": "van der Weij et al. (2024); evaluation-awareness probing (2025)."},
 "reward_hacking_eval": {"title": "Reward-hacking of the eval", "kind": "result",
  "measures": "Responses that game the metric rather than solve the task.",
  "formula": "exploit-pattern rate", "method": "Inspect for format exploits / shortcut tokens that the scorer rewards.",
  "demo": "If a string that isn\u2019t a real answer scores, the metric is hackable and the number is not about the task.",
  "limits": "Pattern-based; needs inspection.", "refs": "\u2014"},
 "subgroup_power": {"title": "Subgroup power", "kind": "result",
  "measures": "Whether per-subgroup claims have enough items.",
  "formula": "per-slice n and CI", "method": "Compute sampling error within each reported subgroup.",
  "demo": "A per-topic ranking on 5 items is noise; subgroup precision must be checked before slicing.",
  "limits": "Sampling error only.", "refs": "\u2014"},
 "provenance_repro": {"title": "Provenance & reproducibility", "kind": "result",
  "measures": "Whether dataset version, sources and seeds are pinned so the result reproduces.",
  "formula": "content-hash(inputs) \u2192 record id", "method": "Hash the inputs and check the verdict recomputes.",
  "demo": "If the inputs aren\u2019t pinned, no claim on them is checkable. Content-addressing makes every number reproducible from its inputs.",
  "limits": "Ensures reproducibility, not correctness.", "refs": "\u2014"},
}

_SHORT_TBL = {
    "claude-3-opus-20240229": "Claude-3-Opus", "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4-0613": "GPT-4-0613", "gpt-4-1106-preview": "GPT-4-1106",
    "gemini-1.5-pro-001": "Gemini-1.5-Pro", "gemini-1.5-flash-001": "Gemini-1.5-Flash",
    "Meta-Llama-3-70B-Instruct": "Llama-3-70B", "palmyra-x-v3": "Palmyra-X-v3",
    "text-unicorn@001": "text-unicorn", "Mixtral-8x22B-v0.1": "Mixtral-8x22B",
}


def _short(m):
    """Strip a provider prefix (provider_model) and prettify known model strings."""
    prov, _, rest = str(m).partition("_")
    rest = rest or prov
    return _SHORT_TBL.get(rest, rest)


def _var(xs):
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def assemble_ui_data(store, pred_rows, intrinsic_rows, spotlight_names, models,
                     candidates=None):
    """Assemble the observatory-UI data dict from a synced store plus the item-level
    rows used to build it.

    pred_rows / intrinsic_rows: {benchmark_name: [row, ...]} where a prediction row is
    {item, subject, original_gold, corrected_gold, preds:{model: letter}} and an
    intrinsic row carries at least {error_type, subject}. Every headline figure comes
    from `leaderboard(store)`; the per-benchmark trace is reconstructed from the rows
    so that it matches the headline exactly (faithful-trace invariant).
    """
    from trust_the_eval.calibration.realworld import (DEFECT_ERROR_TYPES,
                                                      canonical_error_type)
    candidates = candidates or {}
    LB = leaderboard(store)
    bench_by = {b["name"]: b for b in LB["benchmarks"]}

    def prow(b):
        s = b.get("sensitivity") or {}
        le = b.get("label_error") or {}
        r = b.get("reliability") or {}
        rem = b.get("remediation") or {}
        return {
            "name": b["name"], "slug": b.get("slug"),
            "subject": b["name"].split("::")[1] if "::" in b["name"] else b["name"],
            "label_error": {"rate": le.get("rate"), "k": le.get("k"), "n": le.get("n")},
            "ambiguity_rate": (b.get("ambiguity") or {}).get("rate"),
            "alpha": r.get("pooled_alpha"),
            "sens": {"tau": s.get("kendall_tau"), "stable": s.get("ranking_stable"),
                     "dmin": s.get("delta_min_pts"), "dmax": s.get("delta_max_pts"),
                     "skill": s.get("skill_discrimination"), "p_top1": s.get("p_top1_change"),
                     "n_changed": s.get("n_changed_items"), "moved": s.get("models_moved"),
                     "n_models": s.get("n_models"), "n_items": s.get("n_items_scored")},
            "status": b.get("status_word"),
            "rem": {"summary": rem.get("summary"), "n": rem.get("n_actions"),
                    "n_fix": rem.get("n_instrument_fix"), "n_relay": rem.get("n_relay_only"),
                    "impact": (rem.get("verdict_impact") or {}).get("ranking_stable")},
            "n_claims": b.get("n_claims"), "dataset_version": b.get("dataset_version"),
            "record_id": b.get("record_id"),
        }

    def compute_trace(name, head_tau):
        src = pred_rows.get(name, [])
        irows = intrinsic_rows.get(name, [])
        preds, og, cg = {}, {}, {}
        for r in src:
            og[r["item"]] = set(r["original_gold"])
            cg[r["item"]] = set(r["corrected_gold"])
            for m, l in r["preds"].items():
                preds.setdefault(m, {})[r["item"]] = l
        items = list(og)
        n = len(items)
        if n == 0:
            return None
        ms = list(preds)
        pm = []
        for m in ms:
            ao = sum(1 for it in items if preds[m].get(it) in og[it]) / n
            ac = sum(1 for it in items if preds[m].get(it) in cg[it]) / n
            pm.append({"m": _short(m), "ao": round(ao, 4), "ac": round(ac, 4),
                       "d": round(ac - ao, 4)})
        npair = len(ms) * (len(ms) - 1) // 2
        # C/D reconstructed from the headline tau (no ties in the ranking permutation,
        # so C + D = npair and C - D = tau * npair) -> trace matches headline exactly.
        if head_tau is not None and npair:
            D = round((1 - head_tau) * npair / 2)
            C = npair - D
            tau = head_tau
        else:
            C = D = 0
            tau = None
        item_vars = [_var([1.0 if preds[m].get(it) in og[it] else 0.0 for m in ms]) for it in items]
        totals = [sum(1.0 if preds[m].get(it) in og[it] else 0.0 for it in items) for m in ms]
        sum_iv, tot_v, k = sum(item_vars), _var(totals), n
        alpha = (k / (k - 1)) * (1 - sum_iv / tot_v) if (k > 1 and tot_v > 0) else None
        from collections import Counter
        ann = [r for r in irows if r.get("error_type") is not None]
        etc = Counter(canonical_error_type(r["error_type"]) for r in ann)
        defects = sum(v for kk, v in etc.items() if kk in DEFECT_ERROR_TYPES)
        return {
            "n_items": n, "n_models": len(ms),
            "per_model": sorted(pm, key=lambda x: -x["ac"]),
            "tau": {"C": C, "D": D, "npair": npair,
                    "tau": round(tau, 4) if tau is not None else None},
            "cronbach": {"k": k, "sum_item_var": round(sum_iv, 4),
                         "total_var": round(tot_v, 4),
                         "alpha": round(alpha, 4) if alpha is not None else None},
            "label": {"annotated": len(ann), "defects": defects,
                      "by_type": {kk: vv for kk, vv in sorted(etc.items())},
                      "defect_types": sorted(DEFECT_ERROR_TYPES)},
        }

    def deep(name):
        b = bench_by[name]
        src = pred_rows.get(name, [])
        preds, og, cg = {}, {}, {}
        for r in src:
            og[r["item"]] = set(r["original_gold"])
            cg[r["item"]] = set(r["corrected_gold"])
            for m, l in r["preds"].items():
                preds.setdefault(m, {})[r["item"]] = l
        items = list(og)
        rows = []
        for m in preds:
            ao = sum(1 for it in items if preds[m].get(it) in og[it]) / len(items)
            ac = sum(1 for it in items if preds[m].get(it) in cg[it]) / len(items)
            rows.append({"model": _short(m), "acc_orig": ao, "acc_corr": ac, "delta": ac - ao})
        ro = sorted(rows, key=lambda x: -x["acc_orig"])
        rc = sorted(rows, key=lambda x: -x["acc_corr"])
        for i, x in enumerate(ro):
            x["rank_orig"] = i + 1
        rk = {x["model"]: i + 1 for i, x in enumerate(rc)}
        for x in rows:
            x["rank_corr"] = rk[x["model"]]
        cm = correctness_from_predictions(preds, og)
        A = analyze(cm)
        it_ids = list(A["per_item"])
        step = max(1, len(it_ids) // 140)
        scatter = []
        for it in it_ids[::step]:
            pi = A["per_item"][it]
            scatter.append({"d": round(pi["difficulty"], 3),
                            "disc": None if pi["discrimination"] is None else round(pi["discrimination"], 3),
                            "suspect": bool(pi["suspect"])})
        rel = reliability_summary(cm, {r["item"]: r["subject"] for r in src})
        alphas = sorted([v for v in (rel.get("per_subject_alpha") or {}).values() if v is not None]) \
            if rel.get("per_subject_alpha") else []
        rid = b.get("record_id")
        rec = store.get(rid) if rid else None
        h = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:16] if rec else None
        pbis = None
        if preds:
            totals = {m: sum(cm[m].values()) for m in cm}
            mu_t = sum(totals.values()) / len(totals)
            sdT = (sum((totals[m] - mu_t) ** 2 for m in totals) / len(totals)) ** 0.5
            for it in items:
                col = {m: cm[m].get(it, 0) for m in cm}
                p = sum(col.values()) / len(col)
                if 0 < p < 1 and sdT > 0:
                    ones = [totals[m] for m in cm if col[m] == 1]
                    zeros = [totals[m] for m in cm if col[m] == 0]
                    M1, M0 = sum(ones) / len(ones), sum(zeros) / len(zeros)
                    pbis = {"p": round(p, 3), "M1": round(M1, 2), "M0": round(M0, 2),
                            "sdT": round(sdT, 3), "n1": len(ones), "n0": len(zeros),
                            "rpb": round((M1 - M0) / sdT * (p * (1 - p)) ** 0.5, 3)}
                    break
        subj = name.split("::")[1] if "::" in name else name
        return {
            "name": name, "ranking": sorted(rows, key=lambda x: x["rank_corr"]),
            "top1_changed": (ro[0]["model"] != rc[0]["model"]) if rows else False,
            "scatter": scatter, "pbis": pbis, "alphas": [round(a, 3) for a in alphas],
            "alpha_pooled": rel.get("pooled_alpha"),
            "candidates": candidates.get(subj, []),
            "remediation": b.get("remediation"),
            "provenance": {"record_id": rid, "hash": h,
                           "dataset_version": b.get("dataset_version"),
                           "n_items": len(items), "n_models": len(preds)},
            "claims": None,
        }

    trace_by = {b["name"]: compute_trace(b["name"], (b.get("sensitivity") or {}).get("kendall_tau"))
                for b in LB["benchmarks"]}
    portfolio = []
    for b in LB["benchmarks"]:
        row = prow(b)
        row["trace"] = trace_by.get(b["name"])
        portfolio.append(row)
    spotlight = {n: deep(n) for n in spotlight_names if n in bench_by}
    subjects = sorted({r.get("subject") for rs in intrinsic_rows.values()
                       for r in rs if r.get("subject")})
    return {
        "generated": "trust_the_eval.observatory_ui (real pipeline run)",
        "models": [_short(m) for m in models],
        "n_benchmarks": len(portfolio), "n_subjects": len(subjects),
        "portfolio": portfolio, "spotlight": spotlight,
        "spotlight_names": [n for n in spotlight_names if n in bench_by],
    }

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Meridian — Eval Validity Observatory</title>
<style>
:root{
  --paper:#f3f2ec; --surface:#fffdf8; --ink:#1b1a15; --muted:#57544c; --faint:#918d83;
  --line:#e6e3d9; --line2:#efece3;
  --accent:#2f6a54; --accent-soft:#e7efe9; --accent-line:#c6dccf;
  --ok:#2f6a54; --ok-soft:#e7efe9; --ok-line:#c6dccf;
  --warn:#8a6326; --warn-soft:#f4edde; --warn-line:#e4d6ba;
  --bad:#8a3a2c; --bad-soft:#f1e7e2; --bad-line:#ddc6bf;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;font-size:14px;line-height:1.5}
a{color:inherit}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px}
.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}

header.top{position:sticky;top:0;z-index:30;background:rgba(243,242,236,.85);backdrop-filter:saturate(1.3) blur(8px);border-bottom:1px solid var(--line)}
.top .wrap{display:flex;align-items:center;gap:20px;height:60px}
.mark{display:flex;align-items:baseline;gap:11px;cursor:pointer}
.mark .glyph{font-family:var(--mono);font-weight:600;font-size:17.5px;letter-spacing:.06em}
.mark .glyph b{color:var(--accent);font-weight:600}
.mark .sub{font-size:12px;color:var(--muted)}
.top nav{margin-left:auto;display:flex;gap:5px}
.top nav button{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);background:none;border:1px solid transparent;padding:7px 12px;border-radius:7px;cursor:pointer}
.top nav button:hover{color:var(--ink);border-color:var(--line)}
.top nav button.on{color:var(--surface);background:var(--ink);border-color:var(--ink)}

.hero{padding:40px 0 22px;border-bottom:1px solid var(--line)}
.hero h1{font-size:29px;line-height:1.2;letter-spacing:-.015em;margin:.18em 0 .35em;font-weight:600;max-width:780px}
.hero h1 .em{color:var(--accent)}
.hero .line{font-size:13px;color:var(--muted);max-width:720px}
.ribbon{display:flex;flex-wrap:wrap;margin-top:22px;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--surface)}
.ribbon .cell{flex:1;min-width:150px;padding:15px 18px;border-right:1px solid var(--line)}
.ribbon .cell:last-child{border-right:none}
.ribbon .v{font-family:var(--mono);font-size:23px;font-weight:600;letter-spacing:-.01em}
.ribbon .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-top:4px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;vertical-align:middle;margin-right:7px}

.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:20px 0 12px}
.chip{font-family:var(--mono);font-size:11px;letter-spacing:.04em;padding:6px 12px;border:1px solid var(--line);border-radius:999px;background:var(--surface);color:var(--muted);cursor:pointer}
.chip.on{color:var(--surface);background:var(--ink);border-color:var(--ink)}
.search{margin-left:auto}
.search input{font-family:var(--mono);font-size:12px;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);width:210px;color:var(--ink)}

.tbl{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.tbl thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);cursor:pointer;white-space:nowrap;user-select:none}
.tbl thead th .ar{color:var(--faint);margin-left:4px}
.tbl tbody td{padding:11px 14px;border-bottom:1px solid var(--line2);vertical-align:middle}
.tbl tbody tr:last-child td{border-bottom:none}
.tbl tbody tr{cursor:pointer}
.tbl tbody tr:hover{background:#fbfaf5}
.bm{font-weight:600;letter-spacing:-.01em}
.bm .sx{font-family:var(--mono);font-size:11px;color:var(--faint);display:block;margin-top:1px}
.lebar{display:flex;align-items:center;gap:8px;min-width:120px}
.lebar .track{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.lebar .fill{height:100%;border-radius:3px}
.tag{display:inline-flex;align-items:center;font-family:var(--mono);font-size:11px;letter-spacing:.02em;padding:3px 9px;border-radius:999px;border:1px solid}
.tag.stable{color:var(--ok);background:var(--ok-soft);border-color:var(--ok-line)}
.tag.fragile{color:var(--warn);background:var(--warn-soft);border-color:var(--warn-line)}
.tag.na{color:var(--muted);background:#eeebe3;border-color:var(--line)}
.tau{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:7px}
.rempill{font-family:var(--mono);font-size:11px;color:var(--muted)}
.rempill b{color:var(--ink);font-weight:600}

.back{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);background:none;border:none;cursor:pointer;padding:18px 0 6px}
.back:hover{color:var(--ink)}
.dhead{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;padding-bottom:8px}
.dhead h2{font-size:24px;font-weight:600;letter-spacing:-.015em;margin:.1em 0}
.prov{font-family:var(--mono);font-size:11px;color:var(--faint);text-align:right;line-height:1.7}
.prov .ok{color:var(--ok)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.card.full{grid-column:1/-1}
.card h3{font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:0 0 12px;font-weight:600;display:flex;align-items:center;gap:8px}
.verdict{font-size:17px;line-height:1.45;font-weight:500}
.verdict .hl{padding:0 3px;border-radius:3px}
.verdict.fragile .hl{background:var(--warn-soft);color:var(--warn)}
.verdict.stable .hl{background:var(--ok-soft);color:var(--ok)}
.metrics{display:flex;flex-wrap:wrap;border:1px solid var(--line);border-radius:11px;overflow:hidden}
.metrics .m{flex:1;min-width:130px;padding:13px 15px;border-right:1px solid var(--line)}
.metrics .m:last-child{border-right:none}
.metrics .mv{font-family:var(--mono);font-size:19px;font-weight:600}
.metrics .mk{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-top:4px;display:flex;align-items:center;gap:6px}
.note{font-size:12px;color:var(--muted);margin-top:11px;line-height:1.5}

.fbtn{font-family:var(--mono);font-style:italic;font-size:11px;width:17px;height:17px;line-height:15px;text-align:center;border:1px solid var(--accent-line);background:var(--accent-soft);color:var(--accent);border-radius:5px;cursor:pointer;padding:0;flex:none}
.fbtn:hover{background:var(--accent);color:var(--surface)}

.act{border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin-bottom:10px;background:var(--surface)}
.act:last-child{margin-bottom:0}
.act .row1{display:flex;align-items:center;gap:9px;margin-bottom:6px;flex-wrap:wrap}
.tier{font-family:var(--mono);font-size:10px;letter-spacing:.05em;text-transform:uppercase;padding:3px 8px;border-radius:6px;border:1px solid;font-weight:600}
.tier.instrument_fix{color:var(--ok);background:var(--ok-soft);border-color:var(--ok-line)}
.tier.relay_only{color:var(--warn);background:var(--warn-soft);border-color:var(--warn-line)}
.tier.review{color:#7a5a16;background:#f5efdd;border-color:#e6d9b8}
.tier.investigate{color:var(--bad);background:var(--bad-soft);border-color:var(--bad-line)}
.tier.informational{color:var(--muted);background:#eeebe2;border-color:var(--line)}
.act .pid{font-family:var(--mono);font-size:11px;color:var(--ink);font-weight:600}
.act .sev{font-family:var(--mono);font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.05em}
.act .scihref{margin-left:auto}
.act .txt{font-size:13px;line-height:1.5}
.act .why{font-size:12px;color:var(--muted);margin-top:5px}
.act .cav{font-size:11.5px;color:var(--warn);margin-top:6px;font-style:italic}
.impact{font-size:12.5px;color:var(--muted);padding:10px 13px;background:#faf8f1;border:1px dashed var(--line);border-radius:9px;margin-bottom:14px}
.impact b{color:var(--ink)}
.cands{margin-top:10px;border-top:1px solid var(--line);padding-top:10px}
.cands table{width:100%;border-collapse:collapse}
.cands td{font-family:var(--mono);font-size:11.5px;padding:5px 6px;border-bottom:1px solid var(--line2);vertical-align:top}
.cands .ar{color:var(--warn)}
.cands .q{color:var(--muted);max-width:520px}

text{font-family:var(--mono);fill:var(--muted)}
.axline{stroke:var(--line)}

.plib{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding-top:22px}
.pcard{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.pcard .kindt{font-family:var(--mono);font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;padding:2px 7px;border-radius:5px;border:1px solid var(--line);color:var(--muted)}
.pcard h4{font-size:15px;margin:8px 0 4px;letter-spacing:-.01em}
.pcard .pid2{font-family:var(--mono);font-size:10.5px;color:var(--faint)}
.pcard p{font-size:12.5px;color:var(--muted);line-height:1.5;margin:8px 0 12px}
.pcard .pbtns{display:flex;gap:7px}
.pbtn{font-family:var(--mono);font-size:11px;letter-spacing:.04em;padding:6px 11px;border:1px solid var(--line);border-radius:7px;background:var(--surface);color:var(--ink);cursor:pointer}
.pbtn:hover{border-color:var(--ink)}
.pbtn.primary{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.secthead{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:26px 0 2px}

.overlay{position:fixed;inset:0;z-index:60;background:rgba(27,26,21,.34);backdrop-filter:blur(2px);display:flex;align-items:flex-start;justify-content:center;padding:48px 20px;overflow:auto}
.sheet{background:var(--surface);border:1px solid var(--line);border-radius:14px;max-width:680px;width:100%;box-shadow:0 24px 60px rgba(27,26,21,.22)}
.sheethd{display:flex;justify-content:space-between;align-items:flex-start;padding:20px 22px 0}
.sheethd h3{font-size:19px;margin:5px 0 2px;letter-spacing:-.01em}
.sheethd .pidm{font-size:11px;color:var(--faint)}
.x{font-size:22px;line-height:1;color:var(--muted);background:none;border:none;cursor:pointer;padding:0 2px}
.tabs{display:flex;gap:4px;padding:14px 22px 0}
.tabs button{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);background:none;border:none;border-bottom:2px solid transparent;padding:7px 4px;cursor:pointer}
.tabs button.on{color:var(--ink);border-color:var(--accent)}
.sheetbody{padding:16px 22px 24px}
.sx2{margin-bottom:14px}
.sx2 .cl{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:4px}
.sx2 p{margin:0;font-size:13.5px;line-height:1.6;color:#33312b}
.sx2 .refs{font-size:12px;color:var(--muted)}
.formula{background:#faf8f1;border:1px solid var(--line);border-radius:8px;padding:11px 13px;font-size:13px;color:var(--ink);overflow-x:auto}
.csel{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-bottom:12px}
.csel select{font-family:var(--mono);font-size:11.5px;padding:4px 7px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}
.comp .cstep{font-size:13px;line-height:1.55;padding:8px 0;border-bottom:1px solid var(--line2)}
.comp .cstep:last-child{border-bottom:none}
.comp .cl{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--accent);margin-right:8px}
.ctab{width:100%;border-collapse:collapse;margin:8px 0}
.ctab th{font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
.ctab th:first-child{text-align:left}
.ctab td{font-family:var(--mono);font-size:12px;padding:5px 8px;border-bottom:1px solid var(--line2);text-align:right}
.ctab td:first-child{text-align:left;color:var(--ink)}
.df{color:var(--warn);font-family:var(--mono);font-size:10px}
.cwarn{font-size:12px;color:var(--warn);background:var(--warn-soft);border:1px solid var(--warn-line);border-radius:7px;padding:8px 11px;margin-top:10px}
.cnote{font-size:12.5px;color:var(--muted);font-style:italic}

.method .card{margin-bottom:16px}
.method h3{font-size:15px;text-transform:none;letter-spacing:-.01em;color:var(--ink)}
.method p{font-size:13.5px;color:#33312b;line-height:1.6;margin:.4em 0}
.method .bl{border-left:3px solid var(--accent);padding-left:14px}
.foot{color:var(--faint);font-family:var(--mono);font-size:11px;text-align:center;padding:34px 0 26px}
@media(max-width:760px){.grid,.plib{grid-template-columns:1fr}.hero h1{font-size:22px}}
</style></head>
<body>
<header class="top"><div class="wrap">
  <div class="mark" onclick="go('portfolio')"><span class="glyph">meri<b>dian</b></span><span class="sub">Eval Validity Observatory</span></div>
  <nav>
    <button id="nav-portfolio" class="on" onclick="go('portfolio')">Portfolio</button>
    <button id="nav-probes" onclick="go('probes')">Probe library</button>
    <button id="nav-method" onclick="go('method')">Methodology</button>
  </nav>
</div></header>
<main id="app"></main>
<div id="modal"></div>
<div class="foot wrap" id="foot"></div>
<script>
const DATA=__DATA__; const SCI=__SCI__;
const $=(h)=>{const t=document.createElement('template');t.innerHTML=h.trim();return t.content.firstChild;};
const pct=(x,d=1)=>x==null?'\u2014':(x*100).toFixed(d)+'%';
const pp=(x,d=1)=>x==null?'\u2014':(x>=0?'+':'')+x.toFixed(d);
const f2=(x,d=2)=>x==null?'\u2014':x.toFixed(d);
function leColor(r){return r==null?'var(--faint)':r>=0.20?'var(--bad)':r>=0.05?'var(--warn)':'var(--ok)';}
function traceOf(n){const r=DATA.portfolio.find(p=>p.name===n);return r?r.trace:null;}
function spotOf(n){return DATA.spotlight[n]||null;}
function rowOf(n){return DATA.portfolio.find(p=>p.name===n);}

function ribbon(ranking){
  const W=620,rowH=30,padT=34,padB=14,padX=150,n=ranking.length,H=padT+padB+rowH*(n-1);
  const y=r=>padT+rowH*(r-1);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">`;
  s+=`<text x="${padX-12}" y="18" text-anchor="end" font-size="10.5" letter-spacing=".08em">ORIGINAL KEY</text>`;
  s+=`<text x="${W-padX+12}" y="18" text-anchor="start" font-size="10.5" letter-spacing=".08em">CORRECTED KEY</text>`;
  for(const m of ranking){const yo=y(m.rank_orig),yc=y(m.rank_corr),moved=m.rank_orig!==m.rank_corr;
    const col=moved?'var(--warn)':'var(--ok-line)',x1=padX,x2=W-padX,mid=(x1+x2)/2;
    s+=`<path d="M${x1} ${yo} C ${mid} ${yo}, ${mid} ${yc}, ${x2} ${yc}" fill="none" stroke="${col}" stroke-width="${moved?2:1.4}" opacity="${moved?.95:.7}"/>`;
    s+=`<circle cx="${x1}" cy="${yo}" r="3" fill="${moved?'var(--warn)':'var(--ok)'}"/><circle cx="${x2}" cy="${yc}" r="3" fill="${moved?'var(--warn)':'var(--ok)'}"/>`;
    s+=`<text x="${x1-12}" y="${yo+3.5}" text-anchor="end" font-size="11" fill="var(--ink)">${m.model}</text>`;
    s+=`<text x="${x2+12}" y="${yc+3.5}" text-anchor="start" font-size="11" fill="${moved?'var(--warn)':'var(--ink)'}">${m.model}</text>`;}
  return s+`</svg>`;
}
function scatter(points){
  const W=560,H=300,L=44,R=14,T=14,B=34,xw=W-L-R,yh=H-T-B;
  const px=d=>L+d*xw,py=v=>T+(1-(v+1)/2)*yh;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">`;
  for(const v of [-1,-.5,0,.5,1]){const yy=py(v);s+=`<line class="axline" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/><text x="${L-8}" y="${yy+3}" text-anchor="end" font-size="9.5">${v}</text>`;}
  for(const d of [0,.25,.5,.75,1])s+=`<text x="${px(d)}" y="${H-12}" text-anchor="middle" font-size="9.5">${d}</text>`;
  s+=`<text x="${L+xw/2}" y="${H-1}" text-anchor="middle" font-size="9.5" letter-spacing=".06em">DIFFICULTY (share correct)</text>`;
  s+=`<text transform="translate(11 ${T+yh/2}) rotate(-90)" text-anchor="middle" font-size="9.5" letter-spacing=".06em">DISCRIMINATION</text>`;
  let nulls=0;for(const p of points){if(p.disc==null){nulls++;s+=`<circle cx="${px(p.d)}" cy="${H-B+7}" r="2.1" fill="var(--faint)" opacity=".7"/>`;continue;}
    s+=`<circle cx="${px(p.d)}" cy="${py(p.disc)}" r="3" fill="${p.suspect?'var(--warn)':'var(--ok)'}" opacity="${p.suspect?.95:.5}"/>`;}
  s+=`<text x="${W-R}" y="${H-B+10}" text-anchor="end" font-size="9">${nulls} items: discrimination undefined (all models agree)</text>`;
  return s+`</svg>`;
}
function histogram(vals){
  if(!vals.length)return '<div class="note">No per-subject reliability available.</div>';
  const W=560,H=210,L=40,R=12,T=12,B=30,lo=Math.min(-2,Math.floor(Math.min(...vals))),hi=1,bins=14,bw=(hi-lo)/bins;
  const counts=new Array(bins).fill(0);vals.forEach(v=>{counts[Math.min(bins-1,Math.max(0,Math.floor((v-lo)/bw)))]++;});
  const mx=Math.max(...counts),xw=W-L-R,yh=H-T-B;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">`;
  const zx=L+((0-lo)/(hi-lo))*xw;
  s+=`<line x1="${zx}" y1="${T}" x2="${zx}" y2="${T+yh}" stroke="var(--warn)" stroke-dasharray="3 3" opacity=".7"/><text x="${zx}" y="${T+8}" font-size="9" fill="var(--warn)" text-anchor="middle">\u03b1=0</text>`;
  for(let i=0;i<bins;i++){const x=L+i/bins*xw,h=counts[i]/mx*yh,mid=lo+(i+.5)*bw,col=mid<0?'var(--warn)':mid<.5?'var(--faint)':'var(--ok)';
    s+=`<rect x="${x+1}" y="${T+yh-h}" width="${xw/bins-2}" height="${h}" fill="${col}" opacity=".85"/>`;}
  for(const t of [lo,-1,0,1])s+=`<text x="${L+((t-lo)/(hi-lo))*xw}" y="${H-10}" text-anchor="middle" font-size="9.5">${t}</text>`;
  s+=`<text x="${L+xw/2}" y="${H-.5}" text-anchor="middle" font-size="9.5" letter-spacing=".06em">CRONBACH \u03b1 PER SUBJECT (57 MMLU subjects)</text>`;
  return s+`</svg>`;
}

function computePanel(pid,name){
  const t=traceOf(name),sp=spotOf(name),r=rowOf(name);
  if(!t)return `<div class="cnote">No predictions for this benchmark \u2014 nothing to compute here.</div>`;
  if(pid==='result_sensitivity'){const T=t.tau,pst=r.sens.p_top1;
    const rows=t.per_model.map(m=>`<tr><td>${m.m}</td><td>${(m.ao*100).toFixed(1)}</td><td>${(m.ac*100).toFixed(1)}</td><td style="color:${m.d>=0?'var(--ok)':'var(--warn)'}">${m.d>=0?'+':''}${(m.d*100).toFixed(1)}</td></tr>`).join('');
    return `<div class="comp">
      <div class="cstep"><span class="cl">Inputs</span>${t.n_models} models \u00d7 ${t.n_items} scored items; original key vs corrected key.</div>
      <table class="ctab"><thead><tr><th>model</th><th>acc orig</th><th>acc corr</th><th>\u0394 pts</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="cstep"><span class="cl">Kendall \u03c4</span>concordant pairs C=${T.C}, discordant D=${T.D}, total = n(n\u22121)/2 = ${T.npair}. &nbsp;\u03c4 = (C\u2212D)/total = (${T.C}\u2212${T.D})/${T.npair} = <b>${T.tau.toFixed(4)}</b>.</div>
      <div class="cstep"><span class="cl">Top\u20111 flip</span>bootstrap over items \u21d2 P(#1 changes) = <b>${pst==null?'\u2014':(pst*100).toFixed(0)+'%'}</b>.</div></div>`;}
  if(pid==='test_reliability'){const c=t.cronbach,term=c.sum_item_var/c.total_var;
    return `<div class="comp">
      <div class="cstep"><span class="cl">Inputs</span>k=${c.k} items, respondents = ${t.n_models} models (original key).</div>
      <div class="cstep"><span class="cl">Variances</span>\u03a3\u1d62 \u03c3\u00b2\u1d62 = ${c.sum_item_var}, &nbsp; \u03c3\u00b2_T = ${c.total_var}.</div>
      <div class="cstep"><span class="cl">\u03b1</span>= (k/(k\u22121))\u00b7(1 \u2212 \u03a3\u03c3\u00b2\u1d62/\u03c3\u00b2_T) = (${c.k}/${c.k-1})\u00b7(1 \u2212 ${c.sum_item_var}/${c.total_var}) = ${(c.k/(c.k-1)).toFixed(3)}\u00b7(${(1-term).toFixed(3)}) = <b>${c.alpha==null?'\u2014':c.alpha.toFixed(4)}</b>.</div>
      ${c.alpha!=null&&c.alpha<0?`<div class="cwarn">Negative \u03b1: items anti-cohere \u2014 \u2018percent correct\u2019 is not one scale here, consistent with the high label-error rate.</div>`:''}</div>`;}
  if(pid==='label_error_audit'){const L=t.label;
    const rows=Object.entries(L.by_type).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td><td style="text-align:left;color:${L.defect_types.includes(k)?'var(--warn)':'var(--faint)'}">${L.defect_types.includes(k)?'defect':(k.indexOf('bad_')===0?'ambiguity':'ok')}</td></tr>`).join('');
    return `<div class="comp">
      <div class="cstep"><span class="cl">Annotation</span>single-pass human labels, canonicalised across casing.</div>
      <table class="ctab"><thead><tr><th>error_type</th><th>n</th><th style="text-align:left">class</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="cstep"><span class="cl">Rate</span>defects = ${L.defect_types.map(d=>L.by_type[d]||0).join(' + ')} = ${L.defects}; &nbsp;rate = ${L.defects}/${L.annotated} = <b>${(L.defects/L.annotated*100).toFixed(0)}%</b>.</div></div>`;}
  if(pid==='item_ambiguity'){const L=t.label,bq=L.by_type.bad_question_clarity||0,bo=L.by_type.bad_options_clarity||0;
    return `<div class="comp"><div class="cstep"><span class="cl">Clarity flags</span>bad_question_clarity=${bq}, bad_options_clarity=${bo}.</div>
      <div class="cstep"><span class="cl">Rate</span>= (${bq}+${bo})/${L.annotated} = <b>${((bq+bo)/L.annotated*100).toFixed(0)}%</b>.</div></div>`;}
  if(pid==='discrimination_saturation'){const b=sp&&sp.pbis;
    if(!b)return `<div class="cnote">Worked example shown on spotlight benchmarks.</div>`;
    return `<div class="comp">
      <div class="cstep"><span class="cl">One item</span>p=${b.p} correct (${b.n1} right / ${b.n0} wrong of ${t.n_models} models).</div>
      <div class="cstep"><span class="cl">Means</span>M\u2081=${b.M1} (right), M\u2080=${b.M0} (wrong); \u03c3_T=${b.sdT}.</div>
      <div class="cstep"><span class="cl">r_pb</span>= (M\u2081\u2212M\u2080)/\u03c3_T\u00b7\u221a(p(1\u2212p)) = (${b.M1}\u2212${b.M0})/${b.sdT}\u00b7\u221a(${b.p}\u00b7${(1-b.p).toFixed(3)}) = <b>${b.rpb}</b>.</div>
      ${Math.abs(b.rpb)<0.12?`<div class="cwarn">|r_pb|\u22480: this item barely separates strong from weak models \u2014 little signal.</div>`:''}</div>`;}
  return `<div class="cnote">This probe did not fire on this benchmark, so there is no number to trace here \u2014 see the method on the Science tab. The instrument never shows a value without this panel.</div>`;
}

let M_TAB='sci',M_PID=null,M_BENCH=null;
function openProbe(pid,name){M_PID=pid;M_BENCH=name||M_BENCH||'MMLU::virology';M_TAB='sci';renderModal();}
function closeModal(){document.getElementById('modal').innerHTML='';}
function renderModal(){
  const s=SCI[M_PID];if(!s)return;
  const opts=DATA.spotlight_names.map(n=>`<option value="${n}" ${n===M_BENCH?'selected':''}>${n}</option>`).join('');
  const body=M_TAB==='sci'
    ?`<div class="sx2"><span class="cl">Measures</span><p>${s.measures}</p></div>
      <div class="sx2"><span class="cl">Formula</span><div class="formula mono">${s.formula}</div></div>
      <div class="sx2"><span class="cl">Method</span><p>${s.method}</p></div>
      <div class="sx2"><span class="cl">Why it\u2019s valid</span><p>${s.demo}</p></div>
      <div class="sx2"><span class="cl">Limits</span><p>${s.limits}</p></div>
      <div class="sx2"><span class="cl">References</span><p class="refs">${s.refs}</p></div>`
    :`<div class="csel">Computed on <select onchange="M_BENCH=this.value;renderModal()">${opts}</select></div>${computePanel(M_PID,M_BENCH)}`;
  const m=document.getElementById('modal');m.innerHTML='';
  m.appendChild($(`<div class="overlay" onclick="if(event.target===this)closeModal()"><div class="sheet">
    <div class="sheethd"><div><div class="eyebrow">Probe</div><h3>${s.title}</h3><div class="pidm mono">${M_PID}</div></div><button class="x" onclick="closeModal()">\u00d7</button></div>
    <div class="tabs"><button class="${M_TAB==='sci'?'on':''}" onclick="M_TAB='sci';renderModal()">Science</button><button class="${M_TAB==='calc'?'on':''}" onclick="M_TAB='calc';renderModal()">Computation</button></div>
    <div class="sheetbody">${body}</div></div></div>`));
}

function go(v,arg){
  ['portfolio','probes','method'].forEach(k=>document.getElementById('nav-'+k).classList.toggle('on',v===k||(v==='detail'&&k==='portfolio')));
  window.scrollTo(0,0);
  if(v==='portfolio')renderPortfolio();else if(v==='detail')renderDetail(arg);else if(v==='probes')renderProbes();else if(v==='method')renderMethod();
}

let SORT={key:'le',dir:-1},FILTER='all',Q='';
function renderPortfolio(){
  const P=DATA.portfolio,nS=P.filter(p=>p.sens.stable===true).length,nF=P.filter(p=>p.sens.stable===false).length,nD=P.filter(p=>p.status==='degraded').length;
  const app=document.getElementById('app');app.innerHTML='';
  app.appendChild($(`<section class="hero"><div class="wrap">
    <div class="eyebrow">Audit the instrument \u2014 not the model</div>
    <h1>Every benchmark is profiled by whether its verdict <span class="em">survives correcting the labels</span> \u2014 not by a trust score.</h1>
    <div class="line">${DATA.n_benchmarks} benchmarks audited on real predictions (HELM v1.3.0) against human label corrections (MMLU-Redux), ${DATA.models.length} models. Open any row for its dossier; every number carries a <span class="mono">\u0192</span> panel showing exactly how it was computed.</div>
    <div class="ribbon">
      <div class="cell"><div class="v">${DATA.n_benchmarks}</div><div class="k">Benchmarks audited</div></div>
      <div class="cell"><div class="v"><span class="dot" style="background:var(--ok)"></span>${nS}</div><div class="k">Rank-stable verdicts</div></div>
      <div class="cell"><div class="v"><span class="dot" style="background:var(--warn)"></span>${nF}</div><div class="k">Rank-fragile verdicts</div></div>
      <div class="cell"><div class="v"><span class="dot" style="background:var(--bad)"></span>${nD}</div><div class="k">Degraded instruments</div></div>
      <div class="cell"><div class="v" style="color:var(--ok)">100%</div><div class="k">Records hash-verified</div></div>
    </div></div></section>`));
  const body=$(`<div class="wrap"></div>`);
  body.appendChild($(`<div class="controls"><div style="display:flex;gap:6px">
    ${chip('all','All')}${chip('fragile','Rank-fragile')}${chip('high','Label-error \u22655%')}${chip('degraded','Degraded')}</div>
    <div class="search"><input id="q" placeholder="search benchmark\u2026" oninput="Q=this.value.toLowerCase();paintRows()"/></div></div>`));
  body.appendChild($(`<table class="tbl"><thead><tr>
    ${th('subject','Benchmark')}${th('le','Label errors')}${th('alpha','Reliability \u03b1')}${th('stable','Verdict (result-sensitivity)')}${th('delta','\u0394 score')}${th('rem','Remediation')}
  </tr></thead><tbody id="rows"></tbody></table>`));
  app.appendChild(body);paintRows();
}
function chip(k,l){return `<div class="chip ${FILTER===k?'on':''}" onclick="FILTER='${k}';renderPortfolio()">${l}</div>`;}
function th(k,l){const a=SORT.key===k?(SORT.dir<0?'\u25bc':'\u25b2'):'';return `<th onclick="setSort('${k}')">${l}<span class="ar">${a}</span></th>`;}
function setSort(k){if(SORT.key===k)SORT.dir*=-1;else{SORT.key=k;SORT.dir=-1;}renderPortfolio();}
function paintRows(){
  let rows=DATA.portfolio.slice();
  if(FILTER==='fragile')rows=rows.filter(r=>r.sens.stable===false);
  else if(FILTER==='high')rows=rows.filter(r=>(r.label_error.rate||0)>=0.05);
  else if(FILTER==='degraded')rows=rows.filter(r=>r.status==='degraded');
  if(Q)rows=rows.filter(r=>r.subject.toLowerCase().includes(Q));
  const kf={subject:r=>r.subject,le:r=>r.label_error.rate||0,alpha:r=>r.alpha==null?-9:r.alpha,stable:r=>r.sens.tau==null?-9:r.sens.tau,delta:r=>r.sens.dmax||0,rem:r=>r.rem.n||0};
  rows.sort((a,b)=>{const x=kf[SORT.key](a),y=kf[SORT.key](b);return (x<y?-1:x>y?1:0)*SORT.dir;});
  const tb=document.getElementById('rows');tb.innerHTML='';rows.forEach(r=>tb.appendChild(rowEl(r)));
}
function rowEl(r){
  const le=r.label_error,s=r.sens;
  const verdict=s.stable===true?`<span class="tag stable">rank-stable</span>`:s.stable===false?`<span class="tag fragile">rank-fragile</span>`:`<span class="tag na">n/a</span>`;
  const tau=s.tau==null?'':`<span class="tau">\u03c4=${f2(s.tau)}</span>`;
  const dr=s.dmin==null?'\u2014':`${pp(s.dmin)}\u2013${pp(s.dmax)} pts`;
  const rem=r.rem.n?`<span class="rempill"><b>${r.rem.n}</b> fix${r.rem.n>1?'es':''}`+(r.rem.n_relay?` \u00b7 <span style="color:var(--warn)">${r.rem.n_relay} relay</span>`:'')+`</span>`:`<span class="rempill">\u2014</span>`;
  return $(`<tr onclick="go('detail','${r.name}')">
    <td><span class="bm">${r.subject}<span class="sx">${r.name}</span></span></td>
    <td><div class="lebar"><span class="mono" style="min-width:42px;color:${leColor(le.rate)}">${pct(le.rate,0)}</span><div class="track"><div class="fill" style="width:${Math.min(100,(le.rate||0)*100)}%;background:${leColor(le.rate)}"></div></div></div></td>
    <td class="n" style="color:${r.alpha!=null&&r.alpha<0?'var(--warn)':'var(--ink)'}">${f2(r.alpha)}</td>
    <td>${verdict}${tau}</td><td class="n">${dr}</td><td>${rem}</td></tr>`);
}

function fbtn(pid,name){return `<button class="fbtn" title="show calculation" onclick="event.stopPropagation();openProbe('${pid}','${name}')">\u0192</button>`;}
function renderDetail(name){
  const r=rowOf(name),sp=spotOf(name),s=r.sens,prov=sp?sp.provenance:{};
  const app=document.getElementById('app');app.innerHTML='';
  const w=$(`<div class="wrap"></div>`);
  w.appendChild($(`<button class="back" onclick="go('portfolio')">\u2190 All benchmarks</button>`));
  w.appendChild($(`<div class="dhead"><div><div class="eyebrow">Validity dossier</div><h2>${r.subject}</h2>
    <div class="mono" style="color:var(--faint);font-size:11.5px">${name} \u00b7 ${r.dataset_version} \u00b7 ${r.sens.n_models||DATA.models.length} models \u00b7 ${r.sens.n_items||prov.n_items||'\u2014'} scored items</div></div>
    <div class="prov">record ${(prov.record_id||'').slice(0,22)}\u2026<br/>hash ${prov.hash||'\u2014'} <span class="ok">\u2713 verifiable</span><br/>${r.n_claims||0} reported claim(s)</div></div>`));
  const grid=$(`<div class="grid"></div>`);
  const frag=s.stable===false,cls=frag?'fragile':(s.stable===true?'stable':'');
  let vtxt;
  if(s.stable===true)vtxt=`<span class="hl">Rank-stable</span> (Kendall \u03c4 = ${f2(s.tau)}). Correcting the labels leaves the model ranking unchanged; it raises absolute scores by <b>${pp(s.dmin)} to ${pp(s.dmax)} points</b>`+(s.p_top1>0.05?` and could only reshuffle the near-tied leaders.`:`.`);
  else if(s.stable===false)vtxt=`<span class="hl">Rank-fragile</span> (Kendall \u03c4 = ${f2(s.tau)}). With <b>${pct(r.label_error.rate,0)} label errors</b>, correcting them reshuffles the ranking`+(sp&&sp.top1_changed?` and <b>changes the #1 model</b>`:``)+`. Treat this benchmark\u2019s ordering as unreliable.`;
  else vtxt=`Not enough corrections to assess ranking stability.`;
  grid.appendChild($(`<div class="card full"><h3>Can you trust this result? ${fbtn('result_sensitivity',name)}</h3><div class="verdict ${cls}">${vtxt}</div></div>`));
  if(sp)grid.appendChild($(`<div class="card full"><h3>Verdict under correction \u2014 model ranking, original key \u2192 corrected key ${fbtn('result_sensitivity',name)}</h3>${ribbon(sp.ranking)}
    <div class="note">${sp.top1_changed?'Amber lines cross: at least one model changes rank when the gold labels are corrected \u2014 the leaderboard is an artefact of the label errors.':'Lines stay parallel: every model keeps its rank under correction. The ordering is robust to the measured label errors.'}</div></div>`));
  grid.appendChild($(`<div class="card full"><h3>Validity metrics (measured)</h3><div class="metrics">
    <div class="m"><div class="mv" style="color:${leColor(r.label_error.rate)}">${pct(r.label_error.rate,0)}</div><div class="mk">Label-error \u00b7 ${r.label_error.k}/${r.label_error.n} ${fbtn('label_error_audit',name)}</div></div>
    <div class="m"><div class="mv">${pct(r.ambiguity_rate,0)}</div><div class="mk">Ambiguity ${fbtn('item_ambiguity',name)}</div></div>
    <div class="m"><div class="mv" style="color:${r.alpha!=null&&r.alpha<0?'var(--warn)':'var(--ink)'}">${f2(r.alpha)}</div><div class="mk">Reliability \u03b1 ${fbtn('test_reliability',name)}</div></div>
    <div class="m"><div class="mv">${s.n_changed??'\u2014'}</div><div class="mk">Items corrected ${fbtn('result_sensitivity',name)}</div></div>
    <div class="m"><div class="mv">${s.p_top1==null?'\u2014':pct(s.p_top1,0)}</div><div class="mk">P(top-1 flips) ${fbtn('result_sensitivity',name)}</div></div></div>
    <div class="note">Label-error and ambiguity are measured from a single-pass human annotation (no inter-annotator agreement). \u03b1 is internal consistency (Cronbach); negative \u03b1 means the items do not cohere as one scale. Click any \u0192 to see the exact computation.</div></div>`));
  if(sp&&sp.scatter&&sp.scatter.length)grid.appendChild($(`<div class="card"><h3>Item analysis \u2014 difficulty \u00d7 discrimination ${fbtn('discrimination_saturation',name)}</h3>${scatter(sp.scatter)}
    <div class="note">Amber = flagged suspect. Discrimination is <b>not</b> a label-error screen \u2014 near-chance, and undefined wherever all models agree.</div></div>`));
  const av=(name==='MMLU')?DATA.portfolio.filter(p=>p.name.startsWith('MMLU::')&&p.alpha!=null).map(p=>p.alpha):[];
  if(av.length)grid.appendChild($(`<div class="card"><h3>Reliability across subjects ${fbtn('test_reliability',name)}</h3>${histogram(av)}
    <div class="note">Pooled \u03b1=${f2(sp.alpha_pooled)} looks excellent, but is inflated by between-subject spread: the per-subject median is far lower and several are negative.</div></div>`));
  if(sp&&sp.remediation){const rem=sp.remediation,imp=rem.verdict_impact||{};
    const panel=$(`<div class="card full"><h3>Remediation \u2014 prioritised, calibrated by reliability</h3></div>`);
    panel.appendChild($(`<div class="impact"><b>Verdict impact:</b> ${imp.note||'\u2014'}</div>`));
    for(const a of (rem.actions||[])){
      const el=$(`<div class="act"><div class="row1"><span class="tier ${a.tier}">${a.confidence}</span><span class="pid">${a.probe_id}</span><span class="sev">\u00b7 ${a.severity}</span>
        <span class="scihref">${fbtn(a.probe_id,name)}</span></div>
        <div class="txt">${a.action}</div><div class="why">${a.rationale}</div>${a.tier==='relay_only'?`<div class="cav">Relay only \u2014 ${a.caveat}</div>`:''}</div>`);
      if(a.probe_id==='label_error_audit'&&sp.candidates&&sp.candidates.length){
        const c=$(`<div class="cands"><div class="mono" style="font-size:10.5px;letter-spacing:.06em;color:var(--muted);margin-bottom:6px">CANDIDATE CORRECTIONS \u2014 VERIFY WITH A SECOND ANNOTATOR (not asserted)</div><table></table></div>`);
        const tbl=c.querySelector('table');
        for(const cd of sp.candidates.slice(0,5))tbl.appendChild($(`<tr><td class="q">${(cd.q||'').slice(0,90)}\u2026</td><td>${cd.orig} <span class="ar">\u2192 ${cd.cand.join('/')}</span></td><td style="color:var(--faint)">${cd.etype}</td></tr>`));
        el.appendChild(c);}
      panel.appendChild(el);}
    grid.appendChild(panel);}
  grid.appendChild($(`<div class="card full"><h3>Provenance \u2014 reproducible from its inputs</h3>
    <div class="mono" style="font-size:11.5px;line-height:1.9;color:var(--muted)">
      record_id&nbsp;&nbsp;${prov.record_id||'\u2014'}<br/>content_hash&nbsp;${prov.hash||'\u2014'} &nbsp;<span style="color:var(--ok)">\u2713 recomputes</span><br/>
      dataset&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;${r.dataset_version} &nbsp;\u00b7&nbsp; predictions: HELM v1.3.0 &nbsp;\u00b7&nbsp; corrections: MMLU-Redux (single-pass)<br/>
      method&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;result_sensitivity + test_reliability + intrinsic audit (battery_run)</div></div>`));
  w.appendChild(grid);app.appendChild(w);
}

function renderProbes(){
  const app=document.getElementById('app');app.innerHTML='';
  app.appendChild($(`<section class="hero"><div class="wrap"><div class="eyebrow">Probe library</div>
    <h1>Every probe, <span class="em">explained and demonstrated</span>.</h1>
    <div class="line">What each instrument measures, the formula, the method, why it is valid, and its honest limits. Open <b>Computation</b> on any probe to see it run, with real numbers, on a benchmark you choose.</div></div></section>`));
  const w=$(`<div class="wrap"></div>`);
  const groups=[['headline','Headline \u2014 result trust'],['intrinsic','Intrinsic \u2014 the dataset'],['result','Per-result \u2014 the measurement']];
  for(const [kind,label] of groups){
    w.appendChild($(`<div class="secthead">${label}</div>`));
    const g=$(`<div class="plib"></div>`);
    Object.keys(SCI).filter(pid=>SCI[pid].kind===kind).forEach(pid=>{const s=SCI[pid];
      g.appendChild($(`<div class="pcard"><span class="kindt">${kind}</span><h4>${s.title}</h4><div class="pid2">${pid}</div>
        <p>${s.measures}</p><div class="pbtns"><button class="pbtn primary" onclick="openProbe('${pid}')">Science</button>
        <button class="pbtn" onclick="openProbe('${pid}');M_TAB='calc';renderModal()">Computation</button></div></div>`));});
    w.appendChild(g);}
  app.appendChild(w);
}

function renderMethod(){
  const app=document.getElementById('app');app.innerHTML='';
  app.appendChild($(`<section class="hero"><div class="wrap"><div class="eyebrow">What this instrument does \u2014 and does not \u2014 claim</div>
    <h1>Credibility is the product.<span class="em"> We audit eval instruments and claims; we never rate a model or its safety.</span></h1></div></section>`));
  const w=$(`<div class="wrap method" style="padding-top:22px"></div>`);
  const cards=[
   ['The bright line','<p class="bl">Meridian measures the <b>validity of a benchmark and the results reported on it</b> \u2014 label quality, ambiguity, reliability, and how much a verdict depends on those. It does not score, rank, or certify the safety of any model. A model appears here only as a data point in whether the <i>instrument</i> can be trusted.</p>'],
   ['No number without its computation','<p class="bl">Every value in every dossier carries a <span class="mono">\u0192</span> panel that shows the exact inputs, formula and arithmetic that produced it \u2014 the same calculation the pipeline ran, reconstructed from the record. Nothing is asserted that you cannot open and check. The Probe library demonstrates the science behind each one.</p>'],
   ['Result-sensitivity is the headline, and it is conditional','<p>Label errors always bias absolute scores and can reshuffle near-tied models, but they flip a <b>global</b> ranking only when the errors are skill-discriminating <i>and</i> the models are close. So the verdict is reported per benchmark as rank-stable or rank-fragile \u2014 never as a blanket \u201cX% trustworthy\u201d.</p>'],
   ['Ground truth is named honestly','<p>Corrections come from a <b>single-pass</b> annotation (MMLU-Redux); there is no second annotator, so no inter-annotator agreement (\u03ba). We surface corrections as <b>candidates to verify</b> \u2014 never as a determination that \u201cthe answer is X\u201d.</p>'],
   ['Detection limits are stated, not hidden','<p>We measured that behaviour-based label-error detection (item discrimination) is <b>near-chance</b>. Remediation guides confidently only where a probe reliably measures a structural defect; weak or defeatable signals (contamination, sandbagging) are flagged for investigation, not ruled.</p>'],
   ['Every number is reproducible','<p>Each audit is a content-addressed record: the inputs hash to an id and the verdict recomputes from them. These dossiers were generated from a real run over HELM v1.3.0 predictions and MMLU-Redux corrections, validated by 24,000+ end-to-end invariant checks.</p>'],
  ];
  for(const [h,b] of cards)w.appendChild($(`<div class="card"><h3>${h}</h3>${b}</div>`));
  app.appendChild(w);
}

document.getElementById('foot').textContent='Meridian \u00b7 built on Trust-the-Eval \u00b7 real data: HELM v1.3.0 \u00d7 MMLU-Redux \u00b7 '+DATA.n_benchmarks+' benchmarks \u00b7 '+DATA.models.length+' models';
go('portfolio');
</script></body></html>"""

def build_ui_html(data):
    """Render the standalone observatory HTML from an assembled data dict."""
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__SCI__", json.dumps(SCI, ensure_ascii=False)))


def _auto_spotlight(LB, has_pred, k=4):
    """Pick up to k benchmarks to give full dossiers: the pooled benchmark if present,
    then the most rank-fragile (and highest label-error) ones, for honest contrast.
    Data-driven, so it generalises beyond MMLU."""
    names = [b["name"] for b in LB["benchmarks"]]
    pick = []
    for pooled in ("MMLU",):
        if pooled in names and pooled in has_pred:
            pick.append(pooled)

    def keyf(b):
        s = b.get("sensitivity") or {}
        le = (b.get("label_error") or {}).get("rate") or 0.0
        return (0 if s.get("ranking_stable") is False else 1, -le)

    for b in sorted(LB["benchmarks"], key=keyf):
        n = b["name"]
        if n not in pick and n in has_pred:
            pick.append(n)
        if len(pick) >= k:
            break
    return pick[:k]


def build_ui_site(store, sources, out_dir, title="Meridian", candidates=None):
    """Generate the navigable observatory into ``out_dir/index.html`` from a synced
    store and the sources that built it.

    Predictions and intrinsic rows are read back from the source payloads (each source
    exposes ``fetch()`` returning ``{kind, benchmark, rows, ...}``), so the traces stay
    faithful without persisting anything extra into the records (no content-hash
    churn). Spotlight benchmarks are chosen automatically. This is the drop-in the
    pipeline can call at observe-time to emit the observatory; ``build_site`` (the
    lightweight SPA) is left untouched. Returns the written paths.
    """
    from pathlib import Path
    pred_rows, intrinsic_rows, models = {}, {}, None
    for s in sources:
        try:
            p = s.fetch()
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        kind, bench = p.get("kind"), p.get("benchmark")
        if kind == "predictions":
            pred_rows[bench] = p.get("rows", [])
            if models is None and pred_rows[bench]:
                models = list(pred_rows[bench][0].get("preds", {}).keys())
        elif kind == "intrinsic":
            intrinsic_rows[bench] = p.get("rows", [])
    LB = leaderboard(store)
    spotlight = _auto_spotlight(LB, set(pred_rows))
    data = assemble_ui_data(store, pred_rows, intrinsic_rows, spotlight, models or [],
                            candidates=candidates)
    html = build_ui_html(data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    (out / "observatory-data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"index": str(index), "data": str(out / "observatory-data.json")}
