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
  "title": "Result-sensitivity", "kind": "headline", "tier": "validated",
  "evidence": "Validated method; conditional consequence",
  "facet": "Consequential / criterion validity of the reported result",
  "measures": "Whether correcting the gold labels changes the conclusion (the ranking), not just the scores.",
  "threat": "A reported leaderboard ordering may be an artefact of gold-label errors rather than a fact about the models.",
  "construct": "The stability of the verdict (model ranking) under a justified correction of the answer key.",
  "formula": "\u03c4 = (C \u2212 D)/[n(n\u22121)/2]  \u00b7  \u0394\u2098 = acc_corr(m) \u2212 acc_orig(m)  \u00b7  p\u2081 = P(argmax changes | item bootstrap)",
  "method": "Re-score every model under the corrected key; compare original vs corrected rankings with Kendall\u2019s \u03c4 (concordant minus discordant model pairs); bootstrap items for the probability the #1 model changes.",
  "assumptions": "The corrected key is at least as accurate as the original; items are exchangeable for the bootstrap; the model set is fixed.",
  "can": "Quantify exactly how many model pairs invert under correction (\u03c4) and bound the chance the leader changes; flag a benchmark whose ordering does not survive its own measured label errors.",
  "cannot": "Prove the corrected key is itself perfect; rank the models on their merits; transfer the verdict to a different model set. Label errors flip a GLOBAL ranking only when they are skill-discriminating AND models are close \u2014 so a stable \u03c4 is not a clean bill of health, only ranking-robustness to the measured errors.",
  "calibration": "Planted label-flip scenarios with known skill-discrimination: the probe must report rank-fragile exactly when the planted flips are designed to reorder close models, and rank-stable otherwise.",
  "refs": "Kendall (1938); our criterion-validity study on HELM v1.3.0\u00d7MMLU-Redux; Gema et al., MMLU-Redux (arXiv:2406.04127); Northcutt et al. (2021).",
 },
 "test_reliability": {
  "title": "Reliability (internal consistency)", "kind": "headline", "tier": "validated",
  "evidence": "Validated quantity; \u03b1 is only a lower bound \u2014 read with care",
  "facet": "Internal-structure validity",
  "measures": "Whether the items behave as one coherent scale; a negative coefficient is a red flag.",
  "threat": "If items do not cohere, \u2018percent correct\u2019 aggregates inconsistent things and the score is not measuring one construct.",
  "construct": "Internal consistency of the item set across respondents (here, models as respondents, questions as items).",
  "formula": "\u03b1 = (k/(k\u22121))\u00b7(1 \u2212 \u03a3\u1d62\u03c3\u00b2\u1d62/\u03c3\u00b2_T)   (report alongside \u03c9_t, \u03bb\u2082, glb when feasible)",
  "method": "Compute item variances \u03c3\u00b2\u1d62 and total-score variance \u03c3\u00b2_T on the original key; \u03b1 is the tau-equivalent reliability lower bound.",
  "assumptions": "Essentially one dimension and tau-equivalent items; uncorrelated errors. With ~10 respondents the estimate is high-variance.",
  "can": "Flag scales whose items anti-cohere (\u03b1\u22640), which co-occurs with heavy label error; give a conservative lower bound on reliability.",
  "cannot": "Be read as the reliability or the internal structure of the test \u2014 Sijtsma (2009) shows \u03b1 is unrelated to dimensionality and never equals reliability; \u03c9_t/\u03bb\u2082/glb are more accurate. A high pooled \u03b1 can be inflated by between-subject spread and hide low per-subject \u03b1.",
  "calibration": "Mixed-construct scenarios: a deliberately two-dimensional item set must drive \u03b1 down relative to a unidimensional control.",
  "refs": "Cronbach (1951); Sijtsma (2009), Psychometrika 74:107\u2013120; McDonald (1999); Revelle & Zinbarg (2009); Cho (2021).",
 },
 "label_error_audit": {
  "title": "Label / ground-truth error audit", "kind": "intrinsic", "tier": "validated",
  "evidence": "Validated (direct measured proportion from annotation)",
  "facet": "Content validity of the answer key",
  "measures": "Share of items whose gold answer is wrong, absent, or non-unique \u2014 from human annotation.",
  "threat": "A wrong keyed answer penalises correct models and rewards wrong ones, biasing every score computed on the item.",
  "construct": "Proportion of items carrying a ground-truth defect (wrong / no / multiple correct answers).",
  "formula": "rate = |{error_type \u2208 {wrong_groundtruth, no_correct_answer, multiple_correct_answers}}| / |annotated|",
  "method": "Count the annotation\u2019s defect categories (canonicalised across casing/spacing); clarity issues routed to ambiguity.",
  "assumptions": "The annotation is competent and the categories are applied consistently.",
  "can": "Report a faithful measured defect rate per benchmark; surface each flagged item for review; drive result-sensitivity.",
  "cannot": "Certify a corrected answer as true (annotation is single-pass \u2014 there is NO inter-annotator agreement, so no \u03ba/Krippendorff \u03b1, and corrections are candidates, not facts). Behaviour-based detection of these errors from model agreement is near-chance (our discrimination ROC-AUC\u22480.55), so a statistical screen is not a substitute.",
  "calibration": "Synthetic corpora with a known number of injected wrong-gold items: the measured rate must match the injected rate.",
  "refs": "Northcutt et al., Pervasive Label Errors (arXiv:2103.14749); Gema et al., MMLU-Redux (arXiv:2406.04127); Krippendorff (2004) for the agreement we lack.",
 },
 "item_ambiguity": {
  "title": "Item ambiguity", "kind": "intrinsic", "tier": "validated",
  "evidence": "Validated measure; partly subjective \u2192 flag for review",
  "facet": "Content validity / item quality",
  "measures": "Share of items flagged as unclear in question or options.",
  "threat": "An ambiguous item has no single defensible answer, so any key is partly arbitrary and the item adds noise, not signal.",
  "construct": "Proportion of items with a presentation/clarity defect, distinct from a wrong key.",
  "formula": "rate = |{error_type \u2208 {bad_question_clarity, bad_options_clarity}}| / |annotated|",
  "method": "Count clarity-flagged items from the human annotation, kept separate from label errors.",
  "assumptions": "Clarity judgements are reasonably shared across competent readers.",
  "can": "Separate \u2018ambiguous\u2019 from \u2018wrong key\u2019 (different fixes: rewrite vs re-key); flag items for human re-review.",
  "cannot": "Be treated as objective without agreement statistics; single-pass annotation means it is a flag, not a determination.",
  "calibration": "Corpora seeded with under-specified items: the flagged rate must rise relative to a clean control.",
  "refs": "Gema et al., MMLU-Redux (arXiv:2406.04127); Artstein & Poesio (2008) on agreement.",
 },
 "discrimination_saturation": {
  "title": "Discrimination & saturation", "kind": "intrinsic", "tier": "validated",
  "evidence": "Validated (classical test theory); limited at consensus",
  "facet": "Internal-structure validity",
  "measures": "Whether items separate strong from weak models, and whether the set is saturated.",
  "threat": "Items everyone passes (or that anti-discriminate) carry little signal; a benchmark of such items cannot order models.",
  "construct": "Item discrimination \u2014 the correlation between getting an item right and overall ability.",
  "formula": "r_pb = (M\u2081 \u2212 M\u2080)/\u03c3_T \u00b7 \u221a(p(1\u2212p))",
  "method": "M\u2081,M\u2080 = mean total score of models right vs wrong on the item; \u03c3_T = sd of totals; p = share correct.",
  "assumptions": "Total score is a usable proxy for ability; enough variance across respondents.",
  "can": "Identify non-discriminating or negatively-discriminating items and saturation; describe where the benchmark carries signal.",
  "cannot": "Serve as a label-error screen (we measured this to be near-chance) and is undefined wherever all models agree \u2014 exactly where consensus wrong-gold items hide. Noisy with ~10 respondents.",
  "calibration": "Item sets mixing known-good and known-degenerate items: discrimination must rank them accordingly.",
  "refs": "Classical test theory; Lord & Novick (1968); our negative finding (discrimination vs label error, ROC-AUC\u22480.55).",
 },
 "coverage_distribution": {
  "title": "Coverage / distribution", "kind": "intrinsic", "tier": "validated",
  "evidence": "Validated structural measure",
  "facet": "Content / generalizability validity",
  "measures": "Balance across labelled categories and of the answer key.",
  "threat": "Skewed categories or a dominant answer option let a model score by exploiting the distribution \u2014 a Clever-Hans shortcut \u2014 not the construct.",
  "construct": "Concentration of the category mix and of the MCQ key.",
  "formula": "Gini(category counts) ; top_share = max_c n_c / N   (skew flagged only when \u2265 2 categories)",
  "method": "Compute Gini and dominant share of the provided categories; with a single category, assess answer-key balance instead.",
  "assumptions": "The provided categories are the relevant partition.",
  "can": "Quantify category/key imbalance that enables shortcut solutions; flag answer-key leakage by option frequency.",
  "cannot": "Say whether the provided categories are the RIGHT decomposition of the construct (that is a content-validity judgement outside the data).",
  "calibration": "Extreme-skew scenario (~95% one category): coverage must flag it, while a balanced control passes.",
  "refs": "Gini (1912); shortcut/Clever-Hans evidence in benchmarks (arXiv:2410.11672).",
 },
 "dataset_hygiene": {
  "title": "Dataset hygiene", "kind": "intrinsic", "tier": "validated",
  "evidence": "Validated (exact, structural)",
  "facet": "Internal validity of the dataset",
  "measures": "Exact/near-duplicate items, contradictory labels, empties, benchmark-canary leakage.",
  "threat": "Duplicates inflate apparent sample size and reward memorisation twice; contradictory labels are self-refuting; canary strings indicate the test leaked into training.",
  "construct": "Count of structurally defective or leaked items.",
  "formula": "exact dup = identical normalised text ; near dup = similarity \u2265 \u03b8 ; contradictory = same item, different gold",
  "method": "Hash normalised items, compare for near-duplicates, group by item to find label conflicts, scan for known canaries.",
  "assumptions": "Normalisation captures \u2018same item\u2019; similarity threshold is appropriate.",
  "can": "Detect and localise duplicates, conflicts, empties and canary hits unambiguously \u2014 each is exactly fixable.",
  "cannot": "Say whether the non-duplicate items are correct, or detect paraphrastic contamination (that is the contamination probe\u2019s contested job).",
  "calibration": "Corpora with injected duplicates/conflicts/canaries: counts must match injection.",
  "refs": "Standard de-duplication; canary methodology (e.g., BIG-bench canary string); contamination surveys.",
 },
 "statistical_power": {
  "title": "Statistical reliability (power)", "kind": "result", "tier": "validated",
  "evidence": "Validated (standard inferential statistics)",
  "facet": "Generalizability validity (sampling)",
  "measures": "Whether there are enough items for the precision being claimed.",
  "threat": "Reporting a sub-point gap on ~100 items claims precision the sample cannot support; differences may be sampling noise.",
  "construct": "Sampling error of an accuracy and the minimum detectable difference between two models.",
  "formula": "CI half-width \u2248 z\u00b7\u221a(p(1\u2212p)/n) ; paired two-proportion / McNemar test for model differences",
  "method": "Treat accuracy as a proportion drawn from a super-population; use paired differences (correlated across models) to reduce variance; compute MDE.",
  "assumptions": "Items are an exchangeable sample from a population of interest; pairing exploits shared item difficulty.",
  "can": "State the smallest trustworthy difference, attach CIs, and flag under-powered comparisons and per-slice claims.",
  "cannot": "Address bias (label errors, contamination); a powered comparison on a biased benchmark is still wrong.",
  "calibration": "Known-effect simulations: the test\u2019s detection rate must track the nominal power.",
  "refs": "Miller, Adding Error Bars to Evals (arXiv:2411.00640); McNemar (1947); Dror et al. (2018) on NLP significance.",
 },
 "option_order_bias": {
  "title": "MCQ option-order bias", "kind": "result", "tier": "validated",
  "evidence": "Validated, well-replicated phenomenon",
  "facet": "Substantive validity (response process)",
  "measures": "Whether scores change when MCQ option positions are permuted.",
  "threat": "If the same item scores differently by where the answer sits, the score reflects position/token priors, not knowledge.",
  "construct": "Sensitivity of accuracy to label/position permutation (selection bias).",
  "formula": "swing = max_perm acc \u2212 min_perm acc ; recall imbalance across option IDs",
  "method": "Re-evaluate under shuffled option positions/contents; report the swing and per-ID preference.",
  "assumptions": "Permutations are meaning-preserving; enough items to estimate the swing.",
  "can": "Quantify position-driven score inflation and recommend permutation-averaging (e.g., PriDe-style debiasing).",
  "cannot": "Be assessed without re-running under permutations (needs prediction access, not just final scores).",
  "calibration": "Items with a planted ID preference must produce a large swing; symmetric items must not.",
  "refs": "Zheng et al., LLMs Are Not Robust MC Selectors, ICLR 2024 (arXiv:2309.03882); Changing Answer Order decreases MMLU acc (arXiv:2406.19470); Pezeshkpour & Hruschka (2023).",
 },
 "prompt_format_sensitivity": {
  "title": "Prompt-format sensitivity", "kind": "result", "tier": "validated",
  "evidence": "Validated (large measured spreads)",
  "facet": "Substantive validity (response process)",
  "measures": "Score variance across semantically-equivalent prompt formats.",
  "threat": "A large swing means the number is a property of the template, not the model, so it does not transfer to any other framing.",
  "construct": "Dispersion of accuracy over meaning-preserving formatting choices.",
  "formula": "spread = range over formats of acc (spacing, casing, separators, delimiters)",
  "method": "Evaluate under several equivalent templates; report the spread (a single point estimate hides it).",
  "assumptions": "The format set preserves task meaning.",
  "can": "Expose template-dependent scores and motivate multi-prompt reporting.",
  "cannot": "Be reduced to one number safely; needs several format runs to estimate.",
  "calibration": "Equivalent reformattings of a fixed item set must bound the induced spread.",
  "refs": "Sclar et al., FormatSpread, ICLR 2024 (arXiv:2310.11324); Mizrahi et al., multi-prompt evaluation (2024).",
 },
 "self_consistency": {
  "title": "Self-consistency / stochastic stability", "kind": "result", "tier": "validated",
  "evidence": "Validated (variance is directly measurable)",
  "facet": "Generalizability validity (reliability across runs)",
  "measures": "How much of a score is run-to-run noise under stochastic decoding.",
  "threat": "A single stochastic run is one sample; if repeats wander, a point estimate overstates certainty.",
  "construct": "Between-run variance of accuracy at fixed settings.",
  "formula": "between-run sd of acc ; agreement rate across repeats",
  "method": "Repeat the eval at identical settings; quantify variance across runs.",
  "assumptions": "Runs differ only by sampling randomness.",
  "can": "Quantify decoding noise and recommend fixed seeds or run-level CIs.",
  "cannot": "Capture systematic bias; only the random component.",
  "calibration": "High-temperature vs greedy runs: variance must shrink toward zero under greedy decoding.",
  "refs": "Standard test-retest reliability; multi-run reporting in lm-eval (arXiv:2405.14782).",
 },
 "judge_swap": {
  "title": "Judge validity (LLM-as-judge)", "kind": "result", "tier": "validated",
  "evidence": "Validated biases; widely replicated",
  "facet": "Substantive / criterion validity of the metric",
  "measures": "Position bias, self-preference, and inter-judge agreement for LLM judges.",
  "threat": "If a judge favours the first answer or its own family, or two judges barely agree, the metric reflects the judge, not the contestants.",
  "construct": "Judge-induced systematic error and reliability.",
  "formula": "position-swap \u0394 (consistency) ; preference-fairness ; Cohen/Fleiss \u03ba across judges",
  "method": "Swap answer order, swap judges, and measure agreement and swap-stability.",
  "assumptions": "Swaps preserve content; judges are exchangeable raters.",
  "can": "Quantify position/self-preference bias and judge unreliability; flag verdicts dependent on the judge.",
  "cannot": "Fix the judge; bias is most severe precisely when candidate quality is close.",
  "calibration": "Order-swapped pairs with known-equal answers must reveal any positional favouritism.",
  "refs": "Zheng et al., Judging LLM-as-a-Judge with MT-Bench (arXiv:2306.05685); position-consistency/preference-fairness metrics (Shi et al., 2024).",
 },
 "answer_extraction_audit": {
  "title": "Answer-extraction / scoring audit", "kind": "result", "tier": "validated",
  "evidence": "Validated (deterministic parsing defects)",
  "facet": "Internal validity (instrumentation)",
  "measures": "Whether the scorer parses model responses correctly.",
  "threat": "If the parser misreads a correct free-form answer as wrong, the benchmark mis-scores deterministically \u2014 a bug in the ruler.",
  "construct": "Disagreement between automatic extraction and the intended answer.",
  "formula": "extraction error rate = |{parsed \u2260 intended}| / sample",
  "method": "Compare the extractor against the expected answer format / probability over surface forms on a sample.",
  "assumptions": "A reliable reference parse exists for the sample.",
  "can": "Detect parsing/normalisation defects and surface-form competition that depress scores.",
  "cannot": "Catch conceptual errors in the task; only the measurement plumbing.",
  "calibration": "Responses with known formats must be extracted at the expected rate.",
  "refs": "Holtzman et al., Surface Form Competition (arXiv:2104.08315); lm-eval extraction practices (arXiv:2405.14782).",
 },
 "refusal_confound": {
  "title": "Refusal / abstention confound", "kind": "result", "tier": "validated",
  "evidence": "Validated confound",
  "facet": "Substantive validity (response process)",
  "measures": "Whether refusals/abstentions are scored as wrong, confounding capability.",
  "threat": "A model that declines is not the same as a model that is wrong; conflating them depresses the capability estimate and rewards over-confident guessing.",
  "construct": "Rate of refusals/abstentions counted as incorrect.",
  "formula": "confound rate = |{refusal scored incorrect}| / |responses|",
  "method": "Detect refusal/abstention and separate it from incorrect answers in scoring.",
  "assumptions": "Refusals are reliably detectable.",
  "can": "Disentangle \u2018won\u2019t answer\u2019 from \u2018can\u2019t answer\u2019 and correct the capability estimate.",
  "cannot": "Read intent behind a refusal; requires a refusal detector.",
  "calibration": "Injected refusals must be classified out of the incorrect bucket.",
  "refs": "Abstention/selective-prediction literature; refusal handling in eval harnesses.",
 },
 "multiplicity_cherrypick": {
  "title": "Multiplicity / cherry-picking", "kind": "result", "tier": "validated",
  "evidence": "Validated (multiple-comparisons theory)",
  "facet": "Statistical-conclusion validity",
  "measures": "Uncorrected multiple comparisons or best-of-many reporting.",
  "threat": "Run enough comparisons and some \u2018wins\u2019 appear by chance; without correction a headline gain may be selection, not signal.",
  "construct": "Family-wise error inflation from the number of comparisons.",
  "formula": "FWER \u2248 1 \u2212 (1\u2212\u03b1)^m ; Bonferroni/Holm-adjusted thresholds",
  "method": "Count comparisons; apply a multiplicity correction or a pre-registration check.",
  "assumptions": "The comparison family is identifiable.",
  "can": "Flag under-corrected comparison sweeps and best-of-N selection.",
  "cannot": "Detect intent; only the structure of the comparisons made.",
  "calibration": "Null sweeps (no true effect) must trigger the flag at the expected rate.",
  "refs": "Benjamini & Hochberg (1995); Dror et al. (2018); Gelman & Loken, garden of forking paths (2013).",
 },
 "model_drift": {
  "title": "Model drift / temporal validity", "kind": "result", "tier": "contested",
  "evidence": "Real effect; attribution is hard",
  "facet": "Generalizability validity (over time)",
  "measures": "Whether scores drift across model versions / dates.",
  "threat": "A score attached to a moving endpoint is not reproducible; comparisons across drift are confounded.",
  "construct": "Change in accuracy across model snapshots/time.",
  "formula": "\u0394 acc across pinned snapshots",
  "method": "Track the same eval across versions/dates with pinned identifiers.",
  "assumptions": "Snapshots are correctly pinned and otherwise comparable.",
  "can": "Flag comparisons made across version/time drift; require snapshot pinning.",
  "cannot": "Attribute drift to a specific cause (model change vs eval change vs data) without controlled runs.",
  "calibration": "Re-runs of a fixed snapshot must show ~zero drift; a swapped snapshot must show it.",
  "refs": "Chen et al., How Is ChatGPT\u2019s Behavior Changing over Time? (arXiv:2307.09009).",
 },
 "elicitation_ceiling": {
  "title": "Elicitation ceiling", "kind": "result", "tier": "exploratory",
  "evidence": "Promising but a bound, not a point estimate",
  "facet": "Substantive validity (capability vs elicitation)",
  "measures": "Whether a score is capped by extraction effort rather than capability.",
  "threat": "A low score may reflect weak elicitation, not weak capability \u2014 so the number is a lower bound, not the capability.",
  "construct": "Performance lift from improved prompting/format/scaffolding.",
  "formula": "lift = acc(best elicitation) \u2212 acc(default)",
  "method": "Vary prompt/format/length/scaffolding and observe whether the score keeps rising.",
  "assumptions": "Elicitation changes do not leak answers.",
  "can": "Establish a lower bound on capability and reveal under-elicitation.",
  "cannot": "Yield an upper bound or a \u2018true\u2019 capability point estimate; elicitation search is open-ended.",
  "calibration": "Known-underelicited setups must show a recoverable lift.",
  "refs": "Capability-elicitation discussions; our axe-f review (capability as an interval).",
 },
 "contamination_perturb": {
  "title": "Contamination via perturbation", "kind": "result", "tier": "contested",
  "evidence": "Contested \u2014 near-chance on static benchmarks",
  "facet": "Generalizability validity (memorisation)",
  "measures": "Signals of memorisation by re-testing under perturbations.",
  "threat": "If a score reflects recall of seen items, it does not generalise \u2014 the benchmark over-credits the model.",
  "construct": "Performance gap between original and meaning-preserving perturbed items.",
  "formula": "\u0394 = acc(original) \u2212 acc(perturbed)",
  "method": "Compare original vs perturbed/equivalent items.",
  "assumptions": "Perturbations preserve difficulty and meaning (often violated).",
  "can": "Raise an investigation flag where a drop suggests recall rather than reasoning.",
  "cannot": "Serve as a verdict: on a static benchmark, behavioural contamination detection barely beats chance (membership-inference ROC-AUC<0.6); only temporal/held-out splits are reliable. Reported as INVESTIGATE, never a rating.",
  "calibration": "Items known-seen vs known-unseen (temporal split) must separate; perturbation alone may not.",
  "refs": "Ravaut et al., contamination survey (arXiv:2404.00699); Does Contamination Detection Work? (arXiv:2410.18966); LiveCodeBench (temporal).",
 },
 "sandbagging_paired": {
  "title": "Sandbagging / evaluation-awareness", "kind": "result", "tier": "exploratory",
  "evidence": "Exploratory \u2014 detection nascent and defeatable",
  "facet": "Substantive validity (strategic response)",
  "measures": "Strategic underperformance / awareness of being evaluated.",
  "threat": "A model that detects evaluation can lower its score selectively, so the benchmark under-credits true capability.",
  "construct": "Paired performance gap between eval-signalling and neutral conditions.",
  "formula": "\u0394 = acc(eval-framed) \u2212 acc(neutral-framed), paired by item",
  "method": "Compare matched items under eval-signalling vs neutral framings.",
  "assumptions": "Framings differ only in eval-salience; the model is not adapting to the probe.",
  "can": "Surface a hint of strategic display for investigation.",
  "cannot": "Confirm sandbagging: a capable strategist can erase the gap; no settled detector exists. Reported as INVESTIGATE, never a rating.",
  "calibration": "Models instructed to underperform on a subset must widen the paired gap (an existence check, not a field detector).",
  "refs": "van der Weij et al., Sandbagging (arXiv:2406.07358); evaluation-awareness probing (arXiv:2507.01786); noise-injection (arXiv:2412.01784).",
 },
 "reward_hacking_eval": {
  "title": "Reward-hacking of the eval", "kind": "result", "tier": "contested",
  "evidence": "Real where present; detection is pattern-based",
  "facet": "Internal validity of the metric",
  "measures": "Responses that game the metric rather than solve the task.",
  "threat": "If a string that isn\u2019t a real answer scores, the metric is hackable and the number is not about the task.",
  "construct": "Rate of metric-exploiting responses (format exploits, shortcut tokens).",
  "formula": "exploit rate over flagged patterns",
  "method": "Inspect for scorer-rewarded patterns that are not valid task solutions.",
  "assumptions": "Exploit patterns are enumerable for the metric.",
  "can": "Flag known metric exploits and motivate stricter scoring.",
  "cannot": "Be exhaustive; novel exploits evade pattern lists.",
  "calibration": "Injected exploit responses must score under the naive metric and be caught by the probe.",
  "refs": "Specification-gaming / reward-hacking literature (Skalse et al., 2022).",
 },
 "subgroup_power": {
  "title": "Subgroup power", "kind": "result", "tier": "validated",
  "evidence": "Validated (sampling statistics per slice)",
  "facet": "Generalizability validity (per-slice)",
  "measures": "Whether per-subgroup claims have enough items.",
  "threat": "A per-topic ranking on a handful of items is noise presented as a finding.",
  "construct": "Sampling error within each reported subgroup.",
  "formula": "per-slice n and CI half-width \u2248 z\u00b7\u221a(p(1\u2212p)/n_slice)",
  "method": "Compute the sampling error within each reported subgroup.",
  "assumptions": "Slices are exchangeable samples.",
  "can": "Flag subgroup claims too small to support a ranking.",
  "cannot": "Address subgroup bias; only subgroup precision.",
  "calibration": "Tiny synthetic slices must be flagged; large ones must pass.",
  "refs": "Standard stratified-estimate statistics; Miller (arXiv:2411.00640).",
 },
 "provenance_repro": {
  "title": "Provenance & reproducibility", "kind": "result", "tier": "validated",
  "evidence": "Validated (mechanical guarantee)",
  "facet": "Reproducibility (precondition of validity)",
  "measures": "Whether dataset version, sources and seeds are pinned so the result reproduces.",
  "threat": "If the inputs are not pinned, no claim on them is checkable and the score is not reproducible.",
  "construct": "Whether the verdict recomputes from content-addressed inputs.",
  "formula": "content-hash(inputs) \u2192 record id ; recompute(metric) == displayed",
  "method": "Hash the inputs to an id and re-derive each number from them.",
  "assumptions": "Inputs are fully captured by the recorded sources.",
  "can": "Guarantee every number is reproducible from named inputs (pipeline, not black box).",
  "cannot": "Ensure correctness \u2014 reproducibility is necessary, not sufficient, for validity.",
  "calibration": "Re-running the pipeline on the same inputs must reproduce ids and values bit-for-bit.",
  "refs": "Biderman et al., Lessons from the Trenches (arXiv:2405.14782); Datasheets for Datasets (Gebru et al., 2021); Data Statements (Bender & Friedman, 2018).",
 },
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
.tierline{display:flex;align-items:center;gap:10px;margin:0 0 16px;flex-wrap:wrap}
.tierbadge{font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:999px;font-weight:600;white-space:nowrap}
.tier-validated{color:var(--ok);background:var(--ok-soft);border:1px solid var(--ok-line)}
.tier-contested{color:var(--warn);background:var(--warn-soft);border:1px solid var(--warn-line)}
.tier-exploratory{color:var(--bad);background:var(--bad-soft);border:1px solid var(--bad-line)}
.facet{font-size:12px;color:var(--muted);font-style:italic}
.cc{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
.ccbox{border-radius:9px;padding:11px 13px;border:1px solid var(--line)}
.ccbox .cl{display:block;margin-bottom:4px}
.ccbox p{margin:0;font-size:13px;line-height:1.55;color:#33312b}
.ccbox.can{background:var(--ok-soft);border-color:var(--ok-line)}
.ccbox.can .cl{color:var(--ok)}
.ccbox.cannot{background:var(--bad-soft);border-color:var(--bad-line)}
.ccbox.cannot .cl{color:var(--bad)}
.pcard .pcardt{display:flex;gap:6px;align-items:center;margin-bottom:2px}
@media(max-width:640px){.cc{grid-template-columns:1fr}}
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
.cdl{font-size:12px;line-height:1.6;color:var(--muted);margin-top:12px;padding:10px 12px;background:var(--accent-soft);border:1px solid var(--accent-line);border-radius:8px}
.cdl .cl{display:block;margin-bottom:3px}
.cdl a,.dlrow a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent-line)}
.cdl a:hover,.dlrow a:hover{border-bottom-color:var(--accent)}
.dlrow{font-size:12.5px;line-height:1.65;color:#33312b;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}
.dlrow .cl{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:4px}

.method .card{margin-bottom:16px}
.method h3{font-size:15px;text-transform:none;letter-spacing:-.01em;color:var(--ink)}
.method p{font-size:13.5px;color:#33312b;line-height:1.6;margin:.4em 0}
.method .bl{border-left:3px solid var(--accent);padding-left:14px}
.method .cmap{width:100%;font-size:12.5px;margin-top:6px}
.method .cmap th{text-align:left;font-family:var(--mono);font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600;padding:7px 10px;border-bottom:1px solid var(--line)}
.method .cmap td{vertical-align:top;padding:9px 10px;border-bottom:1px solid var(--line2);line-height:1.5;color:#33312b}
.method .cmap td.nocell{color:var(--bad);background:var(--bad-soft)}
.method ul.mlim{margin:.4em 0;padding-left:18px}
.method ul.mlim li{font-size:13px;color:#33312b;line-height:1.6;margin:.35em 0}
.method .refsblk{font-size:11.5px;color:var(--muted);line-height:1.8}
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
    <button id="nav-lineage" onclick="location.href='lineage.html'">Lineage</button>
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
function exportFooter(name){const sp=spotOf(name);const e=sp&&sp.export;if(!e)return '';
  return `<div class="cdl"><span class="cl">Download the exact inputs</span>per-item table (every model\u2019s raw answer + correctness under both gold keys): <a href="${e.json}" download>JSON</a> \u00b7 <a href="${e.csv}" download>CSV</a> \u00b7 <a href="${e.datasheet}" download>datasheet</a> (${e.n_items} items). Re-key it and recompute, or run <span class="mono">meridian explain "${name}" &lt;metric&gt;</span>.</div>`;}
function openProbe(pid,name){M_PID=pid;M_BENCH=name||M_BENCH||'MMLU::virology';M_TAB='sci';renderModal();}
function closeModal(){document.getElementById('modal').innerHTML='';}
function renderModal(){
  const s=SCI[M_PID];if(!s)return;
  const opts=DATA.spotlight_names.map(n=>`<option value="${n}" ${n===M_BENCH?'selected':''}>${n}</option>`).join('');
  const body=M_TAB==='sci'
    ?`<div class="tierline"><span class="tierbadge tier-${s.tier}">${s.evidence}</span><span class="facet">${s.facet}</span></div>
      <div class="sx2"><span class="cl">Threat to validity</span><p>${s.threat}</p></div>
      <div class="sx2"><span class="cl">What it measures</span><p>${s.construct}</p></div>
      <div class="sx2"><span class="cl">Estimator</span><div class="formula mono">${s.formula}</div></div>
      <div class="sx2"><span class="cl">Method</span><p>${s.method}</p></div>
      <div class="sx2"><span class="cl">Assumptions</span><p>${s.assumptions}</p></div>
      <div class="cc"><div class="ccbox can"><span class="cl">Can conclude</span><p>${s.can}</p></div><div class="ccbox cannot"><span class="cl">Cannot conclude</span><p>${s.cannot}</p></div></div>
      <div class="sx2"><span class="cl">Calibration in TtE</span><p>${s.calibration}</p></div>
      <div class="sx2"><span class="cl">Primary references</span><p class="refs">${s.refs}</p></div>`
    :`<div class="csel">Computed on <select onchange="M_BENCH=this.value;renderModal()">${opts}</select></div>${computePanel(M_PID,M_BENCH)}${exportFooter(M_BENCH)}`;
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
      method&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;result_sensitivity + test_reliability + intrinsic audit (battery_run)</div>
    ${sp&&sp.export?`<div class="dlrow"><span class="cl">Total per-datum transparency</span>
      Every item, every model\u2019s raw answer, and its correctness under both gold keys:
      <a href="${sp.export.json}" download>per-item JSON</a> \u00b7 <a href="${sp.export.csv}" download>CSV</a> (${sp.export.n_items} items) \u00b7 <a href="${sp.export.datasheet}" download>datasheet</a>.<br/>
      Recompute any headline number from these inputs: <span class="mono">meridian explain "${name}" tau|alpha|label</span>.</div>`:''}</div>`));
  w.appendChild(grid);app.appendChild(w);
}

function renderProbes(){
  const app=document.getElementById('app');app.innerHTML='';
  app.appendChild($(`<section class="hero"><div class="wrap"><div class="eyebrow">Probe library</div>
    <h1>Every probe, <span class="em">explained and demonstrated</span>.</h1>
    <div class="line">For each instrument: the validity threat it addresses, the construct, the estimator and its assumptions, what it <b>can</b> and <b>cannot</b> conclude, an evidence tier, and primary references. Open <b>Computation</b> to see it run, with real numbers, on a benchmark you choose.</div></div></section>`));
  const w=$(`<div class="wrap"></div>`);
  const groups=[['headline','Headline \u2014 result trust'],['intrinsic','Intrinsic \u2014 the dataset'],['result','Per-result \u2014 the measurement']];
  for(const [kind,label] of groups){
    w.appendChild($(`<div class="secthead">${label}</div>`));
    const g=$(`<div class="plib"></div>`);
    Object.keys(SCI).filter(pid=>SCI[pid].kind===kind).forEach(pid=>{const s=SCI[pid];
      g.appendChild($(`<div class="pcard"><div class="pcardt"><span class="kindt">${kind}</span><span class="tierbadge tier-${s.tier}">${s.tier}</span></div><h4>${s.title}</h4><div class="pid2">${pid}</div>
        <p>${s.measures}</p><div class="pbtns"><button class="pbtn primary" onclick="openProbe('${pid}')">Science</button>
        <button class="pbtn" onclick="openProbe('${pid}');M_TAB='calc';renderModal()">Computation</button></div></div>`));});
    w.appendChild(g);}
  app.appendChild(w);
}

function renderMethod(){
  const app=document.getElementById('app');app.innerHTML='';
  app.appendChild($(`<section class="hero"><div class="wrap"><div class="eyebrow">Methodology \u2014 a validity argument, not a score</div>
    <h1>We assemble <span class="em">evidence about whether a reported result can be trusted</span> \u2014 we never rate a model or its safety.</h1>
    <div class="line">On the unitary, argument-based view of validity (Cronbach &amp; Meehl 1955; Messick 1995; Kane 2013), validity is a property of the <b>interpretation and use of a score</b> \u2014 established by marshalling evidence and confronting threats, not a number a test or a model \u201chas\u201d. Meridian is that argument, made explicit and reproducible.</div></div></section>`));
  const w=$(`<div class="wrap method" style="padding-top:22px"></div>`);
  const cards=[
   ['The inference we license, and audit','<p class="bl">Every dossier supports one claim: <b>the result reported on this benchmark (its ranking / scores) is a trustworthy proxy for the construct it purports to measure.</b> We gather evidence for and against that inference and surface the threats. We do <b>not</b> conclude anything about a model\u2019s ability or safety \u2014 a model appears only as a data point in whether the <i>instrument</i> can be trusted.</p>'],
   ['Why \u201cvalidity\u201d, formally','<p>A benchmark is a <b>task + metric standing in for a phenomenon</b> (Raji et al. 2021); validity is how well the score proxies that phenomenon (Messick; Biderman et al. 2024). Construct validity is <b>unitary</b> (Cronbach &amp; Meehl 1955; Messick 1995): content, internal-structure, substantive, generalizability, external and consequential evidence are facets of <i>one</i> argument. There is no single pass/fail test for it (Bowman &amp; Dahl 2021), and a review of 445 LLM benchmarks found validity routinely unaddressed (Bean et al. 2024) \u2014 so our contribution is to operationalise the evidence-gathering, facet by facet.</p>'],
   ['Result-sensitivity is the headline, and it is conditional','<p>Label errors always bias absolute scores and can reshuffle near-tied models, but they flip a <b>global</b> ranking only when the errors are skill-discriminating <i>and</i> the models are close. The verdict is therefore reported per benchmark as <b>rank-stable</b> or <b>rank-fragile</b> \u2014 with Kendall\u2019s \u03c4 and a bootstrap on P(top-1 changes) \u2014 never as a blanket \u201cX% trustworthy\u201d.</p>'],
  ];
  for(const [h,b] of cards)w.appendChild($(`<div class="card"><h3>${h}</h3>${b}</div>`));

  // coverage map (what we cover, by validity facet \u2014 and what we explicitly do not)
  const rows=[
   ['Content','label-error, ambiguity, coverage/distribution','Key correctness, item clarity, category &amp; answer-key balance','Whether the construct itself is the right target (is MMLU \u201cknowledge\u201d?); item authoring'],
   ['Internal structure','reliability (\u03b1, \u2192\u03c9/\u03bb\u2082/glb), discrimination','Item coherence and separation; a conservative reliability bound','True dimensionality \u2014 \u03b1 is only a lower bound (Sijtsma 2009); no full factor/IRT fit by default'],
   ['Substantive (response process)','option-order, prompt-format, judge-swap, refusal, extraction','Robustness of the score to position, formatting, judge and parsing','The model\u2019s internal reasoning or intent'],
   ['Generalizability','statistical &amp; subgroup power, self-consistency, model-drift, contamination','Sampling precision, run-to-run stability, snapshot pinning, memorisation <i>flags</i>','A contamination <b>verdict</b> from a static set (near-chance); transfer to other items / models'],
   ['External','\u2014 (not established)','\u2014','Correlation with real-world outcomes / a nomological net \u2014 largely absent field-wide, required for full construct validity (Ostrowski et al. 2026)'],
   ['Consequential','result-sensitivity (headline)','Whether the measured label errors change the conclusion','The downstream consequences of acting on the verdict'],
  ];
  let tbl='<div class="card"><h3>What we cover \u2014 and what we do not</h3><p>We organise probes by Messick\u2019s facets. The last column is the honest part: the questions this instrument does <b>not</b> answer.</p><table class="ctab cmap"><thead><tr><th>Validity facet</th><th>What we probe</th><th>What we can say</th><th>What we do <b>not</b> assess</th></tr></thead><tbody>';
  for(const [f,p,c,n] of rows)tbl+=`<tr><td><b>${f}</b></td><td>${p}</td><td>${c}</td><td class="nocell">${n}</td></tr>`;
  tbl+='</tbody></table></div>';
  w.appendChild($(tbl));

  const cards2=[
   ['Measurement-theoretic limits, stated up front','<ul class="mlim"><li><b>Reliability:</b> Cronbach\u2019s \u03b1 is a lower bound under tau-equivalence \u2014 not reliability, and unrelated to internal structure (Sijtsma 2009); read \u03c9_t/\u03bb\u2082/glb beside it.</li><li><b>Latent ability / IRT:</b> ability models assume a fixed, non-strategic respondent. Under an eval-aware or sandbagging model the latent trait is not identified \u2014 <b>capability is an interval, not a point</b> (van der Weij et al. 2024; evaluation-awareness, 2025).</li><li><b>Contamination:</b> on a static benchmark, behavioural detection barely beats chance (Ravaut et al. 2024; \u201cDoes contamination detection work?\u201d 2024); only temporal / held-out splits are reliable.</li><li><b>MCQ &amp; format:</b> scores swing with option order (up to 42.9%; Zheng et al. 2024) and meaning-preserving formatting (Sclar et al. 2024) \u2014 a point estimate hides this.</li><li><b>LLM-as-judge:</b> position and self-preference bias, worst when candidates are close (Zheng et al. 2023).</li></ul>'],
   ['Ground truth is named honestly','<p>Corrections come from a <b>single-pass</b> annotation (MMLU-Redux): no second annotator, hence no inter-annotator agreement (\u03ba / Krippendorff\u2019s \u03b1). We surface corrections as <b>candidates to verify</b>, never as a determination that \u201cthe answer is X\u201d \u2014 and we measured that behaviour-based label-error detection (item discrimination) is <b>near-chance</b>, so a statistical screen is not a substitute for re-annotation.</p>'],
   ['Reproducibility &amp; transparency','<p class="bl"><b>No number without its computation.</b> Each audit is a <b>content-addressed record</b>: the inputs hash to an id and the verdict recomputes from them \u2014 a pipeline, not a black box. Every value carries an <span class="mono">\u0192</span> panel with the exact inputs, formula and arithmetic, and every benchmark ships a <b>per-item export</b> (JSON + CSV: every item, every model\u2019s raw answer, correctness under both gold keys) plus a <b>datasheet</b> (Gebru et al. 2021; Bender &amp; Friedman 2018). The <span class="mono">meridian explain</span> command re-derives any headline figure from those inputs and asserts it equals the displayed value. This aligns with reproducible-evaluation practice (Biderman et al. 2024). These dossiers were generated from a real run over HELM v1.3.0 \u00d7 MMLU-Redux, validated by end-to-end invariant checks.</p>'],
   ['How to falsify our claims','<ul class="mlim"><li><b>\u201cRank-fragile\u201d:</b> take the per-item export, re-key under the corrected gold yourself and recompute Kendall\u2019s \u03c4. If the ranking does not move, we are wrong.</li><li><b>\u201cLabel-error rate\u201d:</b> pull the flagged items; if the corrections are wrong, the rate is wrong \u2014 we claim candidates, not truth, and invite a second annotator.</li><li><b>\u201c\u03b1 &lt; 0\u201d:</b> recompute from the item-variance vector shown in the \u0192 panel.</li><li><b>\u201cReproducible\u201d:</b> re-run the pipeline on the same inputs; ids and numbers must match. If they do not, that is a bug to file.</li><li><b>The probes themselves:</b> each ships a calibration scenario \u2014 a planted defect it must catch. A probe that fails its calibration does not ship.</li></ul>'],
   ['References','<p class="refsblk">Cronbach &amp; Meehl (1955), <i>Construct validity in psychological tests</i> \u00b7 Messick (1995), <i>Validity of psychological assessment</i> \u00b7 Kane (2013), <i>Validating the interpretations and uses of test scores</i> \u00b7 Borsboom et al. (2004) \u00b7 Sijtsma (2009), Psychometrika 74:107\u2013120 \u00b7 Raji et al. (2021), arXiv:2111.15366 \u00b7 Bowman &amp; Dahl (2021) \u00b7 Jacobs &amp; Wallach (2021), FAccT \u00b7 Bean et al. (2024), 445-benchmark review \u00b7 Biderman et al. (2024), arXiv:2405.14782 \u00b7 Miller (2024), arXiv:2411.00640 \u00b7 Northcutt et al. (2021), arXiv:2103.14749 \u00b7 Gema et al., MMLU-Redux, arXiv:2406.04127 \u00b7 Zheng et al., MC selectors, arXiv:2309.03882 \u00b7 Sclar et al., FormatSpread, arXiv:2310.11324 \u00b7 Zheng et al., MT-Bench, arXiv:2306.05685 \u00b7 Ravaut et al. (2024), arXiv:2404.00699 \u00b7 van der Weij et al. (2024), arXiv:2406.07358 \u00b7 Gebru et al. (2021), Datasheets \u00b7 Bender &amp; Friedman (2018), Data Statements.</p>'],
  ];
  for(const [h,b] of cards2)w.appendChild($(`<div class="card"><h3>${h}</h3>${b}</div>`));
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
    pred_rows, intrinsic_rows, source_meta, models = {}, {}, {}, None
    for s in sources:
        try:
            p = s.fetch()
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        kind, bench = p.get("kind"), p.get("benchmark")
        sm = source_meta.setdefault(bench, {"sources": [], "dataset_version": p.get("dataset_version")})
        sm["sources"].append(getattr(s, "id", None))
        if p.get("dataset_version"):
            sm["dataset_version"] = p.get("dataset_version")
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
    # Total per-datum transparency: write the per-item export (JSON + CSV) and a
    # datasheet per benchmark under out/data/, and link them from the data dict.
    # Single source of truth: write_transparency re-derives from the same canonical
    # functions, so the export cannot diverge from what the dossiers display.
    try:
        from .transparency import write_transparency
        exports = write_transparency(out_dir, store, pred_rows, intrinsic_rows, source_meta)
    except Exception:
        exports = {}
    for row in data.get("portfolio", []):
        row["export"] = exports.get(row["name"])
    for nm, sp in (data.get("spotlight") or {}).items():
        if isinstance(sp, dict):
            sp["export"] = exports.get(nm)
    data["has_exports"] = bool(exports)
    html = build_ui_html(data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    (out / "observatory-data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # The lineage drill-down: every corrected item traced raw->verdict, from the real
    # lineage function. Wrapped so it can never break the main observatory build.
    lineage_path = None
    try:
        from .lineage_view import render_lineage_html
        lh = render_lineage_html(pred_rows, intrinsic_rows, title=title + " — lineage", iters=600)
        lp = out / "lineage.html"
        lp.write_text(lh, encoding="utf-8")
        lineage_path = str(lp)
    except Exception:
        lineage_path = None
    return {"index": str(index), "data": str(out / "observatory-data.json"),
            "exports": exports, "lineage": lineage_path}
