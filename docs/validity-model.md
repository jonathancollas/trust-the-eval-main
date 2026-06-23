# The validity model: from a trust score to result sensitivity

This note documents the validity model Trust-the-Eval / Meridian uses to judge an
evaluation **result**, and the empirical study that motivated it. It supersedes
the earlier per-benchmark "trust score".

## The question
A benchmark can carry label errors, ambiguous items, and weak internal
reliability. Does any of that actually change the **conclusions** people draw —
the model **scores** and **rankings**? We tested it on real data instead of
asserting it.

## The criterion study (real data)
HELM v1.3.0 per-item predictions for **10 models** × **57 MMLU subjects**
(5 692 matched items, 99.9% join) joined to MMLU-Redux annotations. We rank the
models on the same scorable set under the **original** gold and under the
**corrected** gold, and ask whether the per-subject label-error rate predicts
where the ranking moves. (See `criterion-validity-study.md` for full numbers.)

Findings:
- **The global MMLU ranking is robust to label correction** — Kendall τ = 1.000
  (subject-cluster 95% CI [0.911, 1.000]); every model's score rises ~uniformly
  (+0.7…+0.9 pts). Model separation absorbs the perturbation.
- **But the signal predicts local instability** — Spearman(label-error rate,
  per-subject ranking inversions) = 0.54, 14/57 subjects move. The conclusion is
  invariant to how the unresolvable residue (5 empty corrections, 33
  `no_correct`) is handled (τ = 1.000, ρ = 0.48–0.54 either way).
- **Mechanism** — on wrong-gold items models match the *corrected* answer 64% vs
  the *wrong* gold 21%; stronger models pick the true answer more often
  (Spearman skill≈0.49), so wrong golds mildly understate strong models.

Consequence: the impact of a label-quality problem is **conditional** on how
close the compared results are. A single per-benchmark badge cannot express
that, so we retired it.

## What we report instead
`result_sensitivity` (the headline). Given a set of models' per-item predictions
plus the original and corrected gold, it computes:
- **Δscore per model** with bootstrap CIs — always material;
- **Δranking** — Kendall τ, inversions, P(top-1 flips) — informative to the
  degree the models are close;
- **skill-discrimination** — whether the errors favour stronger or weaker models
  (the property that decides whether they threaten a ranking at all).

`test_reliability` (Classical Test Theory). Internal-consistency **Cronbach α**
plus item difficulty and point-biserial discrimination, computed from the
model×item matrix. For an aggregate benchmark we report the **per-subject α
distribution** (median), because the pooled α is inflated by cross-subject
difficulty spread; e.g. MMLU pooled α = 0.98 but per-subject median ≈ 0.46. With
~10 respondents (models) α is itself noisy — stated explicitly.

## What does NOT work (honest negatives)
- **Item discrimination is not a label-error screen.** Negative-discrimination
  items vs human-flagged errors: ROC-AUC ≈ 0.55 (chance). The clearest wrong-gold
  items are ones *every* model answers "wrong" under the bad gold → zero variance
  → discrimination undefined, so they are missed.
- **Panel-disagreement-with-gold is a decent screen (AUC ≈ 0.81) but circular** —
  MMLU-Redux candidate errors were seeded by model disagreement, so it measures
  consistency with how the flags were built, not independent validation.
- The least-circular corroboration of a correction is whether the panel picks the
  corrected **answer** (the mechanism number above), not mere disagreement.

## Ground-truth honesty
The label-error and ambiguity numbers come from a **single-pass human
annotation** (MMLU-Redux). No inter-annotator agreement (κ) is available for this
source — the multi-"expert" files are LLM-panel predictions, not multiple human
annotators — so we do **not** call these numbers "ground truth". Provenance is
recorded as `human_annotation_single_pass` with `inter_annotator_agreement: null`,
and the UI states "no inter-annotator agreement available". The label-error rate
therefore carries unquantified annotation uncertainty on top of sampling error.

## Continuous operation
A pipeline source of `kind: "predictions"` (per-item multi-model predictions +
original/corrected gold) is turned into a verifiable `battery_run` record
carrying the `result_sensitivity` and `test_reliability` audits, content-addressed
with an `inputs_hash`. The observatory's validity profile renders them per
benchmark. We rate the eval, never the models.


## End-to-end validation on real data
The full pipeline is exercised on real data (HELM v1.3.0 × MMLU-Redux, 57
subjects) by `scripts/e2e_real_data.py`, which runs **23,000+ counted invariant
checks**: every stored record verifies; each benchmark's label-error equals an
independent recount on the RAW annotations (validating error_type
canonicalisation); each per-subject result_sensitivity and Cronbach α reproduces
a fresh recomputation; the leaderboard, JSON API and `meridian.json` agree
field-by-field; re-running is idempotent; a corrupt source is quarantined; and
the generated SPA parses. A committed slice (`tests/test_e2e_real_data.py`) keeps
this honest in CI without shipping the full dataset.


## Remediation guidance (calibrated by reliability)
Detection without a remedy is half a tool, so each audit finding maps to a
concrete fix — but the guidance is tiered by how reliably the probe measures the
defect. Structural problems the probe detects exactly (duplicates, key/coverage
imbalance, scoring/judge/order/format/refusal/provenance defects) become
high-confidence `instrument_fix` actions. Ambiguous items are flagged for human
`review`. Label corrections are `relay_only`: we surface the specific flagged
items and the candidate correction from the single-pass annotation, labelled for
verification by a second annotator, and **never assert that the answer is X** —
because we measured statistical label-error detection to be near-chance. Weak or
defeatable signals (contamination, sandbagging) are `investigate` prompts, not
verdicts. Every plan is prioritised by result-sensitivity, so an author is told
whether fixing a defect would change the model ranking or only the absolute
scores. The invariant holds throughout: no remediation output asserts ground truth.
