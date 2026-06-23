# Changelog

## [Unreleased] — Validity observatory: result sensitivity

A criterion-validity study on real data (HELM v1.3.0 × MMLU-Redux, 10 models /
57 subjects) reshaped how we judge an evaluation result. See
`docs/validity-model.md` and `criterion-validity-study.md`.

### Changed
- `observe` now renders the **rich navigable observatory** (`observatory_ui`) as the
  live `index.html`, built from the evals just ingested; the lightweight SPA is kept
  alongside as `classic.html`. Eval sources of kind `predictions` are now accepted by
  the source loader, so a benchmark run (model predictions + dataset annotations) can
  feed the observatory directly via `meridian observe --sources <manifest>`; it
  recomputes incrementally as evals change. Covered by `tests/test_observatory_live.py`.

### Added
- `observatory_ui` — the navigable validity observatory generated from a synced
  store: a Portfolio of every benchmark profiled by result-sensitivity (rank-stable
  vs rank-fragile, never a trust score), a per-benchmark dossier with the
  verdict-under-correction ribbon, and a **Probe library** that explains *and*
  demonstrates the science behind each probe. Two invariants are tested: every figure
  carries the computation that produced it — each trace satisfies
  `(C − D) / npair == kendall_tau` exactly — and no model is rated or ranked as an
  endpoint. Pure `assemble_ui_data` + `build_ui_html`; real-data driver in
  `scripts/build_observatory_ui.py`. Covered by `tests/test_observatory_ui.py`.
- `result_sensitivity` — the headline measure: per-model Δscore (bootstrap CIs),
  Δranking (Kendall τ, inversions, P(top-1 flips)) and a skill-discrimination
  coefficient, for how much a result moves when known label errors are corrected.
- `item_analysis` — Classical Test Theory: item difficulty, point-biserial
  discrimination, Cronbach α, and a `concordance` test against external flags.
  Shipped as a **test-reliability** tool, explicitly **not** a validated
  label-error detector (discrimination vs human flags ≈ chance; documented).
- Pipeline source `kind: "predictions"` → a verifiable `battery_run` record
  carrying `result_sensitivity` + `test_reliability` audits (content-addressed,
  with `inputs_hash`).
- Validity profile gains a **Reliability (α)** column (per-subject median for
  aggregates; pooled flagged as inflated; ~10-respondent caveat surfaced).

### Changed
- **Retired the per-benchmark "trust score".** It conflated item-label quality
  with result trustworthiness; the study shows that mapping is weak (global
  rankings are robust to correction; only near-tied/local rankings move). The
  cross-benchmark page is now a **validity profile** + result-sensitivity, not a
  trust ranking.
- Annotation-derived numbers are no longer labelled "ground truth". Provenance is
  `human_annotation_single_pass` with `inter_annotator_agreement: null`; the UI
  states that no inter-annotator agreement is available (κ is not computable from
  this single-pass source).

### Added
- **Remediation guidance layer** (`remediation.py`): turns a benchmark's audit
  findings into a prioritised, *calibrated* list of fixes. Each of the 20 probes
  maps to an action tagged by confidence tier — `instrument_fix` (a structural
  defect the probe measures reliably: deduplicate, rebalance the key/coverage,
  fix scoring/judge/order/format/refusal/provenance), `review` (ambiguous items →
  human re-review), `relay_only` (label corrections surfaced from a *single-pass*
  annotation as candidates to verify with a second annotator — **never asserted**),
  `investigate` (weak/defeatable signals like contamination/sandbagging — a prompt,
  not a verdict), and `informational`. Plans are prioritised by `result_sensitivity`
  (does fixing change the model ranking, or only absolute scores?). Exposed per
  benchmark in the leaderboard and JSON API. Through-line: no output ever asserts
  ground truth.

### Fixed
- **Label-error was mis-measured on annotation files with mixed-case `error_type`.**
  Real files use `Wrong Groundtruth` / `Wrong groundtruth` / `Bad Question Clarity`;
  the code compared raw strings to canonical snake_case sets, silently scoring
  label-error as 0. Added `canonical_error_type` (lower + spaces→underscores),
  applied in the corpus and the fidelity gate. Already-canonical values are
  unchanged, so record ids are preserved.
- **`coverage_distribution` raised a false MEDIUM on single-category corpora**
  (`top_share` is trivially 1.0 with one category). It now only flags
  cross-category skew when ≥2 categories are present, and otherwise assesses
  MCQ answer-key balance — so a clean single-subject benchmark is no longer
  marked "drifting".

### Testing
- **Astronomical real-data end-to-end harness** (`scripts/e2e_real_data.py`):
  drives the full pipeline over 117 real sources (57 per-subject intrinsic + 57
  predictions + pooled MMLU + HELM claims + a corrupt source) and runs **23,000+
  counted invariant checks** — per-record hash verification, per-subject
  reproducibility of result_sensitivity / Cronbach α, per-item psychometric
  invariants, field-level api↔leaderboard equality, idempotency, quarantine, and
  SPA-JS validity. Both bugs above were caught by it.
- A committed real-data E2E test (`tests/test_e2e_real_data.py` + a 3-subject
  fixture) pins real numbers (e.g. virology label-error 41/100) and exercises the
  mixed-case `error_type` path, so the coverage persists without the full dataset.

### Notes
- Bright line unchanged: we rate evaluation instruments and claims, never models
  or their safety.
