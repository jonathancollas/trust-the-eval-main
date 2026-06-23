# Testing

## Unit + integration suite
```bash
python -m pytest -q          # 188 tests, ~12s, zero network
```
Covers the probes, calibration (planted-defect recall/specificity), the record
store and content-addressing, the autonomy pipeline (sync/observe, validation,
quarantine, idempotency), `result_sensitivity`, `item_analysis`, the leaderboard,
and a committed **real-data** end-to-end test.

## Real-data end-to-end test (committed)
`tests/test_e2e_real_data.py` drives the full pipeline over a small committed
fixture (`tests/fixtures/mmlu_redux_real_mini.json` — 3 MMLU-Redux subjects with
their genuine annotations and 10-model HELM predictions). It pins real numbers
(e.g. virology label-error 41/100, college_chemistry 23/100, a clean subject at
0) and deliberately exercises real-world messiness such as mixed-case
`error_type` (`Wrong Groundtruth`, `Wrong groundtruth`, `Bad Question Clarity`).

## Astronomical real-data harness (manual)
`scripts/e2e_real_data.py` runs the full pipeline over **117 real sources** (57
per-subject intrinsic + 57 predictions + a pooled MMLU corpus + HELM claims + a
corrupt source) and asserts **24,000+ counted invariants**:

- every stored record verifies (content hash) and round-trips through `dict`;
- `store.verify_all()` is clean;
- each benchmark's stored label-error equals an independent recount on the RAW
  (Title-case) annotations — validating `error_type` canonicalisation end to end;
- each per-subject `result_sensitivity` (τ, Δscore bounds, n_changed,
  skill-discrimination) and `test_reliability` (Cronbach α) reproduces a fresh
  recomputation;
- per-item psychometric invariants (difficulty ∈ [0,1], discrimination ∈ [−1,1],
  difficulty == mean correctness) hold across all ~5,700 items;
- the leaderboard, JSON API and `meridian.json` agree field-by-field (single
  source of truth), with no "trust score" anywhere;
- re-running `observe()` is idempotent; the corrupt source is quarantined; the
  generated SPA JS parses (`node --check`).

It requires a local clone of the MMLU-Redux dataset repo (it reads
`/home/.../mmlu-redux/mmlu_redux/*.csv` and `outputs/original_helm_combined/*.csv`);
adjust the paths at the top of the script. Two real bugs — the `error_type`
casing mismatch and a false-positive `coverage_distribution` MEDIUM on
single-category corpora — were found and fixed via this harness.
