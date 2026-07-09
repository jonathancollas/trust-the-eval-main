# Trust the Eval

**An open, extensible battery of probes that red-team the _validity_ of an AI evaluation result.**

Everyone runs evaluations. Almost no one can show that a given result is _trustworthy_ - that the score isn't memorised benchmark data, that the LLM judge isn't biased, that the model didn't behave differently because it knew it was being tested. Trust the Eval takes an eval result and attacks its validity from many angles, returning concrete **findings** you can act on.

## Graphical UI (local, zero-dependency)

A local web UI ships with the package — standard library only, no extra installs,
fully offline (no CDN). Launch it with:

```bash
trust-the-eval serve            # opens http://127.0.0.1:8077 in your browser
trust-the-eval serve --port 9000 --no-browser
```

Workflow: **1)** generate synthetic test data (or drag in a real `.json` /
Inspect `.eval` / promptfoo result), **2)** pick model access — *None* for the
static battery, *Demo* for an instant controllable synthetic model (try
`sandbagger`), or *API-compatible* for real endpoints via presets (OpenAI,
Anthropic-compatible gateways, Ollama, LM Studio, vLLM), **3)**
run all 20 probes or a subset. Findings stream in live, severity-sorted and
expandable to show the offending items; export the report as JSON, HTML, or a
`gen_ai.eval.trust` OpenTelemetry event. A long provider run can be stopped
mid-flight.

### Pull from Hugging Face

The UI's **Pull** tab fetches real data straight from the Hugging Face Dataset
Viewer REST API (stdlib `urllib`, no `datasets` dependency, public datasets need
no token). Enter a dataset id (or use a quick-pick), click **Inspect** to detect
its subsets, splits and columns, then **map columns → fields**
(question / gold / model response / score / category / options) — this column
mapping is what lets a huge diversity of datasets flow in. Gated datasets such as
the Open LLM Leaderboard `*-details` (which contain real model outputs + metrics)
work by pasting a Hugging Face access token (sent per-request, never stored).

A plain benchmark (questions + gold, no responses) still lights up the
model-in-the-loop probes once you pick a model in step 2 — they derive responses
themselves. Datasets that already carry model outputs additionally feed the
static-on-real-responses probes.

## What this is - and is not

- **It attacks the _evaluation_ (the measurement instrument), not the _model_.** Every probe answers "can I trust this eval result?" - never "how do I make a model misbehave?".
- It is **not** a jailbreak toolkit, an attack library against deployed systems, or a capability-elicitation kit. PRs in that direction are out of scope (see `CONTRIBUTING.md`).
- It is **decision-support for evaluators**, emitting findings - not an automated certifier handing down a verdict.

## How it works

- A **battery** of independent probes (not a pipeline) runs against one eval artifact. Order doesn't matter; each probe reads the result and emits findings.
- Findings are emitted as an **OpenTelemetry-compatible** event under the proposed `gen_ai.eval.trust.*` namespace, so trust evidence travels in the same pipelines as your eval traces. It **rides** the `gen_ai.*` conventions, it doesn't replace them.
- **Adapters** read existing eval logs (generic JSON today; an Inspect `.eval` adapter is a good first contribution).

## The probes

| id | what it attacks | paper priority | model? | status |
|---|---|---|---|---|
| `statistical_power` | precision of the score; are rankings real or noise | Principle 2 | no | **done** |
| `dataset_hygiene` | duplicates, malformed items, leaked canary | P5 | no | **done** |
| `contamination_perturb` | memorization (verbatim pass, perturbed fail) | P5 | yes | stub |
| `judge_swap` | judge position bias, collusion, kappa vs humans | Axis II | yes | stub |
| `sandbagging_paired` | suppression under eval-salient framing | P4 | yes | stub |
| `elicitation_ceiling` | under-reported capability; can't vs won't | P2 | yes | stub |

**Honest note:** the two `done` probes are static and cheap - and, as the experiment that started this project showed, **low signal** on a clean benchmark. The value is in the four model-in-the-loop stubs. They are stubs *on purpose*: this project is **depth-first** (a shallow probe that gives false confidence is worse than none), and these are the contribution targets.

## Result sensitivity & the validity profile

Beyond per-result probes, Meridian judges a benchmark by a **validity profile**:
measured label-error and ambiguity, internal-consistency reliability (Cronbach α,
per-subject), and — where multi-model per-item predictions exist — **result
sensitivity**: how much the **score** and the **ranking** actually move when the
known label errors are corrected, plus whether those errors favour stronger or
weaker models.

This replaces the earlier per-benchmark "trust score". A criterion-validity study
on real data (HELM v1.3.0 × MMLU-Redux, 10 models / 57 subjects) showed that
label errors always bias absolute scores and reshuffle near-tied models, but flip
the *global* ranking only when the errors are skill-discriminating and the models
are close — so the honest answer is a **sensitivity**, conditional on the
comparison, not a single badge. Details and the honest negative results (e.g.
item discrimination is *not* a usable label-error screen) are in
[`docs/validity-model.md`](docs/validity-model.md); see also `CHANGELOG.md`.

The annotation-derived numbers come from a single-pass human annotation; no
inter-annotator agreement is available for that source, so they are **not**
labelled "ground truth".

## Quickstart

```bash
pip install -e ".[dev]"
trust-the-eval check examples/sample_eval_result.json --otel-out trust.json
python examples/run_demo.py
pytest -q
```

## Interface audit and traceability plan

The recommended product direction for the local UI is documented in
[`docs/interface-audit-plan.md`](docs/interface-audit-plan.md): a six-stage
workflow rail (Source → Mapping → Artifact → Probe plan → Execution → Evidence),
a transformation ledger, row-level lineage, probe trace drawers, and replayable
audit bundles.

## Design principles

1. **Attack the measurement, not the model.** This is the line that keeps the project meta and defensive. It is enforced at review (`CONTRIBUTING.md`).
2. **Battery, not recipe.** Probes are parallel and independent. The value is the probe *content*, not a pipeline UI.
3. **Depth over breadth.** Ship one probe at 100% before the next. Don't pad the registry.
4. **Ride the ecosystem.** Extend OpenTelemetry's `gen_ai.*`; adapt from Inspect / promptfoo logs. Don't compete with the eval harnesses - sit above them.
5. **Community-extensible.** The maintainers ship the framework and the first deep probes; the field adds the long tail.

## Mapping to the evaluator-priorities paper

P4 sandbagging -> `sandbagging_paired` - P5 contamination & benchmark validity -> `contamination_perturb`, `dataset_hygiene` - Axis II judge reliability -> `judge_swap` - P2 capability vs propensity -> `elicitation_ceiling` - Principle 2 decision-grade statistics -> `statistical_power`. Out of scope by design: offensive capability content (P1), research programmes (P3, P7).

## Status

Pre-alpha scaffold. The framework runs; the high-value probes are intentionally unimplemented (depth-first + community). See `CONTRIBUTING.md` to add one.

## Testing

`python -m pytest -q` runs 188 tests with no network. Beyond the synthetic unit
and calibration tests, a committed **real-data** end-to-end test pins real
MMLU-Redux numbers, and a manual harness (`scripts/e2e_real_data.py`) drives the
full pipeline over 117 real sources with **24,000+ counted invariant checks**.
See [`TESTING.md`](TESTING.md).

## License

Apache-2.0 (license-aligned with the OpenTelemetry ecosystem this project extends).
