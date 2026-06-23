# Contributing to Trust the Eval

Thank you for helping make AI evaluations more trustworthy.

## The scope gate (read this first)

Trust the Eval assesses the **validity of an evaluation result**. That is the
entire scope.

**In scope** - probes that answer "can I trust this eval result?":
- contamination / memorization detection
- judge bias, collusion, calibration, human agreement
- sandbagging / evaluation-awareness *detection*
- elicitation-ceiling and capability-vs-propensity measurement
- label / ground-truth error detection, item ambiguity
- prompt-format and answer-extraction sensitivity
- model drift / temporal validity of a published result
- statistical power, multiplicities, provenance

**Out of scope** - PRs will be closed:
- anything that generates attacks, jailbreaks, or exploits against deployed
  models or systems
- harmful-capability elicitation (CBRN, cyber-offense, etc.) or content whose
  primary purpose is to harm a system rather than to assess an evaluation
- "how to game any eval" content (we *detect* whether an eval is gameable; we do
  not publish recipes to game one)

This is an evaluation-integrity project, not an offensive toolkit. When in
doubt, open an issue before writing code.

## How to add a probe

1. Create `src/trust_the_eval/probes/<your_probe>.py`.
2. Subclass `Probe`; set `id`, `name`, `paper_priority`, and `requires_model`.
3. Implement `run(artifact, model) -> list[Finding]`.
4. Decorate the class with `@register` and import it in `probes/__init__.py`.
5. Emit any telemetry under the `gen_ai.eval.trust.*` namespace and register new
   attribute names in `spec/otel-attributes.yaml`.
6. Add a test in `tests/`.

## Quality bar

**Depth over breadth.** A shallow probe that produces false confidence is worse
than no probe. A model-in-the-loop probe must define its sampling, its cost
behaviour, and how it reports evidence (the offending items), not just a number.

## Good first issues

- Implement the **Inspect `.eval` adapter** (`adapters/inspect_log.py`).
- Implement **`contamination_perturb`** (the perturbation re-test).
- Implement **`judge_swap`** (position-bias swap + kappa).
