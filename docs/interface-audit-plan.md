# Interface audit and transformation-traceability plan

This document audits the current Trust the Eval interfaces and proposes a step-by-step plan to make the workflow crystal clear while exposing every data transformation that feeds the processes and probes.

## Executive summary

The product already has the right primitives for trust: probe documentation, per-probe progress events, data export, observatory provenance, and a lineage page for corrected benchmark items. The gap is that those primitives are split across different surfaces and are not organized around the user's mental model: **what data entered, what changed, what ran, what each probe saw, what each probe emitted, and what can be re-run or verified**.

The recommended direction is to make the UI a **traceable workbench** with a persistent workflow rail and an always-available transformation ledger:

1. **Source** — show raw source, adapter choice, raw field schema, row counts, parse warnings, and a sample of raw rows exactly as received.
2. **Mapping** — show field mapping, before/after examples for every mapped column, dropped rows, and a diff from source row to canonical `EvalItem`.
3. **Artifact** — show schema coercions, dropped/filled fields, score/answer canonicalization, hashes, and quality gates.
4. **Probe plan** — show model access, probe eligibility, probe tunables, expected cost, and why any probe will be skipped.
5. **Execution** — show probe lifecycle events, per-probe inputs, model calls, intermediate calculations, cache hits, and findings.
6. **Evidence** — tie each finding to evidence rows, formulas, severity logic, caveats, and remediation; export a portable trace bundle with canonical artifact, transform ledger, run config, probe traces, report, and provenance hashes.

## Current-state audit

### What is strong today

- The local web UI is dependency-free and explicitly positioned as a three-step flow: load data, pick model access, and run probes. This is a good starting skeleton for a clearer wizard/workbench experience.
- The backend reuses the same `run_battery` path as the CLI and exposes progress hooks for probe lifecycle events, so the UI can be made truthful without inventing a separate run model.
- The runner already models probe execution as explicit lifecycle events: `running`, `done`, `skipped`, and `error`. That is the right event vocabulary for a live run timeline.
- The artifact summary endpoint already computes important inspection facts: dataset name, item count, provenance hash, whether responses/scores are present, category counts, and MCQ count.
- The artifact explorer already exposes paginated item slices with question, answer, response, score, category, and options, which can become the first layer of row-level auditability.
- Probe docs are designed to keep scientific description, math, examples, and references near implementation and show them in the web UI.
- The observatory already treats provenance as a first-class concept: records are verified, evidence tiers are surfaced, and public pages differentiate measured, model-free, and literature-derived evidence.
- The existing lineage view proves the desired pattern at small scope: it renders each corrected item as a journey from imported row to correction policy, correctness, accuracy, ranking effect, and verdict, with input, rule, formula, output, and source at every step.

### What blocks a crystal-clear workflow

- The local UI describes steps but does not make them a durable, inspectable state machine. Users can do the work, but they cannot always answer "where am I, what happened before this, and what is next?"
- Transformations are implicit. Adapter detection, Hugging Face column mapping, generic JSON loading, synthetic generation, promptfoo detection, model response derivation, score interpretation, and probe-specific reductions are not written to a unified ledger.
- The current run state stores final findings, skipped probes, errors, progress, and cost, but not per-probe inputs, intermediate values, model requests/responses, cache decisions, or why severity thresholds fired.
- There is no single row-level trace for the local UI equivalent to the observatory lineage page. Users can inspect items and findings, but cannot click one datum and follow it through every adapter, normalization, probe, and finding transformation.
- Probe independence is scientifically important, but the interface can make that feel like a black box unless it distinguishes **workflow order** from **probe independence**. The UI should say: "The workflow is sequential; the probes are independent checks over the same frozen artifact."
- Exported reports are useful but are not yet a complete replay/audit bundle. A rigorous auditor needs raw source metadata, canonical artifact hash, transform events, probe config, code/probe version, model cache/cost metadata, findings, and environment details in one package.

## Target experience

### Top-level information architecture

Replace the current "load/configure/run" feel with a persistent six-stage rail:

| Stage | User question answered | Primary UI object |
|---|---|---|
| 1. Source | What did I provide? | Source card with raw file/API metadata and sample rows |
| 2. Mapping | How did raw fields become eval fields? | Mapping table with before/after examples |
| 3. Artifact | What canonical dataset will probes see? | Frozen artifact summary, schema checks, hash, item explorer |
| 4. Probe plan | Which probes will run and why? | Probe eligibility matrix and tunable diff |
| 5. Execution | What is happening now? | Live run timeline with per-probe trace drawers |
| 6. Evidence | What changed my trust decision? | Findings board with row lineage, formula, thresholds, remediation, exports |

### Core rule

Every panel should follow the same pattern:

```text
Input → Rule/transform → Output → Evidence/source → Reproduce
```

If the UI cannot fill one of those fields, it should explicitly say "not captured yet" instead of hiding the gap.

## Transformation ledger design

Introduce a canonical trace event model that every loader, adapter, normalizer, model call, and probe can append to.

### Trace event schema

```json
{
  "trace_id": "uuid",
  "run_id": "uuid",
  "stage": "ingest|mapping|normalize|probe|finding|export",
  "step_id": "normalize.score.coerce_float",
  "sequence": 42,
  "subject": {"kind": "artifact|item|probe|finding|model_call", "id": "..."},
  "input_ref": {"hash": "sha256:...", "preview": {}},
  "rule": {"name": "canonicalize_score", "version": "...", "parameters": {}},
  "output_ref": {"hash": "sha256:...", "preview": {}},
  "metrics": {"rows_in": 1000, "rows_out": 998, "warnings": 2},
  "source": {"module": "trust_the_eval.adapters.generic_json", "function": "load"},
  "warnings": [],
  "created_at": "2026-07-09T00:00:00Z"
}
```

### Trace capture levels

Provide three modes so the product remains practical:

1. **Summary** — default; captures stage-level counts, hashes, warnings, probe lifecycle, and finding evidence references.
2. **Audit** — captures item-level before/after previews and probe intermediate metrics for all findings and sampled non-findings.
3. **Full replay** — captures all item-level transforms, model request/response metadata, cache keys, and deterministic seeds; redacts secrets by default.

## Detailed plan

### Phase 1 — Make the workflow unmistakable

**Goal:** Users always know what step they are in, what is complete, what is blocked, and what changed.

- Add a persistent workflow rail with six stages: Source, Mapping, Artifact, Probe plan, Execution, Evidence.
- Add stage completion checks, warnings, and "next action" copy.
- Freeze the canonical artifact after Mapping/Artifact so later probe results refer to an immutable hash.
- Rename ambiguous actions to explicit verbs:
  - "Inspect" → "Inspect source schema"
  - "Run" → "Run selected probes on frozen artifact"
  - "Export" → "Export report" and "Export replay bundle"
- Add an "Audit mode" toggle before run start.

**Acceptance criteria:** a new user can describe the workflow from the rail alone; each completed stage shows a data hash, count, or explicit reason it has no data.

### Phase 2 — Show raw-to-canonical transformation

**Goal:** Every input row can be traced into the canonical artifact.

- Instrument all loaders/adapters to emit transformation events.
- Add a Source Preview table showing raw rows exactly as received.
- Add a Mapping Preview table with columns: raw field, mapped canonical field, transform rule, null/default behavior, example input, example output.
- Add row-level item IDs stable across adapter and artifact stages.
- Surface dropped rows, duplicate handling, invalid scores, missing answers, and unsupported option formats as filterable audit events.
- Add a canonical artifact diff drawer: raw row → canonical item → validation warnings.

**Acceptance criteria:** for any canonical item in the explorer, the user can see the raw input fields and the rules that produced question, answer, response, score, category, and options.

### Phase 3 — Make probe planning transparent

**Goal:** Users understand why probes run, skip, or require model access before pressing Run.

- Create a Probe Plan matrix with columns: probe, purpose, requires model, eligible now, reason, tunables, expected inputs, output finding types.
- For skipped probes, show precise reasons such as "requires model", "requires responses", "requires scores", or "insufficient sample size".
- Show default tunables and any overrides as a diff.
- Link each probe to its documentation drawer from the matrix.
- Add "what this probe will read" previews: artifact fields, sample items, and any model prompt shape.

**Acceptance criteria:** the set of probes that will run is predictable before execution, and no skipped probe appears surprising after execution.

### Phase 4 — Add live probe trace drawers

**Goal:** Users can observe what every probe does without turning the UI into a debugger.

- Extend progress events into structured per-probe trace events: started, input snapshot, model calls, calculations, thresholds, findings, skipped, errored.
- Add an Execution timeline grouped by probe with status, duration, cost, cache hits, number of items touched, and findings emitted.
- Add a probe drawer with tabs:
  - **Inputs:** artifact hash, item subset, fields read.
  - **Transforms:** intermediate reductions and formulas.
  - **Model calls:** prompts/responses metadata, redacted payload previews, cache/cost.
  - **Thresholds:** fired and non-fired thresholds.
  - **Findings:** emitted findings and evidence rows.
  - **Code/doc:** implementation source and probe scientific doc.
- Keep model payload capture configurable and redacted.

**Acceptance criteria:** every completed probe has a compact trace explaining how it moved from artifact input to findings or no findings.

### Phase 5 — Build row-level lineage everywhere

**Goal:** A user can click any datum and follow it through every relevant process and probe.

- Add an item lineage drawer in the local UI modeled after the existing observatory lineage page.
- For each item, show stages: source row, mapped fields, normalized artifact item, probe reads, model calls involving the item, score/finding contributions, exports.
- Add item-level badges: used by probe, dropped, transformed, warning, evidence in finding.
- Add cross-links from findings to all evidence items and from items back to findings.
- Add a "why this number" drawer for aggregate metrics showing formula, input rows, included/excluded counts, output, and code source.

**Acceptance criteria:** every finding evidence row is traceable back to source input and forward to the exported report.

### Phase 6 — Export and replay the audit

**Goal:** The interface produces artifacts suitable for review, bug reports, scientific replication, and governance.

- Add a `trace.jsonl` export containing all captured transformation events.
- Add a `manifest.json` with versions, artifact hash, run config, probe set, model provider metadata, redaction policy, and environment.
- Add a zipped replay bundle: raw source metadata or redacted source, canonical artifact, trace, report, HTML report, and optional model cache metadata.
- Add import support for replay bundles so users can reopen a past run and inspect the same trace without rerunning probes.
- Add verification command: `trust-the-eval verify-bundle path/to/bundle.zip`.

**Acceptance criteria:** an auditor can open a bundle and verify that report findings correspond to trace events and the canonical artifact hash.

## Visual recommendations

- Use a **left workflow rail** for stage state and a **right inspector drawer** for details. Avoid hiding crucial provenance behind only hover tooltips.
- Use consistent color semantics:
  - Gray: not started / unavailable.
  - Blue: selected / running.
  - Green: completed / verified.
  - Amber: warning / needs review.
  - Red: error / invalid / high-severity finding.
- Use short, concrete labels: "2 fields unmapped", "17 rows dropped", "5 probes skipped", "3 findings", "artifact sha256:abcd…".
- Prefer **diff tables** over paragraphs for transformations.
- Provide a global search/filter over trace events, item IDs, probe IDs, finding IDs, and hashes.

## Data model changes

Recommended new or extended internal objects:

- `TraceEvent` — append-only event described above.
- `TraceStore` — in-memory for local UI; JSONL export for persistence.
- `ItemLineage` — derived view that groups trace events by stable item ID.
- `ProbeTrace` — derived view grouped by probe ID and run ID.
- `AuditBundle` — manifest + canonical artifact + trace + report + optional redacted raw/source metadata.

## API changes

Recommended local UI endpoints:

- `GET /api/workflow` — current stage state, blockers, artifact hash, run IDs.
- `GET /api/trace?run_id=...&stage=...&probe_id=...&item_id=...` — filtered trace events.
- `GET /api/items/{item_id}/lineage` — row-level lineage.
- `GET /api/probes/{probe_id}/trace?run_id=...` — probe execution trace.
- `POST /api/export/bundle` — generate replay bundle.
- `POST /api/import/bundle` — load a previous bundle into the UI.

## Risks and mitigations

- **Trace volume can grow quickly.** Use capture levels, previews, hashes, pagination, and on-demand full payload expansion.
- **Model calls can contain sensitive data.** Redact secrets, default to metadata/previews, and make full payload capture opt-in.
- **Probe authors may forget instrumentation.** Provide helper APIs and make uninstrumented probes display "not captured yet" in the trace drawer.
- **Too much detail can overwhelm users.** Keep the main rail simple and put details in progressive disclosure drawers.
- **Probe independence could be misread as a pipeline dependency.** The UI copy should repeatedly distinguish sequential workflow stages from independent probe execution.

## Implementation roadmap

### Milestone A — Workflow rail and artifact freeze

- Add workflow state model to the server.
- Render the six-stage rail in the local UI.
- Show artifact hash and source/mapping summary after load.
- Add explicit blocked/ready statuses for each stage.

### Milestone B — Loader trace events

- Add `TraceEvent` and `TraceStore`.
- Instrument synthetic, upload, promptfoo, Inspect, generic JSON, and Hugging Face paths.
- Add raw-to-canonical preview and warning panels.

### Milestone C — Probe plan matrix

- Add eligibility metadata to probes or a central planner.
- Render probe plan before execution.
- Show skipped reasons before and after run.

### Milestone D — Probe traces

- Extend runner progress callbacks with structured event IDs and durations.
- Provide probe helper methods for item subsets, formulas, thresholds, and model calls.
- Render live execution timeline and probe trace drawers.

### Milestone E — Item lineage and finding evidence links

- Add stable item IDs.
- Build item lineage endpoint and drawer.
- Link findings to source rows and probe calculations.

### Milestone F — Replay bundle

- Export manifest, trace JSONL, canonical artifact, report, and HTML.
- Add bundle import and verification command.
- Document the audit/replay contract.

## Success metrics

- 100% of run reports include artifact hash, probe set, run config, and stage summaries.
- 100% of findings link to evidence rows and probe trace events.
- 95%+ of adapter transformations have structured trace events.
- Every skipped probe has a machine-readable skip reason visible before run.
- A replay bundle can regenerate or verify the same report summary from the canonical artifact and trace.

## First changes to prioritize

1. Add a workflow rail and state machine; this immediately clarifies the user journey.
2. Add artifact freeze/hash display; this makes every later result refer to a stable input.
3. Add probe plan/skipped-reason matrix; this reduces confusion before runs.
4. Add a minimal trace event model around loaders and runner progress; this unlocks later lineage work.
5. Add finding-to-item links in the UI; this makes evidence tangible even before full replay bundles exist.
