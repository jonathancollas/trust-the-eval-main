# Canonical evaluation artifact schema v1.0

Trust the Eval normalizes supported inputs into an `EvalArtifact` while keeping
the source evidence needed to audit that normalization.

## Canonical fields

An artifact contains `dataset`, optional `model` and `judge`, `metadata`, and an
ordered list of items. Each item contains:

| Field | Type | Meaning |
|---|---|---|
| `question` | string | input presented to the evaluated model |
| `answer` | string | reference or gold answer |
| `response` | string or null | stored model output |
| `score` | number or null | score assigned by the source evaluation |
| `meta` | object | additional and source-specific evidence |

Every adapter stamps `metadata.schema_version = "1.0"` and
`metadata.source_format`. The generic JSON, promptfoo and Inspect adapters also
store the complete original item under `meta._source_record`. Consumers should
use canonical fields for probes and `_source_record` to verify the mapping or
recover source-specific rubrics, grader outputs and traces.

## Report manifest

JSON and HTML trust reports expose an artifact manifest containing schema and
source format, path, model/judge identifiers, item/response/score counts and the
content hash. The hash intentionally binds questions and gold answers; it is a
dataset-content identity, not a signature over every source metadata field.

## Compatibility policy

- Additive `meta` and `metadata` fields do not change schema v1 compatibility.
- A change to canonical field meaning or type requires a new major schema
  version and a migration function.
- Adapters must retain `_source_record`; normalization must not be the sole copy
  of source evidence.
- Unsupported or empty shapes fail explicitly rather than producing a
  successful zero-item audit.

Version 1.0 does **not** yet guarantee tested compatibility with every historic
Inspect or promptfoo release. Fixtures for each supported upstream release and
explicit migrations remain roadmap work.
