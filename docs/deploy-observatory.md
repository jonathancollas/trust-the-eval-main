# Deploying the Meridian observatory (no hosting required)

The observatory is a **single self-contained `index.html`** — system fonts, inline
SVG, all data embedded, zero network calls. It runs on any static host and even by
double-clicking the file (`file://`). You do **not** need to run a server.

This guide uses **GitHub Pages**, which is free and already part of your workflow.

---

## What the build produces

`meridian observe --sources evals.json --out site/` writes into `site/`:

| file | what it is |
|------|------------|
| `index.html` | the rich navigable observatory (Portfolio, dossiers, verdict-under-correction ribbon, per-number computation panels, probe library) |
| `classic.html` | the lightweight SPA, kept alongside |
| `meridian.json` | the stable API document (schema `meridian-api/v1`) |
| `records.json` | the content-addressed validity records |
| `observatory-data.json` | the data the rich UI is built from |

`observe` is **incremental**: only sources whose fingerprint changed are re-audited.

---

## One-time setup

1. **Enable Pages from Actions.** Repo → **Settings ▸ Pages ▸ Build and deployment ▸
   Source = GitHub Actions**. (No `gh-pages` branch needed.)
2. **Create your manifest.** Copy the example and edit it:
   ```bash
   cp evals.example.json evals.json
   ```
3. **Add your evals** (see the manifest section below), commit, and push.

The workflow at `.github/workflows/observatory.yml` then builds and publishes the
site to `https://<user>.github.io/<repo>/` on:

- every push that touches `evals.json`, `evals/**`, or the engine (`src/**`);
- a **daily schedule** (cron);
- **manual** runs (Actions ▸ observatory ▸ *Run workflow*).

The store is cached between runs, so each publish recomputes only what changed.

---

## The evals manifest

`evals.json` is a JSON list of sources. Three access types — and you can mix **local
files** and **remote URLs** freely:

- `local` — a JSON/JSONL file in the repo (under `evals/`). Offline, fully
  reproducible. Needs `path`.
- `hf_dataset` — a Hugging Face dataset config/split, pinned to its commit SHA.
  Intended for annotation corpora (the `intrinsic` axis). Needs `repo` (and optional
  `config`, `split`).
- `leaderboard_url` — a JSON/JSONL results file at a URL (e.g. a HELM export). Needs
  `url`.

Every source also declares `id`, `kind`, `benchmark`, `dataset_version`, and optional
`anchors` (admission guards, e.g. `{"min_rows": 1}`).

Three eval **kinds**:

- **`intrinsic`** — the dataset's annotations. Rows:
  `{question, choices, answer, error_type, subject}`. Drives label-error, ambiguity,
  coverage and hygiene (model-free).
- **`predictions`** — multi-model answers + both gold keys. Rows:
  `{item, subject, original_gold: [...], corrected_gold: [...], preds: {model: "A"}}`.
  Drives the headline **result-sensitivity** (does correcting the labels change the
  ranking?) and Cronbach reliability. For movement to show, `corrected_gold` must
  differ from `original_gold` somewhere — the corrections come from the `intrinsic`
  eval, the per-item golds from the `predictions` eval.
- **`claim`** — reported scores. Checked against the valid-accuracy ceiling implied by
  the label errors.

See `evals.example.json` for one of each. Edit the `repo`/`url`/`path` values to your
own data before pushing.

---

## Preview locally (still no hosting)

```bash
pip install -e .
meridian observe --sources evals.json --store .meridian-store --out site --always
meridian serve --out site            # http://127.0.0.1:8088
# or just open site/index.html directly
```

---

## Privacy note (important for a public repo)

A **public** repo + Pages makes the published site *and* your eval data
(`predictions`, `claims`, `meridian.json`) publicly readable. By design this is sound
for the observatory itself — it audits the **instrument**, never rates a model — but
your eval inputs would be exposed. If those inputs are sensitive, use a **private**
repo (Pages on private repos requires GitHub Pro/Team), or publish only the built
artifact rather than a public URL.
