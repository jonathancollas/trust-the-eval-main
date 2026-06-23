"""Pull eval results / benchmark datasets from the Hugging Face Dataset Viewer.

Uses the public REST API (https://datasets-server.huggingface.co) over urllib —
no `datasets` library, no extra dependencies. Because dataset schemas vary
wildly, the caller supplies a column->field MAPPING; this is what lets a single
code path ingest a huge diversity of datasets.

A mapping is a dict with any of these keys, each naming a source column:
    question  (required)  -> EvalItem.question
    answer                -> EvalItem.answer   (ground truth)
    response              -> EvalItem.response  (a model output, if the dataset
                                                 is an eval RESULT and not just a
                                                 benchmark)
    score                 -> EvalItem.score    (coerced to float in 0..1)
    category              -> EvalItem.meta["category"]
    options               -> EvalItem.meta["options"] (for MCQ probes)

Note: many HF datasets are *benchmarks* (question+answer only). Those still feed
the model-in-the-loop probes when you select a model (the probes derive the
responses themselves). Datasets that already contain model outputs + metrics
(e.g. the Open LLM Leaderboard `*-details` datasets, which are gated -> need a
token) additionally feed the static-on-real-responses probes.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ..artifact import EvalArtifact, EvalItem

BASE = "https://datasets-server.huggingface.co"
MAX_PAGE = 100            # the /rows endpoint returns at most 100 rows per call
HARD_CAP = 2000           # safety cap on total rows pulled in one request

FIELD_KEYS = ("question", "answer", "response", "score", "category", "options")

# Heuristics: column-name candidates for each field (lowercased, exact match first)
_GUESS = {
    "question": ["question", "prompt", "query", "input", "instruction", "problem",
                 "full_prompt", "doc", "text", "context", "sentence"],
    "answer":   ["answer", "target", "gold", "label", "reference", "solution",
                 "correct_answer", "gold_answer", "ground_truth", "answers"],
    "response": ["response", "output", "prediction", "prediction_text", "generation",
                 "model_output", "filtered_resps", "resps", "completion", "pred"],
    "score":    ["score", "acc", "acc_norm", "accuracy", "exact_match", "is_correct",
                 "correct", "em", "f1", "success", "pass", "metric", "reward"],
    "category": ["category", "subject", "topic", "task", "type", "domain", "subset",
                 "group"],
    "options":  ["options", "choices", "candidates", "endings"],
}

_RELEVANT_TAGS = {
    "evaluation", "eval", "benchmark", "leaderboard", "lm-eval",
    "question-answering", "multiple-choice", "text-generation",
    "text2text-generation", "reasoning", "mmlu", "gsm8k", "truthfulqa",
    "hellaswag", "arc", "bbh", "math",
}
_RELEVANT_ID_TOKENS = {
    "eval", "benchmark", "leaderboard", "mmlu", "gsm8k", "truthful", "hellaswag",
    "arc", "bbh", "math", "lighteval", "mt_bench",
}
_NON_EVAL_MODALITY_TAGS = {
    "audio", "speech", "image", "vision", "video", "object-detection",
    "image-classification", "image-segmentation",
}
_TEXT_EVAL_TAGS = {"question-answering", "text-generation", "text2text-generation"}
# We over-fetch search rows because post-filtering can discard many generic datasets.
# 4x is a practical trade-off: good recall for relevant rows while keeping requests small.
_SEARCH_FILTER_MULTIPLIER = 4


# --------------------------- HTTP ---------------------------
def _get(path: str, params: Dict[str, str], token: Optional[str] = None,
         timeout: int = 30) -> dict:
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "trust-the-eval/0.1"}
    if token:
        headers["Authorization"] = "Bearer " + token.strip()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
            msg = json.loads(body).get("error", body)
        except Exception:
            msg = body or str(e)
        if e.code == 401 or e.code == 403:
            raise PermissionError(
                f"Hugging Face returned {e.code}: {msg} "
                "(gated/private dataset — provide an access token)")
        if e.code == 429:
            raise RuntimeError("Hugging Face rate limit (HTTP 429) — wait and retry.")
        raise RuntimeError(f"Hugging Face error {e.code}: {msg}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error reaching Hugging Face: {e.reason}")


# --------------------------- discovery ---------------------------
def list_splits(dataset: str, token: Optional[str] = None) -> List[Dict[str, str]]:
    """Return [{'config':..., 'split':...}, ...] for a dataset."""
    data = _get("/splits", {"dataset": dataset}, token)
    out = []
    for s in data.get("splits", []):
        out.append({"config": s.get("config"), "split": s.get("split")})
    return out


def _columns_from_features(features: List[dict]) -> List[str]:
    return [f.get("name") for f in features if f.get("name")]


def inspect_dataset(dataset: str, config: Optional[str] = None,
                    split: Optional[str] = None, token: Optional[str] = None,
                    n_preview: int = 3) -> dict:
    """Discover configs/splits/columns and return a small preview + a guessed
    column->field mapping. One /splits call + one /first-rows call."""
    splits = list_splits(dataset, token)
    if not splits:
        raise RuntimeError(f"no splits found for dataset '{dataset}'")
    if config is None or split is None:
        # prefer a split named like a test/validation set, else the first
        pref = next((s for s in splits if s["split"] in ("test", "validation", "valid")), splits[0])
        config = config or pref["config"]
        split = split or pref["split"]
    fr = _get("/first-rows", {"dataset": dataset, "config": config, "split": split}, token)
    columns = _columns_from_features(fr.get("features", []))
    rows = [r.get("row", {}) for r in fr.get("rows", [])[:n_preview]]
    return {
        "dataset": dataset, "config": config, "split": split,
        "configs": sorted({s["config"] for s in splits if s["config"]}),
        "splits": splits,
        "columns": columns,
        "guess": guess_mapping(columns),
        "format": classify_format(columns),
        "sample": rows,
    }


def guess_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    used: set = set()
    for field in FIELD_KEYS:
        chosen = None
        for cand in _GUESS[field]:
            if cand in lower and lower[cand] not in used:
                chosen = lower[cand]
                break
        if chosen is None:  # substring fallback
            for c in columns:
                if c in used:
                    continue
                if any(cand in c.lower() for cand in _GUESS[field]):
                    chosen = c
                    break
        if chosen:
            used.add(chosen)
        mapping[field] = chosen
    return mapping


def classify_format(columns: List[str]) -> Dict[str, Any]:
    """Tell the user whether a dataset is an auditable eval RESULT or just questions.

    Trust the Eval audits *results* (it needs per-sample scores/responses). A
    question-only dataset can only feed coverage/hygiene probes, so we flag it.
    """
    m = guess_mapping(columns)
    has_q, has_resp, has_score = bool(m.get("question")), bool(m.get("response")), bool(m.get("score"))
    if has_score and (has_resp or has_q):
        return {"kind": "results", "auditable": True,
                "mapping": m, "reason": "per-sample scores detected — most probes can run"}
    if has_resp and has_q:
        return {"kind": "responses", "auditable": True,
                "mapping": m, "reason": "model responses but no score column — map a score for full coverage"}
    if has_q:
        return {"kind": "questions_only", "auditable": False,
                "mapping": m, "reason": "questions only (no scores/responses) — only coverage & hygiene probes apply"}
    return {"kind": "unknown", "auditable": False,
            "mapping": m, "reason": "could not detect eval columns — map them by hand below"}


_RESULT_ID_TOKENS = ("details", "-results", "_results", "results-", "leaderboard",
                     "eval-results", "evals", "predictions", "preds", "lm-eval",
                     "lm_eval", "harness", "samples", "logprobs")


def _is_result_dataset(d: Dict[str, Any]) -> bool:
    """Heuristic: does this dataset id look like eval RESULTS (not a raw QA set)?"""
    ds_id = (d.get("id") or "").lower()
    return any(tok in ds_id for tok in _RESULT_ID_TOKENS)


# --------------------------- fetch + map ---------------------------
def _parse_rows_response(payload: dict) -> List[dict]:
    """Turn a /rows or /first-rows JSON envelope into a list of row dicts."""
    return [r.get("row", {}) for r in payload.get("rows", [])]


def fetch_rows(dataset: str, config: str, split: str, n: int,
               token: Optional[str] = None) -> List[dict]:
    """Fetch up to n rows, paginating the /rows endpoint (<=100 per call)."""
    n = max(1, min(int(n), HARD_CAP))
    rows: List[dict] = []
    offset = 0
    while len(rows) < n:
        length = min(MAX_PAGE, n - len(rows))
        payload = _get("/rows", {"dataset": dataset, "config": config,
                                 "split": split, "offset": str(offset),
                                 "length": str(length)}, token)
        batch = _parse_rows_response(payload)
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break
    return rows[:n]


def _coerce_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        v = v[0] if v else ""
    if isinstance(v, dict):
        # common nested shapes: {"text": ...} / {"output": ...}
        for k in ("text", "output", "content", "value"):
            if k in v:
                v = v[k]
                break
        else:
            return json.dumps(v, ensure_ascii=False)
    return str(v)


_TRUE = {"true", "1", "correct", "yes", "pass", "passed"}
_FALSE = {"false", "0", "incorrect", "no", "fail", "failed"}


def _coerce_score(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return _coerce_score(v[0]) if v else None
    if isinstance(v, dict):
        for k in ("acc", "exact_match", "score", "value", "em"):
            if k in v:
                return _coerce_score(v[k])
        return None
    s = str(v).strip().lower()
    if s in _TRUE:
        return 1.0
    if s in _FALSE:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def rows_to_artifact(rows: List[dict], mapping: Dict[str, Optional[str]],
                     dataset_name: str, model: Optional[str] = None,
                     judge: Optional[str] = None) -> EvalArtifact:
    q_col = mapping.get("question")
    if not q_col:
        raise ValueError("mapping must specify a 'question' column")
    items: List[EvalItem] = []
    for row in rows:
        question = _coerce_text(row.get(q_col))
        if not question.strip():
            continue
        meta: Dict[str, Any] = {}
        cat_col = mapping.get("category")
        if cat_col and row.get(cat_col) is not None:
            meta["category"] = _coerce_text(row.get(cat_col))
        opt_col = mapping.get("options")
        if opt_col and isinstance(row.get(opt_col), list):
            meta["options"] = [str(o) for o in row.get(opt_col)]
        ans_col = mapping.get("answer")
        resp_col = mapping.get("response")
        score_col = mapping.get("score")
        items.append(EvalItem(
            question=question,
            answer=_coerce_text(row.get(ans_col)) if ans_col else "",
            response=(_coerce_text(row.get(resp_col)) if resp_col and row.get(resp_col) is not None else None),
            score=(_coerce_score(row.get(score_col)) if score_col else None),
            meta=meta,
        ))
    return EvalArtifact(dataset=dataset_name, items=items, model=model, judge=judge,
                        source_path=f"hf://{dataset_name}", metadata={"source": "huggingface"})


def pull(dataset: str, config: str, split: str, n: int,
         mapping: Dict[str, Optional[str]], token: Optional[str] = None,
         model: Optional[str] = None) -> EvalArtifact:
    rows = fetch_rows(dataset, config, split, n, token)
    name = f"{dataset}/{config}/{split}"
    return rows_to_artifact(rows, mapping, name, model=model)


HUB = "https://huggingface.co"


def _as_text_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, dict):
        vals = []
        for x in v.values():
            vals.extend(_as_text_list(x))
        return vals
    if isinstance(v, list):
        vals = []
        for x in v:
            vals.extend(_as_text_list(x))
        return vals
    return [str(v)]


def _is_relevant_dataset(d: Dict[str, Any]) -> bool:
    ds_id = (d.get("id") or "").lower()
    tags = {t.lower() for t in _as_text_list(d.get("tags"))}
    task_categories = {t.lower() for t in _as_text_list(d.get("task_categories"))}
    all_tags = tags | task_categories

    if any(tok in ds_id for tok in _RELEVANT_ID_TOKENS):
        return True
    if all_tags & _RELEVANT_TAGS:
        return True
    if all_tags & _NON_EVAL_MODALITY_TAGS:
        return False
    # fallback: keep text datasets with likely QA/generation tasks
    textish = {"text", "natural-language-processing", "english"}
    if all_tags & textish and bool(all_tags & _TEXT_EVAL_TAGS):
        return True
    return False


def search_datasets(query: str = "", limit: int = 25, sort: str = "downloads",
                    token: Optional[str] = None, relevant_only: bool = True,
                    results_only: bool = False) -> List[dict]:
    """Search the Hugging Face Hub for datasets (the explorer backend).

    Uses the public Hub API (huggingface.co/api/datasets) over urllib. Returns a
    compact list of {id, downloads, likes, gated, updated} for display.
    """
    limit = max(1, min(int(limit), 100))
    # Ask for a wider page so post-filtering still has enough relevant rows.
    # The Hub API here is single-page in this flow, so we widen one request and cap at 100.
    request_limit = min(100, limit * _SEARCH_FILTER_MULTIPLIER) if relevant_only else limit
    params = {"limit": str(request_limit),
              "sort": sort, "direction": "-1", "full": "false"}
    if query:
        params["search"] = query
    url = HUB + "/api/datasets?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "trust-the-eval/0.1"}
    if token:
        headers["Authorization"] = "Bearer " + token.strip()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            rows = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Hugging Face Hub search error {e.code}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error reaching Hugging Face Hub: {e.reason}")
    out = []
    for d in rows or []:
        if relevant_only and not _is_relevant_dataset(d):
            continue
        if results_only and not _is_result_dataset(d):
            continue
        out.append({
            "id": d.get("id"),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "gated": bool(d.get("gated", False)),
            "updated": d.get("lastModified") or d.get("createdAt"),
        })
        if len(out) >= limit:
            break
    return out
