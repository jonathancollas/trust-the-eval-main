from __future__ import annotations
import json

from ..artifact import EvalArtifact, EvalItem


def load(path: str) -> EvalArtifact:
    """Load a promptfoo results JSON into an EvalArtifact.

    promptfoo shape (v0.x): {"results": {"results": [ {prompt, response, success,
    score, vars, ...} ]}} — we tolerate both the nested and flat layouts.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    res = data.get("results", data)
    rows = res.get("results", res) if isinstance(res, dict) else res
    items: list[EvalItem] = []
    for r in rows or []:
        prompt = r.get("prompt")
        if isinstance(prompt, dict):
            prompt = prompt.get("raw") or prompt.get("label") or json.dumps(prompt)
        resp = r.get("response")
        if isinstance(resp, dict):
            resp = resp.get("output") or resp.get("text") or json.dumps(resp)
        gold = (r.get("vars") or {}).get("expected") or r.get("expected") or ""
        score = r.get("score")
        if score is None and "success" in r:
            score = 1.0 if r["success"] else 0.0
        items.append(EvalItem(question=str(prompt or ""), answer=str(gold),
                              response=(str(resp) if resp is not None else None),
                              score=(float(score) if score is not None else None),
                              meta={k: r[k] for k in ("vars", "testIdx") if k in r}))
    return EvalArtifact(dataset=data.get("description") or "promptfoo_eval",
                        items=items, source_path=path, metadata={})
