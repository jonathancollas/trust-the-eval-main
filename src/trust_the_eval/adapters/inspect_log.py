from __future__ import annotations
import json
import zipfile

from ..artifact import EvalArtifact, EvalItem


def _final_text(obj) -> str:
    """Best-effort extraction of assistant text from an Inspect output blob."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    # common Inspect shapes: {"choices":[{"message":{"content": "..."}}]}
    try:
        ch = obj.get("choices") or []
        if ch:
            msg = ch[0].get("message", {})
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    except AttributeError:
        pass
    return json.dumps(obj)


def load(path: str) -> EvalArtifact:
    """Load an Inspect `.eval` log (a zip of JSON) into an EvalArtifact.

    Inspect stores samples under `samples/*.json` (or a `samples` array in
    `header.json` / `reductions.json` depending on version). We read whichever
    is present and map sample -> EvalItem(question, answer, response, score).
    """
    samples: list[dict] = []
    header: dict = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for hn in ("header.json", "_journal/header.json"):
            if hn in names:
                header = json.loads(z.read(hn))
                break
        sample_files = [n for n in names if n.startswith("samples/") and n.endswith(".json")]
        if sample_files:
            for n in sorted(sample_files):
                samples.append(json.loads(z.read(n)))
        else:
            for cand in ("samples.json", "reductions.json"):
                if cand in names:
                    blob = json.loads(z.read(cand))
                    samples = blob if isinstance(blob, list) else blob.get("samples", [])
                    break

    items: list[EvalItem] = []
    for s in samples:
        q = s.get("input")
        if isinstance(q, list):  # chat-format input
            q = " ".join(m.get("content", "") for m in q if isinstance(m, dict))
        target = s.get("target")
        if isinstance(target, list):
            target = target[0] if target else ""
        score_obj = s.get("score") or {}
        if isinstance(score_obj, dict):
            val = score_obj.get("value")
        else:
            val = score_obj
        score = {"C": 1.0, "I": 0.0, "P": 0.5}.get(val, val) if val is not None else None
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        items.append(EvalItem(
            question=str(q or ""), answer=str(target or ""),
            response=_final_text(s.get("output")), score=score,
            meta={k: s[k] for k in ("metadata", "id") if k in s},
        ))
    task = header.get("eval", {}).get("task") if header else None
    model = header.get("eval", {}).get("model") if header else None
    return EvalArtifact(dataset=task or "inspect_eval", items=items,
                        model=model, source_path=path, metadata=header)
