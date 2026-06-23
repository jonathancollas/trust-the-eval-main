from __future__ import annotations
import json
from typing import Any

from ..artifact import EvalArtifact, EvalItem


def load(path: str) -> EvalArtifact:
    """Load a generic eval-result JSON into an EvalArtifact.

    Expected shape:
    {
      "dataset": "my_eval",
      "model": "provider/model",   # optional
      "judge": "provider/model",   # optional
      "items": [
        {"question": "...", "answer": "...", "response": "...", "score": 1.0}
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    keep = {"question", "answer", "response", "score"}
    items = [
        EvalItem(
            question=it.get("question", ""),
            answer=it.get("answer", ""),
            response=it.get("response"),
            score=it.get("score"),
            meta={k: v for k, v in it.items() if k not in keep},
        )
        for it in data.get("items", [])
    ]
    return EvalArtifact(
        dataset=data.get("dataset", "unknown"),
        items=items,
        model=data.get("model"),
        judge=data.get("judge"),
        metadata=data.get("metadata", {}),
        source_path=path,
    )
