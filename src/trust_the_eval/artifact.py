from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalItem:
    question: str = ""
    answer: str = ""                 # ground-truth / reference
    response: Optional[str] = None   # the model's output, if present in the log
    score: Optional[float] = None    # the score the eval assigned, if present
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalArtifact:
    """A completed evaluation result — the thing Trust the Eval probes attack.

    Probes assess the VALIDITY of this artifact as a measurement.
    They never generate attacks against a deployed model.
    """
    dataset: str
    items: list[EvalItem] = field(default_factory=list)
    model: Optional[str] = None
    judge: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def claimed_accuracy(self) -> Optional[float]:
        scored = [it.score for it in self.items if it.score is not None]
        return sum(scored) / len(scored) if scored else None

    def content_hash(self) -> str:
        blob = json.dumps(
            [(it.question, it.answer) for it in self.items],
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()
