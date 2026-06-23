"""Versioned, model-free annotation corpus (P1) — the intrinsic axis's input.

The intrinsic axis audits the DATASET itself, with no model and no scores. Its
input is a corpus pinned to a version: the benchmark's items plus, where a
re-annotation exists, the per-item human ``error_type`` label. The seed is
MMLU-Redux (Gema et al., NAACL 2025), whose taxonomy marks each MMLU item as
``ok`` or as a specific defect — so the corpus carries genuine human ground
truth about label correctness and item clarity.

Two kinds of model-free fact come out of a corpus:
  - what the dataset-level probes compute from the bare items
    (duplicates / empties / canaries via ``dataset_hygiene``; category balance
    via ``coverage_distribution``);
  - what is MEASURED DIRECTLY from the human annotations when present
    (label-error rate, ambiguity rate, per-subject breakdown, the effective
    accuracy ceiling). This is exact, not a probe estimate.

Rows use the published MMLU-Redux schema: ``question, choices, answer`` (gold
index), ``error_type, subject``. The corpus is content-addressed so a record's
``dataset_version`` is backed by a hash, and the same loader path runs against
the live HF dataset on the user's machine (this sandbox has no network).

Bright line: a corpus describes the validity of an eval dataset, never a model.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .artifact import EvalArtifact, EvalItem
from .calibration.realworld import (
    AMBIGUITY_ERROR_TYPES, DEFECT_ERROR_TYPES, OK_ERROR_TYPE, canonical_error_type,
)


@dataclass
class AnnotationCorpus:
    """A benchmark dataset pinned to a version, optionally human-annotated."""
    benchmark: str
    dataset_version: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    source: Optional[Dict[str, Any]] = None      # citation {title, authors, url, dataset}
    note: Optional[str] = None

    # ----------------------------------------------------------------- basics
    @property
    def n(self) -> int:
        return len(self.rows)

    def _etype(self, r: Dict[str, Any]) -> str:
        return canonical_error_type(r.get("error_type")) or OK_ERROR_TYPE

    @property
    def n_annotated(self) -> int:
        return sum(1 for r in self.rows if r.get("error_type") is not None)

    @property
    def has_annotations(self) -> bool:
        return self.n_annotated > 0

    def content_hash(self) -> str:
        """Deterministic version pin over the items + their annotations."""
        key = [(str(r.get("question", "")), int(r.get("answer", 0) or 0),
                self._etype(r), str(r.get("subject") or r.get("source") or ""))
               for r in self.rows]
        blob = json.dumps(key, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    # ----------------------------------------- model-free annotation measures
    def _rate(self, types) -> Dict[str, Any]:
        ann = [r for r in self.rows if r.get("error_type") is not None]
        if not ann:
            return {"rate": None, "k": 0, "n": 0}
        k = sum(1 for r in ann if self._etype(r) in types)
        return {"rate": k / len(ann), "k": k, "n": len(ann)}

    def label_error(self) -> Dict[str, Any]:
        """Fraction of annotated items whose gold label is genuinely defective."""
        return self._rate(DEFECT_ERROR_TYPES)

    def ambiguity(self) -> Dict[str, Any]:
        """Fraction of annotated items flagged as ill-posed (clarity defects)."""
        return self._rate(AMBIGUITY_ERROR_TYPES)

    def by_subject(self, types=DEFECT_ERROR_TYPES) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in self.rows:
            if r.get("error_type") is None:
                continue
            s = str(r.get("subject") or r.get("source") or "mmlu")
            d = out.setdefault(s, {"k": 0, "n": 0})
            d["n"] += 1
            if self._etype(r) in types:
                d["k"] += 1
        for d in out.values():
            d["rate"] = d["k"] / d["n"] if d["n"] else None
        return out

    def worst_subject(self, types=DEFECT_ERROR_TYPES, min_n: int = 10):
        subs = self.by_subject(types)
        cand = [(s, d) for s, d in subs.items()
                if d["n"] >= min_n and d["rate"] is not None]
        if not cand:
            cand = [(s, d) for s, d in subs.items() if d["rate"] is not None]
        if not cand:
            return None
        s, d = max(cand, key=lambda kv: kv[1]["rate"])
        return {"subject": s, "rate": d["rate"], "k": d["k"], "n": d["n"]}

    # ---------------------------------- artifact for the dataset-level probes
    def to_artifact(self) -> EvalArtifact:
        """Build a (response/score-free) EvalArtifact the model-free probes read."""
        items: List[EvalItem] = []
        for r in self.rows:
            choices = [str(c) for c in (r.get("choices") or [])]
            ai = int(r.get("answer", 0) or 0)
            gold = choices[ai] if 0 <= ai < len(choices) else str(r.get("answer", ""))
            subj = str(r.get("subject") or r.get("source") or "mmlu")
            items.append(EvalItem(
                question=str(r.get("question", "")), answer=gold,
                meta={"options": choices, "choices": choices, "answer_index": ai,
                      "category": subj, "subject": subj, "error_type": self._etype(r)}))
        return EvalArtifact(
            dataset="{}@{}".format(self.benchmark, self.dataset_version),
            items=items, model=None,
            metadata={"dataset_version": self.dataset_version,
                      "corpus_hash": self.content_hash()})

    # ------------------------------------------------------------- loaders ---
    @classmethod
    def from_rows(cls, rows, *, benchmark: str, dataset_version: str,
                  source: Optional[Dict[str, Any]] = None,
                  note: Optional[str] = None) -> "AnnotationCorpus":
        return cls(benchmark=benchmark, dataset_version=dataset_version,
                   rows=list(rows), source=source, note=note)

    @classmethod
    def from_mmlu_redux(cls, path: Optional[str] = None, *,
                        dataset_version: str = "mmlu-redux-2.0",
                        hf_subset: str = "all",
                        source: Optional[Dict[str, Any]] = None) -> "AnnotationCorpus":
        """Load MMLU-Redux as a corpus. `path` reads a local export (offline,
        reproducible); otherwise the package's HF source fetches it (network,
        user machine). Same code path either way."""
        from .calibration.realworld import load_mmlu_redux_rows
        rows = load_mmlu_redux_rows(path=path, hf_subset=hf_subset)
        src = source or {"title": "Are We Done with MMLU? (MMLU-Redux)",
                         "authors": "Gema et al., NAACL 2025",
                         "url": "https://arxiv.org/abs/2406.04127",
                         "dataset": "edinburgh-dawg/mmlu-redux-2.0"}
        return cls(benchmark="MMLU", dataset_version=dataset_version, rows=rows,
                   source=src, note="Human-re-annotated MMLU (error_type per item).")

    @classmethod
    def synthetic_seed(cls, *, seed: int = 0) -> "AnnotationCorpus":
        """A tiny offline corpus with planted annotations, so the intrinsic job
        runs and is testable without network (stands in for MMLU-Redux here)."""
        rng = random.Random(seed)
        plan = {"virology": ("wrong_groundtruth", 0.5),
                "logic": ("multiple_correct_answers", 0.1),
                "history": ("ok", 0.0), "math": ("ok", 0.0)}
        rows: List[Dict[str, Any]] = []
        idx = 0
        for subj, (etype, frac) in plan.items():
            for j in range(12):
                et = etype if (j / 12.0) < frac else "ok"
                if subj == "logic" and j in (10, 11):
                    et = "bad_question_clarity"
                rows.append({"question": "{} question {}?".format(subj, j),
                             "choices": ["opt{}_{}".format(idx, o) for o in range(4)],
                             "answer": j % 4, "error_type": et, "subject": subj})
                idx += 1
        rng.shuffle(rows)
        return cls(benchmark="ToyMMLU", dataset_version="synthetic-seed-1", rows=rows,
                   source={"title": "Synthetic annotated seed (offline)", "url": ""},
                   note="Planted annotations; offline stand-in for MMLU-Redux when no network.")
