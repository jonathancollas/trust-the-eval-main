"""Data sources for autonomous, continuous observation.

A :class:`Source` knows WHERE a kind of data lives and HOW to tell whether it
changed. Two kinds: ``intrinsic`` (an annotation corpus -> the dataset's own
validity) and ``claim`` (a leaderboard of dated results -> the trajectory).
Each Source exposes:

  * ``fingerprint()`` -- a content/version id (file hash, HF commit SHA, ETag);
    the autonomy loop compares it to a stored watermark to skip unchanged data.
  * ``fetch()`` -- the normalised payload
    ``{source_id, kind, benchmark, dataset_version, rows, fingerprint}``.

:class:`LocalFileSource` works fully offline (and in tests). :class:`HFDatasetSource`
and :class:`RemoteLeaderboardSource` use only ``urllib`` (no extra deps) and run
wherever there is network. :func:`build_sources` turns a JSON spec into objects.
Sources carry the data, never a verdict; provenance keeps the source id + fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _read_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
    data = json.loads(txt)
    return data if isinstance(data, list) else data.get("rows", data.get("items", []))


class Source:
    """Base class. Subclasses implement ``fingerprint`` and ``fetch``."""

    def __init__(self, id: str, kind: str, benchmark: str,
                 dataset_version: Optional[str] = None,
                 anchors: Optional[Dict[str, Any]] = None):
        if kind not in ("intrinsic", "claim", "predictions"):
            raise ValueError("kind must be 'intrinsic', 'predictions' or 'claim'")
        self.id = id
        self.kind = kind
        self.benchmark = benchmark
        self.dataset_version = dataset_version or benchmark
        self.anchors = anchors or {}

    def fingerprint(self) -> str:
        raise NotImplementedError

    def fetch(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _payload(self, rows: List[Dict[str, Any]], fingerprint: str) -> Dict[str, Any]:
        return {"source_id": self.id, "kind": self.kind, "benchmark": self.benchmark,
                "dataset_version": self.dataset_version, "rows": rows,
                "fingerprint": fingerprint}


class LocalFileSource(Source):
    """A JSON/JSONL file on disk (offline; the testable + sandbox path)."""

    def __init__(self, path: str, **kw: Any):
        super().__init__(**kw)
        self.path = str(path)

    def fingerprint(self) -> str:
        return _sha(Path(self.path).read_bytes())

    def fetch(self) -> Dict[str, Any]:
        return self._payload(_read_rows(self.path), self.fingerprint())


class HFDatasetSource(Source):
    """A Hugging Face dataset config/split (network).

    Pinned to the dataset's main-branch commit SHA, so the fingerprint changes
    only when the dataset itself changes. Intended for the intrinsic axis
    (annotation corpora like MMLU-Redux).
    """

    def __init__(self, repo: str, config: str = "default", split: str = "test",
                 **kw: Any):
        super().__init__(**kw)
        self.repo = repo
        self.config = config
        self.split = split

    def fingerprint(self) -> str:
        import urllib.request
        url = "https://huggingface.co/api/datasets/%s/refs" % self.repo
        with urllib.request.urlopen(url, timeout=30) as r:
            refs = json.loads(r.read().decode("utf-8"))
        for b in refs.get("branches", []):
            if b.get("name") == "main":
                return "hf:" + str(b.get("targetCommit", ""))[:16]
        return "hf:" + hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest()[:16]

    def fetch(self) -> Dict[str, Any]:
        from .hf import fetch_rows
        raw = fetch_rows(self.repo, self.config, self.split, n=2000)
        rows = [{"question": r.get("question", ""), "choices": r.get("choices", []),
                 "answer": r.get("answer", 0),
                 "error_type": r.get("error_type", "ok"),
                 "subject": r.get("subject") or self.config} for r in raw]
        return self._payload(rows, self.fingerprint())


class RemoteLeaderboardSource(Source):
    """A JSON/JSONL results file at a URL (network) -- e.g. a HELM export."""

    def __init__(self, url: str, **kw: Any):
        super().__init__(**kw)
        self.url = url

    def _bytes(self) -> bytes:
        import urllib.request
        with urllib.request.urlopen(self.url, timeout=60) as r:
            return r.read()

    def fingerprint(self) -> str:
        return _sha(self._bytes())

    def fetch(self) -> Dict[str, Any]:
        raw = self._bytes().decode("utf-8")
        if self.url.endswith(".jsonl"):
            rows = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        else:
            d = json.loads(raw)
            rows = d if isinstance(d, list) else d.get("rows", d.get("items", []))
        return self._payload(rows, self.fingerprint())


_TYPES = {"local": LocalFileSource, "hf_dataset": HFDatasetSource,
          "leaderboard_url": RemoteLeaderboardSource}


def build_sources(spec: List[Dict[str, Any]]) -> List[Source]:
    """Turn a list of ``{"type": ..., **kwargs}`` dicts into Source objects."""
    out: List[Source] = []
    for s in spec:
        s = dict(s)
        cls = _TYPES[s.pop("type")]
        out.append(cls(**s))
    return out


def load_sources(path: str) -> List[Source]:
    return build_sources(json.loads(Path(path).read_text(encoding="utf-8")))
