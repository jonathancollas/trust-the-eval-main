"""Append-only, content-addressed store for ValidityRecords (P0).

The store is a LOG, not a database you edit:

  - every record is immutable and addressed by the SHA-256 of its canonical
    content (its filename IS its hash);
  - writing a record that already exists is a no-op (idempotent);
  - records are never modified or deleted in place — there is no update/delete;
  - integrity is verifiable: recomputing each record's hash must match its id,
    so any out-of-band edit to a stored file is detectable.

Layout (git-friendly, zero-dependency, human-readable):

    <root>/records/<sha>.json     one immutable record per file
    <root>/index.jsonl            append-only log of compact summaries

This plain file store backs the public INTRINSIC axis so it can live in a git
repo and be diffed and audited line-by-line. A high-volume per-claim feed can
later sit on a database; the record schema is identical either way.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .record import ValidityRecord


def _bare(record_id: str) -> str:
    return record_id.split(":", 1)[-1]


def _subject_key(s: Dict[str, Any]) -> str:
    if s.get("kind") == "claim":
        return "claim:{}:{}".format(s.get("benchmark"), s.get("model") or "?")
    return "intrinsic:{}@{}".format(s.get("benchmark"), s.get("dataset_version") or "?")


class RecordStore:
    """A content-addressed, append-only store of ValidityRecords on disk."""

    def __init__(self, root: Any):
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.index_path = self.root / "index.jsonl"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- write
    def put(self, record: ValidityRecord) -> str:
        """Append a record. Idempotent: identical content keeps one file/id."""
        rid = record.record_id or record.finalize().record_id
        if rid != record.compute_id():
            raise ValueError("record_id does not match its content (tampered or stale)")
        path = self.records_dir / (_bare(rid) + ".json")
        if path.exists():
            return rid  # already stored; immutable — no rewrite, no duplicate index line
        payload = json.dumps(record.to_dict(), sort_keys=True,
                             ensure_ascii=False, indent=2)
        tmp = self.records_dir / (_bare(rid) + ".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(path))            # atomic publish
        self._append_index(record.to_dict())
        return rid

    def _append_index(self, d: Dict[str, Any]) -> None:
        s = d.get("subject", {})
        p = d.get("provenance", {})
        line = {"record_id": d.get("record_id"), "kind": s.get("kind"),
                "benchmark": s.get("benchmark"),
                "dataset_version": s.get("dataset_version"),
                "model": s.get("model"), "release": s.get("release"),
                "date": s.get("date"), "method": p.get("method"),
                "created_utc": p.get("created_utc"),
                "subject_key": _subject_key(s)}
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ read
    def has(self, record_id: str) -> bool:
        return (self.records_dir / (_bare(record_id) + ".json")).exists()

    def get(self, record_id: str) -> Dict[str, Any]:
        path = self.records_dir / (_bare(record_id) + ".json")
        if not path.exists():
            raise KeyError(record_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def get_record(self, record_id: str) -> ValidityRecord:
        return ValidityRecord.from_dict(self.get(record_id))

    def all_ids(self) -> List[str]:
        return sorted("sha256:" + p.stem for p in self.records_dir.glob("*.json"))

    def index(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def list(self, kind: Optional[str] = None, benchmark: Optional[str] = None,
             model: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.index()
        if kind is not None:
            rows = [r for r in rows if r.get("kind") == kind]
        if benchmark is not None:
            rows = [r for r in rows if r.get("benchmark") == benchmark]
        if model is not None:
            rows = [r for r in rows if r.get("model") == model]
        return rows

    def history(self, benchmark: str, model: Optional[str] = None,
                kind: str = "claim") -> List[Dict[str, Any]]:
        """The longitudinal thread for a subject, ordered by date then created.

        This is what the observatory's time-series later reads from."""
        rows = [r for r in self.index()
                if r.get("benchmark") == benchmark and r.get("kind") == kind
                and (model is None or r.get("model") == model)]
        rows.sort(key=lambda r: (r.get("date") or "", r.get("created_utc") or ""))
        return rows

    # ------------------------------------------------------------- integrity
    def verify_all(self) -> List[str]:
        """Return ids of any records whose content no longer matches their id."""
        bad: List[str] = []
        for rid in self.all_ids():
            try:
                rec = ValidityRecord.from_dict(self.get(rid))
                if rec.compute_id() != rid:
                    bad.append(rid)
            except Exception:
                bad.append(rid)
        return bad
