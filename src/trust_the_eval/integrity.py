"""Corpus integrity (P5): one digest that attests the whole dataset.

The store already verifies records one by one; this module pins the *entire*
published set to a single :func:`manifest` digest — the SHA-256 over the sorted
content ids. Because each record's id is the hash of its content, adding,
removing, or altering any record changes its id and therefore the corpus digest.
:func:`verify` rolls up record verification, the optional :class:`ProbeRegistry`,
and the manifest into one tamper-evident report a reader can re-check.

Integrity attests the instruments-and-claims dataset; it says nothing about
models or their safety.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest(store: Any) -> Dict[str, Any]:
    """A content digest over all record ids, with counts by kind and method."""
    ids = sorted(store.all_ids())
    digest = "sha256:" + hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    by_kind: Dict[str, int] = {}
    by_method: Dict[str, int] = {}
    for rid in ids:
        d = store.get(rid)
        k = d["subject"]["kind"]
        m = d["provenance"]["method"]
        by_kind[k] = by_kind.get(k, 0) + 1
        by_method[m] = by_method.get(m, 0) + 1
    return {"corpus_digest": digest, "n_records": len(ids),
            "by_kind": by_kind, "by_method": by_method}


def verify(store: Any, registry: Any = None) -> Dict[str, Any]:
    """End-to-end integrity report for the store (and optional registry)."""
    bad: List[str] = store.verify_all()
    rep: Dict[str, Any] = {
        "records_verified": bad == [], "bad_record_ids": bad,
        "generated_utc": _now()}
    rep.update(manifest(store))
    if registry is not None:
        rbad = registry.verify_all()
        rep["registry_verified"] = rbad == []
        rep["bad_admission_ids"] = rbad
        rep["admitted_probes"] = len(registry.admitted())
    rep["verified"] = rep["records_verified"] and (
        registry is None or rep["registry_verified"])
    return rep
