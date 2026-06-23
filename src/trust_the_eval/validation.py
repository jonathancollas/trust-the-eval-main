"""Fidelity gate for autonomous pulls.

Before any fetched payload touches the store, it must pass per-source anchors:
schema presence, row count, and -- crucially -- a parsed statistic inside its
expected band (e.g. "virology label-error must be 50-65%, matching the paper").
A failing payload is QUARANTINED and never ingested. This is the integrity-first
stance applied to automation: schema drift or a mis-parse fails the cycle loudly
rather than silently polluting the longitudinal record.

Anchors (all optional, declared per source):
    min_rows (int)                  -- minimum row count
    require_fields (list[str])      -- columns that must be present in every row
    label_error_between [lo, hi]    -- (intrinsic) parsed label-error band
    score_between [lo, hi]          -- (claim) accept scores only in this range
    dates_parseable (bool)          -- (claim) every date must look like YYYY[-MM]
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .calibration.realworld import DEFECT_ERROR_TYPES, canonical_error_type

_DATE = re.compile(r"^\d{4}(-\d{2})?")
_SCORE_KEYS = ("score", "acc", "accuracy", "acc_norm", "value")


def _label_error_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
    et = [canonical_error_type(r.get("error_type")) for r in rows]
    et = [e for e in et if e]
    if not et:
        return None
    return sum(1 for e in et if e in DEFECT_ERROR_TYPES) / len(et)


def validate(payload: Dict[str, Any],
             anchors: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    anchors = anchors or {}
    rows = payload.get("rows") or []
    reasons: List[str] = []

    n = len(rows)
    if n < int(anchors.get("min_rows", 1)):
        reasons.append("only %d rows (need >= %s)" % (n, anchors.get("min_rows", 1)))
    for f in anchors.get("require_fields", []):
        if not rows or not all(f in r for r in rows):
            reasons.append("required field '%s' missing in some rows" % f)

    if payload.get("kind") == "intrinsic":
        if rows and not any("error_type" in r for r in rows):
            reasons.append("intrinsic source has no 'error_type' column")
        band = anchors.get("label_error_between")
        if band:
            rate = _label_error_rate(rows)
            if rate is None or not (band[0] - 1e-9 <= rate <= band[1] + 1e-9):
                reasons.append("label-error %s outside expected band %s"
                               % ("%.3f" % rate if rate is not None else "n/a", band))

    if payload.get("kind") == "claim":
        lo, hi = anchors.get("score_between", [0.0, 1.0])
        for r in rows:
            v = None
            for k in _SCORE_KEYS:
                if r.get(k) is not None:
                    v = float(r[k])
                    break
            if v is None:
                continue
            if v > 1.0:
                v = v / 100.0
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                reasons.append("a score (%.3f) outside [%s, %s]" % (v, lo, hi))
                break
        if anchors.get("dates_parseable"):
            for r in rows:
                d = r.get("date") or r.get("release") or r.get("release_date")
                if d is not None and not _DATE.match(str(d)):
                    reasons.append("unparseable date '%s'" % d)
                    break

    return (not reasons), reasons


def quarantine(root: Any, source_id: str, fingerprint: Optional[str],
               payload: Dict[str, Any], reasons: List[str]) -> str:
    q = Path(root) / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    fp = (fingerprint or "nofp").replace(":", "_")[:24]
    p = q / ("%s-%s.json" % (source_id, fp))
    p.write_text(json.dumps({
        "source_id": source_id, "fingerprint": fingerprint, "reasons": reasons,
        "n_rows": len(payload.get("rows") or []),
        "quarantined_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)
