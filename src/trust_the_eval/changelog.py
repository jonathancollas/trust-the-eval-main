"""Diff successive observatory states into a feed of notable changes.

This is the active half of "observe": not just rendering the current picture,
but flagging what moved since last cycle -- a status flip, a new dated claim, a
benchmark crossing its label-error ceiling, a saturation verdict changing. The
feed (``feed.json``) is what a watcher, dashboard, or notifier consumes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _by_name(doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {b["name"]: b for b in (doc or {}).get("benchmarks", [])}


def diff_exports(prev: Optional[Dict[str, Any]], curr: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare two ``meridian.json`` documents; return a list of change dicts."""
    changes: List[Dict[str, Any]] = []
    pp, cc = _by_name(prev), _by_name(curr)
    for name, b in cc.items():
        if name not in pp:
            changes.append({"type": "new_benchmark", "benchmark": name,
                            "status": b.get("status")})
            continue
        a = pp[name]
        if a.get("status") != b.get("status"):
            changes.append({"type": "status_change", "benchmark": name,
                            "from": a.get("status"), "to": b.get("status")})
        seen = {(c.get("model"), c.get("date")) for c in a.get("trajectory", [])}
        for c in b.get("trajectory", []):
            if (c.get("model"), c.get("date")) not in seen:
                changes.append({"type": "new_claim", "benchmark": name,
                                "model": c.get("model"), "date": c.get("date"),
                                "score": c.get("score")})
        ta, tb = (a.get("trends") or {}), (b.get("trends") or {})
        if ta.get("saturated") != tb.get("saturated"):
            changes.append({"type": "saturation_change", "benchmark": name,
                            "to": tb.get("saturated"),
                            "reasons": tb.get("saturation_reasons")})
        if (tb.get("over_ceiling") is not None
                and ta.get("over_ceiling") != tb.get("over_ceiling")):
            changes.append({"type": "over_ceiling_change", "benchmark": name,
                            "to": tb.get("over_ceiling")})
    return changes


def write_feed(changes: List[Dict[str, Any]], out_dir: Any,
               *, integrity: Optional[Dict[str, Any]] = None) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "feed.json"
    p.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_changes": len(changes), "changes": changes,
        "corpus_digest": (integrity or {}).get("corpus_digest"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)
