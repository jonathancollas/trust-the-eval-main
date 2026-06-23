"""Validity leaderboard (cross-benchmark): rank reported RESULTS and the
BENCHMARKS that host them by *measurement validity*, not by capability.

This is deliberately NOT a capability leaderboard — that would duplicate HELM
and drift toward rating models. Every result is judged against the measured
effective-accuracy ceiling from the intrinsic profile and the reliability of
the instruments behind the score:

  - above ceiling : the reported score numerically exceeds the measured
                    label-error ceiling — the surplus is annotation noise,
                    not capability (the headline Meridian exists to surface);
  - at ceiling    : within a small band of the ceiling — headroom sits inside
                    the noise;
  - unverifiable  : a score-only claim (no per-sample data published) or a
                    benchmark with no measured ceiling — we honestly can't
                    judge it;
  - headroom      : sits below the ceiling and is not undermined by the audits
                    we ran (never "correct", only "not undermined").

Single source of truth: this module computes the verdicts and the ordering;
the observatory SPA and the JSON API both render exactly what it returns. No
formula is re-implemented in JavaScript.

Bright line: ranks eval instruments and claims, never models or their safety.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .calibration.coverage import TIER_LABEL, evidence_for

# ±2 points of the ceiling counts as "at ceiling" (matches the near-ceiling
# band used in trends.py).
NEAR_BAND = 0.02
_EPS = 1e-9

_SEV_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}
_STATUS = {3: ("degraded", "broken"), 2: ("drifting", "drift"),
           1: ("valid", "valid"), 0: ("valid", "valid")}

# verdict key -> (sort rank [lower = more concerning], colour token, label)
_VERDICT = {
    "over_ceiling": (0, "broken", "above ceiling"),
    "near_ceiling": (1, "drift", "at ceiling"),
    "unverifiable": (2, "slate", "unverifiable"),
    "ok":           (3, "valid", "headroom"),
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "x"


def _max_sev(audits: List[Dict[str, Any]]) -> int:
    return max([_SEV_RANK.get(a.get("severity", "info"), 0) for a in audits] or [0])


def _ceiling_of(audits: List[Dict[str, Any]]) -> Optional[float]:
    for a in audits:
        m = a.get("measured") or {}
        if "effective_accuracy_ceiling" in m:
            return m["effective_accuracy_ceiling"]
    return None


def _measured(audits: List[Dict[str, Any]], probe_id: str) -> Dict[str, Any]:
    for a in audits:
        if a.get("probe_id") == probe_id:
            return a.get("measured") or {}
    return {}


def _verify(d: Dict[str, Any]) -> bool:
    try:
        from .record import ValidityRecord
        return ValidityRecord.from_dict(d).verify()
    except Exception:
        return False


def classify(reported: Optional[float], ceiling: Optional[float],
             verifiable: bool) -> Dict[str, str]:
    """Validity verdict for a single reported score. Pure and unit-tested.

    Returns ``{"key": <verdict>, "reason": <one-line explanation>}``. The
    arithmetic verdicts (over/at ceiling) are valid even for score-only claims,
    because they compare the *reported number* to a *measured* ceiling.
    """
    if reported is None:
        return {"key": "unverifiable", "reason": "No reported score."}
    if ceiling is None:
        return {"key": "unverifiable",
                "reason": "No measured ceiling for this benchmark yet."}
    if reported > ceiling + _EPS:
        return {"key": "over_ceiling",
                "reason": ("reported %.1f%% exceeds the measured ceiling %.1f%% — "
                           "the surplus is label noise, not capability."
                           % (reported * 100, ceiling * 100))}
    if reported >= ceiling - NEAR_BAND:
        return {"key": "near_ceiling",
                "reason": ("within %.0f pts of the measured ceiling %.1f%%; "
                           "remaining headroom sits inside the label noise."
                           % (NEAR_BAND * 100, ceiling * 100))}
    if not verifiable:
        return {"key": "unverifiable",
                "reason": ("score-only: no per-sample data published, so "
                           "gain-in-noise can't be judged.")}
    return {"key": "ok",
            "reason": ("%.1f pts of headroom below the measured ceiling; not "
                       "undermined by the audits we ran."
                       % ((ceiling - reported) * 100))}


from .remediation import remediation_plan  # noqa: E402


def leaderboard(store: Any) -> Dict[str, Any]:
    """Build the validity leaderboard (results + benchmarks) from a store."""
    records = [store.get(rid) for rid in store.all_ids()]

    by_bench: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for d in records:
        s = d["subject"]
        b = by_bench.setdefault(s["benchmark"], {"intrinsic": [], "claims": []})
        (b["claims"] if s["kind"] == "claim" else b["intrinsic"]).append(d)

    # Choose the intrinsic record to profile per benchmark (prefer a real
    # intrinsic_audit over imported literature), and read its ceiling.
    chosen: Dict[str, Optional[Dict[str, Any]]] = {}
    ceiling: Dict[str, Optional[float]] = {}
    for name, b in by_bench.items():
        intr = sorted(b["intrinsic"],
                      key=lambda d: 0 if d["provenance"]["method"] == "intrinsic_audit" else 1)
        chosen[name] = intr[0] if intr else None
        ceiling[name] = _ceiling_of(chosen[name]["audits"]) if chosen[name] else None

    # ---- one row per reported result (claim) ----
    results: List[Dict[str, Any]] = []
    for name, b in by_bench.items():
        c = ceiling[name]
        for d in b["claims"]:
            s = d["subject"]
            r = s.get("reported_score")
            audits = d["audits"]
            verifiable = len(audits) > 0
            cls = classify(r, c, verifiable)
            rank, color, label = _VERDICT[cls["key"]]
            instruments = []
            for a in audits:
                pid = a.get("probe_id")
                try:
                    tier = evidence_for(pid)[0]
                except Exception:
                    tier = None
                instruments.append({"probe_id": pid, "severity": a.get("severity"),
                                    "tier": tier, "tier_label": TIER_LABEL.get(tier, "")})
            headroom = (c - r) if (c is not None and r is not None) else None
            results.append({
                "benchmark": name, "slug": _slug(name),
                "model": s.get("model"), "date": s.get("date"),
                "score": r, "metric": s.get("metric"),
                "ceiling": c, "headroom": headroom,
                "verdict": cls["key"], "verdict_label": label, "color": color,
                "rank": rank, "reason": cls["reason"],
                "verifiable": verifiable, "instruments": instruments,
                "record_id": d["record_id"], "verify": _verify(d),
            })
    # Most concerning first; stable multi-pass so ties read sensibly.
    results.sort(key=lambda r: (r["model"] or ""))
    results.sort(key=lambda r: (r["date"] or ""), reverse=True)
    results.sort(key=lambda r: (r["rank"],
                                r["headroom"] if r["headroom"] is not None else 1e9))

    # ---- one row per benchmark (the instrument itself) ----
    benches: List[Dict[str, Any]] = []
    for name, b in by_bench.items():
        ch = chosen[name]
        audits = ch["audits"] if ch else []
        le = _measured(audits, "label_error_audit")
        am = _measured(audits, "item_ambiguity")
        rank = _max_sev(audits + [a for c_ in b["claims"] for a in c_["audits"]])
        word, color = _STATUS[rank]
        le_rate = le.get("rate", le.get("error_rate")) if le else None
        am_rate = am.get("rate") if am else None
        # result_sensitivity may live in its own record (separate source);
        # scan all intrinsic records for the benchmark, latest match wins.
        sens = None
        for _d in b["intrinsic"]:
            _m = _measured(_d["audits"], "result_sensitivity")
            if _m:
                sens = _m
        rel = None
        for _d in b["intrinsic"]:
            _m = _measured(_d["audits"], "test_reliability")
            if _m:
                rel = _m
        all_audits = [a for _d in (b["intrinsic"] + b["claims"]) for a in _d["audits"]]
        remi = remediation_plan(all_audits, sensitivity=sens)
        benches.append({
            "name": name, "slug": _slug(name),
            "dataset_version": ch["subject"].get("dataset_version") if ch else None,
            "ceiling": ceiling[name],
            "n_claims": len(b["claims"]),
            "label_error": ({"rate": le_rate, "k": le.get("k"),
                             "n": le.get("n_annotated")} if le else None),
            "ambiguity": ({"rate": am_rate, "k": am.get("k"),
                           "n": am.get("n_annotated")} if am else None),
            # result-sensitivity (the honest 'can I trust this result' signal),
            # present only where multi-model predictions + corrections were run:
            "sensitivity": sens,
            # internal-consistency reliability (Cronbach alpha) of the test:
            "reliability": rel,
            "remediation": remi,
            "worst_subject": le.get("worst_subject") if le else None,
            "status_word": word, "status_color": color,
            "method": ch["provenance"]["method"] if ch else None,
            "record_id": ch["record_id"] if ch else None,
        })
    # Order by measured label-error rate (worst first); this is a *profile*
    # table, NOT a trust ranking — see result-sensitivity for result trust.
    benches.sort(key=lambda r: (-((r["label_error"] or {}).get("rate") or -1.0),
                                r["name"]))

    tally = {"over_ceiling": 0, "near_ceiling": 0, "unverifiable": 0, "ok": 0}
    for r in results:
        tally[r["verdict"]] += 1

    return {"results": results, "benchmarks": benches, "tally": tally,
            "near_band": NEAR_BAND, "n_results": len(results)}
