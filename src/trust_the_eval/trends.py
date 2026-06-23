"""Longitudinal trends (P5): what the record history actually shows.

Given a store, :func:`compute_trends` derives, per benchmark, a small set of
falsifiable signals from the claim trajectory and the intrinsic ceiling:

  * **headroom** = ceiling - latest reported score (valid accuracy still below
    the label-error ceiling);
  * **over_ceiling** = the reported score exceeds that ceiling (a red flag: the
    number is partly fitting the benchmark's own label errors / contamination);
  * **last_gain** and whether it is **separable from noise** — using the per-claim
    ``statistical_power`` Wilson intervals when the records carry per-sample data;
  * **saturated** — a conservative roll-up of the above, with its reasons spelled
    out.

Every signal states what it rests on; when a record lacks the per-sample data to
judge noise, the report says so rather than inventing a verdict. Trends describe
the *instruments and claims* over time, never model capability or safety.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .observatory import _summarise

_NEAR_MARGIN = 0.02   # "within 2 points of the ceiling" => effectively at it


def _year(d: Optional[str]) -> Optional[float]:
    if not d:
        return None
    m = re.match(r"(\d{4})(?:-(\d{2}))?", str(d))
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2)) if m.group(2) else 6
    return y + (mo - 0.5) / 12.0


def _halfwidth(record: Dict[str, Any]) -> Optional[float]:
    """The 95% CI half-width of a claim's reported accuracy, if recorded."""
    for a in record.get("audits", []):
        if a.get("probe_id") == "statistical_power":
            m = a.get("measured") or {}
            if m.get("wald_halfwidth") is not None:
                return float(m["wald_halfwidth"])
            if m.get("wilson_lo") is not None and m.get("wilson_hi") is not None:
                return (float(m["wilson_hi"]) - float(m["wilson_lo"])) / 2.0
    return None


def compute_trends(store: Any) -> Dict[str, Dict[str, Any]]:
    """Return per-benchmark trend signals keyed by benchmark name."""
    S = _summarise(store)
    recs = S["records"]
    out: Dict[str, Dict[str, Any]] = {}
    for b in S["benchmarks"]:
        claims = [r for r in recs
                  if r["subject"]["kind"] == "claim"
                  and r["subject"]["benchmark"] == b["name"]
                  and r["subject"].get("reported_score") is not None]
        claims.sort(key=lambda r: r["subject"].get("date") or "")
        C = b["ceiling"]
        t: Dict[str, Any] = {
            "benchmark": b["name"], "slug": b["slug"], "n_claims": len(claims),
            "ceiling": C, "first": None, "latest": None, "span": None,
            "headroom": None, "over_ceiling": None, "near_ceiling": None,
            "last_gain": None, "gain_within_noise": None,
            "noise_basis": "no per-sample CIs in the claim records",
            "slope_per_year": None, "saturated": False, "saturation_reasons": [],
            "notes": []}
        if not claims:
            t["notes"].append("No dated claims recorded yet.")
            out[b["name"]] = t
            continue

        first, latest = claims[0]["subject"], claims[-1]["subject"]
        s0, sN = first["reported_score"], latest["reported_score"]
        d0, dN = first.get("date"), latest.get("date")
        t["first"] = {"date": d0, "score": s0}
        t["latest"] = {"date": dN, "score": sN}
        t["span"] = [d0, dN]

        if C is not None:
            t["headroom"] = round(C - sN, 4)
            t["over_ceiling"] = sN > C + 1e-9
            t["near_ceiling"] = (not t["over_ceiling"]) and (C - sN <= _NEAR_MARGIN)

        if len(claims) >= 2:
            sp = claims[-2]["subject"]["reported_score"]
            t["last_gain"] = round(sN - sp, 4)
            hw_n, hw_p = _halfwidth(claims[-1]), _halfwidth(claims[-2])
            if hw_n is not None and hw_p is not None:
                comb = hw_n + hw_p
                t["gain_within_noise"] = abs(sN - sp) < comb
                t["noise_basis"] = "combined 95% CI half-width \u00b1{:.3f}".format(comb)
            y0, yN = _year(d0), _year(dN)
            if y0 is not None and yN is not None and yN > y0:
                t["slope_per_year"] = round((sN - s0) / (yN - y0), 4)

        reasons = []
        if t["over_ceiling"]:
            reasons.append("reported score exceeds the label-error ceiling")
        elif t["near_ceiling"]:
            reasons.append("within {:.0f} pts of the label-error ceiling".format(_NEAR_MARGIN * 100))
        if t["gain_within_noise"]:
            reasons.append("latest gain is not separable from statistical noise")
        t["saturated"] = bool(reasons)
        t["saturation_reasons"] = reasons
        out[b["name"]] = t
    return out
