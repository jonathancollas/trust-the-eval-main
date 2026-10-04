"""Per-claim verdict: what can this eval result support?

Humble, evidence-backed verdict per CLAIM TYPE (capability / ranking /
longitudinal / judge-scored), mirroring the UI rule in
src/trust_the_eval/ui/index.html (claimVerdict) EXACTLY — the node harness
cross-checks the two implementations on real findings.

Rule (per claim):
- consider only findings from probes relevant to the claim (taxonomy);
- threats = findings of severity MEDIUM or higher;
- 'not supportable'  if a HIGH-severity threat comes from a probe whose
  calibration evidence tier is real_labeled or structural_exact (a probe we
  can actually vouch for);
- 'fragile'          if any HIGH (from a synthetic-floor probe) or any MEDIUM;
- 'not assessed'     if no relevant probe produced a finding;
- 'inconclusive'     if fewer than half of the relevant probes ran;
- 'supportable'      otherwise — meaning *not undermined by the threats we
  tested with adequate coverage*, never "proven correct".

This assesses the SCORE as a measurement; it does not certify the model.
"""
from __future__ import annotations

from typing import Any, Iterable

from .calibration.coverage import TIER_REAL, TIER_STRUCTURAL, evidence_for
from .taxonomy import CLAIM_TYPES

SEV_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}
VERDICT_WORD = {"sup": "supportable", "fra": "fragile", "uns": "not supportable",
                "na": "not assessed", "inc": "inconclusive"}
VERDICT_MEANING = {
    "sup": "Not undermined by the validity threats we test.",
    "fra": "Usable with caveats — address the threats below before relying on it.",
    "uns": "The score can't support this claim as a clean measurement until these are fixed.",
    "na": "No relevant probe produced a measurement.",
    "inc": "Too few relevant probes ran to support or undermine this claim.",
}

_RELIABLE_TIERS = (TIER_REAL, TIER_STRUCTURAL)


def _norm(finding: Any) -> dict:
    """Accept Finding objects or finding dicts; return {probe_id, severity, score, summary}."""
    if isinstance(finding, dict):
        return {"probe_id": finding.get("probe_id"),
                "severity": str(finding.get("severity", "info")),
                "score": finding.get("score"),
                "summary": finding.get("summary", "")}
    sev = getattr(finding, "severity", "info")
    return {"probe_id": getattr(finding, "probe_id", getattr(finding, "id", None)),
            "severity": getattr(sev, "value", str(sev)),
            "score": getattr(finding, "score", None),
            "summary": getattr(finding, "summary", "")}


def _is_reliable(probe_id: str) -> bool:
    try:
        tier, _, _ = evidence_for(probe_id)
    except Exception:
        return False
    return tier in _RELIABLE_TIERS


def claim_verdict(claim: str, findings: Iterable[Any]) -> dict:
    """Verdict for one claim. Returns {verdict, word, meaning, threats, n_probes}."""
    ids = set(CLAIM_TYPES.get(claim, ()))
    rel = [_norm(f) for f in findings]
    rel = [f for f in rel if f["probe_id"] in ids]
    threats = [f for f in rel if SEV_RANK.get(f["severity"], 0) >= 2]
    threats.sort(key=lambda f: (-SEV_RANK.get(f["severity"], 0), f["probe_id"]))
    has_high_reliable = any(f["severity"] == "high" and _is_reliable(f["probe_id"])
                            for f in threats)
    has_high = any(f["severity"] == "high" for f in threats)
    has_med = any(f["severity"] == "medium" for f in threats)
    coverage = len({f["probe_id"] for f in rel}) / len(ids) if ids else 0.0
    if not rel:
        v = "na"
    elif has_high_reliable:
        v = "uns"
    elif has_high or has_med:
        v = "fra"
    elif coverage < 0.5:
        v = "inc"
    else:
        v = "sup"
    return {"verdict": v, "word": VERDICT_WORD[v], "meaning": VERDICT_MEANING[v],
            "threats": threats, "n_probes": len(rel), "coverage": coverage,
            "n_expected": len(ids)}


def claim_verdicts(findings: Iterable[Any]) -> dict[str, dict]:
    """Verdicts for all claim types."""
    fs = list(findings)
    return {claim: claim_verdict(claim, fs) for claim in CLAIM_TYPES}
