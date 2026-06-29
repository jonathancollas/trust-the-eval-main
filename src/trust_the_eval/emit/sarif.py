"""SARIF 2.1.0 emitter — Meridian findings as static-analysis results.

Maps each probe audit in a store to a SARIF ``result`` so the validity verdicts
render natively as code-scanning alerts (GitHub Advanced Security, the Azure DevOps
SARIF viewer) and annotate pull requests — reusing a universal standard instead of a
bespoke UI. The emitter ``reports``; it never gates (that is ``meridian`` policy).

Two disciplines are baked in and unit-tested:

  * **The bright line.** There is NO aggregate "trust score" result — one ``result``
    per probe finding, never a composite, and no 0-100/letter anywhere.
  * **Evidence honesty.** ``level`` is clamped by the probe's *evidence-strength* tier
    (``validated`` / ``contested`` / ``exploratory``), so a ``contested`` or
    ``exploratory`` probe (e.g. contamination, sandbagging) can NEVER be reported as
    ``error`` — it is ``warning`` at most, ``kind: "review"``, message marked
    INVESTIGATE. Conditional verdicts (``result_sensitivity``) are ``review`` too, not
    a hard ``fail``.

Each result carries the recompute pointer (``meridian explain …``) and the
content-addressed ``record_id`` / ``inputs_hash`` so a reader can verify the number —
the thing other scanners' opaque results never offer.

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"

# evidence-STRENGTH tiers (distinct from coverage's ground-truth provenance tiers)
_VALIDATED, _CONTESTED, _EXPLORATORY = "validated", "contested", "exploratory"

# severity (repo vocabulary) -> base SARIF level
_BASE_LEVEL = {"high": "error", "medium": "warning", "low": "note", "info": "none"}
_LEVEL_RANK = {"none": 0, "note": 1, "warning": 2, "error": 3}
_RANK_LEVEL = {v: k for k, v in _LEVEL_RANK.items()}

# validated probes whose verdict is *conditional* (a judgement call, not a hard fault):
# they are reported as `review`, never `fail`, and never escalate to `error`.
_CONDITIONAL_REVIEW = {"result_sensitivity"}

# probes for which `meridian explain <metric>` can re-derive the displayed value
_EXPLAIN_METRIC = {
    "result_sensitivity": "tau", "test_reliability": "alpha",
    "label_error_audit": "label", "item_ambiguity": "ambiguity",
}


def _sci() -> Dict[str, Any]:
    from ..observatory_ui import SCI   # deferred: pure module-level data
    return SCI


def _tier(probe_id: str) -> str:
    """Evidence-strength tier from the probe's science card; unknown -> exploratory
    (the conservative default, so an uncharacterised probe can never reach `error`)."""
    card = _sci().get(probe_id)
    t = (card or {}).get("tier")
    return t if t in (_VALIDATED, _CONTESTED, _EXPLORATORY) else _EXPLORATORY


def _ground_truth(probe_id: str):
    try:
        from ..calibration.coverage import evidence_for
        tier, source, note = evidence_for(probe_id)
        return {"ground_truth_tier": tier, "ground_truth_source": source}
    except Exception:
        return {}


def _level_and_kind(probe_id: str, severity: str, tier: str) -> (str, str):
    """Map (severity, tier) to a SARIF (level, kind), enforcing the honesty clamp."""
    sev = severity if severity in _BASE_LEVEL else "info"
    base = _BASE_LEVEL[sev]
    # kind
    if sev == "info":
        kind = "pass"
    elif tier != _VALIDATED or probe_id in _CONDITIONAL_REVIEW:
        kind = "review"            # contested/exploratory, or a conditional verdict
    else:
        kind = "fail" if sev == "high" else ("review" if sev == "medium" else "pass")
    # level, clamped
    level = base
    if tier != _VALIDATED and _LEVEL_RANK[level] > _LEVEL_RANK["warning"]:
        level = "warning"          # non-validated probes can never be `error`
    if kind == "review" and level == "error":
        level = "warning"          # a review is not a hard error
    if kind == "pass":
        level = "none"
    return level, kind


def _verify_cmd(probe_id: str, benchmark: str) -> Optional[str]:
    m = _EXPLAIN_METRIC.get(probe_id)
    if not m:
        return None
    return 'meridian explain "%s" %s --sources <your-sources.json>' % (benchmark, m)


def _message(summary: str, tier: str) -> str:
    head = "[%s] %s" % (tier, summary or "")
    if tier in (_CONTESTED, _EXPLORATORY):
        head += "  INVESTIGATE \u2014 a flag for review, not a verdict (detection is weak/defeatable on a static set)."
    return head


def _rule(probe_id: str, base_uri: Optional[str]) -> Dict[str, Any]:
    card = _sci().get(probe_id, {})
    tier = _tier(probe_id)
    # default surfacing level for the rule itself: validated -> warning, else note
    default_level = "warning" if tier == _VALIDATED else "note"
    rule: Dict[str, Any] = {
        "id": probe_id,
        "name": card.get("title", probe_id),
        "shortDescription": {"text": card.get("measures", probe_id)},
        "fullDescription": {"text": card.get("threat", card.get("construct", probe_id))},
        "defaultConfiguration": {"level": default_level},
        "properties": {
            "evidence_strength": tier,
            "tags": [t for t in (card.get("facet"), tier) if t],
            **_ground_truth(probe_id),
        },
    }
    if card.get("refs"):
        rule["properties"]["references"] = card["refs"]
    if base_uri:
        rule["helpUri"] = "%s#probe=%s" % (base_uri.rstrip("/") + "/", probe_id)
    return rule


def _result(probe_id: str, benchmark: str, audit: Dict[str, Any],
            *, model: Optional[str], record_id: Optional[str],
            inputs_hash: Optional[str]) -> Dict[str, Any]:
    tier = _tier(probe_id)
    sev = audit.get("severity", "info")
    level, kind = _level_and_kind(probe_id, sev, tier)
    fq = benchmark if not model else "%s [%s]" % (benchmark, model)
    fp = "%s:%s" % (probe_id, benchmark) if not model else "%s:%s:%s" % (probe_id, benchmark, model)
    props: Dict[str, Any] = {"evidence_strength": tier, "severity": sev}
    if record_id:
        props["record_id"] = record_id
    if inputs_hash:
        props["inputs_hash"] = inputs_hash
    if model:
        props["model"] = model
    verify = _verify_cmd(probe_id, benchmark)
    if verify:
        props["verify"] = verify
    # carry a few measured numbers verbatim (no aggregation), JSON-safe scalars only
    measured = audit.get("measured") or {}
    keep = {k: v for k, v in measured.items()
            if isinstance(v, (int, float, str, bool)) or v is None}
    if keep:
        props["measured"] = keep
    return {
        "ruleId": probe_id,
        "level": level,
        "kind": kind,
        "message": {"text": _message(audit.get("summary", ""), tier)},
        "locations": [{"logicalLocations": [{"fullyQualifiedName": fq, "kind": "namespace"}]}],
        "partialFingerprints": {"tte/v1": fp},
        "properties": props,
    }


def to_sarif(store: Any, *, base_uri: Optional[str] = None,
             tool_version: Optional[str] = None) -> Dict[str, Any]:
    """Build a SARIF 2.1.0 log from a store of validity records.

    One result per (probe, benchmark[, model]) finding; deterministic ordering;
    no aggregate result of any kind."""
    try:
        from .. import __version__ as _v
    except Exception:
        _v = "0"
    records = [store.get(rid) for rid in store.all_ids()]
    by_fp: Dict[str, Dict[str, Any]] = {}     # dedupe by identity; last write wins
    seen_rules: Dict[str, None] = {}
    for d in records:
        subj = d.get("subject") or {}
        benchmark = subj.get("benchmark") or "unknown"
        model = subj.get("model") if subj.get("kind") == "claim" else None
        rid = d.get("record_id")
        ih = (d.get("provenance") or {}).get("inputs_hash")
        for a in d.get("audits") or []:
            pid = a.get("probe_id")
            if not pid:
                continue
            seen_rules.setdefault(pid, None)
            res = _result(pid, benchmark, a, model=model, record_id=rid, inputs_hash=ih)
            by_fp[res["partialFingerprints"]["tte/v1"]] = res
    results = [by_fp[k] for k in sorted(by_fp)]
    rules = [_rule(pid, base_uri) for pid in sorted(seen_rules)]
    driver: Dict[str, Any] = {
        "name": "trust-the-eval",
        "version": tool_version or _v,
        "informationUri": base_uri or "https://github.com/jonathancollas/trust-the-eval",
        "rules": rules,
    }
    return {
        "$schema": _SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": driver},
            "columnKind": "utf16CodeUnits",
            "results": results,
        }],
    }


def write_sarif(store: Any, out_dir: Any, *, base_uri: Optional[str] = None,
                tool_version: Optional[str] = None, filename: str = "meridian.sarif") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(to_sarif(store, base_uri=base_uri, tool_version=tool_version),
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
