from __future__ import annotations
from typing import Any

from ..finding import Severity
from ..runner import Report

_MARK = {Severity.INFO: "·", Severity.LOW: "+", Severity.MEDIUM: "!", Severity.HIGH: "x"}
_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}


def to_console(report: Report) -> str:
    a = report.artifact
    out = [
        "Trust the Eval - evaluation validity report",
        f"dataset: {a.dataset} - {a.n} items - {a.content_hash()[:23]}...",
        f"model: {a.model or '-'} - judge: {a.judge or '-'}",
        "",
    ]
    for f in sorted(report.findings, key=lambda f: _ORDER[f.severity]):
        sc = "" if f.score is None else f"  [{f.score:.2f}]"
        out.append(f"  [{_MARK[f.severity]}] {f.probe_id}{sc}  {f.summary}")
    for pid in report.skipped:
        out.append(f"  [o] {pid}  SKIPPED - needs model access (this is where the value is)")
    for pid, err in report.errors.items():
        out.append(f"  [E] {pid}  ERROR - {err}")
    if report.cost:
        out.append("")
        out.append(f"  cost: {report.cost['calls']} model calls "
                   f"({report.cost['cache_hits']} cached), ~{report.cost['est_usd']} USD")
    return "\n".join(out)


def to_dict(report: Report) -> dict[str, Any]:
    return {
        "dataset": report.artifact.dataset,
        "n_items": report.artifact.n,
        "provenance_hash": report.artifact.content_hash(),
        "findings": [
            {"probe": f.probe_id, "severity": f.severity.value, "summary": f.summary,
             "score": f.score, "evidence": f.evidence}
            for f in report.findings
        ],
        "skipped": report.skipped,
        "errors": report.errors,
        "cost": report.cost,
    }
