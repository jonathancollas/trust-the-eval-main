from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .artifact import EvalArtifact
from .cost import CachingClient, CostMeter
from .finding import Finding
from .probe import ModelClient, Probe, all_probes, get_probe

# A progress callback receives a dict per probe lifecycle event:
#   {"i": int, "total": int, "id": str,
#    "status": "running"|"done"|"skipped"|"error",
#    "findings": [Finding] (on done), "error": str (on error)}
ProgressCb = Callable[[Dict[str, Any]], None]


@dataclass
class Report:
    artifact: EvalArtifact
    findings: list[Finding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    cost: dict = field(default_factory=dict)


def run_battery(artifact: EvalArtifact,
                model: Optional[ModelClient] = None,
                probe_ids: Optional[list[str]] = None,
                progress: Optional[ProgressCb] = None,
                should_continue: Optional[Callable[[], bool]] = None,
                overrides: Optional[dict] = None) -> Report:
    """Run a BATTERY of validity probes (parallel & independent; order-free).

    Model-in-the-loop probes are SKIPPED when no model is provided. All model
    calls are routed through a CachingClient so cost is metered and replayable.

    Optional, backward-compatible hooks (used by the web UI):
      - progress(event): called as each probe starts/finishes/skips/errors.
      - should_continue(): checked before each probe; return False to stop early.
    """
    classes: list[type[Probe]] = (
        all_probes() if probe_ids is None else [get_probe(p) for p in probe_ids]
    )
    meter = CostMeter()
    client = CachingClient(model, meter) if model is not None else None
    report = Report(artifact=artifact)
    total = len(classes)
    for i, cls in enumerate(classes, start=1):
        if should_continue is not None and not should_continue():
            break
        probe = cls()
        if overrides and probe.id in overrides:
            probe.set_overrides(overrides[probe.id])
        if probe.requires_model and client is None:
            report.skipped.append(probe.id)
            if progress:
                progress({"i": i, "total": total, "id": probe.id, "status": "skipped"})
            continue
        if progress:
            progress({"i": i, "total": total, "id": probe.id, "status": "running"})
        try:
            fs = probe.run(artifact, client)
            report.findings.extend(fs)
            if progress:
                progress({"i": i, "total": total, "id": probe.id,
                          "status": "done", "findings": fs})
        except Exception as exc:  # a failing probe must not sink the battery
            report.errors[probe.id] = f"{type(exc).__name__}: {exc}"
            if progress:
                progress({"i": i, "total": total, "id": probe.id,
                          "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    if client is not None:
        report.cost = meter.as_dict()
    return report
