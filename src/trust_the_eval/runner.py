from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .artifact import EvalArtifact
from .cost import CallBudgetExceeded, CachingClient, CostMeter
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
    requested: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    stopped_reason: Optional[str] = None
    call_estimate: dict = field(default_factory=dict)

    def coverage(self) -> dict:
        total = len(self.requested)
        return {
            "requested": total,
            "completed": len(self.completed),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "fraction": (len(self.completed) / total) if total else 0.0,
            "complete": bool(total) and len(self.completed) == total,
            "stopped_reason": self.stopped_reason,
        }


def _build_probe(cls: type[Probe], overrides: Optional[dict] = None) -> Probe:
    """Instantiate a probe, applying constructor tunables as well as run tunables."""
    requested = overrides or {}
    ctor = {
        name: requested[name]
        for name, spec in cls.TUNABLES.items()
        if spec.get("ctor") and name in requested and requested[name] is not None
    }
    return cls(**ctor).set_overrides(requested)


def estimate_battery_calls(artifact: EvalArtifact,
                           probe_ids: Optional[list[str]] = None,
                           overrides: Optional[dict] = None) -> dict:
    """Estimate provider calls before launch, per probe and in total.

    Estimates are conservative and do not subtract possible cache hits.  They
    include panel/cross-judge clients declared on the artifact or probe even
    though those clients may be billed independently from the primary model.
    """
    classes = all_probes() if probe_ids is None else [get_probe(p) for p in probe_ids]
    per_probe = {}
    for cls in classes:
        probe = _build_probe(cls, (overrides or {}).get(cls.id))
        per_probe[probe.id] = probe.estimate_model_calls(artifact)
    return {"total": sum(per_probe.values()), "per_probe": per_probe,
            "assumption": "upper estimate before cache hits"}


def run_battery(artifact: EvalArtifact,
                model: Optional[ModelClient] = None,
                probe_ids: Optional[list[str]] = None,
                progress: Optional[ProgressCb] = None,
                should_continue: Optional[Callable[[], bool]] = None,
                overrides: Optional[dict] = None,
                max_model_calls: Optional[int] = None) -> Report:
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
    if max_model_calls is not None and max_model_calls < 0:
        raise ValueError("max_model_calls must be >= 0")
    meter = CostMeter(max_calls=max_model_calls)
    client = CachingClient(model, meter) if model is not None else None
    report = Report(artifact=artifact, requested=[c.id for c in classes],
                    call_estimate=estimate_battery_calls(
                        artifact, [c.id for c in classes], overrides))
    total = len(classes)
    for i, cls in enumerate(classes, start=1):
        if should_continue is not None and not should_continue():
            report.stopped_reason = "cancelled"
            break
        probe = _build_probe(cls, (overrides or {}).get(cls.id))
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
            report.completed.append(probe.id)
            if progress:
                progress({"i": i, "total": total, "id": probe.id,
                          "status": "done", "findings": fs})
        except CallBudgetExceeded as exc:
            report.stopped_reason = str(exc)
            if progress:
                progress({"i": i, "total": total, "id": probe.id,
                          "status": "error", "error": str(exc)})
            break
        except Exception as exc:  # a failing probe must not sink the battery
            report.errors[probe.id] = f"{type(exc).__name__}: {exc}"
            if progress:
                progress({"i": i, "total": total, "id": probe.id,
                          "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    if client is not None:
        report.cost = meter.as_dict()
    return report
