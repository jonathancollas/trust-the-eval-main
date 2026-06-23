from __future__ import annotations
from typing import Any

from ..runner import Report

# Emits a dict following the proposed gen_ai.eval.trust.* convention
# (spec/otel-attributes.yaml). It RIDES OpenTelemetry's gen_ai.* namespace.


def to_otel_event(report: Report) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.eval.trust.spec_version": "0.1",
        "gen_ai.eval.trust.dataset": report.artifact.dataset,
        "gen_ai.eval.trust.n_items": report.artifact.n,
        "gen_ai.eval.trust.provenance.hash": report.artifact.content_hash(),
        "gen_ai.eval.trust.probes_skipped": ",".join(report.skipped) or "none",
    }
    for f in report.findings:
        attrs.update(f.otel_attributes)
    if report.artifact.model:
        attrs["gen_ai.request.model"] = report.artifact.model
    if report.cost:
        attrs["gen_ai.eval.trust.cost.calls"] = report.cost.get("calls", 0)
        attrs["gen_ai.eval.trust.cost.usd"] = report.cost.get("est_usd", 0.0)
    return {"event": "gen_ai.eval.trust", "attributes": attrs}


def emit_via_sdk(report: Report) -> bool:
    """Emit through the real OpenTelemetry SDK if installed and configured.

    Returns True if emitted, False if the SDK is unavailable. Kept optional so
    the package has zero hard dependencies.
    """
    try:
        from opentelemetry import _logs  # type: ignore
        from opentelemetry.sdk._logs import LoggerProvider  # type: ignore
    except Exception:
        return False
    event = to_otel_event(report)
    logger = _logs.get_logger("trust_the_eval")
    try:
        from opentelemetry._logs import LogRecord  # type: ignore
        logger.emit(LogRecord(body=event["event"], attributes=event["attributes"]))
        return True
    except Exception:
        return False
