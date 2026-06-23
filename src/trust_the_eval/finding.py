from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class Severity(enum.Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Finding:
    """What a probe returns: a statement about the validity of an eval result."""
    probe_id: str
    severity: Severity
    summary: str
    score: Optional[float] = None  # 0..1, probe-specific (e.g. risk)
    otel_attributes: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
