from __future__ import annotations
import abc
from typing import Any, Optional

from .artifact import EvalArtifact
from .finding import Finding


class ModelClient(abc.ABC):
    """Minimal interface a model-in-the-loop probe needs.

    Implementations: real providers in `model/providers.py`, controllable local
    models in `model/local.py`. Static probes ignore it.
    """

    @abc.abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.0, **kw: Any) -> str:
        ...

    def sample(self, prompt: str, n: int, *, temperature: float = 1.0, **kw: Any) -> list[str]:
        """Default n-sampling: call complete() n times. Providers may override."""
        return [self.complete(prompt, temperature=temperature, _variant=i, **kw)
                for i in range(n)]


class Probe(abc.ABC):
    id: str = "probe"
    name: str = "Probe"
    paper_priority: str = ""
    requires_model: bool = False
    # {name: {"default":.., "min":.., "max":.., "step":.., "help":".."}} of arbitrary
    # constants surfaced for editing in the UI. Defaults MUST equal the literals
    # previously hard-coded in run(), so behaviour is unchanged unless overridden.
    TUNABLES: dict = {}

    def tune(self, name: str):
        """Return an overridden tunable if set on this instance, else its default."""
        ov = getattr(self, "_overrides", None) or {}
        if name in ov and ov[name] is not None:
            return ov[name]
        return self.TUNABLES[name]["default"]

    def set_overrides(self, overrides):
        """Set per-run overrides for declared TUNABLES (ignores unknown keys)."""
        self._overrides = {k: v for k, v in (overrides or {}).items() if k in self.TUNABLES}
        return self

    @abc.abstractmethod
    def run(self, artifact: EvalArtifact,
            model: Optional[ModelClient] = None) -> list[Finding]:
        ...


_REGISTRY: dict[str, type[Probe]] = {}


def register(cls: type[Probe]) -> type[Probe]:
    if not getattr(cls, "id", None):
        raise ValueError("Probe must define a unique `id`")
    if cls.id in _REGISTRY:
        raise ValueError(f"Duplicate probe id: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def all_probes() -> list[type[Probe]]:
    return list(_REGISTRY.values())


def get_probe(probe_id: str) -> type[Probe]:
    return _REGISTRY[probe_id]
