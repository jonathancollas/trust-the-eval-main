"""Source watermarks: the last fingerprint seen per source.

Lets the autonomy loop ask "is there anything new?" cheaply and skip unchanged
sources, instead of re-pulling and re-deriving everything every cycle. Stored as
``<store>/sources_state.json`` so it travels with the store (and its git history).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SourceState:
    def __init__(self, root: Any):
        self.path = Path(root) / "sources_state.json"
        self._d: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self._d = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._d = {}

    def fingerprint(self, source_id: str) -> Optional[str]:
        e = self._d.get(source_id)
        return e.get("fingerprint") if e else None

    def changed(self, source: Any) -> bool:
        return self.fingerprint(source.id) != source.fingerprint()

    def update(self, source_id: str, fingerprint: str, **extra: Any) -> None:
        self._d[source_id] = dict(extra, fingerprint=fingerprint, last_synced_utc=_now())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._d, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def all(self) -> Dict[str, Any]:
        return dict(self._d)
