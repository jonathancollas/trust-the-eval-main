"""Structured, scientific documentation attached to each probe.

A probe may define a module/class attribute ``DOC = ProbeDoc(...)``. The web UI
reads it (plus the live source via ``inspect.getsource``) so that, for every
probe, a user can see, in one place and always in sync with the code:

  1. science      - a rigorous description of the phenomenon, with references
  2. math         - the exact estimator the probe computes (rendered + LaTeX)
  3. interpretation - the effect to achieve, how to read the score, thresholds
  4. code         - the actual implementation (served separately by the server)

This is data, not logic: keeping it beside the probe makes the science a
first-class, reviewable part of the artifact rather than an afterthought.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Reference:
    authors: str
    title: str
    venue: str
    year: int
    arxiv: Optional[str] = None      # e.g. "2311.04850"
    url: Optional[str] = None
    note: Optional[str] = None       # why this reference matters here

    def to_dict(self) -> dict:
        url = self.url or (f"https://arxiv.org/abs/{self.arxiv}" if self.arxiv else None)
        return {"authors": self.authors, "title": self.title, "venue": self.venue,
                "year": self.year, "arxiv": self.arxiv, "url": url, "note": self.note}


@dataclass
class MathBlock:
    label: str          # e.g. "Contamination risk"
    html: str           # offline-rendered (CSS-based) markup
    latex: str          # copyable LaTeX source

    def to_dict(self) -> dict:
        return {"label": self.label, "html": self.html, "latex": self.latex}


@dataclass
class Threshold:
    when: str           # e.g. "risk \u2265 0.40"
    severity: str       # "high" | "medium" | "low" | "info"
    meaning: str

    def to_dict(self) -> dict:
        return {"when": self.when, "severity": self.severity, "meaning": self.meaning}


@dataclass
class ProbeDoc:
    science: str                                   # paragraphs separated by \n\n
    effect: str                                    # the effect to achieve
    reading: str                                   # how to interpret the score
    references: List[Reference] = field(default_factory=list)
    math: List[MathBlock] = field(default_factory=list)
    terms: List[Tuple[str, str]] = field(default_factory=list)   # (symbol, definition)
    inputs: List[Tuple[str, str, str]] = field(default_factory=list)  # (evidence key, origin, source)
    thresholds: List[Threshold] = field(default_factory=list)
    caveats: Optional[str] = None                  # confounds / what can fool it
    code_refs: List[str] = field(default_factory=list)  # dotted helper names to also show

    def to_dict(self) -> dict:
        return {
            "science": self.science,
            "effect": self.effect,
            "reading": self.reading,
            "references": [r.to_dict() for r in self.references],
            "math": [m.to_dict() for m in self.math],
            "terms": [{"symbol": s, "definition": d} for s, d in self.terms],
            "inputs": [{"key": k, "origin": o, "source": s} for k, o, s in self.inputs],
            "thresholds": [t.to_dict() for t in self.thresholds],
            "caveats": self.caveats,
            "code_refs": self.code_refs,
        }
