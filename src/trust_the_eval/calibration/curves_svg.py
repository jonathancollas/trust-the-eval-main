"""Offline SVG rendering of ROC and precision-recall curves.

Pure-Python string templating — no matplotlib, no JS, no external assets — so
the calibration curves render in any browser and embed directly in the report
or the probe-doc drawer. One function per curve, plus a combined panel.

Design matches the rest of the UI: a light grid, a diagonal reference for ROC,
the prevalence baseline for PR, and the area (AUC / AP) annotated.
"""
from __future__ import annotations

from typing import Optional

from .core import Curves

_W = 320
_H = 320
_PAD = 44


def _scale_x(v: float) -> float:
    return _PAD + v * (_W - 2 * _PAD)


def _scale_y(v: float) -> float:
    # invert: 0 at bottom, 1 at top
    return _H - _PAD - v * (_H - 2 * _PAD)


def _axes(title: str, xlabel: str, ylabel: str) -> list[str]:
    el = []
    # plotting box
    el.append(f'<rect x="{_PAD}" y="{_PAD}" width="{_W-2*_PAD}" height="{_H-2*_PAD}" '
              f'fill="#fff" stroke="#d4d4d8"/>')
    # gridlines at 0.25/0.5/0.75
    for g in (0.25, 0.5, 0.75):
        x = _scale_x(g); y = _scale_y(g)
        el.append(f'<line x1="{x:.1f}" y1="{_PAD}" x2="{x:.1f}" y2="{_H-_PAD}" '
                  f'stroke="#f1f1f4"/>')
        el.append(f'<line x1="{_PAD}" y1="{y:.1f}" x2="{_W-_PAD}" y2="{y:.1f}" '
                  f'stroke="#f1f1f4"/>')
    # tick labels 0 and 1
    el.append(f'<text x="{_PAD}" y="{_H-_PAD+14}" font-size="9" fill="#71717a">0</text>')
    el.append(f'<text x="{_W-_PAD-4}" y="{_H-_PAD+14}" font-size="9" fill="#71717a">1</text>')
    el.append(f'<text x="{_PAD-16}" y="{_H-_PAD+3}" font-size="9" fill="#71717a">0</text>')
    el.append(f'<text x="{_PAD-16}" y="{_PAD+6}" font-size="9" fill="#71717a">1</text>')
    # titles
    el.append(f'<text x="{_W/2:.0f}" y="16" font-size="12" font-weight="600" '
              f'fill="#27272a" text-anchor="middle">{title}</text>')
    el.append(f'<text x="{_W/2:.0f}" y="{_H-8}" font-size="10" fill="#52525b" '
              f'text-anchor="middle">{xlabel}</text>')
    el.append(f'<text x="12" y="{_H/2:.0f}" font-size="10" fill="#52525b" '
              f'text-anchor="middle" transform="rotate(-90 12 {_H/2:.0f})">{ylabel}</text>')
    return el


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    pts = " ".join(f"{_scale_x(x):.1f},{_scale_y(y):.1f}" for x, y in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round"/>')


def roc_svg(curves: Curves, color: str = "#2563eb") -> str:
    """Render the ROC curve (TPR vs FPR) with the chance diagonal and AUC."""
    el = [f'<svg viewBox="0 0 {_W} {_H}" xmlns="http://www.w3.org/2000/svg" '
          f'role="img" aria-label="ROC curve">']
    el += _axes("ROC", "false positive rate", "true positive rate")
    # chance diagonal
    el.append(f'<line x1="{_scale_x(0):.1f}" y1="{_scale_y(0):.1f}" '
              f'x2="{_scale_x(1):.1f}" y2="{_scale_y(1):.1f}" '
              f'stroke="#a1a1aa" stroke-dasharray="4 3"/>')
    if curves.points:
        pts = sorted(((p.fpr, p.tpr) for p in curves.points))
        el.append(_polyline(pts, color))
    if curves.roc_auc is not None:
        el.append(f'<text x="{_W-_PAD-4}" y="{_PAD+16}" font-size="11" '
                  f'font-weight="600" fill="{color}" text-anchor="end">'
                  f'AUC = {curves.roc_auc:.3f}</text>')
    el.append("</svg>")
    return "".join(el)


def pr_svg(curves: Curves, color: str = "#16a34a") -> str:
    """Render the precision-recall curve with the prevalence baseline and AP."""
    el = [f'<svg viewBox="0 0 {_W} {_H}" xmlns="http://www.w3.org/2000/svg" '
          f'role="img" aria-label="precision-recall curve">']
    el += _axes("Precision\u2013Recall", "recall", "precision")
    if curves.prevalence is not None:
        y = _scale_y(curves.prevalence)
        el.append(f'<line x1="{_PAD}" y1="{y:.1f}" x2="{_W-_PAD}" y2="{y:.1f}" '
                  f'stroke="#a1a1aa" stroke-dasharray="4 3"/>')
        el.append(f'<text x="{_PAD+4}" y="{y-4:.1f}" font-size="8" fill="#a1a1aa">'
                  f'prevalence {curves.prevalence:.2f}</text>')
    if curves.points:
        pts = sorted(((p.recall, p.precision) for p in curves.points))
        el.append(_polyline(pts, color))
    if curves.average_precision is not None:
        el.append(f'<text x="{_W-_PAD-4}" y="{_PAD+16}" font-size="11" '
                  f'font-weight="600" fill="{color}" text-anchor="end">'
                  f'AP = {curves.average_precision:.3f}</text>')
    el.append("</svg>")
    return "".join(el)


def curves_panel(curves: Curves) -> str:
    """Both curves side by side; a graceful message if a curve is undefined."""
    if not curves.points:
        return ('<div class="te-curve-empty" style="color:#71717a;font-size:12px;'
                'padding:12px">Curve undefined (only one class present in the '
                'calibration set for this probe).</div>')
    return (f'<div class="te-curves" style="display:flex;gap:16px;flex-wrap:wrap">'
            f'<div>{roc_svg(curves)}</div><div>{pr_svg(curves)}</div></div>')
