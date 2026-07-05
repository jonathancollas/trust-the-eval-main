"""The fragility-atlas view for the observatory.

Renders, as a self-contained static page, the fragility distribution across cells and
the test of the conditional law — straight from the real ``fragility_atlas``, never a
mock. It shows the per-cell table (P(top-1 change), tau, discrimination, closeness),
the law block (correlations with bootstrap CIs and the 2x2 quadrant), and a scatter of
discrimination vs fragility so the sign of the relationship is *visible*.

It audits instruments, not models: no model is scored and no aggregate number rates the
set. No JavaScript, no network.
"""
from __future__ import annotations

import html as _html
from typing import Any, Dict, List

from .atlas import fragility_atlas

_CSS = """
:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;--card:#fff;
--accent:#0f6f6f;--warn:#b45309;--ok:#15803d;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:28px 20px 80px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
h1{font-size:26px;line-height:1.25;margin:.3em 0 .2em;font-weight:600}
.lede{color:var(--mut);max-width:64ch}
.brightline{border-left:3px solid var(--accent);background:#f0f7f7;padding:10px 14px;border-radius:0 8px 8px 0;margin:16px 0;font-size:13.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:18px 0}
h2{font-size:18px;margin:0 0 8px;font-weight:600}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:500}
.bar{display:inline-block;height:8px;background:var(--warn);border-radius:2px;vertical-align:middle}
.sev-high{color:var(--warn);font-weight:600}.sev-medium{color:#a16207}.sev-low{color:var(--ok)}.sev-info{color:var(--mut)}
.law{font-family:var(--mono);font-size:13px;line-height:1.9}
.law b{color:var(--ink)}
.good{color:var(--ok);font-weight:600}.bad{color:var(--warn);font-weight:600}
.quad{display:grid;grid-template-columns:auto 1fr 1fr;gap:2px 10px;font-family:var(--mono);font-size:12.5px;margin-top:8px;max-width:520px}
.quad .h{color:var(--mut)}
.verdict{margin-top:12px;font-size:14px}
figure{margin:6px 0 0}
figcaption{font-size:12px;color:var(--mut);margin-top:6px}
.foot{margin-top:32px;font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:14px}
"""


def _e(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


def _f(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def _scatter(active: List[Dict[str, Any]]) -> str:
    """Hand-drawn SVG: discrimination (x) vs P(top-1 change) (y), one dot per cell."""
    pts = [(r["skill_discrimination"], r["p_top1_change"], r["severity"]) for r in active
           if r["skill_discrimination"] is not None and r["p_top1_change"] is not None]
    if len(pts) < 2:
        return ""
    W, H, pad = 460, 260, 44
    xs = [p[0] for p in pts]
    xmin, xmax = min(xs), max(xs)
    if xmax - xmin < 1e-9:
        xmin, xmax = xmin - 0.1, xmax + 0.1
    ymin, ymax = 0.0, max(0.05, max(p[1] for p in pts))

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (W - 2 * pad)

    def sy(y):
        return H - pad - (y - ymin) / (ymax - ymin) * (H - 2 * pad)

    col = {"high": "#b45309", "medium": "#a16207", "low": "#15803d", "info": "#9ca3af"}
    dots = "".join('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" opacity="0.8"/>'
                   % (sx(x), sy(y), col.get(s, "#6b7280")) for x, y, s in pts)
    x0 = sx(0) if xmin <= 0 <= xmax else None
    zero = ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#d1d5db" stroke-dasharray="3 3"/>'
            % (x0, pad, x0, H - pad)) if x0 is not None else ""
    return (
        '<svg viewBox="0 0 %d %d" width="100%%" style="max-width:480px" role="img" '
        'aria-label="scatter of skill discrimination versus P(top-1 change) across cells">'
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9ca3af"/>'      # y axis
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9ca3af"/>'      # x axis
        '%s%s'
        '<text x="%d" y="%d" font-size="11" fill="#6b7280" text-anchor="middle">skill discrimination of corrected items →</text>'
        '<text x="14" y="%d" font-size="11" fill="#6b7280" transform="rotate(-90 14 %d)" text-anchor="middle">P(top-1 changes) →</text>'
        '<text x="%d" y="%d" font-size="10" fill="#9ca3af">%s</text>'
        '<text x="%d" y="%d" font-size="10" fill="#9ca3af" text-anchor="end">%s</text>'
        '</svg>'
    ) % (W, H,
         pad, pad, pad, H - pad,
         pad, H - pad, W - pad, H - pad,
         zero, dots,
         W // 2, H - 12,
         H // 2, H // 2,
         pad + 2, H - pad + 16, _f(xmin, 2),
         W - pad, H - pad + 16, _f(xmax, 2))


def _law_block(law: Dict[str, Any]) -> str:
    if law.get("insufficient"):
        return '<p class="law">%s (n=%d)</p>' % (_e(law.get("note", "insufficient data")),
                                                 law.get("n", 0))

    def sign(v, want_pos):
        if v is None:
            return "n/a"
        ok = (v > 0) if want_pos else (v < 0)
        cls = "good" if ok else "bad"
        return '<span class="%s">%s</span>' % (cls, _f(v))

    def ci(t):
        return "" if not t or t[0] is None else " [%s, %s]" % (_f(t[0], 2), _f(t[1], 2))

    qm = law["quadrant_mean_p_top1"]
    qc = law["quadrant_counts"]
    rows = ""
    for close, ck in (("close", "close"), ("far", "far")):
        cells = []
        for d, dk in (("discriminating", "disc"), ("flat", "nondisc")):
            k = "%s_%s" % (ck, dk)
            cells.append('<div>%s <span class="mut">(n=%d)</span></div>'
                         % (_f(qm.get(k), 3), qc.get(k, 0)))
        rows += '<div class="h">%s models</div>%s' % (close, "".join(cells))

    verdict = ('<span class="good">HOLDS</span>' if law["law_holds"]
               else '<span class="bad">does NOT cleanly hold</span>')
    return (
        '<p class="law">cells tested: <b>%d</b> · bootstrap CI over cells<br>'
        'corr( P(top1) , discrimination ) = %s%s &nbsp; <span class="mut">(law expects &gt; 0)</span><br>'
        'corr( P(top1) , model gap #1–#2 ) = %s%s &nbsp; <span class="mut">(law expects &lt; 0)</span><br>'
        'corr( P(top1) , correction spread ) = %s%s &nbsp; <span class="mut">(partly mechanical)</span></p>'
        '<div class="quad"><div class="h"></div><div class="h">discriminating</div><div class="h">flat</div>%s</div>'
        '<p class="verdict">verdict: the conditional law %s on this corpus.</p>'
    ) % (law["n"],
         sign(law["corr_p_top1_vs_discrimination"], True), ci(law.get("ci_discrimination")),
         sign(law["corr_p_top1_vs_model_gap"], False), ci(law.get("ci_model_gap")),
         _f(law["corr_p_top1_vs_delta_spread"]), ci(law.get("ci_delta_spread")),
         rows, verdict)


def _cells_table(rows: List[Dict[str, Any]]) -> str:
    active = [r for r in rows if r["n_changed"] > 0 and r["p_top1_change"] is not None]
    if not active:
        return '<p class="mut">No cells with corrections to chart.</p>'
    mx = max(r["p_top1_change"] for r in active) or 1.0
    body = ""
    for r in active:
        w = int(60 * (r["p_top1_change"] / mx)) if mx else 0
        tau = "%s" % _f(r["kendall_tau"], 2)
        if r.get("tau_lo") is not None:
            tau += " [%s,%s]" % (_f(r["tau_lo"], 2), _f(r["tau_hi"], 2))
        body += (
            '<tr><td>%s</td><td>%d</td><td>%s</td>'
            '<td>%s <span class="bar" style="width:%dpx"></span></td>'
            '<td>%s</td><td>%s</td><td class="sev-%s">%s</td></tr>'
        ) % (_e(r["cell"]), r["n_changed"], tau, _f(r["p_top1_change"], 2), w,
             _f(r["skill_discrimination"], 2), _f(r["model_gap_12"], 3),
             _e(r["severity"]), _e(r["severity"]))
    return (
        '<table><thead><tr><th>cell</th><th>chg</th><th>tau</th>'
        '<th>P(top-1 change)</th><th>disc</th><th>gap #1-2</th><th>severity</th></tr></thead>'
        '<tbody>%s</tbody></table>' % body)


def render_atlas_html(pred_rows: Dict[str, List[Dict[str, Any]]],
                      intrinsic_rows: Dict[str, List[Dict[str, Any]]],
                      *, title: str = "Meridian — fragility atlas", iters: int = 1000,
                      seed: int = 0, boot_iters: int = 2000) -> str:
    cells = {b: (pred_rows[b], intrinsic_rows.get(b, [])) for b in pred_rows}
    atlas = fragility_atlas(cells, iters=iters, seed=seed, boot_iters=boot_iters)
    active = [r for r in atlas["rows"] if r["n_changed"] > 0 and r["p_top1_change"] is not None
              and r["skill_discrimination"] is not None]
    scatter = _scatter(active)
    fig = ('<figure>%s<figcaption>Each dot is a benchmark cell. A downward slope means '
           'skill-discriminating corrections are <i>protective</i> of the #1 — the opposite '
           'of the asserted law.</figcaption></figure>' % scatter) if scatter else ""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>%s</title><style>%s</style></head><body><div class=\"wrap\">"
        "<a href=\"index.html\" style=\"display:inline-block;font-size:13px;color:var(--accent);text-decoration:none;margin-bottom:16px\">\u2190 Meridian observatory</a>"
        "<div class=\"eyebrow\">Fragility atlas</div>"
        "<h1>Does correcting known label errors move the ranking — and when?</h1>"
        "<p class=\"lede\">One row per benchmark cell: how often the #1 model changes under "
        "correction (bootstrap), and the two properties that might explain it — whether the "
        "corrected items discriminate skill, and how close the models are.</p>"
        "<div class=\"brightline\">Within-corpus when one overlay meets one model set; widen "
        "with more overlays. No model is scored, and no single number rates the set.</div>"
        "<div class=\"card\"><h2>The conditional law — tested</h2>%s%s</div>"
        "<div class=\"card\"><h2>Cells, by fragility</h2>%s</div>"
        "<div class=\"foot\">Generated from the canonical <span style=\"font-family:%s\">"
        "result_sensitivity</span> (P(top-1), tau, skill_discrimination) — nothing re-derived. "
        "%d cells, %d with corrections. &nbsp;·&nbsp; "
        "<a href=\"lineage.html\" style=\"color:var(--accent)\">Drill into any corrected item\u2019s lineage →</a> "
        "&nbsp;·&nbsp; <a href=\"findings.html\" style=\"color:var(--accent)\">What this means →</a>"
        "</div>"
        "</div></body></html>"
    ) % (_e(title), _CSS, _law_block(atlas["law"]), fig, _cells_table(atlas["rows"]),
         "var(--mono)", atlas["n_cells"], atlas["n_active"])
