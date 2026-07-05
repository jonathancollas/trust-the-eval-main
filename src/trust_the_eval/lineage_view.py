"""The lineage drill-down view for the observatory.

Renders, as a self-contained static page, the per-datum lineage thread for every item
whose key a correction moved — straight from the real ``lineage`` function, never a
mock. Each item expands to its full journey: imported row → correction policy →
correctness (the per-model flip) → accuracy → ranking effect → verdict, every step
showing its input, rule, exact formula, output, and scientific source.

It audits the instrument, never the model: the verdict is about the benchmark's
ranking, and no model is scored. No JavaScript, no network — a plain page an auditor
can open and read.
"""
from __future__ import annotations

import html as _html
from typing import Any, Dict, List, Optional

from .calibration.realworld import canonical_error_type
from .lineage import lineage
from .result_sensitivity import sensitivity
from .transparency import _pog, per_item_records


def _e(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


_CSS = """
:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;--card:#fff;
--accent:#0f6f6f;--warn:#b45309;--ok:#15803d;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:28px 20px 80px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
h1{font-size:26px;line-height:1.25;margin:.3em 0 .2em;font-weight:600}
.lede{color:var(--mut);max-width:62ch}
.brightline{border-left:3px solid var(--accent);background:#f0f7f7;padding:10px 14px;border-radius:0 8px 8px 0;margin:18px 0;font-size:13.5px}
.bench{margin:30px 0 0}
.bench h2{font-size:19px;margin:0 0 4px;font-weight:600}
.summary{font-size:13px;color:var(--mut);font-family:var(--mono);margin-bottom:10px}
.summary b{color:var(--ink)}
.frag{color:var(--warn);font-weight:600}
.stab{color:var(--ok);font-weight:600}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:10px 0;overflow:hidden}
summary{cursor:pointer;padding:12px 14px;font-family:var(--mono);font-size:13px;list-style:none;display:flex;justify-content:space-between;gap:12px;align-items:center}
summary::-webkit-details-marker{display:none}
summary:hover{background:#f9fafb}
.chg{color:var(--warn);font-weight:600}
.thread{padding:4px 14px 14px}
.step{border-top:1px solid var(--line);padding:11px 0 3px}
.step:first-child{border-top:none}
.sname{font-weight:600;font-size:14px}
.srow{font-size:12.5px;margin:3px 0;color:#374151}
.srow .lab{display:inline-block;width:54px;color:var(--mut);font-family:var(--mono)}
.mono{font-family:var(--mono)}
.f{background:#f6f8f8;border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-family:var(--mono);font-size:12px}
.src{font-size:11.5px;color:var(--mut);margin-top:5px}
table.flip{border-collapse:collapse;font-family:var(--mono);font-size:12px;margin:6px 0 2px}
table.flip td{padding:1px 10px 1px 0}
.yes{color:var(--ok)}.no{color:var(--mut)}
.drop{font-size:12.5px;color:var(--mut);margin:8px 0 0}
.empty{color:var(--mut);font-style:italic;margin:14px 0}
footer{margin-top:40px;font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:14px}
"""


def _flip_table(step: Dict[str, Any]) -> str:
    from .observatory_ui import _short
    per = step.get("data", {}).get("per_model", {})
    rows = []
    for m, mb in per.items():
        o = '<span class="yes">✓</span>' if mb.get("correct_orig") else '<span class="no">·</span>'
        c = '<span class="yes">✓</span>' if mb.get("correct_corr") else '<span class="no">·</span>'
        rows.append("<tr><td>%s</td><td>%s</td><td>@orig %s</td><td>@corr %s</td></tr>"
                    % (_e(_short(m)), _e(mb.get("pred")), o, c))
    return '<table class="flip">%s</table>' % "".join(rows) if rows else ""


def _step_html(step: Dict[str, Any]) -> str:
    parts = ['<div class="step">',
             '<div class="sname">%d · %s</div>' % (step["n"], _e(step["name"])),
             '<div class="srow"><span class="lab">in</span> %s</div>' % _e(step["input"]),
             '<div class="srow"><span class="lab">rule</span> %s</div>' % _e(step["rule"])]
    if step.get("formula"):
        parts.append('<div class="srow"><span class="lab">f</span> <span class="f">%s</span></div>' % _e(step["formula"]))
    parts.append('<div class="srow"><span class="lab">out</span> <span class="mono">%s</span></div>' % _e(step["output"]))
    if step["name"].startswith("Score correctness"):
        parts.append(_flip_table(step))
    parts.append('<div class="src">src · %s</div>' % _e(step["source"]))
    parts.append("</div>")
    return "".join(parts)


def _thread_html(thread) -> str:
    d = thread.to_dict()
    return '<div class="thread">%s</div>' % "".join(_step_html(s) for s in d["steps"])


def _bench_section(bench: str, prows: List[Dict[str, Any]], irows: List[Dict[str, Any]],
                   *, iters: int, seed: int) -> Optional[str]:
    from .observatory_ui import _short
    if not prows:
        return None
    records = per_item_records(bench, prows, irows)
    p, o, c, _ = _pog(prows)
    sens = sensitivity(p, o, c, iters=iters, seed=seed)
    pm = sens["per_model"]
    ro = sorted(pm, key=lambda m: -pm[m]["acc_orig"])
    rc = sorted(pm, key=lambda m: -pm[m]["acc_corr"])
    top_o = _short(ro[0]) if ro else "—"
    top_c = _short(rc[0]) if rc else "—"
    tau = sens.get("kendall_tau")
    fragile = (top_o != top_c) or (tau is not None and tau < 0.99)
    verdict = ('<span class="frag">rank-fragile</span>' if fragile
               else '<span class="stab">rank-stable</span>')

    changed = [r for r in prows
               if set(r.get("original_gold") or []) != set(r.get("corrected_gold") or [])]
    items_html = []
    for r in changed:
        th = lineage(bench, r["item"], prows, irows, sens=sens, records=records)
        og = ",".join(r.get("original_gold") or [])
        cg = ",".join(r.get("corrected_gold") or [])
        flips = next((s.data.get("n_flip", 0) for s in th.steps if s.n == 3), 0)
        summary = ('<span>%s</span><span class="chg">key %s → %s · %d models flip</span>'
                   % (_e(r["item"]), _e(og), _e(cg), flips))
        items_html.append("<details><summary>%s</summary>%s</details>"
                          % (summary, _thread_html(th)))

    n_drop = sum(1 for x in irows if canonical_error_type(x.get("error_type")) == "no_correct_answer")
    drop_html = ('<div class="drop">+ %d item(s) flagged <span class="mono">no_correct_answer</span> '
                 'were dropped by policy — excluded from scoring, never silently scored against a bad key.</div>'
                 % n_drop) if n_drop else ""

    tau_s = ("%.3f" % tau) if tau is not None else "n/a"
    summary_line = ('<div class="summary"><b>%d</b> items corrected · τ = <b>%s</b> · '
                    '#1 %s → <b>%s</b> · P(top-1 changes) = <b>%.2f</b> · %s</div>'
                    % (sens.get("n_changed_items", len(changed)), tau_s, _e(top_o), _e(top_c),
                       sens.get("p_top1_change") or 0.0, verdict))

    body = "".join(items_html) or '<div class="empty">No corrected items in this benchmark.</div>'
    return ('<section class="bench"><h2>%s</h2>%s%s%s</section>'
            % (_e(bench), summary_line, body, drop_html))


def render_lineage_html(pred_rows: Dict[str, List[Dict[str, Any]]],
                        intrinsic_rows: Dict[str, List[Dict[str, Any]]],
                        *, title: str = "Meridian — lineage", iters: int = 2000,
                        seed: int = 0) -> str:
    """Render the full lineage drill-down page from real per-benchmark rows."""
    sections = []
    for bench in sorted(pred_rows):
        s = _bench_section(bench, pred_rows[bench], intrinsic_rows.get(bench, []),
                           iters=iters, seed=seed)
        if s:
            sections.append(s)
    body = "".join(sections) or '<div class="empty">No predictions with corrections to trace yet.</div>'
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>%s</title><style>%s</style></head><body><div class=\"wrap\">"
        "<a href=\"index.html\" style=\"display:inline-block;font-size:13px;color:var(--accent);text-decoration:none;margin-bottom:16px\">\u2190 Meridian observatory</a>"
        "<div class=\"eyebrow\">Per-datum lineage</div>"
        "<h1>The journey of every corrected item, raw to verdict.</h1>"
        "<p class=\"lede\">Each item below had its answer key moved by a human correction. "
        "Expand one to see every transformation — with the rule, the exact formula, and the "
        "scientific source — and the per-model correctness flip it causes.</p>"
        "<div class=\"brightline\">Each number is recomputed from its inputs, and each "
        "transformation shows its rule and its source. The verdict is about the benchmark's "
        "ranking under correction — <b>no model is scored or judged here</b>.</div>"
        "%s"
        "<footer>Generated from the same canonical functions the dossiers use "
        "(<span class=\"mono\">per_item_records</span>, <span class=\"mono\">result_sensitivity</span>, "
        "the declared correction policy). Corrections are candidates, not truths. &nbsp;·&nbsp; "
        "<a href=\"atlas.html\" style=\"color:var(--accent)\">See which benchmarks this reshuffles →</a> "
        "&nbsp;·&nbsp; <a href=\"findings.html\" style=\"color:var(--accent)\">What this means →</a>"
        "</footer>"
        "</div></body></html>"
    ) % (_e(title), _CSS, body)
