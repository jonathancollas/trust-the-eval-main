from __future__ import annotations
import html as _html

from ..finding import Severity
from ..runner import Report

_COLOR = {Severity.HIGH: "#c0392b", Severity.MEDIUM: "#e67e22",
          Severity.LOW: "#27ae60", Severity.INFO: "#7f8c8d"}


def to_html(report: Report) -> str:
    a = report.artifact
    rows = []
    for f in report.findings:
        col = _COLOR[f.severity]
        sc = "" if f.score is None else f"<td>{f.score:.2f}</td>"
        if f.score is None:
            sc = "<td>-</td>"
        rows.append(
            f"<tr><td><b>{_html.escape(f.probe_id)}</b></td>"
            f"<td style='color:{col}'>{f.severity.value}</td>{sc}"
            f"<td>{_html.escape(f.summary)}</td></tr>")
    skipped = "".join(f"<li>{_html.escape(s)}</li>" for s in report.skipped)
    cost = (f"<p>Cost: {report.cost['calls']} calls, ~{report.cost['est_usd']} USD</p>"
            if report.cost else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Trust the Eval - {_html.escape(a.dataset)}</title>
<style>body{{font:14px system-ui;margin:2rem;max-width:60rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}}
th{{background:#f5f5f5}}</style></head><body>
<h1>Trust the Eval</h1>
<p><b>{_html.escape(a.dataset)}</b> — {a.n} items — <code>{a.content_hash()[:23]}…</code></p>
<table><thead><tr><th>probe</th><th>severity</th><th>score</th><th>finding</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{cost}
<h3>Skipped (need model access)</h3><ul>{skipped or '<li>none</li>'}</ul>
<p style="color:#7f8c8d">Trust the Eval attacks the validity of an evaluation result, not the safety of a model.</p>
</body></html>"""
