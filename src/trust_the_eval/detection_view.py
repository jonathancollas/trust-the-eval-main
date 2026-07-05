"""The detection-confrontation view for the observatory.

Renders, as a self-contained static page, the label-error detector comparison from the
real ``confront`` — the AUC table with bootstrap CIs and precision@k, and, front and
centre, the circularity caveat: MMLU-Redux surfaced its candidates from model
disagreement, so a disagreement detector's AUC against those labels is a circular upper
bound. It audits the instrument, not models. No aggregate score, no JavaScript.
"""
from __future__ import annotations

import html as _html
from typing import Any, Dict, List

from .detection import confront

_CSS = """
:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;--card:#fff;
--accent:#0f6f6f;--warn:#b45309;--ok:#15803d;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:20px 20px 80px}
.back{display:inline-block;font-size:13px;color:var(--accent);text-decoration:none;margin-bottom:16px}
.back:hover{text-decoration:underline}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
h1{font-size:26px;line-height:1.25;margin:.3em 0 .2em;font-weight:600}
.lede{color:var(--mut);max-width:64ch}
.caveat{border-left:4px solid var(--warn);background:#fffbeb;padding:12px 16px;border-radius:0 8px 8px 0;margin:18px 0;font-size:14px}
.caveat b{color:var(--warn)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:18px 0}
h2{font-size:18px;margin:0 0 4px;font-weight:600}
.sub{color:var(--mut);font-size:13px;font-family:var(--mono);margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:500}
.bar{display:inline-block;height:8px;background:var(--accent);border-radius:2px;vertical-align:middle;margin-left:6px}
.chance{color:var(--mut)}
.findings{font-size:14px;line-height:1.6}
.findings li{margin:4px 0}
.foot{margin-top:32px;font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:14px}
"""


def _e(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


def _f(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def _table(res: Dict[str, Any]) -> str:
    ks = res["ks"]
    head = ("<tr><th>detector</th><th>AUC</th><th>95% CI</th>"
            + "".join("<th>P@%d</th>" % k for k in ks) + "</tr>")
    body = ""
    for name in res["ranked"]:
        m = res["methods"][name]
        auc = m["auc"] or 0.0
        ci = m["ci"]
        ci_s = "n/a" if not ci or ci[0] is None else "[%s, %s]" % (_f(ci[0], 2), _f(ci[1], 2))
        w = int(70 * max(0.0, (auc - 0.5) / 0.5))       # bar from chance (0.5) to 1.0
        pk = "".join("<td>%s</td>" % _f(m["prec_at_k"].get(k), 2) for k in ks)
        body += ("<tr><td>%s</td><td>%s<span class=\"bar\" style=\"width:%dpx\"></span></td>"
                 "<td>%s</td>%s</tr>") % (_e(name), _f(auc, 3), w, ci_s, pk)
    return "<table><thead>%s</thead><tbody>%s</tbody></table>" % (head, body)


def render_detection_html(pred_rows: Dict[str, List[Dict[str, Any]]],
                          intrinsic_rows: Dict[str, List[Dict[str, Any]]],
                          *, title: str = "Meridian — detection confrontation",
                          iters: int = 1500, seed: int = 0) -> str:
    cells = {b: (pred_rows[b], intrinsic_rows.get(b, [])) for b in pred_rows}
    res = confront(cells, iters=iters, seed=seed, ks=(50, 100, 200))
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>%s</title><style>%s</style></head><body><div class=\"wrap\">"
        "<a class=\"back\" href=\"index.html\">\u2190 Meridian observatory</a>"
        "<div class=\"eyebrow\">Detection confrontation</div>"
        "<h1>Can model behaviour find the label errors — and which method?</h1>"
        "<p class=\"lede\">Each detector scores every item for \u201clikely a wrong key\u201d; "
        "we rank them by ROC-AUC against the MMLU-Redux ground truth, with bootstrap CIs "
        "and precision@k.</p>"
        "<div class=\"caveat\"><b>Read this before the numbers.</b> MMLU-Redux surfaced its "
        "candidate errors by inspecting items where models disagreed with the key. So a "
        "disagreement detector scoring high AUC against those labels is <b>circular</b> \u2014 it "
        "partly measures the annotation pipeline, not independent detection. Treat the AUC as "
        "an <b>upper bound</b>, and weigh the per-subject spread (often near chance) over the "
        "pooled figure. No model is scored here; the label under test is the benchmark's key.</div>"
        "<div class=\"card\"><h2>Detectors, ranked</h2>"
        "<div class=\"sub\">positives = %s &middot; %d defects / %d ok &middot; base rate %s</div>"
        "%s<p class=\"chance\" style=\"font-size:12px;margin-top:10px\">AUC 0.5 = chance; bars show "
        "the margin above chance.</p></div>"
        "<div class=\"card\"><h2>What holds, once you account for circularity</h2>"
        "<ul class=\"findings\"><li><b>Ability-weighting adds nothing</b> \u2014 weighting "
        "disagreement by model skill (the IRT mislabel intuition) does not beat the naive "
        "detector here.</li><li><b>Answer entropy is a poor mislabel detector</b> \u2014 it flags "
        "ambiguity, a different construct.</li><li><b>The signal is corpus- and slice-dependent</b> "
        "\u2014 pooled AUC is high, per-subject it is often near chance.</li></ul></div>"
        "<div class=\"foot\">Scored by the canonical detection bench (ROC-AUC via Mann-Whitney, "
        "bootstrap CIs). A statistical screen is a candidate generator, never a substitute for "
        "verification.</div>"
        "</div></body></html>"
    ) % (_e(title), _CSS, ", ".join(res["positive"]), res["n_pos"], res["n_neg"],
         _f(res["base_rate"], 3), _table(res))
