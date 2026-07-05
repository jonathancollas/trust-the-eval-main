"""The Findings hub — the two self-correcting results, framed as the core.

This page exists so the science isn't scattered: it states, up front, the two results
the project actually produced, each with numbers recomputed live (never hardcoded) from
``fragility_atlas`` and ``confront``, and links to the full page for each. Both are
honest self-corrections — one against the project's own asserted law, one against a
shared assumption of the field. It audits instruments, not models. No aggregate score.
"""
from __future__ import annotations

import html as _html
from typing import Any, Dict, List

from .atlas import fragility_atlas
from .detection import confront

_CSS = """
:root{--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--bg:#fafafa;--card:#fff;
--accent:#0f6f6f;--warn:#b45309;--ok:#15803d;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:20px 20px 80px}
.back{display:inline-block;font-size:13px;color:var(--accent);text-decoration:none;margin-bottom:16px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
h1{font-size:27px;line-height:1.22;margin:.3em 0 .2em;font-weight:600}
.lede{color:var(--mut);max-width:64ch}
.brightline{border-left:3px solid var(--accent);background:#f0f7f7;padding:10px 14px;border-radius:0 8px 8px 0;margin:16px 0;font-size:13.5px}
.finding{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:18px 0}
.tag{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--warn);font-weight:600}
.finding h2{font-size:19px;margin:4px 0 8px;font-weight:600}
.num{font-family:var(--mono);font-size:14px;background:#f6f8f8;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0;line-height:1.8}
.num b{color:var(--ink)}
.good{color:var(--ok);font-weight:600}.bad{color:var(--warn);font-weight:600}
.means{font-size:14px;margin:8px 0}
.more{font-size:13px;color:var(--accent);text-decoration:none;font-weight:500}
.more:hover{text-decoration:underline}
.foot{margin-top:30px;font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:14px}
"""


def _e(x: Any) -> str:
    return _html.escape("" if x is None else str(x))


def _f(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def _ci(t):
    return "" if not t or t[0] is None else " [%s, %s]" % (_f(t[0], 2), _f(t[1], 2))


def _finding_law(law: Dict[str, Any]) -> str:
    if law.get("insufficient"):
        return ('<div class="finding"><div class="tag">self-correction</div>'
                '<h2>The conditional law — not enough data yet</h2>'
                '<p class="means">%s (n=%d). Add more overlays to test it.</p>'
                '<a class="more" href="atlas.html">Open the fragility atlas →</a></div>'
                % (_e(law.get("note", "insufficient")), law.get("n", 0)))
    cd = law["corr_p_top1_vs_discrimination"]
    cg = law["corr_p_top1_vs_model_gap"]
    holds = law["law_holds"]
    return (
        '<div class="finding"><div class="tag">self-correction · falsifies our own claim</div>'
        '<h2>Correcting label errors moves the ranking — but not for the reason we asserted</h2>'
        '<div class="num">'
        'corr( P(top-1 change) , discrimination ) = <b class="%s">%s</b>%s '
        '<span style="color:var(--mut)">(the asserted law expected &gt; 0)</span><br>'
        'corr( P(top-1 change) , model gap #1–#2 ) = <b>%s</b>%s '
        '<span style="color:var(--mut)">(closeness aggravates, as expected)</span>'
        '</div>'
        '<p class="means">The project asserted a benchmark is fragile when the corrected items '
        '<i>discriminate skill</i> and the models are close. On this corpus the discrimination '
        'term has the <b>opposite sign</b>, with a CI clearing zero: skill-discriminating '
        'corrections <b>reinforce</b> the existing order and are <i>protective</i> of the #1. '
        'The verdict: the law %s.</p>'
        '<p class="means" style="color:var(--mut)">Reconciled: the tool\u2019s severity logic no '
        'longer escalates on skill-discrimination \u2014 it keys only on measured fragility (\u03c4, '
        'p(top-1)). Not reversed either: one corpus does not license the opposite claim.</p>'
        '<a class="more" href="atlas.html">See the atlas, the scatter, and every cell →</a></div>'
    ) % ("bad" if (cd or 0) < 0 else "good", _f(cd), _ci(law.get("ci_discrimination")),
         _f(cg), _ci(law.get("ci_model_gap")),
         ('<span class="bad">does not hold</span>' if not holds else '<span class="good">holds</span>'))


def _finding_detection(det: Dict[str, Any]) -> str:
    m = det["methods"]
    dis = m.get("disagreement", {})
    aw = m.get("ability_weighted", {})
    return (
        '<div class="finding"><div class="tag">self-correction · a trap the field shares</div>'
        '<h2>Behaviour-based error detection looks strong — but the ground truth is circular</h2>'
        '<div class="num">'
        'disagreement detector AUC = <b>%s</b>%s<br>'
        'ability-weighted (IRT intuition) AUC = <b>%s</b>%s '
        '<span style="color:var(--mut)">(no gain over naive)</span>'
        '</div>'
        '<p class="means">Model disagreement scores a high pooled AUC against the MMLU-Redux '
        'labels — but those labels were <b>surfaced from disagreement</b> in the first place, so '
        'the AUC is a <b>circular upper bound</b>, not independent detection. Per subject the '
        'signal is often near chance. And the IRT-style skill-weighting adds nothing. This tempers '
        'both a naive "detection works" and the field\u2019s "IRT detector is great": both measure '
        'against a label seeded by the signal they test.</p>'
        '<a class="more" href="detection.html">See the full confrontation and the caveat →</a></div>'
    ) % (_f(dis.get("auc"), 3), _ci(dis.get("ci")), _f(aw.get("auc"), 3), _ci(aw.get("ci")))


def render_findings_html(pred_rows: Dict[str, List[Dict[str, Any]]],
                         intrinsic_rows: Dict[str, List[Dict[str, Any]]],
                         *, title: str = "Meridian — findings", iters: int = 800,
                         seed: int = 0, boot_iters: int = 1500) -> str:
    cells = {b: (pred_rows[b], intrinsic_rows.get(b, [])) for b in pred_rows}
    atlas = fragility_atlas(cells, iters=iters, seed=seed, boot_iters=boot_iters)
    det = confront(cells, iters=boot_iters, seed=seed, ks=(50, 100))
    n_cells = atlas["n_active"]
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>%s</title><style>%s</style></head><body><div class=\"wrap\">"
        "<a class=\"back\" href=\"index.html\">\u2190 Meridian observatory</a>"
        "<div class=\"eyebrow\">Findings</div>"
        "<h1>Two results — and both correct a claim, ours or the field\u2019s.</h1>"
        "<p class=\"lede\">The value of a validity tool is turning on its own premises. These are "
        "the two the observatory produced; the numbers below are recomputed from the same "
        "canonical functions the pages use, not restated.</p>"
        "<div class=\"brightline\">Both hold on <b>one corpus</b> so far (%d cells with "
        "corrections). They become results when they replicate on a second overlay. No model is "
        "scored; the subject under audit is always the instrument.</div>"
        "%s%s"
        "<div class=\"foot\">Each finding links to its full page, where every number recomputes "
        "from its inputs. The lineage view shows one datum\u2019s journey the whole chain rests on: "
        "<a class=\"more\" href=\"lineage.html\">open the per-datum lineage →</a></div>"
        "</div></body></html>"
    ) % (_e(title), _CSS, n_cells, _finding_law(atlas["law"]), _finding_detection(det))
