"""The published calibration report — a self-contained, offline HTML artifact.

This is the "voici la preuve mesurée" deliverable: the thing a third party
(auditor, model selector, journalist) can open, read, and cite. It folds
together everything the calibration engine measures:

  * headline pooled precision / recall / specificity with Wilson intervals,
  * a transparent METHODS section (how a probe verdict is scored against ground
    truth) and the THREE evidence tiers,
  * an honest REAL-DATA COVERAGE table (which probes are validated on real human
    labels, which are exact-by-construction, which are synthetic-floor and why),
  * per-probe cards: confusion matrix, P/R/specificity (+CI), ROC & PR curves
    rendered inline as SVG, per-regime specificity, score orientation, and a
    plain "when not to trust this probe" note,
  * a global LIMITATIONS section.

No external assets, no JavaScript, no web fonts — it renders anywhere and embeds
in a repo, a model card, or an email. Bright line preserved throughout: this
measures the validity of an eval result, never model safety.
"""
from __future__ import annotations

import html as _html
from typing import Optional

from .coverage import (PROBE_EVIDENCE, TIER_BEHAVIORAL, TIER_LABEL, TIER_RANK,
                       TIER_REAL, TIER_STRUCTURAL, coverage_summary, evidence_for)
from .curves_svg import pr_svg, roc_svg
from .runner import CalibrationReport

_TIER_COLOR = {
    TIER_REAL: "#15803d",
    TIER_STRUCTURAL: "#1d4ed8",
    TIER_BEHAVIORAL: "#b45309",
}
_TIER_SHORT = {
    TIER_REAL: "real labels",
    TIER_STRUCTURAL: "exact",
    TIER_BEHAVIORAL: "synthetic floor",
}


def _e(s) -> str:
    return _html.escape(str(s))


def _metric_cell(m: dict) -> str:
    """Render a metric dict {value,lo,hi,k,n} as 'value [lo, hi] (k/n)'."""
    v = m.get("value")
    if v is None:
        return '<span class="na">n/a</span>'
    lo, hi, k, n = m.get("lo"), m.get("hi"), m.get("k"), m.get("n")
    ci = f' <span class="ci">[{lo:.2f}, {hi:.2f}]</span>' if lo is not None else ""
    kn = f' <span class="kn">({k}/{n})</span>' if n is not None else ""
    return f'<b>{v:.2f}</b>{ci}{kn}'


def _orientation_note(orientation: str) -> str:
    return {
        "direct": "score increases with defect likelihood",
        "inverted": "score DECREASES with defect (e.g. determinism rate); "
                    "AUC computed on the oriented score",
        "non_monotonic": "score is U-shaped in the defect (both extremes are "
                         "defects); AUC computed on distance-from-clean",
        "degenerate": "only one class present; curve undefined",
    }.get(orientation, orientation)


def _probe_card(cal_dict: dict, caveat: Optional[str]) -> str:
    pid = cal_dict["probe_id"]
    name = cal_dict["probe_name"]
    kind = "model-in-the-loop" if cal_dict["requires_model"] else "static"
    m = cal_dict["matrix"]
    tier, source, basis = evidence_for(pid)
    tcol = _TIER_COLOR[tier]

    # curves: rebuild SVG from the stored curve dict via a light shim
    from .core import CurvePoint, Curves
    cd = cal_dict["curves"]
    pts = [CurvePoint(p["threshold"], p["tpr"], p["fpr"], 0.0, p["tpr"])
           for p in cd.get("roc", [])]
    # attach precision from pr list (same index order is not guaranteed; recompute panel from full curve)
    curves = Curves(points=[], roc_auc=cd.get("roc_auc"),
                    average_precision=cd.get("average_precision"),
                    prevalence=cd.get("prevalence"),
                    orientation=cd.get("orientation", "direct"))
    # reconstruct full points (roc + pr share thresholds)
    roc = {round(p["threshold"], 6): p for p in cd.get("roc", [])}
    full = []
    for p in cd.get("pr", []):
        r = roc.get(round(p["threshold"], 6))
        if r is not None:
            full.append(CurvePoint(p["threshold"], r["tpr"], r["fpr"],
                                   p["precision"], p["recall"]))
    curves.points = full
    curves_html = (f'<div class="curves">{roc_svg(curves)}{pr_svg(curves)}</div>'
                   if full else
                   '<div class="curve-empty">Curve undefined (single class in set).</div>')

    # per-regime specificity rows
    reg_rows = ""
    for reg, d in (cal_dict.get("regime_specificity") or {}).items():
        sp = d.get("specificity", {})
        reg_rows += (f'<tr><td class="reg">{_e(reg)}</td>'
                     f'<td>{_metric_cell(sp)}</td></tr>')
    reg_block = (f'<table class="reg-table"><thead><tr><th>clean regime</th>'
                 f'<th>specificity</th></tr></thead><tbody>{reg_rows}</tbody></table>'
                 if reg_rows else
                 '<p class="muted">no tagged clean regimes</p>')

    caveat_html = (f'<div class="caveat"><span class="caveat-h">When not to trust '
                   f'this probe</span><p>{_e(caveat)}</p></div>') if caveat else ""

    return f"""
<section class="card" id="probe-{_e(pid)}">
  <div class="card-head">
    <h3>{_e(pid)} <span class="kind">{_e(kind)}</span></h3>
    <span class="tier" style="background:{tcol}">{_TIER_SHORT[tier]}</span>
  </div>
  <p class="pname">{_e(name)}</p>
  <p class="basis"><b>Ground-truth basis:</b> {_e(basis)}
     {('<br><b>Real source:</b> <code>' + _e(source) + '</code>') if source else ''}</p>
  <table class="metrics">
    <tbody>
      <tr><td>precision</td><td>{_metric_cell(m['precision'])}</td>
          <td>recall</td><td>{_metric_cell(m['recall'])}</td></tr>
      <tr><td>specificity</td><td>{_metric_cell(m['specificity'])}</td>
          <td>F1</td><td><b>{(m['f1'] if m['f1'] is not None else 0):.2f}</b></td></tr>
      <tr><td>ROC-AUC</td><td><b>{_fmt(cd.get('roc_auc'))}</b></td>
          <td>avg precision</td><td><b>{_fmt(cd.get('average_precision'))}</b></td></tr>
      <tr><td>confusion</td><td colspan="3">TP={m['tp']} · FP={m['fp']} · FN={m['fn']} · TN={m['tn']}
          &nbsp;<span class="muted">(score: {_e(_orientation_note(cd.get('orientation','direct')))})</span></td></tr>
    </tbody>
  </table>
  {curves_html}
  <div class="reg">{reg_block}</div>
  {caveat_html}
</section>"""


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def render_report_html(report: CalibrationReport,
                       caveats: Optional[dict] = None) -> str:
    """Render the full self-contained calibration report as an HTML string.

    ``caveats`` optionally maps probe_id -> a one-line "when not to trust" note
    (the report pulls these from each probe's ProbeDoc when available).
    """
    caveats = caveats or {}
    p = report.pooled
    cov = coverage_summary()
    n_real = cov["counts"].get(TIER_REAL, 0)
    n_struct = cov["counts"].get(TIER_STRUCTURAL, 0)
    n_behav = cov["counts"].get(TIER_BEHAVIORAL, 0)

    # coverage table rows, grouped by tier
    cov_rows = ""
    for pid in sorted(PROBE_EVIDENCE, key=lambda x: (TIER_RANK[PROBE_EVIDENCE[x][0]], x)):
        tier, source, basis = PROBE_EVIDENCE[pid]
        cov_rows += (
            f'<tr><td><code>{_e(pid)}</code></td>'
            f'<td><span class="tier-dot" style="background:{_TIER_COLOR[tier]}"></span>'
            f'{_e(TIER_LABEL[tier])}</td>'
            f'<td>{_e(source) if source else "&mdash;"}</td>'
            f'<td class="basis-cell">{_e(basis)}</td></tr>')

    # per-probe cards sorted by tier then id
    cards = ""
    for cal in sorted(report.probes,
                      key=lambda c: (TIER_RANK[evidence_for(c.probe_id)[0]], c.probe_id)):
        cards += _probe_card(cal.to_dict(), caveats.get(cal.probe_id))

    macro = p.get("macro_roc_auc")
    macro_s = f"{macro:.3f}" if isinstance(macro, (int, float)) else "n/a"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trust the Eval — Probe Calibration Report</title>
<style>
  :root {{ --ink:#18181b; --muted:#71717a; --line:#e4e4e7; --bg:#fafafa;
           --card:#ffffff; --accent:#4338ca; }}
  * {{ box-sizing:border-box; }}
  body {{ font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
          color:var(--ink); background:var(--bg); margin:0; padding:0 0 4rem; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:0 20px; }}
  header.hero {{ background:linear-gradient(180deg,#1e1b4b,#312e81); color:#fff;
                 padding:38px 0 30px; margin-bottom:28px; }}
  header.hero .wrap {{ }}
  header.hero h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:-.3px; }}
  header.hero p {{ margin:2px 0; opacity:.85; font-size:13.5px; }}
  .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:20px; }}
  .kpi {{ background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.16);
          border-radius:10px; padding:12px 16px; min-width:150px; }}
  .kpi .v {{ font-size:24px; font-weight:700; }}
  .kpi .l {{ font-size:11.5px; opacity:.8; text-transform:uppercase; letter-spacing:.4px; }}
  .kpi .ci {{ font-size:11px; opacity:.7; font-weight:400; }}
  h2 {{ font-size:19px; margin:34px 0 10px; padding-bottom:6px;
        border-bottom:2px solid var(--line); letter-spacing:-.2px; }}
  h3 {{ font-size:16px; margin:0; }}
  p.lead {{ color:#3f3f46; }}
  code {{ font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;
          background:#f4f4f5; padding:1px 5px; border-radius:4px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
            vertical-align:top; }}
  th {{ background:#f4f4f5; font-weight:600; font-size:12px; text-transform:uppercase;
        letter-spacing:.3px; color:#52525b; }}
  .tier-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
               margin-right:7px; vertical-align:middle; }}
  .basis-cell {{ color:var(--muted); font-size:12.5px; }}
  .legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:8px 0 4px; font-size:13px; }}
  .legend span b {{ font-weight:600; }}
  .tiers {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin:14px 0; }}
  .tierbox {{ border:1px solid var(--line); border-radius:10px; padding:14px;
              background:var(--card); }}
  .tierbox h4 {{ margin:0 0 6px; font-size:13.5px; }}
  .tierbox .cnt {{ font-size:22px; font-weight:700; }}
  .tierbox p {{ font-size:12px; color:var(--muted); margin:6px 0 0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
           padding:18px; margin:14px 0; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; }}
  .card .kind {{ font-size:11px; font-weight:500; color:var(--muted); margin-left:8px; }}
  .pname {{ color:#3f3f46; margin:4px 0 8px; font-size:13.5px; }}
  .basis {{ font-size:12.5px; color:#52525b; margin:0 0 10px; }}
  .tier {{ color:#fff; font-size:11px; font-weight:600; padding:3px 9px;
           border-radius:20px; white-space:nowrap; }}
  table.metrics td {{ border-bottom:1px solid #f4f4f5; font-size:13px; padding:5px 8px; }}
  table.metrics td:nth-child(odd) {{ color:var(--muted); width:90px; }}
  .ci {{ color:var(--muted); font-size:11.5px; }}
  .kn {{ color:#a1a1aa; font-size:11px; }}
  .na {{ color:#a1a1aa; }}
  .curves {{ display:flex; gap:14px; flex-wrap:wrap; margin:12px 0; }}
  .curve-empty {{ color:var(--muted); font-size:12.5px; padding:10px 0; }}
  .reg-table {{ width:auto; min-width:320px; margin-top:4px; }}
  .reg-table td.reg {{ font-family:ui-monospace,monospace; font-size:12px; }}
  .caveat {{ background:#fffbeb; border:1px solid #fde68a; border-radius:9px;
             padding:10px 13px; margin-top:12px; }}
  .caveat-h {{ font-size:11.5px; font-weight:700; color:#92400e;
               text-transform:uppercase; letter-spacing:.4px; }}
  .caveat p {{ margin:5px 0 0; font-size:12.5px; color:#78350f; }}
  .muted {{ color:var(--muted); }}
  .brightline {{ background:#eef2ff; border:1px solid #c7d2fe; border-radius:10px;
                 padding:13px 16px; margin:18px 0; font-size:13px; color:#3730a3; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:34px;
            border-top:1px solid var(--line); padding-top:14px; }}
</style></head>
<body>
<header class="hero"><div class="wrap">
  <h1>Trust the Eval — Probe Calibration Report</h1>
  <p>Measured precision &amp; recall of each validity probe against known ground truth.</p>
  <p>generated {_e(report.generated_at)} · seed {report.seed} · {len(report.probes)} probes</p>
  <div class="kpis">
    <div class="kpi"><div class="v">{_pct(p['precision'])}</div>
      <div class="l">pooled precision</div><div class="ci">{_ci(p['precision'])}</div></div>
    <div class="kpi"><div class="v">{_pct(p['recall'])}</div>
      <div class="l">pooled recall</div><div class="ci">{_ci(p['recall'])}</div></div>
    <div class="kpi"><div class="v">{_pct(p['specificity'])}</div>
      <div class="l">pooled specificity</div><div class="ci">{_ci(p['specificity'])}</div></div>
    <div class="kpi"><div class="v">{p['probes_perfect']}/{p['probes_total']}</div>
      <div class="l">probes perfect</div><div class="ci">R = spec = 1.0</div></div>
    <div class="kpi"><div class="v">{macro_s}</div>
      <div class="l">macro ROC-AUC</div><div class="ci">mean over probes</div></div>
  </div>
</div></header>

<div class="wrap">
  <p class="lead">Each probe is a diagnostic test: its verdict (defect present /
  absent) is confronted with a <b>known ground truth</b>, yielding a confusion
  matrix and the diagnostic-test rates this tool itself preaches — precision,
  recall, specificity, each with a Wilson 95% interval — plus a threshold sweep
  giving ROC and precision-recall curves. The honest question this report
  answers is the one a sceptic should ask first: <i>why trust the
  validity-checker?</i></p>

  <div class="brightline"><b>Scope.</b> Every probe and every calibration case
  assesses the <b>validity of an eval result</b> as a measurement — never the
  safety of a model. Reward-hacking / gaming probes are detection-only.</div>

  <h2>Method &amp; evidence tiers</h2>
  <p>A probe "flags a defect" when its top finding reaches the operating
  severity (or, for probes whose severity is constant but whose numeric score
  discriminates, when the score crosses a documented threshold). INFO findings
  never count as a positive. Ground truth comes in three tiers, and we are
  explicit about which applies to each probe:</p>
  <div class="tiers">
    <div class="tierbox"><h4 style="color:{_TIER_COLOR[TIER_REAL]}">Tier 1 · real labels</h4>
      <div class="cnt" style="color:{_TIER_COLOR[TIER_REAL]}">{n_real}</div>
      <p>A public dataset labels, per item, whether the defect is present. We
      measure against those human labels. Strongest evidence.</p></div>
    <div class="tierbox"><h4 style="color:{_TIER_COLOR[TIER_STRUCTURAL]}">Tier 2 · exact</h4>
      <div class="cnt" style="color:{_TIER_COLOR[TIER_STRUCTURAL]}">{n_struct}</div>
      <p>The defect is a mathematical property of the artifact (sample size,
      config count, score distribution). Synthetic construction <i>is</i> the
      ground truth — exact, not a proxy.</p></div>
    <div class="tierbox"><h4 style="color:{_TIER_COLOR[TIER_BEHAVIORAL]}">Tier 3 · synthetic floor</h4>
      <div class="cnt" style="color:{_TIER_COLOR[TIER_BEHAVIORAL]}">{n_behav}</div>
      <p>Behavioural defects with no public per-item ground truth. Validated
      against controllable models defective by construction; real-world
      validation pending datasets that mostly do not yet exist.</p></div>
  </div>

  <h2>Real-data coverage</h2>
  <p>Where real human-labelled ground truth exists, we use it. Where the defect
  is structural, the truth is exact by construction. Where it is behavioural and
  unlabelled in public data, we say so plainly rather than overclaim.</p>
  <table><thead><tr><th>probe</th><th>evidence tier</th><th>real source</th>
    <th>ground-truth basis</th></tr></thead>
    <tbody>{cov_rows}</tbody></table>

  <h2>Per-probe calibration</h2>
  <p class="muted">Sorted by evidence tier. Curves render inline (offline SVG).
  Each card shows the confusion matrix, diagnostic rates with Wilson intervals,
  per-regime specificity, the score orientation, and when not to trust it.</p>
  {cards}

  <h2>Limitations — read before citing</h2>
  <ul>
    <li><b>Synthetic floor is internal validity, not external.</b> Tier-3 probes
    are validated against models that exhibit the defect by construction; this
    proves the probe fires on a real instance of the defect, not that its
    real-world precision matches these numbers. External validation awaits
    labelled datasets that, for contamination and most behavioural defects, do
    not publicly exist.</li>
    <li><b>Small-n intervals are wide.</b> Real-data groups (MMLU-Redux) can be
    few; the Wilson intervals reflect that honestly. More groups tighten them.
    A point estimate of 1.00 with a wide interval is not a claim of perfection.</li>
    <li><b>Verdict thresholds are conventions.</b> Severity cut-offs and score
    thresholds are documented rules of thumb, not calibrated decision boundaries;
    a deployment may tune them, which shifts the precision/recall trade-off.</li>
    <li><b>The probes' own caveats still apply.</b> Calibration measures
    detection of a planted/labelled defect; the confounds documented in each
    probe's science notes (e.g. paraphrase difficulty, weak-vs-ambiguous) remain
    real on messy field data.</li>
  </ul>

  <footer>
    Trust the Eval · probe-calibration report · reproducible with
    <code>trust-the-eval calibrate</code> (add <code>--mmlu-redux &lt;path&gt;</code>
    for the real-data acid test). This artifact is self-contained and offline.
  </footer>
</div>
</body></html>"""


def _pct(m: dict) -> str:
    v = m.get("value")
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def _ci(m: dict) -> str:
    lo, hi = m.get("lo"), m.get("hi")
    if lo is None:
        return ""
    return f"95% CI [{lo*100:.0f}, {hi*100:.0f}]"
