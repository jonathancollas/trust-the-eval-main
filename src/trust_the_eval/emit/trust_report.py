"""Run Trust Report — a self-contained, provenance-stamped HTML artifact.

The document you paste into a model card or a procurement file: what was
audited, what the score can and cannot support, every finding with the
flagging probe's MEASURED reliability, and how to verify the report refers
to the same data.

Honesty rules baked in:
- the integrity stamp is the SHA-256 content hash of the audited artifact —
  a provenance stamp, NOT a cryptographic signature of the report;
- per-claim verdicts use the shared rule in trust_the_eval.verdict (humble:
  'supportable' = not undermined by the threats we test, never "proven");
- every probe's reliability numbers come from the calibration engine, with
  its evidence tier named; synthetic-floor probes are flagged as such;
- zero scripts, zero external resources: openable offline, archivable.
"""
from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Optional

from ..taxonomy import CLAIM_LABEL, CLAIM_TYPES, REMEDIATION
from ..verdict import SEV_RANK, claim_verdicts


def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


# --------------------------- calibration index ---------------------------

_CAL_INDEX: dict[int, dict] = {}


def calibration_index(seed: int = 0) -> dict[str, dict]:
    """probe_id -> {tier, precision, recall, specificity} (cached)."""
    if seed in _CAL_INDEX:
        return _CAL_INDEX[seed]
    from ..calibration import run_calibration
    from ..calibration.coverage import evidence_for
    rep = run_calibration(seed=seed)
    idx: dict[str, dict] = {}
    for cal in rep.probes:
        tier, _, _ = evidence_for(cal.probe_id)
        m = cal.matrix
        idx[cal.probe_id] = {
            "tier": tier,
            "precision": m.precision().value,
            "recall": m.recall().value,
            "specificity": m.specificity().value,
        }
    _CAL_INDEX[seed] = idx
    return idx


_TIER_WORD = {"real_labeled": "real labels", "structural_exact": "exact",
              "behavioral_synthetic": "synthetic floor"}
_TIER_COLOR = {"real_labeled": "#15803d", "structural_exact": "#1d4ed8",
               "behavioral_synthetic": "#b45309"}
_SEV_COLOR = {"high": "#dc2626", "medium": "#c2620a", "low": "#15803d", "info": "#52525b"}
_V_BG = {"sup": "#dcfce7", "fra": "#fef3c7", "uns": "#fee2e2"}
_V_FG = {"sup": "#15803d", "fra": "#92400e", "uns": "#b91c1c"}


def _pct(v) -> str:
    return f"{v*100:.0f}%" if v is not None else "\u2014"


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("trust-the-eval")
    except Exception:
        return "dev"


def _finding_dict(f) -> dict:
    sev = getattr(f, "severity", None)
    return {"probe_id": f.probe_id, "severity": getattr(sev, "value", str(sev)),
            "summary": f.summary, "score": getattr(f, "score", None)}


# --------------------------- renderer ---------------------------

def render_trust_report(report, artifact, model_spec: str = "none",
                        calibration: Optional[dict] = None,
                        seed: int = 0) -> str:
    """Render the Run Trust Report. `report` is a runner Report, `artifact`
    the audited EvalArtifact; `calibration` an optional precomputed
    calibration_index() (else computed, cached)."""
    cal = calibration if calibration is not None else calibration_index(seed)
    findings = [_finding_dict(f) for f in report.findings]
    verdicts = claim_verdicts(findings)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    chash = artifact.content_hash()
    ver = _version()

    def rel(pid: str) -> str:
        c = cal.get(pid)
        if not c:
            return '<span class="mut">reliability: not calibrated</span>'
        warn = (' <span class="warn">\u26a0 synthetic floor \u2014 real-world precision unproven</span>'
                if c["tier"] == "behavioral_synthetic" else "")
        return (f'recall <b>{_pct(c["recall"])}</b> \u00b7 specificity <b>{_pct(c["specificity"])}</b> '
                f'\u00b7 precision <b>{_pct(c["precision"])}</b> '
                f'<span class="tier" style="background:{_TIER_COLOR[c["tier"]]}">{_esc(_TIER_WORD[c["tier"]])}</span>{warn}')

    # ---- per-claim verdict cards
    vcards = []
    for claim in CLAIM_TYPES:
        v = verdicts[claim]
        th = "".join(
            f'<div class="th"><span class="sev" style="background:{_SEV_COLOR[t["severity"]]}">{_esc(t["severity"])}</span> '
            f'<code>{_esc(t["probe_id"])}</code> {_esc(t["summary"][:140])}</div>'
            for t in v["threats"][:3])
        more = (f'<div class="mut">+{len(v["threats"])-3} more</div>' if len(v["threats"]) > 3 else "")
        ok = '<div class="ok">\u2713 no medium+ validity threat among the relevant probes</div>' if not v["threats"] else ""
        vcards.append(
            f'<div class="vcard"><div class="vh">{_esc(CLAIM_LABEL[claim])} claim '
            f'<span class="mut">({v["n_probes"]} relevant findings)</span></div>'
            f'<span class="vb" style="background:{_V_BG[v["verdict"]]};color:{_V_FG[v["verdict"]]}">{_esc(v["word"])}</span>'
            f'<div class="vm">{_esc(v["meaning"])}</div>{th}{more}{ok}</div>')

    # ---- dominant threats (medium+, ranked by severity then probe id)
    dom = sorted([f for f in findings if SEV_RANK.get(f["severity"], 0) >= 2],
                 key=lambda f: (-SEV_RANK.get(f["severity"], 0), f["probe_id"]))[:5]
    dom_html = "".join(
        f'<div class="th"><span class="sev" style="background:{_SEV_COLOR[f["severity"]]}">{_esc(f["severity"])}</span> '
        f'<code>{_esc(f["probe_id"])}</code> {_esc(f["summary"])}<div class="rel">{rel(f["probe_id"])}</div></div>'
        for f in dom) or '<div class="ok">\u2713 no medium- or high-severity validity threat on this run</div>'

    # ---- full findings table
    rows = []
    for f in sorted(findings, key=lambda f: (-SEV_RANK.get(f["severity"], 0), f["probe_id"])):
        rem = REMEDIATION.get(f["probe_id"], "")
        rows.append(
            f'<div class="frow"><div class="fh">'
            f'<span class="sev" style="background:{_SEV_COLOR[f["severity"]]}">{_esc(f["severity"])}</span> '
            f'<code>{_esc(f["probe_id"])}</code>'
            + (f'<span class="score">{f["score"]:.2f}</span>' if f.get("score") is not None else "")
            + f'</div><div class="fs">{_esc(f["summary"])}</div>'
            f'<div class="rel">{rel(f["probe_id"])}</div>'
            + (f'<div class="rem"><b>what to do</b> {_esc(rem)}</div>' if rem else "")
            + '</div>')

    skipped = "".join(f'<code>{_esc(s)}</code> ' for s in (report.skipped or []))
    errors = "".join(f'<div class="err"><code>{_esc(k)}</code>: {_esc(v)}</div>'
                     for k, v in (report.errors or {}).items())
    cost = report.cost or {}
    cost_line = (f'{cost.get("calls", 0)} model calls \u00b7 ~${cost.get("est_usd", 0)}'
                 if cost.get("calls") is not None else "static probes only")
    n_probes_fired = len({f["probe_id"] for f in findings})

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Run Trust Report \u2014 {_esc(artifact.dataset)}</title>
<style>
body{{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;color:#18181b;margin:0;background:#fafafa}}
.page{{max-width:880px;margin:0 auto;padding:28px 22px 60px}}
h1{{font-size:21px;margin:0 0 2px}} h2{{font-size:15px;margin:26px 0 10px;border-bottom:1px solid #e4e4e7;padding-bottom:5px}}
.sub{{color:#71717a;font-size:12.5px;margin:0 0 18px}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
.prov{{background:#fff;border:1px solid #e4e4e7;border-radius:10px;padding:14px 16px;font-size:12.5px}}
.prov .row{{display:flex;gap:10px;margin:3px 0}} .prov .k{{color:#71717a;min-width:130px}}
.stamp{{background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:9px 12px;margin-top:10px;font-size:12px;color:#3730a3}}
.frame{{border-left:3px solid #d4d4d8;padding-left:10px;color:#52525b;font-size:12.5px;margin:0 0 14px}}
.vgrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:10px}}
.vcard{{background:#fff;border:1px solid #e4e4e7;border-radius:9px;padding:11px 12px}}
.vh{{font-size:12.5px;font-weight:600}} .vb{{display:inline-block;font-size:10.5px;font-weight:700;padding:2px 9px;border-radius:20px;margin:6px 0;text-transform:uppercase;letter-spacing:.3px}}
.vm{{font-size:11.5px;color:#52525b;margin:0 0 6px}}
.sev{{display:inline-block;color:#fff;font-size:9.5px;font-weight:700;padding:1px 7px;border-radius:20px;text-transform:uppercase}}
.tier{{display:inline-block;color:#fff;font-size:9.5px;font-weight:700;padding:1px 7px;border-radius:20px}}
.th{{font-size:12px;margin:5px 0;line-height:1.45}} .ok{{font-size:12px;color:#15803d}}
.mut{{color:#71717a;font-size:11px}} .warn{{color:#92400e;font-size:10.5px}}
.frow{{background:#fff;border:1px solid #e4e4e7;border-radius:9px;padding:11px 13px;margin:0 0 9px}}
.fh{{display:flex;align-items:center;gap:8px}} .fs{{font-size:12.5px;margin:5px 0 0}}
.score{{margin-left:auto;font-family:ui-monospace,monospace;font-size:12px;color:#52525b}}
.rel{{font-size:11px;color:#52525b;margin-top:6px;background:#f8f8f8;border:1px solid #ececf0;border-radius:6px;padding:5px 8px}}
.rem{{font-size:11.5px;color:#3f3f46;margin-top:6px}} .rem b{{color:#0e7490;font-size:10px;text-transform:uppercase;letter-spacing:.3px}}
.err{{font-size:12px;color:#b91c1c;margin:3px 0}}
.lim{{font-size:12px;color:#52525b;line-height:1.6}}
.foot{{margin-top:26px;color:#71717a;font-size:11.5px;border-top:1px solid #e4e4e7;padding-top:10px}}
</style></head><body><div class="page">
<h1>Run Trust Report</h1>
<p class="sub">Trust the Eval v{_esc(ver)} \u00b7 generated {_esc(now)} \u00b7 validity audit of an eval <i>result</i> \u2014 this report assesses the score as a measurement; it does not certify the model, and it never tests model safety.</p>

<h2>What was audited</h2>
<div class="prov">
  <div class="row"><span class="k">dataset</span><b>{_esc(artifact.dataset)}</b></div>
  <div class="row"><span class="k">items</span>{artifact.n}</div>
  <div class="row"><span class="k">artifact model</span>{_esc(artifact.model or "\u2014")}</div>
  <div class="row"><span class="k">live model (probes)</span><code>{_esc(model_spec)}</code></div>
  <div class="row"><span class="k">probes executed</span>{n_probes_fired} findings-bearing \u00b7 {len(report.skipped or [])} skipped \u00b7 {len(report.errors or {})} errors</div>
  <div class="row"><span class="k">cost</span>{_esc(cost_line)}</div>
  <div class="row"><span class="k">integrity stamp</span><code>{_esc(chash)}</code></div>
  <div class="stamp"><b>How to verify:</b> recompute the SHA-256 content hash of the audited artifact
  (questions + gold answers) and compare with the stamp above \u2014 it proves this report refers to the same data.
  This is a <b>provenance stamp, not a cryptographic signature</b> of the report itself.</div>
</div>

<h2>What can this eval result support?</h2>
<div class="frame">Each claim is judged only against the validity probes relevant to it; reliability numbers are
<b>measured</b> by the calibration engine against known ground truth. \u201cSupportable\u201d means
<b>not undermined by the threats we test</b> \u2014 not proven correct. No single trustworthiness grade is produced, by design.</div>
<div class="vgrid">{''.join(vcards)}</div>

<h2>Dominant validity threats</h2>
{dom_html}

<h2>All findings ({len(findings)})</h2>
{''.join(rows)}

{f'<h2>Skipped \u2014 need model access</h2><div class="mut">{skipped}</div>' if skipped else ''}
{f'<h2>Errors</h2>{errors}' if errors else ''}

<h2>Limitations \u2014 read before citing</h2>
<div class="lim">
<b>Synthetic floor is internal validity, not external:</b> probes tagged \u201csynthetic floor\u201d are validated against
models defective by construction; their field precision on messy real data is unproven.
<b>Per-claim verdicts are rule-based,</b> not learned, and the probe\u2192claim mapping is editorial domain judgement.
<b>Small samples carry wide intervals:</b> point estimates of 100% reliability with few groups are not claims of perfection.
This report rides the OpenTelemetry <code>gen_ai.eval.trust.*</code> namespace and is reproducible:
<code>trust-the-eval check &lt;artifact&gt; --model {_esc(model_spec)} --trust-report report.html</code>.
</div>

<div class="foot">Trust the Eval \u2014 open-source validity audit battery. Apache-2.0.
Calibration evidence (per-probe precision/recall, ROC/PR curves, false-positive regimes):
<code>trust-the-eval calibrate --report-html calibration.html</code>.</div>
</div></body></html>"""
