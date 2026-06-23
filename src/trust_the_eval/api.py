"""Read API + export (P5): the stable, machine-readable contract.

Everything Meridian knows is derived from the store; this module packages it as
one versioned document (:func:`export` / ``meridian.json``) others can consume,
combining the per-benchmark intrinsic profile and trajectory, the computed
:mod:`trends`, the end-to-end :mod:`integrity` report, and — if a
:class:`ProbeRegistry` is supplied — the admitted probes with their reliability.
:func:`build_dashboard_html` renders the trends + integrity surface (the P5 twin).

The API exposes instruments-and-claims data only; it never returns judgements
about models or their safety.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import integrity as _integ
from .observatory import _phase_html, _summarise
from .trends import compute_trends

SCHEMA = "meridian-api/v1"


def export(store: Any, registry: Any = None) -> Dict[str, Any]:
    """Build the full machine-readable Meridian document."""
    S = _summarise(store)
    trends = compute_trends(store)
    benches: List[Dict[str, Any]] = []
    for b in S["benchmarks"]:
        intr = b["intrinsic"]
        benches.append({
            "name": b["name"], "slug": b["slug"], "status": b["status_word"],
            "verdict": b["verdict"], "ceiling": b["ceiling"],
            "intrinsic": ({
                "record_id": intr["record_id"],
                "dataset_version": intr["subject"].get("dataset_version"),
                "method": intr["provenance"]["method"],
                "audits": [{"probe_id": a["probe_id"], "severity": a["severity"],
                            "measured": a.get("measured"),
                            "reliability": a.get("reliability")} for a in intr["audits"]],
            } if intr else None),
            "trajectory": [{"model": c["model"], "date": c["date"], "score": c["score"],
                            "record_id": c["record_id"]} for c in b["claims"]],
            "trends": trends.get(b["name"]),
        })
    doc: Dict[str, Any] = {
        "schema": SCHEMA, "generated_utc": S["generated_utc"],
        "integrity": _integ.verify(store, registry),
        "benchmarks": benches,
        "leaderboard": S["leaderboard"],
    }
    if registry is not None:
        adm = registry.admitted()
        doc["method"] = {
            "policy": (next(iter(adm.values())).result["policy"] if adm else {}),
            "admitted": [{
                "probe_id": pid, "recall": r.result["recall"],
                "specificity": r.result["specificity"],
                "recall_ci": [r.result["recall_lo"], r.result["recall_hi"]],
                "specificity_ci": [r.result["spec_lo"], r.result["spec_hi"]],
                "calibration_version": r.result["calibration_version"],
                "record_id": r.record_id} for pid, r in sorted(adm.items())],
        }
    return doc


def write_api(store: Any, out_dir: Any, registry: Any = None) -> Dict[str, str]:
    """Write ``meridian.json`` (the stable API document) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "meridian.json"
    p.write_text(json.dumps(export(store, registry), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return {"api": str(p)}


# small read helpers over the same document -------------------------------
def benchmark_names(store: Any) -> List[str]:
    return [b["name"] for b in _summarise(store)["benchmarks"]]


def trajectory(store: Any, benchmark: str) -> List[Dict[str, Any]]:
    for b in export(store)["benchmarks"]:
        if b["name"] == benchmark or b["slug"] == benchmark:
            return b["trajectory"]
    return []


def trends_for(store: Any, benchmark: str) -> Optional[Dict[str, Any]]:
    return compute_trends(store).get(benchmark)


# --------------------------------------------------------------- P5 twin
def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_dashboard_html(store: Any, registry: Any = None, *,
                         title: str = "Meridian", phase=None) -> str:
    doc = export(store, registry)
    integ = doc["integrity"]

    def pct(x):
        return "—" if x is None else "{:.1f}%".format(x * 100)

    def headroom_bar(t):
        C = t.get("ceiling")
        lat = (t.get("latest") or {}).get("score")
        if C is None or lat is None:
            return '<div class="hb"><div class="hbnote">no ceiling recorded</div></div>'
        over = t.get("over_ceiling")
        latpos = max(0.0, min(1.0, lat))
        ceilpos = max(0.0, min(1.0, C))
        fill = "var(--broken)" if over else "var(--accent)"
        return (
            '<div class="hb"><div class="track">'
            '<div class="fill" style="width:{lw:.1f}%;background:{fill}"></div>'
            '<div class="ceil" style="left:{cw:.1f}%"></div></div>'
            '<div class="hbrow"><span>latest {lat}</span>'
            '<span class="ceillbl">ceiling {ceil}</span></div></div>'.format(
                lw=latpos * 100, cw=ceilpos * 100, fill=fill,
                lat=pct(lat), ceil=pct(C)))

    cards = []
    for b in doc["benchmarks"]:
        t = b["trends"] or {}
        sat = t.get("saturated")
        chip = ('<span class="pill bad">saturated</span>' if sat
                else '<span class="pill ok">headroom</span>')
        reasons = "".join("<li>{}</li>".format(_esc(x)) for x in t.get("saturation_reasons", []))
        gain = t.get("last_gain")
        gain_txt = "—" if gain is None else "{:+.1f} pts".format(gain * 100)
        noise = ("within noise" if t.get("gain_within_noise") is True
                 else ("separable" if t.get("gain_within_noise") is False else "unknown"))
        slope = t.get("slope_per_year")
        slope_txt = "—" if slope is None else "{:+.1f} pts/yr".format(slope * 100)
        span = t.get("span") or [None, None]
        head = t.get("headroom")
        head_txt = ("—" if head is None
                    else ("over ceiling by {:.1f} pts".format(-head * 100) if head < 0
                          else "{:.1f} pts to ceiling".format(head * 100)))
        cards.append(
            '<div class="tcard"><div class="tc-h"><div class="tc-nm">{nm}</div>{chip}</div>'
            '{bar}'
            '<div class="tc-grid">'
            '<div><span class="k">headroom</span><span class="v">{head}</span></div>'
            '<div><span class="k">last gain</span><span class="v">{gain} <em>({noise})</em></span></div>'
            '<div><span class="k">slope</span><span class="v">{slope}</span></div>'
            '<div><span class="k">claims</span><span class="v">{nclaims} · {d0}→{d1}</span></div>'
            '</div>{reasons}<div class="basis">noise basis · {nb}</div></div>'.format(
                nm=_esc(b["name"]), chip=chip, bar=headroom_bar(t), head=head_txt,
                gain=gain_txt, noise=noise, slope=slope_txt,
                nclaims=t.get("n_claims", 0), d0=_esc(span[0] or "—"), d1=_esc(span[1] or "—"),
                reasons=('<ul class="why">' + reasons + "</ul>") if reasons else "",
                nb=_esc(t.get("noise_basis", "—"))))

    admitted = (doc.get("method") or {}).get("admitted", [])
    html = _DASH_TEMPLATE
    html = html.replace("__PHASE__", _phase_html(phase))
    html = html.replace("__DIGEST__", _esc(integ["corpus_digest"][:23]) + "…")
    html = html.replace("__VERIFIED__", "verified ✓" if integ["verified"] else "FAILED")
    html = html.replace("__NREC__", str(integ["n_records"]))
    html = html.replace("__BYKIND__", _esc(", ".join(
        "{} {}".format(v, k) for k, v in sorted(integ["by_kind"].items()))))
    html = html.replace("__NADM__", str(integ.get("admitted_probes", len(admitted))))
    html = html.replace("__CARDS__", "".join(cards))
    html = html.replace("__SCHEMA__", SCHEMA)
    return html


_DASH_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian · trends &amp; integrity</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0A0C10;--card:#11151C;--card2:#151B23;--line:#1F2630;--line2:#2A333F;--hair:#171D25;--text:#EAEDF1;--muted:#9AA4B2;--faint:#646E7B;
--accent:#5E9CEA;--accent2:#86BBF4;--valid:#43BE8C;--drift:#E2A951;--broken:#E76A50;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter',system-ui,sans-serif;}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1100px 520px at 72% -12%,#121A26 0%,var(--bg) 55%) no-repeat,var(--bg);
color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;letter-spacing:-.005em}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
header{border-bottom:1px solid var(--hair)}
.hbar{display:flex;align-items:center;justify-content:space-between;padding:15px 0;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:11px}.glyph{width:20px;height:20px;position:relative;flex:none}
.glyph i{position:absolute;left:50%;top:0;bottom:0;width:1.5px;background:linear-gradient(var(--accent2),transparent);transform:translateX(-50%)}
.glyph b{position:absolute;left:50%;top:50%;width:6px;height:6px;border-radius:50%;background:var(--accent);transform:translate(-50%,-50%);box-shadow:0 0 0 4px rgba(94,156,234,.18)}
.brand .nm{font-weight:700;font-size:15px}.brand .nm em{font-style:normal;color:var(--muted);font-weight:500}
.hstatus{font-family:var(--mono);font-size:11px;color:var(--faint)}
.phasebar{background:linear-gradient(180deg,rgba(94,156,234,.13),rgba(94,156,234,.02));border-bottom:1px solid var(--line)}
.phasebar .wr{max-width:1080px;margin:0 auto;padding:0 28px;display:flex;gap:12px;align-items:center}
.phasebar .wr.pl{padding-bottom:13px;gap:6px;flex-wrap:wrap}
.phasebar .now{margin-top:13px;background:var(--accent);color:#08111c;font-family:var(--mono);font-weight:700;font-size:12px;border-radius:7px;padding:4px 10px;letter-spacing:.06em}
.phasebar .lab{margin-top:13px;font-size:13px;color:var(--text);font-weight:500}
.phasebar .st{font-family:var(--mono);font-size:10.5px;letter-spacing:.02em;padding:4px 10px;border-radius:999px;border:1px solid var(--line2);color:var(--faint)}
.phasebar .st.active{color:#08111c;background:var(--accent);border-color:var(--accent);font-weight:700}
.phasebar .st.done{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.08)}
.phasebar .sep{color:var(--faint);font-family:var(--mono);font-size:11px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent2);margin-top:30px}
h1{font-size:clamp(26px,3.6vw,38px);font-weight:700;letter-spacing:-.025em;margin:14px 0 10px}
.lede{color:var(--muted);font-size:15px;max-width:72ch;margin:0}
.integrity{margin:26px 0 6px;border:1px solid rgba(67,190,140,.3);border-radius:14px;background:linear-gradient(180deg,rgba(67,190,140,.08),rgba(67,190,140,.02));padding:18px 20px;display:flex;flex-wrap:wrap;gap:22px;align-items:center}
.integrity .blk{display:flex;flex-direction:column;gap:3px}.integrity .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.integrity .v{font-family:var(--mono);font-size:13px;color:var(--text)}.integrity .v.ok{color:var(--valid)}
.ledge{display:flex;align-items:baseline;gap:16px;margin:30px 0 16px}.ledge h3{margin:0;font-size:13px;font-family:var(--mono);letter-spacing:.16em;text-transform:uppercase;font-weight:500;color:var(--muted)}.ledge .ln{flex:1;height:1px;background:var(--hair)}
.tcards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.tcard{border:1px solid var(--line);border-radius:14px;background:var(--card);padding:18px 20px}
.tc-h{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:14px}.tc-nm{font-size:17px;font-weight:600}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.05em;text-transform:uppercase;padding:3px 10px;border-radius:999px;border:1px solid;font-weight:600}
.pill.ok{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.1)}
.pill.bad{color:var(--drift);border-color:rgba(226,169,81,.4);background:rgba(226,169,81,.1)}
.hb{margin:6px 0 14px}.track{position:relative;height:10px;border-radius:6px;background:var(--card2);border:1px solid var(--line);overflow:hidden}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:6px 0 0 6px}
.ceil{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--broken);box-shadow:0 0 6px var(--broken)}
.hbrow{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:6px}.ceillbl{color:var(--broken)}.hbnote{font-family:var(--mono);font-size:11px;color:var(--faint)}
.tc-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-bottom:4px}
.tc-grid .k{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);display:block}
.tc-grid .v{font-family:var(--mono);font-size:13px;color:var(--text)}.tc-grid .v em{color:var(--faint);font-style:normal;font-size:11px}
.why{margin:10px 0 0;padding:0;list-style:none;border-top:1px solid var(--hair);padding-top:10px}.why li{font-family:var(--mono);font-size:11.5px;color:var(--drift);padding:2px 0}
.basis{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:8px}
.api{margin:30px 0 40px;border:1px solid var(--line);border-radius:14px;background:var(--card);padding:18px 20px}
.api h4{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent2);margin:0 0 8px}
.api code{font-family:var(--mono);font-size:12px;color:var(--accent2)}.api .f{font-family:var(--mono);font-size:12px;color:var(--muted)}
footer{border-top:1px solid var(--hair);padding:24px 0 50px;color:var(--faint);font-size:12.5px}
@media (max-width:820px){.tcards{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap hbar">
  <div class="brand"><span class="glyph"><i></i><b></b></span><span class="nm">Meridian <em>· trends &amp; integrity</em></span></div>
  <div class="hstatus">__NREC__ records · __VERIFIED__</div>
</div></header>
__PHASE__
<div class="wrap">
  <div class="eyebrow">What the history shows · and proof it wasn't touched</div>
  <h1>Trends over releases, on a corpus you can re-verify.</h1>
  <p class="lede">Each signal is computed from the records: how much valid accuracy remains below the label-error ceiling, whether the latest gain is separable from statistical noise, and whether a benchmark is saturating. One digest pins the entire dataset; change any record and it changes.</p>
  <div class="integrity">
    <div class="blk"><span class="k">corpus digest</span><span class="v">__DIGEST__</span></div>
    <div class="blk"><span class="k">integrity</span><span class="v ok">__VERIFIED__</span></div>
    <div class="blk"><span class="k">records</span><span class="v">__NREC__ · __BYKIND__</span></div>
    <div class="blk"><span class="k">admitted probes</span><span class="v">__NADM__</span></div>
  </div>
  <div class="ledge"><h3>Per-benchmark trends</h3><span class="ln"></span></div>
  <div class="tcards">__CARDS__</div>
  <div class="api"><h4>Machine-readable API</h4>
    <div class="f">The same data is exported as <code>meridian.json</code> (schema <code>__SCHEMA__</code>): per benchmark a profile + trajectory + trends, the integrity report, and the admitted probes with their reliability. One document, stable contract, re-verifiable.</div></div>
</div>
<footer><div class="wrap">Trends &amp; integrity computed from the store · rates eval instruments &amp; claims, never models or their safety.</div></footer>
</body></html>'''
