"""Static observatory generator (P2): render the public surface from the store.

This makes the observatory REAL: it reads a :class:`RecordStore` and emits a
self-contained, hostable page driven entirely by :class:`ValidityRecord`s — no
hand-written data. It is the data-driven twin of the Meridian mockup.

Output (a tiny static site):
  - ``index.html`` : a single-file SPA with hash-routed permalinks
                     (``#/`` overview, ``#/b/<benchmark>``, ``#/method``);
  - ``records.json``: the raw store export, for transparency / programmatic use.

Each benchmark page shows its INTRINSIC profile (the measured, model-free facts
with their evidence tier and provenance) and its PER-CLAIM trajectory (reported
scores across releases), with the honest distinction between what was measured
here, reported by a study, or not yet measured. Every figure traces back to a
record id and a verifiable provenance.

Bright line: the site rates eval instruments and claims, never models or safety.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calibration.coverage import PROBE_EVIDENCE, TIER_LABEL, coverage_summary
from .leaderboard import leaderboard

_SEV_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}
_STATUS = {3: ("degraded", "broken"), 2: ("drifting", "drift"),
           1: ("valid", "valid"), 0: ("valid", "valid")}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "x"


def _max_sev(audits: List[Dict[str, Any]]) -> int:
    return max([_SEV_RANK.get(a.get("severity", "info"), 0) for a in audits] or [0])


def _summarise(store: Any) -> Dict[str, Any]:
    records = [store.get(rid) for rid in store.all_ids()]
    # attach a verify flag computed by the real code
    for d in records:
        try:
            from .record import ValidityRecord
            d["_verify"] = ValidityRecord.from_dict(d).verify()
        except Exception:
            d["_verify"] = False

    by_bench: Dict[str, Dict[str, Any]] = {}
    for d in records:
        s = d["subject"]
        b = by_bench.setdefault(s["benchmark"], {"name": s["benchmark"],
                                                 "slug": _slug(s["benchmark"]),
                                                 "intrinsic": [], "claims": []})
        if s["kind"] == "claim":
            b["claims"].append(d)
        else:
            b["intrinsic"].append(d)

    benchmarks = []
    for name, b in sorted(by_bench.items()):
        # choose the intrinsic record to profile: prefer a generated audit
        intr = sorted(b["intrinsic"],
                      key=lambda d: 0 if d["provenance"]["method"] == "intrinsic_audit" else 1)
        chosen = intr[0] if intr else None
        claims = sorted(b["claims"], key=lambda d: (d["subject"].get("date") or ""))
        # status / verdict from the most severe audit anywhere on this benchmark
        all_aud = (chosen["audits"] if chosen else []) + [a for c in claims for a in c["audits"]]
        rank = _max_sev(all_aud)
        word, color = _STATUS[rank]
        top = None
        for a in sorted(all_aud, key=lambda a: -_SEV_RANK.get(a.get("severity", "info"), 0)):
            top = a
            break
        verdict = top["summary"] if top else "No findings recorded yet."
        ceiling = None
        if chosen:
            for a in chosen["audits"]:
                m = a.get("measured") or {}
                if "effective_accuracy_ceiling" in m:
                    ceiling = m["effective_accuracy_ceiling"]
        benchmarks.append({
            "name": name, "slug": _slug(name), "status_word": word, "status_color": color,
            "verdict": verdict, "ceiling": ceiling,
            "intrinsic": chosen, "intrinsic_all_ids": [d["record_id"] for d in intr],
            "claims": [{"model": c["subject"].get("model"), "date": c["subject"].get("date"),
                        "score": c["subject"].get("reported_score"),
                        "severity": (c["audits"][0]["severity"] if c["audits"] else "info"),
                        "record_id": c["record_id"], "verify": c.get("_verify", False)}
                       for c in claims],
        })

    method = {
        "coverage": coverage_summary(),
        "probes": sorted(
            [{"probe_id": pid, "tier": ev[0], "tier_label": TIER_LABEL.get(ev[0], ""),
              "source": ev[1], "basis": ev[2]} for pid, ev in PROBE_EVIDENCE.items()],
            key=lambda p: (p["tier"], p["probe_id"])),
    }
    return {"benchmarks": benchmarks, "method": method, "records": records,
            "leaderboard": leaderboard(store),
            "n_records": len(records),
            "verify_all_clean": (store.verify_all() == []),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def _phase_html(phase) -> str:
    if not phase:
        return ""
    code, label = phase if isinstance(phase, (list, tuple)) else ("", str(phase))
    steps = [("P0", "Record & store"), ("P1", "Intrinsic audit"),
             ("P2", "Public site"), ("P3", "Per-claim feed"),
             ("P4", "Probe admission"), ("P5", "Trends & integrity")]
    cur = next((i for i, (c, _n) in enumerate(steps) if c == code), -1)
    chips = []
    for i, (c, nm) in enumerate(steps):
        cls = "active" if i == cur else ("done" if (cur >= 0 and i < cur) else "up")
        mark = "\u2713 " if cls == "done" else ""
        chips.append('<span class="st ' + cls + '">' + mark + c + ' \u00b7 ' + nm + '</span>')
        if i < len(steps) - 1:
            chips.append('<span class="sep">\u2192</span>')
    return ('<div class="phasebar"><div class="wr"><span class="now">' + str(code)
            + '</span><span class="lab">' + str(label) + '</span></div>'
            + '<div class="wr pl">' + "".join(chips) + '</div></div>')


def build_site_html(store: Any, title: str = "Meridian", phase=None) -> str:
    site = _summarise(store)
    site["title"] = title
    html = _TEMPLATE.replace("__SITE__", json.dumps(site, ensure_ascii=False))
    return html.replace("__PHASE__", _phase_html(phase))


def build_site(store: Any, out_dir: Any, title: str = "Meridian", phase=None) -> Dict[str, str]:
    """Generate the static observatory into ``out_dir``. Returns written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    site = _summarise(store)
    site["title"] = title
    index = out / "index.html"
    index.write_text(_TEMPLATE.replace("__SITE__", json.dumps(site, ensure_ascii=False)).replace("__PHASE__", _phase_html(phase)),
                     encoding="utf-8")
    data = out / "records.json"
    data.write_text(json.dumps(site["records"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index": str(index), "data": str(data)}


_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — eval-validity observatory</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0A0C10;--card:#11151C;--card2:#151B23;--line:#1F2630;--line2:#2A333F;--hair:#171D25;--text:#EAEDF1;--muted:#9AA4B2;--faint:#646E7B;
--accent:#5E9CEA;--accent2:#86BBF4;--valid:#43BE8C;--drift:#E2A951;--broken:#E76A50;--slate:#7E8AA0;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter',system-ui,sans-serif;}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1100px 540px at 72% -12%,#121A26 0%,var(--bg) 55%) no-repeat,var(--bg);
color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;letter-spacing:-.005em}
a{color:inherit;text-decoration:none}.mono{font-family:var(--mono)}.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:6px}
header{position:sticky;top:0;z-index:20;background:rgba(10,12,16,.78);backdrop-filter:blur(10px);border-bottom:1px solid var(--hair)}
.hbar{display:flex;align-items:center;justify-content:space-between;padding:15px 0;gap:14px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:11px}.glyph{width:20px;height:20px;position:relative;flex:none}
.glyph i{position:absolute;left:50%;top:0;bottom:0;width:1.5px;background:linear-gradient(var(--accent2),transparent);transform:translateX(-50%)}
.glyph b{position:absolute;left:50%;top:50%;width:6px;height:6px;border-radius:50%;background:var(--accent);transform:translate(-50%,-50%);box-shadow:0 0 0 4px rgba(94,156,234,.18)}
.brand .nm{font-weight:700;font-size:15px}.brand .nm em{font-style:normal;color:var(--muted);font-weight:500}
nav.links{display:flex;gap:4px}nav.links a{padding:8px 13px;font-size:13px;font-weight:500;color:var(--muted);border-radius:8px}
nav.links a[aria-current=true]{color:var(--text);background:var(--card2)}nav.links a:hover{color:var(--text)}
.ping{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);display:flex;align-items:center;gap:7px}
.ping .d{width:6px;height:6px;border-radius:50%;background:var(--valid);box-shadow:0 0 7px var(--valid)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent2)}
h1{font-size:clamp(28px,4.4vw,46px);line-height:1.04;font-weight:700;letter-spacing:-.025em;margin:18px 0 14px}
.lede{font-size:clamp(15px,1.6vw,18px);color:var(--muted);max-width:64ch;margin:0}
.scope{margin-top:20px;display:inline-flex;gap:10px;align-items:center;font-family:var(--mono);font-size:11.5px;color:var(--muted);border:1px solid var(--line);background:rgba(255,255,255,.02);padding:8px 14px;border-radius:999px}
.scope b{color:var(--broken)}
.sec{padding:40px 0}.ledge{display:flex;align-items:baseline;gap:16px;margin:0 0 20px}
.ledge h3{margin:0;font-size:13px;font-family:var(--mono);letter-spacing:.16em;text-transform:uppercase;font-weight:500;color:var(--muted)}.ledge .ln{flex:1;height:1px;background:var(--hair)}.ledge .c{font-family:var(--mono);font-size:11px;color:var(--faint)}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}
.bcard{border:1px solid var(--line);border-radius:16px;background:var(--card);padding:20px;transition:transform .16s,border-color .16s,background .16s;display:block}
.bcard:hover{transform:translateY(-3px);border-color:var(--line2);background:var(--card2)}
.bcard .r1{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}.bcard .nm{font-size:18px;font-weight:600}
.bcard .dv{font-size:11.5px;color:var(--faint);font-family:var(--mono);margin-top:3px}.bcard .verd{font-size:13px;color:var(--muted);margin:12px 0 0;border-top:1px solid var(--hair);padding-top:12px}
.bcard .spark{margin:12px 0 2px}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid;white-space:nowrap;font-weight:500}
.p-valid{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.1)}.p-drift{color:var(--drift);border-color:rgba(226,169,81,.4);background:rgba(226,169,81,.1)}.p-broken{color:var(--broken);border-color:rgba(231,106,80,.4);background:rgba(231,106,80,.1)}
.back{display:inline-block;background:none;border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--muted);padding:7px 12px;margin-bottom:20px}.back:hover{color:var(--text);border-color:var(--line2)}
.bh{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--hair);padding-bottom:18px}
.bh h2{font-size:clamp(26px,4vw,38px);margin:0 0 6px;letter-spacing:-.02em}.bh .dv{font-family:var(--mono);font-size:12px;color:var(--faint)}
.verdict{margin:22px 0 4px;font-size:clamp(17px,2vw,21px);line-height:1.4;max-width:60ch;font-weight:500}
.card{border:1px solid var(--line);border-radius:16px;background:linear-gradient(180deg,var(--card2),var(--card));padding:22px;margin-top:18px;box-shadow:0 18px 44px -30px rgba(0,0,0,.6)}
.chart-cap{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:10px}
.audit{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:16px 18px;margin-bottom:12px}
.audit .top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.sevdot{width:9px;height:9px;border-radius:3px;flex:none}.s-high{background:var(--broken)}.s-medium{background:var(--drift)}.s-low{background:var(--valid)}.s-info{background:var(--faint)}
.audit .pid{font-family:var(--mono);font-size:13px}.sevchip{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:999px;border:1px solid}
.c-high{color:var(--broken);border-color:rgba(231,106,80,.4)}.c-medium{color:var(--drift);border-color:rgba(226,169,81,.4)}.c-low{color:var(--valid);border-color:rgba(67,190,140,.4)}.c-info{color:var(--faint);border-color:var(--line)}
.derived{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent2)}
.summary{font-size:14px;margin:2px 0 12px}.measured{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.kv{font-family:var(--mono);font-size:11px;border:1px solid var(--line);border-radius:8px;padding:5px 9px;background:rgba(255,255,255,.02)}.kv .kk{color:var(--faint)}.kv .vv{color:var(--text)}
.rel{border-top:1px solid var(--hair);padding-top:11px;font-size:12.5px;color:var(--muted)}.rel .tier{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--accent2)}.rel .nums{font-family:var(--mono);color:var(--text)}.rel .basis{color:var(--faint);font-size:11.5px;margin-top:5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.panel{border:1px solid var(--line);border-radius:16px;background:var(--card);padding:20px}
.panel h4{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin:0 0 14px;font-weight:500}
.prov{font-family:var(--mono);font-size:11.5px;color:var(--muted);display:grid;grid-template-columns:1fr;gap:9px}.prov .k{color:var(--faint)}.prov b{color:var(--text);font-weight:500}
.badge{font-family:var(--mono);font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;padding:2px 8px;border-radius:999px;border:1px solid}
.b-run{color:var(--accent2);border-color:rgba(94,156,234,.45);background:rgba(94,156,234,.10)}.b-lit{color:var(--drift);border-color:rgba(226,169,81,.45);background:rgba(226,169,81,.10)}.b-aud{color:var(--valid);border-color:rgba(67,190,140,.45);background:rgba(67,190,140,.10)}
.srcs{margin:12px 0 0;padding:0;list-style:none}.srcs li{font-size:13px;padding:5px 0}.srcs a{color:var(--accent2)}.srcs a:hover{text-decoration:underline}.srcs .ca{font-family:var(--mono);font-size:11px;color:var(--faint);margin-left:6px}
.permalink{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:14px;word-break:break-all}.permalink .pv{color:var(--muted)}
.mcards{display:grid;grid-template-columns:1fr 1fr;gap:14px}.mcard{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:16px 18px}
.mcard .top{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}.mcard .pid{font-family:var(--mono);font-size:12.5px}.mcard .tier{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--accent2)}.mcard .basis{font-size:12px;color:var(--muted)}.mcard .src{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:8px}
.admit{border:1px solid rgba(94,156,234,.3);border-radius:14px;background:linear-gradient(180deg,rgba(94,156,234,.08),rgba(94,156,234,.02));padding:22px;margin:22px 0}
.admit h4{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent2);margin:0 0 10px}.admit p{margin:0;font-size:14px;color:var(--text);max-width:74ch}
.tiers{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 0;font-family:var(--mono);font-size:12px;color:var(--muted)}
.phasebar{background:linear-gradient(180deg,rgba(94,156,234,.13),rgba(94,156,234,.02));border-bottom:1px solid var(--line)}
.phasebar .wr{max-width:1080px;margin:0 auto;padding:0 28px;display:flex;gap:12px;align-items:center}
.phasebar .wr.pl{padding-bottom:13px;gap:6px;flex-wrap:wrap}
.phasebar .now{margin-top:13px;background:var(--accent);color:#08111c;font-family:var(--mono);font-weight:700;font-size:12px;border-radius:7px;padding:4px 10px;letter-spacing:.06em}
.phasebar .lab{margin-top:13px;font-size:13px;color:var(--text);font-weight:500}
.phasebar .st{font-family:var(--mono);font-size:10.5px;letter-spacing:.02em;padding:4px 10px;border-radius:999px;border:1px solid var(--line2);color:var(--faint)}
.phasebar .st.active{color:#08111c;background:var(--accent);border-color:var(--accent);font-weight:700}
.phasebar .st.done{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.08)}
.phasebar .sep{color:var(--faint);font-family:var(--mono);font-size:11px}
footer{border-top:1px solid var(--hair);margin-top:36px;padding:30px 0 56px;color:var(--faint);font-size:12.5px}
.albl{font-family:var(--mono);font-size:9.5px;fill:var(--faint)}
@media (max-width:820px){.cards,.grid2,.mcards{grid-template-columns:1fr}nav.links{display:none}}
.p-slate{color:var(--slate);border-color:rgba(126,138,160,.4);background:rgba(126,138,160,.1)}
.lbtabs{display:flex;gap:8px;margin:8px 0 18px}
.lbtab{font-family:var(--mono);font-size:12px;color:var(--muted);border:1px solid var(--line);padding:7px 13px;border-radius:9px}
.lbtab.on{color:var(--text);background:var(--card2);border-color:var(--line2)}.lbtab .ct{color:var(--faint)}
.lbcap{font-size:13.5px;color:var(--muted);max-width:76ch;margin:0 0 16px}.lbcap b{color:var(--text);font-weight:600}
.lblegend{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:0 0 18px;font-family:var(--mono);font-size:11px}
.lblegend .lc{color:var(--faint);margin-right:12px}
.lblist{display:flex;flex-direction:column;gap:10px}
.lbrow{display:flex;align-items:center;gap:16px;border:1px solid var(--line);border-radius:13px;background:var(--card);padding:15px 18px;transition:border-color .16s,background .16s}
a.lbrow:hover{border-color:var(--line2);background:var(--card2)}
.lbrow.res{align-items:flex-start}
.lbrow .lbnum{font-family:var(--mono);font-size:13px;color:var(--faint);width:22px;flex:none;text-align:right;padding-top:2px}
.lbrow .lbL{flex:none;width:118px;display:flex;flex-direction:column;gap:2px}
.lbrow .lbL .ceil{font-family:var(--mono);font-size:21px;font-weight:600}
.lbrow .lbL .ceil-cap{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}
.lbrow .lbM{flex:1;min-width:0}.lbrow .lbnm{font-size:15px;font-weight:600}
.lbrow .onbench{font-weight:400;color:var(--muted);font-size:13px}
.lbrow .lbmeta{font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-top:3px}
.lbrow .hr.pos{color:var(--valid)}.lbrow .hr.neg{color:var(--broken)}.lbrow .hr.na{color:var(--faint)}
.lbrow .lbwhy{font-size:13px;color:var(--muted);margin-top:7px;max-width:80ch}
.lbrow .lbkv{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}
.lbrow .ins{font-family:var(--mono);font-size:10.5px;color:var(--muted);border:1px solid var(--line);border-radius:7px;padding:3px 8px;background:rgba(255,255,255,.02)}
.lbrow .ins em{font-style:normal;color:var(--accent2)}
.lbrow .ins.so{color:var(--slate);border-color:rgba(126,138,160,.35)}
.lbrow .lbR{flex:none}
.lbrow.v-broken{border-color:rgba(231,106,80,.4);background:linear-gradient(180deg,rgba(231,106,80,.07),transparent)}
@media (max-width:820px){.lbrow{flex-wrap:wrap;gap:10px}.lbrow .lbL{width:auto;flex-direction:row;align-items:baseline;gap:8px}}
.lbfilters{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:2px 0 16px}
.lbfilters .fl{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}
.lbfilters select{font-family:var(--mono);font-size:12px;color:var(--text);background:var(--card2);border:1px solid var(--line);border-radius:8px;padding:7px 11px;cursor:pointer}
.lbchip{cursor:pointer;user-select:none}.lbchip[aria-pressed=true]{outline:2px solid currentColor;outline-offset:1px}
.lbwrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--card);box-shadow:0 18px 44px -34px rgba(0,0,0,.6)}
table.lbt{width:100%;border-collapse:collapse;font-size:13.5px;min-width:780px}
.lbt thead th{position:sticky;top:0;background:var(--card2);text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:500;padding:12px 14px;border-bottom:1px solid var(--line2);white-space:nowrap;cursor:pointer}
.lbt thead th.no{cursor:default}.lbt thead th:hover:not(.no){color:var(--text)}.lbt thead th.on{color:var(--text)}
.lbt thead th .ar{color:var(--accent2);margin-left:5px;font-size:8px;vertical-align:middle}
.lbt th.num,.lbt td.num{text-align:right;font-variant-numeric:tabular-nums}
.lbt tbody td{padding:12px 14px;border-bottom:1px solid var(--hair);white-space:nowrap}
.lbt tbody tr:last-child td{border-bottom:none}.lbt tbody tr:hover td{background:var(--card2)}
.lbt .rk{color:var(--faint);font-family:var(--mono);text-align:right;width:38px}
.lbt .mdl{font-weight:600}.lbt a.bn{color:var(--muted)}.lbt a.bn:hover{color:var(--accent2);text-decoration:underline}
.lbt .dt{color:var(--faint);font-family:var(--mono);font-size:12px}
.lbt .sc{font-weight:600;font-family:var(--mono)}
.lbt .hr{font-family:var(--mono)}.lbt .hr.pos{color:var(--valid)}.lbt .hr.neg{color:var(--broken);font-weight:600}.lbt .hr.na{color:var(--faint)}
.lbt .ev{font-family:var(--mono);font-size:11px;color:var(--muted)}.lbt .ev em{font-style:normal;color:var(--accent2)}.lbt .ev.so{color:var(--slate)}
.lbt .kn{color:var(--faint)}
.lbt tr.r-broken td{background:linear-gradient(90deg,rgba(231,106,80,.12),transparent 55%)}
.lbt tr.r-broken:hover td{background:linear-gradient(90deg,rgba(231,106,80,.18),var(--card2) 65%)}
.lbcap2{font-size:12.5px;color:var(--muted);margin:14px 2px 0;max-width:90ch}.lbcap2 b{color:var(--text);font-weight:600}
.standings{display:flex;flex-direction:column;gap:10px;margin-top:6px}
.lb-stand{display:flex;align-items:center;gap:18px;border:1px solid var(--line);border-radius:14px;background:var(--card);padding:16px 20px;transition:transform .16s,border-color .16s,background .16s}
.lb-stand:hover{transform:translateY(-2px);border-color:var(--line2);background:var(--card2)}
.lb-stand .rk{font-family:var(--mono);font-size:25px;font-weight:700;color:var(--faint);width:42px;text-align:center;flex:none}
.lb-stand.v-broken{border-color:rgba(231,106,80,.32)}.lb-stand.v-broken .rk{color:var(--broken)}
.lb-stand .who{flex:1;min-width:0}.lb-stand .nm{font-size:18px;font-weight:600}
.lb-stand .nm .dv{font-family:var(--mono);font-size:11px;color:var(--faint);font-weight:400;margin-left:8px}
.lb-stand .sub{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:4px}
.lb-stand .scorewrap{flex:none;width:248px;display:flex;align-items:center;gap:14px}
.lb-stand .bar{flex:1;height:8px;border-radius:999px;background:var(--line);overflow:hidden}
.lb-stand .bar i{display:block;height:100%;border-radius:999px}
.lb-stand .bar .f-valid{background:var(--valid)}.lb-stand .bar .f-drift{background:var(--drift)}.lb-stand .bar .f-broken{background:var(--broken)}
.lb-stand .score{font-family:var(--mono);font-size:22px;font-weight:600;white-space:nowrap}.lb-stand .score span{font-size:12px;color:var(--faint);font-weight:400}
.lb-stand .stcol{flex:none}
@media (max-width:820px){.lb-stand{flex-wrap:wrap;gap:12px}.lb-stand .scorewrap{width:100%}.lb-stand .rk{width:32px;text-align:left}}
</style></head><body>
<header><div class="wrap hbar">
  <a class="brand" href="#/"><span class="glyph"><i></i><b></b></span><span class="nm">Meridian <em>· eval-validity observatory</em></span></a>
  <nav class="links"><a id="nav-home" href="#/">Observatory</a><a id="nav-board" href="#/leaderboard">Validity</a><a id="nav-method" href="#/method">Method</a></nav>
  <span class="ping"><span class="d"></span><span id="ping"></span></span>
</div></header>
__PHASE__
<main id="app" class="wrap"></main>
<footer><div class="wrap" id="foot"></div></footer>
<script>
const SITE = __SITE__;
const SEVC={high:"broken",medium:"drift",low:"valid",info:"info"};
function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function mcls(m){return m==="battery_run"?"b-run":(m==="intrinsic_audit"?"b-aud":"b-lit");}
function fmtVal(v){return Array.isArray(v)?v.join("–"):(typeof v==="number"?(""+(Math.round(v*10000)/10000)):esc(v));}
function smooth(p){if(p.length<2)return p.length?`M${p[0][0]} ${p[0][1]}`:"";let d=`M${p[0][0].toFixed(1)} ${p[0][1].toFixed(1)}`;
 for(let i=0;i<p.length-1;i++){const a=p[i-1]||p[i],b=p[i],c=p[i+1],e=p[i+2]||c;
 d+=`C${(b[0]+(c[0]-a[0])/6).toFixed(1)} ${(b[1]+(c[1]-a[1])/6).toFixed(1)} ${(c[0]-(e[0]-b[0])/6).toFixed(1)} ${(c[1]-(e[1]-b[1])/6).toFixed(1)} ${c[0].toFixed(1)} ${c[1].toFixed(1)}`;}return d;}
function spark(claims){
 const cs=claims.filter(c=>c.score!=null);if(cs.length<2)return "";
 const W=300,H=64,n=cs.length,X=i=>6+i/(n-1)*(W-12),Y=v=>6+(1-v)*(H-12);
 const pts=cs.map((c,i)=>[X(i),Y(c.score)]);
 const col=cs[cs.length-1].score>=0.85?"var(--drift)":"var(--accent)";
 return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet"><path d="${smooth(pts)}" fill="none" stroke="${col}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${pts[pts.length-1][0].toFixed(1)}" cy="${pts[pts.length-1][1].toFixed(1)}" r="3" fill="${col}"/></svg>`;
}
function trajectory(claims,ceiling){
 const cs=claims.filter(c=>c.score!=null);if(!cs.length)return "<div class='chart-cap'>No dated claims recorded for this benchmark yet.</div>";
 const W=900,H=260,mL=40,mR=120,mT=20,mB=34,n=cs.length,X=i=>mL+(n<=1?0:i/(n-1)*(W-mL-mR)),Y=v=>mT+(1-v/100)*(H-mT-mB);
 let s=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet">`;
 s+=`<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity=".22"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>`;
 [0,25,50,75,100].forEach(g=>{s+=`<line x1="${mL}" y1="${Y(g)}" x2="${W-mR}" y2="${Y(g)}" stroke="var(--hair)"/><text class="albl" x="${mL-8}" y="${Y(g)+3}" text-anchor="end">${g}%</text>`;});
 if(ceiling!=null){const cy=Y(ceiling*100);s+=`<line x1="${mL}" y1="${cy}" x2="${W-mR}" y2="${cy}" style="stroke:var(--broken)" stroke-width="1.2" stroke-dasharray="4 4" opacity=".8"/><text class="albl" x="${mL+2}" y="${cy-6}" style="fill:var(--broken);font-weight:600">label-error ceiling ≈ ${(ceiling*100).toFixed(1)}%</text>`;}
 const pts=cs.map((c,i)=>[X(i),Y(c.score*100)]);const base=Y(0);
 s+=`<path d="${smooth(pts)}L${pts[pts.length-1][0].toFixed(1)} ${base} L${pts[0][0].toFixed(1)} ${base} Z" fill="url(#g)"/>`;
 s+=`<path d="${smooth(pts)}" fill="none" style="stroke:var(--accent)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>`;
 cs.forEach((c,i)=>{const x=X(i),y=Y(c.score*100);
   s+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.6" fill="var(--accent)"/>`;
   s+=`<text class="albl" x="${x.toFixed(1)}" y="${(y-10).toFixed(1)}" text-anchor="middle" style="fill:var(--text);font-weight:600">${Math.round(c.score*100)}%</text>`;
   s+=`<text class="albl" x="${x.toFixed(1)}" y="${(y-22).toFixed(1)}" text-anchor="middle" style="fill:var(--muted)">${esc(c.model||"")}</text>`;
   s+=`<text class="albl" x="${x.toFixed(1)}" y="${H-9}" text-anchor="middle">${esc(c.date||"")}</text>`;});
 s+=`</svg>`;return s;
}
function relText(a){
 const r=a.reliability||{},m=a.measured||{};
 if(r.recall!=null)return `<span class="nums">recall ${r.recall.toFixed(2)} · spec ${r.specificity.toFixed(2)} (n=${r.recall_n})</span>`;
 if(m.derived==="human_annotation_single_pass")return `<span class="nums">measured from a single-pass human annotation \u2014 no inter-annotator agreement available</span>`;
 if(m.derived==="reported_by_study")return `<span class="nums" style="color:var(--drift)">reported by a cited study (not re-run here)</span>`;
 return `<span class="nums" style="color:var(--faint)">tier only · no live metrics in this record</span>`;
}
function auditCard(a){
 const meas=Object.entries(a.measured||{}).filter(([k])=>k!=="derived").map(([k,v])=>`<span class="kv"><span class="kk">${esc(k)}</span> <span class="vv">${fmtVal(v)}</span></span>`).join("");
 const der=(a.measured||{}).derived?`<span class="derived">${esc((a.measured.derived||"").replace(/_/g," "))}</span>`:"";
 const r=a.reliability||{};
 return `<div class="audit"><div class="top"><span class="sevdot s-${a.severity}"></span><span class="pid">${esc(a.probe_id)}</span><span class="sevchip c-${a.severity}">${esc(a.severity)}</span>${der}</div>
  <div class="summary">${esc(a.summary)}</div>${meas?`<div class="measured">${meas}</div>`:""}
  <div class="rel"><span class="tier">${esc(r.tier_label||r.tier||"")}</span> · ${relText(a)}${r.source?`<div class="basis">source · ${esc(r.source)}</div>`:""}${r.basis?`<div class="basis">${esc(r.basis)}</div>`:""}</div></div>`;
}
function statusPill(w,c){return `<span class="pill p-${c}">${esc(w)}</span>`;}

function renderOverview(){
 let cards=SITE.benchmarks.map(b=>`<a class="bcard" href="#/b/${esc(b.slug)}">
   <div class="r1"><div><div class="nm">${esc(b.name)}</div><div class="dv">${esc(b.intrinsic?b.intrinsic.subject.dataset_version||"":(b.claims.length+" claims"))}</div></div>${statusPill(b.status_word,b.status_color)}</div>
   ${b.claims.length?`<div class="spark">${spark(b.claims)}</div>`:""}
   <div class="verd">${esc(b.verdict)}</div></a>`).join("");
 if(!SITE.benchmarks.length)cards=`<div class="card">The store has no records yet. Generate some with the intrinsic audit job, then rebuild.</div>`;
 return `<section class="sec" style="padding-top:60px">
   <div class="eyebrow">A public record of which benchmarks still measure something</div>
   <h1>Benchmarks decay.<br>We measure when — from records, not opinions.</h1>
   <p class="lede">Every figure on this site is generated from a calibrated, content-addressed validity record and traces back to its provenance. We rate the instruments and the claims, never the models or their safety.</p>
   <span class="scope">SCOPE · we audit <b>eval validity</b>, never model capability or safety</span></section>
  <section class="sec" style="padding-top:6px"><div class="ledge"><h3>Tracked benchmarks</h3><span class="ln"></span><span class="c">${SITE.benchmarks.length} benchmarks · ${SITE.n_records} records</span></div>
   <div class="cards">${cards}</div></section>`;
}
function renderBenchmark(slug){
 const b=SITE.benchmarks.find(x=>x.slug===slug);if(!b)return renderOverview();
 const intr=b.intrinsic;
 const profile=intr?intr.audits.map(auditCard).join(""):"<div class='chart-cap'>No intrinsic profile recorded yet.</div>";
 let prov="",srcs="",permal="";
 if(intr){const p=intr.provenance;
  prov=`<div class="prov"><div><span class="k">method</span> <span class="badge ${mcls(p.method)}">${esc(p.method.replace(/_/g," "))}</span></div>
   <div><span class="k">inputs hash</span> <b>${p.inputs_hash?esc(p.inputs_hash.slice(0,24))+"…":"— · not a live run"}</b></div>
   <div><span class="k">tool / probe-set / calibration</span> <b>${esc(p.tool_version)} · ${esc(p.probe_set_version)} · ${esc(p.calibration_version)}</b></div>
   ${p.note?`<div><span class="k">note</span> ${esc(p.note)}</div>`:""}</div>`;
  if(p.sources&&p.sources.length)srcs=`<ul class="srcs">`+p.sources.map(x=>`<li><a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.title)}</a><span class="ca">${esc(x.authors||"")} ↗</span></li>`).join("")+`</ul>`;
  permal=`<div class="permalink">permalink · <span class="pv">#/b/${esc(b.slug)}</span><br>record · <span class="pv">${esc(intr.record_id)}</span> ${intr._verify?"· verified ✓":""}</div>`;
 }
 return `<section class="sec" style="padding-top:34px"><a class="back" href="#/">← All benchmarks</a>
   <div class="bh"><div><h2>${esc(b.name)}</h2><div class="dv">${esc(intr?intr.subject.dataset_version||"":"")}</div></div>${statusPill(b.status_word,b.status_color)}</div>
   <p class="verdict">${esc(b.verdict)}</p>
   <div class="card"><div class="ledge"><h3>Per-claim trajectory</h3><span class="ln"></span></div>${trajectory(b.claims,b.ceiling)}<div class="chart-cap">Reported accuracy across releases, from the per-claim records${b.ceiling!=null?" · dashed = the label-error ceiling from the intrinsic profile":""}.</div></div>
   <div class="grid2"><div class="panel"><h4>Provenance</h4>${prov||"—"}${srcs}${permal}</div>
     <div class="panel"><h4>How to read this</h4><p style="font-size:13.5px;color:var(--muted);margin:0">Each probe audit below carries its evidence tier and how its number was obtained: <b style="color:var(--valid)">measured from human annotations</b>, from a <b style="color:var(--accent2)">model-free probe run</b>, or <b style="color:var(--drift)">reported by a cited study</b>. What isn't listed isn't yet measured.</p></div></div>
   <div class="ledge" style="margin-top:36px"><h3>Intrinsic profile · probe audits</h3><span class="ln"></span><span class="c">${intr?intr.audits.length:0}</span></div>${profile}</section>`;
}
function renderMethod(){
 const M=SITE.method,cov=M.coverage,counts=cov.counts;
 const cards=M.probes.map(p=>`<div class="mcard"><div class="top"><span class="pid">${esc(p.probe_id)}</span><span class="tier">${esc(p.tier_label||p.tier)}</span></div>
   <div class="basis">${esc(p.basis)}</div>${p.source?`<div class="src">source · ${esc(p.source)} ↗</div>`:""}</div>`).join("");
 return `<section class="sec" style="padding-top:40px"><div class="eyebrow">Method</div>
   <h1 style="font-size:clamp(26px,3.6vw,40px)">Every probe states the ground truth it's validated on.</h1>
   <p class="lede">Measuring eval validity is a young, empirical discipline. A probe is a hypothesis about one way a benchmark can stop being valid; it enters only once it clears calibration against planted defects, and every finding carries that measured reliability.</p>
   <div class="admit"><h4>Calibration is the bar for admission</h4><p>A new probe runs over synthetic cases — some with a known planted defect, some clean — and its recall and specificity must clear the threshold. The number beside every finding is that measurement, not a claim. This keeps the registry open to contributions without losing trust.</p>
   <div class="tiers"><span>real-labeled · ${counts.real_labeled||0}</span><span>structural-exact · ${counts.structural_exact||0}</span><span>behavioral-synthetic · ${counts.behavioral_synthetic||0}</span></div></div>
   <div class="ledge"><h3>Probe registry</h3><span class="ln"></span><span class="c">${M.probes.length} probes</span></div><div class="mcards">${cards}</div></section>`;
}
function fmtRate(o){return o&&o.rate!=null?`${(o.rate*100).toFixed(0)}%${o.k!=null?` (${o.k}/${o.n})`:""}`:"\u2014";}
function relCell(r){
 if(!r)return `<span class="ev so">\u2014</span>`;
 const cav=`${r.n_models} models \u2014 \u03b1 is noisy at this respondent count`;
 if(r.per_subject&&r.median_alpha!=null)
   return `<span class="ev" title="${cav}; pooled \u03b1=${r.pooled_alpha} is inflated by cross-subject spread">median \u03b1=${r.median_alpha.toFixed(2)} <span class="dt">${r.n_subjects} subtests</span></span>`;
 if(r.pooled_alpha!=null)
   return `<span class="ev" title="${cav}">\u03b1=${r.pooled_alpha.toFixed(2)}</span>`;
 return `<span class="ev so">n/a</span>`;
}
function sensCell(s){
 if(!s)return `<span class="ev so">needs multi-model preds</span>`;
 const tau=s.kendall_tau==null?1:s.kendall_tau;
 const rk=s.ranking_stable?`\u03c4=${tau.toFixed(2)} <em>rank-stable</em>`:`\u03c4=${tau.toFixed(2)} <span style="color:var(--broken);font-weight:600">rank-fragile</span>`;
 const sc=`\u0394 ${s.delta_min_pts>=0?"+":""}${s.delta_min_pts.toFixed(1)}\u2026${s.delta_max_pts>=0?"+":""}${s.delta_max_pts.toFixed(1)} pts`;
 const sd=s.skill_discrimination;
 const sk=sd==null?"":(Math.abs(sd)<0.2?"skill-neutral":(sd>0?"skill-discriminating":"skill-inverted"));
 return `<span class="ev">${rk} \u00b7 ${sc}${sk?` \u00b7 ${sk}`:""}</span>`;
}
function renderLeaderboard(arg){
 const LB=SITE.leaderboard, rows=LB.benchmarks;
 const META='style="font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-top:5px"';
 const head=`<tr><th class="no rk">#</th><th>Benchmark</th><th class="num">Label-error</th><th class="num">Ambiguity</th><th class="num">Reliability (\u03b1)</th><th>Result sensitivity (labels corrected)</th><th>Status</th></tr>`;
 const body=rows.length?rows.map((b,i)=>`<tr>
   <td class="rk">${i+1}</td>
   <td class="mdl"><a class="bn" href="#/b/${esc(b.slug)}">${esc(b.name)}</a>${b.dataset_version?` <span class="dt">${esc(b.dataset_version)}</span>`:""}</td>
   <td class="num">${fmtRate(b.label_error)}</td>
   <td class="num">${fmtRate(b.ambiguity)}</td>
   <td>${relCell(b.reliability)}</td>
   <td>${sensCell(b.sensitivity)}</td>
   <td>${statusPill(b.status_word,b.status_color)}</td></tr>`).join(""):`<tr><td colspan="7" style="padding:22px;text-align:center;color:var(--faint)">No benchmarks recorded yet.</td></tr>`;
 const det=rows.filter(b=>b.sensitivity).map(b=>{
   const s=b.sensitivity, moved=s.models_moved||[], mech=s.mechanism, parts=[];
   parts.push(`correcting <b>${s.n_changed_items}</b> known label errors over <b>${s.n_models}</b> models`);
   parts.push(`ranking <b>${s.ranking_stable?"unchanged":"changed"}</b> (\u03c4=${(s.kendall_tau==null?1:s.kendall_tau).toFixed(3)}${s.tau_lo!=null?`, 95% CI [${s.tau_lo.toFixed(3)}, ${s.tau_hi.toFixed(3)}]`:""})`);
   if(s.p_top1_change!=null&&s.p_top1_change>=0.05)parts.push(`top-1 flips in <b>${(s.p_top1_change*100).toFixed(0)}%</b> of resamples (near-tie)`);
   parts.push(`every score shifts ${s.delta_min_pts>=0?"+":""}${s.delta_min_pts.toFixed(1)} to ${s.delta_max_pts>=0?"+":""}${s.delta_max_pts.toFixed(1)} pts`);
   if(s.skill_discrimination!=null)parts.push(`skill-discrimination \u03c1=${s.skill_discrimination.toFixed(2)}`);
   const mt=mech?`<div ${META}>on corrected items, models match the <b>wrong</b> gold ${(mech.agree_original_gold*100).toFixed(0)}% vs the <b>true</b> answer ${(mech.agree_corrected_gold*100).toFixed(0)}%.</div>`:"";
   return `<div class="card" style="margin-top:10px"><div style="font-size:15px;font-weight:600">${esc(b.name)}</div>
     <div ${META.replace('--faint','--muted')}>${parts.join(" \u00b7 ")}.</div>${mt}
     ${moved.length?`<div ${META}>models that change rank: ${moved.map(esc).join(", ")}.</div>`:""}</div>`;}).join("");
 return `<section class="sec" style="padding-top:40px"><div class="eyebrow">Validity profiles & result sensitivity</div>
   <h1 style="font-size:clamp(26px,3.6vw,40px)">How much can you trust each result?</h1>
   <p class="lede">Not a trust ranking. For each benchmark: the <b>measured label-error and ambiguity</b>, and \u2014 where multi-model predictions exist \u2014 the <b>result sensitivity</b>: how much the score and the ranking actually move when the known label errors are corrected.</p>
   <div class="lbwrap"><table class="lbt"><thead>${head}</thead><tbody>${body}</tbody></table></div>
   ${det?`<h2 style="font-size:18px;margin:26px 0 6px">Result-sensitivity detail</h2>${det}`:""}
   <div class="lbcap2">Empirically (HELM \u00d7 MMLU-Redux, 10 models): label errors always bias <b>absolute scores</b> and reshuffle <b>near-tied</b> models, but flip the <b>overall</b> ranking only when the errors are skill-discriminating and the compared models are close \u2014 which is why the honest answer is a <b>sensitivity</b>, not a single per-benchmark trust score. We rate the eval, never the models.</div>
   </section>`;
}
function setNav(route){
 document.getElementById("nav-home").setAttribute("aria-current", route==="home"?"true":"false");
 document.getElementById("nav-method").setAttribute("aria-current", route==="method"?"true":"false");
 const _bd=document.getElementById("nav-board");if(_bd)_bd.setAttribute("aria-current", route==="board"?"true":"false");
}
function render(){
 const h=(location.hash||"#/").replace(/^#/,"");let html,route="home";
 if(h.startsWith("/method")){html=renderMethod();route="method";}
 else if(h.startsWith("/leaderboard")){html=renderLeaderboard(h.split("/")[2]);route="board";}
 else if(h.startsWith("/b/")){html=renderBenchmark(decodeURIComponent(h.slice(3)));route="bench";}
 else{html=renderOverview();route="home";}
 document.getElementById("app").innerHTML=html;setNav(route);window.scrollTo(0,0);
}
document.getElementById("ping").textContent=`${SITE.n_records} records · integrity ${SITE.verify_all_clean?"clean":"FAIL"}`;
document.getElementById("foot").innerHTML=`Generated from <b style="color:var(--text)">${SITE.n_records}</b> content-addressed records · integrity ${SITE.verify_all_clean?"verified ✓":"FAIL"} · ${esc(SITE.generated_utc)}. Rates eval instruments &amp; claims — never models or their safety.`;
window.addEventListener("hashchange",render);render();
</script></body></html>'''
