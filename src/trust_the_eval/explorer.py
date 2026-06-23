"""Record/store explorer view — the data twin for the P0/P1 milestones.

`build_explorer_html(store)` renders the raw contents of a :class:`RecordStore`
as a navigable page: every :class:`ValidityRecord` with its subject, content id
and verify status, its provenance (method / inputs hash / sources), and each
probe audit with its measured quantities and reliability snapshot. It exists so
the record + store layers (P0) and the model-free intrinsic audit (P1) have a
faithful, reproducible visual — generated from the real store, never by hand.

This complements :mod:`trust_the_eval.observatory` (the public benchmark site,
P2/P3): the explorer shows the records themselves; the observatory shows the
reader-facing pages built from them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .observatory import _phase_html, _summarise


def build_explorer_html(store: Any, title: str = "Meridian", phase=None) -> str:
    site = _summarise(store)
    data = {"records": site["records"], "n": site["n_records"],
            "verify_all_clean": site["verify_all_clean"],
            "generated_utc": site["generated_utc"], "title": title}
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__PHASE__", _phase_html(phase)))


def build_explorer(store: Any, out_dir: Any, title: str = "Meridian", phase=None) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "explorer.html"
    p.write_text(build_explorer_html(store, title=title, phase=phase), encoding="utf-8")
    return {"explorer": str(p)}


_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian · record explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0A0C10;--card:#11151C;--card2:#151B23;--line:#1F2630;--line2:#2A333F;--hair:#171D25;--text:#EAEDF1;--muted:#9AA4B2;--faint:#646E7B;
--accent:#5E9CEA;--accent2:#86BBF4;--valid:#43BE8C;--drift:#E2A951;--broken:#E76A50;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter',system-ui,sans-serif;}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1100px 520px at 72% -12%,#121A26 0%,var(--bg) 55%) no-repeat,var(--bg);
color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;letter-spacing:-.005em}
a{color:var(--accent2);text-decoration:none}a:hover{text-decoration:underline}.mono{font-family:var(--mono)}.wrap{max-width:1180px;margin:0 auto;padding:0 28px}
header{border-bottom:1px solid var(--hair);position:sticky;top:0;background:rgba(10,12,16,.78);backdrop-filter:blur(10px);z-index:10}
.hbar{display:flex;align-items:center;justify-content:space-between;padding:15px 0;gap:16px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:11px}.glyph{width:20px;height:20px;position:relative;flex:none}
.glyph i{position:absolute;left:50%;top:0;bottom:0;width:1.5px;background:linear-gradient(var(--accent2),transparent);transform:translateX(-50%)}
.glyph b{position:absolute;left:50%;top:50%;width:6px;height:6px;border-radius:50%;background:var(--accent);transform:translate(-50%,-50%);box-shadow:0 0 0 4px rgba(94,156,234,.18)}
.brand .nm{font-weight:700;font-size:15px}.brand .nm em{font-style:normal;color:var(--muted);font-weight:500}
.hstatus{font-family:var(--mono);font-size:11px;color:var(--faint);display:flex;gap:14px;align-items:center}.ok{color:var(--valid)}
.hstatus .d{width:6px;height:6px;border-radius:50%;background:var(--valid);box-shadow:0 0 7px var(--valid);display:inline-block;margin-right:6px}
.phasebar{background:linear-gradient(180deg,rgba(94,156,234,.13),rgba(94,156,234,.02));border-bottom:1px solid var(--line)}
.phasebar .wr{max-width:1180px;margin:0 auto;padding:0 28px;display:flex;gap:12px;align-items:center}
.phasebar .wr.pl{padding-bottom:13px;gap:6px;flex-wrap:wrap}
.phasebar .now{margin-top:13px;background:var(--accent);color:#08111c;font-family:var(--mono);font-weight:700;font-size:12px;border-radius:7px;padding:4px 10px;letter-spacing:.06em}
.phasebar .lab{margin-top:13px;font-size:13px;color:var(--text);font-weight:500}
.phasebar .st{font-family:var(--mono);font-size:10.5px;letter-spacing:.02em;padding:4px 10px;border-radius:999px;border:1px solid var(--line2);color:var(--faint)}
.phasebar .st.active{color:#08111c;background:var(--accent);border-color:var(--accent);font-weight:700}
.phasebar .st.done{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.08)}
.phasebar .sep{color:var(--faint);font-family:var(--mono);font-size:11px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent2)}
h1{font-size:24px;font-weight:700;letter-spacing:-.02em;margin:26px 0 4px}.sub{color:var(--muted);font-size:14px;max-width:74ch;margin:0 0 8px}
.card{border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,var(--card2),var(--card));padding:18px 20px;box-shadow:0 18px 44px -30px rgba(0,0,0,.6)}
.traj{margin:18px 0 6px}.traj .t{display:flex;align-items:baseline;justify-content:space-between;gap:14px;margin-bottom:4px}.traj .t h3{margin:0;font-size:16px;font-weight:600}.traj .t .note{font-family:var(--mono);font-size:11px;color:var(--faint)}
.grid{display:grid;grid-template-columns:340px 1fr;gap:20px;margin:22px 0 40px;align-items:start}
.reclist{border:1px solid var(--line);border-radius:14px;background:var(--card);overflow:hidden;position:sticky;top:74px}
.rl-sec{font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent2);padding:14px 16px 8px;background:rgba(94,156,234,.05);border-bottom:1px solid var(--hair)}
.row{display:block;width:100%;text-align:left;background:none;border:0;border-bottom:1px solid var(--hair);padding:13px 16px;cursor:pointer;color:inherit;transition:background .12s}
.row:hover{background:var(--card2)}.row[aria-current=true]{background:rgba(94,156,234,.10);box-shadow:inset 2px 0 0 var(--accent)}
.row .rt{font-weight:600;font-size:14px}.row .rm{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:4px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{font-family:var(--mono);font-size:9.5px;letter-spacing:.05em;text-transform:uppercase;padding:2px 7px;border-radius:999px;border:1px solid}
.b-run{color:var(--accent2);border-color:rgba(94,156,234,.45);background:rgba(94,156,234,.10)}.b-lit{color:var(--drift);border-color:rgba(226,169,81,.45);background:rgba(226,169,81,.10)}.b-aud{color:var(--valid);border-color:rgba(67,190,140,.45);background:rgba(67,190,140,.10)}
.detail h2{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
.kind{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid var(--line2);color:var(--muted);vertical-align:middle;margin-left:10px}
.subline{color:var(--muted);font-size:13.5px;margin:0 0 14px}.subline b{color:var(--text)}
.idrow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-family:var(--mono);font-size:11.5px;color:var(--faint);border-top:1px dashed var(--line);border-bottom:1px dashed var(--line);padding:12px 0;margin-bottom:18px}.idrow .id{color:var(--muted)}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid}.p-ok{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.1)}
.h4{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);margin:0 0 12px;font-weight:500}
.prov{display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;font-family:var(--mono);font-size:12px;margin-bottom:8px}.prov .k{color:var(--faint)}.prov .v{color:var(--text)}
.srcs{margin:12px 0 0;padding:0;list-style:none}.srcs li{font-size:13px;padding:5px 0;color:var(--muted)}.srcs .ca{font-family:var(--mono);font-size:11px;color:var(--faint);margin-left:6px}
.audit{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:16px 18px;margin-bottom:12px}
.audit .top{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.sevdot{width:9px;height:9px;border-radius:3px;flex:none}.s-high{background:var(--broken)}.s-medium{background:var(--drift)}.s-low{background:var(--valid)}.s-info{background:var(--faint)}
.audit .pid{font-family:var(--mono);font-size:13px}.sevchip{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:999px;border:1px solid}
.c-high{color:var(--broken);border-color:rgba(231,106,80,.4)}.c-medium{color:var(--drift);border-color:rgba(226,169,81,.4)}.c-low{color:var(--valid);border-color:rgba(67,190,140,.4)}.c-info{color:var(--faint);border-color:var(--line)}
.derived{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent2)}
.summary{font-size:14px;margin:2px 0 12px}.measured{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.kv{font-family:var(--mono);font-size:11px;border:1px solid var(--line);border-radius:8px;padding:5px 9px;background:rgba(255,255,255,.02)}.kv .kk{color:var(--faint)}.kv .vv{color:var(--text)}
.rel{border-top:1px solid var(--hair);padding-top:11px;font-size:12.5px;color:var(--muted)}.rel .tier{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--accent2)}.rel .nums{font-family:var(--mono);color:var(--text)}.rel .basis{color:var(--faint);font-size:11.5px;margin-top:5px}
footer{border-top:1px solid var(--hair);padding:24px 0 50px;color:var(--faint);font-size:12.5px}
.albl{font-family:var(--mono);font-size:9.5px;fill:var(--faint)}
@media (max-width:820px){.grid{grid-template-columns:1fr}.reclist{position:static}.prov{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap hbar">
  <div class="brand"><span class="glyph"><i></i><b></b></span><span class="nm">Meridian <em>· record explorer</em></span></div>
  <div class="hstatus"><span><span class="d"></span><span id="hs"></span></span><span class="mono">content-addressed store</span></div>
</div></header>
__PHASE__
<div class="wrap">
  <div class="eyebrow" style="margin-top:26px">Live data · generated from the store</div>
  <h1>Real ValidityRecords, straight from the store.</h1>
  <p class="sub">Every record below was produced by the actual code, addressed by the SHA-256 of its content, and integrity-checked. Click a record to inspect its subject, provenance, and each probe audit's reliability snapshot.</p>
  <div class="traj card"><div class="t"><h3>Per-claim trajectory (if any)</h3><span class="note">from the claim records</span></div><div id="trajchart"></div></div>
  <div class="grid"><div class="reclist" id="reclist"></div><div class="detail card" id="detail"></div></div>
</div>
<footer><div class="wrap">Genuine output · <span id="fcount"></span> records · integrity verified at export.</div></footer>
<script>
const DATA = __DATA__;
const SEVC={high:"broken",medium:"drift",low:"valid",info:"info"};
function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function mcls(m){return m==="battery_run"?"b-run":(m==="intrinsic_audit"?"b-aud":"b-lit");}
function fmtVal(v){return Array.isArray(v)?v.join("–"):(typeof v==="number"?(""+(Math.round(v*10000)/10000)):esc(v));}
function smooth(p){if(p.length<2)return p.length?`M${p[0][0]} ${p[0][1]}`:"";let d=`M${p[0][0].toFixed(1)} ${p[0][1].toFixed(1)}`;
 for(let i=0;i<p.length-1;i++){const a=p[i-1]||p[i],b=p[i],c=p[i+1],e=p[i+2]||c;
 d+=`C${(b[0]+(c[0]-a[0])/6).toFixed(1)} ${(b[1]+(c[1]-a[1])/6).toFixed(1)} ${(c[0]-(e[0]-b[0])/6).toFixed(1)} ${(c[1]-(e[1]-b[1])/6).toFixed(1)} ${c[0].toFixed(1)} ${c[1].toFixed(1)}`;}return d;}
function trajectory(){
 const cs=DATA.records.filter(r=>r.subject.kind==="claim"&&r.subject.reported_score!=null)
   .sort((a,b)=>(a.subject.date||"").localeCompare(b.subject.date||""));
 if(!cs.length){document.getElementById("trajchart").innerHTML="<div style='font-family:var(--mono);font-size:11px;color:var(--faint)'>No claim records in this store.</div>";return;}
 const W=1040,H=220,mL=40,mR=120,mT=20,mB=34,n=cs.length,X=i=>mL+(n<=1?0:i/(n-1)*(W-mL-mR)),Y=v=>mT+(1-v/100)*(H-mT-mB);
 let s=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet">`;
 s+=`<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity=".22"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>`;
 [0,50,100].forEach(g=>{s+=`<line x1="${mL}" y1="${Y(g)}" x2="${W-mR}" y2="${Y(g)}" stroke="var(--hair)"/><text class="albl" x="${mL-8}" y="${Y(g)+3}" text-anchor="end">${g}%</text>`;});
 const pts=cs.map((c,i)=>[X(i),Y(c.subject.reported_score*100)]);const base=Y(0);
 s+=`<path d="${smooth(pts)}L${pts[pts.length-1][0].toFixed(1)} ${base} L${pts[0][0].toFixed(1)} ${base} Z" fill="url(#g)"/>`;
 s+=`<path d="${smooth(pts)}" fill="none" style="stroke:var(--accent)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>`;
 cs.forEach((c,i)=>{const x=X(i),y=Y(c.subject.reported_score*100);
   s+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.6" fill="var(--accent)"/>`;
   s+=`<text class="albl" x="${x.toFixed(1)}" y="${(y-10).toFixed(1)}" text-anchor="middle" style="fill:var(--text);font-weight:600">${Math.round(c.subject.reported_score*100)}%</text>`;
   s+=`<text class="albl" x="${x.toFixed(1)}" y="${(y-22).toFixed(1)}" text-anchor="middle" style="fill:var(--muted)">${esc(c.subject.model||"")}</text>`;
   s+=`<text class="albl" x="${x.toFixed(1)}" y="${H-9}" text-anchor="middle">${esc(c.subject.date||"")}</text>`;});
 s+=`</svg>`;document.getElementById("trajchart").innerHTML=s;
}
function shortId(r){return r.record_id.slice(7,19);}
function rowLabel(r){const s=r.subject;return s.kind==="claim"?`${s.benchmark} · ${s.model}`:`${s.benchmark} · ${s.dataset_version||""}`;}
let SEL=null;
function renderList(){
 const intr=DATA.records.filter(r=>r.subject.kind==="intrinsic");
 const clm=DATA.records.filter(r=>r.subject.kind==="claim").sort((a,b)=>(a.subject.date||"").localeCompare(b.subject.date||""));
 const host=document.getElementById("reclist");let h="";
 function sec(title,arr){if(!arr.length)return;h+=`<div class="rl-sec">${title} · ${arr.length}</div>`;
   arr.forEach(r=>{h+=`<button class="row" data-id="${r.record_id}" aria-current="${r.record_id===SEL}"><div class="rt">${esc(rowLabel(r))}</div>
     <div class="rm"><span class="badge ${mcls(r.provenance.method)}">${esc(r.provenance.method.replace(/_/g," "))}</span>${r.subject.date?`<span>${esc(r.subject.date)}</span>`:""}<span>${shortId(r)}</span></div></button>`;});}
 sec("Intrinsic",intr);sec("Per-claim",clm);host.innerHTML=h;
 host.querySelectorAll(".row").forEach(b=>b.onclick=()=>{SEL=b.dataset.id;renderList();renderDetail();});
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
function renderDetail(){
 const r=DATA.records.find(x=>x.record_id===SEL)||DATA.records[0];if(!r){document.getElementById("detail").innerHTML="<div class='sub'>Store is empty.</div>";return;}SEL=r.record_id;
 const s=r.subject,p=r.provenance;
 let subline=s.kind==="claim"
   ? `model <b>${esc(s.model)}</b> · date <b>${esc(s.date||"—")}</b> · reported <b>${s.reported_score!=null?(s.reported_score*100).toFixed(1)+"%":"—"}</b> ${s.metric?`(${esc(s.metric)})`:""}`
   : `dataset version <b>${esc(s.dataset_version||"—")}</b>`;
 let prov=`<div class="prov">
   <div><span class="k">method</span><br><span class="badge ${mcls(p.method)}">${esc(p.method.replace(/_/g," "))}</span></div>
   <div><span class="k">inputs hash</span><br><span class="v">${p.inputs_hash?esc(p.inputs_hash.slice(0,22))+"…":"— · not a live run"}</span></div>
   <div><span class="k">tool</span><br><span class="v">${esc(p.tool_version)}</span></div>
   <div><span class="k">probe-set / calibration</span><br><span class="v">${esc(p.probe_set_version)} · ${esc(p.calibration_version)}</span></div></div>
   ${p.note?`<div style="font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-top:8px">${esc(p.note)}</div>`:""}`;
 let srcs=(p.sources&&p.sources.length)?`<ul class="srcs">`+p.sources.map(x=>`<li><a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.title)}</a><span class="ca">${esc(x.authors||"")} ↗</span></li>`).join("")+`</ul>`:"";
 let audits=r.audits.length?r.audits.map(auditCard).join(""):"<div class='sub'>No audits in this record.</div>";
 document.getElementById("detail").innerHTML=`
   <h2>${esc(s.benchmark)}<span class="kind">${esc(s.kind)}</span></h2>
   <p class="subline">${subline}</p>
   <div class="idrow"><span class="id">${esc(r.record_id)}</span><span class="pill p-ok">verified ✓</span><span>${esc(r.schema_version)}</span></div>
   <div class="h4">Provenance</div>${prov}${srcs}
   <div class="h4" style="margin-top:20px">Probe audits · ${r.audits.length}</div>${audits}`;
}
document.getElementById("hs").textContent=`${DATA.n} records · integrity ${DATA.verify_all_clean?"clean":"FAIL"}`;
document.getElementById("fcount").textContent=DATA.n;
SEL=(DATA.records.find(r=>r.provenance.method==="intrinsic_audit")||DATA.records[0]||{}).record_id;
trajectory();renderList();renderDetail();
</script></body></html>'''
