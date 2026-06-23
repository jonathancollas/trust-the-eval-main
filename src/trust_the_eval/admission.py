"""Probe admission (P4): calibration is the gate, made opposable.

Anyone can write a probe; not every probe earns a place. A probe — built-in or
third-party — is admitted to the registry only if it clears **recall** and
**specificity** against PLANTED-DEFECT calibration cases (artifacts whose
ground-truth defect status is known because the package constructed them). The
admission decision is a content-addressed, verifiable :class:`AdmissionRecord`,
kept append-only in a :class:`ProbeRegistry`, so "calibration is the bar" stops
being a slogan and becomes a check with a paper trail.

A contributed probe submits: a :class:`Probe` subclass and a set of
:class:`CalibrationCase`s (or relies on the package's scenario generator for a
known probe id). The gate runs the probe over those cases, builds the confusion
matrix, and applies an :class:`AdmissionPolicy`. By default the policy judges the
POINT estimate (so well-behaved probes pass on modest case counts) and always
reports the Wilson interval; ``conservative=True`` demands the lower CI bound to
clear — the rigorous setting for high-stakes admission.

Bright line: a probe is admitted on its ability to detect a VALIDITY defect in
an eval artifact, never anything about model safety.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calibration import (
    build_cases, calibrate_probe, has_scenarios, score_threshold_for,
)
from .calibration.coverage import TIER_LABEL, evidence_for
from .probe import all_probes, get_probe
from .record import _short, evidence_version


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------- policy
@dataclass
class AdmissionPolicy:
    """The bar a probe must clear to be admitted."""
    min_recall: float = 0.80
    min_specificity: float = 0.80
    min_per_class: int = 2         # need >= this many defect AND clean cases
    conservative: bool = False     # require the Wilson LOWER bound to clear

    def to_dict(self) -> Dict[str, Any]:
        return {"min_recall": self.min_recall, "min_specificity": self.min_specificity,
                "min_per_class": self.min_per_class, "conservative": self.conservative}


@dataclass
class AdmissionResult:
    probe_id: str
    admitted: bool
    recall: Optional[float] = None
    recall_lo: Optional[float] = None
    recall_hi: Optional[float] = None
    specificity: Optional[float] = None
    spec_lo: Optional[float] = None
    spec_hi: Optional[float] = None
    n_cases: int = 0
    n_pos: int = 0
    n_neg: int = 0
    tier: str = "contributed"
    tier_label: str = "contributed (tier not certified)"
    reasons: List[str] = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)
    calibration_version: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id, "admitted": self.admitted,
            "recall": self.recall, "recall_lo": self.recall_lo, "recall_hi": self.recall_hi,
            "specificity": self.specificity, "spec_lo": self.spec_lo, "spec_hi": self.spec_hi,
            "n_cases": self.n_cases, "n_pos": self.n_pos, "n_neg": self.n_neg,
            "tier": self.tier, "tier_label": self.tier_label, "reasons": list(self.reasons),
            "policy": dict(self.policy), "calibration_version": self.calibration_version,
            "error": self.error}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdmissionResult":
        return cls(**{k: d.get(k) for k in (
            "probe_id", "admitted", "recall", "recall_lo", "recall_hi", "specificity",
            "spec_lo", "spec_hi", "n_cases", "n_pos", "n_neg", "tier", "tier_label",
            "reasons", "policy", "calibration_version", "error")})


def _tier_of(pid: str):
    try:
        tier, _src, _note = evidence_for(pid)
        return tier, TIER_LABEL.get(tier, tier)
    except Exception:
        return "contributed", "contributed (tier not certified)"


def evaluate_probe(probe_or_id, cases=None, *, policy: Optional[AdmissionPolicy] = None,
                   score_threshold: Optional[float] = None, seed: int = 0) -> AdmissionResult:
    """Run a probe through the admission gate and return a reasoned verdict."""
    policy = policy or AdmissionPolicy()
    probe = get_probe(probe_or_id)() if isinstance(probe_or_id, str) else probe_or_id
    pid = getattr(probe, "id", str(probe_or_id))
    cv = evidence_version()
    tier, tier_label = _tier_of(pid)

    if cases is None:
        if not has_scenarios(pid):
            return AdmissionResult(pid, False, tier=tier, tier_label=tier_label,
                                   policy=policy.to_dict(), calibration_version=cv,
                                   reasons=["no calibration scenarios for this probe; "
                                            "supply planted-defect cases to be admissible"])
        cases = build_cases(pid, seed=seed)
        if score_threshold is None:
            score_threshold = score_threshold_for(pid)

    cal = calibrate_probe(probe, cases, score_threshold=score_threshold)
    if getattr(cal, "error", None):
        return AdmissionResult(pid, False, tier=tier, tier_label=tier_label,
                               policy=policy.to_dict(), calibration_version=cv,
                               reasons=["calibration error: " + str(cal.error)],
                               error=str(cal.error))
    m = cal.matrix
    rec, spc = m.recall(), m.specificity()
    n_pos, n_neg = m.tp + m.fn, m.tn + m.fp
    n = n_pos + n_neg
    rv = rec.lo if policy.conservative else rec.value
    sv = spc.lo if policy.conservative else spc.value
    which = "Wilson lower bound" if policy.conservative else "point"

    reasons: List[str] = []
    ok_pos = n_pos >= policy.min_per_class
    ok_neg = n_neg >= policy.min_per_class
    if not ok_pos:
        reasons.append("only {} defect case(s) (need \u2265 {})".format(n_pos, policy.min_per_class))
    if not ok_neg:
        reasons.append("only {} clean case(s) (need \u2265 {})".format(n_neg, policy.min_per_class))
    ok_r = ok_pos and rv >= policy.min_recall
    ok_s = ok_neg and sv >= policy.min_specificity
    reasons.append("recall {:.2f} ({}) {} {:.2f}".format(
        rv, which, "\u2265" if rv >= policy.min_recall else "<", policy.min_recall))
    reasons.append("specificity {:.2f} ({}) {} {:.2f}".format(
        sv, which, "\u2265" if sv >= policy.min_specificity else "<", policy.min_specificity))
    admitted = bool(ok_r and ok_s)

    return AdmissionResult(
        probe_id=pid, admitted=admitted,
        recall=rec.value, recall_lo=rec.lo, recall_hi=rec.hi,
        specificity=spc.value, spec_lo=spc.lo, spec_hi=spc.hi,
        n_cases=n, n_pos=n_pos, n_neg=n_neg, tier=tier, tier_label=tier_label,
        reasons=reasons, policy=policy.to_dict(), calibration_version=cv)


def gate(probes=None, *, policy: Optional[AdmissionPolicy] = None,
         seed: int = 0) -> Dict[str, AdmissionResult]:
    """Run a set of probes (default: all registered) through the gate."""
    if probes is None:
        probes = [p() for p in all_probes()]
    return {getattr(p, "id", str(i)): evaluate_probe(p, policy=policy, seed=seed)
            for i, p in enumerate(probes)}


# ------------------------------------------------- content-addressed records
@dataclass
class AdmissionRecord:
    """A verifiable, content-addressed record of one admission decision."""
    result: Dict[str, Any]
    tool_version: str = ""
    created_utc: str = ""
    record_id: str = ""
    schema_version: str = "admission/v1"

    def canonical(self) -> str:
        return json.dumps({"schema_version": self.schema_version, "result": self.result,
                           "tool_version": self.tool_version},
                          sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def compute_id(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def finalize(self) -> "AdmissionRecord":
        self.record_id = self.compute_id()
        return self

    def verify(self) -> bool:
        return bool(self.record_id) and self.record_id == self.compute_id()

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "record_id": self.record_id,
                "created_utc": self.created_utc, "tool_version": self.tool_version,
                "result": self.result}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdmissionRecord":
        return cls(result=d["result"], tool_version=d.get("tool_version", ""),
                   created_utc=d.get("created_utc", ""), record_id=d.get("record_id", ""),
                   schema_version=d.get("schema_version", "admission/v1"))

    @classmethod
    def of(cls, result: AdmissionResult) -> "AdmissionRecord":
        from . import __version__ as _v
        return cls(result=result.to_dict(), tool_version=_v, created_utc=_now()).finalize()


class ProbeRegistry:
    """Append-only, content-addressed store of admission records."""

    def __init__(self, root: Any):
        self.root = Path(root)
        (self.root / "admissions").mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def _path(self, rid: str) -> Path:
        return self.root / "admissions" / (rid.split(":", 1)[1] + ".json")

    def put(self, rec: AdmissionRecord) -> AdmissionRecord:
        if not rec.record_id:
            rec.finalize()
        p = self._path(rec.record_id)
        if not p.exists():
            p.write_text(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"record_id": rec.record_id,
                                     "probe_id": rec.result["probe_id"],
                                     "admitted": rec.result["admitted"],
                                     "created_utc": rec.created_utc}) + "\n")
        return rec

    def admit(self, probe_or_id, **kw) -> AdmissionRecord:
        return self.put(AdmissionRecord.of(evaluate_probe(probe_or_id, **kw)))

    def index(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        out = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def all_ids(self) -> List[str]:
        return [e["record_id"] for e in self.index()]

    def has(self, rid: str) -> bool:
        return self._path(rid).exists()

    def get(self, rid: str) -> Dict[str, Any]:
        return json.loads(self._path(rid).read_text(encoding="utf-8"))

    def get_record(self, rid: str) -> AdmissionRecord:
        return AdmissionRecord.from_dict(self.get(rid))

    def verify_all(self) -> List[str]:
        bad = []
        for rid in self.all_ids():
            try:
                if not self.get_record(rid).verify():
                    bad.append(rid)
            except Exception:
                bad.append(rid)
        return bad

    def admitted(self) -> Dict[str, AdmissionRecord]:
        """Latest ADMITTED record per probe id (by creation time)."""
        best: Dict[str, AdmissionRecord] = {}
        for e in sorted(self.index(), key=lambda e: e.get("created_utc", "")):
            if not e.get("admitted"):
                continue
            rec = self.get_record(e["record_id"])
            best[e["probe_id"]] = rec
        return best


# --------------------------------------------------------------------- twin
def _ci(lo, hi):
    if lo is None or hi is None:
        return ""
    return "[{:.2f}, {:.2f}]".format(lo, hi)


def build_admission_html(results: Dict[str, AdmissionResult], policy: AdmissionPolicy,
                         *, title: str = "Meridian", phase=None,
                         contributed: Optional[List[str]] = None) -> str:
    """Render the admission gate as a Meridian-styled page (the P4 twin)."""
    from .observatory import _phase_html
    contributed = set(contributed or [])

    def metric(v, lo, hi, bar):
        if v is None:
            return '<span class="faint">—</span>'
        cls = "ok" if v >= bar else "bad"
        return ('<span class="m {0}">{1:.2f}</span> <span class="ci">{2}</span>'
                .format(cls, v, _ci(lo, hi)))

    rows = []
    ordered = sorted(results.values(),
                     key=lambda r: (r.probe_id not in contributed, not r.admitted, r.probe_id))
    for r in ordered:
        verdict = ('<span class="pill ok">ADMIT</span>' if r.admitted
                   else '<span class="pill bad">REJECT</span>')
        tag = '<span class="contrib">contributed</span>' if r.probe_id in contributed else ""
        reasons = "".join("<li>{}</li>".format(_esc(x)) for x in r.reasons)
        rows.append(
            '<tr><td class="pid">{pid}{tag}<div class="tier">{tier}</div></td>'
            '<td>{rec}</td><td>{spc}</td><td class="num">{npos}/{nneg}</td>'
            '<td>{verdict}</td><td class="why"><ul>{reasons}</ul></td></tr>'.format(
                pid=_esc(r.probe_id), tag=tag, tier=_esc(r.tier_label),
                rec=metric(r.recall, r.recall_lo, r.recall_hi, policy.min_recall),
                spc=metric(r.specificity, r.spec_lo, r.spec_hi, policy.min_specificity),
                npos=r.n_pos, nneg=r.n_neg, verdict=verdict, reasons=reasons))

    n_admit = sum(1 for r in results.values() if r.admitted)
    body = _ADM_TEMPLATE
    body = body.replace("__PHASE__", _phase_html(phase))
    body = body.replace("__POLICY__", (
        "recall \u2265 {:.0%} &nbsp;·&nbsp; specificity \u2265 {:.0%} &nbsp;·&nbsp; "
        "\u2265 {} planted cases per class &nbsp;·&nbsp; judged on the {}"
        .format(policy.min_recall, policy.min_specificity, policy.min_per_class,
                "Wilson lower bound" if policy.conservative else "point estimate")))
    body = body.replace("__COUNT__", "{} of {} candidates admitted".format(n_admit, len(results)))
    body = body.replace("__ROWS__", "".join(rows))
    return body


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_ADM_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian · probe admission</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0A0C10;--card:#11151C;--card2:#151B23;--line:#1F2630;--line2:#2A333F;--hair:#171D25;--text:#EAEDF1;--muted:#9AA4B2;--faint:#646E7B;
--accent:#5E9CEA;--accent2:#86BBF4;--valid:#43BE8C;--drift:#E2A951;--broken:#E76A50;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter',system-ui,sans-serif;}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1100px 520px at 72% -12%,#121A26 0%,var(--bg) 55%) no-repeat,var(--bg);
color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;letter-spacing:-.005em}
a{color:var(--accent2);text-decoration:none}.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
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
.lede{color:var(--muted);font-size:15px;max-width:70ch;margin:0}
.policy{margin:26px 0 8px;border:1px solid rgba(94,156,234,.3);border-radius:14px;background:linear-gradient(180deg,rgba(94,156,234,.08),rgba(94,156,234,.02));padding:18px 20px}
.policy .h{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent2);margin-bottom:8px}
.policy .v{font-family:var(--mono);font-size:13px;color:var(--text)}
.policy .c{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:6px}
.tablewrap{border:1px solid var(--line);border-radius:14px;overflow:hidden;margin:18px 0 40px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
thead th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:500;padding:13px 16px;background:rgba(94,156,234,.05);border-bottom:1px solid var(--hair)}
tbody td{padding:14px 16px;border-bottom:1px solid var(--hair);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
.pid{font-family:var(--mono);font-size:13px;color:var(--text);white-space:nowrap}
.pid .tier{font-family:var(--sans);font-size:11px;color:var(--faint);margin-top:5px;white-space:normal;max-width:22ch}
.contrib{font-family:var(--mono);font-size:9px;letter-spacing:.05em;text-transform:uppercase;color:var(--accent2);border:1px solid rgba(94,156,234,.45);background:rgba(94,156,234,.10);border-radius:999px;padding:2px 7px;margin-left:8px;vertical-align:middle}
.m{font-family:var(--mono);font-weight:600}.m.ok{color:var(--valid)}.m.bad{color:var(--broken)}.ci{font-family:var(--mono);font-size:11px;color:var(--faint)}
.num{font-family:var(--mono);color:var(--muted)}.faint{color:var(--faint)}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:3px 10px;border-radius:999px;border:1px solid;font-weight:700;white-space:nowrap}
.pill.ok{color:var(--valid);border-color:rgba(67,190,140,.4);background:rgba(67,190,140,.1)}
.pill.bad{color:var(--broken);border-color:rgba(231,106,80,.4);background:rgba(231,106,80,.1)}
.why ul{margin:0;padding:0;list-style:none}.why li{font-family:var(--mono);font-size:11.5px;color:var(--muted);padding:2px 0}
footer{border-top:1px solid var(--hair);padding:24px 0 50px;color:var(--faint);font-size:12.5px}
@media (max-width:820px){.why{display:none}}
</style></head><body>
<header><div class="wrap hbar">
  <div class="brand"><span class="glyph"><i></i><b></b></span><span class="nm">Meridian <em>· probe admission</em></span></div>
  <div class="hstatus">__COUNT__</div>
</div></header>
__PHASE__
<div class="wrap">
  <div class="eyebrow">Calibration is the bar</div>
  <h1>A probe is admitted only if it can prove it works.</h1>
  <p class="lede">Each candidate runs over planted-defect cases — artifacts whose true defect status the package controls. We measure recall and specificity with Wilson intervals and apply one published policy. The same gate judges built-ins and outside contributions; nothing enters on trust.</p>
  <div class="policy"><div class="h">Admission policy</div><div class="v">__POLICY__</div>
    <div class="c">A contribution submits a probe + planted cases (or relies on a known probe's scenarios). The decision is recorded as a content-addressed, verifiable AdmissionRecord.</div></div>
  <div class="tablewrap"><table>
    <thead><tr><th>Probe</th><th>Recall (95% CI)</th><th>Specificity (95% CI)</th><th>def/clean</th><th>Verdict</th><th>Reasoning</th></tr></thead>
    <tbody>__ROWS__</tbody>
  </table></div>
</div>
<footer><div class="wrap">Admission decisions are content-addressed and re-verifiable · the gate is the same for every probe.</div></footer>
</body></html>'''
