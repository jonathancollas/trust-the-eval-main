"""Per-claim feed (P3): dated model-on-benchmark results -> claim records.

This makes the per-claim axis REAL. A "claim" is a dated assertion that some
model scored some number on some benchmark (the unit a leaderboard publishes).
The feed ingests such rows — shaped like an Open LLM Leaderboard *details*
export — and emits ``kind="claim"`` :class:`ValidityRecord`s carrying the
reported score, the date/release, and provenance.

When the source also provides PER-SAMPLE results (per-question scores, and
ideally a subject per item), the feed audits that specific result with the
model-free, result-level probes — ``statistical_power`` (is the reported number
precise enough to separate models?) and ``subgroup_power`` (do the per-subject
sub-scores have the power to be read?) — each with its calibrated reliability.
A score-only claim carries no fabricated audits; its provenance says so.

Network honesty (kept consistent with the rest): this sandbox has no outbound
network, so the live HF pull runs on the user's machine; here the same code path
reads a local export via ``path=``. Records are content-addressed and verifiable.

Bright line: a claim record rates the reported result's validity, never the
model or its safety.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import probes as _probes  # noqa: F401  (register probes)
from .artifact import EvalArtifact, EvalItem
from .calibration import (
    build_cases, calibrate_probe, has_scenarios, score_threshold_for,
)
from .probe import get_probe
from .record import (
    ProbeAudit, Provenance, ReliabilitySnapshot, Subject, ValidityRecord,
    _sha_of, _short, evidence_version,
)
from .runner import run_battery

# model-free probes that audit ONE dated result (need recorded per-item scores)
RESULT_PROBES = ["statistical_power", "subgroup_power"]

_MODEL_KEYS = ("model", "model_name", "model_id", "name")
_SCORE_KEYS = ("score", "acc", "accuracy", "acc_norm", "value", "metric_value")
_DATE_KEYS = ("date", "release", "release_date", "timestamp")
_N_KEYS = ("n", "num_samples", "n_samples", "count")
_SUBJECT_KEYS = ("subject", "category", "subtask", "task", "topic")


def _pick(d: Dict[str, Any], keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _norm_score(v) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v / 100.0 if v > 1.0 else v          # accept 0-100 or 0-1


def normalize_row(row: Dict[str, Any], *, benchmark: Optional[str] = None) -> Dict[str, Any]:
    """Map a heterogeneous leaderboard row to a canonical claim dict."""
    bm = row.get("benchmark") or benchmark or "?"
    per = row.get("per_sample") or row.get("samples") or row.get("per_item")
    scores: Optional[List[float]] = None
    subjects: Optional[List[Any]] = None
    responses: Optional[List[Any]] = None
    if isinstance(per, list) and per and isinstance(per[0], dict):
        scores = [float(p.get("score")) for p in per if p.get("score") is not None]
        subjects = [_pick(p, _SUBJECT_KEYS) for p in per]
        if any("response" in p for p in per):
            responses = [p.get("response") for p in per]
    elif isinstance(row.get("scores"), list):
        scores = [float(s) for s in row["scores"] if s is not None]
        subjects = row.get("subjects")
        responses = row.get("responses")
    return {"benchmark": bm, "model": _pick(row, _MODEL_KEYS),
            "score": _norm_score(_pick(row, _SCORE_KEYS)),
            "date": _pick(row, _DATE_KEYS), "release": row.get("release"),
            "metric": row.get("metric") or "accuracy", "n": _pick(row, _N_KEYS),
            "scores": scores, "subjects": subjects, "responses": responses}


def _result_artifact(c: Dict[str, Any]) -> EvalArtifact:
    sc = c["scores"] or []
    subs = c["subjects"] or []
    resp = c["responses"] or []
    items: List[EvalItem] = []
    for i, s in enumerate(sc):
        subj = subs[i] if i < len(subs) else None
        items.append(EvalItem(
            question="item {}".format(i), answer="",
            response=(resp[i] if i < len(resp) else None),
            score=(None if s is None else float(s)),
            meta=({"category": subj} if subj else {})))
    return EvalArtifact(dataset="{}@result".format(c["benchmark"]),
                        items=items, model=c["model"])


def calibrate_result_probes(seed: int = 0) -> Dict[str, Any]:
    """Calibrate result-level probes so claim audits carry recall/spec."""
    cals: Dict[str, Any] = {}
    for pid in RESULT_PROBES:
        if has_scenarios(pid):
            cals[pid] = calibrate_probe(get_probe(pid)(), build_cases(pid, seed=seed),
                                        score_threshold=score_threshold_for(pid))
    return cals


def claim_record(row: Dict[str, Any], *, benchmark: Optional[str] = None,
                 sources: Optional[List[Dict[str, Any]]] = None,
                 calibrations: Optional[Dict[str, Any]] = None,
                 method: str = "claim_feed",
                 note: Optional[str] = None) -> ValidityRecord:
    """Build a claim ValidityRecord from one leaderboard row."""
    c = normalize_row(row, benchmark=benchmark)
    cv = evidence_version()
    cals = calibrations or {}
    audits: List[ProbeAudit] = []
    if c["scores"]:
        probe_ids = list(RESULT_PROBES)
        if c["responses"]:
            probe_ids.append("answer_extraction_audit")
        report = run_battery(_result_artifact(c), model=None, probe_ids=probe_ids)
        for f in report.findings:
            rel = (ReliabilitySnapshot.from_calibration(cals[f.probe_id], cv)
                   if f.probe_id in cals
                   else ReliabilitySnapshot.from_evidence(f.probe_id, cv))
            audits.append(ProbeAudit.from_finding(f, rel))
    applied = sorted(set(a.probe_id for a in audits))
    psv = _short(_sha_of(applied)) if applied else _short(_sha_of(["claim_feed"]))
    ident = {"benchmark": c["benchmark"], "model": c["model"], "date": c["date"],
             "score": c["score"], "metric": c["metric"], "n": c["n"],
             "scores": c["scores"]}
    from . import __version__ as _v
    prov = Provenance(
        method=method, tool_version=_v, probe_set_version=psv, calibration_version=cv,
        sources=list(sources or []), inputs_hash=_sha_of(ident),
        note=note or ("Per-sample results audited at result level."
                      if c["scores"] else
                      "Score-only claim; no per-sample data published to audit."))
    subj = Subject(kind="claim", benchmark=c["benchmark"], model=c["model"],
                   release=c["release"], date=c["date"],
                   reported_score=c["score"], metric=c["metric"])
    return ValidityRecord(subject=subj, audits=audits, provenance=prov).finalize()


def ingest_claims(rows, *, store: Any = None, benchmark: Optional[str] = None,
                  sources: Optional[List[Dict[str, Any]]] = None,
                  calibrations: Optional[Dict[str, Any]] = None,
                  method: str = "claim_feed") -> List[ValidityRecord]:
    """Build (and optionally store) claim records from an iterable of rows."""
    out: List[ValidityRecord] = []
    for row in rows:
        rec = claim_record(row, benchmark=benchmark, sources=sources,
                            calibrations=calibrations, method=method)
        if store is not None:
            store.put(rec)
        out.append(rec)
    return out


# ----------------------------------------------------------------- adapters
def load_leaderboard_rows(path: Optional[str] = None,
                          repo: Optional[str] = None, **pull_kw: Any) -> List[Dict[str, Any]]:
    """Load leaderboard result rows.

    ``path`` reads a local JSON/JSONL export (offline, reproducible — the path
    used here). Otherwise, with ``repo`` set, the package's HF source fetches a
    details dataset on the user's machine (network). Rows are passed through
    :func:`normalize_row`, so common column names are accepted.
    """
    if path:
        import json
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith(".jsonl"):
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            else:
                data = json.load(fh)
                rows = data if isinstance(data, list) else data.get("rows", data.get("items", []))
        return rows
    if repo:                                    # network path (user machine)
        from .sources.hf import pull
        art = pull(repo, **pull_kw)
        return [dict(it.meta, response=it.response, score=it.score) for it in art.items]
    return []


def claims_from_hf_leaderboard(path: Optional[str] = None, *, benchmark: str = "MMLU",
                               repo: Optional[str] = None,
                               source: Optional[Dict[str, Any]] = None,
                               **pull_kw: Any):
    """Convenience: load leaderboard rows and tag them with a benchmark + source."""
    rows = load_leaderboard_rows(path=path, repo=repo, **pull_kw)
    src = source or {"title": "Open LLM Leaderboard (details)",
                     "url": "https://huggingface.co/open-llm-leaderboard"}
    for r in rows:
        r.setdefault("benchmark", benchmark)
    return rows, [src]


def synthetic_claims(benchmark: str = "MMLU", *, seed: int = 0) -> List[Dict[str, Any]]:
    """Offline dated claims WITH deterministic per-sample scores, so the
    result-level probes run and the trajectory is real without network."""
    subjects = ["math", "history", "law", "biology"]
    spec = [("GPT-3", "2021", 0.439), ("GPT-4", "2023-03", 0.864),
            ("GPT-4o", "2024-05", 0.887), ("GPT-4.1", "2025-02", 0.902)]
    rows: List[Dict[str, Any]] = []
    n = 200
    for model, date, acc in spec:
        k = round(acc * n)
        per = []
        for i in range(n):
            per.append({"score": 1.0 if i < k else 0.0,
                        "subject": subjects[i % len(subjects)]})
        rows.append({"benchmark": benchmark, "model": model, "date": date,
                     "acc": acc, "n": n, "per_sample": per})
    return rows
