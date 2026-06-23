"""A local, dependency-free web UI for Trust the Eval.

Served by the Python standard library only (http.server). Single-page frontend
(index.html, inlined CSS/JS, no CDN -> works offline). Batteries run in a
background thread with live progress and cooperative stop, reusing the exact
same run_battery code path as the CLI.

Launch:  trust-the-eval serve   (or)   python -m trust_the_eval.cli serve
"""
from __future__ import annotations
import base64
import importlib
import inspect
import json
import os
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .. import datagen
from ..adapters import generic_json, inspect_log, promptfoo
from ..emit import html as html_mod, otel as otel_mod, report as report_mod
from ..finding import Finding
from ..model import build_model
from ..probe import all_probes
from ..runner import run_battery
from ..sources import hf as hf_source
from .. import probes as _probes  # noqa: F401  self-registers all probes

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")

_LOCK = threading.Lock()
ARTIFACTS: dict[str, object] = {}
RUNS: dict[str, dict] = {}


# ----------------------------- helpers -----------------------------
def _finding_json(f: Finding) -> dict:
    return {"probe_id": f.probe_id, "severity": f.severity.value,
            "summary": f.summary, "score": f.score,
            "otel_attributes": f.otel_attributes, "evidence": f.evidence}


def _artifact_summary(art) -> dict:
    cats: dict[str, int] = {}
    n_mcq = 0
    for it in art.items:
        c = it.meta.get("category") or it.meta.get("topic")
        if c:
            cats[c] = cats.get(c, 0) + 1
        opts = it.meta.get("options") or it.meta.get("choices")
        if isinstance(opts, list) and len(opts) >= 2:
            n_mcq += 1
    top = dict(sorted(cats.items(), key=lambda kv: -kv[1])[:8])
    return {
        "dataset": art.dataset, "n": art.n,
        "provenance_hash": art.content_hash(),
        "has_responses": any(it.response is not None for it in art.items),
        "has_scores": any(it.score is not None for it in art.items),
        "n_categories": len(cats), "categories": top, "n_mcq": n_mcq,
    }


def _artifact_items(art, offset: int, limit: int) -> dict:
    """Return a paginated slice of items for the dataset explorer."""
    n = art.n
    offset = max(0, offset)
    limit = max(1, min(limit, 200))
    rows = []
    for it in art.items[offset:offset + limit]:
        opts = it.meta.get("options") or it.meta.get("choices")
        rows.append({
            "question": it.question,
            "answer": it.answer,
            "response": it.response,
            "score": it.score,
            "category": it.meta.get("category") or it.meta.get("topic"),
            "options": opts if isinstance(opts, list) else None,
        })
    return {"total": n, "offset": offset, "limit": limit,
            "has_responses": any(it.response is not None for it in art.items),
            "has_scores": any(it.score is not None for it in art.items),
            "items": rows}


def _looks_like_promptfoo(data) -> bool:
    """Detect a promptfoo results export by SHAPE, not filename."""
    if not isinstance(data, dict):
        return False
    res = data.get("results")
    if isinstance(res, dict) and isinstance(res.get("results"), list):
        return True
    rows = res if isinstance(res, list) else None
    r0 = (rows or [None])[0]
    if isinstance(r0, dict) and ({"success", "score", "vars", "testIdx"} & set(r0)):
        return True
    return ("evalId" in data) or ("version" in data and "results" in data)


def _load_from_request(body: dict):
    """Return an EvalArtifact from a /api/load request body."""
    if body.get("type") == "synthetic":
        n = int(body.get("n", 160)); seed = int(body.get("seed", 7))
        data = datagen.generate(n=n, seed=seed)
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return generic_json.load(path)
    if body.get("type") == "upload":
        name = body.get("filename", "upload.json")
        raw = base64.b64decode(body.get("content_b64", ""))
        suffix = ".eval" if name.endswith(".eval") else (".zip" if name.endswith(".zip") else ".json")
        fd, path = tempfile.mkstemp(suffix=suffix); os.close(fd)
        with open(path, "wb") as fh:
            fh.write(raw)
        if name.endswith(".eval") or name.endswith(".zip"):
            return inspect_log.load(path)
        # Detect promptfoo by CONTENT (any filename), then fall back to generic JSON.
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as _fh:
                _data = _json.load(_fh)
            if "promptfoo" in name.lower() or _looks_like_promptfoo(_data):
                return promptfoo.load(path)
        except Exception:
            pass
        return generic_json.load(path)
    raise ValueError("unknown load type")


def _run_thread(run_id: str, artifact, model_spec, probe_ids, api_key=None, overrides=None):
    def progress(ev):
        with _LOCK:
            r = RUNS[run_id]
            r["progress"] = {"i": ev["i"], "total": ev["total"], "id": ev["id"]}
            if ev["status"] == "done":
                r["findings"].extend(_finding_json(f) for f in ev.get("findings", []))
            elif ev["status"] == "skipped":
                r["skipped"].append(ev["id"])
            elif ev["status"] == "error":
                r["errors"][ev["id"]] = ev.get("error", "error")

    def should_continue():
        with _LOCK:
            return not RUNS[run_id]["cancel"]

    try:
        model = build_model(model_spec, api_key=api_key)
        report = run_battery(artifact, model=model, probe_ids=probe_ids,
                             progress=progress, should_continue=should_continue,
                             overrides=overrides)
        with _LOCK:
            r = RUNS[run_id]
            r["report"] = report
            r["cost"] = report.cost
            # ensure final findings/skipped/errors reflect the report exactly
            r["findings"] = [_finding_json(f) for f in report.findings]
            r["skipped"] = list(report.skipped)
            r["errors"] = dict(report.errors)
            r["status"] = "stopped" if r["cancel"] else "done"
    except Exception as exc:  # pragma: no cover - defensive
        with _LOCK:
            RUNS[run_id]["status"] = "error"
            RUNS[run_id]["errors"]["_run"] = f"{type(exc).__name__}: {exc}"


def _run_state(run_id: str) -> dict:
    with _LOCK:
        r = RUNS[run_id]
        return {"status": r["status"], "progress": r["progress"],
                "findings": list(r["findings"]), "skipped": list(r["skipped"]),
                "errors": dict(r["errors"]), "cost": r["cost"],
                "n_items": r["n_items"], "dataset": r["dataset"]}


# ----------------------------- HTTP -----------------------------
def _resolve_source(dotted: str) -> Optional[dict]:
    """Return {'name', 'source'} for a dotted function/callable name, or None.

    Handles plain functions (pkg.mod.func) and class attributes/methods
    (pkg.mod.Class.method) by walking attributes after the importable module.
    """
    parts = dotted.split(".")
    # find the longest importable module prefix, then getattr the rest
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        obj = mod
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
            return {"name": dotted, "source": inspect.getsource(obj)}
        except Exception:
            return None
    return None


def _probe_doc_payload(probe_id: str) -> Optional[dict]:
    """Assemble the scientific doc + the live implementation for one probe."""
    cls = None
    for c in all_probes():
        if c.id == probe_id:
            cls = c
            break
    if cls is None:
        return None
    payload: dict = {
        "id": cls.id, "name": cls.name,
        "requires_model": bool(cls().requires_model),
        "paper_priority": cls.paper_priority,
        "documented": False,
    }
    payload["tunables"] = getattr(cls, "TUNABLES", {})
    # live source of the probe class (always in sync with what runs)
    try:
        payload["code"] = {"probe": inspect.getsource(cls), "helpers": []}
    except Exception as exc:
        payload["code"] = {"probe": f"# source unavailable: {exc}", "helpers": []}
    doc = getattr(cls, "DOC", None)
    if doc is not None:
        payload.update(doc.to_dict())
        payload["documented"] = True
        helpers = []
        for ref in getattr(doc, "code_refs", []) or []:
            r = _resolve_source(ref)
            if r:
                helpers.append(r)
        payload["code"]["helpers"] = helpers
    return payload


# Calibration is computed once and cached (it runs every probe over its
# scenarios). The UI reads this to show measured precision/recall + curves.
_CALIBRATION_CACHE: dict = {}


def _calibration_payload(seed: int = 0) -> dict:
    """Run (or return cached) calibration and shape it for the UI.

    Includes the evidence-tier coverage summary, per-probe metrics, inline SVG
    curves, per-regime specificity, and a one-line caveat pulled from each
    probe's ProbeDoc — the same data the published HTML report uses.
    """
    if seed in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[seed]
    from ..calibration import (run_calibration, coverage_summary, evidence_for,
                               roc_svg, pr_svg)
    from ..calibration.core import CurvePoint, Curves
    from ..probe import get_probe

    try:
        rep = run_calibration(seed=seed)
    except Exception as exc:   # never 500 the endpoint: return a valid, empty-but-typed payload
        return {"generated_at": "", "seed": seed, "pooled": None,
                "coverage": {"counts": {}, "labels": {}, "total": 0},
                "probes": [], "errors": {"run_calibration": f"{type(exc).__name__}: {exc}"}}

    def _caveat(pid: str):
        try:
            c = (get_probe(pid)().DOC.caveats or "").strip()
            first = c.split(". ")[0]
            return (first[:240] + ("\u2026" if len(first) > 240 else "")) if first else None
        except Exception:
            return None

    def _curves_svg(cd: dict):
        roc = {round(p["threshold"], 6): p for p in cd.get("roc", [])}
        full = []
        for p in cd.get("pr", []):
            r = roc.get(round(p["threshold"], 6))
            if r is not None:
                full.append(CurvePoint(p["threshold"], r["tpr"], r["fpr"],
                                       p["precision"], p["recall"]))
        if not full:
            return None, None
        cv = Curves(points=full, roc_auc=cd.get("roc_auc"),
                    average_precision=cd.get("average_precision"),
                    prevalence=cd.get("prevalence"),
                    orientation=cd.get("orientation", "direct"))
        return roc_svg(cv), pr_svg(cv)

    probes_out = []
    asm_errors: dict = {}
    for cal in rep.probes:
      try:
        d = cal.to_dict()
        tier, source, basis = evidence_for(cal.probe_id)
        try:
            roc_s, pr_s = _curves_svg(d["curves"])
        except Exception:
            roc_s, pr_s = None, None
        probes_out.append({
            "id": cal.probe_id,
            "name": cal.probe_name,
            "requires_model": cal.requires_model,
            "tier": tier,
            "tier_source": source,
            "tier_basis": basis,
            "matrix": d["matrix"],
            "curves_meta": {"roc_auc": d["curves"]["roc_auc"],
                            "average_precision": d["curves"]["average_precision"],
                            "orientation": d["curves"]["orientation"],
                            "prevalence": d["curves"]["prevalence"]},
            "roc_svg": roc_s,
            "pr_svg": pr_s,
            "regime_specificity": d["regime_specificity"],
            "caveat": _caveat(cal.probe_id),
        })
      except Exception as exc:
        asm_errors[cal.probe_id] = f"assembly: {type(exc).__name__}: {exc}"

    try:
        _cov = coverage_summary()
    except Exception:
        _cov = {"counts": {}, "labels": {}, "total": len(probes_out)}
    _errs = dict(rep.meta.get("errors") or {}); _errs.update(asm_errors)
    payload = {
        "generated_at": rep.generated_at,
        "seed": rep.seed,
        "pooled": rep.pooled,
        "coverage": _cov,
        "probes": probes_out,
        "errors": _errs,
    }
    _CALIBRATION_CACHE[seed] = payload
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "TrustTheEval/0.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, payload, ctype="application/json", headers=None):
        body = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # -------- GET --------
    def do_GET(self):
        u = urlparse(self.path); path = u.path
        if path == "/" or path == "/index.html":
            with open(INDEX, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if path == "/api/probes":
            data = [{"id": c.id, "name": c.name,
                     "requires_model": bool(c().requires_model),
                     "paper_priority": c.paper_priority} for c in all_probes()]
            return self._send(200, data)
        if path == "/api/env":
            return self._send(200, {"anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                                    "openai_key": bool(os.environ.get("OPENAI_API_KEY"))})
        if path == "/api/calibration":
            seed = int(parse_qs(u.query).get("seed", ["0"])[0])
            try:
                return self._send(200, _calibration_payload(seed))
            except Exception as exc:
                return self._send(500, {"error": f"calibration failed: {exc}"})
        if path.startswith("/api/probe/"):
            probe_id = path.split("/api/probe/", 1)[1].strip("/")
            payload = _probe_doc_payload(probe_id)
            if payload is None:
                return self._send(404, {"error": f"no such probe: {probe_id}"})
            return self._send(200, payload)
        if path.startswith("/api/run/") and path.endswith("/export"):
            run_id = path.split("/")[3]
            fmt = (parse_qs(u.query).get("fmt", ["json"])[0])
            with _LOCK:
                r = RUNS.get(run_id)
                report = r.get("report") if r else None
            if not report:
                return self._send(409, {"error": "run not finished"})
            if fmt == "trust":
                with _LOCK:
                    art = ARTIFACTS.get(r.get("ds_id"))
                    model_spec = r.get("model", "none")
                if art is None:
                    return self._send(409, {"error": "dataset no longer loaded"})
                from ..emit.trust_report import render_trust_report
                try:
                    html_doc = render_trust_report(report, art, model_spec=model_spec)
                except Exception as exc:
                    return self._send(500, {"error": f"trust report failed: {exc}"})
                return self._send(200, html_doc.encode("utf-8"),
                                  "text/html; charset=utf-8",
                                  {"Content-Disposition": "attachment; filename=trust-report.html"})
            if fmt == "html":
                return self._send(200, html_mod.to_html(report).encode("utf-8"),
                                  "text/html; charset=utf-8",
                                  {"Content-Disposition": "attachment; filename=trust-the-eval.html"})
            if fmt == "otel":
                return self._send(200, json.dumps(otel_mod.to_otel_event(report), indent=2).encode("utf-8"),
                                  "application/json",
                                  {"Content-Disposition": "attachment; filename=gen_ai.eval.trust.json"})
            return self._send(200, json.dumps(report_mod.to_dict(report), indent=2).encode("utf-8"),
                              "application/json",
                              {"Content-Disposition": "attachment; filename=trust-the-eval-report.json"})
        if path.startswith("/api/dataset/") and path.endswith("/items"):
            ds_id = path.split("/")[3]
            with _LOCK:
                art = ARTIFACTS.get(ds_id)
            if art is None:
                return self._send(404, {"error": "dataset not loaded"})
            q = parse_qs(u.query)
            offset = int(q.get("offset", ["0"])[0])
            limit = int(q.get("limit", ["25"])[0])
            return self._send(200, _artifact_items(art, offset, limit))
        if path.startswith("/api/run/"):
            run_id = path.split("/")[3]
            with _LOCK:
                exists = run_id in RUNS
            if not exists:
                return self._send(404, {"error": "no such run"})
            return self._send(200, _run_state(run_id))
        return self._send(404, {"error": "not found"})

    # -------- POST --------
    def do_POST(self):
        u = urlparse(self.path); path = u.path
        try:
            if path == "/api/load":
                art = _load_from_request(self._body())
                ds_id = uuid.uuid4().hex[:12]
                with _LOCK:
                    ARTIFACTS[ds_id] = art
                summary = _artifact_summary(art); summary["dataset_id"] = ds_id
                return self._send(200, summary)

            if path == "/api/run":
                b = self._body()
                ds_id = b.get("dataset_id")
                with _LOCK:
                    art = ARTIFACTS.get(ds_id)
                if art is None:
                    return self._send(400, {"error": "dataset not loaded"})
                probe_ids = b.get("probes") or None
                model_spec = b.get("model", "none")
                api_key = b.get("api_key") or None
                overrides = b.get("params") or None
                run_id = uuid.uuid4().hex[:12]
                with _LOCK:
                    RUNS[run_id] = {"status": "running",
                                    "progress": {"i": 0, "total": len(probe_ids) if probe_ids else len(all_probes()), "id": ""},
                                    "findings": [], "skipped": [], "errors": {}, "cost": {},
                                    "cancel": False, "report": None,
                                    "ds_id": ds_id, "model": model_spec,
                                    "n_items": art.n, "dataset": art.dataset}
                threading.Thread(target=_run_thread,
                                 args=(run_id, art, model_spec, probe_ids, api_key, overrides),
                                 daemon=True).start()
                return self._send(200, {"run_id": run_id})

            if path == "/api/compare":
                b = self._body()
                ra_id, rb_id = b.get("run_a"), b.get("run_b")
                with _LOCK:
                    ra, rb = RUNS.get(ra_id), RUNS.get(rb_id)
                if not ra or not rb:
                    return self._send(404, {"error": "unknown run id"})
                if not ra.get("findings") and ra.get("status") == "running":
                    return self._send(409, {"error": "run A not finished"})
                if not rb.get("findings") and rb.get("status") == "running":
                    return self._send(409, {"error": "run B not finished"})
                with _LOCK:
                    art_a = ARTIFACTS.get(ra.get("ds_id"))
                    art_b = ARTIFACTS.get(rb.get("ds_id"))
                    fa, fb = list(ra["findings"]), list(rb["findings"])
                    la = b.get("label_a") or ra.get("dataset") or "A"
                    lb = b.get("label_b") or rb.get("dataset") or "B"
                if art_a is None or art_b is None:
                    return self._send(409, {"error": "dataset no longer loaded"})
                from ..compare import compare as _compare
                try:
                    return self._send(200, _compare(art_a, fa, art_b, fb,
                                                    label_a=la, label_b=lb))
                except Exception as exc:
                    return self._send(500, {"error": f"compare failed: {exc}"})
            if path.startswith("/api/run/") and path.endswith("/cancel"):
                run_id = path.split("/")[3]
                with _LOCK:
                    if run_id in RUNS:
                        RUNS[run_id]["cancel"] = True
                return self._send(200, {"ok": True})

            if path == "/api/hf/search":
                b = self._body()
                results = hf_source.search_datasets(
                    query=(b.get("query") or "").strip(),
                    limit=int(b.get("limit", 25)),
                    sort=b.get("sort") or "downloads",
                    token=b.get("token") or None,
                    relevant_only=bool(b.get("relevant_only", True)),
                    results_only=bool(b.get("results_only", False)))
                return self._send(200, {"results": results})

            if path == "/api/hf/inspect":
                b = self._body()
                dataset = (b.get("dataset") or "").strip()
                if not dataset:
                    return self._send(400, {"error": "dataset id required"})
                info = hf_source.inspect_dataset(
                    dataset, config=b.get("config") or None,
                    split=b.get("split") or None, token=b.get("token") or None)
                return self._send(200, info)

            if path == "/api/hf/pull":
                b = self._body()
                dataset = (b.get("dataset") or "").strip()
                mapping = b.get("mapping") or {}
                if not dataset:
                    return self._send(400, {"error": "dataset id required"})
                if not mapping.get("question"):
                    return self._send(400, {"error": "map a 'question' column first"})
                art = hf_source.pull(
                    dataset=dataset, config=b.get("config"), split=b.get("split"),
                    n=int(b.get("n", 100)), mapping=mapping, token=b.get("token") or None)
                if art.n == 0:
                    return self._send(400, {"error": "no usable rows (check the question mapping)"})
                ds_id = uuid.uuid4().hex[:12]
                with _LOCK:
                    ARTIFACTS[ds_id] = art
                summary = _artifact_summary(art); summary["dataset_id"] = ds_id
                return self._send(200, summary)

            return self._send(404, {"error": "not found"})
        except Exception as exc:
            return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "127.0.0.1", port: int = 8077, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Trust the Eval UI -> {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return 0
