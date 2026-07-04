"""Meridian CLI — one command to build the whole observatory.

    meridian build  [--corpus mmlu-redux.jsonl] [--leaderboard board.jsonl]
                    [--benchmark MMLU] [--store DIR] [--out DIR]
    meridian verify [--store DIR]
    meridian serve  [--out DIR] [--port 8088]

``build`` chains every phase end to end: it turns an annotation corpus into a
model-free INTRINSIC record (P1), a leaderboard export into dated CLAIM records
(P3), stores them content-addressed (P0), runs the calibration ADMISSION gate
over the built-in probes into a registry (P4), generates the public static site
(P2/P3), writes the machine-readable ``meridian.json`` and prints the INTEGRITY
report (P5). With no inputs it runs a fully offline synthetic demo so the command
works out of the box. Everything is reproducible and re-verifiable; it rates
eval instruments and claims, never models or their safety.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def _p(msg: str) -> None:
    print(msg, flush=True)


def _registry_dir(store_dir: str) -> Path:
    return Path(store_dir) / "registry"


def _build(args) -> int:
    from . import api, integrity
    from .admission import AdmissionPolicy, ProbeRegistry
    from .calibration import has_scenarios
    from .claims import (
        calibrate_result_probes, claims_from_hf_leaderboard, ingest_claims, synthetic_claims,
    )
    from .corpus import AnnotationCorpus
    from .intrinsic import audit_corpus, calibrate_dataset_probes
    from .observatory import build_site
    from .probe import all_probes
    from .store import RecordStore

    demo = not args.corpus and not args.leaderboard
    benchmark = args.benchmark or ("ToyMMLU" if demo else "MMLU")
    store = RecordStore(args.store)
    cd = calibrate_dataset_probes(seed=args.seed)
    cr = calibrate_result_probes(seed=args.seed)

    _p("meridian build")
    if demo:
        _p("  (no inputs given \u2014 running an offline synthetic demo)")

    # 1 \u00b7 intrinsic axis (corpus \u2192 model-free record)
    before = len(store.all_ids())
    if args.corpus:
        corpus = AnnotationCorpus.from_mmlu_redux(path=args.corpus)
        if args.benchmark:
            corpus.benchmark = args.benchmark
    else:
        corpus = AnnotationCorpus.synthetic_seed()
        benchmark = corpus.benchmark
    audit_corpus(corpus, calibrations=cd, store=store)
    _p("  P1 intrinsic   \u00b7 audited corpus '{}' ({}) \u2192 +{} record".format(
        args.corpus or "synthetic-seed", corpus.benchmark, len(store.all_ids()) - before))

    # 2 \u00b7 per-claim axis (leaderboard \u2192 dated claim records)
    before = len(store.all_ids())
    if args.leaderboard:
        rows, sources = claims_from_hf_leaderboard(path=args.leaderboard, benchmark=benchmark)
    else:
        rows, sources = synthetic_claims(benchmark), [
            {"title": "synthetic demo claims", "url": "about:blank"}]
    ingest_claims(rows, store=store, sources=sources, calibrations=cr)
    _p("  P3 claims      \u00b7 ingested {} dated claim(s) for {} \u2192 +{} record".format(
        len(rows), benchmark, len(store.all_ids()) - before))

    # 3 \u00b7 admission gate (calibration as the bar) \u2192 registry
    registry = None
    if not args.no_admission:
        registry = ProbeRegistry(_registry_dir(args.store))
        policy = AdmissionPolicy(conservative=args.conservative)
        gated = admitted = 0
        for P in all_probes():
            pid = P().id
            if has_scenarios(pid):
                gated += 1
                rec = registry.admit(pid, policy=policy, seed=args.seed)
                admitted += 1 if rec.result["admitted"] else 0
        _p("  P4 admission   \u00b7 gated {} calibratable probe(s) \u2192 {} admitted".format(gated, admitted))

    # 4 \u00b7 public static site (P2/P3)
    site = build_site(store, args.out, title=args.title)
    _p("  P2 site        \u00b7 {}".format(site["index"]))

    # 5 \u00b7 machine-readable API document (P5)
    apidoc = api.write_api(store, args.out, registry=registry)
    _p("  P5 api         \u00b7 {}".format(apidoc["api"]))

    # 6 \u00b7 integrity report (P5)
    rep = integrity.verify(store, registry)
    _p("  integrity      \u00b7 {} \u00b7 digest {} \u00b7 {} records {}{}".format(
        "verified \u2713" if rep["verified"] else "FAILED",
        rep["corpus_digest"][:19] + "\u2026", rep["n_records"], rep["by_kind"],
        " \u00b7 {} admitted probes".format(rep["admitted_probes"]) if "admitted_probes" in rep else ""))

    _p("\nDone. Open {} or host the directory; consume {}.".format(
        site["index"], Path(apidoc["api"]).name))
    return 0 if rep["verified"] else 1


def _verify(args) -> int:
    from . import integrity
    from .admission import ProbeRegistry
    from .store import RecordStore

    store = RecordStore(args.store)
    reg = None
    if (_registry_dir(args.store) / "index.jsonl").exists():
        reg = ProbeRegistry(_registry_dir(args.store))
    rep = integrity.verify(store, reg)
    _p("integrity \u00b7 {}".format("verified \u2713" if rep["verified"] else "FAILED"))
    _p("  corpus digest : {}".format(rep["corpus_digest"]))
    _p("  records       : {} {}".format(rep["n_records"], rep["by_kind"]))
    if rep["bad_record_ids"]:
        _p("  TAMPERED      : {}".format(rep["bad_record_ids"]))
    if reg is not None:
        _p("  registry      : {} \u00b7 {} admitted".format(
            "ok" if rep["registry_verified"] else "FAILED", rep["admitted_probes"]))
    return 0 if rep["verified"] else 1


def _serve(args) -> int:
    import functools
    import http.server
    import socketserver

    out = Path(args.out)
    if not (out / "index.html").exists():
        _p("No site at {} \u2014 run 'meridian build --out {}' first.".format(out, out))
        return 1
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        _p("Serving {} at http://{}:{}/  (Ctrl-C to stop)".format(out, args.host, args.port))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            _p("\nstopped")
    return 0


def _load_sources(path: str):
    from .sources.feeds import load_sources
    return load_sources(path)


def _sync(args) -> int:
    from .pipeline import sync
    from .state import SourceState
    from .store import RecordStore
    store = RecordStore(args.store)
    state = SourceState(args.store)
    _p("meridian sync")
    outs = sync(store, _load_sources(args.sources), state, root=args.store, force=args.force)
    for o in outs:
        tail = ""
        if o.get("new_records") is not None:
            tail = " (+%d records, %d rows)" % (o["new_records"], o.get("n_rows", 0))
        elif o.get("reasons"):
            tail = " — " + "; ".join(o["reasons"])
        elif o.get("reason"):
            tail = " — " + o["reason"]
        _p("  %-24s %-11s%s" % (o["source"], o["status"], tail))
    return 0


def _observe(args) -> int:
    from .pipeline import observe
    rep = observe(args.store, _load_sources(args.sources), args.out, always=args.always)
    if not rep.get("changed"):
        _p("meridian observe · no change across %d source(s); site untouched."
           % len(rep.get("sync", [])))
        return 0
    _p("meridian observe")
    for o in rep["sync"]:
        tail = " (+%d)" % o["new_records"] if o.get("new_records") else ""
        _p("  %-24s %-11s%s" % (o["source"], o["status"], tail))
    integ = rep["integrity"]
    _p("  site + api    · %s" % rep["out"])
    _p("  integrity     · %s · %d records %s" % (
        "verified ✓" if integ["verified"] else "FAILED", integ["n_records"], integ["by_kind"]))
    _p("  changes       · %d → %s" % (len(rep["changes"]), os.path.basename(rep["feed"])))
    for c in rep["changes"][:8]:
        _p("      - " + json.dumps(c, ensure_ascii=False))
    return 0 if integ["verified"] else 1


def _watch(args) -> int:
    import time
    from .pipeline import observe
    _p("meridian watch · every %ds (Ctrl-C to stop)" % args.interval)
    try:
        while True:
            r = observe(args.store, _load_sources(args.sources), args.out, always=False)
            _p("  cycle · %s" % ("rebuilt" if r.get("changed") else "no change"))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        _p("\nstopped")
        return 0


def _explain(args) -> int:
    """Re-derive a displayed metric from the named inputs and show recompute==displayed."""
    from .store import RecordStore
    from .transparency import explain, format_explanation
    sources = _load_sources(args.sources)
    pred_rows, intr_rows = {}, {}
    for s in sources:
        try:
            p = s.fetch()
        except Exception:
            continue
        if not isinstance(p, dict) or p.get("benchmark") != args.benchmark:
            continue
        if p.get("kind") == "predictions":
            pred_rows[args.benchmark] = p.get("rows", [])
        elif p.get("kind") == "intrinsic":
            intr_rows[args.benchmark] = p.get("rows", [])
    if args.benchmark not in pred_rows and args.benchmark not in intr_rows:
        _p("meridian explain · no sources for benchmark %r in %s" % (args.benchmark, args.sources))
        return 2
    store = RecordStore(args.store)
    metrics = [args.metric] if args.metric else ["tau", "alpha", "label", "ambiguity", "top1", "accuracy"]
    _p("meridian explain · %s" % args.benchmark)
    rc = 0
    for i, mtr in enumerate(metrics):
        e = explain(store, args.benchmark, mtr,
                    pred_rows=pred_rows.get(args.benchmark, []),
                    intrinsic_rows=intr_rows.get(args.benchmark, []))
        _p("")
        _p(format_explanation(e))
        if e.get("comparable") and not e["match"]:
            rc = 1
    return rc


def _sarif(args) -> int:
    """Emit SARIF 2.1.0 from a store (one finding per probe; no aggregate score)."""
    from collections import Counter
    from .store import RecordStore
    from .emit.sarif import to_sarif, write_sarif
    store = RecordStore(args.store)
    log = to_sarif(store, base_uri=args.base_uri)
    run = log["runs"][0]
    path = write_sarif(store, args.out, base_uri=args.base_uri)
    levels = Counter(r["level"] for r in run["results"])
    _p("meridian sarif \u2192 %s" % path)
    _p("  rules: %d  results: %d" % (len(run["tool"]["driver"]["rules"]), len(run["results"])))
    _p("  levels: %s" % dict(levels))
    _p("  (one finding per probe; evidence tier clamps level; no aggregate score)")
    return 0


def _policy(args) -> int:
    from .corrections import render_policy_table
    _p(render_policy_table())
    return 0


def _lineage(args) -> int:
    import json as _json
    from .lineage import format_lineage, lineage
    sources = _load_sources(args.sources)
    pred_rows, intr_rows = [], []
    for s in sources:
        try:
            p = s.fetch()
        except Exception:
            continue
        if not isinstance(p, dict) or p.get("benchmark") != args.benchmark:
            continue
        if p.get("kind") == "predictions":
            pred_rows = p.get("rows", [])
        elif p.get("kind") == "intrinsic":
            intr_rows = p.get("rows", [])
    if not pred_rows:
        _p("meridian lineage · no predictions for benchmark %r in %s" % (args.benchmark, args.sources))
        return 2
    item = args.item
    if item is None:
        changed = [r["item"] for r in pred_rows
                   if set(r.get("original_gold") or []) != set(r.get("corrected_gold") or [])]
        pool = changed if (args.changed or changed) else [r["item"] for r in pred_rows]
        if not pool:
            _p("meridian lineage · no %sitems to trace" % ("changed " if args.changed else ""))
            return 2
        item = pool[0]
    th = lineage(args.benchmark, item, pred_rows, intr_rows)
    _p(_json.dumps(th.to_dict(), ensure_ascii=False, indent=2) if args.json else format_lineage(th))
    return 0


def _read_json_rows(path: str):
    import json
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return []
    if path.endswith(".jsonl") or (txt[0] != "[" and "\n" in txt):
        return [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
    data = json.loads(txt)
    return data if isinstance(data, list) else data.get("rows", [data])


def _import_corrections(args) -> int:
    import json
    import os
    from .ingest import format_journal, ingest_mmlu_redux, ingest_platinum
    if args.format not in ("mmlu-redux", "platinum"):
        _p("meridian import-corrections · --format must be mmlu-redux or platinum")
        return 2
    rows = _read_json_rows(args.infile)
    if args.format == "platinum":
        corr, _intr, journal = ingest_platinum(rows, subject=args.subject)
    else:
        corr, _intr, journal = ingest_mmlu_redux(rows, subject=args.subject)
    os.makedirs(args.out, exist_ok=True)
    by_subj = {}
    for r in corr:
        d = by_subj.setdefault(r.subject, {"corr": [], "intr": []})
        d["corr"].append(r.to_dict())
        d["intr"].append(r.intrinsic())
    for subj, d in by_subj.items():
        slug = str(subj).lower().replace(" ", "_")
        with open(os.path.join(args.out, "mmlu_%s_intrinsic.json" % slug), "w", encoding="utf-8") as f:
            json.dump(d["intr"], f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out, "mmlu_%s_corrections.json" % slug), "w", encoding="utf-8") as f:
            json.dump(d["corr"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "import_journal.json"), "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)
    n_changed = sum(1 for r in corr if r.changed)
    n_dropped = sum(1 for r in corr if not r.scored)
    _p("meridian import-corrections · %d items across %d subjects → %s"
       % (len(corr), len(by_subj), args.out))
    _p("  changed: %d · dropped (not scorable): %d · kept: %d"
       % (n_changed, n_dropped, len(corr) - n_changed - n_dropped))
    _p("  files: mmlu_<subject>_intrinsic.json · mmlu_<subject>_corrections.json · import_journal.json")
    if args.journal:
        _p("")
        _p(format_journal(journal, limit=8, source=args.format))
    return 0


def _load_gold_map(corrections_dir: str):
    import glob
    import json
    import os
    gm = {}
    for fp in glob.glob(os.path.join(corrections_dir, "*_corrections.json")):
        try:
            recs = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for r in recs:
            gm[r["item"]] = {"original_gold": r.get("original_gold"),
                             "corrected_gold": r.get("corrected_gold"),
                             "scored": r.get("scored", True)}
    return gm


def _import_predictions(args) -> int:
    import json
    import os
    from .ingest import (format_predictions_journal, ingest_helm,
                         ingest_inspect, ingest_lm_eval, ingest_lm_eval_freeform,
                         stamp_predictions)
    if args.format not in ("lm-eval", "helm", "inspect"):
        _p("meridian import-predictions · --format must be lm-eval, helm, or inspect")
        return 2
    samples_by_model = {}
    for spec in (args.samples or []):
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, os.path.splitext(os.path.basename(name))[0]
        if args.format == "helm":
            samples_by_model[name] = json.load(open(path, encoding="utf-8"))
        elif args.format == "inspect":
            if path.endswith(".eval"):
                try:
                    from inspect_ai.log import read_eval_log
                except Exception:
                    _p("meridian import-predictions · reading .eval directly needs inspect_ai installed.")
                    _p("  Either `pip install inspect_ai`, or run `inspect log dump %s > out.json` and pass the JSON." % path)
                    return 2
                log = read_eval_log(path)
                samples_by_model[name] = log.model_dump(mode="json") if hasattr(log, "model_dump") else log
            else:
                samples_by_model[name] = json.load(open(path, encoding="utf-8"))
        else:
            samples_by_model[name] = _read_json_rows(path)
    if not samples_by_model:
        _p("meridian import-predictions · pass at least one --samples MODEL=PATH")
        return 2
    if args.format == "helm":
        pred_rows, journal = ingest_helm(samples_by_model, subject=args.subject)
    elif args.format == "inspect":
        pred_rows, journal = ingest_inspect(samples_by_model, subject=args.subject)
    elif args.answers == "freeform":
        pred_rows, journal = ingest_lm_eval_freeform(samples_by_model, subject=args.subject)
    else:
        pred_rows, journal = ingest_lm_eval(samples_by_model, subject=args.subject)

    dropped = []
    if args.corrections:
        gm = _load_gold_map(args.corrections)
        bare = [{"item": r["item"], "subject": r["subject"], "preds": r["preds"]} for r in pred_rows]
        pred_rows, dropped = stamp_predictions(bare, gm)

    os.makedirs(args.out, exist_ok=True)
    by_subj = {}
    for r in pred_rows:
        by_subj.setdefault(r["subject"], []).append(r)
    for subj, rows in by_subj.items():
        slug = str(subj).lower().replace(" ", "_")
        with open(os.path.join(args.out, "mmlu_%s_predictions.json" % slug), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "predictions_journal.json"), "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)
    _p("meridian import-predictions · %d items · %d models · %d subjects → %s"
       % (len(pred_rows), len(samples_by_model), len(by_subj), args.out))
    if args.corrections:
        _p("  joined with corrections in %s · dropped (not scorable): %d" % (args.corrections, len(dropped)))
    _p("  files: mmlu_<subject>_predictions.json · predictions_journal.json")
    if args.journal:
        _p("")
        _p(format_predictions_journal(journal, limit=8, source=args.format))
    return 0


def _lineage_site(args) -> int:
    import os
    from .lineage_view import render_lineage_html
    sources = _load_sources(args.sources)
    pred_rows, intrinsic_rows = {}, {}
    for s in sources:
        try:
            p = s.fetch()
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        if p.get("kind") == "predictions":
            pred_rows[p.get("benchmark")] = p.get("rows", [])
        elif p.get("kind") == "intrinsic":
            intrinsic_rows[p.get("benchmark")] = p.get("rows", [])
    if not pred_rows:
        _p("meridian lineage-site · no predictions found in %s" % args.sources)
        return 2
    html = render_lineage_html(pred_rows, intrinsic_rows, iters=args.iters)
    out = args.out
    if out.endswith("/") or os.path.isdir(out):
        os.makedirs(out, exist_ok=True)
        out = os.path.join(out, "lineage.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    n = sum(len([r for r in rows
                 if set(r.get("original_gold") or []) != set(r.get("corrected_gold") or [])])
            for rows in pred_rows.values())
    _p("meridian lineage-site · %d benchmarks · %d corrected items traced → %s"
       % (len(pred_rows), n, out))
    return 0


def _atlas(args) -> int:
    from .atlas import fragility_atlas, format_atlas
    sources = _load_sources(args.sources)
    pred, intr = {}, {}
    for s in sources:
        try:
            p = s.fetch()
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        if p.get("kind") == "predictions":
            pred[p.get("benchmark")] = p.get("rows", [])
        elif p.get("kind") == "intrinsic":
            intr[p.get("benchmark")] = p.get("rows", [])
    if not pred:
        _p("meridian atlas · no predictions found in %s" % args.sources)
        return 2
    cells = {b: (pred[b], intr.get(b, [])) for b in pred}
    atlas = fragility_atlas(cells, iters=args.iters, boot_iters=args.boot)
    _p(format_atlas(atlas, top=args.top))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meridian", description="Build and verify the Meridian eval-validity observatory.")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the whole observatory from inputs (or a demo)")
    b.add_argument("--corpus", default=None, help="annotation corpus (MMLU-Redux JSON/JSONL) for the intrinsic axis")
    b.add_argument("--leaderboard", default=None, help="leaderboard export (JSON/JSONL) for the per-claim axis")
    b.add_argument("--benchmark", default=None, help="benchmark name to tag records with")
    b.add_argument("--store", default="meridian-store", help="content-addressed store directory")
    b.add_argument("--out", default="meridian-site", help="output directory for the static site + meridian.json")
    b.add_argument("--title", default="Meridian", help="site title")
    b.add_argument("--seed", type=int, default=0, help="calibration seed")
    b.add_argument("--conservative", action="store_true", help="admit on the Wilson lower bound, not the point estimate")
    b.add_argument("--no-admission", action="store_true", help="skip the probe admission gate")
    b.set_defaults(func=_build)

    v = sub.add_parser("verify", help="re-verify a store's integrity")
    v.add_argument("--store", default="meridian-store")
    v.set_defaults(func=_verify)

    s = sub.add_parser("serve", help="serve a built site directory over HTTP")
    s.add_argument("--out", default="meridian-site")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8088)
    s.set_defaults(func=_serve)

    sy = sub.add_parser("sync", help="pull only changed sources into the store (no rebuild)")
    sy.add_argument("--store", default="meridian-store")
    sy.add_argument("--sources", required=True, help="JSON spec of sources to watch")
    sy.add_argument("--force", action="store_true", help="ignore watermarks; re-pull all")
    sy.set_defaults(func=_sync)

    ob = sub.add_parser("observe", help="one cycle: sync -> admission -> site -> verify -> feed")
    ob.add_argument("--store", default="meridian-store")
    ob.add_argument("--sources", required=True)
    ob.add_argument("--out", default="meridian-site")
    ob.add_argument("--always", action="store_true", help="rebuild even if nothing changed")
    ob.set_defaults(func=_observe)

    w = sub.add_parser("watch", help="run an observation cycle on a loop")
    w.add_argument("--store", default="meridian-store")
    w.add_argument("--sources", required=True)
    w.add_argument("--out", default="meridian-site")
    w.add_argument("--interval", type=int, default=3600)
    w.set_defaults(func=_watch)

    ex = sub.add_parser("explain", help="re-derive a displayed metric from the inputs (proves recompute==displayed)")
    ex.add_argument("metric", nargs="?", default=None,
                    help="tau | alpha | label | ambiguity | top1 | accuracy (default: all)")
    ex.add_argument("--benchmark", required=True, help="benchmark name, e.g. 'MMLU::virology'")
    ex.add_argument("--sources", required=True, help="the same JSON source spec used to build the site")
    ex.add_argument("--store", default="meridian-store")
    ex.set_defaults(func=_explain)

    sf = sub.add_parser("sarif", help="emit SARIF 2.1.0 findings from a store (for code-scanning / PR annotations)")
    sf.add_argument("--store", default="meridian-store")
    sf.add_argument("--out", default="site", help="directory to write meridian.sarif into")
    sf.add_argument("--base-uri", default=None, help="base URL for rule helpUri / informationUri")
    sf.set_defaults(func=_sarif)

    pol = sub.add_parser("policy", help="print the declared correction policy (error_type -> action, with sources)")
    pol.set_defaults(func=_policy)

    li = sub.add_parser("lineage", help="trace one item from imported row to verdict (each step recomputable, formula + source)")
    li.add_argument("item", nargs="?", default=None, help="item id or a substring; if omitted, a changed item is chosen")
    li.add_argument("--benchmark", required=True, help="benchmark name, e.g. 'MMLU::college_chemistry'")
    li.add_argument("--sources", required=True, help="the same JSON source spec used to build the site")
    li.add_argument("--changed", action="store_true", help="when no item is given, require one whose key was corrected")
    li.add_argument("--json", action="store_true", help="emit the thread as JSON instead of text")
    li.set_defaults(func=_lineage)

    ic = sub.add_parser("import-corrections", help="import a corrections source (MMLU-Redux) into canonical rows, with a transform journal")
    ic.add_argument("infile", help="path to a JSON list or JSONL of raw rows")
    ic.add_argument("--format", default="mmlu-redux", help="source format: mmlu-redux or platinum")
    ic.add_argument("--subject", default=None, help="subject override if rows lack a subject field")
    ic.add_argument("--out", default="imported", help="output directory")
    ic.add_argument("--journal", action="store_true", help="print an excerpt of the transform journal")
    ic.set_defaults(func=_import_corrections)

    ip = sub.add_parser("import-predictions", help="import model predictions (lm-eval, HELM, or Inspect) into canonical rows; optionally join corrections")
    ip.add_argument("--samples", action="append", metavar="MODEL=PATH",
                    help="a model's samples/scenario_state/.eval file; repeat for several models")
    ip.add_argument("--format", default="lm-eval", help="source format: lm-eval (also Open LLM Leaderboard details), helm, or inspect (.eval or dumped JSON)")
    ip.add_argument("--answers", default="mcq", choices=["mcq", "freeform"], help="mcq (A-D letters) or freeform (answer strings, e.g. GSM8K-Platinum); lm-eval only")
    ip.add_argument("--subject", default=None, help="subject override if samples lack a subject field")
    ip.add_argument("--corrections", default=None, help="a directory of imported corrections to join (applies the gold decision, drops non-scorable items)")
    ip.add_argument("--out", default="imported", help="output directory")
    ip.add_argument("--journal", action="store_true", help="print an excerpt of the decode journal")
    ip.set_defaults(func=_import_predictions)

    ls = sub.add_parser("lineage-site", help="generate the lineage drill-down page (every corrected item, raw→verdict) from sources")
    ls.add_argument("--sources", required=True, help="the same JSON source spec used to build the site")
    ls.add_argument("--out", default="lineage.html", help="output HTML file (or a directory)")
    ls.add_argument("--iters", type=int, default=2000, help="bootstrap iterations for the ranking step")
    ls.set_defaults(func=_lineage_site)

    at = sub.add_parser("atlas", help="fragility atlas across benchmarks: P(top-1 change) per cell + a test of the conditional law")
    at.add_argument("--sources", required=True, help="the same JSON source spec used to build the site")
    at.add_argument("--iters", type=int, default=2000, help="bootstrap iterations per cell")
    at.add_argument("--boot", type=int, default=2000, help="bootstrap iterations for the law's correlation CIs")
    at.add_argument("--top", type=int, default=None, help="show only the N most fragile cells")
    at.set_defaults(func=_atlas)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
