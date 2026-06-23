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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
