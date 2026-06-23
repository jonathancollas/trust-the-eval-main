"""Autonomous observation pipeline.

``sync()`` pulls only the sources whose fingerprint changed, runs each through
the fidelity gate, and ingests the survivors into the store (idempotently).
``observe()`` runs one full cycle: sync, then -- only if something changed --
re-run the admission gate, regenerate the public site and ``meridian.json``,
re-verify integrity, and diff against the previous export to emit a change feed.
A scheduler (cron / ``meridian watch``) calls ``observe()`` on a cadence; the
content-addressed store makes every cycle safe to repeat.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import api, integrity
from .admission import ProbeRegistry
from .calibration import has_scenarios
from .changelog import diff_exports, write_feed
from .claims import calibrate_result_probes, ingest_claims
from .corpus import AnnotationCorpus
from .intrinsic import audit_corpus, calibrate_dataset_probes
from .observatory import build_site
from .probe import all_probes
from .result_sensitivity import build_record
from .state import SourceState
from .store import RecordStore
from .validation import quarantine, validate


def sync(store: Any, sources: List[Any], state: SourceState, *, root: Any,
         calibrations_dataset: Optional[Dict[str, Any]] = None,
         calibrations_result: Optional[Dict[str, Any]] = None,
         force: bool = False) -> List[Dict[str, Any]]:
    """Pull changed sources into the store. Returns a per-source outcome list."""
    cd = calibrations_dataset if calibrations_dataset is not None else calibrate_dataset_probes()
    cr = calibrations_result if calibrations_result is not None else calibrate_result_probes()
    outcomes: List[Dict[str, Any]] = []
    for src in sources:
        try:
            fp = src.fingerprint()
        except Exception as e:                       # network / source error
            outcomes.append({"source": src.id, "status": "error",
                             "reason": "fingerprint: %s" % e})
            continue
        if not force and state.fingerprint(src.id) == fp:
            outcomes.append({"source": src.id, "status": "skipped", "fingerprint": fp})
            continue
        try:
            payload = src.fetch()
        except Exception as e:
            outcomes.append({"source": src.id, "status": "error", "reason": "fetch: %s" % e})
            continue
        ok, reasons = validate(payload, getattr(src, "anchors", {}))
        if not ok:
            quarantine(root, src.id, payload.get("fingerprint", fp), payload, reasons)
            outcomes.append({"source": src.id, "status": "quarantined", "reasons": reasons})
            continue
        before = len(store.all_ids())
        if payload["kind"] == "intrinsic":
            corpus = AnnotationCorpus.from_rows(
                payload["rows"], benchmark=payload["benchmark"],
                dataset_version=payload["dataset_version"],
                source={"title": "source: " + src.id})
            audit_corpus(corpus, calibrations=cd, store=store)
        elif payload["kind"] == "predictions":
            # multi-model per-item predictions + original/corrected gold ->
            # a result_sensitivity record (how much the result moves under
            # label correction). Rows: {item, original_gold[], corrected_gold[],
            # preds:{model: letter}}.
            preds: Dict[str, Dict[str, Any]] = {}
            og: Dict[str, Any] = {}
            cg: Dict[str, Any] = {}
            for r in payload["rows"]:
                iid = str(r["item"])
                og[iid] = list(r.get("original_gold") or [])
                cg[iid] = list(r.get("corrected_gold") or og[iid])
                for m, letter in (r.get("preds") or {}).items():
                    preds.setdefault(m, {})[iid] = letter
            # test reliability (Cronbach alpha) under the ORIGINAL gold, with
            # per-subject breakdown when rows carry a 'subject'.
            from .item_analysis import correctness_from_predictions, reliability_audit
            cm = correctness_from_predictions(preds, {i: set(v) for i, v in og.items()})
            subjects = {str(r["item"]): r["subject"] for r in payload["rows"]
                        if r.get("subject") is not None} or None
            rec, _res = build_record(
                payload["benchmark"], preds, og, cg,
                dataset_version=payload.get("dataset_version"),
                sources=[{"title": "source: " + src.id}],
                extra_audits=[reliability_audit(cm, subjects)])
            store.put(rec)
        else:
            ingest_claims(payload["rows"], store=store, benchmark=payload["benchmark"],
                          sources=[{"title": "source: " + src.id}], calibrations=cr)
        delta = len(store.all_ids()) - before
        state.update(src.id, payload.get("fingerprint", fp), kind=payload["kind"],
                     benchmark=payload["benchmark"], n_rows=len(payload["rows"]))
        outcomes.append({"source": src.id, "status": "ingested",
                         "new_records": delta, "n_rows": len(payload["rows"])})
    return outcomes


def observe(store_dir: Any, sources: List[Any], out_dir: Any,
            *, always: bool = False) -> Dict[str, Any]:
    """One observation cycle. Rebuilds only when sources changed (or always)."""
    store = RecordStore(store_dir)
    state = SourceState(store_dir)
    out = Path(out_dir)

    prev: Optional[Dict[str, Any]] = None
    pj = out / "meridian.json"
    if pj.exists():
        try:
            prev = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            prev = None

    outcomes = sync(store, sources, state, root=store_dir)
    changed = any(o["status"] == "ingested" for o in outcomes)
    if not changed and not always and prev is not None:
        return {"changed": False, "sync": outcomes}

    # admission gate -> registry (re-measured each rebuild)
    reg = ProbeRegistry(Path(store_dir) / "registry")
    for P in all_probes():
        pid = P().id
        if has_scenarios(pid):
            reg.admit(pid)

    build_site(store, out)
    # The live observatory is the rich, navigable UI built from the evals just
    # ingested (Portfolio + dossiers + computation panels + probe science). The
    # lightweight SPA is kept alongside as classic.html. Build is resilient: if the
    # rich render fails for any reason, the SPA is restored as index.html.
    try:
        idx = out / "index.html"
        if idx.exists():
            idx.replace(out / "classic.html")
        from .observatory_ui import build_ui_site
        build_ui_site(store, sources, out)
    except Exception:
        if (out / "classic.html").exists() and not (out / "index.html").exists():
            (out / "classic.html").replace(out / "index.html")
    api.write_api(store, out, registry=reg)
    rep = integrity.verify(store, reg)
    curr = api.export(store, reg)
    changes = diff_exports(prev, curr)
    feed = write_feed(changes, out, integrity=rep)
    return {"changed": True, "sync": outcomes, "integrity": rep,
            "changes": changes, "feed": feed, "out": str(out)}
