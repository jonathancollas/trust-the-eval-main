"""Total per-datum transparency: per-item export, per-source datasheet, and a
recompute guarantee.

The contract is strict: every number Meridian displays must be re-derivable from
the named inputs by the *same* canonical functions that produced it. This module
does not define a single new formula. It orchestrates:

  * ``result_sensitivity.sensitivity``      -> per-model accuracy, Kendall tau, top-1
  * ``item_analysis.cronbach_terms``        -> reliability (alpha) and its terms
  * ``item_analysis.analyze``               -> per-item difficulty / discrimination
  * ``item_analysis.correctness_from_predictions`` -> the 0/1 correctness used everywhere
  * ``corpus.AnnotationCorpus.label_error`` -> label-error / ambiguity counts

``recompute_headline`` reproduces the displayed headline; ``explain`` compares it
to what ``leaderboard(store)`` shows and asserts they match (pipeline, not black
box). ``write_transparency`` emits, per benchmark, the full per-item table (JSON +
CSV) and a datasheet, so a reader can re-key the data themselves.

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .calibration.realworld import (AMBIGUITY_ERROR_TYPES, DEFECT_ERROR_TYPES,
                                    canonical_error_type)
from .corpus import AnnotationCorpus
from .item_analysis import (analyze, correctness_from_predictions,
                            cronbach_terms, reliability_summary)
from .leaderboard import _slug, leaderboard
from .result_sensitivity import sensitivity

# metric aliases -> canonical key used internally
_METRIC_ALIASES = {
    "tau": "kendall_tau", "kendall_tau": "kendall_tau", "ranking": "kendall_tau",
    "alpha": "pooled_alpha", "cronbach": "pooled_alpha", "reliability": "pooled_alpha",
    "pooled_alpha": "pooled_alpha",
    "label": "label_error", "label_error": "label_error", "label_rate": "label_error",
    "ambiguity": "ambiguity", "ambiguity_rate": "ambiguity",
    "acc": "accuracy", "accuracy": "accuracy", "per_model": "accuracy",
    "top1": "top1_changed", "top1_changed": "top1_changed",
}

METRICS = ("kendall_tau", "pooled_alpha", "label_error", "ambiguity", "top1_changed", "accuracy")


def _short(m: str) -> str:
    from .observatory_ui import _short as s   # deferred to avoid an import cycle
    return s(m)


def _pog(pred_rows: List[Dict[str, Any]]):
    """prediction rows -> (predictions, original_gold, corrected_gold, subjects)."""
    preds: Dict[str, Dict[str, Optional[str]]] = {}
    og: Dict[str, List[str]] = {}
    cg: Dict[str, List[str]] = {}
    subj: Dict[str, Any] = {}
    for r in pred_rows:
        i = str(r["item"])
        og[i] = list(r.get("original_gold") or [])
        cg[i] = list(r.get("corrected_gold") or og[i])
        if r.get("subject") is not None:
            subj[i] = r["subject"]
        for m, letter in (r.get("preds") or {}).items():
            preds.setdefault(m, {})[i] = letter
    return preds, og, cg, subj


# --------------------------------------------------------------------------- export
def per_item_records(benchmark: str, pred_rows: List[Dict[str, Any]],
                     intrinsic_rows: Optional[List[Dict[str, Any]]] = None
                     ) -> Tuple[List[str], List[Dict[str, Any]]]:
    """The atomic ground truth: one record per item with every model's raw answer,
    its correctness under BOTH gold keys, the correction flag, and the per-item
    difficulty/discrimination (the same values the dossier scatter uses).

    Correctness comes from ``correctness_from_predictions`` -- the one helper the
    reliability and discrimination probes also use -- so the export cannot diverge
    from the headline. Returns ``(models, records)``.
    """
    preds, og, cg, subj = _pog(pred_rows)
    items = [i for i in og if i in cg]
    cm_o = correctness_from_predictions(preds, {i: set(og[i]) for i in items})
    cm_c = correctness_from_predictions(preds, {i: set(cg[i]) for i in items})
    A = analyze(cm_o)
    per_item = A["per_item"]
    # optional per-item error_type, only if intrinsic rows carry a matching `item` id
    et_by: Dict[str, Any] = {}
    for r in (intrinsic_rows or []):
        if r.get("item") is not None:
            et_by[str(r["item"])] = r.get("error_type")
    models = sorted(preds)
    out: List[Dict[str, Any]] = []
    for i in items:
        et = canonical_error_type(et_by[i]) if i in et_by else None
        models_block = {m: {"pred": preds[m].get(i),
                            "correct_orig": int(cm_o[m].get(i, 0)),
                            "correct_corr": int(cm_c[m].get(i, 0))} for m in models}
        pi = per_item.get(i, {})
        out.append({
            "item": i,
            "subject": subj.get(i),
            "original_gold": sorted(set(og[i])),
            "corrected_gold": sorted(set(cg[i])),
            "changed": set(og[i]) != set(cg[i]),
            "error_type": et,
            "is_label_defect": (et in DEFECT_ERROR_TYPES) if et else None,
            "is_ambiguity": (et in AMBIGUITY_ERROR_TYPES) if et else None,
            "difficulty": pi.get("difficulty"),
            "discrimination": pi.get("discrimination"),
            "models": models_block,
        })
    return models, out


def records_to_csv(models: List[str], records: List[Dict[str, Any]]) -> str:
    """Flat, spreadsheet-friendly CSV: one row per item, three columns per model.

    Column headers use the *exact source model id* (not a shortened display name) so
    the export round-trips to the predictions it was built from and cannot collide
    when two provider-prefixed ids shorten to the same label."""
    buf = io.StringIO()
    cols = ["item", "subject", "changed", "error_type", "original_gold",
            "corrected_gold", "difficulty", "discrimination"]
    for m in models:
        cols += ["%s__pred" % m, "%s__correct_orig" % m, "%s__correct_corr" % m]
    w = csv.writer(buf)
    w.writerow(cols)
    for r in records:
        base = [r["item"], r.get("subject"), int(r["changed"]),
                r.get("error_type") or "",
                "|".join(r["original_gold"]), "|".join(r["corrected_gold"]),
                "" if r["difficulty"] is None else round(r["difficulty"], 4),
                "" if r["discrimination"] is None else round(r["discrimination"], 4)]
        for m in models:
            mb = r["models"][m]
            base += [mb["pred"] if mb["pred"] is not None else "",
                     mb["correct_orig"], mb["correct_corr"]]
        w.writerow(base)
    return buf.getvalue()


# --------------------------------------------------------- recompute / explain
def recompute_headline(benchmark: str, pred_rows: List[Dict[str, Any]],
                       intrinsic_rows: Optional[List[Dict[str, Any]]] = None,
                       *, iters: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Re-derive every displayed headline figure from the inputs, using the same
    canonical functions and the same defaults (iters/seed) the pipeline used. This
    is the single source of truth, re-run."""
    preds, og, cg, subj = _pog(pred_rows)
    out: Dict[str, Any] = {"benchmark": benchmark}
    if preds:
        res = sensitivity(preds, og, cg, iters=iters, seed=seed)
        out["accuracy"] = {_short(m): {"acc_orig": v["acc_orig"], "acc_corr": v["acc_corr"],
                                       "delta": v["delta"]}
                           for m, v in res["per_model"].items()}
        out["kendall_tau"] = res["kendall_tau"]
        out["inversions"] = res["inversions"]
        out["n_pairs"] = res["n_pairs"]
        out["n_changed"] = res["n_changed_items"]
        out["n_items"] = res["n_items_scored"]
        out["n_models"] = res["n_models"]
        out["top1_changed"] = (bool(res["ranking_orig"] and res["ranking_corr"]
                                    and res["ranking_orig"][0] != res["ranking_corr"][0]))
        items = [i for i in og if i in cg]
        cm = correctness_from_predictions(preds, {i: set(og[i]) for i in items})
        ct = cronbach_terms(cm, items)
        rel = reliability_summary(cm, subj or None)
        out["pooled_alpha"] = rel.get("pooled_alpha")
        out["cronbach"] = ct  # {k, sum_item_var, total_var, alpha}
    if intrinsic_rows:
        corpus = AnnotationCorpus.from_rows(
            intrinsic_rows, benchmark=benchmark,
            dataset_version="", source=None)
        out["label_error"] = corpus.label_error()
        out["ambiguity"] = corpus.ambiguity()
    return out


def _displayed(store: Any, benchmark: str) -> Dict[str, Any]:
    LB = leaderboard(store)
    for b in LB["benchmarks"]:
        if b["name"] == benchmark:
            return b
    raise KeyError("benchmark not in store: %s" % benchmark)


def explain(store: Any, benchmark: str, metric: str, *,
            pred_rows: List[Dict[str, Any]],
            intrinsic_rows: Optional[List[Dict[str, Any]]] = None,
            iters: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Re-derive one metric from the inputs and compare it to the displayed value.

    Returns ``{benchmark, metric, displayed, recomputed, match, formula, terms,
    inputs}``. ``match`` is the recompute guarantee: it is True iff the value the
    pipeline shows equals the value re-derived here from the named inputs.
    """
    key = _METRIC_ALIASES.get(metric, metric)
    if key not in METRICS:
        raise ValueError("unknown metric %r; one of %s" % (metric, ", ".join(sorted(_METRIC_ALIASES))))
    b = _displayed(store, benchmark)
    rc = recompute_headline(benchmark, pred_rows, intrinsic_rows, iters=iters, seed=seed)
    inputs = {"n_items": rc.get("n_items"), "n_models": rc.get("n_models"),
              "n_changed": rc.get("n_changed")}

    def close(a, b_, tol=1e-9):
        if a is None or b_ is None:
            return a == b_
        return abs(a - b_) <= tol

    if key == "kendall_tau":
        disp = (b.get("sensitivity") or {}).get("kendall_tau")
        got = rc.get("kendall_tau")
        C, D, npair = rc.get("n_pairs", 0) - rc.get("inversions", 0), rc.get("inversions", 0), rc.get("n_pairs", 0)
        terms = {"concordant_C": C, "discordant_D": D, "n_pairs": npair}
        formula = "tau = (C - D) / [n(n-1)/2]"
        match = close(disp, got)
    elif key == "pooled_alpha":
        disp = (b.get("reliability") or {}).get("pooled_alpha")
        ct = rc.get("cronbach") or {}
        got = round(ct["alpha"], 4) if ct else None
        terms = {"k": ct.get("k"), "sum_item_var": ct.get("sum_item_var"),
                 "total_var": ct.get("total_var")}
        formula = "alpha = (k/(k-1)) * (1 - sum_i var_i / var_total)"
        match = close(disp, got)
    elif key == "label_error":
        le = b.get("label_error") or {}
        disp = {"k": le.get("k"), "n": le.get("n"), "rate": le.get("rate")}
        got = rc.get("label_error") or {}
        terms = {"defect_types": sorted(DEFECT_ERROR_TYPES)}
        formula = "rate = k / n_annotated, k = #(error_type in defect set)"
        match = (disp["k"] == got.get("k") and disp["n"] == got.get("n")
                 and close(disp["rate"], got.get("rate")))
    elif key == "ambiguity":
        am = b.get("ambiguity") or {}
        disp = {"k": am.get("k"), "n": am.get("n"), "rate": am.get("rate")}
        got = rc.get("ambiguity") or {}
        terms = {"ambiguity_types": sorted(AMBIGUITY_ERROR_TYPES)}
        formula = "rate = k / n_annotated, k = #(error_type in clarity set)"
        match = (disp["k"] == got.get("k") and disp["n"] == got.get("n")
                 and close(disp["rate"], got.get("rate")))
    elif key == "top1_changed":
        # not stored in the leaderboard summary; the displayed value lives in the
        # dossier and is itself a recompute, so we report the derivation only.
        disp = None
        got = rc.get("top1_changed")
        terms = {"note": "point-estimate: ranking_orig[0] vs ranking_corr[0]"}
        formula = "top1_changed = (argmax acc_orig != argmax acc_corr)"
        match = (disp is None)  # nothing to contradict
    else:  # accuracy (per-model), compared against the displayed trace rounding
        disp = None
        got = rc.get("accuracy")
        terms = {"rounded_to": 4}
        formula = "acc(model) = (# items correct) / n_items, under each gold key"
        match = got is not None
    # `comparable` is True only when the leaderboard stores a value to check against;
    # top-1 and per-model accuracy are derivation-only (shown in the dossier, not the
    # leaderboard summary), so `match` is not a claim of agreement for them.
    comparable = ((key in ("kendall_tau", "pooled_alpha") and disp is not None)
                  or (key in ("label_error", "ambiguity")
                      and isinstance(disp, dict) and disp.get("rate") is not None))
    return {"benchmark": benchmark, "metric": key, "displayed": disp,
            "recomputed": got, "match": bool(match), "comparable": bool(comparable),
            "formula": formula, "terms": terms, "inputs": inputs}


def format_explanation(e: Dict[str, Any]) -> str:
    """Human-readable derivation for the CLI."""
    L = []
    L.append("benchmark : %s" % e["benchmark"])
    L.append("metric    : %s" % e["metric"])
    L.append("inputs    : %s" % json.dumps(e["inputs"]))
    L.append("formula   : %s" % e["formula"])
    L.append("terms     : %s" % json.dumps(e["terms"], default=str))
    L.append("displayed : %s" % json.dumps(e["displayed"], default=str))
    L.append("recomputed: %s" % json.dumps(e["recomputed"], default=str))
    if e.get("comparable"):
        L.append("match     : %s" % ("YES (recompute == displayed)" if e["match"]
                                      else "NO -- MISMATCH, this is a bug to file"))
    else:
        L.append("match     : derivation only (no stored value to compare)")
    return "\n".join(L)


# ----------------------------------------------------------------- datasheet
def datasheet(benchmark: str, *, source_meta: Dict[str, Any],
              pred_rows: List[Dict[str, Any]],
              intrinsic_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A per-source datasheet (Gebru et al. 2021; Bender & Friedman 2018): provenance,
    version, license, annotation process, composition, measured stats, known limits."""
    preds, og, cg, subj = _pog(pred_rows)
    items = [i for i in og if i in cg]
    dv = source_meta.get("dataset_version") or ""
    le = am = {"rate": None, "k": 0, "n": 0}
    by_type: Dict[str, int] = {}
    if intrinsic_rows:
        corpus = AnnotationCorpus.from_rows(intrinsic_rows, benchmark=benchmark,
                                            dataset_version=dv, source=None)
        le, am = corpus.label_error(), corpus.ambiguity()
        from collections import Counter
        by_type = dict(Counter(canonical_error_type(r.get("error_type"))
                               for r in intrinsic_rows if r.get("error_type") is not None))
    is_redux = "mmlu-redux" in str(dv).lower() or "mmlu-redux" in benchmark.lower()
    lic = source_meta.get("license") or ("CC-BY-4.0 (MMLU-Redux 2.0)" if is_redux else "see source")
    pred_prov = source_meta.get("predictions_provenance") or (
        "HELM v1.3.0 per-instance predictions" if is_redux else "as provided")
    return {
        "benchmark": benchmark,
        "dataset_version": dv or None,
        "license": lic,
        "sources": source_meta.get("sources", []),
        "annotation": {
            "label_source": "human annotation" if intrinsic_rows else "n/a",
            "process": "single-pass (one annotation per item)",
            "inter_annotator_agreement": None,
            "agreement_note": ("no Cohen/Fleiss kappa or Krippendorff alpha is "
                               "available: the corrections are single-pass, so they "
                               "are candidates to verify, not adjudicated truth."),
            "predictions_provenance": pred_prov,
        },
        "composition": {
            "n_items_scored": len(items),
            "n_annotated": le.get("n"),
            "n_models": len(preds),
            "models": sorted(_short(m) for m in preds),
            "subjects": sorted({str(s) for s in subj.values()}) if subj else [],
        },
        "measured": {
            "label_error": le,
            "ambiguity": am,
            "error_type_counts": {k: by_type[k] for k in sorted(by_type)},
        },
        "intended_use": ("Auditing the measurement validity of this benchmark and the "
                         "results reported on it. NOT a model capability or safety ranking."),
        "known_limitations": [
            "Single-pass annotation: no inter-annotator agreement; corrections are candidates.",
            "Predictions are a fixed snapshot (not re-generated live).",
            "Cronbach alpha is a lower bound and noisy at ~%d respondents (models)." % len(preds),
            ("Behaviour-based label-error detection (item discrimination) is near-chance as a "
             "screen on MMLU-Redux-style data (pooled ROC-AUC ~0.55 against human flags); the "
             "corrections here come from the annotation, not from a statistical screen."),
        ],
    }


# ------------------------------------------------------------ write to a site
def write_transparency(out_dir: Any, store: Any,
                       pred_rows: Dict[str, List[Dict[str, Any]]],
                       intrinsic_rows: Dict[str, List[Dict[str, Any]]],
                       source_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """For every benchmark with predictions, write ``data/<slug>.items.json``,
    ``data/<slug>.items.csv`` and ``data/<slug>.datasheet.json`` under ``out_dir``.
    Returns a map ``{benchmark: {json, csv, datasheet, n_items}}`` of relative URLs.
    """
    data_dir = Path(out_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Dict[str, Any]] = {}
    for name, rows in pred_rows.items():
        if not rows:
            continue
        slug = _slug(name)
        irows = intrinsic_rows.get(name, [])
        models, recs = per_item_records(name, rows, irows)
        meta = source_meta.get(name, {})
        head = recompute_headline(name, rows, irows)
        payload = {
            "benchmark": name,
            "dataset_version": meta.get("dataset_version"),
            "generated_by": "trust_the_eval.transparency.per_item_records",
            "note": ("Every value in the observatory derives from these rows via the "
                     "canonical functions; re-key the data and recompute to check."),
            "headline_recomputed": {
                "kendall_tau": head.get("kendall_tau"),
                "pooled_alpha": head.get("pooled_alpha"),
                "label_error": head.get("label_error"),
                "ambiguity": head.get("ambiguity"),
                "top1_changed": head.get("top1_changed"),
            },
            "models": [_short(m) for m in models],
            "n_items": len(recs),
            "items": recs,
        }
        (data_dir / ("%s.items.json" % slug)).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (data_dir / ("%s.items.csv" % slug)).write_text(
            records_to_csv(models, recs), encoding="utf-8")
        (data_dir / ("%s.datasheet.json" % slug)).write_text(
            json.dumps(datasheet(name, source_meta=meta, pred_rows=rows,
                                 intrinsic_rows=irows), ensure_ascii=False),
            encoding="utf-8")
        index[name] = {
            "json": "data/%s.items.json" % slug,
            "csv": "data/%s.items.csv" % slug,
            "datasheet": "data/%s.datasheet.json" % slug,
            "n_items": len(recs),
        }
    return index
