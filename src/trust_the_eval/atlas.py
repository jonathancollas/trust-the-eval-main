"""The fragility atlas — testing the conditional law across many cells.

Claim B of the project has been an *assertion*: a benchmark's ranking is fragile
under label correction only when the corrected items discriminate skill AND the
models are close. This module turns that into a *tested regularity*. For each cell
(a benchmark / subject with its predictions and its correction overlay) it reads,
from the canonical ``sensitivity`` — never re-derived:

  * fragility            = ``p_top1_change`` (bootstrap over items)
  * discrimination       = ``skill_discrimination`` (do stronger models pick the
                           corrected answer more? a Spearman already computed there)
  * model closeness      = the #1–#2 accuracy gap under the original key (small = close)

then asks, across cells, whether fragility is predicted by discrimination and
closeness — with the honest verdict reported either way, including when it is not.

This is a within-corpus test when run on one overlay × one model set (e.g. the
57 MMLU-Redux subjects × HELM). Feed more overlays (Platinum diffs, SWE-bench
Verified, KMMLU-Redux) via the import adapters to widen it. No aggregate score is
produced; the atlas is a distribution and a regularity test, not a rating.

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import statistics as _stats
from typing import Any, Dict, List, Optional, Tuple

from .result_sensitivity import sensitivity, severity_for, spearman
from .transparency import _pog


def cell_metrics(cell: str, pred_rows: List[Dict[str, Any]],
                 intrinsic_rows: Optional[List[Dict[str, Any]]] = None,
                 *, iters: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """One row of the atlas — all quantities read from the canonical sensitivity."""
    p, o, c, _ = _pog(pred_rows)
    r = sensitivity(p, o, c, iters=iters, seed=seed)
    pm = r["per_model"]
    accs = sorted((v["acc_orig"] for v in pm.values()), reverse=True)
    gap12 = (accs[0] - accs[1]) if len(accs) >= 2 else None
    deltas = [v["delta"] for v in pm.values()]
    delta_spread = (max(deltas) - min(deltas)) if deltas else None
    acc_spread = _stats.pstdev(accs) if len(accs) >= 2 else None
    ro = r.get("ranking_orig") or []
    rc = r.get("ranking_corr") or []
    return {
        "cell": cell,
        "n_models": r["n_models"], "n_items_scored": r["n_items_scored"],
        "n_changed": r["n_changed_items"],
        "kendall_tau": r["kendall_tau"], "tau_lo": r.get("tau_lo"), "tau_hi": r.get("tau_hi"),
        "p_top1_change": r.get("p_top1_change"),
        "skill_discrimination": r.get("skill_discrimination"),
        "model_gap_12": gap12, "model_acc_spread": acc_spread, "delta_spread": delta_spread,
        "top1_moved": bool(ro and rc and ro[0] != rc[0]),
        "n_models_moved": len(r.get("models_moved") or []),
        "severity": severity_for(r),
    }


def _vec(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [r[key] for r in rows]


def _boot_corr(xs: List[float], ys: List[float], iters: int = 2000,
               seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for a Spearman correlation, resampling cells."""
    import random
    n = len(xs)
    if n < 4:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        rc = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if rc is not None:
            vals.append(rc)
    if not vals:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def conditional_law_test(rows: List[Dict[str, Any]], *, boot_iters: int = 2000,
                         seed: int = 0) -> Dict[str, Any]:
    """Across cells, does fragility track discrimination and closeness?

    Reports Spearman correlations (fragility vs each predictor) and a 2x2
    median-split of fragility by discrimination and closeness. The law predicts the
    highest fragility in the close-and-discriminating quadrant. The verdict is honest
    either way.
    """
    act = [r for r in rows if r["n_changed"] > 0 and r["p_top1_change"] is not None
           and r["model_gap_12"] is not None and r["skill_discrimination"] is not None]
    n = len(act)
    if n < 4:
        return {"n": n, "insufficient": True,
                "note": "need >= 4 active cells (changed items, >=2 models) to test the law"}
    ptop = _vec(act, "p_top1_change")
    disc = _vec(act, "skill_discrimination")
    gap = _vec(act, "model_gap_12")
    spread = _vec(act, "delta_spread")

    corr_disc = spearman(disc, ptop)     # expect > 0 : more discriminating -> more fragile
    corr_gap = spearman(gap, ptop)       # expect < 0 : bigger gap (less close) -> less fragile
    corr_spread = spearman(spread, ptop)
    ci_disc = _boot_corr(disc, ptop, iters=boot_iters, seed=seed)
    ci_gap = _boot_corr(gap, ptop, iters=boot_iters, seed=seed)
    ci_spread = _boot_corr(spread, ptop, iters=boot_iters, seed=seed)

    md_disc = _stats.median(disc)
    md_gap = _stats.median(gap)
    quad: Dict[str, List[float]] = {"close_disc": [], "close_nondisc": [],
                                    "far_disc": [], "far_nondisc": []}
    for r in act:
        d = "disc" if r["skill_discrimination"] >= md_disc else "nondisc"
        close = "close" if r["model_gap_12"] <= md_gap else "far"
        quad["%s_%s" % (close, d)].append(r["p_top1_change"])
    quad_mean = {k: (sum(v) / len(v) if v else None) for k, v in quad.items()}
    means = {k: v for k, v in quad_mean.items() if v is not None}
    law_holds = bool(means and max(means, key=lambda k: means[k]) == "close_disc"
                     and (corr_disc or 0) > 0 and (corr_gap or 0) < 0)

    return {
        "n": n, "insufficient": False,
        "corr_p_top1_vs_discrimination": corr_disc, "ci_discrimination": ci_disc,
        "corr_p_top1_vs_model_gap": corr_gap, "ci_model_gap": ci_gap,
        "corr_p_top1_vs_delta_spread": corr_spread, "ci_delta_spread": ci_spread,
        "median_discrimination": md_disc, "median_gap": md_gap,
        "quadrant_mean_p_top1": quad_mean,
        "quadrant_counts": {k: len(v) for k, v in quad.items()},
        "law_holds": law_holds,
    }


def fragility_atlas(cells: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
                    *, iters: int = 2000, seed: int = 0, boot_iters: int = 2000) -> Dict[str, Any]:
    """Build the atlas over cells: {name: (pred_rows, intrinsic_rows)}."""
    rows: List[Dict[str, Any]] = []
    for name, pair in cells.items():
        pred, intr = pair
        try:
            rows.append(cell_metrics(name, pred, intr, iters=iters, seed=seed))
        except Exception:
            continue
    rows.sort(key=lambda r: (r["p_top1_change"] if r["p_top1_change"] is not None else -1.0),
              reverse=True)
    return {"rows": rows, "law": conditional_law_test(rows, boot_iters=boot_iters, seed=seed),
            "n_cells": len(rows), "n_active": sum(1 for r in rows if r["n_changed"] > 0)}


def _f(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else ("%.*f" % (nd, x))


def format_atlas(atlas: Dict[str, Any], top: Optional[int] = None) -> str:
    rows = atlas["rows"]
    shown = rows if top is None else rows[:top]
    out = ["fragility atlas · %d cells (%d with corrections)"
           % (atlas["n_cells"], atlas["n_active"]),
           "note: within-corpus when one overlay x one model set; widen with more overlays",
           "",
           "%-26s %5s %6s %8s %7s %6s %-7s"
           % ("cell", "chg", "tau", "P(top1)", "disc", "gap12", "severity"),
           "-" * 78]
    for r in shown:
        out.append("%-26s %5d %6s %8s %7s %6s %-7s"
                   % (str(r["cell"])[:26], r["n_changed"], _f(r["kendall_tau"], 2),
                      _f(r["p_top1_change"], 2), _f(r["skill_discrimination"], 2),
                      _f(r["model_gap_12"], 3), r["severity"]))
    if top is not None and len(rows) > top:
        out.append("… %d more" % (len(rows) - top))

    law = atlas["law"]
    out += ["", "conditional law — does fragility track discrimination AND closeness?"]
    if law.get("insufficient"):
        out.append("  %s (n=%d)" % (law.get("note", "insufficient data"), law.get("n", 0)))
        return "\n".join(out)
    def _ci(t):
        return "n/a" if not t or t[0] is None else "[%s, %s]" % (_f(t[0], 2), _f(t[1], 2))
    out += [
        "  cells tested: %d  (bootstrap CI over cells)" % law["n"],
        "  corr( P(top1) , discrimination ) = %s  %s   (law expects > 0)"
        % (_f(law["corr_p_top1_vs_discrimination"]), _ci(law.get("ci_discrimination"))),
        "  corr( P(top1) , model gap #1-#2 ) = %s  %s   (law expects < 0)"
        % (_f(law["corr_p_top1_vs_model_gap"]), _ci(law.get("ci_model_gap"))),
        "  corr( P(top1) , correction spread ) = %s  %s   (partly mechanical)"
        % (_f(law["corr_p_top1_vs_delta_spread"]), _ci(law.get("ci_delta_spread"))),
        "  mean P(top1) by quadrant (median split on discrimination x closeness):",
    ]
    qm = law["quadrant_mean_p_top1"]
    qc = law["quadrant_counts"]
    labels = {"close_disc": "close + discriminating", "close_nondisc": "close + flat",
              "far_disc": "far + discriminating", "far_nondisc": "far + flat"}
    for k in ("close_disc", "close_nondisc", "far_disc", "far_nondisc"):
        out.append("    %-24s %s   (n=%d)" % (labels[k], _f(qm.get(k), 3), qc.get(k, 0)))
    out.append("  verdict: the law %s on this corpus."
               % ("HOLDS" if law["law_holds"] else "does NOT cleanly hold"))
    return "\n".join(out)
