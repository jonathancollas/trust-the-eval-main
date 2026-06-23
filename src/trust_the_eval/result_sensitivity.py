"""Result sensitivity to label correction — the honest 'can I trust this result?'
number, replacing a context-free per-benchmark trust score.

Given a set of models' per-item predictions on a benchmark, the ORIGINAL gold,
and a CORRECTED gold (e.g. from MMLU-Redux annotations), this measures how much
the *result* actually moves when the known label errors are fixed:

  * Delta-score per model (with bootstrap CI) — always informative;
  * Delta-ranking (Kendall tau, inversions, which models move, P(top-1 flips)) —
    informative only to the extent the compared models are close;
  * skill-discrimination — whether the errors systematically favour weaker or
    stronger models (the property that determines whether they threaten a
    ranking at all).

Empirical basis (criterion study on HELM x MMLU-Redux, 10 models / 57 subjects):
correcting labels left the *global* MMLU ranking unchanged (tau = 1.0) yet moved
*per-subject* rankings in proportion to the label-error rate (Spearman ~0.53),
because the errors are moderately skill-discriminating (Spearman ~0.49). A single
per-benchmark badge cannot express that; this function does.

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def kendall_inversions(order_a: Sequence[str], order_b: Sequence[str]) -> int:
    """Discordant pairs between two rankings of the same items (0 .. C(n,2))."""
    pos = {m: i for i, m in enumerate(order_b)}
    seq = [pos[m] for m in order_a]
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq))
               if seq[i] > seq[j])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None

    def _rank(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = _rank(xs), _rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denom = n * (n * n - 1)
    return 1.0 - 6.0 * d2 / denom if denom else None


def _ranking(scores: Dict[str, float]) -> List[str]:
    # deterministic: accuracy desc, then model name asc for ties
    return sorted(scores, key=lambda m: (-scores[m], m))


def ranking_stable(result: Dict[str, object]) -> bool:
    """Does the POINT-estimate ranking hold under correction (no inversions)?
    Bootstrap top-1 fragility is reported separately (p_top1_change)."""
    tau = result.get("kendall_tau")
    return True if tau is None else tau >= 0.99


def severity_for(result: Dict[str, object]) -> str:
    """Map a sensitivity result to a Severity string (repo vocabulary).

    HIGH   the point-estimate ranking reorders under correction, OR the top-1
           flips in >15% of bootstrap resamples
    MEDIUM ranking holds at the point estimate but the top-1 is a near-tie
           (5-15% bootstrap flips) or the errors are skill-correlated with a
           material score spread
    LOW    known errors present but the result is robust
    info   nothing changed
    """
    if result.get("n_changed_items", 0) == 0:
        return "info"
    tau = result.get("kendall_tau")
    ptop = result.get("p_top1_change")
    if tau is not None and tau < 0.99:
        return "high"
    if ptop is not None and ptop > 0.15:
        return "high"
    sd = result.get("skill_discrimination")
    deltas = [v["delta"] for v in result.get("per_model", {}).values()]
    spread = (max(deltas) - min(deltas)) if deltas else 0.0
    if (ptop is not None and ptop >= 0.05) or (sd is not None and sd > 0.3 and spread > 0.005):
        return "medium"
    return "low"


def audit_measured(result: Dict[str, object]) -> Dict[str, object]:
    """Compact, serialisable summary suitable as a ProbeAudit `measured` dict."""
    pm = result.get("per_model", {})
    deltas = sorted(v["delta"] for v in pm.values())
    med = deltas[len(deltas) // 2] if deltas else 0.0
    return {
        "derived": "label_correction_sensitivity",
        "n_models": result.get("n_models"),
        "n_items_scored": result.get("n_items_scored"),
        "n_changed_items": result.get("n_changed_items"),
        "kendall_tau": result.get("kendall_tau"),
        "tau_lo": result.get("tau_lo"), "tau_hi": result.get("tau_hi"),
        "p_top1_change": result.get("p_top1_change"),
        "ranking_stable": ranking_stable(result),
        "delta_median_pts": round(med * 100, 3),
        "delta_min_pts": round(deltas[0] * 100, 3) if deltas else 0.0,
        "delta_max_pts": round(deltas[-1] * 100, 3) if deltas else 0.0,
        "skill_discrimination": result.get("skill_discrimination"),
        "models_moved": result.get("models_moved"),
        "mechanism": result.get("mechanism"),
    }


def summary_line(result: Dict[str, object]) -> str:
    m = audit_measured(result)
    if m["n_changed_items"] == 0:
        return "No known label errors to correct; result unaffected."
    tau = m["kendall_tau"]
    rank = "rank-stable" if m["ranking_stable"] else "rank-fragile"
    sd = m["skill_discrimination"]
    skill = ("skill-neutral" if sd is None or abs(sd) < 0.2
             else ("skill-discriminating" if sd > 0 else "skill-inverted"))
    return ("Correcting %d known label errors over %d models: ranking tau=%.2f (%s), "
            "score %+.1f..%+.1f pts, errors %s."
            % (m["n_changed_items"], m["n_models"], tau if tau is not None else 1.0,
               rank, m["delta_min_pts"], m["delta_max_pts"], skill))


def sensitivity(predictions: Dict[str, Dict[str, Optional[str]]],
                original_gold: Dict[str, Iterable[str]],
                corrected_gold: Dict[str, Iterable[str]],
                *, iters: int = 2000, seed: int = 0) -> Dict[str, object]:
    """Measure how a result moves when labels are corrected.

    predictions: model -> {item_id -> predicted_label or None}
    original_gold / corrected_gold: item_id -> acceptable labels
    Items scored = those present in BOTH golds. A missing/None prediction counts
    as incorrect. All numbers are computed here (single source of truth).
    """
    models = sorted(predictions)
    items = [i for i in original_gold if i in corrected_gold]
    og = {i: set(original_gold[i]) for i in items}
    cg = {i: set(corrected_gold[i]) for i in items}
    changed = [i for i in items if og[i] != cg[i]]

    # correctness matrices (0/1) per model per item, under each gold
    cor_o: Dict[str, List[int]] = {}
    cor_c: Dict[str, List[int]] = {}
    for m in models:
        pm = predictions[m]
        cor_o[m] = [1 if (pm.get(i) in og[i]) else 0 for i in items]
        cor_c[m] = [1 if (pm.get(i) in cg[i]) else 0 for i in items]

    nI = len(items)
    acc_o = {m: (sum(cor_o[m]) / nI if nI else 0.0) for m in models}
    acc_c = {m: (sum(cor_c[m]) / nI if nI else 0.0) for m in models}
    ord_o, ord_c = _ranking(acc_o), _ranking(acc_c)
    n = len(models)
    C = n * (n - 1) // 2

    # bootstrap over items (shared resample => paired deltas, tau, top-1)
    rng = random.Random(seed)
    deltas: Dict[str, List[float]] = {m: [] for m in models}
    taus: List[float] = []
    top1_changes = 0
    if nI and C:
        for _ in range(iters):
            idx = [rng.randrange(nI) for _ in range(nI)]
            ao, ac = {}, {}
            for m in models:
                co, cc = cor_o[m], cor_c[m]
                so = sum(co[k] for k in idx)
                sc = sum(cc[k] for k in idx)
                ao[m] = so / nI
                ac[m] = sc / nI
                deltas[m].append((sc - so) / nI)
            bo, bc = _ranking(ao), _ranking(ac)
            taus.append(1 - 2 * kendall_inversions(bo, bc) / C)
            if bo[0] != bc[0]:
                top1_changes += 1

    def _ci(v: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not v:
            return (None, None)
        s = sorted(v)
        return (s[int(0.025 * len(s))], s[int(0.975 * len(s))])

    per_model = {}
    rank_o = {m: i + 1 for i, m in enumerate(ord_o)}
    rank_c = {m: i + 1 for i, m in enumerate(ord_c)}
    for m in models:
        lo, hi = _ci(deltas[m])
        per_model[m] = {
            "acc_orig": acc_o[m], "acc_corr": acc_c[m],
            "delta": acc_c[m] - acc_o[m], "delta_lo": lo, "delta_hi": hi,
            "rank_orig": rank_o[m], "rank_corr": rank_c[m],
        }

    inv = kendall_inversions(ord_o, ord_c) if C else 0
    tau_ci = _ci(taus)

    # skill-discrimination on changed items: do stronger models pick the
    # corrected answer more often?
    skill_disc = None
    mech = None
    if changed:
        cidx = [items.index(i) for i in changed]
        picks_corr = {}
        ag_o = ag_c = 0.0
        cnt = 0
        for m in models:
            cc = cor_c[m]
            co = cor_o[m]
            picks_corr[m] = sum(cc[k] for k in cidx) / len(cidx)
            ag_c += sum(cc[k] for k in cidx)
            ag_o += sum(co[k] for k in cidx)
            cnt += len(cidx)
        skill_disc = spearman([acc_o[m] for m in models],
                              [picks_corr[m] for m in models])
        mech = {"agree_original_gold": ag_o / cnt, "agree_corrected_gold": ag_c / cnt}

    return {
        "n_models": n, "n_items_scored": nI, "n_changed_items": len(changed),
        "per_model": per_model,
        "ranking_orig": ord_o, "ranking_corr": ord_c,
        "kendall_tau": (1 - 2 * inv / C) if C else None,
        "inversions": inv, "n_pairs": C,
        "tau_lo": tau_ci[0], "tau_hi": tau_ci[1],
        "p_top1_change": (top1_changes / iters) if (taus) else None,
        "models_moved": [m for m in models if rank_o[m] != rank_c[m]],
        "skill_discrimination": skill_disc,
        "mechanism": mech,
    }


def build_record(benchmark: str,
                 predictions: Dict[str, Dict[str, Optional[str]]],
                 original_gold: Dict[str, Iterable[str]],
                 corrected_gold: Dict[str, Iterable[str]],
                 *, dataset_version: Optional[str] = None,
                 sources: Optional[list] = None,
                 iters: int = 2000, seed: int = 0,
                 note: Optional[str] = None,
                 extra_audits: Optional[list] = None):
    """Compute sensitivity and wrap it as a battery_run ValidityRecord.

    Returns (record, result). The record is content-addressed with an
    inputs_hash over the exact predictions+golds, so it is reproducible and
    verifiable like any other battery_run record. It carries a single
    `result_sensitivity` audit (severity + summary + measured)."""
    from . import __version__ as _v
    from .record import (Provenance, ProbeAudit, Subject, ValidityRecord,
                         _sha_of, _short, evidence_version)
    res = sensitivity(predictions, original_gold, corrected_gold, iters=iters, seed=seed)
    audit = ProbeAudit("result_sensitivity", severity_for(res), summary_line(res),
                       measured=audit_measured(res))
    canon = {
        "pred": {m: dict(sorted(predictions[m].items())) for m in sorted(predictions)},
        "og": {k: sorted(str(x) for x in original_gold[k]) for k in sorted(original_gold)},
        "cg": {k: sorted(str(x) for x in corrected_gold[k]) for k in sorted(corrected_gold)},
    }
    subj = Subject(kind="intrinsic", benchmark=benchmark, dataset_version=dataset_version)
    prov = Provenance(method="battery_run", tool_version=_v,
                      probe_set_version=_short(_sha_of(["result_sensitivity"])),
                      calibration_version=evidence_version(),
                      sources=list(sources or []), inputs_hash=_sha_of(canon), note=note)
    return ValidityRecord(subject=subj, audits=[audit] + list(extra_audits or []),
                          provenance=prov).finalize(), res
