"""Detection confrontation — can model behaviour flag label errors, and which method?

The project's honest negative result was that *naive* behavioural detection of label
errors sits near chance. This bench turns that into a comparison: several distinct
detector families are scored against the MMLU-Redux ground truth (the ``error_type``
labels), with ROC-AUC, bootstrap CIs, and precision@k — so we can see which, if any,
beats chance, and whether ability-weighting (the intuition behind IRT mislabel
indicators) actually helps.

Detectors (higher score = more likely a label error), all read from the same
predictions:

  * ``disagreement``       — fraction of models whose answer differs from the gold
                             (the naive behavioural baseline).
  * ``ability_weighted``   — the same disagreement, weighted by each model's overall
                             skill: strong models disagreeing counts more (a robust,
                             always-defined stand-in for the IRT mislabel signal).
  * ``consensus_disagree`` — models converge on a *single* non-gold answer
                             (plurality-off-gold minus gold share).
  * ``answer_entropy``     — spread of the answers (a hard/ambiguous-item signal; kept
                             as a contrast — it should detect ambiguity, not wrong keys).

This audits the instrument: the label under test is the benchmark item's key, never a
model. No aggregate score. Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .calibration.realworld import DEFECT_ERROR_TYPES, canonical_error_type
from .lineage import _join_error_type

LETTERS = "ABCD"


def _agree(pred: Any, gold: Any) -> bool:
    return pred in set(gold or [])


def _model_ability(pred_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Each model's overall agreement-with-gold rate (a skill proxy), over all items."""
    hit: Dict[str, int] = {}
    tot: Dict[str, int] = {}
    for r in pred_rows:
        gold = r.get("original_gold") or []
        for m, p in (r.get("preds") or {}).items():
            tot[m] = tot.get(m, 0) + 1
            if _agree(p, gold):
                hit[m] = hit.get(m, 0) + 1
    return {m: (hit.get(m, 0) / tot[m]) for m in tot if tot[m]}


def _item_scores(preds: Dict[str, str], gold: List[str],
                 ability: Dict[str, float]) -> Dict[str, Optional[float]]:
    models = [m for m in preds if m in ability]
    n = len(models)
    if n == 0:
        return {k: None for k in ("disagreement", "ability_weighted",
                                  "consensus_disagree", "answer_entropy")}
    disagree = {m: (0.0 if _agree(preds[m], gold) else 1.0) for m in models}
    disagreement = sum(disagree.values()) / n

    wsum = sum(ability[m] for m in models)
    ability_weighted = (sum(ability[m] * disagree[m] for m in models) / wsum) if wsum > 0 else disagreement

    counts: Dict[str, int] = {}
    for m in models:
        counts[preds[m]] = counts.get(preds[m], 0) + 1
    gold_share = sum(counts.get(g, 0) for g in set(gold or [])) / n
    off_gold = [c for a, c in counts.items() if a not in set(gold or [])]
    plurality_off = (max(off_gold) / n) if off_gold else 0.0
    consensus_disagree = plurality_off - gold_share

    ent = 0.0
    for a, c in counts.items():
        pr = c / n
        ent -= pr * math.log(pr, 2)
    answer_entropy = ent / math.log(len(LETTERS), 2)

    return {"disagreement": disagreement, "ability_weighted": ability_weighted,
            "consensus_disagree": consensus_disagree, "answer_entropy": answer_entropy}


def _labeled_scores(cells: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
                    positive: set) -> Tuple[List[int], Dict[str, List[float]]]:
    """Pool per-cell labelled items; label 1 if the key is a defect, 0 if ok, else skip.

    Ability is global (pooled across cells) for stability; the per-cell intrinsic index
    is used for the join, so there is no cross-subject collision.
    """
    ability = _model_ability([r for pr, _ in cells.values() for r in pr])
    labels: List[int] = []
    methods: Dict[str, List[float]] = {}
    for _name, (pred_rows, intr_rows) in cells.items():
        for r in pred_rows:
            et = canonical_error_type(_join_error_type(str(r["item"]), intr_rows))
            if et in positive:
                lab = 1
            elif et == "ok":
                lab = 0
            else:
                continue
            sc = _item_scores(r.get("preds") or {}, r.get("original_gold") or [], ability)
            if any(v is None for v in sc.values()):
                continue
            labels.append(lab)
            for k, v in sc.items():
                methods.setdefault(k, []).append(v)
    return labels, methods


def roc_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties."""
    n = len(labels)
    npos = sum(labels)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    u = sum_pos - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def _auc_ci(labels: List[int], scores: List[float], iters: int, seed: int
            ) -> Tuple[Optional[float], Optional[float]]:
    import random
    n = len(labels)
    if n < 8:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        a = roc_auc([labels[i] for i in idx], [scores[i] for i in idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def precision_at_k(labels: List[int], scores: List[float], k: int) -> Optional[float]:
    if not labels or k <= 0:
        return None
    k = min(k, len(labels))
    top = sorted(range(len(labels)), key=lambda i: -scores[i])[:k]
    return sum(labels[i] for i in top) / k


def confront(cells: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
             *, positive: Optional[set] = None, iters: int = 2000, seed: int = 0,
             ks: Tuple[int, ...] = (50, 100)) -> Dict[str, Any]:
    """Run every detector against the ground truth; return AUC + CI + precision@k each."""
    positive = positive or set(DEFECT_ERROR_TYPES)
    labels, methods = _labeled_scores(cells, positive)
    npos, nneg = sum(labels), len(labels) - sum(labels)
    out_methods = {}
    for name, scores in methods.items():
        auc = roc_auc(labels, scores)
        out_methods[name] = {
            "auc": auc, "ci": _auc_ci(labels, scores, iters, seed),
            "prec_at_k": {k: precision_at_k(labels, scores, k) for k in ks},
        }
    ranked = sorted(out_methods, key=lambda m: (out_methods[m]["auc"] or 0.0), reverse=True)
    return {"n_pos": npos, "n_neg": nneg, "positive": sorted(positive),
            "base_rate": (npos / (npos + nneg)) if (npos + nneg) else None,
            "methods": out_methods, "ranked": ranked, "ks": list(ks)}


def _f(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def format_confront(res: Dict[str, Any]) -> str:
    out = ["detection confrontation · label-error ground truth (MMLU-Redux error_type)",
           "positives = %s" % ", ".join(res["positive"]),
           "labelled items: %d defects / %d ok  (base rate %s)"
           % (res["n_pos"], res["n_neg"], _f(res["base_rate"], 3)),
           "",
           "%-20s %7s %-16s %s"
           % ("detector", "AUC", "95% CI", "  ".join("P@%d" % k for k in res["ks"])),
           "-" * 72]
    for name in res["ranked"]:
        m = res["methods"][name]
        ci = m["ci"]
        ci_s = "n/a" if not ci or ci[0] is None else "[%s, %s]" % (_f(ci[0], 2), _f(ci[1], 2))
        pk = "  ".join(_f(m["prec_at_k"].get(k), 2) for k in res["ks"])
        out.append("%-20s %7s %-16s %s" % (name, _f(m["auc"], 3), ci_s, pk))
    out += ["", "AUC 0.5 = chance. A CI clearing 0.5 means the detector carries real signal;",
            "compare ability_weighted against the naive disagreement to see if skill-weighting helps.",
            "",
            "CAUTION — circularity: if the ground-truth labels were themselves surfaced using",
            "model disagreement (MMLU-Redux inspected items where models disagreed with the key),",
            "then a disagreement detector's AUC is inflated — it partly measures the annotation",
            "pipeline, not independent detection. Read high AUC here as an upper bound, and weigh",
            "the per-subject spread (often near chance) over the pooled figure."]
    return "\n".join(out)
