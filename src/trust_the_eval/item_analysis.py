"""Classical Test Theory item analysis as an evaluator-reliability tool.

Given a panel of models' per-item correctness on a benchmark, this measures the
*psychometric* quality of the test itself — independently of any single gold
label:

  * difficulty p_i        — share of models that get item i right;
  * discrimination r_i     — does item-correctness track overall model skill?
                             (point-biserial of item correctness vs ability);
                             a TEST-QUALITY measure: does the item separate
                             strong from weak models?
  * Cronbach's alpha       — internal-consistency reliability of the test;
  * gold_disagreement      — share of the panel that DISAGREES with the gold;
                             a screening signal for a possibly-wrong gold.

EMPIRICAL HONESTY (HELM x MMLU-Redux, 10 models, 57 subjects):
  * negative item discrimination is NOT a usable label-error screen here
    (ROC-AUC vs human-flagged errors = 0.55, ~chance). A key reason: the
    clearest wrong-gold items are ones EVERY model answers 'wrong' under the
    bad gold -> zero variance -> discrimination is undefined, not negative.
    Discrimination answers "does this item rank models?", not "is the gold
    right?".
  * panel `gold_disagreement` IS a decent screen (AUC = 0.81) but is PARTLY
    CIRCULAR: MMLU-Redux candidate errors were themselves seeded by model
    disagreement, so this measures consistency with how the flags were built,
    not independent validation.
  * pooled Cronbach alpha (0.975) is inflated by cross-subject difficulty
    spread; PER-SUBJECT reliability is low (median ~0.46, negative for some
    subjects), and with only ~10 respondents alpha is itself noisy.

Bottom line: use this for TEST RELIABILITY/quality, not as proof a gold is
wrong. The least-circular corroboration of a correction is whether the panel
picks the corrected ANSWER (see result_sensitivity.mechanism), not mere
disagreement.

Why CTT and not a 2PL IRT fit: with only ~10 models ('examinees') and hundreds
of items, per-item 2PL parameters are badly under-identified (10 binary
responses per item). CTT statistics are the robust choice at this respondent
count. `discrimination` uses a stable GLOBAL ability proxy (each model's overall
accuracy), i.e. "do better models get this item right?".

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:           # no variance (constant item or ability)
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / (sxx * syy) ** 0.5


def correctness_from_predictions(predictions: Dict[str, Dict[str, Optional[str]]],
                                 gold: Dict[str, "object"]) -> Dict[str, Dict[str, int]]:
    """predictions[model][item]=letter, gold[item]=acceptable letters ->
    correctness[model][item] in {0,1}, over items present in gold."""
    out: Dict[str, Dict[str, int]] = {}
    for m, preds in predictions.items():
        row = {}
        for i, g in gold.items():
            row[i] = 1 if preds.get(i) in set(g) else 0
        out[m] = row
    return out


def cronbach_terms(correctness: Dict[str, Dict[str, int]], items: List[str]) -> Optional[Dict[str, float]]:
    """The exact intermediate terms behind Cronbach's alpha, so a derivation can be
    shown without re-implementing the formula. Returns
    ``{k, sum_item_var, total_var, alpha}`` or ``None`` when undefined.
    Population variances across models, matching the original implementation."""
    models = list(correctness)
    k = len(items)
    if k < 2 or len(models) < 2:
        return None
    item_var = 0.0
    for i in items:
        col = [correctness[m].get(i, 0) for m in models]
        mu = sum(col) / len(col)
        item_var += sum((c - mu) ** 2 for c in col) / len(col)
    totals = [sum(correctness[m].get(i, 0) for i in items) for m in models]
    mt = sum(totals) / len(totals)
    total_var = sum((t - mt) ** 2 for t in totals) / len(totals)
    if total_var <= 0:
        return None
    alpha = (k / (k - 1)) * (1 - item_var / total_var)
    return {"k": k, "sum_item_var": item_var, "total_var": total_var, "alpha": alpha}


def cronbach_alpha(correctness: Dict[str, Dict[str, int]], items: List[str]) -> Optional[float]:
    t = cronbach_terms(correctness, items)
    return t["alpha"] if t is not None else None


def analyze(correctness: Dict[str, Dict[str, int]],
            *, suspect_threshold: float = 0.0) -> Dict[str, object]:
    """Per-item difficulty/discrimination + Cronbach alpha + suspect flags."""
    models = list(correctness)
    items = sorted({i for m in models for i in correctness[m]})
    ability = {m: (sum(correctness[m].values()) / len(correctness[m])
                   if correctness[m] else 0.0) for m in models}
    abil_vec = [ability[m] for m in models]

    per_item = {}
    suspects = []
    for i in items:
        col = [correctness[m].get(i, 0) for m in models]
        p = sum(col) / len(col)
        disc = _pearson([float(c) for c in col], abil_vec)
        susp = (disc is not None) and (disc < suspect_threshold)
        per_item[i] = {"difficulty": p, "discrimination": disc, "suspect": susp}
        if susp:
            suspects.append(i)

    return {
        "n_models": len(models), "n_items": len(items),
        "cronbach_alpha": cronbach_alpha(correctness, items),
        "ability": ability,
        "per_item": per_item,
        "suspect_items": suspects,
        "n_suspect": len(suspects),
    }


def concordance(analysis: Dict[str, object],
                human_flagged: Dict[str, bool]) -> Dict[str, object]:
    """How well does the statistical suspect signal recover externally-flagged
    (e.g. human-annotated) bad items? Reports precision/recall/F1 of the binary
    suspect flag and the ROC-AUC of the continuous score (-discrimination)
    against the human flag.
    """
    per_item = analysis["per_item"]            # type: ignore[index]
    items = [i for i in per_item if i in human_flagged]
    if not items:
        return {"n": 0}
    yh = [1 if human_flagged[i] else 0 for i in items]
    ys = [1 if per_item[i]["suspect"] else 0 for i in items]
    tp = sum(1 for k in range(len(items)) if ys[k] and yh[k])
    fp = sum(1 for k in range(len(items)) if ys[k] and not yh[k])
    fn = sum(1 for k in range(len(items)) if not ys[k] and yh[k])
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None

    # ROC-AUC of score = -discrimination (None disc -> 0, neutral) vs human flag
    score = [(-(per_item[i]["discrimination"]) if per_item[i]["discrimination"] is not None else 0.0)
             for i in items]
    pos = [score[k] for k in range(len(items)) if yh[k]]
    neg = [score[k] for k in range(len(items)) if not yh[k]]
    auc = None
    if pos and neg:
        wins = 0.0
        for a in pos:
            for b in neg:
                wins += 1.0 if a > b else (0.5 if a == b else 0.0)
        auc = wins / (len(pos) * len(neg))

    return {"n": len(items), "n_human_flagged": sum(yh), "n_suspect": sum(ys),
            "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc,
            "tp": tp, "fp": fp, "fn": fn}


def gold_disagreement(analysis: Dict[str, object]) -> Dict[str, float]:
    """Per-item share of the panel that DISAGREES with the gold (= 1 - difficulty).

    This is the label-error-relevant screen (high disagreement -> gold may be
    wrong). NOTE the circularity caveat in the module docstring: on MMLU-Redux
    this tracks the human flags (AUC ~0.81) partly because those flags were
    seeded from model disagreement. It is a screen, not independent proof.
    """
    per_item = analysis["per_item"]  # type: ignore[index]
    return {i: 1.0 - per_item[i]["difficulty"] for i in per_item}


def reliability_summary(correctness: Dict[str, Dict[str, int]],
                        subjects: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Internal-consistency reliability for the validity profile.

    Reports the POOLED Cronbach alpha and, when per-item `subjects` are given,
    the PER-SUBJECT alpha distribution (the honest figure: pooled alpha is
    inflated by cross-subject difficulty spread). Carries an explicit caveat
    that alpha is noisy at this respondent (model) count.
    """
    items = sorted({i for m in correctness for i in correctness[m]})
    k = len(correctness)
    pooled = cronbach_alpha(correctness, items)
    out: Dict[str, object] = {
        "derived": "classical_test_theory", "n_models": k, "n_items": len(items),
        "pooled_alpha": (round(pooled, 4) if pooled is not None else None),
        "per_subject": False,
        "respondent_note": "alpha is noisy at %d respondents (models)" % k,
    }
    if subjects:
        by: Dict[str, List[str]] = {}
        for i in items:
            s = subjects.get(i)
            if s is not None:
                by.setdefault(s, []).append(i)
        al = []
        for s, its in by.items():
            a = cronbach_alpha(correctness, its)
            if a is not None:
                al.append(a)
        if al:
            al.sort()
            out.update({
                "per_subject": True, "n_subjects": len(al),
                "median_alpha": round(al[len(al) // 2], 4),
                "alpha_min": round(al[0], 4), "alpha_max": round(al[-1], 4),
                "note": ("pooled alpha is inflated by cross-subject difficulty "
                         "spread; per-subject reliability is the honest figure"),
            })
    return out


def _reliability_severity(measured: Dict[str, object]) -> str:
    a = measured.get("median_alpha") if measured.get("per_subject") else measured.get("pooled_alpha")
    if a is None:
        return "info"
    return "medium" if a < 0.5 else "low"


def reliability_audit(correctness: Dict[str, Dict[str, int]],
                      subjects: Optional[Dict[str, str]] = None):
    """A `test_reliability` ProbeAudit wrapping `reliability_summary`."""
    from .record import ProbeAudit
    m = reliability_summary(correctness, subjects)
    if m.get("per_subject") and m.get("median_alpha") is not None:
        summ = ("Internal-consistency reliability: per-subject Cronbach "
                "alpha median %.2f over %d subtests (pooled %.2f, inflated). %s."
                % (m["median_alpha"], m["n_subjects"], m["pooled_alpha"], m["respondent_note"]))
    elif m.get("pooled_alpha") is not None:
        summ = ("Internal-consistency reliability: Cronbach alpha=%.2f over %d "
                "items, %d models. %s."
                % (m["pooled_alpha"], m["n_items"], m["n_models"], m["respondent_note"]))
    else:
        summ = "Reliability not computable (insufficient variance or items)."
    return ProbeAudit("test_reliability", _reliability_severity(m), summ, measured=m)
