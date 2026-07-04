"""Per-datum lineage — the journey of one item from its imported row to the verdict.

This is the spine of the transparency promise: pick one item and see, step by step,
the input, the transformation rule, the exact formula, the output, and the
scientific source — every number recomputable. It assembles, never re-implements:
correctness comes from ``transparency.per_item_records`` (the same correctness the
reliability and discrimination probes use), the ranking effect from
``result_sensitivity.sensitivity``, and the correction step from the declared
``corrections`` policy.

A dropped item (``no_correct_answer`` / a known-wrong key with no correction) is not
hidden: its thread stops at the policy step and says so, so nothing disappears
silently from the chain.

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .calibration.realworld import canonical_error_type
from .corrections import DEFAULT_ACTION, DEFAULT_POLICY, _ACTION_SOURCE
from .result_sensitivity import sensitivity
from .transparency import _pog, per_item_records


@dataclass
class LineageStep:
    n: int
    name: str
    input: str
    rule: str
    formula: Optional[str]
    output: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "name": self.name, "input": self.input, "rule": self.rule,
                "formula": self.formula, "output": self.output, "source": self.source,
                "data": self.data}


@dataclass
class LineageThread:
    benchmark: str
    item: str
    scored: bool
    steps: List[LineageStep]

    def to_dict(self) -> Dict[str, Any]:
        return {"benchmark": self.benchmark, "item": self.item, "scored": self.scored,
                "steps": [s.to_dict() for s in self.steps]}


def _nq(s: Any) -> str:
    s = (str(s) if s is not None else "").lower()
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", s).strip()


def _join_error_type(item: str, intrinsic_rows: List[Dict[str, Any]]) -> Optional[str]:
    """Best-effort join from a prediction item id to its annotation's error_type.

    The item id is ``subject::<normalised-question-prefix>::<answer_idx>``; the prefix
    is a truncation of the normalised question, so we match an intrinsic row whose
    normalised question starts with it and whose answer index agrees. Robust to the
    exact truncation length. Returns the raw error_type or None if it can't be joined.
    """
    parts = item.split("::")
    if len(parts) < 3:
        return None
    try:
        idx = int(parts[-1])
    except ValueError:
        return None
    middle = "::".join(parts[1:-1])
    for r in intrinsic_rows:
        try:
            ai = int(r.get("answer", -1))
        except (TypeError, ValueError):
            ai = -1
        if ai == idx and _nq(r.get("question", "")).startswith(middle):
            return r.get("error_type")
    return None


def _action_for(error_type: Optional[str], og: List[str], cg: Optional[List[str]]) -> str:
    """The declared policy action; if the annotation didn't join, infer it from the
    observed gold change (keep / union / change)."""
    canon = canonical_error_type(error_type)
    if canon is not None:
        return DEFAULT_POLICY.get(canon, {}).get("action", DEFAULT_ACTION)
    if cg is None:
        return "drop"
    so, sc = set(og), set(cg)
    if so == sc:
        return "keep"
    if so < sc:
        return "union"
    return "change"


def lineage(benchmark: str, item: str, pred_rows: List[Dict[str, Any]],
            intrinsic_rows: Optional[List[Dict[str, Any]]] = None,
            *, iters: int = 2000, seed: int = 0,
            records: Optional[Any] = None, sens: Optional[Dict[str, Any]] = None) -> LineageThread:
    """Build the full lineage thread for one item, from the real functions.

    ``records`` (the ``per_item_records`` tuple) and ``sens`` (a ``sensitivity`` dict)
    may be passed in precomputed — they are benchmark-level, so a site generator
    computes them once per benchmark and reuses them across every item's thread.
    """
    intrinsic_rows = intrinsic_rows or []
    # locate the item (exact, else case-insensitive substring)
    row = next((r for r in pred_rows if str(r["item"]) == item), None)
    if row is None:
        cands = [r for r in pred_rows if item.lower() in str(r["item"]).lower()]
        row = cands[0] if cands else None

    # a dropped item: present in the annotation, absent from the scored predictions
    if row is None:
        et = next((r.get("error_type") for r in intrinsic_rows
                   if item.lower() in _nq(r.get("question", ""))), None)
        if et is not None:
            canon = canonical_error_type(et)
            action = DEFAULT_POLICY.get(canon, {}).get("action", DEFAULT_ACTION)
            steps = [LineageStep(
                1, "Correction policy — dropped",
                input="error_type = %s" % et,
                rule="declared policy: %s" % DEFAULT_POLICY.get(canon, {}).get("note", ""),
                formula=None,
                output="action = %s → excluded from scoring (no valid key)" % action,
                source=_ACTION_SOURCE.get(action, ""),
                data={"error_type": et, "action": action, "scored": False})]
            return LineageThread(benchmark, item, scored=False, steps=steps)
        raise KeyError("item not found in predictions or annotations: %s" % item)

    item = str(row["item"])
    og = sorted(set(row.get("original_gold") or []))
    cg = sorted(set(row.get("corrected_gold") or og))
    preds = row.get("preds") or {}
    et = _join_error_type(item, intrinsic_rows)
    action = _action_for(et, og, cg)
    changed = set(og) != set(cg)

    # per-item correctness — single source (the same the reliability/discrimination probes use)
    models, recs = records if records is not None else per_item_records(benchmark, pred_rows, intrinsic_rows)
    rec = next((r for r in recs if r["item"] == item), None)

    # benchmark-level ranking effect — single source
    if sens is None:
        p, o, c, _ = _pog(pred_rows)
        sens = sensitivity(p, o, c, iters=iters, seed=seed)
    res = sens
    pm = res["per_model"]
    ro = sorted(pm, key=lambda m: -pm[m]["acc_orig"])
    rc = sorted(pm, key=lambda m: -pm[m]["acc_corr"])

    def short(m: str) -> str:
        from .observatory_ui import _short
        return _short(m)

    steps: List[LineageStep] = []

    steps.append(LineageStep(
        1, "Imported, normalised row",
        input="raw eval files (predictions + corrections) — see the import adapter",
        rule="parse each source format into one canonical row (no formula, no judgement)",
        formula=None,
        output="item=%s · original_gold=%s · %d models answered" % (item, og, len(preds)),
        source="import adapter (every transform logged; nothing inferred)",
        data={"item": item, "original_gold": og, "n_models": len(preds)}))

    steps.append(LineageStep(
        2, "Apply the correction policy",
        input="error_type = %s · original_gold = %s" % (et or "(not joined)", og),
        rule="declared policy: %s" % DEFAULT_POLICY.get(canonical_error_type(et) or "ok", {}).get("note", "see policy table"),
        formula=None,
        output="action = %s → corrected_gold = %s%s" % (action, cg, "  (changed)" if changed else "  (unchanged)"),
        source=_ACTION_SOURCE.get(action, "") + "  Correction is a candidate, not a truth.",
        data={"error_type": et, "action": action, "corrected_gold": cg, "changed": changed}))

    if rec is not None:
        flips = [m for m in models
                 if rec["models"][m]["correct_orig"] != rec["models"][m]["correct_corr"]]
        rows_txt = []
        for m in models:
            mb = rec["models"][m]
            mark = lambda x: "1" if x else "0"
            rows_txt.append("%s: %s  @orig=%s @corr=%s" % (short(m), mb["pred"],
                            mark(mb["correct_orig"]), mark(mb["correct_corr"])))
        steps.append(LineageStep(
            3, "Score correctness — per model, both keys",
            input="each model's answer vs original_gold and corrected_gold",
            rule="a prediction is correct if it is in the gold set",
            formula="correct = 1 if pred ∈ gold else 0",
            output="%d of %d models flip ✓↔✗ on this item" % (len(flips), len(models)),
            source="exact-match scoring (classical test theory)",
            data={"per_model": {m: rec["models"][m] for m in models}, "n_flip": len(flips),
                  "detail": rows_txt}))

    # per-model accuracy (benchmark-level) — highlight models that flip on this item
    hi = [m for m in models if rec and rec["models"][m]["correct_orig"] != rec["models"][m]["correct_corr"]][:4] or ro[:3]
    acc_txt = ["%s: %.3f → %.3f (%s%.3f)" % (short(m), pm[m]["acc_orig"], pm[m]["acc_corr"],
               "+" if pm[m]["delta"] >= 0 else "", pm[m]["delta"]) for m in hi]
    steps.append(LineageStep(
        4, "Per-model accuracy (over %d scored items)" % res["n_items_scored"],
        input="correctness across all scored items, under each key",
        rule="accuracy is the proportion correct; this item is one of %d corrected" % res["n_changed_items"],
        formula="acc(model) = (# correct) / n_items",
        output=" · ".join(acc_txt),
        source="proportion correct (the metric the benchmark already reports)",
        data={"per_model_acc": {short(m): {"orig": pm[m]["acc_orig"], "corr": pm[m]["acc_corr"]} for m in hi}}))

    C = res["n_pairs"] - res["inversions"]
    tau = res["kendall_tau"]
    top_o = short(ro[0]) if ro else None
    top_c = short(rc[0]) if rc else None
    steps.append(LineageStep(
        5, "Ranking effect — Kendall τ + bootstrap",
        input="models ranked by accuracy, original key vs corrected key",
        rule="compare the two rankings; resample items for the top-1 stability",
        formula="τ = (C − D) / [n(n−1)/2]   ·   P(top-1 changes) by bootstrap",
        output="τ = (%d−%d)/%d = %.3f · #1 %s→%s · P(top-1 changes)=%.2f"
               % (C, res["inversions"], res["n_pairs"], tau if tau is not None else float("nan"),
                  top_o, top_c, res.get("p_top1_change") or 0.0),
        source="Kendall (1938) · bootstrap error bars on evals — Miller (2024)",
        data={"kendall_tau": tau, "C": C, "D": res["inversions"], "n_pairs": res["n_pairs"],
              "top1_orig": top_o, "top1_corr": top_c, "p_top1_change": res.get("p_top1_change")}))

    fragile = (top_o != top_c) or (tau is not None and tau < 0.99)
    steps.append(LineageStep(
        6, "Verdict — about the benchmark, never the model",
        input="the ranking effect above",
        rule="label errors flip a global ranking only when skill-discriminating AND models are close",
        formula=None,
        output=("rank-fragile: correcting known label errors moves the order here — read it as uncertain"
                if fragile else
                "rank-stable: the order survives correcting known label errors"),
        source="result-sensitivity is conditional (project methodology). No model is scored or judged.",
        data={"rank_fragile": bool(fragile)}))

    return LineageThread(benchmark, item, scored=True, steps=steps)


def format_lineage(thread: LineageThread) -> str:
    """Render the thread as a vertical text lineage."""
    out = ["lineage · %s · %s" % (thread.benchmark, thread.item),
           "scored: %s" % ("yes" if thread.scored else "NO (dropped by policy)"), ""]
    for s in thread.steps:
        out.append("[%d] %s" % (s.n, s.name))
        out.append("    in  : %s" % s.input)
        out.append("    rule: %s" % s.rule)
        if s.formula:
            out.append("    f   : %s" % s.formula)
        out.append("    out : %s" % s.output)
        if s.name.startswith("Score correctness") and s.data.get("detail"):
            for line in s.data["detail"]:
                out.append("          %s" % line)
        out.append("    src : %s" % s.source)
        out.append("")
    return "\n".join(out)
