"""Compare two eval artifacts (A vs B): is the ranking claim real or noise?

The model-selector's question. Three honest components:

1. Score gap with uncertainty.
   - Unpaired (different items): Newcombe's MOVER interval for the difference
     of two independent proportions, built from the same Wilson intervals the
     rest of the tool uses (Newcombe 1998, method 10 analogue / square-and-add).
   - Paired (same items in both artifacts): sign analysis on discordant pairs —
     among items where exactly one side is correct, the Wilson interval on the
     share won by A. If it excludes 0.5, the gap is real at 95%.
     Paired analysis is used whenever item overlap is high; it is strictly
     more powerful than the unpaired bound.

2. Findings diff — per probe, severity in A vs B, classified
   (worse / better / same / a_only / b_only).

3. Combined ranking verdict — "A > B" (or B > A) is supportable only if
   (a) BOTH runs' ranking-relevant validity is not 'not supportable',
   (b) an aggregate score gap is quantifiable, and
   (c) the gap exceeds the noise margin.
   Anything less is 'fragile' or 'not supportable', with explicit reasons.

Correctness convention mirrors the statistical_power probe: an item counts as
correct when its recorded score >= 0.5; items without a score are excluded
(here both p and the CI use n = number of scored items).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from .artifact import EvalArtifact
from .stats import wilson_ci
from .verdict import SEV_RANK, claim_verdict

PAIRING_OVERLAP = 0.9     # fraction of the smaller artifact that must match
PAIRING_MIN = 10          # minimum matched pairs to trust paired analysis


# ----------------------------- score extraction -----------------------------

def _item_key(it) -> tuple[str, str]:
    return (it.question or "", it.answer or "")


def _correct_map(art: EvalArtifact) -> dict[tuple[str, str], bool]:
    """item key -> correctness, for scored items only (first occurrence wins)."""
    out: dict[tuple[str, str], bool] = {}
    for it in art.items:
        if it.score is None:
            continue
        k = _item_key(it)
        if k not in out:
            out[k] = it.score >= 0.5
    return out


def accuracy_block(art: EvalArtifact) -> Optional[dict]:
    scored = [it.score for it in art.items if it.score is not None]
    if not scored:
        return None
    n = len(scored)
    k = sum(1 for s in scored if s >= 0.5)
    lo, hi = wilson_ci(k, n)
    return {"acc": k / n, "k": k, "n": n, "lo": lo, "hi": hi}


# ----------------------------- gap analysis -----------------------------

def newcombe_diff(a: dict, b: dict) -> dict:
    """MOVER interval for p_a - p_b from the two Wilson intervals."""
    d = a["acc"] - b["acc"]
    lo = d - math.sqrt((a["acc"] - a["lo"]) ** 2 + (b["hi"] - b["acc"]) ** 2)
    hi = d + math.sqrt((a["hi"] - a["acc"]) ** 2 + (b["acc"] - b["lo"]) ** 2)
    return {"d": d, "lo": lo, "hi": hi,
            "exceeds_noise": lo > 0 or hi < 0,
            "method": "newcombe_unpaired"}


def paired_block(map_a: dict, map_b: dict) -> Optional[dict]:
    keys = set(map_a) & set(map_b)
    n_min = min(len(map_a), len(map_b)) or 1
    if len(keys) < PAIRING_MIN or len(keys) < PAIRING_OVERLAP * n_min:
        return None
    both = neither = a_wins = b_wins = 0
    for k in keys:
        ca, cb = map_a[k], map_b[k]
        if ca and cb:
            both += 1
        elif not ca and not cb:
            neither += 1
        elif ca:
            a_wins += 1
        else:
            b_wins += 1
    m = a_wins + b_wins
    out = {"n_pairs": len(keys), "both": both, "neither": neither,
           "a_wins": a_wins, "b_wins": b_wins, "discordant": m,
           "d": (a_wins - b_wins) / len(keys), "method": "paired_sign_wilson"}
    if m == 0:
        out.update({"sign_lo": None, "sign_hi": None, "exceeds_noise": False})
        return out
    lo, hi = wilson_ci(a_wins, m)
    out.update({"sign_lo": lo, "sign_hi": hi,
                "exceeds_noise": lo > 0.5 or hi < 0.5})
    return out


# ----------------------------- findings diff -----------------------------

def _sev_of(findings: Iterable[dict], probe_id: str) -> Optional[str]:
    best = None
    for f in findings:
        if f.get("probe_id") != probe_id:
            continue
        s = str(f.get("severity", "info"))
        if best is None or SEV_RANK.get(s, 0) > SEV_RANK.get(best, 0):
            best = s
    return best


def findings_diff(findings_a: list[dict], findings_b: list[dict]) -> list[dict]:
    ids = sorted(({f.get("probe_id") for f in findings_a} |
                  {f.get("probe_id") for f in findings_b}) - {None})
    rows = []
    for pid in ids:
        sa, sb = _sev_of(findings_a, pid), _sev_of(findings_b, pid)
        ra, rb = SEV_RANK.get(sa or "", 0), SEV_RANK.get(sb or "", 0)
        if sa is None and sb is not None:
            change = "b_only"
        elif sb is None and sa is not None:
            change = "a_only"
        elif rb > ra:
            change = "worse"
        elif rb < ra:
            change = "better"
        else:
            change = "same"
        rows.append({"probe_id": pid, "sev_a": sa, "sev_b": sb, "change": change,
                     "max_rank": max(ra, rb)})
    rows.sort(key=lambda r: (-r["max_rank"], r["probe_id"]))
    return rows


# ----------------------------- combined verdict -----------------------------

def ranking_verdict(findings_a: list[dict], findings_b: list[dict],
                    gap: Optional[dict], paired: Optional[dict]) -> dict:
    """Can 'A ranks above/below B' be supported? Verdict + explicit reasons."""
    va = claim_verdict("ranking", findings_a)
    vb = claim_verdict("ranking", findings_b)
    reasons: list[str] = []
    eff = paired if paired is not None else gap

    if va["verdict"] == "uns" or vb["verdict"] == "uns":
        side = "A" if va["verdict"] == "uns" else "B"
        reasons.append(f"run {side}'s ranking-relevant validity is not supportable on its own")
        return {"verdict": "uns", "reasons": reasons, "a": va["verdict"], "b": vb["verdict"]}

    if eff is None:
        reasons.append("no comparable aggregate score could be extracted from the artifacts")
        return {"verdict": "uns", "reasons": reasons, "a": va["verdict"], "b": vb["verdict"]}

    if not eff.get("exceeds_noise"):
        if paired is not None:
            reasons.append(
                f"paired analysis: among {paired['discordant']} discordant items, "
                f"A wins {paired['a_wins']} vs {paired['b_wins']} — the Wilson interval "
                f"includes 0.5, so the gap is indistinguishable from noise")
        else:
            reasons.append(
                f"the A\u2212B gap of {gap['d']*100:+.1f} pts has 95% interval "
                f"[{gap['lo']*100:+.1f}, {gap['hi']*100:+.1f}] pts, which crosses 0 — noise")
        return {"verdict": "uns", "reasons": reasons, "a": va["verdict"], "b": vb["verdict"]}

    if va["verdict"] == "fra" or vb["verdict"] == "fra":
        for side, v in (("A", va), ("B", vb)):
            if v["verdict"] == "fra":
                tops = ", ".join(t["probe_id"] for t in v["threats"][:2]) or "threats"
                reasons.append(f"run {side} carries medium+ validity threats ({tops})")
        reasons.append("the gap exceeds noise, but fix the threats before publishing the ranking")
        return {"verdict": "fra", "reasons": reasons, "a": va["verdict"], "b": vb["verdict"]}

    reasons.append("gap exceeds the 95% noise margin and both runs are clean on ranking-relevant probes")
    return {"verdict": "sup", "reasons": reasons, "a": va["verdict"], "b": vb["verdict"]}


# ----------------------------- entry point -----------------------------

def compare(art_a: EvalArtifact, findings_a: list[dict],
            art_b: EvalArtifact, findings_b: list[dict],
            label_a: str = "A", label_b: str = "B") -> dict:
    acc_a, acc_b = accuracy_block(art_a), accuracy_block(art_b)
    map_a, map_b = _correct_map(art_a), _correct_map(art_b)
    paired = paired_block(map_a, map_b) if (acc_a and acc_b) else None
    gap = newcombe_diff(acc_a, acc_b) if (acc_a and acc_b) else None

    same_hash = art_a.content_hash() == art_b.content_hash()
    notes = []
    if same_hash:
        notes.append("identical artifact content: the score gap is zero by construction; "
                     "differences reflect probe behaviour (e.g. live-model checks), not the data")
    if paired:
        notes.append(f"paired on {paired['n_pairs']} shared items — paired analysis supersedes the unpaired bound")
    elif acc_a and acc_b:
        notes.append("items differ between artifacts: unpaired comparison (weaker; different measurements)")
    if not (acc_a and acc_b):
        notes.append("aggregate accuracy unavailable on at least one side (no per-item scores)")

    return {
        "labels": {"a": label_a, "b": label_b},
        "score": {"a": acc_a, "b": acc_b, "gap": gap, "paired": paired},
        "probes": findings_diff(findings_a, findings_b),
        "ranking": ranking_verdict(findings_a, findings_b, gap, paired),
        "comparability": {
            "same_content_hash": same_hash,
            "n_a": art_a.n, "n_b": art_b.n,
            "paired": paired is not None,
            "notes": notes,
        },
    }
