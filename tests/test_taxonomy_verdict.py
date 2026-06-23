"""Taxonomy + per-claim verdict tests, including the JS↔Python consistency guard."""
from __future__ import annotations

import re
from pathlib import Path

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.probe import all_probes
from trust_the_eval.taxonomy import CLAIM_TYPES, REMEDIATION, claims_of
from trust_the_eval.verdict import claim_verdict, claim_verdicts

UI = Path(__file__).resolve().parents[1] / "src" / "trust_the_eval" / "ui" / "index.html"


def F(pid, sev, summary="x"):
    return {"probe_id": pid, "severity": sev, "summary": summary}


# ----------------------------- taxonomy -----------------------------

def test_every_probe_has_claims_and_remediation():
    for p in all_probes():
        assert claims_of(p.id), f"{p.id} maps to no claim"
        assert p.id in REMEDIATION and REMEDIATION[p.id].strip(), f"{p.id} has no remediation"


def test_ui_js_maps_match_python_taxonomy():
    """The UI keeps JS copies of CLAIM_TYPES and REMEDIATION for offline
    rendering; they must stay byte-identical in content to taxonomy.py."""
    html = UI.read_text(encoding="utf-8")

    js_claims_block = re.search(r"const CLAIM_TYPES = \{(.*?)\n\};", html, re.S).group(1)
    for claim, ids in CLAIM_TYPES.items():
        m = re.search(re.escape(claim) + r"\"?\s*:\s*\[(.*?)\]", js_claims_block, re.S)
        assert m, f"claim {claim} missing from UI CLAIM_TYPES"
        js_ids = set(re.findall(r'"([a-z_]+)"', m.group(1)))
        assert js_ids == set(ids), f"claim {claim}: UI={js_ids} != taxonomy={set(ids)}"

    js_rem_block = re.search(r"const REMEDIATION = \{(.*?)\n\};", html, re.S).group(1)
    js_rem = dict(re.findall(r'([a-z_]+)\s*:\s*"((?:[^"\\]|\\.)*)"', js_rem_block))
    assert set(js_rem) == set(REMEDIATION), "UI REMEDIATION keys differ from taxonomy"
    for pid, txt in REMEDIATION.items():
        assert js_rem[pid] == txt, f"remediation text differs for {pid}"


# ----------------------------- verdict rule -----------------------------

def test_high_from_reliable_probe_is_not_supportable():
    # statistical_power is structural_exact -> a HIGH from it sinks the ranking claim
    v = claim_verdict("ranking", [F("statistical_power", "high")])
    assert v["verdict"] == "uns"


def test_high_from_synthetic_floor_probe_is_only_fragile():
    # sandbagging_paired is behavioral_synthetic -> HIGH gives 'fragile', not 'uns'
    v = claim_verdict("capability", [F("sandbagging_paired", "high")])
    assert v["verdict"] == "fra"


def test_medium_threat_is_fragile_and_low_is_supportable():
    assert claim_verdict("capability", [F("contamination_perturb", "medium")])["verdict"] == "fra"
    assert claim_verdict("capability", [F("contamination_perturb", "low")])["verdict"] == "sup"


def test_irrelevant_probes_do_not_affect_a_claim():
    # contamination is not a judge-scored probe
    v = claim_verdict("judge-scored", [F("contamination_perturb", "high")])
    assert v["verdict"] == "sup" and v["n_probes"] == 0


def test_claim_verdicts_accepts_finding_objects():
    from trust_the_eval.finding import Finding, Severity
    fs = [Finding("judge_swap", Severity.HIGH, "judge order flips outcome")]
    out = claim_verdicts(fs)
    assert out["judge-scored"]["verdict"] in ("uns", "fra")
    assert out["capability"]["verdict"] == "sup"


# ----------------------------- validity-argument table -----------------------------

def test_validity_argument_covers_every_probe():
    from trust_the_eval.validity_argument import CASES, FACET_ORDER
    ids = {p.id for p in all_probes()}
    assert set(CASES) == ids, f"validity-argument CASES != probes: {set(CASES) ^ ids}"
    for pid, (facet, dual, examined, ok, bad) in CASES.items():
        assert facet in FACET_ORDER, f"{pid}: bad facet {facet}"
        assert dual is None or dual in FACET_ORDER, f"{pid}: bad dual {dual}"
        assert examined and ok and bad, f"{pid}: empty argument cell"


def test_validity_argument_renders_with_live_calibration():
    from trust_the_eval.validity_argument import render_validity_argument_md
    md = render_validity_argument_md(seed=0)
    assert md.startswith("# Trust the Eval")
    # every probe id appears, and the three evidence tiers are surfaced honestly
    for p in all_probes():
        assert f"`{p.id}`" in md
    assert "Synthetic floor" in md and "Real labels" in md and "Exact" in md
    assert "Borsboom" in md and "Kane" in md  # the limit + the honesty rule are stated
