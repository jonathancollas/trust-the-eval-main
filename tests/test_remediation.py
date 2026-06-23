"""Unit tests for the remediation guidance layer."""
import trust_the_eval.probes  # noqa: F401  (populates the probe registry)
from trust_the_eval.probe import all_probes
from trust_the_eval.remediation import (PROBE_IDS, REMEDIATION, TIER_CAPTION,
                                        remediation_plan,
                                        remediation_for_benchmark)

VALID_TIERS = set(TIER_CAPTION)


def _audit(pid, severity="medium", measured=None):
    return {"probe_id": pid, "severity": severity, "summary": "",
            "measured": measured or {}}


def test_registry_covers_every_probe():
    # every one of the 20 probes has a remediation mapping
    for pid in PROBE_IDS:
        assert pid in REMEDIATION, "missing remediation for %s" % pid
    # and the live registry (if populated) matches PROBE_IDS exactly
    live = {P().id for P in all_probes()}
    if live:
        assert live == set(PROBE_IDS), (live ^ set(PROBE_IDS))


def test_all_tiers_valid():
    for pid, (tier, action, rationale) in REMEDIATION.items():
        assert tier in VALID_TIERS, (pid, tier)
        assert action and rationale


def test_label_error_is_relay_only_and_never_asserts():
    tier, action, _ = REMEDIATION["label_error_audit"]
    assert tier == "relay_only"
    low = action.lower()
    assert "verify" in low and "second annotator" in low
    # must not assert a determination
    for forbidden in ("the correct answer is", "the answer is", "change the key to"):
        assert forbidden not in low
    plan = remediation_plan([_audit("label_error_audit", "high", {"k": 41})])
    act = plan["actions"][0]
    assert act["tier"] == "relay_only"
    assert act["candidates_available"] is True
    assert act["asserts_correction"] is False
    assert act["n_items_affected"] == 41


def test_structural_probes_are_instrument_fix():
    for pid in ("dataset_hygiene", "coverage_distribution", "option_order_bias",
                "prompt_format_sensitivity", "provenance_repro",
                "refusal_confound", "judge_swap"):
        assert REMEDIATION[pid][0] == "instrument_fix"


def test_weak_signals_are_investigate():
    for pid in ("contamination_perturb", "sandbagging_paired", "reward_hacking_eval"):
        assert REMEDIATION[pid][0] == "investigate"
        # caveat must say it is not a verdict
        assert "not a verdict" in TIER_CAPTION["investigate"][1]


def test_info_severity_is_not_remediated():
    plan = remediation_plan([_audit("dataset_hygiene", "info")])
    assert plan["n_actions"] == 0
    assert "No remediation needed" in plan["summary"]


def test_result_sensitivity_is_prioritiser_not_action():
    plan = remediation_plan([_audit("result_sensitivity", "medium"),
                             _audit("dataset_hygiene", "medium", {"near_dups": 3})])
    pids = [a["probe_id"] for a in plan["actions"]]
    assert "result_sensitivity" not in pids
    assert "dataset_hygiene" in pids


def test_verdict_impact_stable_vs_fragile():
    audits = [_audit("dataset_hygiene", "medium", {"near_dups": 2})]
    stable = remediation_plan(audits, sensitivity={"ranking_stable": True,
                                                   "models_moved": 0})
    assert stable["verdict_impact"]["ranking_stable"] is True
    assert "does NOT change the model ranking" in stable["verdict_impact"]["note"]
    fragile = remediation_plan(audits, sensitivity={"ranking_stable": False,
                                                    "models_moved": 3})
    assert fragile["verdict_impact"]["ranking_stable"] is False
    assert "CAN change" in fragile["verdict_impact"]["note"]
    # unknown when no sensitivity
    none = remediation_plan(audits)
    assert none["verdict_impact"]["known"] is False


def test_priority_puts_confident_fixes_first_when_ranking_fragile():
    audits = [_audit("sandbagging_paired", "high"),       # investigate
              _audit("dataset_hygiene", "low", {"near_dups": 1}),  # instrument_fix
              _audit("label_error_audit", "high", {"k": 5})]       # relay_only
    plan = remediation_plan(audits, sensitivity={"ranking_stable": False,
                                                 "models_moved": 2})
    tiers = [a["tier"] for a in plan["actions"]]
    # the weak 'investigate' signal must not be ranked above the instrument fix
    assert tiers.index("instrument_fix") < tiers.index("investigate")


def test_dedup_keeps_highest_severity():
    audits = [_audit("dataset_hygiene", "low", {"near_dups": 1}),
              _audit("dataset_hygiene", "high", {"near_dups": 9})]
    plan = remediation_plan(audits)
    assert plan["n_actions"] == 1
    assert plan["actions"][0]["severity"] == "high"
    # kept the high-severity instance's measured (its detail shows near_dups=9)
    assert "near_dups=9" in plan["actions"][0]["action"]


def test_every_action_has_required_fields():
    audits = [_audit(pid, "medium") for pid in list(PROBE_IDS)[:8]]
    plan = remediation_plan(audits)
    for a in plan["actions"]:
        for f in ("probe_id", "severity", "tier", "confidence", "action",
                  "rationale", "caveat", "asserts_correction"):
            assert f in a
        assert a["asserts_correction"] is False
