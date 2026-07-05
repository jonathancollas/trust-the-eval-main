"""Locks the reconciliation of the tool's logic with the fragility-atlas finding.

The atlas refuted using skill_discrimination as a fragility signal. severity_for must no
longer escalate on it — it keys only on directly measured fragility (tau, p_top1).
"""
from trust_the_eval.result_sensitivity import CONDITIONAL_LAW_STATUS, severity_for


def _result(*, tau, ptop, sd, deltas, n_changed=10):
    return {"n_changed_items": n_changed, "kendall_tau": tau, "p_top1_change": ptop,
            "skill_discrimination": sd,
            "per_model": {("m%d" % i): {"delta": d} for i, d in enumerate(deltas)}}


def test_high_discrimination_no_longer_escalates_severity():
    # the exact refuted scenario: strong skill-discrimination and a material score spread,
    # but the ranking is point-stable and the top-1 is not a bootstrap near-tie.
    r = _result(tau=1.0, ptop=0.0, sd=0.6, deltas=[0.0, 0.05])   # spread 0.05 > 0.005
    assert severity_for(r) == "low"        # was "medium" under the refuted heuristic


def test_measured_fragility_still_escalates():
    assert severity_for(_result(tau=0.5, ptop=0.0, sd=0.0, deltas=[0.0, 0.0])) == "high"   # reorders
    assert severity_for(_result(tau=1.0, ptop=0.20, sd=0.0, deltas=[0.0, 0.0])) == "high"  # top-1 flips a lot
    assert severity_for(_result(tau=1.0, ptop=0.08, sd=0.0, deltas=[0.0, 0.0])) == "medium"  # near-tie
    assert severity_for(_result(tau=1.0, ptop=0.0, sd=0.0, deltas=[0.0, 0.0])) == "low"      # robust


def test_severity_ignores_skill_discrimination_entirely():
    # holding measured fragility fixed, the verdict must not depend on skill_discrimination
    base = dict(tau=1.0, ptop=0.0, deltas=[0.0, 0.05])
    assert severity_for(_result(sd=0.9, **base)) == severity_for(_result(sd=-0.9, **base))


def test_conditional_law_status_documents_the_refutation():
    s = CONDITIONAL_LAW_STATUS.lower()
    assert "provisional" in s and "protective" in s
    assert "no longer" in s          # the code states it stopped using the signal
