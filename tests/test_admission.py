"""Tests for trust_the_eval.admission — calibration as the admission gate (P4)."""
from collections import Counter

import trust_the_eval.probes  # noqa: F401
from trust_the_eval.admission import (
    AdmissionPolicy, AdmissionRecord, ProbeRegistry, build_admission_html,
    evaluate_probe, gate,
)
from trust_the_eval.artifact import EvalArtifact, EvalItem
from trust_the_eval.calibration.core import CalibrationCase
from trust_the_eval.finding import Finding, Severity
from trust_the_eval.probe import Probe


# ---- a genuine third-party contribution: detect a degenerate answer key ----
class DegenerateKeyProbe(Probe):
    id = "contrib_degenerate_key"
    name = "Degenerate answer key"
    requires_model = False

    def run(self, artifact, model=None):
        golds = [(it.answer or "").strip() for it in artifact.items if (it.answer or "").strip()]
        if not golds:
            return [Finding(self.id, Severity.INFO, "no gold answers")]
        share = max(Counter(golds).values()) / len(golds)
        if share >= 0.9:
            return [Finding(self.id, Severity.HIGH, "answer key {:.0f}% one label".format(share * 100), score=share)]
        if share >= 0.6:
            return [Finding(self.id, Severity.MEDIUM, "answer key skewed", score=share)]
        return [Finding(self.id, Severity.INFO, "answer key balanced", score=share)]


class AlwaysHigh(Probe):
    id = "broken_always_high"
    name = "always fires"
    requires_model = False

    def run(self, artifact, model=None):
        return [Finding(self.id, Severity.HIGH, "flag")]


class NeverFlags(Probe):
    id = "broken_never"
    name = "never fires"
    requires_model = False

    def run(self, artifact, model=None):
        return [Finding(self.id, Severity.INFO, "nothing")]


def _art(answers):
    return EvalArtifact(dataset="key", model=None,
                        items=[EvalItem(question="q", answer=a, response=None, score=None)
                               for a in answers])


def _cases(npos=6, nneg=6):
    pos = [CalibrationCase("deg{}".format(i), _art(["A"] * 20), True) for i in range(npos)]
    neg = [CalibrationCase("bal{}".format(i), _art(["A", "B", "C", "D"] * 5), False) for i in range(nneg)]
    return pos + neg


def test_builtin_probe_is_admitted():
    r = evaluate_probe("statistical_power")
    assert r.admitted and r.recall is not None and r.n_pos > 0 and r.n_neg > 0


def test_contributed_probe_passes_the_gate():
    r = evaluate_probe(DegenerateKeyProbe(), cases=_cases())
    assert r.admitted
    assert r.recall >= 0.8 and r.specificity >= 0.8 and r.n_cases == 12


def test_broken_probes_are_rejected():
    high = evaluate_probe(AlwaysHigh(), cases=_cases())
    assert not high.admitted and high.specificity == 0.0
    assert any("specificity" in x for x in high.reasons)
    never = evaluate_probe(NeverFlags(), cases=_cases())
    assert not never.admitted and never.recall == 0.0
    assert any("recall" in x for x in never.reasons)


def test_conservative_policy_uses_lower_bound():
    lax = evaluate_probe("statistical_power")
    strict = evaluate_probe("statistical_power", policy=AdmissionPolicy(conservative=True))
    # same probe, but demanding the Wilson lower bound clears is stricter
    assert lax.admitted and not strict.admitted
    assert any("lower bound" in x for x in strict.reasons)


def test_admission_record_is_content_addressed_and_verifiable():
    rec = AdmissionRecord.of(evaluate_probe(DegenerateKeyProbe(), cases=_cases()))
    assert rec.record_id.startswith("sha256:") and rec.verify()
    rec.result["admitted"] = not rec.result["admitted"]
    assert not rec.verify()                       # tamper is detected


def test_probe_registry_stores_and_lists_admitted(tmp_path):
    reg = ProbeRegistry(tmp_path)
    reg.admit(DegenerateKeyProbe(), cases=_cases())
    reg.admit(AlwaysHigh(), cases=_cases())       # rejected, still recorded
    assert reg.verify_all() == []
    adm = reg.admitted()
    assert "contrib_degenerate_key" in adm and "broken_always_high" not in adm
    n = len(reg.all_ids())
    reg.put(list(adm.values())[0])                # idempotent
    assert len(reg.all_ids()) == n


def test_build_admission_html_smoke():
    results = {
        "contrib_degenerate_key": evaluate_probe(DegenerateKeyProbe(), cases=_cases()),
        "broken_always_high": evaluate_probe(AlwaysHigh(), cases=_cases()),
        "statistical_power": evaluate_probe("statistical_power"),
    }
    html = build_admission_html(results, AdmissionPolicy(), phase=("P4", "Probe admission"),
                                contributed=["contrib_degenerate_key", "broken_always_high"])
    assert "ADMIT" in html and "REJECT" in html
    assert "Admission policy" in html and "P4" in html and "__ROWS__" not in html


def test_gate_runs_over_builtins():
    g = gate()
    assert "statistical_power" in g and g["statistical_power"].admitted
