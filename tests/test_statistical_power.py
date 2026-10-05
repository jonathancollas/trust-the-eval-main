from trust_the_eval.artifact import EvalArtifact, EvalItem
from trust_the_eval.probes.statistical_power import StatisticalPower


def test_power_emits_min_gap():
    # mix of correct/incorrect so p is not degenerate (p(1-p) > 0)
    items = [EvalItem(question=f"q{i}", answer="a", score=float(i % 2)) for i in range(20)]
    f = StatisticalPower().run(EvalArtifact(dataset="t", items=items))[0]
    assert f.otel_attributes["gen_ai.eval.trust.stat.min_sig_gap_pts"] > 0
    assert 0 <= f.otel_attributes["gen_ai.eval.trust.stat.wilson_lo"] <= 1


def test_one_perfect_item_is_not_reported_as_perfectly_precise():
    art = EvalArtifact(dataset="tiny", items=[EvalItem(question="q", answer="a", score=1.0)])
    f = StatisticalPower().run(art)[0]
    assert f.severity.value == "medium"
    assert f.evidence["wilson_halfwidth"] > 0.3
    assert "+/-0.0 pts" not in f.summary
