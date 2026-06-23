"""Tests for result_sensitivity (deterministic, known-answer cases)."""
from trust_the_eval.result_sensitivity import (kendall_inversions, sensitivity,
                                               spearman)


def test_kendall_and_spearman_basics():
    assert kendall_inversions(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert kendall_inversions(["a", "b"], ["b", "a"]) == 1
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_no_change_is_identity():
    preds = {"m1": {"i1": "A", "i2": "A"}, "m2": {"i1": "A", "i2": "B"}}
    gold = {"i1": ["A"], "i2": ["A"]}
    r = sensitivity(preds, gold, gold, iters=50)
    assert r["n_changed_items"] == 0
    assert r["kendall_tau"] == 1.0
    assert r["models_moved"] == []
    assert all(abs(v["delta"]) < 1e-12 for v in r["per_model"].values())
    assert r["skill_discrimination"] is None


def test_correction_can_flip_ranking():
    # original gold makes m1 == m2 (tie -> m1 first by name); correcting i1's
    # gold from A to B makes m2 strictly better -> ranking flips.
    preds = {"m1": {"i1": "A", "i2": "B"},   # orig 1/2 ; corr: i1 A(wrong) i2 B(wrong) -> 0/2? 
             "m2": {"i1": "B", "i2": "A"}}   # orig 1/2 ; corr: i1 B(correct) i2 A(correct) -> 2/2
    og = {"i1": ["A"], "i2": ["A"]}
    cg = {"i1": ["B"], "i2": ["A"]}
    r = sensitivity(preds, og, cg, iters=50)
    assert r["n_changed_items"] == 1
    assert r["ranking_orig"] == ["m1", "m2"]      # tie broken by name
    assert r["ranking_corr"] == ["m2", "m1"]      # m2 now strictly ahead
    assert r["inversions"] == 1 and r["kendall_tau"] == -1.0
    assert set(r["models_moved"]) == {"m1", "m2"}
    pm = r["per_model"]
    assert abs(pm["m2"]["delta"] - 0.5) < 1e-9    # m2 +0.5
    assert pm["m2"]["delta_lo"] is not None        # bootstrap CI present
    # mechanism present on the changed item
    assert r["mechanism"]["agree_corrected_gold"] >= r["mechanism"]["agree_original_gold"]


def test_severity_and_summary_helpers():
    from trust_the_eval.result_sensitivity import (audit_measured, severity_for,
                                                   summary_line)
    # rank-stable, no change -> info
    preds = {"m1": {"i1": "A"}, "m2": {"i1": "B"}}
    g = {"i1": ["A"]}
    r = sensitivity(preds, g, g, iters=20)
    assert severity_for(r) == "info"
    m = audit_measured(r)
    assert m["n_changed_items"] == 0 and m["ranking_stable"] is True
    assert "unaffected" in summary_line(r)
    # a rank-flipping change -> high
    r2 = sensitivity({"m1": {"i1": "A", "i2": "B"}, "m2": {"i1": "B", "i2": "A"}},
                     {"i1": ["A"], "i2": ["A"]}, {"i1": ["B"], "i2": ["A"]}, iters=20)
    assert severity_for(r2) == "high"  # top-1 flips -> not rank-stable
    assert "tau=" in summary_line(r2)
    # stronger model (m_strong) picks the corrected answer on the changed item,
    # weaker does not -> positive skill-discrimination.
    preds = {
        "m_strong": {"i1": "A", "i2": "A", "i3": "B"},  # orig 2/3
        "m_weak":   {"i1": "C", "i2": "C", "i3": "C"},  # orig 0/3
    }
    og = {"i1": ["A"], "i2": ["A"], "i3": ["A"]}
    cg = {"i1": ["A"], "i2": ["A"], "i3": ["B"]}         # i3 corrected to B
    r = sensitivity(preds, og, cg, iters=50)
    assert r["n_changed_items"] == 1
    sd = r["skill_discrimination"]
    assert sd is not None and sd > 0  # strong model picks corrected, weak doesn't
