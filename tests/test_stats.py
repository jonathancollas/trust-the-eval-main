import math
from trust_the_eval.stats import (cohens_kappa, wilson_ci, min_significant_gap,
                                   two_proportion_pvalue, shannon_entropy, gini)


def test_wilson_bounds():
    lo, hi = wilson_ci(5, 10)
    assert 0 <= lo < 0.5 < hi <= 1

def test_kappa_perfect_and_chance():
    assert abs(cohens_kappa([1,0,1,0], [1,0,1,0]) - 1.0) < 1e-9
    assert cohens_kappa([1,1,1,1], [0,0,0,0]) == 0.0

def test_min_gap_shrinks_with_n():
    assert min_significant_gap(0.5, 100) > min_significant_gap(0.5, 10000)

def test_two_proportion_pvalue_significant():
    assert two_proportion_pvalue(90, 100, 60, 100) < 0.01

def test_entropy_and_gini():
    assert abs(shannon_entropy([1,1]) - 1.0) < 1e-9
    assert gini([1,1,1,1]) == 0.0
