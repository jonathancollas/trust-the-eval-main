from trust_the_eval.sampling import subsample
from trust_the_eval.perturb import paraphrase, numeric_renumber, reformat_templates
from trust_the_eval.grading import default_grader, robust_grader, extract_final
from trust_the_eval.cost import CachingClient, CostMeter
from trust_the_eval.model.local import HonestModel


def test_subsample_deterministic():
    items = list(range(100))
    assert subsample(items, 10, seed=1) == subsample(items, 10, seed=1)
    assert subsample(items, 10, seed=1) != subsample(items, 10, seed=2)

def test_paraphrase_changes_text_only():
    assert paraphrase("What is 2 + 2?", seed=3) != "What is 2 + 2?"

def test_numeric_renumber_recomputes_gold():
    q, g = numeric_renumber("What is 2 + 2?", "4", seed=0)
    a = int(q.split("+")[0].split()[-1]); b = int(q.split("+")[1].split()[0].rstrip("?"))
    assert "+" in q and int(g) == a + b

def test_grader_lenient_vs_robust():
    assert default_grader("blah 1 2 4 5", "4") is True       # lenient: '4' appears
    assert robust_grader("the final value is 4", "4") is True   # final number is 4
    assert robust_grader("the answer is 7", "4") is False    # final number is 7, not 4

def test_caching_client_meters_and_caches():
    m = CostMeter()
    c = CachingClient(HonestModel(), m)
    c.complete("What is 2 + 2?")
    c.complete("What is 2 + 2?")  # served from cache
    assert m.calls == 1 and m.cache_hits == 1
