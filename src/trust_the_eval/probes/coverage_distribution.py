from __future__ import annotations
from collections import Counter
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..probe import ModelClient, Probe, register
from ..stats import gini
from ..grading import answer_kind, gold_choices


@register
class CoverageDistribution(Probe):
    """Check construct coverage: a benchmark dominated by one category does not
    measure the breadth it claims. Uses item metadata 'category'/'topic'."""
    id = "coverage_distribution"
    name = "Coverage / distribution"
    paper_priority = "P5"
    requires_model = False

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        cats = [it.meta.get("category") or it.meta.get("topic") for it in artifact.items]
        cats = [c for c in cats if c]
        if cats and len(set(cats)) > 1:
            counts = Counter(cats)
            n = sum(counts.values())
            share = {k: v / n for k, v in counts.most_common()}
            top_cat, top_share = counts.most_common(1)[0][0], counts.most_common(1)[0][1] / n
            g = gini(list(counts.values()))
            sev = (Severity.MEDIUM if top_share > self.tune("top_share_hi") or g > self.tune("gini_hi") else Severity.LOW)
            return [Finding(self.id, sev,
                            f"{len(counts)} categories; top '{top_cat}' = {top_share:.0%}; gini {g:.2f}",
                            score=round(g, 3),
                            otel_attributes={"gen_ai.eval.trust.coverage.gini": round(g, 3),
                                             "gen_ai.eval.trust.coverage.top_share": round(top_share, 3)},
                            evidence={"basis": "category_metadata",
                                      "shares": {k: round(s, 3) for k, s in list(share.items())[:10]}})]
        key = Counter()
        for it in artifact.items:
            if answer_kind(it.answer) == "mcq":
                cs = gold_choices(it.answer)
                if len(cs) == 1:
                    key[next(iter(cs))] += 1
        mcq = sum(key.values())
        if mcq >= self.tune("mcq_min"):
            share = {k: v / mcq for k, v in key.most_common()}
            top_opt, top_share = key.most_common(1)[0][0], key.most_common(1)[0][1] / mcq
            g = gini(list(key.values()))
            sev = Severity.MEDIUM if top_share > self.tune("ans_key_share_hi") else Severity.LOW
            return [Finding(self.id, sev,
                            f"no category metadata; answer-key balance over {mcq} MCQ items: "
                            f"most-common option '{top_opt}' = {top_share:.0%} (gini {g:.2f})",
                            score=round(top_share, 3),
                            otel_attributes={"gen_ai.eval.trust.coverage.key_top_share": round(top_share, 3),
                                             "gen_ai.eval.trust.coverage.key_gini": round(g, 3)},
                            evidence={"basis": "answer_key", "mcq_items": mcq,
                                      "key_shares": {k: round(s, 3) for k, s in share.items()}})]
        return [Finding(self.id, Severity.INFO,
                        "no item category metadata and no MCQ answer key; coverage not assessable",
                        evidence={"basis": None})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

CoverageDistribution.DOC = ProbeDoc(
    science=(
        "A benchmark's headline score is implicitly read as a measure of the whole "
        "construct it names \u2014 'math ability', 'medical knowledge', 'reasoning' "
        "\u2014 but that reading is only valid if the items actually SPAN the "
        "construct. When a few categories dominate the item pool, the aggregate "
        "score is a weighted average that mostly reflects those categories, and "
        "the model's competence (or incompetence) on the under-represented ones is "
        "invisible. This is a construct-validity problem: Raji et al. (2021) argue "
        "that influential benchmarks are routinely framed as 'general' measures of "
        "progress they cannot support, because their construction does not cover "
        "the breadth claimed. Holistic evaluation responds by making coverage "
        "explicit \u2014 HELM (Liang et al., 2022) deliberately spans 42 scenarios "
        "across domains precisely so that breadth is measured rather than assumed. "
        "And the problem is widespread: Bean et al. (2025), reviewing 445 LLM "
        "benchmarks, find recurring validity failures in what phenomena are "
        "actually measured, and recommend reporting coverage rather than a single "
        "aggregate.\n\n"
        "This probe gives a fast, static read on coverage skew using the items' "
        "own category metadata ('category' or 'topic'). It counts items per "
        "category and summarizes the imbalance two ways: the share of the single "
        "largest category (how much one bucket dominates) and the Gini coefficient "
        "of the category counts (how unequal the whole distribution is, 0 = "
        "perfectly even, \u2192 1 = all mass in one category). High values mean the "
        "aggregate score is dominated by a subset of the construct and that "
        "per-category reporting is needed before the number is read as a measure "
        "of the whole.\n\n"
        "Scope and honesty: this assesses the validity of the eval's CONSTRUCT "
        "COVERAGE \u2014 whether the score represents the breadth it implies \u2014 "
        "not the safety of the model. It is purely structural: it measures the "
        "balance of the LABELLED categories you give it and says nothing about "
        "whether those categories are the RIGHT decomposition of the construct (a "
        "perfectly even split over a badly-chosen taxonomy still has poor construct "
        "validity), nor whether any category has enough items to be statistically "
        "meaningful (that is subgroup_power). With no category metadata it falls back to auditing the multiple-choice ANSWER-KEY balance (the distribution of gold option letters) "
        "-- itself a validity signal that interacts with selection bias (Zheng et al., 2023); only when neither categories nor an MCQ key are present is it INFO."
    ),
    references=[
        Reference("Raji, Bender, Paullada, Denton, Hanna",
                  "AI and the Everything in the Whole Wide World Benchmark",
                  "NeurIPS (Datasets & Benchmarks)", 2021, arxiv="2111.15366",
                  note="Position paper on construct validity: influential benchmarks are framed as 'general' measures they cannot support because they do not cover the claimed breadth \u2014 the failure this probe screens for."),
        Reference("Liang, Bommasani, Lee, Tsipras, Soylu, et al.",
                  "Holistic Evaluation of Language Models (HELM)",
                  "TMLR", 2022, arxiv="2211.09110",
                  note="Makes coverage a first-class goal by spanning 42 scenarios across domains \u2014 the methodological argument for measuring breadth instead of assuming it."),
        Reference("Bean, Kearns, Romanou, Hafner, Mayne, et al.",
                  "Measuring what Matters: Construct Validity in Large Language Model Benchmarks",
                  "arXiv", 2025, arxiv="2511.04703",
                  note="Systematic review of 445 LLM benchmarks finding pervasive construct-validity issues, with recommendations to report coverage rather than a single aggregate."),
    ],
    math=[
        MathBlock(
            label="Top-category share & category shares",
            html=(
                '<span class="mrow">top_share = '
                '<span class="frac"><span class="num">max<sub>c</sub> n<sub>c</sub></span><span class="den">N</span></span>'
                ' ,&nbsp;&nbsp; share(c) = n<sub>c</sub> / N ,&nbsp; N = &Sigma;<sub>c</sub> n<sub>c</sub></span>'
            ),
            latex=r"\mathrm{top\_share}=\frac{\max_c n_c}{N},\qquad \mathrm{share}(c)=\frac{n_c}{N},\quad N=\sum_c n_c",
        ),
        MathBlock(
            label="Gini coefficient of category counts (the score)",
            html=(
                '<span class="mrow">G = '
                '<span class="frac"><span class="num">&Sigma;<sub>i</sub> &Sigma;<sub>j</sub> |n<sub>i</sub> &minus; n<sub>j</sub>|</span>'
                '<span class="den">2 k &Sigma;<sub>i</sub> n<sub>i</sub></span></span></span>'
            ),
            latex=r"G=\frac{\sum_{i}\sum_{j}\lvert n_i-n_j\rvert}{2k\sum_i n_i}\in[0,1),\quad k=\#\text{categories}",
        ),
    ],
    terms=[
        ("category", "each item's label from meta['category'] (or meta['topic']); items without one are excluded"),
        ("k", "number of distinct categories present"),
        ("n_c", "number of items in category c"),
        ("N", "total number of categorized items (\u03a3 n_c)"),
        ("share(c)", "n_c / N \u2014 fraction of items in category c"),
        ("top_share", "the largest category's share \u2014 how much one bucket dominates the score"),
        ("G (gini)", "Gini coefficient of the counts {n_c}: 0 = perfectly even coverage, \u21921 = one category holds nearly all items (the probe's score)"),
    ],
    thresholds=[
        Threshold("top_share > 0.60  OR  gini > 0.60", "medium",
                  "One category holds most items, or the distribution is highly unequal \u2014 the aggregate score mostly reflects a subset of the construct; report per-category accuracy, not just the headline number."),
        Threshold("otherwise (categories present)", "low",
                  "Coverage is reasonably balanced across the labelled categories on this benchmark."),
        Threshold("no categories: MCQ answer key dominated by one option (>40%)", "medium",
                  "Without categories the probe audits answer-key balance; one option holding >40% of golds is a validity flaw that interacts with model selection bias (Zheng et al.)."),
        Threshold("no categories: MCQ answer key reasonably balanced", "low",
                  "Answer-key balance looks fine over the multiple-choice items on this set."),
        Threshold("no categories and not MCQ", "info",
                  "No category/topic labels and no MCQ answer key, so coverage cannot be assessed (map a category column under Pull, or add one)."),
    ],
    effect=(
        "Ensure the headline score represents the breadth the benchmark implies, "
        "not just its largest bucket. The probe quantifies how concentrated the "
        "item pool is across categories \u2014 via the dominant category's share "
        "and the Gini of category counts \u2014 so coverage blind spots are made "
        "explicit and per-category reporting can replace a misleading single "
        "aggregate."
    ),
    reading=(
        "Low top_share and low Gini mean items are spread across categories, so the "
        "aggregate score is a fair summary of the whole construct. A high top_share "
        "means one bucket dominates (the 'reasoning' score is really the "
        "arithmetic score); a high Gini means the imbalance is pervasive across "
        "many categories. Either way, read per-category results (the shares are "
        "returned as evidence) rather than the single number, and weight or "
        "rebalance if you need a construct-level claim. The score is the Gini "
        "coefficient; top_share is reported alongside because a low Gini can still "
        "hide one oversized bucket among many small ones."
    ),
    caveats=(
        "(a) Coverage \u2260 right taxonomy: this measures BALANCE over the "
        "categories provided, not whether those categories validly decompose the "
        "construct \u2014 an even split over a poorly-chosen or coarse taxonomy "
        "still lacks construct validity (Raji et al.), which no automatic metric "
        "can certify. (b) Balance \u2260 power: an even distribution can still leave "
        "every category with too few items to be meaningful \u2014 use "
        "subgroup_power for per-category sample sizes, and statistical_power for "
        "the aggregate. (c) Label-dependent and skip-prone: items lacking a "
        "category/topic field are excluded, so the shares describe only the "
        "labelled subset; a benchmark with no categories falls back to MCQ answer-key balance (or INFO if not MCQ), which is "
        "a weaker signal than per-category coverage. (d) Granularity "
        "sensitivity: Gini and top_share depend on how finely categories are "
        "defined \u2014 merging or splitting buckets changes both, so compare "
        "coverage only across consistent taxonomies. (e) Purely structural: it "
        "does not look at per-category ACCURACY (whether the model is actually "
        "worse on under-covered areas) \u2014 pair with subgroup_power to connect "
        "coverage to performance."
    ),
    code_refs=["trust_the_eval.stats.gini",
               "trust_the_eval.grading.answer_kind",
               "trust_the_eval.grading.gold_choices"],
)

CoverageDistribution.TUNABLES = {'top_share_hi': {'default': 0.6, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'top-category share > this -> MEDIUM'}, 'gini_hi': {'default': 0.6, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'gini > this -> MEDIUM'}, 'mcq_min': {'default': 10, 'min': 1, 'max': 200, 'step': 1, 'help': 'min MCQ items to check answer-key balance'}, 'ans_key_share_hi': {'default': 0.4, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'answer-key letter share > this -> MEDIUM'}}
