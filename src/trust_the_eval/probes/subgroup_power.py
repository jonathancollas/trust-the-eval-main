from __future__ import annotations
from collections import defaultdict
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..probe import ModelClient, Probe, register
from ..stats import wilson_ci


@register
class SubgroupPower(Probe):
    """Per-subcategory subscores ('strong at X, weak at Y') are often noise: the
    aggregate n is fine but each subgroup n is tiny. Flags underpowered groups."""
    id = "subgroup_power"
    name = "Subgroup statistical power"
    paper_priority = "Principle 2"
    requires_model = False

    def __init__(self, min_n: int = 30):
        self.min_n = min_n

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        groups: dict[str, list[float]] = defaultdict(list)
        for it in artifact.items:
            cat = it.meta.get("category") or it.meta.get("topic")
            if cat and it.score is not None:
                groups[cat].append(it.score)
        if not groups:
            return [Finding(self.id, Severity.INFO,
                            "no per-item category scores; subgroup power not assessable")]
        under = []
        for cat, scores in sorted(groups.items()):
            n = len(scores)
            k = sum(1 for s in scores if s >= self.tune("correct_cutoff"))
            lo, hi = wilson_ci(k, n)
            if n < self.min_n:
                under.append({"category": cat, "n": n,
                              "acc": round(k / n, 3), "ci": [round(lo, 2), round(hi, 2)]})
        sev = (Severity.MEDIUM if len(under) >= max(1, len(groups) // 2) else
               Severity.LOW if under else Severity.LOW)
        return [Finding(self.id, sev,
                        f"{len(under)}/{len(groups)} subgroups underpowered (n<{self.min_n})",
                        score=round(len(under) / len(groups), 3),
                        otel_attributes={"gen_ai.eval.trust.subgroup.underpowered": len(under),
                                         "gen_ai.eval.trust.subgroup.total": len(groups)},
                        evidence={"underpowered": under[:8]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

SubgroupPower.DOC = ProbeDoc(
    science=(
        "Leaderboards and model cards love per-category breakdowns \u2014 'strong "
        "at algebra, weak at geometry', 'great on French, poor on Swahili'. The "
        "trap is that the AGGREGATE sample can be perfectly adequate while each "
        "SUBGROUP is tiny, so the sub-scores are mostly sampling noise and the "
        "'strengths' and 'weaknesses' reshuffle on a re-run. This is the "
        "small-n-per-cell problem, and it has two well-established warnings. In "
        "NLP evaluation, Card et al. (2020) show statistical power is widely "
        "neglected and that small test sets leave comparisons underpowered \u2014 "
        "a fortiori for a slice of an already-small set. In psychometrics, "
        "Haberman (2008) established that a sub-score is only worth reporting if "
        "it is more reliable than the total score; Sinharay's follow-up work found "
        "that, in practice, sub-scores rarely clear that bar. The lesson is the "
        "same: a per-subgroup number needs its own power before it can be read as "
        "a measurement.\n\n"
        "This probe gives each subgroup the uncertainty it deserves. Using the "
        "items' category metadata and per-item scores, it groups scores by "
        "category, computes each subgroup's accuracy and its Wilson 95% interval "
        "(Wilson, 1927 \u2014 well-behaved at small n and extreme p, unlike the "
        "naive Wald interval), and flags every subgroup with fewer than a minimum "
        "number of items (default 30) as underpowered. The score is the fraction "
        "of subgroups that are underpowered. A high fraction means most of the "
        "per-category story is noise and only the aggregate (or pooled larger "
        "groups) supports a claim.\n\n"
        "Scope and honesty: this assesses the validity of per-subgroup CLAIMS "
        "\u2014 whether sub-scores have the statistical power to be interpreted "
        "\u2014 not the safety of the model. The n<30 rule is a convention (a "
        "rule-of-thumb for the normal approximation), not a hard law: 29 is not "
        "meaningfully worse than 31, and the right threshold depends on the effect "
        "size you care about. The Wilson interval is the honest summary the probe "
        "reports for EVERY subgroup, underpowered or not, so a reader can see the "
        "actual precision rather than rely on the flag alone. It is the power-side "
        "complement of coverage_distribution (which checks how items are "
        "distributed across categories) and a per-subgroup refinement of "
        "statistical_power (which does the same for the aggregate)."
    ),
    references=[
        Reference("Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky",
                  "With Little Power Comes Great Responsibility",
                  "EMNLP", 2020, arxiv="2010.06595",
                  note="Statistical power is widely neglected in NLP and small test sets leave comparisons underpowered \u2014 the aggregate concern this probe localizes to each subgroup."),
        Reference("Haberman",
                  "When Can Subscores Have Value?",
                  "Journal of Educational and Behavioral Statistics 33(2), 204\u2013229", 2008,
                  url="https://doi.org/10.3102/1076998607302636",
                  note="Foundational psychometric result: a subscore is only worth reporting if it is more reliable than the total score \u2014 the rigorous basis for being sceptical of small per-category sub-scores."),
        Reference("Wilson",
                  "Probable Inference, the Law of Succession, and Statistical Inference",
                  "Journal of the American Statistical Association 22(158), 209\u2013212", 1927,
                  url="https://doi.org/10.1080/01621459.1927.10502953",
                  note="The Wilson score interval reported per subgroup here; accurate for the small n and extreme p typical of thin category slices."),
    ],
    math=[
        MathBlock(
            label="Per-subgroup accuracy & Wilson 95% interval",
            html=(
                '<span class="mrow">p&#770;<sub>c</sub> = k<sub>c</sub> / n<sub>c</sub> ,&nbsp;&nbsp; '
                'CI<sub>c</sub> = '
                '<span class="frac"><span class="num">p&#770;<sub>c</sub> + z&sup2;/2n<sub>c</sub> &nbsp;\u00b1&nbsp; '
                'z<span class="sqrt"><span class="rad">&radic;</span><span class="rnd">p&#770;<sub>c</sub>(1\u2212p&#770;<sub>c</sub>)/n<sub>c</sub> + z&sup2;/4n<sub>c</sub>&sup2;</span></span></span>'
                '<span class="den">1 + z&sup2;/n<sub>c</sub></span></span></span>'
            ),
            latex=r"\hat p_c=\frac{k_c}{n_c},\qquad \mathrm{CI}_c=\frac{\hat p_c+\frac{z^2}{2n_c}\pm z\sqrt{\frac{\hat p_c(1-\hat p_c)}{n_c}+\frac{z^2}{4n_c^2}}}{1+\frac{z^2}{n_c}}",
        ),
        MathBlock(
            label="Underpowered fraction (the score)",
            html=(
                '<span class="mrow">underpowered_fraction = '
                '<span class="frac"><span class="num">|{ c : n<sub>c</sub> &lt; min_n }|</span>'
                '<span class="den">|{ c }|</span></span></span>'
            ),
            latex=r"\mathrm{underpowered\_fraction}=\frac{\bigl|\{\,c:\ n_c<\mathrm{min\_n}\,\}\bigr|}{\bigl|\{\,c\,\}\bigr|}",
        ),
    ],
    terms=[
        ("c", "a category/subgroup, from item meta['category'] (or meta['topic'])"),
        ("n_c", "number of items in subgroup c that have a per-item score"),
        ("k_c", "items in c scored correct (score \u2265 0.5)"),
        ("p\u0302_c", "subgroup c's accuracy, k_c / n_c"),
        ("z", "normal quantile for the confidence level; z = 1.96 for 95%"),
        ("CI_c", "the Wilson 95% interval for subgroup c's true accuracy (reported for every subgroup)"),
        ("min_n", "minimum items for a subgroup to be considered powered (default 30, a normal-approximation rule of thumb)"),
        ("underpowered_fraction", "share of subgroups with n_c < min_n \u2014 the probe's score"),
    ],
    thresholds=[
        Threshold("underpowered fraction \u2265 0.50 (at least half the subgroups have n < min_n)", "medium",
                  "Most per-category sub-scores lack the sample size to be interpreted \u2014 treat per-subgroup 'strengths/weaknesses' as noise and report only the aggregate or pooled larger groups."),
        Threshold("some but fewer than half underpowered", "low",
                  "A minority of subgroups are thin; read those categories' Wilson intervals before drawing conclusions, but the larger subgroups support claims."),
        Threshold("no per-item category scores", "info",
                  "Items have no category labels with scores, so subgroup power cannot be assessed (need both a category and a per-item score)."),
    ],
    effect=(
        "Stop per-category sub-scores from being over-read. By attaching a Wilson "
        "confidence interval to each subgroup and flagging those below a minimum "
        "size, the probe shows which 'strong at X, weak at Y' claims are backed by "
        "enough data and which are sampling noise \u2014 so breakdowns are reported "
        "with their uncertainty instead of as point facts."
    ),
    reading=(
        "A low underpowered fraction means most subgroups have enough items that "
        "their sub-scores are interpretable (still read the intervals). A high "
        "fraction means the per-category breakdown is largely noise: the ranking "
        "of categories would reshuffle on a re-run, so cite the aggregate or pool "
        "thin categories together, and never compare two models on a small "
        "subgroup. The score is the fraction of underpowered subgroups; the "
        "evidence lists the offending categories with their n, accuracy and Wilson "
        "interval \u2014 a wide interval is the direct, threshold-free way to see a "
        "sub-score you cannot trust."
    ),
    caveats=(
        "(a) n<30 is a convention, not a law: it is a normal-approximation rule of "
        "thumb, so a subgroup just under the cut is not meaningfully worse than one "
        "just over, and the truly adequate n depends on the effect size you need "
        "to detect \u2014 read the per-subgroup Wilson interval, which the probe "
        "reports regardless of the flag. (b) Counts power, not reliability: it "
        "flags thin subgroups but does not run Haberman's added-value test, so a "
        "well-powered sub-score can still fail to beat the total score for "
        "diagnostic purposes \u2014 'enough items' is necessary, not sufficient, "
        "for a useful sub-score. (c) Independence assumed: the Wilson interval "
        "treats items as independent; clustered or near-duplicate items within a "
        "category inflate effective precision (interacts with dataset_hygiene). "
        "(d) Label/score dependent: subgroups come from category metadata and need "
        "per-item scores; without both it returns INFO, and the taxonomy's "
        "granularity sets the subgroup sizes (cf. coverage_distribution). (e) "
        "Multiplicity: many subgroups also means many comparisons, so some will "
        "look significant by chance even when powered \u2014 pair with "
        "multiplicity_cherrypick. (f) Partial-credit scores are binarized at 0.5 "
        "for the proportion."
    ),
    code_refs=["trust_the_eval.stats.wilson_ci"],
)

SubgroupPower.TUNABLES = {'min_n': {'default': 30, 'min': 1, 'max': 500, 'step': 1, 'help': 'min subgroup size to be adequately powered', 'ctor': True}, 'correct_cutoff': {'default': 0.5, 'min': 0, 'max': 1, 'step': 0.05, 'help': 'score >= this counts as correct'}}
