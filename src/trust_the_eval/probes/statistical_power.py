from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..probe import ModelClient, Probe, register
from ..stats import min_significant_gap, proportion_halfwidth, wilson_ci


@register
class StatisticalPower(Probe):
    """Is the score precise, and are model rankings real or noise?"""
    id = "statistical_power"
    name = "Statistical reliability"
    paper_priority = "Principle 2"
    requires_model = False

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        n = artifact.n
        if n == 0:
            return [Finding(self.id, Severity.HIGH, "empty eval (n=0)",
                            evidence={"n": 0, "n_scored": 0, "scoring_coverage": 0.0})]
        scored = [it.score for it in artifact.items if it.score is not None]
        n_scored = len(scored)
        if n_scored == 0:
            return [Finding(self.id, Severity.MEDIUM,
                            f"n={n}: no per-item scores recorded \u2014 precision and "
                            f"ranking reliability cannot be assessed",
                            evidence={"n": n, "n_scored": 0, "scoring_coverage": 0.0})]
        k = sum(1 for sc in scored if sc >= self.tune("correct_cutoff"))
        p = k / n_scored
        half = proportion_halfwidth(p, n_scored)
        gap = min_significant_gap(p, n_scored)
        lo, hi = wilson_ci(k, n_scored)
        coverage = n_scored / n
        sev = Severity.MEDIUM if gap > self.tune("gap_medium") else Severity.LOW
        note = ""
        if coverage < self.tune("cov_high"):
            sev = Severity.HIGH
            note = (f"; only {n_scored}/{n} items scored ({coverage*100:.0f}%) \u2014 "
                    f"accuracy is over a subset and may be biased")
        elif coverage < self.tune("cov_medium"):
            if sev == Severity.LOW:
                sev = Severity.MEDIUM
            note = f"; {n - n_scored}/{n} items unscored ({coverage*100:.0f}% scored)"
        return [Finding(self.id, sev,
                        f"n_scored={n_scored}/{n}: acc {p:.2f}, 95% CI [{lo:.2f},{hi:.2f}] "
                        f"(+/-{half*100:.1f} pts); two models within "
                        f"{gap*100:.1f} pts = noise{note}",
                        score=round(gap, 3),
                        otel_attributes={"gen_ai.eval.trust.stat.ci95_pts": round(half * 100, 2),
                                         "gen_ai.eval.trust.stat.min_sig_gap_pts": round(gap * 100, 2),
                                         "gen_ai.eval.trust.stat.wilson_lo": round(lo, 3),
                                         "gen_ai.eval.trust.stat.wilson_hi": round(hi, 3),
                                         "gen_ai.eval.trust.stat.scoring_coverage": round(coverage, 3)},
                        evidence={"n": n, "n_scored": n_scored,
                                  "scoring_coverage": round(coverage, 4),
                                  "k": k, "accuracy": round(p, 4), "z": 1.96,
                                  "wald_halfwidth": round(half, 4),
                                  "min_sig_gap": round(gap, 4),
                                  "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3)})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

StatisticalPower.DOC = ProbeDoc(
    science=(
        "A benchmark score is an estimate from a finite sample, so it carries "
        "sampling uncertainty \u2014 yet leaderboards routinely report a single "
        "number to two decimals and rank models by hundredths of a point. "
        "Reporting accuracy without an interval, and declaring model A better "
        "than B on a gap smaller than noise, are basic validity failures. Card et "
        "al. (2020) show this is endemic in NLP: statistical power has been "
        "largely ignored, several GLUE tasks have test sets too small to "
        "adequately power the comparisons people run on them, and underpowered "
        "experiments both miss real effects and exaggerate the ones they do "
        "report. Dror et al. (2018) make the companion point that significance "
        "testing is under-used and often done wrong. This probe brings the "
        "minimum statistical hygiene to a single eval result.\n\n"
        "Model accuracy on n independent items is a binomial proportion. The naive "
        "Wald interval p\u0302 \u00b1 z\u221a(p\u0302(1\u2212p\u0302)/n) has poor "
        "coverage for small n or p near 0/1 (it can even leave [0,1]); Brown, Cai "
        "& DasGupta (2001) recommend the Wilson score interval (Wilson, 1927) "
        "instead, which stays in range and is accurate at the extremes. This probe "
        "reports the Wilson 95% interval for the score, plus two readouts: the "
        "Wald half-width as a familiar '\u00b1 points' precision figure, and the "
        "minimum significant gap \u2014 the smallest accuracy difference between "
        "two equal-n models that would not be explained by noise, derived from the "
        "standard error of a difference of proportions (\u221a2 times the "
        "single-proportion SE). Two models closer than that gap are a tie at this "
        "n.\n\n"
        "Scope and honesty: this is a static, model-free probe \u2014 it reads the "
        "recorded per-item scores and n, and assesses whether the NUMBER is precise "
        "enough to support the claims made about it (its precision and whether "
        "rankings are real), never the safety of the model. It deliberately uses a "
        "simple binomial model; the caveats below note where that model is "
        "optimistic (paired comparisons, clustered or non-independent items, "
        "multiple comparisons) and point to the matching probes."
    ),
    references=[
        Reference("Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky",
                  "With Little Power Comes Great Responsibility",
                  "EMNLP", 2020, arxiv="2010.06595",
                  note="Shows statistical power is largely ignored in NLP and that small test sets leave common model comparisons underpowered \u2014 the motivation for this probe."),
        Reference("Dror, Baumer, Shlomov, Reichart",
                  "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing",
                  "ACL", 2018,
                  url="https://aclanthology.org/P18-1128/",
                  note="Companion case that significance testing is under-used/mis-used in NLP evaluation; argues for principled hypothesis testing."),
        Reference("Wilson",
                  "Probable Inference, the Law of Succession, and Statistical Inference",
                  "Journal of the American Statistical Association 22(158), 209\u2013212", 1927,
                  url="https://doi.org/10.1080/01621459.1927.10502953",
                  note="The Wilson score interval used here for the accuracy CI \u2014 well-behaved for small n and extreme p."),
        Reference("Brown, Cai, DasGupta",
                  "Interval Estimation for a Binomial Proportion",
                  "Statistical Science 16(2), 101\u2013133", 2001,
                  url="https://doi.org/10.1214/ss/1009213286",
                  note="Shows the naive Wald interval has poor coverage and recommends Wilson / Agresti\u2013Coull \u2014 why this probe reports Wilson rather than Wald."),
    ],
    math=[
        MathBlock(
            label="Wilson score 95% interval for accuracy",
            html=(
                '<span class="mrow">'
                '<span class="frac"><span class="num">p&#770; + z&sup2;/2n &nbsp;\u00b1&nbsp; '
                'z<span class="sqrt"><span class="rad">&radic;</span><span class="rnd">p&#770;(1\u2212p&#770;)/n + z&sup2;/4n&sup2;</span></span></span>'
                '<span class="den">1 + z&sup2;/n</span></span></span>'
            ),
            latex=r"\mathrm{CI}_{95}=\frac{\hat p+\frac{z^2}{2n}\ \pm\ z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z=1.96",
        ),
        MathBlock(
            label="Minimum significant gap (two equal-n models)",
            html=(
                '<span class="mrow">&Delta;<sub>min</sub> = '
                'z&nbsp;<span class="sqrt"><span class="rad">&radic;</span><span class="rnd">2</span></span>&nbsp;'
                '<span class="sqrt"><span class="rad">&radic;</span><span class="rnd">p&#770;(1\u2212p&#770;)/n</span></span>'
                ' &nbsp;&nbsp;(Wald half-width: half = z<span class="sqrt"><span class="rad">&radic;</span><span class="rnd">p&#770;(1\u2212p&#770;)/n</span></span>)</span>'
            ),
            latex=r"\Delta_{\min}=z\sqrt{2}\,\sqrt{\frac{\hat p(1-\hat p)}{n}},\qquad "
                  r"\text{half}=z\sqrt{\frac{\hat p(1-\hat p)}{n}}",
        ),
    ],
    terms=[
        ("n", "number of eval items (artifact.n); when only some carry scores, the math uses n_scored below"),
        ("p\u0302", "observed accuracy = k / n_scored, where k = #items scored \u2265 0.5"),
        ("n_scored", "items carrying a recorded score (\u2264 n); the interval, half-width and gap rest on this, not n"),
        ("k", "number of items scored correct (score \u2265 0.5)"),
        ("z", "normal quantile for the confidence level; z = 1.96 for 95%"),
        ("CI\u2089\u2085", "Wilson score 95% interval for the true accuracy (reported as [lo, hi])"),
        ("half", "Wald half-width z\u221a(p\u0302(1\u2212p\u0302)/n), reported as a familiar \u00b1 points precision figure"),
        ("\u0394_min", "minimum significant gap: smallest A\u2212B accuracy difference at this n not attributable to noise (the probe's score)"),
    ],
    thresholds=[
        Threshold("min significant gap > 0.03 (i.e. > 3 points)", "medium",
                  "The eval can only resolve model differences larger than ~3 points; finer rankings on this benchmark are within noise \u2014 increase n or report intervals."),
        Threshold("min significant gap \u2264 0.03", "low",
                  "Adequately powered to resolve sub-3-point differences at this n; still report the interval rather than a bare number."),
        Threshold("n = 0", "high",
                  "Empty eval \u2014 no items to estimate anything from."),
        Threshold("fewer than 50% of items carry a score", "high",
                  "Accuracy is computed over a minority of items; non-random missingness can bias it, and the headline number does not represent the full set."),
    ],
    inputs=[
        ("n", "measured", "number of items in the eval run (artifact.n)"),
        ("n_scored", "measured", "items carrying a recorded score; accuracy, CI and gap rest on this, not n"),
        ("scoring_coverage", "derived", "n_scored / n \u2014 fraction of items actually scored"),
        ("k", "measured", "items with recorded score \u2265 0.5, among the scored items"),
        ("accuracy", "derived", "k / n_scored"),
        ("z", "parameter", "normal quantile fixed at 1.96 for 95% confidence"),
        ("wald_halfwidth", "derived", "z\u00b7\u221a(p\u0302(1\u2212p\u0302)/n_scored) \u2014 stats.proportion_halfwidth"),
        ("min_sig_gap", "derived", "z\u00b7\u221a2\u00b7\u221a(p\u0302(1\u2212p\u0302)/n_scored) \u2014 stats.min_significant_gap (the score)"),
        ("wilson_lo", "derived", "lower Wilson 95% bound on n_scored \u2014 stats.wilson_ci"),
        ("wilson_hi", "derived", "upper Wilson 95% bound on n_scored \u2014 stats.wilson_ci"),
    ],
    effect=(
        "Turn a bare score into a measurement with a stated uncertainty, and make "
        "model rankings honest: report the Wilson 95% interval, the \u00b1-point "
        "precision, and the smallest gap this benchmark can actually resolve, so "
        "ties are called ties instead of being read off the third decimal."
    ),
    reading=(
        "Read the score together with n. The Wilson interval [lo, hi] is the "
        "plausible range for the true accuracy; the \u00b1 half-width is its "
        "precision. The headline `score` field is the minimum significant gap "
        "\u0394_min: if two models differ by less than \u0394_min on this "
        "benchmark, treat them as tied \u2014 the ordering is noise. A large "
        "\u0394_min (small n, or p near 0.5 where variance peaks) means the "
        "benchmark is underpowered for the comparisons people will want to make; "
        "the remedy is more items or explicit intervals, not more decimal places."
    ),
    caveats=(
        "The binomial model is deliberately simple and, in several common "
        "situations, OPTIMISTIC about precision. (a) Paired comparisons: when two "
        "models are run on the SAME items, a paired test (e.g. McNemar) is more "
        "powerful and appropriate than this two-independent-proportions gap; "
        "\u0394_min is a conservative, unpaired bound. (b) Non-independent items: "
        "if items are clustered (same passage, near-duplicates) the effective n is "
        "smaller than the raw count, so the true interval is wider \u2014 see "
        "subgroup_power and dataset_hygiene. (c) Multiple comparisons: ranking many "
        "models or configs inflates false 'significant' gaps; see "
        "multiplicity_cherrypick. (d) p\u0302 uses a 0.5 score threshold to count "
        "correct items, so partial-credit scores are binarized for the proportion "
        "math. (e) The interval quantifies sampling error only \u2014 not "
        "contamination, judge error, or label noise, which the other probes "
        "address; a tight interval around a biased score is still precise about "
        "the wrong number."
    ),
    code_refs=["trust_the_eval.stats.wilson_ci",
               "trust_the_eval.stats.min_significant_gap",
               "trust_the_eval.stats.proportion_halfwidth"],
)

StatisticalPower.TUNABLES = {'gap_medium': {'default': 0.03, 'min': 0, 'max': 1, 'step': 0.005, 'help': 'min-significant-gap > this -> MEDIUM'}, 'cov_high': {'default': 0.5, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'scoring coverage < this -> HIGH'}, 'cov_medium': {'default': 0.9, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'coverage < this -> at least MEDIUM'}, 'correct_cutoff': {'default': 0.5, 'min': 0, 'max': 1, 'step': 0.05, 'help': 'score >= this counts as correct'}}
