from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..probe import ModelClient, Probe, register
from ..stats import point_biserial


@register
class DiscriminationSaturation(Probe):
    """Detect ceiling/floor: if (near) every item is right or wrong, the score no
    longer discriminates models. Uses the recorded per-item scores."""
    id = "discrimination_saturation"
    name = "Discrimination & saturation"
    paper_priority = "Principle 2"
    requires_model = False

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        scored = [it.score for it in artifact.items if it.score is not None]
        if not scored:
            return [Finding(self.id, Severity.INFO, "no per-item scores recorded",
                            evidence={"scores_present": False})]
        n = len(scored)
        acc = sum(scored) / n
        binary = all(s <= 1e-9 or s >= 1 - 1e-9 for s in scored)
        ceiling, floor = acc >= self.tune("ceiling"), acc <= self.tune("floor")
        if acc >= self.tune("hi_ceiling") or acc <= self.tune("hi_floor"):
            sat_sev = Severity.HIGH
        elif ceiling or floor:
            sat_sev = Severity.MEDIUM
        else:
            sat_sev = Severity.LOW
        # Item discrimination (point-biserial, CTT/IRT) needs MANY takers/models;
        # a handful of models is too few for a stable per-item correlation.
        MIN_TAKERS = self.tune("min_takers")
        matrix = artifact.metadata.get("item_model_scores")
        n_models = (len(matrix[0]) if isinstance(matrix, list) and matrix
                    and isinstance(matrix[0], list) else 0)
        disc_share = None
        if (isinstance(matrix, list) and matrix
                and all(isinstance(r, list) and len(r) == n_models for r in matrix)
                and n_models >= MIN_TAKERS):
            totals = [sum(row[j] for row in matrix) for j in range(n_models)]
            low = sum(1 for row in matrix
                      if (lambda pb: pb == pb and pb < self.tune("pb_cutoff"))(
                          point_biserial(totals, [1 if x >= self.tune("correct_cutoff") else 0 for x in row])))
            disc_share = low / len(matrix)
        rank = {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3}
        sev = sat_sev
        if disc_share is not None and disc_share > self.tune("disc_share_hi") and rank[Severity.MEDIUM] > rank[sev]:
            sev = Severity.MEDIUM
        sat_tag = (" \u2014 at ceiling (saturated)" if ceiling
                   else " \u2014 at floor (saturated)" if floor else "")
        if disc_share is not None:
            msg = (f"accuracy {acc:.2f}{sat_tag}; {disc_share:.0%} of items barely "
                   f"discriminate (point-biserial < 0.1 across {n_models} models)")
        elif n_models >= 2:
            msg = (f"accuracy {acc:.2f}{sat_tag}; item\u00d7model matrix has only {n_models} "
                   f"models \u2014 need \u2265{MIN_TAKERS} takers for reliable discrimination")
        else:
            msg = (f"accuracy {acc:.2f}{sat_tag}; per-item discrimination needs an "
                   f"item\u00d7model matrix (single-model "
                   f"{'binary ' if binary else ''}scores can't reveal it)")
        return [Finding(self.id, sev, msg, score=round(acc, 3),
                        otel_attributes={"gen_ai.eval.trust.saturation.accuracy": round(acc, 3),
                                         "gen_ai.eval.trust.saturation.ceiling": ceiling,
                                         "gen_ai.eval.trust.discrimination.low_share":
                                             (round(disc_share, 3) if disc_share is not None else -1.0)},
                        evidence={"n": n, "accuracy": round(acc, 4), "binary": binary,
                                  "ceiling": ceiling, "floor": floor, "n_models": n_models,
                                  "discrimination_assessable": disc_share is not None,
                                  "low_discrimination_share":
                                      (round(disc_share, 3) if disc_share is not None else None)})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

DiscriminationSaturation.DOC = ProbeDoc(
    science=(
        "A benchmark earns its keep by DISCRIMINATING: spreading models out so "
        "their scores carry information. When nearly every item is answered "
        "correctly (a ceiling) or incorrectly (a floor), the benchmark stops "
        "telling models apart \u2014 a 96% and a 98% are a coin-flip apart, and the "
        "ranking is noise. This is the psychometric notion of item "
        "discrimination: Lalor et al. (2016), bringing Item Response Theory to NLP "
        "evaluation, show that test items differ in difficulty and DISCRIMINATING "
        "POWER, and that items everyone passes or everyone fails contribute almost "
        "no signal about ability. At the whole-benchmark level the same effect is "
        "called saturation, and it is now the dominant failure mode of popular "
        "evals: Ott et al. (2022), mapping the global dynamics of AI benchmarks in "
        "Nature Communications, find that a large fraction saturate, often within "
        "a couple of years of release. Kiela et al. (2021), introducing Dynabench, "
        "make the consequence explicit \u2014 models reach outstanding benchmark "
        "scores while the test loses its ability to drive or measure progress \u2014 "
        "and argue for dynamic, ceiling-resistant benchmarks.\n\n"
        "This probe gives a fast, static read on saturation from the recorded "
        "per-item scores. It computes the overall accuracy and flags ceiling/floor saturation; given an item x model matrix it also computes per-item "
        "point-biserial discrimination (needs >= 5 models). Items everyone passes or "
        "everyone fails carry no IRT signal; they are the "
        "kind a saturated benchmark is full of. A benchmark "
        "is flagged saturated when accuracy is at a ceiling (>= 0.95) or floor "
        "(<= 0.05). The headline "
        "accuracy is reported as the score, alongside the discrimination read that tells "
        "you whether that accuracy still leaves room to discriminate.\n\n"
        "Scope and honesty: this assesses the validity of the eval as a MEASURING "
        "INSTRUMENT \u2014 whether it can still separate models \u2014 not the safety "
        "of the model. With a single model only saturation is read; a ceiling/floor means low "
        "discrimination, but per-item correctness is a coarse proxy for the full "
        "IRT discrimination parameter (which needs many models' response "
        "patterns). Saturation is also relative to the population of models you "
        "care about \u2014 a benchmark saturated for frontier models can still "
        "discriminate weaker ones (Lalor et al.) \u2014 so a ceiling here means 'not "
        "informative for models at this level', and the remedy is a harder or "
        "dynamic benchmark (Kiela et al.), not a tweak to the score."
    ),
    references=[
        Reference("Lalor, Wu, Yu",
                  "Building an Evaluation Scale using Item Response Theory",
                  "EMNLP", 2016, arxiv="1605.08889",
                  note="Brings IRT to NLP evaluation: items have difficulty and DISCRIMINATING POWER, and items everyone passes/fails give almost no signal \u2014 the psychometric basis for measuring saturation."),
        Reference("Ott, Barbosa-Silva, Blagec, Brauner, Samwald",
                  "Mapping Global Dynamics of Benchmark Creation and Saturation in Artificial Intelligence",
                  "Nature Communications 13, 6793", 2022,
                  url="https://doi.org/10.1038/s41467-022-34591-0",
                  note="Empirically maps benchmark saturation across AI: a large fraction saturate, often within a couple of years \u2014 evidence that saturation is the dominant validity threat this probe screens for."),
        Reference("Kiela, Bartolo, Nie, Kaushik, et al.",
                  "Dynabench: Rethinking Benchmarking in NLP",
                  "NAACL", 2021, arxiv="2104.14337",
                  note="Documents that models reach outstanding scores while benchmarks lose discriminative value, and proposes dynamic, ceiling-resistant benchmarking \u2014 the remedy when this probe flags saturation."),
    ],
    math=[
        MathBlock(
            label="Accuracy & saturation flag",
            html=(
                '<span class="mrow">acc = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i</sub> s<sub>i</sub> ,&nbsp;&nbsp; '
                'saturated = <b>1</b>[ acc &ge; 0.95 &or; acc &le; 0.05 ]</span>'
            ),
            latex=r"\mathrm{acc}=\frac1n\sum_i s_i,\qquad "
                  r"\mathrm{saturated}=\mathbf{1}[\mathrm{acc}\ge 0.95\ \vee\ \mathrm{acc}\le 0.05]",
        ),
        MathBlock(
            label="Item discrimination (point-biserial; needs an item x model matrix)",
            html=(
                '<span class="mrow">r<sub>pb</sub>(item) = corr( correct<sub>item</sub> , total<sub>model</sub> ) ,&nbsp; '
                'low_share = #{ items : r<sub>pb</sub> &lt; 0.1 } / N&nbsp; (only when &ge; 5 models supplied)</span>'
            ),
            latex=r"r_{pb}=\mathrm{corr}(\text{correct}_{\text{item}},\text{total}_{\text{model}}),\ "
                  r"\text{low\_share}=|\{r_{pb}<0.1\}|/N\ (\ge 5\ \text{models})",
        ),
    ],
    terms=[
        ("n", "number of items with a recorded per-item score"),
        ("s\u1d62", "the recorded score of item i (typically 0/1 correctness, or a graded value in [0,1])"),
        ("acc", "overall accuracy, mean of s\u1d62 (the probe's score)"),
        ("ceiling / floor", "acc >= 0.95 (ceiling) or acc <= 0.05 (floor) -- the saturation flags"),
        ("item x model matrix", "metadata['item_model_scores']: per-item scores across models; >= 5 needed for discrimination"),
        ("r_pb / low_share", "per-item point-biserial of correctness vs model total; low_share = fraction with r_pb < 0.1"),
    ],
    thresholds=[
        Threshold("accuracy >= 0.98 or <= 0.02", "high",
                  "Severe ceiling/floor: the score is essentially uninformative for models at this level; move to a harder or dynamic benchmark."),
        Threshold("accuracy >= 0.95 or <= 0.05, or >30% low-discrimination items (with >= 5 models)", "medium",
                  "Saturated, or (given a real item x model matrix) most items barely discriminate -- score gaps are largely noise."),
        Threshold("otherwise (scores present, mid-range)", "low",
                  "Accuracy sits away from the extremes; the benchmark still discriminates on this run. Per-item discrimination needs an item x model matrix (>= 5 models)."),
        Threshold("no per-item scores", "info",
                  "No recorded per-item scores, so saturation/discrimination cannot be assessed."),
    ],
    effect=(
        "Tell whether the benchmark can still separate models or has hit a "
        "ceiling/floor where the score is uninformative. By reporting accuracy "
        "and, with an item x model matrix, the share of low-discrimination items, the probe flags when a "
        "leaderboard's tight cluster of high scores is noise rather than signal "
        "\u2014 the cue to retire or harden the benchmark."
    ),
    reading=(
        "A mid-range accuracy (away from ceiling/floor) means the benchmark "
        "discriminates: score gaps carry information. Accuracy at a ceiling "
        "(>= 0.95) or floor (<= 0.05) means "
        "almost every item lands all-right or all-wrong \u2014 those items add no "
        "separation, so ranking models by this score is reading noise (a 96% vs "
        "98% is within sampling error; see statistical_power). The score is the "
        "accuracy; per-item discrimination (point-biserial) needs an item x model matrix. Saturation is "
        "relative to the models you test: a ceiling for frontier models may still "
        "separate weaker ones, so the fix is a harder or dynamic benchmark (Kiela "
        "et al.), not a reinterpretation of the number."
    ),
    caveats=(
        "(a) Coarse proxy for discrimination: per-item correctness extremes "
        "approximate, but are not, the IRT discrimination parameter, which is "
        "estimated from MANY models' response patterns (Lalor et al.) \u2014 this "
        "single-run view can miss items that are non-extreme yet still "
        "non-discriminating. (b) Population-relative: 'saturated' means "
        "uninformative for the model(s) that produced these scores; the same "
        "benchmark may discriminate a different ability range, so do not read a "
        "ceiling as 'the benchmark is bad' in absolute terms. (c) Thresholds are "
        "conventions: 0.95/0.10 and the 0.9 extreme-share cut are rules of thumb, "
        "not calibrated boundaries. (d) Needs per-item scores and binarizes graded "
        "values at the extremes; partial-credit datasets concentrate less at 0/1, "
        "which can understate saturation. (e) Complementary, not redundant: pair "
        "with statistical_power (is a given gap significant at this n?) and "
        "contamination_perturb (is a ceiling real mastery or memorization?), and "
        "note that a floor can also signal a broken harness/grader rather than a "
        "hard benchmark (cf. answer_extraction_audit)."
    ),
    code_refs=["trust_the_eval.stats.point_biserial"],
)

DiscriminationSaturation.TUNABLES = {'ceiling': {'default': 0.95, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'acc >= this = ceiling (MEDIUM)'}, 'floor': {'default': 0.05, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'acc <= this = floor (MEDIUM)'}, 'hi_ceiling': {'default': 0.98, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'acc >= this -> HIGH'}, 'hi_floor': {'default': 0.02, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'acc <= this -> HIGH'}, 'min_takers': {'default': 5, 'min': 2, 'max': 50, 'step': 1, 'help': 'min models for point-biserial'}, 'pb_cutoff': {'default': 0.1, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'point-biserial < this = poor discrimination'}, 'disc_share_hi': {'default': 0.3, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'share of poor-discriminating items > this -> MEDIUM'}, 'correct_cutoff': {'default': 0.5, 'min': 0, 'max': 1, 'step': 0.05, 'help': 'score >= this counts as correct'}}
