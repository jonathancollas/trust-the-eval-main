from __future__ import annotations
from collections import Counter
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..grading import extract_final
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class SelfConsistency(Probe):
    """Resample each item; a high answer-flip rate means a single-run score is
    partly luck (unstable measurement)."""
    id = "self_consistency"
    name = "Self-consistency / stochastic stability"
    paper_priority = "Principle 2"
    requires_model = True

    def __init__(self, sample_size: int = 30, votes: int = 5, seed: int = 0):
        self.sample_size, self.votes, self.seed = sample_size, votes, seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        instabilities, n = [], 0
        for _, it in subsample(artifact.items, self.sample_size, self.seed):
            if not it.question.strip():
                continue
            finals = [extract_final(s) for s in
                      model.sample(it.question, self.votes, temperature=self.tune("temp"))]
            finals = [f for f in finals if f]
            if len(finals) < 2:
                continue
            n += 1
            top = Counter(finals).most_common(1)[0][1]
            instabilities.append(1 - top / len(finals))  # fraction disagreeing with mode
        mean_inst = sum(instabilities) / len(instabilities) if instabilities else 0.0
        sev = (Severity.HIGH if mean_inst >= self.tune("inst_high") else Severity.MEDIUM if mean_inst >= self.tune("inst_medium")
               else Severity.LOW if n else Severity.INFO)
        return [Finding(self.id, sev,
                        f"mean answer instability {mean_inst:.2f} over {n} items "
                        f"({self.votes} samples each)",
                        score=round(mean_inst, 3),
                        otel_attributes={"gen_ai.eval.trust.self_consistency.instability": round(mean_inst, 3)},
                        evidence={"items": n, "votes_each": self.votes})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

SelfConsistency.DOC = ProbeDoc(
    science=(
        "LLMs are stochastic: sampled at non-zero temperature, the same prompt "
        "yields different completions. A benchmark that scores one sample per item "
        "therefore measures capability plus luck \u2014 re-run it and the score "
        "moves. This probe quantifies that instability so a single-run number can "
        "be read with the right amount of trust.\n\n"
        "The idea rests on a now-standard observation. Wang et al. (2022) "
        "introduced self-consistency: sample several reasoning paths and take the "
        "majority answer; that the majority beats a single greedy decode is "
        "direct evidence that individual samples disagree, and that the spread "
        "carries signal. Kuhn et al. (2023) turn the spread itself into a "
        "measurement \u2014 the dispersion of resampled answers (their semantic "
        "entropy) is an unsupervised predictor of when a model's output can be "
        "trusted. And the stochasticity is not incidental to evaluation: Chen et "
        "al. (2023), studying behaviour drift, note that even nominally fixed "
        "model services give non-deterministic answers to identical inputs. If a "
        "model returns different final answers to the same question across "
        "resamples, then which answer the benchmark happened to record \u2014 and "
        "thus whether the item counts as correct \u2014 is partly an accident of "
        "sampling.\n\n"
        "Concretely, the probe takes a deterministic subset of items and, for "
        "each, draws several samples at temperature 0.7, extracts each sample's "
        "final answer, and measures how often the answers disagree with the modal "
        "(most common) answer. Averaged over items, this answer-instability is the "
        "score: 0 means every resample agrees (the item's correctness is a stable "
        "fact), high values mean the recorded pass/fail was a coin-flip. High "
        "instability also says best-of-1 understates achievable accuracy and "
        "majority-vote (self-consistency) decoding would both raise and stabilize "
        "the score.\n\n"
        "Scope and honesty: this assesses the reliability of the measurement (its "
        "run-to-run reproducibility), not the safety of the model, and not "
        "correctness \u2014 it is gold-free, measuring agreement among the model's "
        "own resamples. A confidently-wrong model can be perfectly self-consistent "
        "(low instability, still wrong), so read this alongside the accuracy and "
        "the contamination/elicitation probes rather than as a quality score."
    ),
    references=[
        Reference("Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou",
                  "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
                  "ICLR", 2023, arxiv="2203.11171",
                  note="Shows resampled answers disagree and that aggregating them (majority vote) beats a single decode \u2014 the basis for measuring, and mitigating, answer instability."),
        Reference("Kuhn, Gal, Farquhar",
                  "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation",
                  "ICLR (Oral)", 2023, arxiv="2302.09664",
                  note="Turns the dispersion of resampled answers into an unsupervised, single-model uncertainty signal \u2014 the conceptual parent of this instability metric."),
        Reference("Chen, Zaharia, Zou",
                  "How Is ChatGPT's Behavior Changing over Time?",
                  "arXiv (Harvard Data Science Review)", 2023, arxiv="2307.09009",
                  note="Documents that nominally fixed LLM services return non-deterministic answers to identical inputs \u2014 why single-run scores are not reproducible by default."),
    ],
    math=[
        MathBlock(
            label="Per-item answer instability (disagreement with the mode)",
            html=(
                '<span class="mrow">inst<sub>i</sub> = 1 &minus; '
                '<span class="frac"><span class="num">max<sub>a</sub> count<sub>i</sub>(a)</span>'
                '<span class="den">m<sub>i</sub></span></span></span>'
            ),
            latex=r"\mathrm{inst}_i = 1-\frac{\max_a \mathrm{count}_i(a)}{m_i}",
        ),
        MathBlock(
            label="Mean instability over items (the probe's score)",
            html=(
                '<span class="mrow">instability = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i&isin;S</sub> inst<sub>i</sub> ,&nbsp;&nbsp; '
                'a<sub>i,j</sub> = extract_final(sample<sub>j</sub>(q<sub>i</sub>; T=0.7))</span>'
            ),
            latex=r"\mathrm{instability}=\frac1n\sum_{i\in S}\mathrm{inst}_i,\qquad "
                  r"a_{i,j}=\mathrm{extract\_final}\big(\mathrm{sample}_j(q_i;\,T{=}0.7)\big)",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled items with a non-empty question (sample_size, default 30)"),
        ("m\u1d62", "number of non-empty extracted final answers obtained for item i (up to `votes`, default 5)"),
        ("votes", "samples drawn per item at temperature 0.7"),
        ("a\u1d62\u2c7c", "the final answer extracted from the j-th sample of item i (via extract_final)"),
        ("count\u1d62(a)", "how many of item i's samples gave final answer a"),
        ("mode", "the most common final answer for an item; max_a count\u1d62(a) is its vote count"),
        ("inst\u1d62", "1 minus the modal fraction \u2014 the share of resamples that disagree with the item's majority answer"),
        ("instability", "mean of inst\u1d62 over scored items (the probe's score), in [0,1)"),
        ("n", "number of items with \u2265 2 valid samples (those actually contributing to the mean)"),
    ],
    thresholds=[
        Threshold("instability \u2265 0.30", "high",
                  "Resamples frequently disagree \u2014 a single-run score is substantially luck; report majority-vote accuracy with a run-to-run interval, not one number."),
        Threshold("0.15 \u2264 instability < 0.30", "medium",
                  "Noticeable run-to-run variation; single-sample comparisons between models are fragile at this level."),
        Threshold("instability < 0.15 (with items measured)", "low",
                  "Answers are largely stable across resamples on this sample; the single-run score is reproducible here."),
        Threshold("no items with \u2265 2 valid samples", "info",
                  "Could not extract enough comparable answers to assess stability (e.g. empty/un-parseable outputs)."),
    ],
    effect=(
        "Tell how much of a single-run score is signal versus sampling luck. By "
        "resampling each item and measuring how often the model's own final "
        "answers disagree, the probe converts 'LLMs are stochastic' into a "
        "concrete instability number \u2014 so a benchmark result can be reported "
        "with a run-to-run interval and, if needed, stabilized with majority-vote "
        "decoding."
    ),
    reading=(
        "instability \u2248 0 means the model returns the same final answer on "
        "every resample, so the item's correctness \u2014 and the aggregate score "
        "\u2014 is reproducible. High instability means the recorded pass/fail was "
        "partly chance: re-running the eval would move the number, and ranking two "
        "models by a single run at this noise level is unreliable. The practical "
        "fix is to report majority-vote (self-consistency) accuracy over k samples "
        "and a run-to-run interval. Crucially this is gold-free: it measures "
        "agreement, not correctness, so a model that is consistently wrong scores "
        "LOW instability \u2014 pair it with the accuracy and with statistical_power "
        "(sampling error) and model_drift (across-time change)."
    ),
    caveats=(
        "(a) Not a quality metric: low instability means reproducible, not "
        "correct \u2014 a confidently-wrong model is self-consistent. Always read "
        "with accuracy. (b) Extraction-bound: instability is computed on "
        "extract_final outputs, so formatting variation that the extractor does "
        "not normalize (e.g. '12' vs 'twelve', units, ordering) can inflate it; "
        "for free-form generation a semantic-equivalence grouping (Kuhn et al.'s "
        "semantic entropy) is more faithful than exact-match on finals. (c) "
        "Temperature-dependent: the 0.7 sampling temperature sets the noise floor; "
        "an eval run at temperature 0 will be more stable than this probe suggests, "
        "and one run hotter, less \u2014 the score characterizes stability at THIS "
        "temperature, not the eval's. (d) Few votes / few items: with `votes`=5 "
        "the modal fraction is coarse (steps of 0.2) and small n is noisy; raise "
        "both for a tighter estimate at higher cost (calls scale with votes, "
        "metered by the CachingClient). (e) Mode ties are resolved by Counter's "
        "most_common ordering; with multimodal answers the single-mode fraction "
        "slightly understates true dispersion \u2014 entropy would be the richer "
        "summary."
    ),
    code_refs=["trust_the_eval.grading.extract_final",
               "trust_the_eval.sampling.subsample"],
)

SelfConsistency.TUNABLES = {'inst_high': {'default': 0.3, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'mean instability >= this -> HIGH'}, 'inst_medium': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': '>= this -> MEDIUM'}, 'temp': {'default': 0.7, 'min': 0, 'max': 2, 'step': 0.05, 'help': 'resampling temperature'}, 'sample_size': {'default': 30, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'votes': {'default': 5, 'min': 1, 'max': 20, 'step': 1, 'help': 'samples per item', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
