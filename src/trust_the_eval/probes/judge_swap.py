from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..judges import judge_correct
from ..probe import ModelClient, Probe, register
from ..sampling import subsample
from ..stats import cohens_kappa


@register
class JudgeSwap(Probe):
    """Attack LLM-as-judge validity: (a) position bias via order swap; (b) self-
    preference/collusion via a cross-family judge (pass one as `model`); (c)
    Cohen's kappa vs human labels in meta['human_label']."""
    id = "judge_swap"
    name = "Judge validity (position bias, collusion, kappa)"
    paper_priority = "Axis II"
    requires_model = True

    def __init__(self, sample_size: int = 50, cross_family_judge: Optional[ModelClient] = None,
                 seed: int = 0):
        self.sample_size = sample_size
        self.cross = cross_family_judge
        self.seed = seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None  # `model` is the (same-family) judge under test
        items = [(i, it) for i, it in subsample(artifact.items, self.sample_size, self.seed)
                 if it.response is not None and it.answer.strip()]
        if not items:
            return [Finding(self.id, Severity.INFO,
                            "no (response, gold) pairs to judge")]
        n = swaps = cross_disagree = 0
        own_labels, cross_labels, human_labels = [], [], []
        for _, it in items:
            n += 1
            s_first = judge_correct(model, it.response, it.answer, first=True, author="fam-A")
            s_last = judge_correct(model, it.response, it.answer, first=False, author="fam-A")
            if s_first != s_last:
                swaps += 1
            own = s_last
            own_labels.append(own)
            if self.cross is not None:
                c = judge_correct(self.cross, it.response, it.answer, first=False, author="fam-B")
                cross_labels.append(c)
                if c != own:
                    cross_disagree += 1
            hl = it.meta.get("human_label")
            if hl is not None:
                human_labels.append(int(hl))
        pos_bias = swaps / n
        attrs = {"gen_ai.eval.trust.judge.position_bias": round(pos_bias, 3)}
        bits = [f"position bias {pos_bias:.2f} ({swaps}/{n} verdicts flip on order swap)"]
        sev = Severity.MEDIUM if pos_bias >= self.tune("pos_bias_medium") else Severity.LOW
        if cross_labels:
            collusion = cross_disagree / n
            attrs["gen_ai.eval.trust.judge.cross_family_disagree"] = round(collusion, 3)
            bits.append(f"cross-family disagreement {collusion:.2f}")
            if collusion >= self.tune("collusion_high"):
                sev = Severity.HIGH
        if human_labels and len(human_labels) == len(own_labels[:len(human_labels)]):
            k = cohens_kappa(own_labels[:len(human_labels)], human_labels)
            attrs["gen_ai.eval.trust.judge.kappa"] = round(k, 3)
            bits.append(f"kappa vs humans {k:.2f}")
            if k < self.tune("kappa_high"):
                sev = Severity.HIGH
        return [Finding(self.id, sev, "; ".join(bits), score=round(pos_bias, 3),
                        otel_attributes=attrs,
                        evidence={"judged": n, "order_flips": swaps,
                                  "cross_family_disagree": cross_disagree})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

JudgeSwap.DOC = ProbeDoc(
    science=(
        "Using a strong LLM as an automatic grader ('LLM-as-a-judge') is now "
        "routine for open-ended evaluations, but the judge is itself a model with "
        "systematic biases, so an eval scored this way inherits them. Zheng et "
        "al. (2023), the reference study, catalogue three: position bias (the "
        "verdict depends on the order in which candidates are shown), verbosity "
        "bias (longer answers are favoured), and self-enhancement bias (a judge "
        "favours answers in its own style \u2014 they report GPT-4 favouring "
        "itself by roughly ten points of win rate). They also establish the "
        "positive baseline this probe is measured against: a strong judge can "
        "reach over 80% agreement with humans, about the level humans agree with "
        "each other \u2014 so the goal is not 'no LLM judge' but 'a judge whose "
        "errors are bounded and measured'.\n\n"
        "Each bias has direct empirical grounding. Position bias: Wang et al. "
        "(2023) show that simply swapping the order of two answers can flip an "
        "LLM judge's preference, and propose order-balancing as a mitigation \u2014 "
        "exactly the manipulation this probe performs. Self-preference: "
        "Panickssery et al. (2024) show LLM judges can recognise their own "
        "outputs (GPT-4 ~73.5% accurate at distinguishing itself) and that this "
        "self-recognition is causally linked to scoring their own generations "
        "higher than humans judge them to deserve \u2014 a within-family judge is "
        "therefore a conflicted referee. Agreement with a human gold standard is "
        "quantified with Cohen's \u03ba (Cohen, 1960), which corrects raw "
        "agreement for chance and is the standard inter-rater statistic.\n\n"
        "This probe stress-tests the judge in three independent ways on the "
        "eval's own (response, gold) pairs. (1) Position bias: each response is "
        "judged twice, presented first vs not-first; the fraction of verdicts "
        "that flip is the position-bias rate. (2) Cross-family disagreement: an "
        "optional second judge from a different model family re-scores the same "
        "pairs; high disagreement flags self-preference / within-family collusion "
        "(passing the eval's own judge as a different-family judge would be a "
        "conflict of interest). (3) Human agreement: where items carry a human "
        "label, Cohen's \u03ba between the judge and the humans is reported. The "
        "probe attacks the validity of the scoring step \u2014 whether the judge "
        "measures correctly \u2014 never the safety of the model under test.\n\n"
        "Scope and honesty: position-flip rate and cross-family disagreement are "
        "necessary-not-sufficient signals (a judge can be consistently wrong yet "
        "order-stable). \u03ba needs human labels to mean anything, and its "
        "interpretation bands are conventional, not laws. The probe diagnoses; it "
        "does not certify a judge as correct."
    ),
    references=[
        Reference("Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica",
                  "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
                  "NeurIPS", 2023, arxiv="2306.05685",
                  note="The reference study: defines position, verbosity and self-enhancement bias, and shows strong judges reach >80% agreement with humans (the baseline this probe measures against)."),
        Reference("Wang, Li, Chen, Zhu, Lin, Cao, Liu, Liu, Sui",
                  "Large Language Models are not Fair Evaluators",
                  "arXiv", 2023, arxiv="2305.17926",
                  note="Demonstrates position bias \u2014 swapping answer order flips the judge's verdict \u2014 and proposes order-balancing, the manipulation this probe runs."),
        Reference("Panickssery, Bowman, Feng",
                  "LLM Evaluators Recognize and Favor Their Own Generations",
                  "NeurIPS", 2024, arxiv="2404.13076",
                  note="Self-preference: judges recognise their own outputs (GPT-4 ~73.5%) and score them higher than humans do \u2014 the basis for the cross-family / collusion check."),
        Reference("Cohen",
                  "A Coefficient of Agreement for Nominal Scales",
                  "Educational and Psychological Measurement, 20(1), 37\u201346", 1960,
                  url="https://doi.org/10.1177/001316446002000104",
                  note="Defines Cohen's \u03ba, the chance-corrected inter-rater agreement statistic used here to compare the judge to human labels."),
    ],
    math=[
        MathBlock(
            label="Position-bias rate (order-swap flip rate)",
            html=(
                '<span class="mrow">pos_bias = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i</sub> <b>1</b>[ s<sub>i</sub><sup>first</sup> &ne; s<sub>i</sub><sup>not-first</sup> ]</span>'
            ),
            latex=r"\mathrm{pos\_bias}=\frac1n\sum_{i=1}^{n}\mathbf{1}\!\left[s_i^{\text{first}}\neq s_i^{\text{not-first}}\right]",
        ),
        MathBlock(
            label="Cross-family disagreement",
            html=(
                '<span class="mrow">disagree = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i</sub> <b>1</b>[ s<sub>i</sub><sup>own</sup> &ne; s<sub>i</sub><sup>cross</sup> ]</span>'
            ),
            latex=r"\mathrm{disagree}=\frac1n\sum_{i=1}^{n}\mathbf{1}\!\left[s_i^{\text{own}}\neq s_i^{\text{cross}}\right]",
        ),
        MathBlock(
            label="Cohen's \u03ba vs human labels (chance-corrected agreement)",
            html=(
                '<span class="mrow">&kappa; = '
                '<span class="frac"><span class="num">p<sub>o</sub> &minus; p<sub>e</sub></span>'
                '<span class="den">1 &minus; p<sub>e</sub></span></span></span>'
            ),
            latex=r"\kappa=\frac{p_o-p_e}{1-p_e},\quad "
                  r"p_o=\tfrac1n\sum_i \mathbf{1}[s_i=h_i],\quad "
                  r"p_e=\sum_{c\in\{0,1\}}\hat p^{s}_c\,\hat p^{h}_c",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled subset of items with a stored response and a non-empty gold answer (size n)"),
        ("model (the judge)", "the judge under test; `model` is the SAME-family judge passed to the battery"),
        ("cross", "an optional second judge from a DIFFERENT model family (constructor arg cross_family_judge)"),
        ("s\u1d62^first / s\u1d62^not-first", "the judge's 0/1 correctness verdict when the candidate is presented first vs not, via an order flag in the prompt"),
        ("s\u1d62^own / s\u1d62^cross", "verdicts from the same-family judge vs the cross-family judge on item i"),
        ("s\u1d62 , h\u1d62", "the judge's verdict and the human label for item i (h from meta['human_label'])"),
        ("p_o", "observed agreement: fraction of items where judge and human agree"),
        ("p_e", "chance agreement: \u03a3_c (judge's rate of class c)\u00b7(humans' rate of class c)"),
        ("\u03ba", "Cohen's kappa: (p_o \u2212 p_e)/(1 \u2212 p_e); 1 = perfect, 0 = chance-level, <0 = worse than chance"),
    ],
    thresholds=[
        Threshold("cross-family disagreement \u2265 0.15", "high",
                  "The eval's judge and an independent-family judge disagree on a large share of items \u2014 strong evidence the verdict depends on the choice of judge (self-preference / collusion risk)."),
        Threshold("Cohen's \u03ba < 0.60 (when human labels present)", "high",
                  "Judge\u2013human agreement is only moderate or worse once corrected for chance \u2014 the automatic score is a weak proxy for human judgement."),
        Threshold("position-bias rate \u2265 0.10", "medium",
                  "At least one in ten verdicts flips purely on presentation order \u2014 order-balance the scoring and treat margins as noisy."),
        Threshold("otherwise", "low",
                  "Order-stable, and (where checkable) agreeing with the cross-family judge and humans on this sample."),
    ],
    effect=(
        "Ensure the score produced by an LLM judge reflects answer quality rather "
        "than artifacts of the judge \u2014 presentation order, kinship between "
        "judge and candidate, or drift from human judgement. The probe quantifies "
        "each of these separately so a judged eval can report bounded, measured "
        "grader error instead of an unexamined verdict."
    ),
    reading=(
        "Low values on all three sub-measures mean the judge behaves like a stable "
        "instrument on this sample: order-invariant, family-agnostic, and "
        "(where human labels exist) in chance-corrected agreement with people. A "
        "high position-bias rate means verdicts are partly an ordering artifact "
        "(mitigate by averaging over orders). High cross-family disagreement means "
        "the score is judge-dependent \u2014 a within-family judge may be flattering "
        "the candidate. \u03ba is the most decision-relevant when human labels are "
        "present: \u03ba\u22650.8 is strong, 0.6\u20130.8 substantial, <0.6 means the "
        "judge is a weak stand-in for humans. The headline `score` field is the "
        "position-bias rate; the other measures are attached as evidence/attributes."
    ),
    caveats=(
        "(a) Necessary, not sufficient: order-stability and cross-family agreement "
        "do not prove correctness \u2014 two judges can share the same blind spot, "
        "and a judge can be consistently wrong yet self-consistent. (b) \u03ba is "
        "only computed where meta['human_label'] exists, over those items; a small "
        "labelled subset gives a noisy \u03ba, and \u03ba is sensitive to label "
        "prevalence (high agreement can still yield low \u03ba when one class "
        "dominates). (c) The cross-family signal requires actually passing a "
        "different-family judge; without one, only position bias and \u03ba are "
        "available. (d) This probe targets pointwise correctness judging as the "
        "tool uses it; verbosity bias and pairwise-preference artifacts (Zheng et "
        "al.) are real but out of this probe's current scope. (e) The judge calls "
        "go through the CachingClient, so cost is metered and repeated identical "
        "judgements are deduped."
    ),
    code_refs=["trust_the_eval.judges.judge_correct",
               "trust_the_eval.stats.cohens_kappa"],
)

JudgeSwap.TUNABLES = {'pos_bias_medium': {'default': 0.1, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'position-bias rate >= this -> MEDIUM'}, 'collusion_high': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'cross-family disagreement >= this -> HIGH'}, 'kappa_high': {'default': 0.6, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'Cohen kappa vs humans < this -> HIGH'}, 'sample_size': {'default': 50, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}}
