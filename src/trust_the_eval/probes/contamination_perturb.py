from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..evidence import trim
from ..finding import Finding, Severity
from ..grading import default_grader
from ..perturb import paraphrase
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class ContaminationPerturbation(Probe):
    """Detect MEMORIZATION: items the model passes verbatim but fails when the
    question is paraphrased (answer preserved) signal contamination, not skill."""
    id = "contamination_perturb"
    name = "Contamination via perturbation re-test"
    paper_priority = "P5"
    requires_model = True

    def __init__(self, sample_size: int = 40, seed: int = 0):
        self.sample_size, self.seed = sample_size, seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        rows, n_checked, n_flipped = [], 0, 0
        for idx, it in subsample(artifact.items, self.sample_size, self.seed):
            if not it.question.strip() or not it.answer.strip():
                continue
            verbatim = model.complete(it.question, temperature=0.0)
            if not default_grader(verbatim, it.answer):
                continue  # only items it gets right verbatim are candidates
            n_checked += 1
            para = paraphrase(it.question, seed=self.seed + idx)
            re_ans = model.complete(para, temperature=0.0)
            if not default_grader(re_ans, it.answer):
                n_flipped += 1
                rows.append({"item": idx, "question": trim(it.question),
                             "paraphrase": trim(para), "gold": trim(it.answer)})
        risk = (n_flipped / n_checked) if n_checked else 0.0
        sev = (Severity.HIGH if risk >= self.tune("risk_high") else Severity.MEDIUM if risk >= self.tune("risk_medium")
               else Severity.LOW if n_checked else Severity.INFO)
        summary = (f"contamination risk {risk:.2f} "
                   f"({n_flipped}/{n_checked} pass verbatim but fail paraphrased)"
                   if n_checked else "no verbatim-correct items to test")
        return [Finding(self.id, sev, summary, score=round(risk, 3),
                        otel_attributes={"gen_ai.eval.trust.contamination.risk": round(risk, 3)},
                        evidence={"checked": n_checked, "flipped": n_flipped, "examples": rows[:5]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

ContaminationPerturbation.DOC = ProbeDoc(
    science=(
        "Benchmark (test-set) contamination is the leakage of evaluation items "
        "into a model's training data. When it occurs, a benchmark stops "
        "measuring capability and starts measuring memorization, inflating the "
        "reported score. It is especially hard to rule out for proprietary "
        "models whose pretraining corpora are not public, and the usual defence "
        "\u2014 string-level decontamination by n-gram overlap \u2014 is "
        "insufficient: Yang et al. (2023) show that simple meaning-preserving "
        "variations of test items (paraphrase, translation) bypass n-gram "
        "filters entirely, yet a 13B model overfit on such rephrased samples "
        "can reach GPT-4-level scores on MMLU, GSM8K and HumanEval. The "
        "underlying question \u2014 separating genuine generalization from test "
        "memorization \u2014 is the one Magar & Schwartz (2022) and Oren et al. "
        "(2024) formalize.\n\n"
        "This probe operationalizes the rephrased-sample insight as a black-box "
        "behavioural test. The principle: a meaning-preserving paraphrase leaves "
        "the correct answer unchanged, so a model that truly possesses the "
        "underlying capability should answer a paraphrase as well as the "
        "verbatim item. If the model passes the verbatim item but fails a "
        "semantically equivalent rewrite, its success was bound to the surface "
        "form \u2014 the signature of memorized test strings rather than "
        "competence. Unlike order-based proofs (Oren et al., 2024, which exploit "
        "an exchangeable benchmark's canonical ordering) or likelihood / "
        "membership-inference methods (Shi et al., 2024, Min-K% Prob; Golchin & "
        "Surdeanu, 2024, guided prompting), it needs no weights, no token "
        "log-probabilities and no access to the pretraining data \u2014 only the "
        "ability to query the model and grade answers.\n\n"
        "Scope and honesty: the output is a screening signal (a 'risk'), not a "
        "proof of contamination, and it assesses the validity of the evaluation "
        "result \u2014 whether the score is a real capability measurement \u2014 "
        "never the safety of the model. A high risk warrants corroboration with "
        "an order-based or likelihood-based test and human review of the flagged "
        "items, which the probe returns as evidence."
    ),
    references=[
        Reference("Yang, Chiang, Zheng, Gonzalez, Stoica",
                  "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples",
                  "arXiv", 2023, arxiv="2311.04850",
                  note="Backbone of this probe: paraphrased/translated test items evade n-gram "
                       "decontamination yet still massively inflate scores."),
        Reference("Oren, Meister, Chatterji, Ladhak, Hashimoto",
                  "Proving Test Set Contamination in Black Box Language Models",
                  "ICLR (Outstanding Paper Honorable Mention)", 2024, arxiv="2310.17623",
                  note="Complementary black-box proof via the canonical ordering of an "
                       "exchangeable benchmark."),
        Reference("Shi, Ajith, Xia, Huang, Liu, Blevins, Chen, Zettlemoyer",
                  "Detecting Pretraining Data from Large Language Models (Min-K% Prob)",
                  "ICLR", 2024, arxiv="2310.16789",
                  note="Likelihood/membership-inference alternative when token log-probs are available."),
        Reference("Golchin, Surdeanu",
                  "Time Travel in LLMs: Tracing Data Contamination in Large Language Models",
                  "ICLR", 2024, arxiv="2308.08493",
                  note="Guided-prompting detection; corroborating evidence for a flagged benchmark."),
        Reference("Magar, Schwartz",
                  "Data Contamination: From Memorization to Exploitation",
                  "ACL", 2022, arxiv="2203.08242",
                  note="Formalizes the memorization-vs-exploitation distinction this probe leverages."),
        Reference("Sainz, Campos, Garc\u00eda-Ferrero, Etxaniz, de Lacalle, Agirre",
                  "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark",
                  "EMNLP Findings", 2023, arxiv="2310.18018",
                  note="Argues contamination must be measured per benchmark \u2014 the motivation for shipping this as a probe."),
    ],
    math=[
        MathBlock(
            label="Contamination risk (conditional flip rate)",
            html=(
                '<span class="mrow">risk = '
                '<span class="frac"><span class="num">&Sigma;<sub>i&isin;S</sub> '
                'v<sub>i</sub>(1 &minus; p<sub>i</sub>)</span>'
                '<span class="den">&Sigma;<sub>i&isin;S</sub> v<sub>i</sub></span></span>'
                ' = n<sub>flip</sub> / n<sub>check</sub></span>'
            ),
            latex=r"\mathrm{risk}=\frac{\sum_{i\in S} v_i\,(1-p_i)}{\sum_{i\in S} v_i}"
                  r"=\frac{n_{\mathrm{flip}}}{n_{\mathrm{check}}}",
        ),
        MathBlock(
            label="Per-item indicators",
            html=(
                '<span class="mrow">v<sub>i</sub> = '
                '<b>1</b>[ grade(model(q<sub>i</sub>)) = correct ]'
                ' &nbsp;&nbsp; p<sub>i</sub> = '
                '<b>1</b>[ grade(model(paraphrase(q<sub>i</sub>))) = correct ]</span>'
            ),
            latex=r"v_i=\mathbf{1}[\,\mathrm{grade}(\mathrm{model}(q_i))=\text{correct}\,],\quad "
                  r"p_i=\mathbf{1}[\,\mathrm{grade}(\mathrm{model}(\mathrm{paraphrase}(q_i)))=\text{correct}\,]",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled subset of eval items (size set by sample_size, default 40)"),
        ("q\u1d62", "the i-th item's question (verbatim, as it appears in the benchmark)"),
        ("v\u1d62", "1 if the model answers the verbatim item correctly, else 0 (the verbatim-correct subset are the only candidates tested)"),
        ("p\u1d62", "1 if the model answers a meaning-preserving paraphrase of q\u1d62 correctly, else 0"),
        ("n_check", "\u03a3 v\u1d62 \u2014 number of items the model gets right verbatim (the conditioning set)"),
        ("n_flip", "\u03a3 v\u1d62(1\u2212p\u1d62) \u2014 of those, how many it then fails after paraphrase"),
        ("risk", "n_flip / n_check \u2014 the empirical estimate of P(fail paraphrase | pass verbatim)"),
    ],
    thresholds=[
        Threshold("risk \u2265 0.40", "high",
                  "Most verbatim-correct answers collapse under reformulation \u2014 strong memorization signal; treat the headline score as inflated."),
        Threshold("0.15 \u2264 risk < 0.40", "medium",
                  "A sizable minority of answers depend on surface form; investigate the flagged items and corroborate."),
        Threshold("0 < risk < 0.15", "low",
                  "Mostly robust to reformulation, within plausible model brittleness."),
        Threshold("n_check = 0", "info",
                  "The model gets no sampled item right verbatim, so there is nothing to test for memorization here."),
    ],
    effect=(
        "Ensure the reported score reflects capability that survives a "
        "meaning-preserving reformulation, rather than recall of memorized test "
        "strings. The probe isolates, on the model's own correct answers, the "
        "fraction that evaporate when the question is paraphrased \u2014 a direct, "
        "black-box proxy for rephrased-sample contamination that n-gram "
        "decontamination cannot see."
    ),
    reading=(
        "risk \u2248 0 means correctness is stable across equivalent phrasings: "
        "the score behaves like a capability measurement. A high risk means a "
        "meaningful share of 'correct' answers were tied to the exact wording, "
        "so the benchmark is, in part, measuring memorization and the headline "
        "number is optimistic. The score is a conditional probability in [0,1]; "
        "read it together with n_check (how many items it is computed over \u2014 "
        "a tiny n_check is noisy) and inspect the offending items returned as "
        "evidence before concluding."
    ),
    caveats=(
        "This is a screening signal, not proof. Confounds that can inflate risk "
        "without contamination: (a) the paraphrase genuinely changes difficulty "
        "or introduces ambiguity; (b) the model is simply brittle to surface "
        "form despite no leakage. Confounds that can deflate it: a lenient "
        "grader may count a near-miss as correct on both versions. Mitigations: "
        "use a small, controlled paraphrase set; corroborate with an order-based "
        "(Oren et al.) or likelihood-based (Min-K%) test; and review flagged "
        "items by hand. A Wilson interval on the proportion (n = n_check, see "
        "trust_the_eval.stats.wilson_ci) quantifies the estimate's uncertainty; "
        "the probe currently reports the point estimate."
    ),
    code_refs=["trust_the_eval.perturb.paraphrase",
               "trust_the_eval.grading.default_grader"],
)

ContaminationPerturbation.TUNABLES = {'risk_high': {'default': 0.4, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'flip risk >= this -> HIGH'}, 'risk_medium': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'flip risk >= this -> MEDIUM'}, 'sample_size': {'default': 40, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'sampling seed', 'ctor': True}}
