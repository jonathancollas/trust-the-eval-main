from __future__ import annotations
import re
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..grading import extract_final, normalize
from ..perturb import permute_options
from ..probe import ModelClient, Probe, register
from ..sampling import subsample

_LETTERS = "ABCDEFGH"


def _mcq(it) -> Optional[list[str]]:
    opts = it.meta.get("options") or it.meta.get("choices")
    return opts if isinstance(opts, list) and len(opts) >= 2 else None


@register
class OptionOrderBias(Probe):
    """For MCQ items, permute option order; if the chosen CONTENT changes, the
    score reflects position/label artifacts, not capability."""
    id = "option_order_bias"
    name = "MCQ option-order bias"
    paper_priority = "P5"
    requires_model = True

    def __init__(self, sample_size: int = 40, seed: int = 0):
        self.sample_size, self.seed = sample_size, seed

    def _ask(self, model, question, options):
        n = len(options)
        valid = _LETTERS[:n] + _LETTERS[:n].lower()
        block = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(options))
        out = model.complete(
            f"{question}\n{block}\nReply with the letter of the correct option.",
            temperature=0.0)
        # 1) an explicit letter answer ("answer is B", leading "C.", "(D)") —
        #    NOT any letter in prose (avoids the 'a' in "answer").
        m = (re.search(rf"(?:answer|option|choice|letter)\b[^A-Za-z0-9]*([{valid}])\b", out)
             or re.search(rf"^\s*\(?([{valid}])\)?[\).:,\s]", out)
             or re.search(rf"\(([{valid}])\)", out))
        if m:
            return options[_LETTERS.index(m.group(1).upper())]
        # 2) fall back: the model's final answer equals an option
        fin = extract_final(out)
        for o in options:
            if normalize(o) == fin or normalize(o) == normalize(out):
                return o
        # 3) last resort: an option mentioned at a word boundary (no '1' in '13')
        for o in options:
            if re.search(rf"(?<!\w){re.escape(normalize(o))}(?!\w)", normalize(out)):
                return o
        return None

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        mcq = [(i, it, _mcq(it)) for i, it in subsample(artifact.items, self.sample_size, self.seed)]
        mcq = [(i, it, o) for i, it, o in mcq if o]
        if not mcq:
            return [Finding(self.id, Severity.INFO, "no MCQ items (need meta.options)")]
        n, changed, dropped = 0, 0, 0
        for i, it, opts in mcq:
            a = self._ask(model, it.question, opts)
            shuffled, _ = permute_options(opts, seed=self.seed + i)
            b = self._ask(model, it.question, shuffled)
            if a is None or b is None:
                dropped += 1                     # model's option choice not parseable
                continue
            n += 1
            if normalize(a) != normalize(b):
                changed += 1
        if n == 0:
            return [Finding(self.id, Severity.INFO,
                            f"could not parse the model's option choice on any of "
                            f"{dropped} MCQ items (no order-bias signal)",
                            evidence={"mcq_candidates": len(mcq), "unparsed_dropped": dropped})]
        rate = changed / n
        sev = (Severity.HIGH if rate >= self.tune("high") else Severity.MEDIUM if rate >= self.tune("medium")
               else Severity.LOW)
        return [Finding(self.id, sev,
                        f"answer changed on reorder in {rate:.2f} of MCQ items "
                        f"({changed}/{n}); {dropped} unparseable (dropped)",
                        score=round(rate, 3),
                        otel_attributes={"gen_ai.eval.trust.option_order.flip_rate": round(rate, 3),
                                         "gen_ai.eval.trust.option_order.unparsed": dropped},
                        evidence={"mcq_checked": n, "flipped": changed, "unparsed_dropped": dropped})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

OptionOrderBias.DOC = ProbeDoc(
    science=(
        "Multiple-choice questions are a dominant eval format because they are "
        "trivial to grade, but that convenience hides a validity trap: an LLM's "
        "choice can depend on WHERE an option sits and WHICH letter labels it, "
        "not on which option is correct. When it does, MCQ accuracy partly "
        "measures a positional/label artifact rather than knowledge. Pezeshkpour "
        "& Hruschka (2024) show the effect is large \u2014 reordering the answer "
        "options moves accuracy by roughly 13% to 75% across benchmarks, even "
        "with few-shot demonstrations \u2014 and conjecture it surfaces when the "
        "model is uncertain among its top choices. Zheng et al. (2024) trace the "
        "mechanism: LLMs have an inherent 'selection bias', a prior preference "
        "for specific option IDs (e.g. 'A'), driven mainly by token bias \u2014 "
        "extra probability mass on certain ID tokens \u2014 and they find it "
        "across 20 LLMs and three benchmarks, not well fixed by chain-of-thought. "
        "Their debiasing method, PriDe, estimates the prior by PERMUTING OPTION "
        "CONTENTS on a few samples \u2014 the same intervention this probe uses to "
        "detect the bias.\n\n"
        "Concretely, for each MCQ item (an item carrying meta['options'] or "
        "meta['choices']) the probe asks the model twice: once in the original "
        "option order and once with the options permuted by a fixed seed. It "
        "parses the model's chosen letter back to the option CONTENT (with "
        "fallbacks to matching the answer text), then checks whether the chosen "
        "content is the same across the two orderings. The fraction of MCQ items "
        "whose selected content flips is the order-bias flip rate. A position- or "
        "label-robust model picks the same answer regardless of arrangement "
        "(flip rate \u2248 0); a high flip rate means the score is contaminated by "
        "presentation and MCQ comparisons are unsafe.\n\n"
        "Scope and honesty: this assesses the validity of an MCQ eval RESULT \u2014 "
        "whether option arrangement, not knowledge, drives the score \u2014 not the "
        "safety of the model. It detects INSTABILITY under reordering, which is "
        "necessary but not sufficient for unbiased scoring (a model could be "
        "stably wrong, or stably biased toward the content now in a favored slot); "
        "the principled fix is content-permutation debiasing (PriDe) or "
        "permutation-averaged scoring, of which this probe is the diagnostic half. "
        "It is the MCQ, item-side analogue of the judge's position bias measured "
        "in judge_swap, and complements prompt_format_sensitivity (wrapper "
        "templates) rather than duplicating it."
    ),
    references=[
        Reference("Zheng, Zhou, Meng, Zhou, Huang",
                  "Large Language Models Are Not Robust Multiple Choice Selectors",
                  "ICLR (Spotlight)", 2024, arxiv="2309.03882",
                  note="Identifies MCQ 'selection bias' (a prior for option IDs like 'A'), traces it to token bias, and debiases by permuting option contents \u2014 the mechanism and the intervention this probe uses."),
        Reference("Pezeshkpour, Hruschka",
                  "Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions",
                  "NAACL Findings", 2024, arxiv="2308.11483",
                  note="Quantifies the effect: reordering options shifts accuracy by ~13\u201375% across benchmarks \u2014 direct evidence that MCQ scores can be a position artifact."),
    ],
    math=[
        MathBlock(
            label="Order-bias flip rate (selected content changes on reorder)",
            html=(
                '<span class="mrow">flip_rate = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i&isin;M</sub> <b>1</b>[ choice(q<sub>i</sub>, O<sub>i</sub>) &ne; choice(q<sub>i</sub>, &pi;(O<sub>i</sub>)) ]</span>'
            ),
            latex=r"\mathrm{flip\_rate}=\frac1n\sum_{i\in M}\mathbf{1}\!\left[\mathrm{choice}(q_i,O_i)\neq\mathrm{choice}(q_i,\pi(O_i))\right]",
        ),
        MathBlock(
            label="Choice is compared by CONTENT, not letter",
            html=(
                '<span class="mrow">choice(q, O) = the option CONTENT the model selects '
                'when shown options O ,&nbsp; &pi; = a fixed seeded permutation</span>'
            ),
            latex=r"\mathrm{choice}(q,O)\in O\ \text{(the selected option's text)},\quad "
                  r"\pi=\text{seeded permutation of the options}",
        ),
    ],
    terms=[
        ("M", "the sampled MCQ items (meta['options']/['choices'], >=2 options); n = items with a PARSABLE choice in BOTH orderings; the rest are counted 'dropped' (unparseable) and reported, not silently ignored"),
        ("q\u1d62", "the i-th question text"),
        ("O\u1d62", "the original ordered list of options for item i"),
        ("\u03c0(O\u1d62)", "the options permuted by a fixed seed (seed + item index)"),
        ("choice(q, O)", "the option CONTENT the model picks given options O \u2014 parsed from its letter answer, with fallbacks to matching the final answer / option text"),
        ("flip", "the selected CONTENT differs between O\u1d62 and \u03c0(O\u1d62) (compared normalized, so a letter change with same content is NOT a flip)"),
        ("flip_rate", "fraction of comparable MCQ items that flip (the probe's score) \u2014 the share of MCQ answers driven by arrangement"),
    ],
    thresholds=[
        Threshold("flip_rate \u2265 0.20", "high",
                  "At least 1 in 5 MCQ answers changes under reordering \u2014 the score is substantially a position/label artifact; debias (permutation-average) before trusting or comparing MCQ accuracy."),
        Threshold("0.10 \u2264 flip_rate < 0.20", "medium",
                  "A meaningful order effect; report permutation-averaged accuracy rather than a single arrangement."),
        Threshold("flip_rate < 0.10 (with MCQ items measured)", "low",
                  "Choices are largely stable under reordering on this sample; MCQ scoring is robust here."),
        Threshold("no MCQ items (no options metadata)", "info",
                  "The dataset has no multiple-choice items to test (map an options/choices column under Pull, or the eval has none)."),
        Threshold("MCQ present but choice unparseable on all items", "info",
                  "Options exist but the model's selection could not be parsed on any item (reported as 'dropped'); no order-bias signal -- check the model's answer format."),
    ],
    effect=(
        "Ensure MCQ accuracy reflects knowing the answer, not the slot or letter "
        "it was placed in. By asking each MCQ item in two option orders and "
        "checking whether the chosen CONTENT changes, the probe exposes when an "
        "MCQ score \u2014 and rankings built from it \u2014 is driven by selection/"
        "position bias rather than capability."
    ),
    reading=(
        "flip_rate \u2248 0 means the model picks the same content however the "
        "options are arranged: MCQ accuracy is robust and comparable. A high flip "
        "rate means arrangement is steering answers, so the single-order score is "
        "partly an artifact \u2014 especially on items where the model is "
        "uncertain (Pezeshkpour & Hruschka) \u2014 and the remedy is to average "
        "over option permutations or apply content-permutation debiasing (PriDe; "
        "Zheng et al.). Compared by CONTENT not letter, so a model that always "
        "says 'A' will flip heavily here, exposing exactly the selection bias "
        "those papers describe."
    ),
    caveats=(
        "(a) Instability \u2260 unbiasedness: a low flip rate means stable, not "
        "necessarily correct or unbiased \u2014 a model could stably prefer "
        "whatever content lands in a favored slot, which a single permutation may "
        "not reveal; permutation-AVERAGED accuracy is the fuller treatment. (b) "
        "Single permutation: only one reorder is compared per item (seeded), so "
        "flip_rate is a lower-variance but coarse estimate \u2014 the literature "
        "cycles all positions; more permutations tighten it at higher cost "
        "(metered/cached by the CachingClient). (c) Parsing dependence: the choice "
        "is recovered from the model's letter (then answer-text fallbacks); a "
        "verbose or malformed answer that cannot be parsed in either ordering is "
        "dropped from n, which can bias the sample toward compliant items. (d) "
        "Needs MCQ metadata: items without an options/choices list are silently "
        "skipped, so this probe is INFO on free-form benchmarks. (e) One sample at "
        "temperature 0 per ordering: with a stochastic model some flips are "
        "sampling noise (see self_consistency), and small n is noisy (see "
        "statistical_power). (f) Distinct from judge_swap (judge position bias) "
        "and prompt_format_sensitivity (wrapper templates)."
    ),
    code_refs=["trust_the_eval.perturb.permute_options",
               "trust_the_eval.grading.extract_final"],
)

OptionOrderBias.TUNABLES = {'high': {'default': 0.2, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'flip rate >= this -> HIGH'}, 'medium': {'default': 0.1, 'min': 0, 'max': 1, 'step': 0.01, 'help': '>= this -> MEDIUM'}, 'sample_size': {'default': 40, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
