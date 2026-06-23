from __future__ import annotations
from collections import Counter
from typing import Optional

from ..artifact import EvalArtifact
from ..evidence import trim
from ..finding import Finding, Severity
from ..grading import extract_final
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class ItemAmbiguity(Probe):
    """Flag ill-posed items: high dispersion of a capable model's answers across
    resamples (the question admits multiple defensible readings)."""
    id = "item_ambiguity"
    name = "Item ambiguity"
    paper_priority = "P5"
    requires_model = True

    def __init__(self, sample_size: int = 30, votes: int = 5, seed: int = 0):
        self.sample_size, self.votes, self.seed = sample_size, votes, seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        panel = [model] + list(artifact.metadata.get("panel") or [])
        cross = len(panel) >= 2
        rows, n_checked, n_amb = [], 0, 0
        for idx, it in subsample(artifact.items, self.sample_size, self.seed):
            if not it.question.strip():
                continue
            n_checked += 1
            if cross:
                # Item ambiguity is model-INDEPENDENT: flag iff distinct models
                # disagree with EACH OTHER (robust to any one model's decoding
                # noise). Each model votes by its own resample-majority.
                majs = []
                for m in panel:
                    fs = [extract_final(s) for s in m.sample(it.question, self.votes, temperature=self.tune("temp_cross"))]
                    fs = [f for f in fs if f]
                    if fs:
                        majs.append(max(set(fs), key=fs.count))
                if len(majs) >= 2 and len(set(majs)) >= 2:
                    n_amb += 1
                    rows.append({"item": idx, "question": trim(it.question),
                                 "model_majorities": majs})
            else:
                # Single-model fallback: NO convergence target -> nearly every
                # resample is a DIFFERENT answer. Separates a genuinely
                # under-specified item from a model merely noisy around one mode,
                # but cannot fully isolate ambiguity from model stochasticity
                # (supply metadata['panel'] of >=2 models for that).
                finals = [extract_final(s) for s in
                          model.sample(it.question, self.votes, temperature=self.tune("temp_single"))]
                finals = [f for f in finals if f]
                if len(finals) < 4:
                    continue
                distinct = len(set(finals))
                if distinct >= self.tune("distinct_min") and distinct >= self.tune("unique_frac") * len(finals):
                    n_amb += 1
                    rows.append({"item": idx, "question": trim(it.question),
                                 "distinct_answers": distinct, "samples": len(finals)})
        rate = (n_amb / n_checked) if n_checked else 0.0
        sev = (Severity.MEDIUM if rate >= self.tune("rate_medium") else Severity.LOW if n_checked else Severity.INFO)
        mode = ("cross-model disagreement" if cross else
                "single-model (no panel); supply metadata['panel'] of >=2 models "
                "to separate item ambiguity from model noise")
        return [Finding(self.id, sev,
                        f"ambiguous-item rate {rate:.2f} ({n_amb}/{n_checked}); {mode}",
                        score=round(rate, 3),
                        otel_attributes={"gen_ai.eval.trust.item_ambiguity.rate": round(rate, 3),
                                         "gen_ai.eval.trust.item_ambiguity.cross_model": cross},
                        evidence={"checked": n_checked, "ambiguous": n_amb,
                                  "cross_model": cross, "examples": rows[:5]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

ItemAmbiguity.DOC = ProbeDoc(
    science=(
        "Grading assumes each item has a single defensible answer. When an item is "
        "ill-posed \u2014 underspecified, multiply interpretable, or missing "
        "context \u2014 there is no fact of the matter for the gold to encode, so "
        "marking it right or wrong measures the question's ambiguity, not the "
        "model. Ambiguity is a documented, distinct failure of benchmark "
        "validity: Gema et al. (2024), re-annotating MMLU, separate errors into "
        "'incorrect ground truth' and 'ambiguous question' and find some subsets "
        "are dominated by the latter \u2014 ambiguity and mislabeling are different "
        "defects needing different fixes. More broadly, annotation artifacts and "
        "ill-specified items let models exploit shortcuts: Gururangan et al. "
        "(2018) famously showed a hypothesis-only classifier scores ~67% on SNLI "
        "(chance 33%), evidence that ill-posed items make benchmark numbers "
        "overstate genuine ability.\n\n"
        "To find ill-posed items without a human pass, this probe uses answer "
        "DISPERSION under resampling as a proxy for ambiguity, in the spirit of "
        "semantic uncertainty (Kuhn et al., 2023), where the spread of a model's "
        "resampled answers is an unsupervised reliability signal. The reasoning: a "
        "well-posed question pins a capable model to one answer across samples; a "
        "question that admits several defensible readings makes the model scatter "
        "across several distinct answers. High, MULTI-MODAL dispersion is the "
        "fingerprint of an ill-posed item.\n\n"
        "Concretely, over a deterministic sample the probe draws several samples "
        "per item at temperature 0.8, extracts each final answer, and computes the "
        "Shannon entropy of the answer distribution. An item is flagged ambiguous "
        "when entropy \u2265 1 bit AND there are \u2265 3 distinct answers \u2014 "
        "requiring both a high-entropy spread and genuine multi-modality, not just "
        "a single alternative. The ambiguous-item rate is the score; flagged "
        "items (with their distinct-answer count and entropy) are returned for "
        "review.\n\n"
        "Scope and honesty: this assesses the validity of the eval ITEMS \u2014 "
        "whether questions are well-posed enough to grade \u2014 not the safety of "
        "the model. It is the dispersion-as-ambiguity counterpart to "
        "self_consistency (which measures the SAME dispersion as run-to-run "
        "reliability) and the complement of label_error_audit (a CONFIDENT "
        "consensus AGAINST the gold = bad label; a SCATTERED consensus = ambiguous "
        "item). Crucially, high dispersion can also mean the model is simply weak "
        "on the item rather than the item being ambiguous, so the rate is a triage "
        "signal for human review, not a proof of ill-posedness."
    ),
    references=[
        Reference("Gema, Leang, Hong, Devoto, Mancino, Saxena, He, Zhao, Du, Minervini, et al.",
                  "Are We Done with MMLU?",
                  "NAACL", 2024, arxiv="2406.04127",
                  note="Distinguishes 'ambiguous question' from 'incorrect ground truth' as separate MMLU error types, with some subsets dominated by ambiguity \u2014 the exact defect this probe targets."),
        Reference("Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith",
                  "Annotation Artifacts in Natural Language Inference Data",
                  "NAACL", 2018, arxiv="1803.02324",
                  note="Shows ill-specified items let a hypothesis-only model reach ~67% on SNLI \u2014 evidence that poorly-posed items make benchmark scores overstate ability."),
        Reference("Kuhn, Gal, Farquhar",
                  "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation",
                  "ICLR (Oral)", 2023, arxiv="2302.09664",
                  note="Establishes the spread/entropy of resampled answers as an unsupervised uncertainty signal \u2014 the basis for using answer dispersion as an ambiguity proxy."),
    ],
    math=[
        MathBlock(
            label="Per-model majority & distinct-answer count (per item)",
            html=(
                "<span class=\"mrow\">maj<sub>m</sub>(i) = mode of model m's resampled finals on item i ,&nbsp; "
                'distinct<sub>i</sub> = |{ extracted finals of the primary model on item i }|</span>'
            ),
            latex=r"\mathrm{maj}_m(i)=\mathrm{mode}\{\text{finals of }m\text{ on }i\},\quad "
                  r"\mathrm{distinct}_i=|\{\text{primary finals on }i\}|",
        ),
        MathBlock(
            label="Ambiguity flag (cross-model primary) & rate (the score)",
            html=(
                '<span class="mrow">cross-model: ambiguous<sub>i</sub> = <b>1</b>[ |{ maj<sub>m</sub>(i) : m &isin; panel }| &ge; 2 ] '
                '(models disagree) ;&nbsp; single-model: <b>1</b>[ distinct<sub>i</sub> &ge; 3 &and; distinct<sub>i</sub> &ge; 0.8&middot;m<sub>i</sub> ] ;&nbsp; '
                'rate = &Sigma;<sub>i</sub> ambiguous<sub>i</sub> / n<sub>checked</sub></span>'
            ),
            latex=r"\text{cross: }\mathbf{1}[\,|\{\mathrm{maj}_m(i)\}|\ge 2\,];\ "
                  r"\text{single: }\mathbf{1}[\mathrm{distinct}_i\ge 3\wedge \mathrm{distinct}_i\ge 0.8\,m_i];\ "
                  r"\mathrm{rate}=\tfrac{\sum_i \mathrm{ambiguous}_i}{n_{\mathrm{checked}}}",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled items with a non-empty question (sample_size, default 30)"),
        ("votes", "samples drawn per item at temperature 0.8 (default 5)"),
        ("panel", "primary model + metadata['panel'] models; cross-model mode needs >= 2"),
        ("maj_m(i)", "model m's resample-majority answer on item i"),
        ("distinct_i", "number of DISTINCT extracted finals for the primary model on item i"),
        ("ambiguous_i (cross)", "1 if >= 2 panel models give DIFFERENT majority answers (item splits independent models)"),
        ("ambiguous_i (single)", "1 if distinct_i >= 3 AND >= 80% of resamples are unique (no convergence target); UNCONFIRMED -- may be model noise"),
        ("n_checked", "items with \u22652 valid extracted answers (those that can be assessed)"),
        ("rate", "fraction of checked items flagged ambiguous (the probe's score)"),
    ],
    thresholds=[
        Threshold("ambiguous-item rate >= 0.10", "medium",
                  "At least 1 in 10 items make >= 2 independent models disagree (cross-model), or leave a single model with no convergence target -- enough ill-posed items to distort the score; human-review and repair/remove."),
        Threshold("rate < 0.10 (with items checked)", "low",
                  "Few items show ambiguous, multi-modal answer dispersion on this sample; the questions look largely well-posed."),
        Threshold("no checkable items", "info",
                  "No items yielded \u22652 extractable answers, so item ambiguity cannot be assessed (e.g. empty or un-parseable outputs)."),
    ],
    effect=(
        "Find questions that are not well-posed enough to grade, by detecting "
        "where independent models disagree (or one model finds no convergence target) across distinct "
        "options. Separates 'the item is ambiguous' from 'the model is wrong' (and "
        "from 'the label is wrong'), and returns the offending items so they can "
        "be repaired or excluded before they distort the score."
    ),
    reading=(
        "rate \u2248 0 means resampled answers concentrate: items are well-posed and "
        "gradable. A high rate means many questions split a capable model across "
        "\u22653 defensible-looking answers \u2014 grading those items measures "
        "ambiguity, not capability, and inflates noise (and can let shortcuts "
        "inflate scores; Gururangan et al.). Treat it as triage: open the returned "
        "examples (distinct-answer count, entropy) and have a human decide if the "
        "item is truly ambiguous. Distinguish from neighbours: a CONFIDENT "
        "consensus against the gold is a label error (label_error_audit); a "
        "SCATTERED consensus is ambiguity (here); the same dispersion read as "
        "reproducibility is self_consistency."
    ),
    caveats=(
        "(a) Ambiguity vs incapacity: high dispersion can mean the model is simply "
        "weak on the item, not that the item is ill-posed \u2014 so the rate "
        "over-counts on hard-but-well-posed questions; human review is required "
        "the CROSS-MODEL mode (>= 2 models disagreeing) largely removes this confound; single-model results are UNCONFIRMED. (b) Extraction-bound: "
        "entropy is over extract_final outputs, so formatting variants the "
        "extractor doesn't normalize (e.g. '12' vs 'twelve', ordering, units) "
        "inflate distinct-answer counts and thus apparent ambiguity \u2014 a "
        "semantic-equivalence grouping (Kuhn et al.) is more faithful than "
        "exact-match on finals. (c) Thresholds and votes: the \u22651 bit and "
        "\u22653-distinct rule with votes=5 is coarse (entropy is quantized by the "
        "5 samples), so the flag is a heuristic, not a calibrated test; more votes "
        "tighten it at higher cost (metered/cached by the CachingClient). (d) "
        "Temperature 0.8 sets the dispersion floor: the same items would scatter "
        "less at lower temperature, so this measures ambiguity-as-seen-at-0.8, an "
        "upper-ish estimate. (e) Small n makes the rate noisy (read with "
        "statistical_power); and this is a validity-of-items signal, not a model "
        "quality score."
    ),
    code_refs=["trust_the_eval.grading.extract_final",
               "trust_the_eval.grading.extract_final",
               "trust_the_eval.sampling.subsample"],
)

ItemAmbiguity.TUNABLES = {'rate_medium': {'default': 0.1, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'ambiguous rate >= this -> MEDIUM'}, 'distinct_min': {'default': 3, 'min': 2, 'max': 20, 'step': 1, 'help': 'min distinct answers to flag'}, 'unique_frac': {'default': 0.8, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'distinct/samples >= this to flag (single-model)'}, 'temp_cross': {'default': 0.7, 'min': 0, 'max': 2, 'step': 0.05, 'help': 'temp for cross-model majorities'}, 'temp_single': {'default': 0.8, 'min': 0, 'max': 2, 'step': 0.05, 'help': 'temp for single-model resampling'}, 'sample_size': {'default': 30, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'votes': {'default': 5, 'min': 1, 'max': 20, 'step': 1, 'help': 'samples per item', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
