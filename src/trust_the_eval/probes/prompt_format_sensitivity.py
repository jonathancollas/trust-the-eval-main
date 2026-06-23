from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..grading import default_grader
from ..perturb import reformat_templates
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class PromptFormatSensitivity(Probe):
    """Re-ask a sample under equivalent templates; a large accuracy spread means
    the score reflects presentation, not capability."""
    id = "prompt_format_sensitivity"
    name = "Prompt-format sensitivity"
    paper_priority = "P5"
    requires_model = True

    def __init__(self, sample_size: int = 40, seed: int = 0):
        self.sample_size, self.seed = sample_size, seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        items = [(i, it) for i, it in subsample(artifact.items, self.sample_size, self.seed)
                 if it.question.strip() and it.answer.strip()]
        if not items:
            return [Finding(self.id, Severity.INFO, "no scorable items in sample")]
        per_template: dict[str, float] = {}
        names = [n for n, _ in reformat_templates(items[0][1].question)]
        for name in names:
            correct = 0
            for _, it in items:
                prompt = dict(reformat_templates(it.question))[name]
                if default_grader(model.complete(prompt, temperature=0.0), it.answer):
                    correct += 1
            per_template[name] = correct / len(items)
        spread = max(per_template.values()) - min(per_template.values())
        sev = (Severity.HIGH if spread >= self.tune("high") else Severity.MEDIUM if spread >= self.tune("medium")
               else Severity.LOW)
        return [Finding(self.id, sev,
                        f"accuracy spread {spread:.2f} across {len(names)} equivalent templates",
                        score=round(spread, 3),
                        otel_attributes={"gen_ai.eval.trust.format.spread_pts": round(spread * 100, 1)},
                        evidence={"per_template": {k: round(v, 3) for k, v in per_template.items()}})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

PromptFormatSensitivity.DOC = ProbeDoc(
    science=(
        "A capability score should depend on whether the model can do the task, "
        "not on the incidental template the question is wrapped in. But LLMs are "
        "strikingly sensitive to meaning-preserving formatting, so a benchmark "
        "that fixes one arbitrary template measures that template as much as the "
        "model. Sclar et al. (2024) quantified this: across plausible, "
        "semantically-equivalent prompt formats they observed performance swings "
        "of up to 76 accuracy points on a single model, sensitivity that persists "
        "with larger models, more few-shot examples, and instruction tuning \u2014 "
        "and, decisively for evaluation, format performance correlates only weakly "
        "between models, so comparing models under one fixed format is "
        "methodologically questionable. Mizrahi et al. (2024), over 6.5M "
        "instances, 20 LLMs and 39 tasks, reach the same conclusion at scale: "
        "different instruction templates change both absolute scores and the "
        "relative ranking of models, and they recommend reporting a performance "
        "spread over multiple paraphrases rather than a single-template number.\n\n"
        "This probe operationalizes that recommendation directly on the eval's own "
        "items. It re-asks a deterministic sample under a small set of equivalent "
        "presentations of the same question \u2014 plain; a 'Question:/Answer:' "
        "frame; an 'answer concisely' instruction; and a 'give only the final "
        "answer' frame \u2014 grading each against the same gold. The accuracy "
        "SPREAD (best template minus worst) is the score: it is the performance "
        "spread of Mizrahi et al. computed locally. A large spread means the "
        "headline number is an artifact of the chosen wrapper and that "
        "single-format model comparisons are unsafe; a small spread means the "
        "score is robust to presentation.\n\n"
        "Scope and honesty: this assesses the validity of the evaluation result "
        "\u2014 whether incidental presentation moves the score \u2014 not the "
        "safety of the model. It varies wrapper templates, a deliberately narrow, "
        "meaning-preserving slice of prompt design; it does not vary few-shot "
        "ordering, system prompts, or MCQ option order (that is option_order_bias), "
        "and it is distinct from contamination_perturb, which paraphrases the "
        "question CONTENT to probe memorization rather than re-wrapping it. The "
        "spread is a lower bound on true format sensitivity, since only four "
        "templates are tried."
    ),
    references=[
        Reference("Sclar, Choi, Tsvetkov, Suhr",
                  "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design (FormatSpread)",
                  "ICLR", 2024, arxiv="2310.11324",
                  note="Up to 76 accuracy points of variation across equivalent prompt formats, and only weak between-model correlation \u2014 the core evidence that single-format scores are unreliable."),
        Reference("Mizrahi, Kaplan, Malkin, Dror, Shahaf, Stanovsky",
                  "State of What Art? A Call for Multi-Prompt LLM Evaluation",
                  "TACL", 2024, arxiv="2401.00595",
                  note="At 6.5M instances / 20 LLMs / 39 tasks: templates change absolute scores AND rankings; proposes the 'performance spread' over paraphrases \u2014 exactly this probe's metric."),
    ],
    math=[
        MathBlock(
            label="Per-template accuracy",
            html=(
                '<span class="mrow">acc(t) = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i&isin;S</sub> <b>1</b>[ grade(model(template<sub>t</sub>(q<sub>i</sub>); T=0), g<sub>i</sub>) ]</span>'
            ),
            latex=r"\mathrm{acc}(t)=\frac1n\sum_{i\in S}\mathbf{1}\!\left[\mathrm{grade}\big(\mathrm{model}(\mathrm{template}_t(q_i);T{=}0),\,g_i\big)\right]",
        ),
        MathBlock(
            label="Accuracy spread across equivalent templates (the score)",
            html=(
                '<span class="mrow">spread = max<sub>t&isin;T</sub> acc(t) &minus; min<sub>t&isin;T</sub> acc(t)</span>'
            ),
            latex=r"\mathrm{spread}=\max_{t\in T}\mathrm{acc}(t)-\min_{t\in T}\mathrm{acc}(t)",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled items with a non-empty question and gold (size n; sample_size default 40)"),
        ("n", "number of scored items per template"),
        ("T", "the set of equivalent templates: plain, qa ('Question:/Answer:'), instructed ('Answer concisely'), boxed ('Give only the final answer')"),
        ("template_t(q\u1d62)", "question i rendered under template t (meaning preserved, only the wrapper changes)"),
        ("g\u1d62", "the gold answer for item i (shared across templates)"),
        ("acc(t)", "accuracy under template t (each item asked once at temperature 0)"),
        ("spread", "max acc(t) \u2212 min acc(t) \u2014 the performance spread; the probe's score and the share of accuracy attributable to wrapper choice"),
    ],
    thresholds=[
        Threshold("spread \u2265 0.20", "high",
                  "Equivalent wrappers move accuracy by \u226520 points \u2014 the score largely reflects presentation, and single-format model comparisons on this benchmark are unsafe."),
        Threshold("0.10 \u2264 spread < 0.20", "medium",
                  "A meaningful format effect; report accuracy across templates rather than a single number."),
        Threshold("spread < 0.10", "low",
                  "Accuracy is stable across the tried templates; the score is largely robust to presentation on this sample."),
        Threshold("no scorable items", "info",
                  "The sample had no item with both a question and a gold answer, so format sensitivity cannot be computed."),
    ],
    effect=(
        "Ensure the reported score measures capability, not the arbitrary template "
        "it was collected under. By re-asking the eval's own items across "
        "equivalent presentations and reporting the best-minus-worst accuracy "
        "spread, the probe exposes when a benchmark number \u2014 and any ranking "
        "built from it at a single fixed format \u2014 is an artifact of "
        "presentation."
    ),
    reading=(
        "spread \u2248 0 means the four templates agree: the score is robust to "
        "presentation and safe to compare. A large spread means accuracy depends "
        "on the wrapper, so the single-template headline is optimistic-or-"
        "pessimistic by accident and, since format effects barely correlate across "
        "models (Sclar et al.), ranking models under one fixed format is "
        "unreliable \u2014 report the per-template breakdown (returned as evidence) "
        "or an average/spread over templates instead. The score is a lower bound: "
        "only four wrappers are tried, so a small spread does not prove full format "
        "robustness."
    ),
    caveats=(
        "(a) Lower bound: only four hand-chosen wrappers are tested; true "
        "sensitivity (FormatSpread explores hundreds) is at least this large, so a "
        "low spread is reassuring but not conclusive. (b) Wrapper-only: this "
        "varies the surrounding template, NOT few-shot example order, system "
        "prompts, or MCQ option order (option_order_bias) \u2014 and is distinct "
        "from contamination_perturb, which paraphrases question CONTENT for "
        "memorization. (c) Extraction confound: a template like 'give only the "
        "final answer' interacts with answer extraction, so part of a spread can "
        "be the grader/extractor rather than the model \u2014 corroborate with "
        "answer_extraction_audit. (d) One sample per (item, template) at "
        "temperature 0: with a stochastic model some spread is sampling noise "
        "(see self_consistency); and small n makes the spread itself noisy (see "
        "statistical_power). (e) Cost scales with the number of templates \u00d7 "
        "items; calls are metered and cached by the CachingClient."
    ),
    code_refs=["trust_the_eval.perturb.reformat_templates",
               "trust_the_eval.grading.default_grader"],
)

PromptFormatSensitivity.TUNABLES = {'high': {'default': 0.2, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'accuracy spread >= this -> HIGH'}, 'medium': {'default': 0.1, 'min': 0, 'max': 1, 'step': 0.01, 'help': '>= this -> MEDIUM'}, 'sample_size': {'default': 40, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
