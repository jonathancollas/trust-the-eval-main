from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..evidence import trim
from ..finding import Finding, Severity
from ..grading import grade_task_aware
from ..probe import ModelClient, Probe, register


@register
class AnswerExtractionAudit(Probe):
    """Compare the eval's recorded score to a robust re-grade of the SAME stored
    responses. Disagreements are mis-scoring (a validity leak in the scorer),
    needs no model calls when responses are present in the log."""
    id = "answer_extraction_audit"
    name = "Answer-extraction / scoring audit"
    paper_priority = "P5"
    requires_model = False

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        rows, n, fn, fp, skipped = [], 0, 0, 0, 0
        for idx, it in enumerate(artifact.items):
            if it.response is None or it.score is None or not it.answer.strip():
                continue
            gradeable, ref_correct = grade_task_aware(it.response, it.answer)
            if not gradeable:
                skipped += 1
                continue
            n += 1
            eval_correct = it.score >= self.tune("correct_cutoff")
            if ref_correct and not eval_correct:
                fn += 1
                rows.append({"item": idx, "type": "false_negative",
                             "response": trim(it.response), "gold": trim(it.answer)})
            elif eval_correct and not ref_correct:
                fp += 1
                rows.append({"item": idx, "type": "false_positive",
                             "response": trim(it.response), "gold": trim(it.answer)})
        if n == 0:
            return [Finding(self.id, Severity.INFO,
                            f"no reliably-gradeable items ({skipped} free-text skipped \u2014 "
                            f"use the LLM-judge probe for those)",
                            evidence={"auditable": 0, "skipped_free_text": skipped})]
        rate = (fn + fp) / n
        sev = (Severity.HIGH if rate >= self.tune("high") else Severity.MEDIUM if rate >= self.tune("medium")
               else Severity.LOW)
        return [Finding(self.id, sev,
                        f"scoring disagreement {rate:.2f}: {fn} false-neg, {fp} false-pos of "
                        f"{n} task-gradeable items ({skipped} free-text skipped)",
                        score=round(rate, 3),
                        otel_attributes={"gen_ai.eval.trust.extraction.error_rate": round(rate, 3),
                                         "gen_ai.eval.trust.extraction.false_neg": fn,
                                         "gen_ai.eval.trust.extraction.false_pos": fp},
                        evidence={"audited": n, "skipped_free_text": skipped, "examples": rows[:5]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

AnswerExtractionAudit.DOC = ProbeDoc(
    science=(
        "Between a model's free-form output and a 0/1 score sits a fallible step: "
        "extracting the answer and matching it to the gold. If the score depends "
        "on how that extraction is done, the benchmark is partly measuring the "
        "PARSER, not the model. The effect is large and well documented. "
        "Alzahrani et al. (2024) show that on popular MCQ benchmarks like MMLU, "
        "merely changing the answer-selection method (or option order) shifts "
        "leaderboard rankings by up to eight positions, and recommend hybrid "
        "scoring. Yu et al. (2024), building xFinder, find that the regular-"
        "expression answer extractors used by standard harnesses (LM-Eval-Harness, "
        "OpenCompass) frequently misfire \u2014 a correct response in a non-"
        "standard format is scored wrong \u2014 and that models can even shape "
        "outputs to fit the regex. Molfese et al. (2025) add that traditional "
        "exact-match strategies systematically UNDER-estimate capability, with a "
        "fundamental trade-off between strict formats and free-form reasoning. In "
        "all three, the scorer itself is a validity leak.\n\n"
        "This probe audits that leak directly, with no model calls, by re-grading "
        "the eval's OWN stored responses two ways and comparing to the recorded "
        "score. It re-grades each response with a TASK-AWARE grader (multiple-choice by option letter, numeric by final value), skipping free-text items that a "
        "heuristic cannot reliably adjudicate. Two "
        "disagreement types are counted: a FALSE NEGATIVE is a response the eval "
        "marked wrong but that re-grades as genuinely right (a strict/correct "
        "answer the original scorer fumbled \u2014 understating capability); a "
        "FALSE POSITIVE is a response the eval marked right that the task-aware re-grade marks WRONG "
        "(the recorded label over-credited a response the careful re-grade rejects, e.g. a gold string buried in "
        "the text rather than a clean answer \u2014 overstating it). The "
        "disagreement rate (false-neg + false-pos over audited items) is the "
        "score, with the offending items returned for inspection.\n\n"
        "Scope and honesty: this assesses the validity of the SCORING step \u2014 "
        "whether the recorded score faithfully reflects the stored responses "
        "\u2014 not the safety of the model. It is the audit complement of "
        "reward_hacking_eval (which uses the same lenient-vs-robust gap to find "
        "PASSES that were not clean solves); here both directions are checked "
        "against the eval's recorded label. Crucially it judges only RELATIVE to "
        "this tool's task-aware grader, which is still simple (option-letter / "
        "final-value matching) and skip free text \u2014 they are a "
        "reference point for detecting scorer disagreement, not a certified-"
        "correct oracle, and free-form generation needs a semantic or human check."
    ),
    references=[
        Reference("Alzahrani, Alyahya, Alnumay, AlRashed, et al.",
                  "When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards",
                  "ACL", 2024, arxiv="2402.01781",
                  note="Changing the answer-SELECTION method (not the model) moves MMLU-style rankings by up to 8 positions \u2014 direct evidence that the scorer/extractor, not capability, can drive the result."),
        Reference("Yu, Yang, Cao, Cai, et al. (xFinder)",
                  "xFinder: Large Language Models as Automated Evaluators for Reliable Evaluation",
                  "ICLR", 2024, arxiv="2405.11874",
                  note="Shows RegEx answer extractors in standard harnesses (LM-Eval-Harness, OpenCompass) frequently misfire, scoring correct-but-oddly-formatted answers wrong \u2014 the false-negative failure this probe catches."),
        Reference("Molfese, Conia, Orlando, Navigli, et al.",
                  "Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice Question Answering",
                  "arXiv", 2025, arxiv="2503.14996",
                  note="Finds traditional exact-match strategies systematically under-estimate capability and a strict-format vs free-form trade-off \u2014 motivating a robust re-grade against a lenient one."),
    ],
    math=[
        MathBlock(
            label="Scoring-disagreement rate (the score)",
            html=(
                '<span class="mrow">disagreement = '
                '<span class="frac"><span class="num">fn + fp</span><span class="den">n</span></span></span>'
            ),
            latex=r"\mathrm{disagreement}=\frac{\mathrm{fn}+\mathrm{fp}}{n}",
        ),
        MathBlock(
            label="False negatives & positives (recorded vs task-aware re-grade)",
            html=(
                '<span class="mrow">over gradeable items: fn = &Sigma; <b>1</b>[ correct &and; &not;eval_ok ] ,&nbsp; '
                'fp = &Sigma; <b>1</b>[ eval_ok &and; &not;correct ]</span>'
            ),
            latex=r"\mathrm{fn}=\sum_i \mathbf{1}[c_i\wedge\neg e_i],\quad "
                  r"\mathrm{fp}=\sum_i \mathbf{1}[e_i\wedge\neg c_i]\quad(\text{gradeable }i)",
        ),
    ],
    terms=[
        ("n", "auditable items: those with a stored response, a recorded score, and a non-empty gold"),
        ("r\u1d62 , g\u1d62", "the stored response and gold answer for item i"),
        ("eval_ok\u1d62", "the eval's recorded verdict for item i (score \u2265 0.5)"),
        ("regrade (task-aware)", "grade_task_aware: MCQ by option letters, numeric by final value; FREE TEXT is skipped"),
        ("gradeable / skipped", "items the regrade can adjudicate (MCQ/numeric) vs free-text items it skips"),
        ("fn (false negatives)", "regrade-correct but recorded wrong -- the scorer understated capability"),
        ("fp (false positives)", "recorded right but regrade-wrong -- the scorer overstated it"),
        ("disagreement", "(fn + fp)/n \u2014 the probe's score: how often the recorded score disagrees with a robust re-grade"),
    ],
    thresholds=[
        Threshold("disagreement \u2265 0.15", "high",
                  "More than ~1 in 7 recorded scores disagree with a robust re-grade of the same responses \u2014 the score is substantially a scoring artifact; fix answer extraction / grading and re-score before trusting the number."),
        Threshold("0.05 \u2264 disagreement < 0.15", "medium",
                  "A non-trivial scoring-disagreement rate \u2014 audit the flagged false-neg/false-pos items and tighten the extractor."),
        Threshold("disagreement < 0.05 (with items audited)", "low",
                  "The recorded scores mostly agree with a robust re-grade on this run; the scorer looks reliable here."),
        Threshold("no task-gradeable items, or no responses+scores", "info",
                  "Nothing the heuristic can adjudicate: items lack responses/scores, or all golds are free text (use the LLM-judge probe for those)."),
    ],
    effect=(
        "Ensure the recorded score reflects the model's actual responses, not the "
        "quirks of an answer extractor. By re-grading the eval's own stored "
        "responses strictly and leniently and comparing to the recorded label, the "
        "probe quantifies how many scores are wrong in each direction \u2014 "
        "correct answers marked wrong, and lenient matches credited as clean "
        "solves \u2014 and returns the specific items to fix."
    ),
    reading=(
        "disagreement \u2248 0 means the recorded scores agree with a robust "
        "re-grade: the scorer is faithful on this run. A high rate means the "
        "number is partly a scoring artifact. The two directions matter "
        "differently: false NEGATIVES (correct answers the extractor missed) mean "
        "the score UNDER-states capability and comparisons penalize models with "
        "non-standard formatting (Yu et al.; Molfese et al.); false POSITIVES "
        "(lenient-only passes) mean it OVER-states capability. The fix is a better "
        "extractor / hybrid scoring (Alzahrani et al.), then re-score \u2014 open "
        "the returned examples to see which way the scorer is failing. The score "
        "is the combined disagreement rate; read fn and fp separately to know the "
        "direction."
    ),
    caveats=(
        "(a) Relative to THIS tool's graders: 'robust' and 'lenient' are simple "
        "(extracted-final equality vs substring), so the audit detects "
        "DISAGREEMENT with them, not ground-truth correctness \u2014 they can "
        "themselves be wrong on free-form or semantically-equivalent answers, "
        "where a semantic or human grader is needed. (b) Short-answer/MCQ bias: "
        "the graders are tuned for extractable final answers; on open-ended "
        "generation both directions can misfire, so pair with judge_swap and human "
        "review. (c) Needs stored responses AND scores: a benchmark logging only "
        "scores (no responses) returns INFO \u2014 absence of audit, not a clean "
        "bill. (d) Direction asymmetry: false positives require the lenient match "
        "to succeed, so a response that passes the recorded score by some OTHER "
        "rule the eval used (not captured by lenient) is not flagged \u2014 the "
        "probe sees disagreement with its own graders, not with the eval's exact "
        "internal scorer. (e) Complement of reward_hacking_eval (same lenient-vs-"
        "robust gap, there only on PASSES to find gamed solves) and "
        "prompt_format_sensitivity (a 'give only the final answer' template "
        "interacts with extraction); small n makes the rate noisy (statistical_power)."
    ),
    code_refs=["trust_the_eval.grading.grade_task_aware",
               "trust_the_eval.grading.gold_choices",
               "trust_the_eval.grading.response_choices"],
)

AnswerExtractionAudit.TUNABLES = {'high': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'rate >= this -> HIGH'}, 'medium': {'default': 0.05, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'rate >= this -> MEDIUM'}, 'correct_cutoff': {'default': 0.5, 'min': 0, 'max': 1, 'step': 0.05, 'help': 'score >= this counts as correct'}}
