from __future__ import annotations
import re
from typing import Optional

from ..artifact import EvalArtifact
from ..evidence import trim
from ..finding import Finding, Severity
from ..grading import gold_choices, normalize
from ..probe import ModelClient, Probe, register

_LEAK_MARKERS = ("the answer is", "gold:", "correct answer:", "ground truth")


@register
class RewardHackingEval(Probe):
    """DETECTION ONLY (never a recipe). Flags items the eval scored as PASS that
    fail an independent semantic re-check, or where the answer was leaked in the
    prompt/metadata — i.e. the score was obtained by exploiting the measurement
    rather than doing the task."""
    id = "reward_hacking_eval"
    name = "Reward-hacking of the eval (detection)"
    paper_priority = "P3 / Shift 1"
    requires_model = False

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        rows, n_pass, n_leak, n_empty = [], 0, 0, 0
        markers = ("the answer is", "gold", "correct answer", "ground truth")
        for idx, it in enumerate(artifact.items):
            gold = it.answer or ""
            gnorm = normalize(gold)
            nq = normalize(it.question or "")
            # Leak detection is limited to DISTINCTIVE free-text golds. MCQ options
            # and bare numbers recur in prompts and few-shot exemplars, so a
            # marker+value match there is usually coincidence, not a leak. A long
            # free-text answer appearing right after a reveal marker is a real leak.
            # (Scorer disagreement on a graded item is owned by answer_extraction_audit.)
            is_free = (not gold_choices(gold)
                       and not re.fullmatch(r"-?\d[\d,]*(\.\d+)?", gold.strip())
                       and len(gnorm) >= 6)
            leaked = bool(is_free and any(
                re.search(re.escape(mk) + r"[\s:=.\-]*" + re.escape(gnorm) + r"\b", nq)
                for mk in markers))
            if leaked:
                n_leak += 1
                rows.append({"item": idx, "type": "answer_leaked_in_prompt",
                             "question": trim(it.question)})
            if it.score is None or it.response is None or not gold.strip():
                continue
            if it.score >= self.tune("correct_cutoff"):
                n_pass += 1
                if not normalize(it.response):              # empty/blank response scored PASS
                    n_empty += 1
                    rows.append({"item": idx, "type": "empty_response_scored_pass",
                                 "response": trim(it.response), "gold": trim(it.answer)})
        sev = (Severity.HIGH if n_leak else Severity.MEDIUM if n_empty else Severity.LOW)
        return [Finding(self.id, sev,
                        f"{n_leak} items leak the answer in the prompt; "
                        f"{n_empty}/{n_pass} passes have an empty response scored correct",
                        score=round((n_leak + n_empty) / max(1, len(artifact.items)), 3),
                        otel_attributes={"gen_ai.eval.trust.reward_hacking.answer_leaks": n_leak,
                                         "gen_ai.eval.trust.reward_hacking.empty_passes": n_empty},
                        evidence={"passes": n_pass, "answer_leaks": n_leak,
                                  "empty_passes": n_empty, "examples": rows[:5]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

RewardHackingEval.DOC = ProbeDoc(
    science=(
        "An evaluation score is a proxy for a capability, and any proxy can be "
        "satisfied without achieving the thing it stands for. This is Goodhart's "
        "law \u2014 'when a measure becomes a target, it ceases to be a good "
        "measure' (Strathern, 1997, after Goodhart, 1975). In AI the failure has "
        "two named forms: specification gaming, behaviour that satisfies the "
        "literal objective while missing its intent (Krakovna et al., 2020), and "
        "reward hacking, formalized by Skalse et al. (2022) as optimizing an "
        "imperfect proxy reward in a way that lowers the true reward. Gao et al. "
        "(2022) show empirically that pushing on a proxy (a reward model) past a "
        "point degrades ground-truth performance \u2014 over-optimization. Applied "
        "to evaluations, the worry is concrete: a response can clear the eval's "
        "scorer without genuinely solving the task, so the headline pass rate "
        "over-states capability.\n\n"
        "This probe targets that specific validity threat \u2014 the eval being "
        "gamed \u2014 with two static, model-free checks on the recorded results. "
        "(1) Answer leakage: it flags items whose prompt reveals the gold answer "
        "right after a reveal marker ('the answer is', 'gold', 'correct answer', "
        "'ground truth'). This is restricted to distinctive free-text golds, "
        "because multiple-choice options and bare numbers recur in prompts and "
        "few-shot exemplars by chance \u2014 a leaked answer can be 'passed' by "
        "copying, not solving. (2) Empty passes: it flags items the eval marked "
        "PASS whose response is empty or blank after normalization \u2014 the scorer "
        "credited a non-answer. Scorer-vs-regrade DISAGREEMENT on graded items "
        "(a lenient scorer crediting a wrong answer) is deliberately NOT counted "
        "here; that is answer_extraction_audit's job, so the two probes never "
        "double-count. Both are signatures of a score earned by exploiting the "
        "measurement rather than solving the task.\n\n"
        "Scope and honesty \u2014 this matters most here: the probe is strictly "
        "DETECTION-ONLY. It surfaces eval items that look gamed so an evaluator "
        "can fix the scorer or remove the leak; it does not, and must not, "
        "describe how to make a model game an eval, and the tool's contribution "
        "guidelines reject any such request. The grader-strictness gap is a "
        "heuristic, not proof of intent: a lenient-only pass can also be a genuine "
        "solve that the strict extractor mis-parsed, and a leak marker can appear "
        "innocently. Read flagged items by hand. The check also only sees the "
        "scoring artifacts present in the log \u2014 environment-level exploits in "
        "agentic evals are out of this static probe's reach."
    ),
    references=[
        Reference("Skalse, Howe, Krasheninnikov, Krueger",
                  "Defining and Characterizing Reward Hacking",
                  "NeurIPS", 2022, arxiv="2209.13085",
                  note="First formal definition of reward hacking: optimizing an imperfect proxy reward can lower the true reward \u2014 the phenomenon this probe screens an eval for."),
        Reference("Krakovna, Uesato, Mikulik, Rahtz, Everitt, Kumar, Kenton, Leike, Legg",
                  "Specification Gaming: the Flip Side of AI Ingenuity",
                  "DeepMind Safety Research (blog)", 2020,
                  url="https://deepmindsafetyresearch.medium.com/specification-gaming-the-flip-side-of-ai-ingenuity-c85bdb0deeb4",
                  note="Canonical framing and example catalogue: satisfying the literal objective while missing its intent \u2014 exactly what a 'passed but not solved' item is."),
        Reference("Gao, Schulman, Hilton",
                  "Scaling Laws for Reward Model Overoptimization",
                  "ICML / arXiv", 2022, arxiv="2210.10760",
                  note="Empirically characterizes over-optimization (Goodhart in ML): pushing on a proxy past a point degrades true performance \u2014 why a high proxy score need not mean capability."),
        Reference("Strathern (after Goodhart, 1975)",
                  "'Improving Ratings': Audit in the British University System (the modern statement of Goodhart's law)",
                  "European Review, 5(3), 305\u2013321", 1997,
                  url="https://doi.org/10.1002/(SICI)1234-981X(199707)5:3<305::AID-EURO184>3.0.CO;2-4",
                  note="Source of 'when a measure becomes a target, it ceases to be a good measure' \u2014 the conceptual root of why eval scores can be gamed."),
    ],
    math=[
        MathBlock(
            label="Empty-pass count (empty/blank response scored as PASS)",
            html=(
                '<span class="mrow">n<sub>empty</sub> = &Sigma;<sub>i: PASS</sub> '
                '<b>1</b>[ norm(r<sub>i</sub>) = &empty; ]</span>'
            ),
            latex=r"n_{\mathrm{empty}}=\sum_{i:\,\mathrm{PASS}}\mathbf{1}[\mathrm{norm}(r_i)=\varnothing]",
        ),
        MathBlock(
            label="Answer-leak count (free-text gold right after a reveal marker)",
            html=(
                '<span class="mrow">n<sub>leak</sub> = &Sigma;<sub>i</sub> '
                '<b>1</b>[ free(g<sub>i</sub>) &and; (&exist; m&isin;M : "m &hellip; g<sub>i</sub>" &sub; norm(q<sub>i</sub>)) ]</span>'
            ),
            latex=r"n_{\mathrm{leak}}=\sum_i \mathbf{1}\!\left[\mathrm{free}(g_i)\wedge \exists m\in M:\ m\!\cdot\! g_i\subseteq \mathrm{norm}(q_i)\right]",
        ),
    ],
    terms=[
        ("PASS item", "an item the eval recorded as correct: score \u2265 0.5 with a stored response and a non-empty gold"),
        ("r\u1d62 , g\u1d62", "the stored model response and the gold answer for item i"),
        ("q\u1d62", "the item's prompt text (searched for leak markers and the gold)"),
        ("M", "reveal markers {'the answer is', 'gold', 'correct answer', 'ground truth'}"),
        ("free(g)", "gold is a distinctive free-text answer (not MCQ, not a bare number, >= 6 chars)"),
        ("n_pass", "number of PASS items examined"),
        ("n_empty", "PASS items whose response is empty/blank after normalization -- the scorer was gamed"),
        ("n_leak", "items where a free-text gold appears directly after a reveal marker in the prompt"),
    ],
    thresholds=[
        Threshold("any answer leak (n_leak > 0)", "high",
                  "A free-text answer is revealed in the prompt -- the item can be passed by copying, not solving. Remove the leak."),
        Threshold("empty passes (n_empty > 0)", "medium",
                  "Empty/blank responses were scored as correct -- the scorer is gamed; fix it and re-score."),
        Threshold("no leaks and no empty passes", "low",
                  "No answer leakage or empty-response passes detected. (Scorer-vs-regrade disagreement is reported by answer_extraction_audit.)"),
    ],
    effect=(
        "Ensure a PASS means the task was solved, not that the scorer (or a leaky "
        "prompt) was exploited. The probe flags items whose "
        "prompt hands the answer to the model, and PASS items with an empty response, "
        "turning a vague worry about "
        "Goodharting into specific, fixable items."
    ),
    reading=(
        "No leaks and no empty passes means nothing here undermines the score: the "
        "score is trustworthy on this axis. A leak means the item is passable by copying, and an empty pass means the scorer is broken; the headline "
        "may then over-state the true "
        "capability, and the fix is to remove the leak / fix the "
        "scorer. Any answer leak is treated as high severity on its "
        "own, because a single leaked item proves the benchmark can be passed "
        "without solving. Always open the flagged items (returned as evidence): "
        "some lenient-only passes are real solves the strict parser missed, so "
        "this is a triage signal, not a verdict."
    ),
    caveats=(
        "Detection-only, and a heuristic. (a) False positives: a flagged leak "
        "can occur innocently when a reveal marker sits near a matching string (e.g. unusual "
        "answer formatting), and a leak marker can occur innocently in prose \u2014 "
        "hand-review is required. (b) False negatives: the probe only sees scoring "
        "artifacts in the log; it cannot detect environment- or tool-level exploits "
        "in agentic evals, prompt-injection of the grader, or gaming that still "
        "yields the correct final token. (c) It depends on stored responses AND "
        "scores; items missing either are skipped, so the checks run only "
        "over auditable passes. (d) Leak detection is conservative -- distinctive free-text golds only -- so systematic "
        "numeric/MCQ leakage is not flagged; the empty-pass check is exact, and "
        "for free-form generation, pair this "
        "with judge_swap and human review. (e) Strictly a validity probe: it never "
        "instructs how to game an eval, only flags where one may have been gamed."
    ),
    code_refs=["trust_the_eval.grading.gold_choices",
               "trust_the_eval.grading.normalize"],
)

RewardHackingEval.TUNABLES = {'correct_cutoff': {'default': 0.5, 'min': 0, 'max': 1, 'step': 0.05, 'help': 'score >= this counts as a PASS'}}
