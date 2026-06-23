from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..evidence import trim
from ..finding import Finding, Severity
from ..grading import default_grader
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class ModelDrift(Probe):
    """Closed models change under the same name. Re-run a sample on the CURRENT
    model and compare correctness to the eval's recorded responses: if it shifted,
    the published score has expired. Needs stored responses to compare against."""
    id = "model_drift"
    name = "Model drift / temporal validity"
    paper_priority = "Shift 2"
    requires_model = True

    def __init__(self, sample_size: int = 40, seed: int = 0):
        self.sample_size, self.seed = sample_size, seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        rows, n, changed = [], 0, 0
        for idx, it in subsample(artifact.items, self.sample_size, self.seed):
            if it.response is None or not it.answer.strip() or not it.question.strip():
                continue
            n += 1
            then_ok = default_grader(it.response, it.answer)
            now_ok = default_grader(model.complete(it.question, temperature=0.0), it.answer)
            if then_ok != now_ok:
                changed += 1
                rows.append({"item": idx, "question": trim(it.question),
                             "was_correct": then_ok, "now_correct": now_ok})
        if n == 0:
            return [Finding(self.id, Severity.INFO,
                            "no stored responses to compare against current model",
                            evidence={"comparable": False})]
        rate = changed / n
        sev = (Severity.HIGH if rate >= self.tune("high") else Severity.MEDIUM if rate >= self.tune("medium")
               else Severity.LOW)
        return [Finding(self.id, sev,
                        f"{rate:.0%} of items changed correctness vs recorded run "
                        f"({changed}/{n}) — published score may have drifted",
                        score=round(rate, 3),
                        otel_attributes={"gen_ai.eval.trust.drift.change_rate": round(rate, 3)},
                        evidence={"compared": n, "changed": changed, "examples": rows[:5]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

ModelDrift.DOC = ProbeDoc(
    science=(
        "A benchmark score is attached to a model identifier (e.g. "
        "'gpt-4-0613'), but behind a commercial API the system serving that name "
        "can change without notice. When it does, a previously published score "
        "stops describing the model you query today \u2014 it has expired. Chen, "
        "Zaharia & Zou (2023) documented this directly: evaluating the March vs "
        "June 2023 versions of GPT-3.5 and GPT-4 on the same tasks, they found "
        "large behaviour shifts \u2014 GPT-4's accuracy at identifying prime vs "
        "composite numbers fell from 84% to 51% over three months \u2014 and "
        "concluded that the behaviour of the 'same' LLM service can change "
        "substantially in a short time, motivating continuous monitoring. Pradeep "
        "et al. (2023) make the methodological consequence explicit: results from "
        "proprietary models behind opaque API endpoints are non-reproducible and "
        "non-deterministic, which threatens the validity of anything built on "
        "them.\n\n"
        "This probe measures temporal validity by re-running a deterministic "
        "sample of the eval's items on the CURRENT model and comparing per-item "
        "correctness to the responses recorded in the artifact. The quantity is a "
        "correctness-change rate: the fraction of items whose pass/fail status "
        "flips between the recorded run and now. A high rate means the model that "
        "produced the stored score is not the model answering today, so the "
        "published number should be re-measured before it is trusted or compared. "
        "Crucially it requires stored responses to compare against \u2014 without "
        "them there is no 'then' to compare 'now' to, and the probe abstains.\n\n"
        "Scope and honesty: this assesses the validity of a previously recorded "
        "eval RESULT over time, not the safety of the model. The change rate is "
        "symmetric \u2014 it counts both regressions (was right, now wrong) and "
        "improvements (was wrong, now right); drift is drift, and the evidence "
        "rows record the direction per item. It does not attribute a cause "
        "(version bump, infrastructure change, decoding/temperature differences, "
        "or grader noise), only that the score no longer reproduces. Because "
        "correctness is judged at temperature 0 with a single sample, a small "
        "change rate can be ordinary nondeterminism rather than a true update \u2014"
        " see the caveats and pair with self_consistency and provenance_repro."
    ),
    references=[
        Reference("Chen, Zaharia, Zou",
                  "How Is ChatGPT's Behavior Changing over Time?",
                  "arXiv (Harvard Data Science Review)", 2023, arxiv="2307.09009",
                  note="Direct evidence of drift: large behaviour/accuracy shifts in GPT-3.5/GPT-4 between March and June 2023 under the same name \u2014 the phenomenon this probe detects."),
        Reference("Pradeep, Sharifymoghaddam, Lin",
                  "RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models",
                  "arXiv", 2023, arxiv="2309.15088",
                  note="Argues results from proprietary models behind opaque APIs are non-reproducible and non-deterministic \u2014 the validity stakes of model drift."),
    ],
    math=[
        MathBlock(
            label="Correctness-change rate (recorded run vs current model)",
            html=(
                '<span class="mrow">change_rate = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i&isin;S</sub> <b>1</b>[ then_ok<sub>i</sub> &ne; now_ok<sub>i</sub> ]</span>'
            ),
            latex=r"\mathrm{change\_rate}=\frac1n\sum_{i\in S}\mathbf{1}\!\left[\mathrm{then\_ok}_i\neq\mathrm{now\_ok}_i\right]",
        ),
        MathBlock(
            label="Per-item correctness (then = recorded, now = current)",
            html=(
                '<span class="mrow">then_ok<sub>i</sub> = grade(r<sub>i</sub><sup>rec</sup>, g<sub>i</sub>)'
                ' ,&nbsp;&nbsp; now_ok<sub>i</sub> = grade(model(q<sub>i</sub>; T=0), g<sub>i</sub>)</span>'
            ),
            latex=r"\mathrm{then\_ok}_i=\mathrm{grade}(r_i^{\mathrm{rec}},g_i),\quad "
                  r"\mathrm{now\_ok}_i=\mathrm{grade}(\mathrm{model}(q_i;T{=}0),g_i)",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled items that have a stored response, a non-empty question AND a non-empty gold (size n)"),
        ("n", "number of comparable items (those with a recorded response to compare against); sample_size default 40"),
        ("q\u1d62 , g\u1d62", "the i-th item's question and gold answer"),
        ("r\u1d62^rec", "the response recorded in the artifact (the 'then' run)"),
        ("then_ok\u1d62", "whether the recorded response was graded correct"),
        ("now_ok\u1d62", "whether the current model's fresh temperature-0 answer is graded correct"),
        ("change_rate", "fraction of items whose correctness flipped (the probe's score) \u2014 symmetric over regressions and improvements"),
    ],
    thresholds=[
        Threshold("change_rate \u2265 0.15", "high",
                  "More than ~1 in 7 items flip correctness vs the recorded run \u2014 the published score likely no longer describes the current model; re-measure before trusting or comparing it."),
        Threshold("0.05 \u2264 change_rate < 0.15", "medium",
                  "A non-trivial drift signal; some of it may be nondeterminism, so re-run and corroborate before acting."),
        Threshold("change_rate < 0.05", "low",
                  "Current behaviour closely reproduces the recorded run on this sample; the score is temporally stable here."),
        Threshold("no comparable items (no stored responses)", "info",
                  "The artifact has no recorded responses to compare against, so temporal drift cannot be assessed."),
    ],
    effect=(
        "Tell whether a previously recorded eval score still describes the model "
        "you can query now. By replaying the eval's own items against the current "
        "endpoint and comparing per-item correctness, the probe turns 'the API "
        "may have changed' into a concrete change rate with the specific flipped "
        "items attached \u2014 so a stale leaderboard number is caught before it is "
        "reused."
    ),
    reading=(
        "change_rate \u2248 0 means the current model reproduces the recorded "
        "results on this sample: the score is still valid in time. A high "
        "change_rate means today's model behaves differently from the one that "
        "produced the stored responses, so the published number is stale \u2014 "
        "re-run the full eval rather than citing the old score, and never compare "
        "a fresh model against a competitor's months-old number. The flips are "
        "symmetric (the model may have improved or regressed); the evidence rows "
        "show which items went was_correct\u2192now_wrong and vice-versa so you can "
        "see the direction and magnitude, not just the rate."
    ),
    caveats=(
        "(a) Nondeterminism vs true drift: correctness 'now' is one temperature-0 "
        "sample; even at T=0 many APIs are not bit-reproducible, so a small "
        "change_rate can be sampling noise rather than a model update \u2014 "
        "corroborate with self_consistency (intra-run stability) and "
        "provenance_repro (re-run determinism). (b) Confounded cause: the probe "
        "detects that the score changed, not why \u2014 a version bump, a serving/"
        "infra change, different decoding settings between the recorded run and "
        "now, or grader sensitivity all register as drift. (c) Depends on stored "
        "responses AND gold: items lacking either are skipped, so change_rate is "
        "computed only over comparable items and a small n is noisy (read it with "
        "statistical_power). (d) Apples-to-apples requires the same prompting as "
        "the recorded run; if the artifact's responses came from a different "
        "harness/prompt, some 'drift' is really a harness difference \u2014 the "
        "fair use is comparing the same model name across time under the same "
        "protocol."
    ),
    code_refs=["trust_the_eval.grading.default_grader",
               "trust_the_eval.sampling.subsample"],
)

ModelDrift.TUNABLES = {'high': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'change rate >= this -> HIGH'}, 'medium': {'default': 0.05, 'min': 0, 'max': 1, 'step': 0.01, 'help': '>= this -> MEDIUM'}, 'sample_size': {'default': 40, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
