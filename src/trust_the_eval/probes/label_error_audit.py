from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..evidence import trim
from ..finding import Finding, Severity
from ..grading import default_grader, extract_final
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class LabelErrorAudit(Probe):
    """Detect suspected wrong/ambiguous GROUND TRUTH: where a strong adjudicator
    is self-consistent across samples yet consistently DISAGREES with the gold,
    the label is suspect (the eval, not the model, may be wrong)."""
    id = "label_error_audit"
    name = "Label / ground-truth error audit"
    paper_priority = "P5"
    requires_model = True

    def __init__(self, sample_size: int = 40, votes: int = 3, seed: int = 0):
        self.sample_size, self.votes, self.seed = sample_size, votes, seed

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        panel = [model] + list(artifact.metadata.get("panel") or [])
        cross = len(panel) >= 2
        rows, n_checked, n_suspect = [], 0, 0
        for idx, it in subsample(artifact.items, self.sample_size, self.seed):
            if not it.question.strip() or not it.answer.strip():
                continue
            n_checked += 1
            verdicts = []  # (resample-majority, is_self_consistent) per model
            for m in panel:
                fs = [extract_final(s) for s in m.sample(it.question, self.votes, temperature=self.tune("temp"))]
                fs = [f for f in fs if f]
                if not fs:
                    continue
                maj = max(set(fs), key=fs.count)
                verdicts.append((maj, fs.count(maj) / len(fs) >= self.tune("consistency")))
            if not verdicts:
                continue
            if cross:
                # Require INDEPENDENT models to AGREE with each other on an answer
                # that DISAGREES with the gold (Northcutt-style multi-model
                # confidence). A single confidently-wrong model cannot trigger this.
                majs = [m for m, _ in verdicts]
                consensus = max(set(majs), key=majs.count)
                agree = majs.count(consensus) / len(majs)
                if agree >= self.tune("agree") and not default_grader(f"answer: {consensus}", it.answer):
                    n_suspect += 1
                    rows.append({"item": idx, "question": trim(it.question),
                                 "gold": trim(it.answer), "model_consensus": consensus,
                                 "models_agreeing": f"{majs.count(consensus)}/{len(majs)}"})
            else:
                maj, consistent = verdicts[0]
                if consistent and not default_grader(f"answer: {maj}", it.answer):
                    n_suspect += 1
                    rows.append({"item": idx, "question": trim(it.question),
                                 "gold": trim(it.answer), "model_answer": maj})
        rate = (n_suspect / n_checked) if n_checked else 0.0
        if cross:
            sev = (Severity.HIGH if rate >= self.tune("cross_high") else Severity.MEDIUM if rate >= self.tune("cross_medium")
                   else Severity.LOW if n_checked else Severity.INFO)
            mode = "cross-model consensus disagrees with gold"
        else:
            # A single model cannot separate a wrong GOLD from a confidently-wrong
            # MODEL; report UNCONFIRMED candidates and never escalate to HIGH.
            sev = (Severity.MEDIUM if rate >= self.tune("single_medium") else Severity.LOW if n_checked else Severity.INFO)
            mode = ("single-model candidates (UNCONFIRMED \u2014 a confidently-wrong "
                    "model alone yields false positives; supply metadata['panel'] of >=2 models)")
        return [Finding(self.id, sev,
                        f"suspected label-error rate {rate:.2f} ({n_suspect}/{n_checked}); {mode}",
                        score=round(rate, 3),
                        otel_attributes={"gen_ai.eval.trust.label_error.rate": round(rate, 3),
                                         "gen_ai.eval.trust.label_error.cross_model": cross},
                        evidence={"checked": n_checked, "suspect": n_suspect,
                                  "cross_model": cross, "examples": rows[:5]})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

LabelErrorAudit.DOC = ProbeDoc(
    science=(
        "Every graded benchmark assumes its gold answers are correct, but that "
        "assumption frequently fails, and when it does the score is wrong no "
        "matter how good the model is. Northcutt, Athalye & Mueller (2021) "
        "algorithmically found, then human-verified, label errors in the test "
        "sets of ten of the most-used ML benchmarks \u2014 averaging at least 3.3% "
        "errors, e.g. ~6% of the ImageNet validation set \u2014 and showed the "
        "consequence is not cosmetic: correcting the labels can REVERSE model "
        "rankings (a ResNet-18 overtakes a ResNet-50), so a leaderboard built on a "
        "noisy test set can recommend the wrong model. The problem reaches "
        "LLM-era benchmarks directly: Gema et al. (2024), re-annotating MMLU, "
        "estimate ~6.5% of questions contain errors (57% in the Virology subset) "
        "and find the original scores diverge significantly from the corrected "
        "ones. Mislabeled gold caps measured accuracy and adds noise that "
        "compresses the gap between models.\n\n"
        "Detecting label errors without a second human pass needs a signal that "
        "separates 'the model is wrong' from 'the gold is wrong'. The insight "
        "this probe uses, in the spirit of confident learning (Northcutt et al., "
        "2021, the method behind the benchmark study), is confident disagreement: "
        "if a strong adjudicator is highly SELF-CONSISTENT on an item (its "
        "resampled answers agree) yet that confident consensus DISAGREES with the "
        "gold, the gold is the suspect. A one-off model error tends to be "
        "unstable across samples; a stable, repeated consensus against the label "
        "is the fingerprint of a bad label, not a bad guess.\n\n"
        "Concretely, over a deterministic sample the probe draws several "
        "adjudicator samples per item at temperature 0.7, extracts each final "
        "answer, and takes the modal answer. An item is flagged as a suspected "
        "label error when the adjudicator is consistent (the mode covers \u226580% "
        "of valid samples) AND the consensus is graded as disagreeing with the "
        "gold. The suspected-label-error rate is the score, and the flagged items "
        "(question, gold, model consensus) are returned for human review.\n\n"
        "Scope and honesty: this audits the validity of the eval's GROUND TRUTH "
        "\u2014 whether the answer key can be trusted \u2014 not the safety of the "
        "model. It is a triage signal, explicitly necessary-not-sufficient: a "
        "confident consensus can be confidently wrong (especially on contaminated "
        "or popular items the adjudicator memorized), so a flag means 'a human "
        "should check this label', never 'the label is definitely wrong'. It also "
        "cannot tell a wrong label from a genuinely ambiguous item (that is "
        "item_ambiguity's job), and it inherits the grader's matching limits."
    ),
    references=[
        Reference("Northcutt, Athalye, Mueller",
                  "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks",
                  "NeurIPS (Datasets & Benchmarks)", 2021, arxiv="2103.14749",
                  note="Human-verified label errors (avg \u22653.3%) in ten major benchmarks, and the key stakes: correcting labels can REVERSE model rankings \u2014 why a noisy answer key invalidates the score."),
        Reference("Gema, Leang, Hong, Devoto, Mancino, Saxena, He, Zhao, Du, Minervini, et al.",
                  "Are We Done with MMLU?",
                  "NAACL", 2024, arxiv="2406.04127",
                  note="Re-annotates MMLU: ~6.5% of questions have ground-truth errors (57% in Virology) and corrected scores differ significantly from reported ones \u2014 direct evidence for LLM benchmarks."),
        Reference("Northcutt, Jiang, Chuang",
                  "Confident Learning: Estimating Uncertainty in Dataset Labels",
                  "JAIR 70", 2021, arxiv="1911.00068",
                  note="The confident-learning method behind the benchmark study; the 'confident disagreement' principle this probe operationalizes to flag suspect labels."),
    ],
    math=[
        MathBlock(
            label="Suspected-label-error rate (the score)",
            html=(
                '<span class="mrow">label_error_rate = '
                '<span class="frac"><span class="num">n<sub>suspect</sub></span><span class="den">n<sub>checked</sub></span></span>'
                ' ,&nbsp; suspect<sub>i</sub> = <b>1</b>[ agree<sub>i</sub> &and; &not;grade(consensus<sub>i</sub>, g<sub>i</sub>) ]</span>'
            ),
            latex=r"\mathrm{label\_error\_rate}=\frac{n_{\mathrm{suspect}}}{n_{\mathrm{checked}}},\quad "
                  r"\mathrm{suspect}_i=\mathbf{1}[\,\mathrm{agree}_i\wedge\neg\,\mathrm{grade}(\mathrm{consensus}_i,g_i)\,]",
        ),
        MathBlock(
            label="Cross-model agreement (>= 2 models concur) vs single-model fallback",
            html=(
                '<span class="mrow">cross: agree<sub>i</sub> = <b>1</b>[ &ge;80% of panel majorities = consensus<sub>i</sub> ] ;&nbsp; '
                'single (UNCONFIRMED): one model self-consistent (mode &ge;80% of its resamples) and contradicts the gold</span>'
            ),
            latex=r"\text{cross: }\mathrm{agree}_i=\mathbf{1}\!\left[\tfrac{|\{m:\mathrm{maj}_m=\mathrm{consensus}_i\}|}{|\mathrm{panel}|}\ge 0.8\right];\ "
                  r"\text{single: one model, mode}\ge 0.8,\ \neg\,\mathrm{grade}",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled items with a non-empty question and gold (sample_size, default 40)"),
        ("votes", "adjudicator samples drawn per item at temperature 0.7 (default 3)"),
        ("panel", "primary model + metadata['panel']; cross-model mode needs >= 2 independent models"),
        ("maj_m(i)", "model m's resample-majority answer on item i"),
        ("consensus_i", "the answer a MAJORITY of the panel's models agree on for item i"),
        ("agree_i (cross)", "1 if >= 80% of panel models share consensus_i (independent models concur)"),
        ("disagrees-gold_i", "1 if consensus_i (cross) / the single model's majority is graded NOT matching gold g_i"),
        ("g\u1d62", "the benchmark's gold answer for item i (the thing under audit)"),
        ("n_checked", "items with a usable question, gold and \u22651 extracted answer"),
        ("n_suspect", "items that are BOTH consistent and disagree (suspected bad labels)"),
        ("label_error_rate", "n_suspect / n_checked \u2014 the probe's score"),
    ],
    thresholds=[
        Threshold("CROSS-MODEL: rate >= 0.15", "high",
                  ">= 2 independent models AGREE on an answer that contradicts the gold, on a large share of items -- the key is likely substantially wrong; human-review before trusting any ranking. (Single-model never reaches HIGH.)"),
        Threshold("cross-model 0.05-0.15, OR single-model rate >= 0.15 (UNCONFIRMED)", "medium",
                  "Cross-model: non-trivial multi-model disagreement with the gold (~3-7% in real benchmarks). Single-model: candidates only -- a confidently-wrong model alone produces these, so confirm with a panel."),
        Threshold("label_error_rate < 0.05 (with items checked)", "low",
                  "Few confident disagreements with the gold on this sample; the answer key looks largely sound here."),
        Threshold("no checkable items", "info",
                  "No items had a question, a gold, and an extractable consensus, so the answer key cannot be audited."),
    ],
    effect=(
        "Protect the score from a wrong answer key by finding items where a "
        "panel of >= 2 independent models agrees on an answer that contradicts the gold "
        "\u2014 the signature of a mislabeled (or broken) benchmark item rather "
        "than a model mistake. Returns the specific suspect items so the labels "
        "can be human-reviewed and corrected."
    ),
    reading=(
        "label_error_rate \u2248 0 means the adjudicator's confident answers line "
        "up with the gold: the answer key looks trustworthy on this sample. A high "
        "rate means many golds are contradicted by a stable consensus \u2014 the "
        "benchmark, not the model, may be wrong, which caps measured accuracy and "
        "can flip rankings (Northcutt et al.). Treat the score as a triage rate, "
        "not a correction: open the returned examples (question, gold, model "
        "consensus) and have a human adjudicate. Read it with item_ambiguity "
        "(is the item merely ambiguous rather than mislabeled?) and "
        "contamination_perturb (could the adjudicator's confidence come from "
        "memorization rather than knowledge?)."
    ),
    caveats=(
        "(a) Necessary not sufficient \u2014 confidently wrong is possible: a "
        "memorized or popular item can yield a stable consensus that is itself "
        "wrong, inflating the rate; corroborate suspect items with "
        "contamination_perturb and always human-review. (b) Single vs panel: "
        "one model judges the gold, so its systematic blind spots become apparent "
        "'label errors' \u2014 a multi-model panel (cross-family, cf. judge_swap) "
        "is the reliable mode; this probe now supports it via metadata['panel'], and single-model results are reported UNCONFIRMED (capped below HIGH). (c) Mislabel vs "
        "ambiguity: a flag does not distinguish a wrong label from a question with "
        "no single defensible answer \u2014 that separation is item_ambiguity's "
        "job. (d) Grader-bound: 'disagrees' uses default_grader, so answer-format "
        "mismatches the grader can't normalize may masquerade as disagreement "
        "(interacts with answer_extraction_audit); and the 0.8 consistency cut "
        "with few votes is coarse (with votes=3, 'consistent' effectively means "
        "all extracted finals agree). (e) Small n makes the rate noisy (read with "
        "statistical_power); cost scales with votes (metered/cached by the "
        "CachingClient)."
    ),
    code_refs=["trust_the_eval.grading.extract_final",
               "trust_the_eval.grading.default_grader",
               "trust_the_eval.sampling.subsample"],
)

LabelErrorAudit.TUNABLES = {'agree': {'default': 0.8, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'fraction of panel that must agree (cross-model)'}, 'consistency': {'default': 0.8, 'min': 0, 'max': 1, 'step': 0.01, 'help': "a model's resample-majority share to count as self-consistent"}, 'cross_high': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'cross-model rate >= this -> HIGH'}, 'cross_medium': {'default': 0.05, 'min': 0, 'max': 1, 'step': 0.01, 'help': '>= this -> MEDIUM'}, 'single_medium': {'default': 0.15, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'single-model rate >= this -> MEDIUM (capped)'}, 'temp': {'default': 0.7, 'min': 0, 'max': 2, 'step': 0.05, 'help': 'adjudicator sampling temperature'}, 'sample_size': {'default': 40, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'votes': {'default': 3, 'min': 1, 'max': 20, 'step': 1, 'help': 'adjudicator samples per item', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
