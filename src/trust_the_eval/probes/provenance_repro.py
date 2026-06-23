from __future__ import annotations
import hashlib
import json
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..grading import normalize
from ..probe import ModelClient, Probe, register
from ..sampling import subsample


@register
class ProvenanceRepro(Probe):
    """Produce a signed evidence bundle (content hash + config) and, with a model,
    measure determinism by re-running a small sample twice at temperature 0."""
    id = "provenance_repro"
    name = "Provenance & reproducibility"
    paper_priority = "Principle 3"
    requires_model = True

    def __init__(self, sample_size: int = 20, seed: int = 0,
                 signing_key: str = "trust-the-eval"):
        self.sample_size, self.seed, self.signing_key = sample_size, seed, signing_key

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        assert model is not None
        items = [(i, it) for i, it in subsample(artifact.items, self.sample_size, self.seed)
                 if it.question.strip()]
        n = stable = 0
        for _, it in items:
            n += 1
            a = normalize(model.complete(it.question, temperature=0.0))
            b = normalize(model.complete(it.question, temperature=0.0))
            if a == b:
                stable += 1
        determinism = (stable / n) if n else 1.0
        bundle = {"dataset": artifact.dataset, "content_hash": artifact.content_hash(),
                  "model": artifact.model, "n_items": artifact.n,
                  "determinism_sample": n, "determinism": round(determinism, 3)}
        sig = hashlib.sha256((self.signing_key + json.dumps(bundle, sort_keys=True)).encode()).hexdigest()
        sev = (Severity.MEDIUM if determinism < self.tune("determinism_min") else Severity.LOW)
        return [Finding(self.id, sev,
                        f"determinism {determinism:.2f} over {n} re-runs; "
                        f"signed provenance bundle issued",
                        score=round(determinism, 3),
                        otel_attributes={"gen_ai.eval.trust.repro.determinism": round(determinism, 3),
                                         "gen_ai.eval.trust.provenance.signature": "sha256:" + sig[:16]},
                        evidence={"bundle": bundle, "signature": "sha256:" + sig})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

ProvenanceRepro.DOC = ProbeDoc(
    science=(
        "A result you cannot reproduce is not yet a measurement. Reproducibility "
        "\u2014 obtaining the same result from the same data and code \u2014 is the "
        "step that turns a reported number into a verifiable finding, and the ML "
        "community has formalized it: Pineau et al. (2021), reporting the NeurIPS "
        "2019 reproducibility program, introduced a code-submission policy and a "
        "reproducibility checklist (seeds, full configuration, data provenance) "
        "precisely because results were often unverifiable. Gundersen & Kjensmo "
        "(2018) surveyed AI papers and found most lack the documentation needed to "
        "reproduce their reported results at all. For LLM evaluation two distinct "
        "things must hold: the artifact must be PROVENANCED (you know exactly which "
        "data and model produced the score, sealed so tampering is detectable), and "
        "the scoring must be REPRODUCIBLE (re-running it yields the same answers).\n\n"
        "Reproducibility is the harder half for LLMs because they are not "
        "deterministic by default: Chen et al. (2023), studying behaviour drift, "
        "note that even nominally fixed model services return different answers to "
        "identical inputs, and most APIs are not bit-reproducible even at "
        "temperature 0 (batching, hardware, and decoding implementation all leak "
        "in). If the same question, asked twice the same way, yields different "
        "graded outcomes, then a single-run score is not reconstructable and "
        "comparisons across time or labs are unsafe.\n\n"
        "This probe does both jobs. For provenance it emits a signed bundle: the "
        "dataset's content hash (a SHA-256 over the canonical items \u2014 any edit "
        "changes it), the model id, item count, and the determinism result, all "
        "sealed with a SHA-256 signature over the bundle so later tampering is "
        "evident. For reproducibility it measures determinism directly: it asks a "
        "deterministic sample of items twice at temperature 0 and reports the "
        "fraction whose (normalized) outputs match. Determinism = 1 means the run "
        "is exactly replayable on this sample; lower means the score itself carries "
        "run-to-run irreproducibility.\n\n"
        "Scope and honesty: this assesses the validity of the eval RESULT \u2014 its "
        "auditability and replayability \u2014 not the safety of the model. The "
        "signature is an integrity/provenance seal (it proves the bundle hasn't "
        "changed and binds data+model+result), NOT a cryptographic authenticity "
        "claim about who produced it: the default key is public, so it detects "
        "accidental tampering, not a determined forger. Determinism is measured on "
        "raw outputs (it is the strict, exact-match notion); a low score flags "
        "irreproducibility but does not by itself separate harmless formatting "
        "jitter from outcome-changing nondeterminism \u2014 read it with "
        "self_consistency (answer-level stability) and model_drift (across-time "
        "change)."
    ),
    references=[
        Reference("Pineau, Vincent-Lamarre, Sinha, Larivière, Beygelzimer, d'Alché-Buc, Fox, Larochelle",
                  "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)",
                  "JMLR 22(164)", 2021, arxiv="2003.12206",
                  note="Defines reproducibility as obtaining the same result from the same data/code and introduces the seeds/config/provenance checklist this probe's bundle operationalizes."),
        Reference("Gundersen, Kjensmo",
                  "State of the Art: Reproducibility in Artificial Intelligence",
                  "AAAI", 2018,
                  url="https://doi.org/10.1609/aaai.v32i1.11503",
                  note="Surveys AI research and finds most papers lack the documentation needed to reproduce results \u2014 the motivation for a machine-checkable provenance bundle."),
        Reference("Chen, Zaharia, Zou",
                  "How Is ChatGPT's Behavior Changing over Time?",
                  "arXiv (Harvard Data Science Review)", 2023, arxiv="2307.09009",
                  note="Documents that nominally fixed LLM services give non-deterministic answers to identical inputs \u2014 why determinism must be measured, not assumed, for a reproducible score."),
    ],
    math=[
        MathBlock(
            label="Determinism (exact replay rate at temperature 0)",
            html=(
                '<span class="mrow">determinism = '
                '<span class="frac"><span class="num">1</span><span class="den">n</span></span>'
                '&Sigma;<sub>i&isin;S</sub> <b>1</b>[ norm(y<sub>i</sub><sup>(1)</sup>) = norm(y<sub>i</sub><sup>(2)</sup>) ]</span>'
            ),
            latex=r"\mathrm{determinism}=\frac1n\sum_{i\in S}\mathbf{1}\!\left[\mathrm{norm}(y_i^{(1)})=\mathrm{norm}(y_i^{(2)})\right]",
        ),
        MathBlock(
            label="Content hash & signed provenance bundle",
            html=(
                '<span class="mrow">content_hash = SHA256(canonical(items)) ,&nbsp;&nbsp;'
                'signature = SHA256( key \u2225 JSON<sub>sorted</sub>(bundle) )</span>'
            ),
            latex=r"\mathrm{content\_hash}=\mathrm{SHA256}(\mathrm{canonical}(\text{items})),\quad "
                  r"\mathrm{signature}=\mathrm{SHA256}\big(\text{key}\,\Vert\,\mathrm{JSON}_{\text{sorted}}(\text{bundle})\big)",
        ),
    ],
    terms=[
        ("S", "the deterministically sampled items with a non-empty question (sample_size, default 20)"),
        ("n", "number of items re-run twice (the determinism denominator)"),
        ("y\u1d62\u207d\u00b9\u207e, y\u1d62\u207d\u00b2\u207e", "the model's two temperature-0 completions for item i"),
        ("norm(\u00b7)", "normalization (strip/lowercase/whitespace) applied before comparing the two outputs"),
        ("determinism", "fraction of items whose two runs match after normalization (the probe's score)"),
        ("content_hash", "SHA-256 over the artifact's canonical items \u2014 any change to the data changes it (EvalArtifact.content_hash)"),
        ("bundle", "{dataset, content_hash, model, n_items, determinism_sample, determinism} \u2014 the provenance record"),
        ("signature", "SHA-256 of (signing_key \u2225 sorted-JSON bundle) \u2014 an integrity seal over the bundle"),
    ],
    thresholds=[
        Threshold("determinism < 0.90", "medium",
                  "More than ~1 in 10 items return different outputs on an identical re-run \u2014 the score is not exactly replayable; report it with a run-to-run interval and pin decoding settings."),
        Threshold("determinism \u2265 0.90", "low",
                  "Runs are (near-)exactly replayable on this sample; a signed provenance bundle is issued binding data + model + result."),
    ],
    effect=(
        "Make the eval result auditable and replayable: seal exactly which data "
        "and model produced it (so tampering is detectable) and measure whether "
        "re-running the scoring reproduces the same outputs. Turns 'trust this "
        "number' into a verifiable bundle plus a concrete determinism rate."
    ),
    reading=(
        "determinism \u2248 1 means the run replays exactly on this sample, so the "
        "score is reproducible and the issued bundle pins its provenance. A value "
        "below 1 means identical inputs gave different outputs \u2014 the score "
        "carries irreproducibility, single-run numbers should come with a "
        "run-to-run interval, and decoding settings (temperature, seed, "
        "implementation) need pinning before the result is compared across time or "
        "labs. The content_hash lets anyone confirm the dataset is byte-identical "
        "to the one scored; the signature lets them confirm the bundle wasn't "
        "altered. Read determinism with self_consistency (does the ANSWER stay "
        "stable, not just the text) and model_drift (has the model changed since)."
    ),
    caveats=(
        "(a) Integrity, not authenticity: the signature seals the bundle's "
        "contents with a default PUBLIC key, so it detects accidental tampering "
        "and binds data+model+result \u2014 it is not proof of authorship and a "
        "determined forger with the key could re-sign; use a private signing_key "
        "for stronger guarantees. (b) Strict, exact-match determinism: outputs are "
        "compared after light normalization, so harmless formatting jitter counts "
        "as non-determinism \u2014 the score is a conservative (low) bound on "
        "replayability; semantic stability is self_consistency's job. (c) Two "
        "samples per item: determinism is estimated from a single pair at small n, "
        "so it is noisy (read with statistical_power) and cannot quantify the "
        "DISTRIBUTION of outputs, only whether two draws matched. (d) "
        "Provenance \u2260 quality: a perfectly reproducible, well-sealed score can "
        "still be invalid for other reasons (contamination, judge bias, label "
        "error) \u2014 this probe certifies you can re-run and verify the number, "
        "not that the number is right."
    ),
    code_refs=["trust_the_eval.artifact.EvalArtifact.content_hash",
               "trust_the_eval.sampling.subsample"],
)

ProvenanceRepro.TUNABLES = {'determinism_min': {'default': 0.9, 'min': 0, 'max': 1, 'step': 0.01, 'help': 'determinism < this -> MEDIUM'}, 'sample_size': {'default': 20, 'min': 1, 'max': 1000, 'step': 1, 'help': 'items sampled', 'ctor': True}, 'seed': {'default': 0, 'min': 0, 'max': 99999, 'step': 1, 'help': 'seed', 'ctor': True}}
