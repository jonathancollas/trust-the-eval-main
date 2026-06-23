from __future__ import annotations
from typing import Optional

from ..artifact import EvalArtifact
from ..finding import Finding, Severity
from ..probe import ModelClient, Probe, register
from ..stats import expected_max_gap


@register
class MultiplicityCherrypick(Probe):
    """Researcher degrees of freedom: if many configs were tried and only the
    best reported, the headline is inflated. Reads metadata['configs'] (list of
    scores) or metadata['n_configs']."""
    id = "multiplicity_cherrypick"
    name = "Multiplicity / cherry-picking"
    paper_priority = "Principle 2"
    requires_model = False

    def run(self, artifact: EvalArtifact, model: Optional[ModelClient] = None):
        md = artifact.metadata
        configs = md.get("configs")
        n_cfg = md.get("n_configs") or (len(configs) if isinstance(configs, list) else None)
        if not n_cfg or n_cfg < self.tune("min_cfg"):
            return [Finding(self.id, Severity.INFO,
                            "no multi-config info; provide metadata['configs'] (per-config scores) "
                            "or ['n_configs'] to assess cherry-picking",
                            evidence={"n_configs": n_cfg})]
        attrs = {"gen_ai.eval.trust.multiplicity.n_configs": n_cfg}
        if isinstance(configs, list) and all(isinstance(c, (int, float)) for c in configs):
            best, mean = max(configs), sum(configs) / len(configs)
            inflation = best - mean
            noise = expected_max_gap(n_cfg, mean, artifact.n)
            bonf = self.tune("alpha") / n_cfg
            within = inflation <= noise
            attrs["gen_ai.eval.trust.multiplicity.inflation_pts"] = round(inflation * 100, 1)
            attrs["gen_ai.eval.trust.multiplicity.noise_pts"] = round(noise * 100, 1)
            if n_cfg >= self.tune("big_cfg"):
                sev = Severity.HIGH if within else Severity.MEDIUM
            elif inflation > noise:
                sev = Severity.MEDIUM
            else:
                sev = Severity.LOW
            verdict = ("WITHIN multiplicity noise - the lead may be luck" if within
                       else "exceeds noise - lead is likely real")
            msg = (f"{n_cfg} configs; reported-best is +{inflation*100:.1f} pts above the mean vs "
                   f"+{noise*100:.1f} pts expected from noise alone ({verdict}); "
                   f"Bonferroni alpha = {bonf:.4f}")
            return [Finding(self.id, sev, msg, score=round(inflation, 3), otel_attributes=attrs,
                            evidence={"n_configs": n_cfg, "best": round(best, 4),
                                      "mean": round(mean, 4), "inflation": round(inflation, 4),
                                      "noise_inflation": round(noise, 4),
                                      "bonferroni_alpha": round(bonf, 4)})]
        sev = Severity.MEDIUM if n_cfg >= self.tune("big_cfg") else Severity.LOW
        return [Finding(self.id, sev,
                        f"{n_cfg} configurations tried (no per-config scores; "
                        f"Bonferroni alpha = {0.05/n_cfg:.4f} would apply)",
                        score=None, otel_attributes=attrs,
                        evidence={"n_configs": n_cfg, "bonferroni_alpha": round(0.05 / n_cfg, 4)})]


# ---------------------------------------------------------------------------
# Scientific documentation (surfaced in the UI; see trust_the_eval.probedoc)
# ---------------------------------------------------------------------------
from ..probedoc import ProbeDoc, Reference, MathBlock, Threshold  # noqa: E402

MultiplicityCherrypick.DOC = ProbeDoc(
    science=(
        "If you try many configurations \u2014 prompts, temperatures, few-shot "
        "seeds, decoding settings \u2014 and report only the best, the headline "
        "number is inflated: you have selected the top of a noisy distribution, "
        "not estimated a fixed quantity. This is the multiple-comparisons problem "
        "in its evaluation guise, and it is driven by 'researcher degrees of "
        "freedom': Simmons, Nelson & Simonsohn (2011) showed that undisclosed "
        "flexibility in how analyses are run makes it possible to present almost "
        "anything as significant. Gelman & Loken (2014) sharpened this with the "
        "'garden of forking paths' \u2014 the inflation does not require deliberate "
        "fishing; simply choosing the best-looking option after seeing the "
        "results is enough to bias the reported score upward. For LLM evaluation, "
        "where re-prompting is cheap and many configs are routinely swept, the "
        "best-of-N number can sit well above the typical-config number purely by "
        "selection.\n\n"
        "This probe surfaces that selection effect from the run's own metadata. "
        "It reads how many configurations were tried (metadata['n_configs'], or "
        "the length of metadata['configs']) and, when the per-config scores are "
        "available, computes the optimism gap between the reported best and the "
        "mean configuration. More configs means more opportunities for the "
        "maximum to drift above the true value; a large best-minus-mean gap is the "
        "direct evidence of how much of the headline is selection rather than "
        "capability. The standard remedy for testing under multiplicity is to "
        "control the false discovery rate (Benjamini & Hochberg, 1995) or to "
        "report the full distribution rather than the maximum.\n\n"
        "Scope and honesty: this assesses the validity of a reported score under "
        "configuration search \u2014 whether it is an honest estimate or a "
        "selected maximum \u2014 not the safety of the model. It is metadata-"
        "driven and therefore an honesty aid, not a detector: it can only see the "
        "configs the run DISCLOSES, so it rewards transparency and is blind to "
        "undisclosed sweeps (the very degrees of freedom Simmons et al. warn "
        "about). The inflation gap is descriptive, not a significance-corrected "
        "quantity; it quantifies optimism, and the fix is to pre-register the "
        "config, report all configs, or apply a multiplicity correction \u2014 not "
        "to read the maximum as the result."
    ),
    references=[
        Reference("Simmons, Nelson, Simonsohn",
                  "False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant",
                  "Psychological Science 22(11), 1359\u20131366", 2011,
                  url="https://doi.org/10.1177/0956797611417632",
                  note="Coins 'researcher degrees of freedom' and shows undisclosed flexibility lets one present almost anything as significant \u2014 the mechanism behind a cherry-picked best config."),
        Reference("Gelman, Loken",
                  "The Garden of Forking Paths: Why Multiple Comparisons Can Be a Problem, Even When There Was No 'Fishing Expedition'",
                  "American Scientist 102(6), 460", 2014,
                  url="https://doi.org/10.1511/2014.111.460",
                  note="Shows selection inflation arises even without deliberate fishing \u2014 choosing the best option after seeing results biases the reported number upward."),
        Reference("Benjamini, Hochberg",
                  "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
                  "Journal of the Royal Statistical Society B 57(1), 289\u2013300", 1995,
                  url="https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
                  note="The standard correction for multiplicity \u2014 the principled alternative to reporting the single best of many configurations."),
    ],
    math=[
        MathBlock(
            label="Optimism gap (reported-best minus mean config)",
            html=(
                '<span class="mrow">inflation = max<sub>j</sub> s<sub>j</sub> &minus; '
                '<span class="frac"><span class="num">1</span><span class="den">m</span></span>'
                '&Sigma;<sub>j=1</sub><sup>m</sup> s<sub>j</sub></span>'
            ),
            latex=r"\mathrm{inflation}=\max_{j} s_j-\frac1m\sum_{j=1}^{m} s_j",
        ),
        MathBlock(
            label="Multiplicity exposure",
            html=(
                '<span class="mrow">m = n_configs = number of configurations tried '
                '(more m \u21d2 more upward selection of max<sub>j</sub> s<sub>j</sub>)</span>'
            ),
            latex=r"m=n\_configs\quad(\text{larger } m \Rightarrow \text{greater optimism of } \max_j s_j)",
        ),
        MathBlock(
            label="Noise baseline (expected best-minus-mean) & Bonferroni level",
            html=(
                '<span class="mrow">noise = E[max of m] &middot; se ,&nbsp; se = &radic;(p(1&minus;p)/n) ,&nbsp; '
                '&alpha;<sub>Bonf</sub> = 0.05 / m</span>'
            ),
            latex=r"\mathrm{noise}=\mathbb{E}[\max\nolimits_m Z]\cdot\sqrt{p(1-p)/n},\quad \alpha_{\mathrm{Bonf}}=0.05/m",
        ),
    ],
    terms=[
        ("m (n_configs)", "number of configurations tried, from metadata['n_configs'] or len(metadata['configs'])"),
        ("s\u2c7c", "the score of the j-th configuration (from metadata['configs'] when it is a numeric list)"),
        ("max\u2c7c s\u2c7c", "the reported best score \u2014 the headline being scrutinized"),
        ("mean config", "(1/m) \u03a3 s\u2c7c \u2014 the average configuration's score, a less optimistic summary"),
        ("inflation", "best minus mean -- how much the headline exceeds a typical config"),
        ("noise_inflation", "expected best-minus-mean under pure noise = E[max of m] * se (Blom approx)"),
        ("bonferroni_alpha", "0.05 / m -- the multiplicity-corrected significance level for m comparisons"),
    ],
    thresholds=[
        Threshold(">= 5 configs and the lead is within noise", "high",
                  "Many configs and the reported best is within what multiplicity noise alone would produce -- the lead is likely luck; report the full distribution or a corrected result."),
        Threshold(">= 5 configs (lead exceeds noise), or < 5 configs with lead exceeding noise", "medium",
                  "Selection is present; pre-register the config or report all configs / a multiplicity-corrected (Bonferroni/BH) result."),
        Threshold("few configs with the lead within noise", "low",
                  "A few configurations with a lead no larger than noise; note the selection but the headline is not strongly inflated."),
        Threshold("n_configs absent or < 2", "info",
                  "No multi-configuration metadata, so cherry-picking cannot be assessed (undisclosed sweeps are invisible here)."),
    ],
    effect=(
        "Tell whether a reported score is an honest estimate or the lucky maximum "
        "of a configuration sweep. From the run's own metadata the probe reports "
        "how many configs were tried and how far the reported best sits above the "
        "mean config \u2014 turning hidden 'researcher degrees of freedom' into an "
        "explicit optimism figure that argues for pre-registration or "
        "full-distribution reporting."
    ),
    reading=(
        "A small inflation with few configs means the reported number is close to "
        "a typical run \u2014 little selection. A large best-minus-mean gap, or many "
        "configs, means the headline is substantially a selected maximum: it would "
        "likely regress toward the mean on fresh data, so do not compare it "
        "against a single-config competitor, and prefer reporting all configs (or "
        "a multiplicity-corrected / pre-registered result). The score is the "
        "inflation gap in score units; n_configs is the exposure that makes a "
        "large gap likely. Because the probe only sees disclosed configs, a clean "
        "result here is reassuring only to the extent the run was transparent "
        "about what it swept."
    ),
    caveats=(
        "(a) Metadata-driven, not a detector: it can only assess the configs a run "
        "DISCLOSES via metadata, so undisclosed sweeps \u2014 the core of "
        "'researcher degrees of freedom' (Simmons et al.) \u2014 are invisible, and "
        "absence of multi-config metadata returns INFO, not a clean bill. (b) "
        "Descriptive plus corrected: the probe reports the best-minus-mean gap, compares it to the noise-only spread (Blom) and the Bonferroni level "
        "; for full inference under "
        "multiplicity use a false-discovery-rate or family-wise correction "
        "(Benjamini & Hochberg) rather than this raw gap. (c) Mean is a rough "
        "baseline: best-minus-MEAN can understate selection when many configs are "
        "poor (dragging the mean down) or overstate it when configs are near-"
        "duplicates; the full distribution (or best-minus-median / variance) is "
        "more informative, and the gap ignores each config's own sampling error "
        "(read with statistical_power). (d) Forking paths beyond configs: choices "
        "like prompt wording, data filtering and metric definition are also "
        "degrees of freedom (Gelman & Loken) that this config-count probe does not "
        "capture \u2014 it addresses one disclosed slice of a broader problem. (e) "
        "Related but distinct from subgroup_power's multiplicity note: there the "
        "issue is many SUBGROUPS, here it is many CONFIGURATIONS of the same eval."
    ),
    code_refs=["trust_the_eval.stats.expected_max_gap"],
)

MultiplicityCherrypick.TUNABLES = {'min_cfg': {'default': 2, 'min': 2, 'max': 20, 'step': 1, 'help': 'min configs to assess'}, 'big_cfg': {'default': 5, 'min': 2, 'max': 50, 'step': 1, 'help': 'configs >= this -> stricter verdict'}, 'alpha': {'default': 0.05, 'min': 0, 'max': 0.2, 'step': 0.005, 'help': 'Bonferroni family alpha'}}
