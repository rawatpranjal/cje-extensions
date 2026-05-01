# Classical Calibration, Validation Studies, and CJE

This memo compares the newly ingested classical calibration sources to Mean
CJE and CVaR-CJE. Source status is intentionally strict: "full local text"
means a local source was converted and read; "metadata/abstract only" means the
full paper or book was not locally available under the open-only rule.

## Source Status

| Source | Status | Best for | One-line takeaway | What CJE borrows | What CJE changes |
|---|---|---|---|---|---|
| Deville and Sarndal (1992), "Calibration Estimators in Survey Sampling" | Crossref metadata only | classical calibration weights | Calibrated weights use auxiliary totals and connect to GREG. | Use cheap auxiliary information to recover a better population estimand. | Calibrate a judge-to-oracle outcome map and audit policy transport, not only survey weights. |
| Sarndal, Swensson, and Wretman (1992), *Model Assisted Survey Sampling* | Springer metadata only | model-assisted design logic | A working model improves precision while design remains the protection. | Keep prediction and design/audit conceptually separate. | The "design" is the oracle budget and audit plan across policies. |
| Ta, Shao, Li, and Wang (2020), high-dimensional GREG | full local XML/text | modern GREG with many covariates | GREG can keep efficiency properties with high-dimensional structure. | Rich calibration covariates can be useful. | Rich covariates are admissible only if target-policy audits pass. |
| Carroll et al. (2006), *Measurement Error in Nonlinear Models* | Crossref metadata only | measurement-error theory | Validation data learn truth from error-prone proxies. | Treat cheap judge scores as proxies for expensive oracle labels. | Test reuse across target policies instead of assuming one stable proxy model. |
| Liao, Spiegelman, and Carroll (2013) | full local XML/text | regression-calibration failure modes | Regression calibration fails when the needed validation-study conditioning set is missing. | Discipline about which covariates belong in `f(s_i, x_i)`. | Use residual audits to detect missing-context bias. |
| Boe et al. (2023), "Issues in Implementing Regression Calibration Analyses" | arXiv abstract only | practical regression calibration | Calibration equation, standard errors, and mediator choices matter. | Propagate calibration uncertainty. | For CVaR, bootstrap must also re-optimize `t_hat`. |
| Amorim et al. (2021), two-phase validation design | full local XML/text | choosing validation samples | Targeted validation sampling can beat simple random sampling. | Tail-stratified oracle labeling is worth considering. | Include sampling correction if the oracle slice is not uniform. |
| Whittemore and Halpern (2013), external risk-model validation | full local XML/text | cost-efficient validation design | The best design depends on the target performance parameter. | Make oracle allocation estimand-driven. | Mean, CVaR, and audit moments may need different label allocation. |
| Yang and Ding (2025), two-phase rejective sampling | arXiv source converted from LaTeX | modern two-phase design | Phase-1 auxiliary variables can guide efficient second-phase sampling. | Use cheap scores and prompt strata to guide expensive labels. | Adapt the design to policy comparison and lower-tail risk. |
| Prentice (1989), Buyse and Molenberghs (1998), Alonso et al. (2004), Green et al. (2008) | PubMed metadata/abstracts only | surrogate endpoint validation | Correlation with the final outcome is not enough. | Surrogate skepticism and multi-criterion validation. | Replace broad surrogacy claims with policy-and-estimand-specific audit gates. |

## What Classical Calibration Is Doing

Survey calibration and GREG start from a sampling problem. The analyst observes
the outcome for sampled units and knows auxiliary information, often population
totals, for everyone or for the frame. Calibration changes the weights so the
sample matches those known auxiliary totals. GREG uses a model for efficiency,
but the model is "assisting" the design rather than replacing it.

The CJE analogue is partial, not literal. CJE also uses cheap auxiliary
information, the judge score `s_i` and covariates `x_i`, to reduce oracle-label
cost. But Direct CJE does not just adjust weights to known totals. It learns a
calibration map `f(s_i, x_i)` for oracle labels `y_i`, applies that map to
target-policy responses, and then asks whether the residual moment
`mean(y_i - f(s_i, x_i)) = 0` holds in the target context.

So the closest classical cousin is not pure survey calibration. It is a
validation-study or missing-outcome estimator: cheap proxy observed broadly,
gold-standard variable observed on a smaller slice, prediction plus residual
correction, and uncertainty that accounts for the calibration step.

## Mean CJE Versus the Classical Approach

Mean CJE estimates a policy mean `mean(y)` by fitting `f(s, x)` on an oracle
slice and averaging `f(s_i, x_i)` over target-policy responses. The critical
extra object is the policy-wise transport audit. Classical regression
calibration often assumes the validation relationship applies in the analysis
sample after conditioning on the right variables. CJE refuses to leave that as
an unchecked assumption: for each target policy, the audit estimates
`mean(y_i - f(s_i, x_i))`.

This makes CJE more operational than the classical surrogate-endpoint
literature. Prentice-style surrogacy is a strong distributional condition.
Mean CJE only needs the weaker mean residual condition for a level claim, but
it also requires an audit sample to test that condition. If the audit fails, the
right comparison is not "calibrated estimate versus uncalibrated estimate"; it
is "recalibrate, fall back to oracle-only, or refuse the level claim."

## CVaR-CJE Versus Mean CJE

CVaR-CJE changes the estimand. It is not enough for the judge to be unbiased on
average. For a threshold `t`, define the shortfall outcome
`z_i(t) = max(t - y_i, 0)`. CVaR-CJE fits a shortfall calibrator
`h_t(s_i, x_i)`, evaluates the lower-tail objective over a fixed threshold
grid, and chooses `t_hat`.

That creates two audit requirements:

1. The selected threshold must have the intended lower-tail mass:
   `mean(1{y_i <= t_hat}) - alpha = 0`.
2. The shortfall residual must transport:
   `mean(max(t_hat - y_i, 0) - h_t_hat(s_i, x_i)) = 0`.

This is stricter than Mean CJE. A policy can pass the mean audit because
positive and negative residuals cancel, while still fail the tail audit because
the judge is miscalibrated exactly on the worst responses. That is the main
lesson from the surrogate-endpoint literature for CVaR-CJE: validating a
surrogate for one estimand does not validate it for every estimand.

## What We Should Learn

1. Treat oracle-slice design as part of the method. A uniform oracle slice is a
   clean baseline, but the validation-design papers argue that targeted
   sampling can be much more efficient. For CVaR-CJE, useful strata are cheap
   judge score quantiles, prompt family, policy family, response length, and
   regions near the estimated tail threshold.

2. If the oracle slice is tail-stratified, include the sampling correction.
   Tail-stratified labels are attractive only if the estimator and audit know
   the inclusion probabilities. Otherwise the calibration model can learn from
   an intentionally distorted oracle slice and produce biased level claims.

3. Separate the mean gate from the tail gate. The current workflow is right to
   validate Mean CJE first, but that only licenses the mean. CVaR-CJE still
   needs its own threshold and shortfall residual audit.

4. Report calibration uncertainty as first-order, not cosmetic. Regression
   calibration papers emphasize valid standard errors after estimating the
   calibration equation. CJE's calibration-aware bootstrap is the correct
   analogue, especially because CVaR also re-optimizes `t_hat`.

5. Be careful with covariates. Classical regression calibration warns that the
   calibration equation must include variables needed for validity, but
   conditioning choices can also change interpretation. In CJE, response length
   and prompt stratum can reduce bias, but the target-policy audit must remain
   the final arbiter.

6. Use surrogate-endpoint language sparingly. We do not need to claim full
   Prentice-style surrogacy. The defensible claim is narrower: for a named
   policy and estimand, the calibrated surrogate is admissible only if the
   relevant residual audit passes.

## Bottom Line

Classical calibration says: use cheap auxiliary information, but keep the
design and variance accounting honest. Validation-study work says: learn the
truth-from-proxy map on gold-standard data, but condition on the right variables
and propagate first-stage uncertainty. Surrogate-endpoint work says: a proxy can
look good and still fail for the estimand that matters.

Mean CJE combines these lessons into an audit-gated mean estimator. CVaR-CJE
pushes the same logic into the lower tail, where the audit must be tail-specific
because mean validity does not imply tail validity.
