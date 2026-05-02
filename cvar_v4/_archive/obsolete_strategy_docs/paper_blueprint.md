# CVaR-CJE Paper Blueprint

## Target Shape

Working title: **CVaR-CJE: Audit-Gated Tail Evaluation for LLM Judges**.

The paper should read as an LLM-evaluation methods paper with a new risk estimand, not as a generic CVaR paper. The core claim is:

> Mean CJE estimates how good a policy is on average; CVaR-CJE estimates how bad the policy is when it fails, and refuses level claims when the tail bridge does not transport.

Recommended length:

| Part | Target length | Role |
|---|---:|---|
| Main paper | 9-10 pages excluding references | One clean method, one new benchmark, one distinctive empirical story |
| References | 2-3 pages | Compact; cite only papers used in framing or design |
| Online appendix | 20-40 pages | Formal notation, proofs, estimator variants, Monte Carlos, dataset card, prompts, extra tables |

If this becomes a conference submission, the best fit depends on emphasis:

- **ICLR / NeurIPS / ICML** if the estimator, audit, inference, and Monte Carlo evidence are the center.
- **ACL / EMNLP / COLM** if Arena-Tail and LLM-as-judge benchmarking dominate.
- **FAccT** if the contribution is framed around audit-gated deployment, refusal to certify unsafe cells, and high-stakes evaluation governance.

## Main Paper Structure

1. **Introduction and Motivation** (1.5 pages)
   - The mean can be acceptable while the lower tail is unacceptable.
   - Existing LLM-as-judge work largely reports means, win rates, or calibration-corrected means.
   - CVaR-CJE extends CJE from average value to lower-tail severity.
   - The audit is front and center: rejected rows are not interpreted as valid level claims.

2. **Setup and Estimands** (1 page)
   - Prompt `X`, response `A`, cheap judge score `S`, oracle label `Y`, logger policy `pi0`, target policy `pi`.
   - Mean value `V(pi)`, quantile `q_alpha(pi)`, and lower-tail CVaR.
   - State the optimization-form CVaR as the primary definition.
   - Explain that quantiles and CVaR are complementary: quantile locates the cutoff, CVaR measures severity inside the cutoff.

3. **Audit-Gated CVaR-CJE** (2 pages)
   - Direct stop-loss estimator: fit `h_t(S, X)` for `(t - Y)_+`, average over target-policy judge scores, maximize over `t`.
   - Estimator variants: oracle-only, direct, direct+covariates, quantile companion.
   - Audit moments:
     - `g1 = 1{Y <= t_hat} - alpha`
     - `g2 = (t_hat - Y)_+ - h_t_hat(S, X)`
   - Decision rule: pass -> report CVaR level and CI; fail -> report diagnostic/refusal, not a level claim.
   - Show that `alpha = 1` recovers the mean-CJE audit up to sign.

4. **Arena-Tail Benchmark** (1.5 pages)
   - The benchmark is a contribution, not a footnote.
   - Design goals: continuous bounded oracle `Y`, heavy lower tail, similar means with different tails, audit-positive policy, retained strata metadata.
   - Proposed construction: Arena/WildChat general prompts + HealthBench-style medical prompts + BeaverTails/WildJailbreak/HarmBench/StrongREJECT safety prompts.
   - Policies: base, near-clone, premium, safety-tuned/over-refusal, adversarial/jailbroken.

5. **Empirical Results** (2 pages)
   - Figure 1: audit-gated leaderboard.
   - Figure 2: mean vs CVaR scatter showing policies with similar means but different tails.
   - Figure 3: tail-risk curves over `alpha`.
   - Table 1: per-policy mean, quantile, CVaR, audit p-value, gate status.
   - Table 2: estimator comparison against oracle-only and naive judges.
   - All headline claims are audit-gated.

6. **Monte Carlo Summary** (0.5 page in main, full appendix)
   - Main text only reports the summary table: size under transport, power under shortfall-transport failure, coverage after audit pass, sample-size scaling.

7. **Related Work and Limitations** (1 page)
   - CJE and calibrated LLM-as-judge inference.
   - LLM-as-judge benchmarks.
   - CVaR / expected shortfall.
   - Safety and medical benchmark datasets.
   - Limitations: oracle quality, discrete outcomes, alpha selection, prompt-source ethics, adaptive benchmark overfitting.

## What To Cite

Core CJE and LLM-as-judge inference:

- Landesberg and Narayan, *Causal Judge Evaluation*.
- Lee et al., *How to Correctly Report LLM-as-a-Judge Evaluations*.
- Chen et al., *Efficient Inference for Noisy LLM-as-a-Judge Evaluation*.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
- Liu et al., *G-Eval*.
- Li et al., *CalibraEval*.

Risk estimands:

- Rockafellar and Uryasev on CVaR optimization.
- Acerbi and Tasche on expected shortfall / coherence.

Benchmark and prompt sources:

- HealthBench for rubric-based high-stakes health evaluation and worst-at-k framing.
- Chatbot Arena / MT-Bench for the original LLM evaluation context.
- WildChat for real user prompts and toxicity/jailbreak tails.
- BeaverTails / PKU-SafeRLHF for helpfulness vs harmlessness and safety labels.
- WildJailbreak, HarmBench, StrongREJECT for adversarial and refusal behavior.
- Prometheus 2 / RewardBench as judge-resource context, not primary evidence.

Audit framing:

- Fleming and DeMets on surrogate endpoints.
- Lipsitch et al. on negative controls.
- Tchetgen Tchetgen et al. on proximal causal inference.

## Dataset Scope

The new dataset should be called **Arena-Tail** unless a better name emerges.

Minimum viable benchmark:

| Component | Target |
|---|---|
| Prompt pool | 50k-100k prompts |
| Policies | 5 controlled policies |
| Cheap judge coverage | 100% |
| Oracle coverage | 15-25%, tail-stratified |
| Audit slice | Held out from calibration; per policy |
| Outcome | Continuous score in `[0,1]`, preferably weighted rubric fraction |
| Headline alphas | `0.10` and `0.05`; `0.01` only if sample size supports it |

Avoid:

- A pure 5k Arena reuse as the only real-data result. It is useful as a diagnostic but too small for deep-tail claims.
- Pairwise win labels as the main `Y`. CVaR of pairwise wins is not the same object as lower-tail response quality.
- Likert-only `1-5` ratings as the main `Y`. They create atoms at the quantile and make the naive conditional-tail definition unstable.
- A benchmark where every bad prompt is obviously harmful. That shows safety classification, not tail evaluation under a realistic prompt mixture.
- Using the same model family as target policy, cheap judge, and oracle without a self-preference audit.

Explore:

- HealthBench-style weighted rubrics for continuous `Y`.
- Prompt strata retained as metadata: general, safety-sensitive, adversarial-harmful, adversarial-benign, medical, legal/support.
- Policies engineered to have similar means but different tails.
- At least one audit-positive policy that the paper refuses to certify.
- A small human/expert validation slice for oracle-judge quality, especially in medical or safety strata.

## Estimator Scope

Keep the main estimator set small:

1. **Naive judge CVaR**: compute tail on raw cheap judge scores. Baseline only.
2. **Oracle-only CVaR**: compute from the expensive oracle slice. High variance baseline.
3. **Direct CVaR-CJE**: stop-loss calibration and grid maximization.
4. **Direct+cov CVaR-CJE**: same, with response-length / stratum covariates if the mean-CJE pipeline supports it.
5. **Quantile-CJE companion**: report `q_alpha` beside CVaR, not as a replacement.

Avoid:

- Reporting a rejected CVaR estimate as if it were calibrated. The rejection is the result.
- Too many estimator variants in the main paper. DR, IPS, stacking, and alternative audits can move to the appendix unless they change the headline.
- Selecting `alpha` after seeing results. Pre-register `0.10` and `0.05`; put `0.01` in appendix unless powered.
- Calling the naive conditional mean `E[Y | Y <= q]` the definition when `Y` has atoms.

Explore:

- One-moment audit using only `g2` as a robustness check.
- Tail-stratified oracle allocation, with proper inverse sampling corrections if the oracle slice is not simple random.
- Cross-fitted stop-loss calibration so the audit and estimator do not share nuisance overfit.
- Calibration-aware cluster bootstrap with threshold re-optimization inside each replicate.

## Distinctive Results And Graphs

The paper needs one visual signature: **audit-gated tail ranking**.

Recommended main figures:

- **Audit-gated leaderboard**: rows are policies; columns are mean, `q_0.10`, `CVaR_0.10`, CI, audit status. Rejected rows are grey/hatched and labelled "refuse level claim."
- **Mean vs tail scatter**: x-axis mean value, y-axis CVaR. Show near-equal means but separated lower tails.
- **Tail-risk curves**: x-axis `alpha`, y-axis CVaR. Each policy is a curve; separation should grow as alpha shrinks.
- **Audit diagnostic plane**: x-axis `g1` mean, y-axis `g2` mean, confidence ellipse; accepted policies near origin, rejected policy away from origin.
- **Power/sample-size curve**: Monte Carlo only if space permits; otherwise appendix.

Avoid:

- A table of point estimates without gate status.
- Only showing error bars. The method's distinctive feature is the refusal decision.
- A result where all policies pass and all tails are ordered the same as means; that undersells the contribution.

## Motivation Framing

Use three domains, but keep them concrete and not sensational:

- **Clinical/medical assistants**: HealthBench explicitly studies worst-at-k reliability; missed emergency referral and unsafe medical compliance are tail events.
- **Content moderation and safety assistants**: refusal failures and over-refusals both live in tails; WildJailbreak's harmful/benign adversarial split is ideal.
- **Customer support / policy guidance**: a small number of hallucinated policy claims can dominate legal and reputational risk even when average satisfaction is high.

The line to use:

> Quantiles tell us where the bad region begins. CVaR tells us how bad the bad region is. The audit tells us whether the bridge that estimates that region is allowed to travel to this policy.

## Appendix Contract

The appendix should explicitly answer Eddie's concerns.

Include:

- Formal estimand and notation.
- Proof that the optimization-form CVaR equals the lower-tail conditional mean under continuity.
- Discrete-outcome definition using partial mass at the quantile.
- Proof/sketch that `alpha = 1` recovers mean-CJE.
- Two-moment Wald audit and one-moment alternative.
- Transport assumptions: mean transport vs threshold-indexed shortfall transport.
- Calibration-aware bootstrap details.
- Monte Carlo DGPs:
  - null where shortfall transport holds;
  - mean transport holds but shortfall transport fails;
  - wrong threshold / atom at quantile;
  - audit-positive adversarial policy;
  - sample-size and oracle-budget sweeps.
- Dataset card and labeling prompts for Arena-Tail.

