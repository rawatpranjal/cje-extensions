# cvar_v5 deferred work

Anchors in this file are referenced from `NotImplementedError` strings in code. When a flag in `Config` is set to True for a deferred feature, the corresponding `[anchor]` here describes what's missing.

## `[grid-tail-dense]` — closed 2026-05-02

Hypothesis: at small α the optimum t̂_α = q_α(Y) sits in [0, 0.1], where the population saddle objective Ψ_α(t) has curvature −p_Y(t)/α — so any grid spacing wider than ~√(α / p_Y) introduces discretization bias on the argmax. Default uniform 61-point grid has spacing 1/60 ≈ 0.017, which over uniform-Y at α=0.01 produces a population (no-MC) ĈVaR underestimate of ~0.002.

Verified by population calculation (`Ψ(t) = t − t²/(2α)` for Y~U(0,1)): grid argmax at α=0.01 is 0.00277 vs truth 0.005. **Discretization bias = +0.002, before any sampling noise.**

**Fix shipped**: added `make_t_grid(...)` in `cvar_cje/calibrator.py` with `grid_kind ∈ {"uniform", "tail_dense"}`. `tail_dense` (default) places 21 of 61 points uniformly in [0, 0.1], 41 in [0.1, 1]. Same |T| = 61, same calibrator-fit cost.

**Empirical result** (R=25, n_oracle=600, n_eval=1000):

| α | policy | uniform RMSE | tail_dense RMSE | improvement |
|---|---|---|---|---|
| 0.01 | uniform | 0.00373 | 0.00261 | **−30%** |
| 0.05 | tail_heavy | 0.00187 | 0.00064 | **−66%** |
| 0.10 | (any) | ~ | ~ | unchanged (within MC noise) |
| 0.50 | (any) | ~ | ~ | unchanged |

`tail_dense` is **strictly ≥ uniform** on every cell tested. Default switched.

`Config.grid_kind = "tail_dense"`. Provenance for the comparison: ad-hoc benchmark, not a sweep harness — see chat transcript / commit message.

## `[grid-adaptive]` — closed-investigated 2026-05-02 (do not revisit without signal)

Hypothesis: a grid that adapts to (Y_calib, α) — concentrating points around `quantile(Y_calib, α)` — would beat the static `tail_dense` grid by following the policy's actual quantile location.

**Investigated, equivalent to tail_dense**, with one strict downside.

R=25 on the 4-policy × 5-α panel: adaptive matches tail_dense to within MC noise on every cell. Both fix the small-α discretization bias. tail_dense's hard-coded `[0, 0.1]` window happens to cover the small-α optimum for all 4 policies on this DGP.

**Strict downside of adaptive**: the grid depends on α at fit time, so one fitted calibrator can't be queried at multiple α values (each α wants a different grid). tail_dense doesn't have this lock-in.

**Decision**: tail_dense is the default. Adaptive is dropped — no implementation lives in cvar_v5/. If a real-data setting shows the optimum t̂_α sits outside [0, 0.1] (e.g., a policy with mass concentrated mid-range and a moderate α), revisit. Until then, don't.

## `[ci-bootstrap]`

Full-pipeline cluster bootstrap CI on the Direct CVaR point estimate. Per `cvar_v4/sections/method.tex:98-113`: resamples train/eval/audit jointly, refits shortfall calibrators, re-optimizes `t̂_α` per replicate, returns 95% percentile CI on the plug-in centre.

`Config.bootstrap_ci`. Out of MVP because the user said "no need to fix coverage at this moment or CIs / focus on the estimate itself purely."

## `[var-cal-jackknife]`

Leave-one-out jackknife on calibrator-fit variance, contributing to the variance decomposition `Var_total ≈ Var_cal + Var_audit`. See `cvar_v4/eda/deeper/_estimator.py:jackknife_var_cal` for v4 implementation.

`Config.jackknife_var_cal`.

## `[plugin-quantile]`, `[plugin-ru-dual]` — closed-falsified 2026-05-02

Hypothesis: classical plug-in CVaR variants on the calibrated mean reward `m̂(s) = 1 − ĥ_1(s)` could match or beat the Direct saddle-point's accuracy at lower computational cost. Two variants implemented and compared head-to-head with Direct on the parametric Beta panel:

- `plugin_quantile`: empirical α-quantile + tail mean of `m̂(s_eval)`
- `plugin_ru_dual`: Rockafellar–Uryasev dual `Ψ̃_α(t) = t − (1/(αn)) Σ (t − m̂(s_i))_+`, argmax over `T`

**Falsified.** The Direct saddle-point dominates across all 12 (policy × α) cells.

| metric | Direct | plug-in quantile | plug-in RU-dual |
|---|---|---|---|
| RMSE ratio (vs Direct, mean over cells) | 1.00 | **14.6×** | **14.4×** |
| bias (worst case across cells) | -0.002 | +0.116 | +0.115 |
| var_calib ratio (mean over cells) | 1.00 | 30.4× | 28.6× |

**Theoretical explanation** (Jensen on `(t − y)_+`):

    E[(t − Y)_+ | s]  ≥  (t − E[Y | s])_+  =  (t − m̂(s))_+

Plug-in RU systematically *underestimates* conditional shortfall, which biases ĈVaR upward. Plug-in quantile loses the within-stratum variability of `Y | s` entirely — it only sees `m̂(s)`. The empirical bias of +0.05 to +0.12 across the panel matches the sign of the Jensen gap.

**Default stays Direct saddle-point** (`cvar_cje/estimator.py::estimate_direct_cvar`). Both plug-in implementations preserved at `cvar_v5/_archive/plugin_estimators.py` for reference.

**Provenance**: `cvar_v5/mc/runs/2026-05-02T155532_estimator_comparison/` (R=20, n_oracle=600, n_eval=1000, K=5, α ∈ {0.05, 0.10, 0.20}, all 4 policies).

`Config.plugin_variants` stays a deferred-feature flag that raises `NotImplementedError`. Plug-ins are research-grade, not paper-canonical; do not re-investigate without a strong reason (e.g., a real-data setting where m̂(s) approximates Y closely and within-stratum variance is negligible).

## `[two-stage-calibrator]`

Two-stage calibrator: isotonic on `s`, then linear residual adjustment on `response_length`. NOT canonical in `cvar_v4/sections/method.tex` (CVaR section specifies single-stage isotonic). Tracked here for the eventual real-data adapter where two-stage is needed for paper-quality numbers on HealthBench / Arena.

`Config.two_stage_calibrator`.

## `[ttc-gate]`

Target-Typicality Coverage gate (≥ 0.70) — Mean-CJE convention from the original CJE paper, used to gate IPS estimators. Not in the cvar_v4 method.tex CVaR spec. Not necessary for the Direct CVaR-CJE MVP. Add if/when we move to mixed Direct/IPS reporting.

(No `Config` flag — silently absent in MVP.)

## `[omega-research]` — open (omega_sweep run 2026-05-02T093648)

Initial omega_sweep at n_oracle=600 (n_audit ≈ 104), R=100, δ ∈ {0, 0.1, 0.3, 0.5, 1.0}, all 4 policies × α=0.10:

| Ω̂ | size at δ=0 | size_dev | mean power (δ>0) |
|---|---|---|---|
| `boot_remax_ridge`    | 0.000 | 0.050 | 0.000 |
| `boot_remax_no_ridge` | 0.170 | 0.120 | 0.211 |
| `analytical`          | 0.185 | 0.135 | 0.342 |
| `boot_fixed`          | 0.190 | 0.140 | 0.344 |

**No variant cleared the size_dev < 0.10 floor.** Type-I error is either 0 (ridge dominance — useless audit) or 3–4× nominal η=0.05 (argmax-variance unaccounted — over-rejects at null).

**Current default**: `Config.omega_estimator = "boot_remax_ridge"` stays. Conservative — never falsely rejects, but also never detects real transport breaks at this n_audit. **Reported numbers must NOT claim audit-validated until this is resolved.** The other three variants are kept in `audit.py` because the test suite (`test_omega_variants_give_different_W`) and the MC report depend on the family.

**Provenance**: `cvar_v5/mc/runs/2026-05-02T093648_omega_sweep/`

### Sub-anchor: `[omega-research-n-audit]` — investigated 2026-05-02; **closed-falsified**

Hypothesis: variants calibrate at larger n_audit. **Falsified, with a stronger negative finding: size gets WORSE as n_audit grows.**

| n_audit | analytical | boot_fixed | boot_remax_no_ridge | boot_remax_ridge |
|---|---|---|---|---|
| 100  | 0.185 | 0.190 | 0.170 | 0.000 |
| 249  | 0.165 | 0.165 | 0.120 | 0.000 |
| 504  | 0.230 | 0.240 | 0.150 | 0.000 |
| 1002 | 0.410 | 0.425 | 0.190 | 0.000 |

Provenance: `cvar_v5/mc/runs/2026-05-02T153030_omega_n_audit/`.

**Theoretical explanation**: the audit's null is "ĥ_t = h_t* exactly", which never holds at finite n_calib. The audit's g-vector has a non-zero asymptotic mean (the calibrator's finite-sample bias `ε_calib(n_calib)`), and the Wald statistic scales as `n_audit · ε_calib^T Var^(-1) ε_calib` — linear in n_audit. So at large n_audit, the audit reliably detects the calibrator's bias, NOT transport violations.

`boot_remax_ridge` masks this because the ridge floor `1/n_audit` shrinks proportionally with the bootstrap covariance, leaving the ratio approximately constant near zero.

**Implication**: increasing n_audit on this audit machinery is the wrong fix. The structural problem is that the audit's null doesn't account for finite-sample calibrator bias. This escalates to `[omega-research-derivation]`.

### Sub-anchor: `[omega-research-derivation]` — partial calibration achieved 2026-05-02

Re-derive the audit so its null accounts for finite-sample calibrator bias.

**Empirical evidence from `mean_cje/` study (2026-05-02):** on a known-truth Beta DGP, the **delete-one-fold jackknife on `V̂_cal` is the load-bearing piece** for converting catastrophically narrow CIs (33% coverage) into nominal-coverage CIs (89–92% with jackknife alone, 94–96% with full-pipeline bootstrap). AIPW one-step is **inert** on this DGP (adds noise, no bias removal). See `mean_cje/README.md`.

The original CJE paper's machinery is more layered than I initially read:

```
V̂_eval (per-row IF, treats f̂ fixed)             ←  ≈ 33% coverage
  + V̂_cal (delete-one-fold jackknife on f̂)       ←  89–92% coverage
  + bootstrap (full-pipeline, refits f̂, B=2000)   ←  94–96% coverage
```

Concrete approaches to apply to the cvar_v5 audit Ω̂:

1. **Jackknife OUA on (g_1, g_2)** (planned now). Per oracle fold k:
    - refit `ĥ_t^(−k)` on CALIB \ fold_k
    - apply to AUDIT slice: `h_t^(−k)(s_audit)`
    - recompute audit moments `ḡ^(−k)` at the original `t̂`
    - `V̂_cal_g = ((K−1)/K) · Σ_k (ḡ^(−k) − ḡ̄)(ḡ^(−k) − ḡ̄)ᵀ`
    - `Ω̂ = Ω̂_audit + V̂_cal_g`. Add as a new variant (e.g. `analytical_oua`).

2. **Full-pipeline bootstrap on the audit** (later, if jackknife alone misses size). Refit `ĥ_t`, re-optimize `t̂`, recompute `ḡ`, take sample covariance over B reps.

3. **Argmax remax + V̂_cal** combined. The mean_cje study didn't have an argmax nuisance; cvar audit does. May need both.

Until size calibrates, **no audit-validated claims at production n_audit**. Default stays `boot_remax_ridge` (conservative, zero detection).

**Update 2026-05-02 — `boot_remax_oua` calibrates at original n_audit scale (249):**

| n_audit | analytical | analytical_oua | boot_remax_no_ridge | **boot_remax_oua** | boot_remax_ridge |
|---|---|---|---|---|---|
| 249  | 0.165 | 0.120 | 0.120 | **0.090** | 0.000 |
| 504  | 0.230 | 0.200 | 0.150 | 0.110 | 0.000 |
| 1002 | 0.410 | 0.360 | 0.190 | 0.175 | 0.000 |

`boot_remax_oua` = bootstrap-with-remax (captures argmax nuisance) + jackknife V̂_cal_g (captures calibrator-fit nuisance, additive). The two corrections compose: each captures a distinct nuisance the other misses.

At n_audit ≈ 249 (original spec scale): **size_dev = 0.040 < 0.050 — calibrates per the lock-in threshold.** At larger n_audit (504, 1002), residual bias-into-null leaks through (calibrator's `ε_calib` enters as a center shift, not a variance issue, and grows in `√n_audit · ε_calib` units).

**Provenance**: `cvar_v5/mc/runs/2026-05-02T194018_omega_n_audit/`.

**Decision (pending)**:
- Lock `boot_remax_oua` as the default for n_audit ≤ ~250 (the canonical workhorse audit size).
- For larger n_audit, options remain: (a) bias-correct ḡ via the same jackknife (subtract estimated calibrator bias before forming W); (b) calibrate the null empirically; (c) use a one-sided test on g_1 only.

## `[joint-calibrator]` — closed 2026-05-02

Hypothesis: a joint `ĝ(s, t)` exploiting structure across `t` (smoothness, monotonicity, parametric form) could cut `var_calib` materially below per-t isotonic. **Falsified.**

Four methods tried (smoothed_per_t, distribution_regression, bivariate_isotonic_pooled, gbm_monotone). None met the bar (`var_calib_ratio < 0.7` on ≥ 3 of 4 policies AND `rmse_ratio ≤ 1.05` AND `time_ratio ≤ 5`).

The cleanest theoretical reason — observed via the bivariate-isotonic experiment — is that **the joint constraints are inactive on `z_{i,k} = max(t_k − Y_i, 0)`**. Each row already satisfies 1-Lipschitz-in-t and non-decreasing-in-t exactly by construction; per-t isotonic in s preserves this; per-t isotonic IS the constrained-feasible squared-error minimizer. There's no additional information for a joint method to extract at our data structure.

GBM-with-monotone-constraints over-regularizes and collapses to constants on most policies. `distribution_regression` works only on U-shaped Y (`tail_heavy`, Beta(0.5, 0.5)) — 93% `var_calib` reduction there, but blows up bias 2-3× elsewhere.

**Default stays `per_t_isotonic`.** Implementations and the sweep harness were deleted after the study; if a real-data adapter ever shows a strongly U-shaped score distribution, re-implement `distribution_regression` per the formula in `calibrator.py`'s math contract.

Do not re-investigate joint methods unless the data structure changes (e.g., a target other than `(t-Y)_+` that doesn't have the row-wise 1-Lipschitz property).

## `[scaling-tracking]`

MC outputs currently go to filesystem-timestamped dirs (`mc/runs/<ts>_<mode>/`) with `run_config.json` next to each `results_mc.csv`. This is sufficient while runs are sporadic.

If MC becomes a recurring job — say ≥ 5 runs/week sustained for a month — switch to MLflow (or W&B) for cross-run comparison, leaderboards, and artifact retention. Trigger: weekly cadence sustained for a month, OR a need to compare runs across machines / users.

Until then: filesystem timestamps win on simplicity.
