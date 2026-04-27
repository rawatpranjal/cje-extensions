# SCHEMA — Authors' Arena Pipeline (citations)

Source of truth for the CJE paper's Arena experiment pipeline. Every numeric
constant below is cited to file:line in the authors' repo at
`~/Dropbox/cvar-cje-data/cje-arena-experiments/` or to the pip-installed
`cje-eval==0.2.10` package. Any deviation in `cvar_v3/workhorse.py` must be
justified here.

## 1. Dataset

**Location**: `~/Dropbox/cvar-cje-data/cje-arena-experiments/data/cje_dataset.jsonl`

**Size**: 9.5 MB, **4989 rows** (paper reports 4961 after TF filtering).

**sha256** (2026-04-24):
`c17fdc6a2765a207e3f11dc5579c0102e76bf871359b84411cb9558570a70b47`

**JSONL schema per row** (verified against `head -1`):
```json
{
  "prompt": "...",
  "response": "...",                       // base-policy response
  "base_policy_logprob": -151.74,
  "target_policy_logprobs": {
    "clone": -152.38,
    "parallel_universe_prompt": -159.49,
    "premium": -261.02,
    "unhelpful": -208.18
  },
  "metadata": {
    "prompt_id": "arena_0",
    "judge_score": 0.85,                   // GPT-4.1-nano, already in [0,1]
    "oracle_label": 0.7                    // GPT-5, already in [0,1]
  }
}
```

No further normalization needed — scores are already in [0, 1].

**Target-policy fresh draws**: `data/responses/{policy}_responses.jsonl` (one file per non-base policy). Each row has its own `metadata.judge_score` and `metadata.oracle_label` for that policy's response to a given prompt. These are required for the Direct estimator (judge the fresh response, no IPS needed).

## 2. Oracle coverage mechanism

**Source**: `ablations/core/base.py:100-143`

Not hash-based. Uses `random.sample` seeded with the **experiment seed**, so the
specific 25% slice varies across seeds:

```python
random.seed(seed)        # base.py:100 -- spec.seed, not 42
np.random.seed(seed)
oracle_indices = [i for i, s in enumerate(dataset.samples) if s.oracle_label is not None]
n_keep = max(2, int(len(oracle_indices) * spec.oracle_coverage))
keep_indices = set(random.sample(oracle_indices, min(n_keep, len(oracle_indices))))
# Samples NOT in keep_indices have oracle_label set to None
```

There is no fixed-42 oracle mask in the paper's pipeline; across-seed
variability of which 25% becomes the oracle slice is part of the experiment.
`workhorse.estimate_all_policies` mirrors this by defaulting `mask_seed=seed`.
For diagnostic runs that need a frozen slice (e.g. `validate_mean.py`), pass
an explicit `mask_seed` -- but be aware that fixing it deviates from the
paper's protocol.

## 3. Cross-fitting folds and calibration mode

**Source**: `ablations/core/base.py:535-549` + `ablations/config.py:36`

```python
# config.py:31-36
"use_covariates": [False, True],         # ablation toggle
"reward_calibration_mode": "auto",        # forwarded into spec.extra

# base.py:535-549 (with spec.extra populated from config.py)
reward_calibration_mode = spec.extra.get("reward_calibration_mode", "monotone")
# -> resolves to "auto" because config.py:36 sets it explicitly.

calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    enable_cross_fit=True,
    n_folds=5,
    calibration_mode="auto",              # FlexibleCalibrator selects mode
    random_seed=seed,
    covariate_names=["response_length"]   # for direct+cov
)
```

The `"monotone"` literal in `base.py:535` is only the fallback when `spec.extra`
is empty; the active experiment config in `config.py:36` is `"auto"`. With
`"auto"`, `cje.calibration.judge.FlexibleCalibrator` picks between
`monotone` (plain isotonic) and `two_stage` (flexible first stage + isotonic
debias) per fold. On the 25%-oracle Arena slice with `response_length`
covariates, it consistently selects `"two_stage"` (verified: lower in-fold
RMSE, ~0.180 vs ~0.192). So the active backbone for `direct+cov` is
**two-stage flexible calibration**, not plain monotone isotonic.

Folds are hash-based on `prompt_id`, computed inside `cje.calibrate_dataset`.

## 4. Direct+cov estimator

**Source**: `ablations/core/base.py:204-213` (estimator map for "direct")

```python
CalibratedDirectEstimator(
    target_policies=list(s.target_policies),
    reward_calibrator=cal_result.calibrator,     # monotone isotonic, fit on 25% oracle slice
    run_diagnostics=True,
    oua_jackknife=True,                           # oracle-uncertainty augmentation
    inference_method="cluster_robust",           # cluster-robust SE on prompt_id
    n_bootstrap=2000,                             # used when inference_method="bootstrap"
    use_augmented_estimator=True,                 # AIPW-style debiasing
)
```

From `cje.estimators.direct_method` in `cje-eval==0.2.10` (pinned in
`requirements.txt` of the authors' repo).

**Two-stage ("direct+cov")** happens inside `calibrate_dataset` via
`covariate_names=["response_length"]`. Stage 1 = isotonic
`m1(s) ~ E[Y|S=s]`. Stage 2 = fit residual on covariate. Response length is
computed by `cje` from the response text — raw numeric, no log / standardize
observed at this level.

## 5. Bootstrap / inference

**Source**: `ablations/core/base.py:184-212, 667-684`

- Default `inference_method="cluster_robust"` — closed-form cluster-robust SE,
  **not** bootstrap, for point-estimate accuracy experiments.
- `inference_method="bootstrap"` available via `spec.extra`, with
  `n_bootstrap=2000` and cluster by `prompt_id`.
- Fresh-draw oracle labels are masked to match the calibration oracle slice
  (base.py:667-684) — prevents the bootstrap from "seeing" more oracle than
  the calibration used.

## 6. TTC

Not in local code. Lives inside `cje-eval`'s diagnostics (`compute_overlap_metrics`
at base.py:464-484). Exact formula visible only in the installed library;
read there when we add our own TTC check in the workhorse.

## 7. Oracle ground truth per policy

**Source**: `ablations/core/base.py:354-411` (`_load_oracle_ground_truth`)

For each target policy, oracle mean is literally:

```python
df.groupby('policy').oracle_label.mean()
```

where target-policy oracle labels come from
`data/responses/{policy}_responses.jsonl`. Approximate values (from the
authors' README example output):

| Policy | Oracle mean |
|---|---|
| base | ≈ 0.758 |
| clone | ≈ 0.762 |
| parallel_universe_prompt | ≈ 0.771 |
| premium | ≈ 0.762 |
| unhelpful | ≈ 0.143 |

Our Mean CJE must land within 0.01 of these AND its 95% CI must contain
them. This is the blocking benchmark in `cvar_v3/validate_mean.py`.

## 8. Full sweep config

**Source**: `ablations/config.py`
- `oracle_coverage ∈ {0.05, 0.10, 0.25, 0.50, 1.00}` (line 24)
- `sample_sizes ∈ {250, 500, 1000, 2500, 5000}` (line 23)
- `seeds = range(50)` (line 38)
- `n_folds = 5` (line 86)
- `var_cap = 1.0` (line 44) — SIMCal variance budget

**Our choice for CVaR run**: `oracle_coverage=0.25`, `sample_sizes=5000`,
`seeds = range(20)` (reduced from 50 for runtime), `alphas ∈ {0.05, 0.10, 0.20}`.

## 9. Benchmark numbers (measured end-to-end, seed=42, full oracle, single-stage monotone)

Run on 2026-04-24 using `cje-eval==0.2.10` (dist-info), `cje.__version__ == 0.2.5`
(internal), `CalibratedDirectEstimator` with `inference_method="cluster_robust"`,
`oua_jackknife=True`, `use_augmented_estimator=True`:

| Policy | Direct estimate | Oracle (README) | |Δ| | Pass? |
|---|---|---|---|---|
| clone                    | 0.76201 | 0.762 | 0.00001 | ✓ |
| premium                  | 0.76333 | 0.762 | 0.00133 | ✓ |
| parallel_universe_prompt | 0.76722 | 0.771 | 0.00378 | ✓ |
| unhelpful                | 0.43444 | 0.143 | 0.29144 | ✗ by design (extreme shift) |

3 of 4 policies pass the |Δ| ≤ 0.01 benchmark at full oracle. `unhelpful` is
known to fail at any oracle coverage due to catastrophic distribution shift
("NaN by design" per README). Our blocking benchmark in `validate_mean.py`
therefore excludes `unhelpful` and requires 3/3 non-adversarial policies to
match within 0.01 at 25% oracle.

## 10. What the workhorse will do

- **Mean path**: call `cje.CalibratedDirectEstimator` with the same parameters
  as `ablations/core/base.py:204-213` — paper match guaranteed.
- **CVaR path**: reuse `cje.calibrate_dataset` to get the fitted calibrator,
  then apply our grid-search stop-loss isotonic calibrator
  (`estimate_direct_cvar_isotonic`, lifted from `cvar_v3/demo_utils.py:328-341`)
  on top. Bootstrap by resampling prompts (cluster bootstrap) and refitting the
  CVaR threshold each rep.
- **Transport audit**: implement `two_moment_wald_audit` from
  `cvar_v3/demo_utils.py:434-459` in the workhorse module.

## 11. Semi-synthetic DGP for power analysis

**Source**: `cvar_v3/dgp.py`. Used by `cvar_v3/run_monte_carlo.py` and
`cvar_v3/tests_dgp.py`.

**Per-policy DGP** (5 policies: base + 4 targets):

- **Y marginal**: empirical CDF over the policy's observed oracle labels.
  For `unhelpful` (`p_zero ≈ 0.31` mass at Y=0), a mixture of `P(Y=0)=p_zero`
  plus the empirical CDF on Y > 0. (`dgp.py:_fit_one_policy`,
  `_sample_y`.)
- **S | Y**: monotone isotonic `m_p(Y)` (`IsotonicRegression(increasing=True)`)
  + heteroscedastic Gaussian noise binned by Y-quartile (4 bins). Output
  clipped to [0,1]. For `unhelpful` Y=0 mass, S draws from
  `N(mean_s_at_zero, sigma_s_at_zero)` regardless of `m_override` since base
  never observes Y=0. (`dgp.py:_fit_one_policy`, `sample_synthetic`.)
- **Cross-policy joint** (`r(Y_base, Y_target) ≈ 0.81` in real data) is
  **not preserved** — each policy is sampled independently. Per-policy
  marginals are what the calibrate-on-base-eval-on-target estimator
  actually consumes.

**Mis-specification knob** δ via `sample_synthetic(..., delta, perturbation, m_override)`:

- `m_override=PolicyDGP`: the conditional mean comes from this DGP's `m_iso`
  (typically pass `dgp_base` to mimic the calibrator's training distribution).
- `perturbation="uniform"`: `m_target(y) ← m_base(y) + δ`. Tests audit moment g₂.
- `perturbation="tail"` (default in `run_monte_carlo.py`): `m_target(y) ← m_base(y) − δ`
  if `y ≤ q_α(Y_base)` else `m_base(y)`. Tests both g₁ and g₂.

**Cells in `cvar_v3/run_monte_carlo.py`** (smoke vs full):

| cell_kind        | calib  | eval   | δ                    | n_eval         |
|------------------|--------|--------|----------------------|----------------|
| size_diagnostic  | base   | base   | 0                    | 2,500 / 5,000  |
| power_curve      | base   | clone  | {0, 0.02, 0.05, 0.10, 0.20} | 2,500 / 5,000 |
| scaling          | base   | clone  | 0.05                 | {500, 2,500}   |

`--medium` runs all 4 targets at α=0.10 with 100 reps; `--full` widens to
multi-α and 4 sample sizes with 200 reps.

**Truth**: `cvar_truth(dgp, alpha, n_truth=200_000)` — population CVaR of
the eval DGP. δ-invariant (Y is δ-invariant; only S shifts).

## 12. Audit Σ̂ correction (appendix gap viii)

**Source**: `cvar_v3/workhorse.py:two_moment_wald_audit_xf` (the xf in the
name is historical — the implementation is paired bootstrap, not K-fold).

The naive `two_moment_wald_audit` uses sample-cov on `s_audit` for Σ̂.
Asymptotic theory says

  n_audit · Var(ḡ) = Σ_eval + (variance from plug-in t̂ and ĝ)

The naive Σ̂ misses the plug-in term. Empirically on the truest null
(base→base, δ=0, n_oracle=625, n_audit=2000): naive size = 0.51,
target = 0.05 (W mean = 9.6, χ²₂ target = 2.0).

**Fix**: paired non-parametric bootstrap of the (s_train, s_audit)
clusters with **t̂ re-maximised inside each rep**. For B replicates b:

```python
idx_t = rng.integers(0, n_train, size=n_train)   # bootstrap train
idx_a = rng.integers(0, n_audit, size=n_audit)   # bootstrap audit
# Re-maximize t̂_b on bootstrap data via the dual:
_, t_b, _, _ = estimate_direct_cvar_isotonic(
    s_train[idx_t], y_train[idx_t], s_audit[idx_a], alpha, grid_size,
)
# Compute moments at t_b on bootstrap data:
g1_b = mean( 1{y_audit[idx_a] ≤ t_b} − α )
g2_b = mean( (t_b − y_audit[idx_a])_+ − ĝ_b(s_audit[idx_a]) )
```

Then `Σ̂_boot = sample_cov(g_per_boot)` and `W = ḡ_full_data' Σ̂_boot⁻¹ ḡ_full_data ~ χ²₂`.

**Calibration achieved** on the truest null: empirical size 0.05 (target
0.05), W mean 2.13 (target 2.0), W median 1.50 (target 1.39).

**What didn't work and why**:

1. K-fold cross-fitting alone (residuals on held-out folds): captures
   pointwise calibrator error but NOT the integrated-bias variance term.
   Empirical size still ≈ 0.50.
2. Paired bootstrap of (train, audit) with **t̂ held fixed**: captures
   calibrator-fit and audit-set variance but misses t̂'s sampling
   variance. Empirical size dropped to 0.34 — better but not nominal.
3. Re-maximizing t̂ inside the bootstrap is what closes the gap. The
   non-smooth argmax is the dominant missing-variance source for the
   χ²₂ Wald in this estimator.

**Cost**: B isotonic + grid-search per audit call. With B=80 and
grid_size=31: ~0.5 sec/audit on n_train=625. The medium MC runs
2500 outer reps in ~18 min on 6 workers.

## 13. Audit-gated CI coverage (framework's intended interpretation)

**Source**: `cvar_v3/make_power_report.py` §3, validated empirically on
medium MC (`cvar_v3/power_analysis.md`).

The cluster bootstrap CI is calibrated for **variance**, not **bias**.
Under transport failure (calibrator fit on base, applied to a target with
different Y marginal), the Direct CVaR estimator inherits a small
finite-sample bias and the CI is centered slightly off-truth, producing
unconditional under-coverage (0.78 at base→clone, δ=0).

**The framework's prescription** — and what the medium MC empirically
confirms — is to interpret the CI **conditional on the audit accepting**:

| Target (δ=0) | Coverage all reps | Coverage \| audit accepts |
|---|---|---|
| base→base    | 0.93 | **0.95** ✓ |
| base→premium | 0.92 | **0.94** ✓ |
| base→pup     | 0.89 | **0.89** ✓ |
| base→clone   | 0.78 | 0.81 △ (audit at small natural mis-spec is low-power) |
| base→unhelpful | 0.18 | **1.00** ✓ (n=10; audit catches catastrophic transport) |

**Why BCa won't help**: BCa corrects bias *within the bootstrap
distribution* (when bootstrap reps are skewed around the original
estimate). Our bias is *between Ĉ and C_true* — the bootstrap reps are
correctly centered around Ĉ, just at the wrong place. BCa is the wrong
tool. Audit-gated refusal IS the right tool, and it works.

**Implication for the appendix**: report unconditional CI coverage AS
WELL AS the audit-gated coverage; cite the latter as the operative
metric for level claims. When the audit rejects, refuse-level
(per Sec.~4 of the main paper).

## 14. Information flow — what each component is allowed to see

Explicit accounting of which observable each computation in the
pipeline consumes. Anyone reviewing for oracle leakage should be able
to walk this table and confirm the estimator and CI never touch
eval-side oracle labels.

| Component (file:function)                                    | Reads `s_calib` | Reads `y_calib` | Reads `s_eval` | Reads `y_eval` | Reads `C_true` |
|--------------------------------------------------------------|:---:|:---:|:---:|:---:|:---:|
| `dgp.py:fit_arena_dgp`                                       | (real Arena `(S,Y)` once at fit time) | | | | — |
| `dgp.py:sample_synthetic`                                    | — | — | — | — | — |
| `dgp.py:cvar_truth` (defines `C_true`)                       | — | — | — | — | (200k separate sample) |
| `workhorse.py:estimate_direct_cvar_isotonic` (point Ĉ, t̂)   | ✓ | ✓ | ✓ | **✗** | ✗ |
| `workhorse.py:cluster_bootstrap_cvar` (CI)                   | ✓ | ✓ | ✓ | **✗** | ✗ |
| `workhorse.py:two_moment_wald_audit_xf` (audit)              | ✓ | ✓ | ✓ | ✓ (by design) | ✗ |
| `run_monte_carlo.py:_run_one` `ci_covers_truth` (diagnostic) | — | — | — | — | ✓ (diagnostic only, never feeds back) |

Key invariants:

1. **The estimator and CI never see `y_eval`.** Their function
   signatures literally do not accept it. A reviewer can grep for
   `y_eval` in `cluster_bootstrap_cvar` and `estimate_direct_cvar_isotonic`
   and find nothing.
2. **The audit sees `y_eval` because that's its purpose.** The
   χ²₂ Wald moments `g_1 = 1{Y ≤ t̂} − α` and `g_2 = (t−Y)_+ − ĝ(S)`
   are explicitly defined to use held-out oracle labels (Sec.~4 of the
   main paper). In deployment this would be a small annotated audit
   slice; in the MC we use the full eval set and document the resulting
   power inflation.
3. **`C_true` is a separate-RNG 200,000-draw quantity** computed
   before the MC starts. It is read only at the diagnostic step
   (`ci_covers_truth`, signed bias) and never enters any inferential
   computation. Audit-gated coverage `P(CI ∋ C_true | audit accepts)`
   is computed *post-hoc* over the MC's CSV; it is the deployment-relevant
   subgroup metric, not a leakage path.
4. **The DGP's perturbation parameters** (e.g., `q_low_threshold` for
   the `tail` shape) are part of the DGP's *construction*, not knowledge
   passed to the estimator. The estimator observes only the realised
   `(S, Y)` samples.
5. **`m_override`** in `sample_synthetic` is a DGP-side knob that
   defines what δ=0 means under our null (target's S follows base's m).
   The estimator does not see it.

What this table does not certify:

- **The audit is correlated with the CI** (both depend on the same
  `(s_train, y_train, s_eval, y_eval)`). This is finite-sample dependence,
  not leakage; the paired bootstrap with t̂ re-maximisation captures it.
- **The DGP itself is fit from real data** (standard semi-synthetic
  design); results are conditional on the DGP being a plausible model.
- **Audit-gated coverage is a data-derived subgroup metric** — by design,
  conditioning on audit-accepts selects reps where data look transport-y,
  which correlates with the estimator being closer to truth. This is the
  audit's *intended* effect, not double-dipping; we report unconditional
  and gated coverage side-by-side in `cvar_v3/power_analysis.md`.
