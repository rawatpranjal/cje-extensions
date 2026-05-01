# Design × audit comparison

coverage = `0.25`, α = `0.1`, seed = `42`. Estimator: HT-weighted isotonic Direct CVaR-CJE. Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). Full-oracle truth = CVaR_α computed on every row's oracle label.

### base

Full-oracle CVaR_α truth = `-0.181` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 27 | -0.154 | [-0.387, +0.080] | 0.027 | ✓ | 0.80 | 1.000  | +0.000 |
| floor_tail_band | 25 | -0.140 | [-0.321, +0.040] | 0.041 | ✓ | 0.76 | 1.000  | +0.000 |
| uniform | 26 | -0.146 | [-0.429, +0.137] | 0.035 | ✓ | 0.63 | 1.000  | +0.000 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.919 ✅ | p=1.000 ✅ | p=0.995 ✅ |
| floor_tail_band | p=0.809 ✅ | p=1.000 ✅ | p=0.976 ✅ |
| uniform | p=0.754 ✅ | p=1.000 ✅ | p=0.957 ✅ |

### clone

Full-oracle CVaR_α truth = `-0.272` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 27 | -0.164 | [-0.421, +0.093] | 0.107 | ✓ | 0.81 | 0.682  | -0.037 |
| floor_tail_band | 23 | -0.139 | [-0.314, +0.036] | 0.132 | ✓ | 0.76 | 0.072  | -0.117 |
| uniform | 26 | -0.174 | [-0.504, +0.156] | 0.097 | ✓ | 0.66 | 0.852  | +0.009 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.641 ✅ | p=0.951 ✅ | p=0.909 ✅ |
| floor_tail_band | p=0.849 ✅ | p=0.931 ✅ | p=0.984 ✅ |
| uniform | p=0.906 ✅ | p=0.982 ✅ | p=0.994 ✅ |

### premium

Full-oracle CVaR_α truth = `-0.082` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 26 | -0.104 | [-0.284, +0.075] | 0.022 | ✓ | 0.80 | 0.745  | -0.016 |
| floor_tail_band | 26 | -0.116 | [-0.291, +0.059] | 0.034 | ✓ | 0.74 | 0.599  | -0.032 |
| uniform | 26 | -0.061 | [-0.206, +0.083] | 0.021 | ✓ | 0.46 | 0.014 🔥 | +0.102 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.840 ✅ | p=0.997 ✅ | p=0.981 ✅ |
| floor_tail_band | p=0.831 ✅ | p=0.961 ✅ | p=0.977 ✅ |
| uniform | p=0.937 ✅ | p=0.956 ✅ | p=0.996 ✅ |

### parallel_universe_prompt

Full-oracle CVaR_α truth = `-0.319` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 29 | -0.262 | [-0.621, +0.096] | 0.057 | ✓ | 0.80 | 0.548  | -0.047 |
| floor_tail_band | 25 | -0.184 | [-0.385, +0.018] | 0.136 | ✓ | 0.62 | 0.217  | -0.071 |
| uniform | 26 | -0.311 | [-0.888, +0.267] | 0.009 | ✓ | 0.71 | 0.258  | -0.057 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.646 ✅ | p=0.994 ✅ | p=0.914 ✅ |
| floor_tail_band | p=0.961 ✅ | p=0.987 ✅ | p=0.999 ✅ |
| uniform | p=0.638 ✅ | p=0.957 ✅ | p=0.925 ✅ |

### unhelpful

Full-oracle CVaR_α truth = `-0.502` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 26 | -0.485 | [-1.041, +0.070] | 0.017 | ✓ | 0.83 | 0.535  | -0.030 |
| floor_tail_band | 25 | -0.286 | [-0.597, +0.025] | 0.216 | ✓ | 0.70 | 0.032 🔥 | -0.167 |
| uniform | 26 | -0.508 | [-1.434, +0.417] | 0.006 | ✓ | 0.75 | 0.497  | +0.031 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.803 ✅ | p=0.984 ✅ | p=0.982 ✅ |
| floor_tail_band | p=0.214 ✅ | p=0.715 ✅ | p=0.645 ✅ |
| uniform | p=0.854 ✅ | p=0.923 ✅ | p=0.978 ✅ |


## Aggregate summary

Across all policies. coverage = 0.25, α = 0.1.

| design | audit | mean \|err\| | reject_rate | n_cells |
|---|---|---:|---:|---:|
| floor_tail | g1_only | 0.046 | 0.00 | 5 |
| floor_tail | g2_only | 0.046 | 0.00 | 5 |
| floor_tail | two_moment | 0.046 | 0.00 | 5 |
| floor_tail_band | g1_only | 0.112 | 0.00 | 5 |
| floor_tail_band | g2_only | 0.112 | 0.00 | 5 |
| floor_tail_band | two_moment | 0.112 | 0.00 | 5 |
| uniform | g1_only | 0.034 | 0.00 | 5 |
| uniform | g2_only | 0.034 | 0.00 | 5 |
| uniform | two_moment | 0.034 | 0.00 | 5 |
