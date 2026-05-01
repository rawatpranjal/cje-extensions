# Design × audit comparison

coverage = `0.25`, α = `0.2`, seed = `42`. Estimator: HT-weighted isotonic Direct CVaR-CJE. Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). Full-oracle truth = CVaR_α computed on every row's oracle label.

### base

Full-oracle CVaR_α truth = `-0.087` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 27 | -0.051 | [-0.219, +0.117] | 0.036 | ✓ | 0.78 | 1.000  | +0.000 |
| floor_tail_band | 25 | -0.076 | [-0.321, +0.168] | 0.010 | ✓ | 0.78 | 1.000  | +0.000 |
| uniform | 26 | -0.057 | [-0.227, +0.113] | 0.030 | ✓ | 0.72 | 1.000  | +0.000 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.886 ✅ | p=1.000 ✅ | p=0.990 ✅ |
| floor_tail_band | p=0.747 ✅ | p=1.000 ✅ | p=0.960 ✅ |
| uniform | p=0.875 ✅ | p=1.000 ✅ | p=0.990 ✅ |

### clone

Full-oracle CVaR_α truth = `-0.118` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 27 | -0.056 | [-0.239, +0.127] | 0.062 | ✓ | 0.79 | 0.682  | -0.037 |
| floor_tail_band | 23 | -0.074 | [-0.313, +0.165] | 0.044 | ✓ | 0.78 | 0.072  | -0.117 |
| uniform | 26 | -0.071 | [-0.266, +0.123] | 0.047 | ✓ | 0.73 | 0.852  | +0.009 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.351 ✅ | p=0.837 ✅ | p=0.699 ✅ |
| floor_tail_band | p=0.310 ✅ | p=0.891 ✅ | p=0.685 ✅ |
| uniform | p=0.583 ✅ | p=0.943 ✅ | p=0.891 ✅ |

### premium

Full-oracle CVaR_α truth = `-0.044` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 26 | -0.000 | [-0.132, +0.131] | 0.043 | ✓ | 0.69 | 0.745  | -0.016 |
| floor_tail_band | 26 | -0.032 | [-0.324, +0.259] | 0.011 | ✓ | 0.79 | 0.599  | -0.032 |
| uniform | 26 | -0.002 | [-0.113, +0.108] | 0.041 | ✓ | 0.63 | 0.014 🔥 | +0.102 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.943 ✅ | p=0.974 ✅ | p=0.997 ✅ |
| floor_tail_band | p=0.413 ✅ | p=0.951 ✅ | p=0.724 ✅ |
| uniform | p=0.969 ✅ | p=0.943 ✅ | p=0.997 ✅ |

### parallel_universe_prompt

Full-oracle CVaR_α truth = `-0.231` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 29 | -0.157 | [-0.396, +0.083] | 0.074 | ✓ | 0.78 | 0.548  | -0.047 |
| floor_tail_band | 33 | -0.124 | [-0.294, +0.045] | 0.106 | ✓ | 0.71 | 0.304  | -0.051 |
| uniform | 26 | -0.154 | [-0.463, +0.155] | 0.077 | ✓ | 0.64 | 0.258  | -0.057 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.537 ✅ | p=0.955 ✅ | p=0.876 ✅ |
| floor_tail_band | p=0.880 ✅ | p=0.983 ✅ | p=0.991 ✅ |
| uniform | p=0.347 ✅ | p=0.950 ✅ | p=0.714 ✅ |

### unhelpful

Full-oracle CVaR_α truth = `-0.377` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 26 | -0.374 | [-0.884, +0.136] | 0.003 | ✓ | 0.78 | 0.535  | -0.030 |
| floor_tail_band | 33 | -0.195 | [-0.438, +0.049] | 0.183 | ✓ | 0.61 | 0.015 🔥 | -0.154 |
| uniform | 26 | -0.504 | [-1.407, +0.399] | 0.127 | ✓ | 0.74 | 0.497  | +0.031 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.466 ✅ | p=0.988 ✅ | p=0.874 ✅ |
| floor_tail_band | p=0.032 🔥 | p=0.577 ✅ | p=0.289 ✅ |
| uniform | p=0.658 ✅ | p=0.937 ✅ | p=0.909 ✅ |


## Aggregate summary

Across all policies. coverage = 0.25, α = 0.2.

| design | audit | mean \|err\| | reject_rate | n_cells |
|---|---|---:|---:|---:|
| floor_tail | g1_only | 0.044 | 0.00 | 5 |
| floor_tail | g2_only | 0.044 | 0.00 | 5 |
| floor_tail | two_moment | 0.044 | 0.00 | 5 |
| floor_tail_band | g1_only | 0.071 | 0.20 | 5 |
| floor_tail_band | g2_only | 0.071 | 0.00 | 5 |
| floor_tail_band | two_moment | 0.071 | 0.00 | 5 |
| uniform | g1_only | 0.064 | 0.00 | 5 |
| uniform | g2_only | 0.064 | 0.00 | 5 |
| uniform | two_moment | 0.064 | 0.00 | 5 |
