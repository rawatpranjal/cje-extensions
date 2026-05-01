# Design × audit comparison

coverage = `0.25`, α = `0.3`, seed = `42`. Estimator: HT-weighted isotonic Direct CVaR-CJE. Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). Full-oracle truth = CVaR_α computed on every row's oracle label.

### base

Full-oracle CVaR_α truth = `-0.033` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 27 | +0.016 | [-0.110, +0.143] | 0.049 | ✓ | 0.60 | 1.000  | +0.000 |
| floor_tail_band | 33 | -0.041 | [-0.295, +0.212] | 0.008 | ✓ | 0.75 | 1.000  | +0.000 |
| uniform | 26 | -0.000 | [-0.123, +0.123] | 0.033 | ✓ | 0.66 | 1.000  | +0.000 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.971 ✅ | p=1.000 ✅ | p=0.999 ✅ |
| floor_tail_band | p=0.468 ✅ | p=1.000 ✅ | p=0.835 ✅ |
| uniform | p=0.875 ✅ | p=1.000 ✅ | p=0.990 ✅ |

### clone

Full-oracle CVaR_α truth = `-0.090` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 27 | +0.014 | [-0.121, +0.150] | 0.105 | ✓ | 0.65 | 0.682  | -0.037 |
| floor_tail_band | 23 | -0.036 | [-0.293, +0.221] | 0.054 | ✓ | 0.75 | 0.099  | -0.100 |
| uniform | 26 | -0.009 | [-0.149, +0.130] | 0.081 | ✓ | 0.68 | 0.852  | +0.009 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.327 ✅ | p=0.789 ✅ | p=0.692 ✅ |
| floor_tail_band | p=0.413 ✅ | p=0.830 ✅ | p=0.771 ✅ |
| uniform | p=0.666 ✅ | p=0.921 ✅ | p=0.928 ✅ |

### premium

Full-oracle CVaR_α truth = `+0.028` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 26 | +0.078 | [-0.046, +0.203] | 0.050 | ✓ | 0.54 | 0.745  | -0.016 |
| floor_tail_band | 30 | +0.026 | [-0.252, +0.305] | 0.002 | ✓ | 0.73 | 0.290  | -0.055 |
| uniform | 26 | +0.047 | [-0.059, +0.153] | 0.019 | ✓ | 0.57 | 0.014 🔥 | +0.102 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.644 ✅ | p=0.968 ✅ | p=0.903 ✅ |
| floor_tail_band | p=0.192 ✅ | p=0.992 ✅ | p=0.489 ✅ |
| uniform | p=0.875 ✅ | p=0.925 ✅ | p=0.986 ✅ |

### parallel_universe_prompt

Full-oracle CVaR_α truth = `-0.169` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 29 | -0.106 | [-0.280, +0.067] | 0.062 | ✓ | 0.76 | 0.548  | -0.047 |
| floor_tail_band | 33 | -0.104 | [-0.257, +0.049] | 0.065 | ✓ | 0.76 | 0.636  | -0.025 |
| uniform | 26 | -0.099 | [-0.310, +0.111] | 0.069 | ✓ | 0.64 | 0.258  | -0.057 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.937 ✅ | p=0.950 ✅ | p=0.996 ✅ |
| floor_tail_band | p=0.828 ✅ | p=0.980 ✅ | p=0.983 ✅ |
| uniform | p=0.666 ✅ | p=0.934 ✅ | p=0.928 ✅ |

### unhelpful

Full-oracle CVaR_α truth = `-0.317` (n_total=100)

**CI + variance decomposition + mean audit (per design):**

| design | n_slice | est | 95% CI (cal-aware) | \|err\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |
|---|---:|---:|---|---:|:--:|---:|---:|---:|
| floor_tail | 26 | -0.303 | [-0.761, +0.155] | 0.013 | ✓ | 0.79 | 0.535  | -0.030 |
| floor_tail_band | 33 | -0.160 | [-0.334, +0.014] | 0.156 | ✓ | 0.66 | 0.015 🔥 | -0.154 |
| uniform | 26 | -0.451 | [-1.257, +0.355] | 0.135 | ✓ | 0.73 | 0.497  | +0.031 |

**CVaR audit p-values (per design × audit_variant):**

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | p=0.804 ✅ | p=0.997 ✅ | p=0.982 ✅ |
| floor_tail_band | p=0.259 ✅ | p=0.602 ✅ | p=0.670 ✅ |
| uniform | p=0.111 ✅ | p=0.910 ✅ | p=0.336 ✅ |


## Aggregate summary

Across all policies. coverage = 0.25, α = 0.3.

| design | audit | mean \|err\| | reject_rate | n_cells |
|---|---|---:|---:|---:|
| floor_tail | g1_only | 0.056 | 0.00 | 5 |
| floor_tail | g2_only | 0.056 | 0.00 | 5 |
| floor_tail | two_moment | 0.056 | 0.00 | 5 |
| floor_tail_band | g1_only | 0.057 | 0.00 | 5 |
| floor_tail_band | g2_only | 0.057 | 0.00 | 5 |
| floor_tail_band | two_moment | 0.057 | 0.00 | 5 |
| uniform | g1_only | 0.067 | 0.00 | 5 |
| uniform | g2_only | 0.067 | 0.00 | 5 |
| uniform | two_moment | 0.067 | 0.00 | 5 |
