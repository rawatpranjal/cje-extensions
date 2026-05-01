# Design × audit comparison

coverage = `0.25`, α = `0.05`, seed = `42`. Estimator: HT-weighted isotonic Direct CVaR-CJE. Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). Full-oracle truth = CVaR_α computed on every row's oracle label.

### base

Full-oracle CVaR_α truth = `-0.273` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.258 \|err\|=0.01 p=0.97 ✅ | est=-0.258 \|err\|=0.01 p=1.00 ✅ | est=-0.258 \|err\|=0.01 p=1.00 ✅ |
| floor_tail_band | est=-0.172 \|err\|=0.10 p=0.90 ✅ | est=-0.172 \|err\|=0.10 p=1.00 ✅ | est=-0.172 \|err\|=0.10 p=0.99 ✅ |
| uniform | est=-0.288 \|err\|=0.02 p=0.95 ✅ | est=-0.288 \|err\|=0.02 p=1.00 ✅ | est=-0.288 \|err\|=0.02 p=1.00 ✅ |

### clone

Full-oracle CVaR_α truth = `-0.357` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.270 \|err\|=0.09 p=0.63 ✅ | est=-0.270 \|err\|=0.09 p=0.98 ✅ | est=-0.270 \|err\|=0.09 p=0.89 ✅ |
| floor_tail_band | est=-0.170 \|err\|=0.19 p=0.67 ✅ | est=-0.170 \|err\|=0.19 p=0.93 ✅ | est=-0.170 \|err\|=0.19 p=0.93 ✅ |
| uniform | est=-0.342 \|err\|=0.01 p=0.89 ✅ | est=-0.342 \|err\|=0.01 p=0.98 ✅ | est=-0.342 \|err\|=0.01 p=0.99 ✅ |

### premium

Full-oracle CVaR_α truth = `-0.195` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.196 \|err\|=0.00 p=0.92 ✅ | est=-0.196 \|err\|=0.00 p=1.00 ✅ | est=-0.196 \|err\|=0.00 p=1.00 ✅ |
| floor_tail_band | est=-0.156 \|err\|=0.04 p=0.84 ✅ | est=-0.156 \|err\|=0.04 p=0.99 ✅ | est=-0.156 \|err\|=0.04 p=0.98 ✅ |
| uniform | est=-0.127 \|err\|=0.07 p=0.95 ✅ | est=-0.127 \|err\|=0.07 p=0.96 ✅ | est=-0.127 \|err\|=0.07 p=1.00 ✅ |

### parallel_universe_prompt

Full-oracle CVaR_α truth = `-0.380` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.374 \|err\|=0.01 p=0.88 ✅ | est=-0.374 \|err\|=0.01 p=1.00 ✅ | est=-0.374 \|err\|=0.01 p=0.99 ✅ |
| floor_tail_band | est=-0.252 \|err\|=0.13 p=0.94 ✅ | est=-0.252 \|err\|=0.13 p=0.99 ✅ | est=-0.252 \|err\|=0.13 p=1.00 ✅ |
| uniform | est=-0.501 \|err\|=0.12 p=0.95 ✅ | est=-0.501 \|err\|=0.12 p=1.00 ✅ | est=-0.501 \|err\|=0.12 p=1.00 ✅ |

### unhelpful

Full-oracle CVaR_α truth = `-0.624` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.507 \|err\|=0.12 p=0.85 ✅ | est=-0.507 \|err\|=0.12 p=0.97 ✅ | est=-0.507 \|err\|=0.12 p=0.99 ✅ |
| floor_tail_band | est=-0.339 \|err\|=0.28 p=0.53 ✅ | est=-0.339 \|err\|=0.28 p=0.80 ✅ | est=-0.339 \|err\|=0.28 p=0.87 ✅ |
| uniform | est=-0.514 \|err\|=0.11 p=0.97 ✅ | est=-0.514 \|err\|=0.11 p=0.93 ✅ | est=-0.514 \|err\|=0.11 p=0.99 ✅ |


## Aggregate summary

Across all policies. coverage = 0.25, α = 0.05.

| design | audit | mean \|err\| | reject_rate | n_cells |
|---|---|---:|---:|---:|
| floor_tail | g1_only | 0.045 | 0.00 | 5 |
| floor_tail | g2_only | 0.045 | 0.00 | 5 |
| floor_tail | two_moment | 0.045 | 0.00 | 5 |
| floor_tail_band | g1_only | 0.148 | 0.00 | 5 |
| floor_tail_band | g2_only | 0.148 | 0.00 | 5 |
| floor_tail_band | two_moment | 0.148 | 0.00 | 5 |
| uniform | g1_only | 0.066 | 0.00 | 5 |
| uniform | g2_only | 0.066 | 0.00 | 5 |
| uniform | two_moment | 0.066 | 0.00 | 5 |
