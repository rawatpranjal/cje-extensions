# Design × audit comparison

coverage = `0.5`, α = `0.1`, seed = `42`. Estimator: HT-weighted isotonic Direct CVaR-CJE. Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). Full-oracle truth = CVaR_α computed on every row's oracle label.

### base

Full-oracle CVaR_α truth = `+0.000` (n_total=10)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=+0.000 \|err\|=0.00 p=0.98 ✅ | est=+0.000 \|err\|=0.00 p=1.00 ✅ | est=+0.000 \|err\|=0.00 p=1.00 ✅ |
| uniform | est=-0.003 \|err\|=0.00 p=0.84 ✅ | est=-0.003 \|err\|=0.00 p=1.00 ✅ | est=-0.003 \|err\|=0.00 p=0.99 ✅ |

### clone

Full-oracle CVaR_α truth = `+0.000` (n_total=10)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=+0.164 \|err\|=0.16 p=0.91 ✅ | est=+0.164 \|err\|=0.16 p=1.00 ✅ | est=+0.164 \|err\|=0.16 p=0.99 ✅ |
| uniform | est=+0.241 \|err\|=0.24 p=0.42 ✅ | est=+0.241 \|err\|=0.24 p=0.86 ✅ | est=+0.241 \|err\|=0.24 p=0.79 ✅ |

### premium

Full-oracle CVaR_α truth = `+0.000` (n_total=10)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.004 \|err\|=0.00 p=0.98 ✅ | est=-0.004 \|err\|=0.00 p=1.00 ✅ | est=-0.004 \|err\|=0.00 p=1.00 ✅ |
| uniform | est=-0.003 \|err\|=0.00 p=0.84 ✅ | est=-0.003 \|err\|=0.00 p=1.00 ✅ | est=-0.003 \|err\|=0.00 p=0.98 ✅ |

### parallel_universe_prompt

Full-oracle CVaR_α truth = `+0.000` (n_total=10)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.007 \|err\|=0.01 p=0.98 ✅ | est=-0.007 \|err\|=0.01 p=1.00 ✅ | est=-0.007 \|err\|=0.01 p=1.00 ✅ |
| uniform | est=-0.003 \|err\|=0.00 p=0.84 ✅ | est=-0.003 \|err\|=0.00 p=1.00 ✅ | est=-0.003 \|err\|=0.00 p=0.99 ✅ |

### unhelpful

Full-oracle CVaR_α truth = `-0.209` (n_total=10)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.015 \|err\|=0.19 p=0.98 ✅ | est=-0.015 \|err\|=0.19 p=0.99 ✅ | est=-0.015 \|err\|=0.19 p=1.00 ✅ |
| uniform | est=-0.003 \|err\|=0.21 p=0.84 ✅ | est=-0.003 \|err\|=0.21 p=1.00 ✅ | est=-0.003 \|err\|=0.21 p=0.99 ✅ |


## Aggregate summary

Across all policies. coverage = 0.5, α = 0.1.

| design | audit | mean \|err\| | reject_rate | n_cells |
|---|---|---:|---:|---:|
| floor_tail | g1_only | 0.074 | 0.00 | 5 |
| floor_tail | g2_only | 0.074 | 0.00 | 5 |
| floor_tail | two_moment | 0.074 | 0.00 | 5 |
| uniform | g1_only | 0.091 | 0.00 | 5 |
| uniform | g2_only | 0.091 | 0.00 | 5 |
| uniform | two_moment | 0.091 | 0.00 | 5 |
