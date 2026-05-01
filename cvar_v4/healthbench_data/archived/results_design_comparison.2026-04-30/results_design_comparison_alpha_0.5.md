# Design × audit comparison

coverage = `0.25`, α = `0.5`, seed = `42`. Estimator: HT-weighted isotonic Direct CVaR-CJE. Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). Full-oracle truth = CVaR_α computed on every row's oracle label.

### base

Full-oracle CVaR_α truth = `+0.076` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=+0.130 \|err\|=0.05 p=0.95 ✅ | est=+0.130 \|err\|=0.05 p=1.00 ✅ | est=+0.130 \|err\|=0.05 p=1.00 ✅ |
| floor_tail_band | est=+0.112 \|err\|=0.04 p=0.62 ✅ | est=+0.112 \|err\|=0.04 p=1.00 ✅ | est=+0.112 \|err\|=0.04 p=0.92 ✅ |
| uniform | est=+0.083 \|err\|=0.01 p=0.69 ✅ | est=+0.083 \|err\|=0.01 p=1.00 ✅ | est=+0.083 \|err\|=0.01 p=0.94 ✅ |

### clone

Full-oracle CVaR_α truth = `+0.019` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=+0.129 \|err\|=0.11 p=1.00 ✅ | est=+0.129 \|err\|=0.11 p=0.72 ✅ | est=+0.129 \|err\|=0.11 p=0.94 ✅ |
| floor_tail_band | est=+0.118 \|err\|=0.10 p=0.74 ✅ | est=+0.118 \|err\|=0.10 p=0.88 ✅ | est=+0.118 \|err\|=0.10 p=0.96 ✅ |
| uniform | est=+0.082 \|err\|=0.06 p=0.69 ✅ | est=+0.082 \|err\|=0.06 p=0.85 ✅ | est=+0.082 \|err\|=0.06 p=0.93 ✅ |

### premium

Full-oracle CVaR_α truth = `+0.154` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=+0.198 \|err\|=0.04 p=0.64 ✅ | est=+0.198 \|err\|=0.04 p=0.97 ✅ | est=+0.198 \|err\|=0.04 p=0.90 ✅ |
| floor_tail_band | est=+0.216 \|err\|=0.06 p=0.26 ✅ | est=+0.216 \|err\|=0.06 p=0.84 ✅ | est=+0.216 \|err\|=0.06 p=0.68 ✅ |
| uniform | est=+0.149 \|err\|=0.01 p=0.56 ✅ | est=+0.149 \|err\|=0.01 p=0.90 ✅ | est=+0.149 \|err\|=0.01 p=0.87 ✅ |

### parallel_universe_prompt

Full-oracle CVaR_α truth = `-0.094` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.024 \|err\|=0.07 p=0.69 ✅ | est=-0.024 \|err\|=0.07 p=0.90 ✅ | est=-0.024 \|err\|=0.07 p=0.94 ✅ |
| floor_tail_band | est=-0.020 \|err\|=0.07 p=0.68 ✅ | est=-0.020 \|err\|=0.07 p=0.92 ✅ | est=-0.020 \|err\|=0.07 p=0.95 ✅ |
| uniform | est=-0.019 \|err\|=0.07 p=1.00 ✅ | est=-0.019 \|err\|=0.07 p=0.85 ✅ | est=-0.019 \|err\|=0.07 p=0.98 ✅ |

### unhelpful

Full-oracle CVaR_α truth = `-0.233` (n_total=100)

| design \ audit | g1_only | g2_only | two_moment |
|---|---|---|---|
| floor_tail | est=-0.198 \|err\|=0.04 p=0.63 ✅ | est=-0.198 \|err\|=0.04 p=0.93 ✅ | est=-0.198 \|err\|=0.04 p=0.93 ✅ |
| floor_tail_band | est=-0.086 \|err\|=0.15 p=0.02 🔥 | est=-0.086 \|err\|=0.15 p=0.54 ✅ | est=-0.086 \|err\|=0.15 p=0.18 ✅ |
| uniform | est=-0.271 \|err\|=0.04 p=0.12 ✅ | est=-0.271 \|err\|=0.04 p=0.91 ✅ | est=-0.271 \|err\|=0.04 p=0.47 ✅ |


## Aggregate summary

Across all policies. coverage = 0.25, α = 0.5.

| design | audit | mean \|err\| | reject_rate | n_cells |
|---|---|---:|---:|---:|
| floor_tail | g1_only | 0.063 | 0.00 | 5 |
| floor_tail | g2_only | 0.063 | 0.00 | 5 |
| floor_tail | two_moment | 0.063 | 0.00 | 5 |
| floor_tail_band | g1_only | 0.084 | 0.20 | 5 |
| floor_tail_band | g2_only | 0.084 | 0.00 | 5 |
| floor_tail_band | two_moment | 0.084 | 0.00 | 5 |
| uniform | g1_only | 0.038 | 0.00 | 5 |
| uniform | g2_only | 0.038 | 0.00 | 5 |
| uniform | two_moment | 0.038 | 0.00 | 5 |
