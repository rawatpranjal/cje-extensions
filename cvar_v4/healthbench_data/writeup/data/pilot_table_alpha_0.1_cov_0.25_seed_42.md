# CVaR-CJE pilot table

- **α** = `0.1`
- **coverage** = `0.25` (uniform oracle slice)
- **seed** = `42`
- **n_total per policy** = `500`
- **truth** = atom-split CVaR_α on full oracle panel
- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)
- **verdict heuristic** = PASS if both \|mean_g1\|, \|mean_g2\| ≤ 0.05; otherwise tag offending moment

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 26 | -0.276 | -0.367 | 0.091 | -0.023 | +0.006 | PASS |
| clone | 132 | -0.279 | -0.354 | 0.075 | -0.002 | -0.005 | PASS |
| premium | 132 | -0.233 | -0.264 | 0.031 | +0.006 | -0.003 | PASS |
| parallel_universe_prompt | 132 | -0.348 | -0.479 | 0.132 | +0.044 | +0.013 | PASS |
| unhelpful | 132 | -0.411 | -0.492 | 0.081 | +0.044 | -0.003 | PASS |

**Notes**:
- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).
- mean_g1 = mean of `1{y_audit ≤ t̂} − α` (tail-mass deviation).
- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).
- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.
