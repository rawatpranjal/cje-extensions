# CVaR-CJE pilot table

- **α** = `0.1`
- **coverage** = `0.25` (uniform oracle slice)
- **seed** = `42`
- **n_total per policy** = `100`
- **truth** = atom-split CVaR_α on full oracle panel
- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)
- **verdict heuristic** = PASS if both \|mean_g1\|, \|mean_g2\| ≤ 0.05; otherwise tag offending moment

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 5 | -0.141 | -0.181 | 0.040 | -0.100 | +0.000 | FLAG_TAIL |
| clone | 26 | -0.169 | -0.272 | 0.103 | -0.023 | +0.005 | PASS |
| premium | 26 | -0.054 | -0.123 | 0.070 | +0.015 | -0.010 | PASS |
| parallel_universe_prompt | 26 | -0.308 | -0.319 | 0.012 | +0.131 | +0.012 | FLAG_TAIL |
| unhelpful | 26 | -0.515 | -0.502 | 0.012 | -0.062 | +0.016 | FLAG_TAIL |

**Notes**:
- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).
- mean_g1 = mean of `1{y_audit ≤ t̂} − α` (tail-mass deviation).
- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).
- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.
