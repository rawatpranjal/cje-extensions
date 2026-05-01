# CVaR-CJE pilot table

- **α** = `0.3`
- **coverage** = `0.25` (uniform oracle slice)
- **seed** = `42`
- **n_total per policy** = `100`
- **truth** = atom-split CVaR_α on full oracle panel
- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)
- **verdict heuristic** = PASS if both \|mean_g1\|, \|mean_g2\| ≤ 0.05; otherwise tag offending moment

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 26 | -0.000 | -0.033 | 0.033 | -0.031 | +0.000 | PASS |
| clone | 26 | -0.009 | -0.090 | 0.081 | +0.085 | +0.019 | FLAG_TAIL |
| premium | 26 | +0.047 | +0.028 | 0.019 | -0.031 | -0.018 | PASS |
| parallel_universe_prompt | 26 | -0.099 | -0.169 | 0.069 | +0.085 | +0.016 | FLAG_TAIL |
| unhelpful | 26 | -0.451 | -0.317 | 0.135 | +0.585 | -0.022 | FLAG_TAIL |

**Notes**:
- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).
- mean_g1 = mean of `1{y_audit ≤ t̂} − α` (tail-mass deviation).
- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).
- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.
