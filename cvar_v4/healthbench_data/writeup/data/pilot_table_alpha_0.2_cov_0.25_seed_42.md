# CVaR-CJE pilot table

- **α** = `0.2`
- **coverage** = `0.25` (uniform oracle slice)
- **seed** = `42`
- **n_total per policy** = `500`
- **truth** = atom-split CVaR_α on full oracle panel
- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)
- **verdict heuristic** = PASS if both \|mean_g1\|, \|mean_g2\| ≤ 0.05; otherwise tag offending moment

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 26 | -0.149 | -0.197 | 0.049 | -0.008 | +0.008 | PASS |
| clone | 132 | -0.153 | -0.185 | 0.032 | +0.027 | -0.006 | PASS |
| premium | 132 | -0.116 | -0.101 | 0.016 | -0.018 | -0.006 | PASS |
| parallel_universe_prompt | 132 | -0.237 | -0.322 | 0.086 | +0.042 | +0.018 | PASS |
| unhelpful | 132 | -0.297 | -0.363 | 0.066 | +0.164 | +0.008 | FLAG_TAIL |

**Notes**:
- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).
- mean_g1 = mean of `1{y_audit ≤ t̂} − α` (tail-mass deviation).
- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).
- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.
