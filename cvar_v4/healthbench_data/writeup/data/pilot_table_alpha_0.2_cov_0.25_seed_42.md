# CVaR-CJE pilot table

- **α** = `0.2`
- **coverage** = `0.25` (uniform oracle slice)
- **seed** = `42`
- **n_total per policy** = `500`
- **truth** = atom-split CVaR_α on full oracle panel
- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)
- **verdict heuristic** = PASS if both \|mean_g1\|, \|mean_g2\| ≤ 0.05; otherwise tag offending moment

| policy | n_slice | CVaR_hat | truth | error | audit_only (95% CI) | t̂ | t* | gap | n_oracle_eq | cost× | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 26 | -0.149 | -0.197 | 0.049 | -0.156 [-0.373, +0.042] | +0.008 | +0.000 | +0.008 | 121 | 4.63× | -0.008 | +0.008 | PASS |
| clone | 132 | -0.153 | -0.185 | 0.032 | -0.141 [-0.211, -0.073] | +0.008 | +0.012 | -0.003 | 75 | 0.56× | +0.027 | -0.006 | PASS |
| premium | 132 | -0.116 | -0.101 | 0.016 | -0.104 [-0.190, -0.018] | +0.033 | +0.135 | -0.101 | 119 | 0.89× | -0.018 | -0.006 | PASS |
| parallel_universe_prompt | 132 | -0.237 | -0.322 | 0.086 | -0.329 [-0.435, -0.231] | -0.092 | -0.109 | +0.017 | 118 | 0.89× | +0.042 | +0.018 | PASS |
| unhelpful | 132 | -0.297 | -0.363 | 0.066 | -0.305 [-0.352, -0.264] | -0.142 | -0.200 | +0.058 | 11 | 0.08× | +0.164 | +0.008 | FLAG_TAIL |
| risky | 132 | -0.231 | -0.301 | 0.070 | -0.210 [-0.277, -0.143] | -0.092 | -0.059 | -0.033 | 54 | 0.41× | -0.048 | -0.004 | PASS |

**Notes**:
- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).
- `audit_only` = empirical CVaR_α on Y_audit alone (no calibrator). 95% CI is a 1-D percentile bootstrap of n_audit oracle rows (B=2000). The skeptic baseline.
- `t̂` = saddle-point threshold from the FULL cheap target distribution; `t*` = atom-split α-quantile of full oracle Y. `gap = t̂ − t*` is a mechanism diagnostic.
- `n_oracle_eq` = pure-oracle rows needed to match Direct's variance. `cost× = Var(audit_only) / Var(Direct) = n_oracle_eq / n_audit`. `>1` means Direct is more efficient than just averaging the audit slice; `<1` means Direct's calibrator-resampling variance exceeds the gain. Honest envelope uses Var_cal + Var_audit for Direct.
- mean_g1 = mean of `1{y_audit ≤ t̂} − α` (tail-mass deviation).
- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).
- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.
