# CVaR-CJE pilot table

- **α** = `0.1`
- **coverage** = `0.25` (uniform oracle slice)
- **seed** = `42`
- **n_total per policy** = `500`
- **truth** = atom-split CVaR_α on full oracle panel
- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)
- **verdict heuristic** = PASS if both \|mean_g1\|, \|mean_g2\| ≤ 0.05; otherwise tag offending moment

| policy | n_slice | CVaR_hat | truth | error | audit_only (95% CI) | t̂ | t* | gap | n_oracle_eq | cost× | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 26 | -0.276 | -0.367 | 0.091 | -0.299 [-0.500, +0.000] | -0.108 | -0.103 | -0.004 | 99 | 3.79× | -0.023 | +0.006 | PASS |
| clone | 132 | -0.279 | -0.354 | 0.075 | -0.245 [-0.325, -0.147] | -0.108 | -0.083 | -0.025 | 53 | 0.40× | -0.002 | -0.005 | PASS |
| premium | 132 | -0.233 | -0.264 | 0.031 | -0.223 [-0.342, -0.101] | -0.057 | +0.000 | -0.057 | 95 | 0.71× | +0.006 | -0.003 | PASS |
| parallel_universe_prompt | 132 | -0.348 | -0.479 | 0.132 | -0.475 [-0.628, -0.323] | -0.185 | -0.219 | +0.034 | 102 | 0.77× | +0.044 | +0.013 | PASS |
| unhelpful | 132 | -0.411 | -0.492 | 0.081 | -0.371 [-0.434, -0.308] | -0.236 | -0.286 | +0.049 | 8 | 0.06× | +0.044 | -0.003 | PASS |
| risky | 132 | -0.341 | -0.492 | 0.150 | -0.319 [-0.366, -0.224] | -0.159 | -0.189 | +0.029 | 25 | 0.19× | +0.006 | -0.002 | PASS |

**Notes**:
- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).
- `audit_only` = empirical CVaR_α on Y_audit alone (no calibrator). 95% CI is a 1-D percentile bootstrap of n_audit oracle rows (B=2000). The skeptic baseline.
- `t̂` = saddle-point threshold from the FULL cheap target distribution; `t*` = atom-split α-quantile of full oracle Y. `gap = t̂ − t*` is a mechanism diagnostic.
- `n_oracle_eq` = pure-oracle rows needed to match Direct's variance. `cost× = Var(audit_only) / Var(Direct) = n_oracle_eq / n_audit`. `>1` means Direct is more efficient than just averaging the audit slice; `<1` means Direct's calibrator-resampling variance exceeds the gain. Honest envelope uses Var_cal + Var_audit for Direct.
- mean_g1 = mean of `1{y_audit ≤ t̂} − α` (tail-mass deviation).
- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).
- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.
