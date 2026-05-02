# estimator_comparison sweep — report

Run dir: `/Users/pranjal/Code/cvar-cje/cvar_v5/mc/runs/2026-05-02T155532_estimator_comparison`

Setting: n_oracle=600, n_eval=1000, R=20, K=5, alphas=[0.05, 0.1, 0.2], all 4 policies. Same calibrator shared across estimators per rep — only the reduction differs.

## Per-cell numbers

| policy | α | estimator | bias | RMSE | rmse_ratio | var_calib | var_calib_ratio |
|---|---|---|---|---|---|---|---|
| left_skew | 0.05 | direct_saddle | +0.00400 | 0.01611 | 1.000 | 2.11e-04 | 1.000 |
| left_skew | 0.05 | plugin_quantile | +0.11589 | 0.11849 | 7.353 | 5.98e-04 | 2.829 |
| left_skew | 0.05 | plugin_ru_dual | +0.11456 | 0.11718 | 7.272 | 5.83e-04 | 2.758 |
| left_skew | 0.10 | direct_saddle | +0.00140 | 0.01222 | 1.000 | 1.21e-04 | 1.000 |
| left_skew | 0.10 | plugin_quantile | +0.09952 | 0.10163 | 8.316 | 2.94e-04 | 2.423 |
| left_skew | 0.10 | plugin_ru_dual | +0.09874 | 0.10086 | 8.252 | 2.99e-04 | 2.470 |
| left_skew | 0.20 | direct_saddle | -0.00061 | 0.01077 | 1.000 | 7.75e-05 | 1.000 |
| left_skew | 0.20 | plugin_quantile | +0.08331 | 0.08484 | 7.881 | 1.90e-04 | 2.455 |
| left_skew | 0.20 | plugin_ru_dual | +0.08268 | 0.08417 | 7.819 | 1.83e-04 | 2.364 |
| right_skew | 0.05 | direct_saddle | -0.00146 | 0.00549 | 1.000 | 2.85e-05 | 1.000 |
| right_skew | 0.05 | plugin_quantile | +0.08505 | 0.08555 | 15.571 | 9.79e-05 | 3.437 |
| right_skew | 0.05 | plugin_ru_dual | +0.08338 | 0.08384 | 15.260 | 9.31e-05 | 3.266 |
| right_skew | 0.10 | direct_saddle | -0.00135 | 0.00594 | 1.000 | 3.20e-05 | 1.000 |
| right_skew | 0.10 | plugin_quantile | +0.07994 | 0.08038 | 13.538 | 6.20e-05 | 1.938 |
| right_skew | 0.10 | plugin_ru_dual | +0.07896 | 0.07945 | 13.382 | 6.95e-05 | 2.169 |
| right_skew | 0.20 | direct_saddle | -0.00207 | 0.00593 | 1.000 | 2.85e-05 | 1.000 |
| right_skew | 0.20 | plugin_quantile | +0.07172 | 0.07217 | 12.166 | 4.97e-05 | 1.743 |
| right_skew | 0.20 | plugin_ru_dual | +0.07085 | 0.07129 | 12.017 | 4.50e-05 | 1.578 |
| tail_heavy | 0.05 | direct_saddle | -0.00165 | 0.00185 | 1.000 | 7.02e-07 | 1.000 |
| tail_heavy | 0.05 | plugin_quantile | +0.04939 | 0.05122 | 27.718 | 1.86e-04 | 265.217 |
| tail_heavy | 0.05 | plugin_ru_dual | +0.04775 | 0.04961 | 26.847 | 1.74e-04 | 247.522 |
| tail_heavy | 0.10 | direct_saddle | +0.00032 | 0.00210 | 1.000 | 3.24e-06 | 1.000 |
| tail_heavy | 0.10 | plugin_quantile | +0.05784 | 0.05937 | 28.323 | 1.66e-04 | 51.077 |
| tail_heavy | 0.10 | plugin_ru_dual | +0.05722 | 0.05878 | 28.042 | 1.63e-04 | 50.340 |
| tail_heavy | 0.20 | direct_saddle | +0.00120 | 0.00536 | 1.000 | 1.55e-05 | 1.000 |
| tail_heavy | 0.20 | plugin_quantile | +0.06022 | 0.06112 | 11.403 | 1.03e-04 | 6.672 |
| tail_heavy | 0.20 | plugin_ru_dual | +0.05982 | 0.06073 | 11.330 | 1.03e-04 | 6.670 |
| uniform | 0.05 | direct_saddle | -0.00110 | 0.00670 | 1.000 | 4.15e-05 | 1.000 |
| uniform | 0.05 | plugin_quantile | +0.06907 | 0.07205 | 10.759 | 4.18e-04 | 10.078 |
| uniform | 0.05 | plugin_ru_dual | +0.06790 | 0.07095 | 10.596 | 4.23e-04 | 10.185 |
| uniform | 0.10 | direct_saddle | -0.00045 | 0.00744 | 1.000 | 4.11e-05 | 1.000 |
| uniform | 0.10 | plugin_quantile | +0.07473 | 0.07643 | 10.269 | 2.40e-04 | 5.834 |
| uniform | 0.10 | plugin_ru_dual | +0.07408 | 0.07576 | 10.178 | 2.39e-04 | 5.801 |
| uniform | 0.20 | direct_saddle | +0.00084 | 0.00906 | 1.000 | 4.51e-05 | 1.000 |
| uniform | 0.20 | plugin_quantile | +0.06933 | 0.07071 | 7.806 | 1.67e-04 | 3.711 |
| uniform | 0.20 | plugin_ru_dual | +0.06877 | 0.07013 | 7.743 | 1.62e-04 | 3.594 |

## Decision (at α = 0.1)

Per the plan: a plug-in is flagged as a candidate iff
`rmse_ratio < 0.95` AND `var_calib_ratio < 0.95` on >= 2 of 4 policies.

| estimator | policies with both ratios < 0.95 |
|---|---|
| plugin_quantile | (none) (0/4) |
| plugin_ru_dual | (none) (0/4) |

**Verdict**: Direct saddle-point dominates. Close `[plugin-quantile]` and `[plugin-ru-dual]` in TODO.md as 'comparison done; no headroom over Direct on the parametric panel'. Preserve archived implementations for reference.