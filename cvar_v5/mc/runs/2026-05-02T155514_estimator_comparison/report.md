# estimator_comparison sweep — report

Run dir: `/Users/pranjal/Code/cvar-cje/cvar_v5/mc/runs/2026-05-02T155514_estimator_comparison`

Setting: n_oracle=600, n_eval=1000, R=2, K=5, alphas=[0.05, 0.1, 0.2], all 4 policies. Same calibrator shared across estimators per rep — only the reduction differs.

## Per-cell numbers

| policy | α | estimator | bias | RMSE | rmse_ratio | var_calib | var_calib_ratio |
|---|---|---|---|---|---|---|---|
| left_skew | 0.05 | direct_saddle | -0.00990 | 0.00999 | 1.000 | 1.03e-05 | 1.000 |
| left_skew | 0.05 | plugin_quantile | +0.10729 | 0.11208 | 11.220 | 2.05e-03 | 198.564 |
| left_skew | 0.05 | plugin_ru_dual | +0.10682 | 0.11175 | 11.187 | 2.09e-03 | 203.162 |
| left_skew | 0.10 | direct_saddle | -0.00811 | 0.01007 | 1.000 | 2.13e-05 | 1.000 |
| left_skew | 0.10 | plugin_quantile | +0.08349 | 0.09067 | 9.005 | 1.65e-03 | 77.320 |
| left_skew | 0.10 | plugin_ru_dual | +0.08285 | 0.09000 | 8.938 | 1.61e-03 | 75.753 |
| left_skew | 0.20 | direct_saddle | -0.01121 | 0.01352 | 1.000 | 2.96e-05 | 1.000 |
| left_skew | 0.20 | plugin_quantile | +0.07172 | 0.07347 | 5.434 | 9.46e-05 | 3.193 |
| left_skew | 0.20 | plugin_ru_dual | +0.07146 | 0.07319 | 5.414 | 8.99e-05 | 3.033 |
| right_skew | 0.05 | direct_saddle | +0.00524 | 0.00578 | 1.000 | 1.47e-06 | 1.000 |
| right_skew | 0.05 | plugin_quantile | +0.08753 | 0.08754 | 15.138 | 8.20e-07 | 0.558 |
| right_skew | 0.05 | plugin_ru_dual | +0.08588 | 0.08588 | 14.850 | 5.09e-06 | 3.463 |
| right_skew | 0.10 | direct_saddle | +0.00446 | 0.00595 | 1.000 | 6.01e-06 | 1.000 |
| right_skew | 0.10 | plugin_quantile | +0.07993 | 0.08036 | 13.508 | 9.11e-06 | 1.516 |
| right_skew | 0.10 | plugin_ru_dual | +0.07926 | 0.07974 | 13.403 | 1.54e-05 | 2.556 |
| right_skew | 0.20 | direct_saddle | +0.00151 | 0.00466 | 1.000 | 6.81e-06 | 1.000 |
| right_skew | 0.20 | plugin_quantile | +0.07040 | 0.07138 | 15.333 | 8.12e-05 | 11.915 |
| right_skew | 0.20 | plugin_ru_dual | +0.06944 | 0.07052 | 15.148 | 9.11e-05 | 13.375 |
| tail_heavy | 0.05 | direct_saddle | -0.00205 | 0.00205 | 1.000 | 1.02e-07 | 1.000 |
| tail_heavy | 0.05 | plugin_quantile | +0.04039 | 0.04051 | 19.727 | 1.48e-04 | 1448.789 |
| tail_heavy | 0.05 | plugin_ru_dual | +0.03764 | 0.03809 | 18.548 | 1.36e-04 | 1325.349 |
| tail_heavy | 0.10 | direct_saddle | -0.00074 | 0.00087 | 1.000 | 6.85e-07 | 1.000 |
| tail_heavy | 0.10 | plugin_quantile | +0.04504 | 0.04573 | 52.530 | 1.33e-04 | 194.000 |
| tail_heavy | 0.10 | plugin_ru_dual | +0.04488 | 0.04559 | 52.367 | 9.87e-05 | 144.031 |
| tail_heavy | 0.20 | direct_saddle | +0.00178 | 0.00178 | 1.000 | 2.06e-06 | 1.000 |
| tail_heavy | 0.20 | plugin_quantile | +0.05929 | 0.05929 | 33.364 | 1.10e-05 | 5.365 |
| tail_heavy | 0.20 | plugin_ru_dual | +0.05903 | 0.05903 | 33.218 | 1.15e-05 | 5.591 |
| uniform | 0.05 | direct_saddle | +0.00211 | 0.00263 | 1.000 | 8.52e-08 | 1.000 |
| uniform | 0.05 | plugin_quantile | +0.09237 | 0.09488 | 36.105 | 2.46e-04 | 2883.498 |
| uniform | 0.05 | plugin_ru_dual | +0.09212 | 0.09458 | 35.991 | 2.24e-04 | 2632.078 |
| uniform | 0.10 | direct_saddle | +0.00350 | 0.00705 | 1.000 | 2.23e-05 | 1.000 |
| uniform | 0.10 | plugin_quantile | +0.09558 | 0.09719 | 13.793 | 1.60e-04 | 7.180 |
| uniform | 0.10 | plugin_ru_dual | +0.09522 | 0.09678 | 13.735 | 1.58e-04 | 7.070 |
| uniform | 0.20 | direct_saddle | +0.00730 | 0.01058 | 1.000 | 1.30e-05 | 1.000 |
| uniform | 0.20 | plugin_quantile | +0.08307 | 0.08417 | 7.954 | 1.07e-04 | 8.238 |
| uniform | 0.20 | plugin_ru_dual | +0.08293 | 0.08401 | 7.939 | 1.09e-04 | 8.381 |

## Decision (at α = 0.1)

Per the plan: a plug-in is flagged as a candidate iff
`rmse_ratio < 0.95` AND `var_calib_ratio < 0.95` on >= 2 of 4 policies.

| estimator | policies with both ratios < 0.95 |
|---|---|
| plugin_quantile | (none) (0/4) |
| plugin_ru_dual | (none) (0/4) |

**Verdict**: Direct saddle-point dominates. Close `[plugin-quantile]` and `[plugin-ru-dual]` in TODO.md as 'comparison done; no headroom over Direct on the parametric panel'. Preserve archived implementations for reference.