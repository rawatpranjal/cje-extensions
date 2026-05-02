# Multi-seed sweep — plug-in vs augmented Direct CVaR-CJE

n_seeds=10 (start=0)  coverage=0.25  B=200  wall=243.4s

Coverage at 95% nominal (closer to 0.95 = better calibrated). MSE = E[(V̂ - truth)^2]; lower is better. Bias = E[V̂ - truth].

## α = 0.1

| policy | n_au | truth | bias_plug | bias_aug | RMSE_plug | RMSE_aug | MSE_plug | MSE_aug | cov_plug | cov_aug | width_plug | width_aug | uniq t̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 24 | -0.367 | -0.0688 | -0.0681 | 0.1865 | 0.2093 | 0.03479 | 0.04382 | 0.80 | 0.80 | 0.645 | 0.725 | 190 |
| clone | 121 | -0.354 | -0.0873 | -0.0919 | 0.1933 | 0.2012 | 0.03737 | 0.04048 | 0.80 | 0.80 | 0.648 | 0.702 | 192 |
| premium | 121 | -0.264 | -0.1238 | -0.1323 | 0.2151 | 0.2257 | 0.04629 | 0.05096 | 0.80 | 0.80 | 0.612 | 0.670 | 190 |
| parallel | 121 | -0.479 | -0.0836 | -0.0851 | 0.1959 | 0.2072 | 0.03839 | 0.04293 | 0.90 | 0.90 | 0.697 | 0.758 | 194 |
| unhelpful | 121 | -0.492 | -0.2037 | -0.2052 | 0.2927 | 0.2996 | 0.08565 | 0.08973 | 0.80 | 0.80 | 0.722 | 0.785 | 186 |

**Verdicts (this α):**
- base (n_au=24): MSE winner = **plug** (gap 20.6%), coverage closer to 0.95 = **aug**
- clone (n_au=121): MSE winner = **plug** (gap 7.7%), coverage closer to 0.95 = **aug**
- premium (n_au=121): MSE winner = **plug** (gap 9.2%), coverage closer to 0.95 = **aug**
- parallel (n_au=121): MSE winner = **plug** (gap 10.6%), coverage closer to 0.95 = **aug**
- unhelpful (n_au=121): MSE winner = **plug** (gap 4.6%), coverage closer to 0.95 = **aug**

## α = 0.2

| policy | n_au | truth | bias_plug | bias_aug | RMSE_plug | RMSE_aug | MSE_plug | MSE_aug | cov_plug | cov_aug | width_plug | width_aug | uniq t̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 24 | -0.197 | -0.0364 | -0.0378 | 0.1002 | 0.1284 | 0.01004 | 0.01649 | 0.80 | 0.80 | 0.378 | 0.478 | 190 |
| clone | 121 | -0.185 | -0.0528 | -0.0599 | 0.1062 | 0.1153 | 0.01128 | 0.01329 | 0.80 | 0.90 | 0.381 | 0.463 | 190 |
| premium | 121 | -0.101 | -0.0930 | -0.1061 | 0.1327 | 0.1475 | 0.01762 | 0.02175 | 0.90 | 0.90 | 0.362 | 0.433 | 192 |
| parallel | 121 | -0.322 | -0.0297 | -0.0288 | 0.1135 | 0.1270 | 0.01288 | 0.01613 | 0.80 | 0.80 | 0.460 | 0.549 | 195 |
| unhelpful | 121 | -0.363 | -0.1686 | -0.1886 | 0.2457 | 0.2842 | 0.06037 | 0.08077 | 0.90 | 0.90 | 0.562 | 0.651 | 191 |

**Verdicts (this α):**
- base (n_au=24): MSE winner = **plug** (gap 39.1%), coverage closer to 0.95 = **aug**
- clone (n_au=121): MSE winner = **plug** (gap 15.1%), coverage closer to 0.95 = **aug**
- premium (n_au=121): MSE winner = **plug** (gap 19.0%), coverage closer to 0.95 = **aug**
- parallel (n_au=121): MSE winner = **plug** (gap 20.1%), coverage closer to 0.95 = **aug**
- unhelpful (n_au=121): MSE winner = **plug** (gap 25.3%), coverage closer to 0.95 = **aug**

## Hypothesis check

Inspector hypothesis: plug-in wins MSE at small n_audit (base, n_au≈26); augmented wins or ties at larger n_audit (others, n_au≈132).

- base (n_au≈26): plug-in wins MSE in 2/2 α-cells.
- non-base (n_au≈132): plug-in wins MSE in 8/8 cells; aug wins in 0/8.