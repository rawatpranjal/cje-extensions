# Coverage sweep on existing HealthBench panel

`n_seeds = 20`, oracle coverage = `0.25`, seed_base = `1000`.

Per-variant coverage rate is the fraction of seeds whose 95% CI covers the full-oracle truth.

Wilson 95% interval on the coverage rate (handles small-N).


## α = 0.1

| policy | variant | n | covered | rate | Wilson 95% |
|---|---|---|---|---|---|
| base | cvar_aug_cov_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | cvar_aug_var_total | 20 | 17 | 0.850 | [0.640, 0.948] |
| base | cvar_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| base | cvar_eval_only | 20 | 17 | 0.850 | [0.640, 0.948] |
| base | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | mean_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | mean_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| base | mean_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | mean_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | cvar_aug_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | cvar_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| clone | cvar_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | cvar_eval_only | 20 | 17 | 0.850 | [0.640, 0.948] |
| clone | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| clone | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_eval_only | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| parallel_universe_prompt | cvar_aug_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| parallel_universe_prompt | cvar_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| parallel_universe_prompt | cvar_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| parallel_universe_prompt | cvar_eval_only | 20 | 18 | 0.900 | [0.699, 0.972] |
| parallel_universe_prompt | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| parallel_universe_prompt | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| parallel_universe_prompt | mean_cov_var_total | 20 | 6 | 0.300 | [0.145, 0.519] |
| parallel_universe_prompt | mean_eval_only | 20 | 16 | 0.800 | [0.584, 0.919] |
| parallel_universe_prompt | mean_var_total | 20 | 18 | 0.900 | [0.699, 0.972] |
| premium | cvar_aug_cov_var_total | 20 | 17 | 0.850 | [0.640, 0.948] |
| premium | cvar_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| premium | cvar_cov_var_total | 20 | 17 | 0.850 | [0.640, 0.948] |
| premium | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| premium | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| premium | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| premium | mean_cov_var_total | 20 | 7 | 0.350 | [0.181, 0.567] |
| premium | mean_eval_only | 20 | 5 | 0.250 | [0.112, 0.469] |
| premium | mean_var_total | 20 | 14 | 0.700 | [0.481, 0.855] |
| risky | cvar_aug_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| risky | cvar_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | cvar_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| risky | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| risky | mean_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| risky | mean_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | mean_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| unhelpful | cvar_aug_cov_var_total | 20 | 17 | 0.850 | [0.640, 0.948] |
| unhelpful | cvar_aug_var_total | 20 | 15 | 0.750 | [0.531, 0.888] |
| unhelpful | cvar_cov_var_total | 20 | 17 | 0.850 | [0.640, 0.948] |
| unhelpful | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| unhelpful | cvar_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| unhelpful | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| unhelpful | mean_cov_var_total | 20 | 12 | 0.600 | [0.387, 0.781] |
| unhelpful | mean_eval_only | 20 | 18 | 0.900 | [0.699, 0.972] |
| unhelpful | mean_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |

## α = 0.2

| policy | variant | n | covered | rate | Wilson 95% |
|---|---|---|---|---|---|
| base | cvar_aug_cov_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | cvar_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| base | cvar_cov_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | cvar_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| base | mean_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | mean_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| base | mean_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| base | mean_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | cvar_aug_cov_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| clone | cvar_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | cvar_cov_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| clone | cvar_eval_only | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | cvar_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_cov_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_eval_only | 20 | 20 | 1.000 | [0.839, 1.000] |
| clone | mean_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| parallel_universe_prompt | cvar_aug_cov_var_total | 20 | 14 | 0.700 | [0.481, 0.855] |
| parallel_universe_prompt | cvar_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| parallel_universe_prompt | cvar_cov_var_total | 20 | 15 | 0.750 | [0.531, 0.888] |
| parallel_universe_prompt | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| parallel_universe_prompt | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| parallel_universe_prompt | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| parallel_universe_prompt | mean_cov_var_total | 20 | 6 | 0.300 | [0.145, 0.519] |
| parallel_universe_prompt | mean_eval_only | 20 | 16 | 0.800 | [0.584, 0.919] |
| parallel_universe_prompt | mean_var_total | 20 | 18 | 0.900 | [0.699, 0.972] |
| premium | cvar_aug_cov_var_total | 20 | 13 | 0.650 | [0.433, 0.819] |
| premium | cvar_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| premium | cvar_cov_var_total | 20 | 14 | 0.700 | [0.481, 0.855] |
| premium | cvar_eval_only | 20 | 20 | 1.000 | [0.839, 1.000] |
| premium | cvar_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| premium | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| premium | mean_cov_var_total | 20 | 7 | 0.350 | [0.181, 0.567] |
| premium | mean_eval_only | 20 | 5 | 0.250 | [0.112, 0.469] |
| premium | mean_var_total | 20 | 14 | 0.700 | [0.481, 0.855] |
| risky | cvar_aug_cov_var_total | 20 | 14 | 0.700 | [0.481, 0.855] |
| risky | cvar_aug_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | cvar_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| risky | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | cvar_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| risky | mean_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| risky | mean_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| risky | mean_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| unhelpful | cvar_aug_cov_var_total | 20 | 11 | 0.550 | [0.342, 0.742] |
| unhelpful | cvar_aug_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| unhelpful | cvar_cov_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| unhelpful | cvar_eval_only | 20 | 19 | 0.950 | [0.764, 0.991] |
| unhelpful | cvar_var_total | 20 | 16 | 0.800 | [0.584, 0.919] |
| unhelpful | mean_aug_var_total | 20 | 20 | 1.000 | [0.839, 1.000] |
| unhelpful | mean_cov_var_total | 20 | 12 | 0.600 | [0.387, 0.781] |
| unhelpful | mean_eval_only | 20 | 18 | 0.900 | [0.699, 0.972] |
| unhelpful | mean_var_total | 20 | 19 | 0.950 | [0.764, 0.991] |
