# HH red-team-attempts — pairwise (mean, CVaR_0.05, CVaR_0.01) with 95% bootstrap CIs

_n=38,961 transcripts; 12 (model_type | num_params) cells; B=1,000 bootstrap reps; Y = `min_harmlessness_score_transcript`_

## Cellwise statistics (sorted by mean ascending — π0 candidates at top)

| policy | n | mean [95% CI] | q_0.05 | CVaR_0.05 [95% CI] | CVaR_0.01 [95% CI] |
|---|---|---|---|---|---|
| `context distillation | 52B` | 17,739 | -0.829 [-0.853, -0.804] | -3.752 | -4.498 [-4.558, -4.432] | -5.530 [-5.633, -5.412] |
| `plain lm | 13B` | 2,587 | -0.772 [-0.837, -0.705] | -3.658 | -4.377 [-4.517, -4.218] | -5.327 [-5.505, -5.093] |
| `plain lm | 2.7B` | 1,151 | -0.748 [-0.839, -0.652] | -3.657 | -4.299 [-4.502, -4.077] | -5.143 [-5.400, -4.833] |
| `context distillation | 13B` | 2,174 | -0.729 [-0.802, -0.656] | -3.708 | -4.310 [-4.452, -4.159] | -5.132 [-5.416, -4.857] |
| `context distillation | 2.7B` | 2,044 | -0.649 [-0.725, -0.580] | -3.922 | -4.488 [-4.622, -4.333] | -5.191 [-5.412, -4.939] |
| `plain lm | 52B` | 1,414 | -0.633 [-0.724, -0.548] | -3.662 | -4.340 [-4.555, -4.110] | -5.384 [-5.668, -4.983] |
| `rlhf | 2.7B` | 2,498 | +0.524 [+0.481, +0.571] | -1.866 | -2.506 [-2.638, -2.357] | -3.370 [-3.626, -3.093] |
| `rlhf | 13B` | 2,292 | +1.426 [+1.398, +1.456] | +0.269 | -0.175 [-0.298, -0.056] | -1.089 [-1.399, -0.706] |
| `rejection sampling | 2.7B` | 1,445 | +1.492 [+1.448, +1.536] | -0.104 | -1.142 [-1.398, -0.840] | -2.399 [-2.714, -1.982] |
| `rejection sampling | 13B` | 1,263 | +1.581 [+1.540, +1.620] | +0.283 | -0.207 [-0.348, -0.053] | -0.864 [-1.122, -0.614] |
| `rlhf | 52B` | 3,081 | +1.703 [+1.671, +1.737] | +0.065 | -0.643 [-0.790, -0.493] | -1.739 [-2.032, -1.412] |
| `rejection sampling | 52B` | 1,273 | +1.723 [+1.681, +1.763] | +0.415 | -0.326 [-0.582, -0.063] | -1.591 [-2.112, -0.977] |

## Same-mean / different-tail pairs

Criterion: 95% mean CIs overlap (means statistically indistinguishable) **AND** 95% CVaR_0.05 CIs do NOT overlap (tails clearly distinguishable). These are the cells where mean-CJE would tie but CVaR-CJE would resolve.

| A | B | Δmean | ΔCVaR_0.05 | ratio | A.mean / A.CVaR / B.mean / B.CVaR |
|---|---|---|---|---|---|
| `rlhf | 13B` | `rejection sampling | 2.7B` | 0.066 | 0.967 | **14.6×** | +1.426 / -0.175 / +1.492 / -1.142 |
