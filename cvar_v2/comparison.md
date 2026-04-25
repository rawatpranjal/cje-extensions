# Direct Mean vs Direct CVaR-CJE on Chatbot Arena (n≈5000)

Run on the CJE paper's Arena data (20 fold-assignment seeds, 25% oracle, B=500 cluster bootstrap). Direct Mean via `cje-eval==0.2.10` `CalibratedDirectEstimator` (cluster-robust SE, OUA jackknife, augmented). Direct CVaR-CJE via grid-search stop-loss isotonic calibrator (`cvar/workhorse.py`).

## Primary table — CVaR@10%

| Policy | Mean (95% CI) | Mean truth | CVaR (95% CI) | CVaR truth | Transport audit p | Audit reject rate | Rank: Mean → CVaR |
|---|---|---|---|---|---|---|---|
| `parallel_universe_prompt` | 0.772 [0.764, 0.780] | 0.771 | 0.255 [0.230, 0.283] | 0.267 | 0.0364 | 55% | 1 → 1 |
| `premium` | 0.765 [0.758, 0.773] | 0.762 | 0.251 [0.228, 0.278] | 0.257 | 0.00329 | 60% | 2 → 2 |
| `clone` | 0.761 [0.754, 0.769] | 0.762 | 0.249 [0.223, 0.276] | 0.269 | 0.0221 | 55% | 3 → 3 |
| `unhelpful` | 0.421 [0.299, 0.514] | 0.143 | 0.036 [0.019, 0.098] | 0.000 | 0 | 100% | 4 → 4 |

_95% CIs are median per-seed CIs across 20 seeds. Mean CI uses cluster-robust closed-form SE (paper convention); CVaR CI is cluster bootstrap (B=500). Across-seed slice variability is reported separately below._

### Across-seed slice variability (which 25% becomes the oracle slice)

| Policy | Mean across-seed range | CVaR across-seed range |
|---|---|---|
| `parallel_universe_prompt` | [0.764, 0.783] | [0.227, 0.271] |
| `premium` | [0.758, 0.775] | [0.225, 0.269] |
| `clone` | [0.754, 0.771] | [0.221, 0.267] |
| `unhelpful` | [0.229, 0.542] | [0.015, 0.060] |

## Alpha sensitivity — CVaR rank by tail depth

| Policy | CVaR@5% | CVaR@10% | CVaR@20% | Mean |
|---|---|---|---|---|
| `parallel_universe_prompt` | 0.173 | 0.255 | 0.379 | 0.772 |
| `premium` | 0.173 | 0.251 | 0.373 | 0.765 |
| `clone` | 0.171 | 0.249 | 0.369 | 0.761 |
| `unhelpful` | 0.025 | 0.036 | 0.061 | 0.421 |

## Headline

- **CVaR amplifies the signal against `unhelpful`.** On Mean, `unhelpful` is ~0.42 vs ~0.76 for the others (ratio 0.55). On CVaR@10%, it collapses to 0.04 vs ~0.25 (ratio 0.14) — tail risk is what makes this policy unacceptable.
- **Among the 3 good policies, the CVaR estimator ranks `parallel_universe_prompt` first, but the empirical CVaR truth ranks `clone` first.** The estimator tracks the Mean ordering (`parallel_universe_prompt` on top), not the tail ordering — consistent with base-policy calibration failing to transport on the tail task.
- **Transport audit rejects at ≥50% rate for**: `parallel_universe_prompt`, `premium`, `clone`, `unhelpful`. The calibrator fit on the base-policy oracle slice does not transport well to the tail task — so CVaR point estimates carry extra uncertainty beyond the bootstrap CI. This is the audit catching the estimator-vs-truth mismatch.
- **Mean − CVaR gap on the 3 good policies**: `parallel_universe_prompt` = +0.52, `premium` = +0.51, `clone` = +0.51. The gap is roughly constant (~0.51), so the Mean alone cannot rank these policies by tail risk — which is exactly why CVaR is a useful complement.

## Does the transport audit have bite?

Definition of bite: when the audit rejects, are CVaR estimates farther from the empirical truth?

| Audit rejected? | Mean \|err\| vs truth | Median \|err\| | n rows |
|---|---|---|---|
| No (accept) | 0.0112 | 0.0081 | 84 |
| Yes (reject) | 0.0281 | 0.0266 | 156 |

Pearson correlation(audit p-value, |err|) = **-0.363** (negative → audit has bite).

_Notes: Mean estimator is the paper's `direct+cov` (`CalibratedDirectEstimator` with `covariate_names=["response_length"]`, `calibration_mode="auto"`, fresh-draw oracle masked to the calibration slice per `ablations/core/base.py:667-684`). 95% CI is cluster-robust closed-form via t_4. CVaR estimator is `estimate_direct_cvar_isotonic` (`cvar/workhorse.py`). CVaR 95% CI is the cluster bootstrap (B=500) median across 20 seeds. "truth" columns are full-oracle empirical means / lower-tail means on the 5000-row per-policy fresh-draw set._
