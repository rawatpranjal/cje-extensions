# Notation reference — CJE audit comparison

Plain-text notation only. No LaTeX. Use this as a glossary when reading the appendix or chat answers about the audit.

## Indices and slices

| Symbol | Meaning |
|---|---|
| `i` | row index inside the audit slice |
| `j` | row index inside the oracle / calibration slice |
| `k` | row index inside the fresh-draw set under target policy |
| `n_a` | audit slice size |
| `m` | oracle / calibration slice size |
| `n` | fresh-draw size under the target policy |
| `B` | number of bootstrap replicates (paper: 500–2000; cluster by prompt id) |

## Policies

| Symbol | Meaning |
|---|---|
| `π_0` | logger policy — always `base` in the Arena experiment |
| `π'` | target policy — one of `clone`, `premium`, `parallel_universe_prompt`, `unhelpful` |
| `unhelpful` | known-bad stress-test target; never a logger |

## Per-row variables

| Symbol | Meaning | Range |
|---|---|---|
| `x_i` | prompt features | — |
| `a_i` | response under whichever policy generated row `i` | text |
| `s_i` | judge score (cheap, surrogate) | `[0, 1]` |
| `y_i` | oracle label (expensive, ground truth) | `[0, 1]` |

The audit slice has both `s_i` and `y_i` observed; the fresh-draw set has only `s_k`.

## Calibrators (fit on the logger's oracle slice)

| Symbol | Meaning | Used by |
|---|---|---|
| `f(s)` | mean calibrator: predicts `E[y \| s]` | main paper |
| `g_t(s)` | shortfall calibrator at threshold `t`: predicts `E[max(t - y, 0) \| s]` | CVaR appendix |
| `f_(-k)` | mean calibrator refit with oracle fold `k` held out (OUA jackknife) | main paper SE |

Both are isotonic + response-length covariate (two-stage). Cross-fitted across 5 folds (hash of `prompt_id`).

## Tail-level / threshold

| Symbol | Meaning |
|---|---|
| `α` | lower-tail level; headline `α = 0.10` (worst tenth of responses) |
| `q_α(π')` | true α-quantile of `y` under `π'` (estimand-side) |
| `t_hat` | estimated threshold from grid maximisation: `t_hat = argmax_{t in G} { t - Z_bar_t / α }` |
| `G` | grid of 61 candidate thresholds, anchored at the oracle-side α-quantile of `y` |
| `Z_t` | population shortfall mean: `E_{π'}[ max(t - y, 0) ]` |
| `Z_bar_t` | sample estimate: `(1/n) Σ_k g_t(s_k)` over fresh draws under `π'` |

## Estimands and estimators

| Symbol | Meaning |
|---|---|
| `V(π')` | mean policy value: `E_{π'}[y]` |
| `V_hat(π')` | direct estimate: `(1/n) Σ_k f(s_k)` over fresh draws under `π'` |
| `CVaR_α(π')` | population CVaR — formally `sup_t { t - α^(-1) · E_{π'}[max(t - y, 0)] }`; equals `E[y \| y ≤ q_α(π')]` only when `y` has no atom at `q_α(π')` |
| `CVaR_hat(π')` | direct CVaR estimate: `t_hat - Z_bar_{t_hat} / α` |

## Audit moments (per row)

| Symbol | Definition | What it checks |
|---|---|---|
| `r_i` | `y_i - f(s_i)` | mean residual; paper's transport test |
| `g1_i` | `1[y_i ≤ t_hat] - α` | did `t_hat` land at the right quantile (share below ≈ α) |
| `g2_i` | `max(t_hat - y_i, 0) - g_{t_hat}(s_i)` | shortfall residual; threshold-indexed transport |

Note: at `α = 1`, `g1_i` collapses to `0` and `g2_i` collapses to `-r_i` — the CVaR audit reduces to the mean audit.

## Aggregates and test statistics

| Symbol | Definition | Reference dist |
|---|---|---|
| `r_bar` | `(1/n_a) Σ_i r_i` | — |
| `g_bar` | `[ mean(g1), mean(g2) ]` (a 2-vector) | — |
| `SE(r_bar)` | `sqrt( var(r_i) / n_a )` plus OUA jackknife correction for calibration noise | — |
| `t_stat` | `r_bar / SE(r_bar)` | t / N(0,1), 1 df |
| `Σ` | 2×2 covariance of `g_bar` from cluster bootstrap, with `t_hat` re-maximised inside each replicate | — |
| `Σ_inv` | `Σ^(-1)` | — |
| `W` | `g_bar^T · Σ_inv · g_bar` | chi-squared, 2 df |

## Identifying assumptions

| Name | Statement | Where used |
|---|---|---|
| **(T1)** mean transport | `E_{π'}[ y - f(s) ] = 0` | main paper |
| **shortfall transport** | `E_{π'}[ max(t - y, 0) - g_t(s) ] = 0` for `t` near `t_hat` (or stronger: same identity conditional on `s`) | CVaR appendix |
| **no atom at quantile** | `P_{π'}(y = q_α(π')) = 0` | needed for `g1` to have mean zero, and for the CVaR variational form to equal the conditional-mean form |
| **unique interior maximiser** | the variational objective `t - Z_t/α` has a unique argmax in the interior of `G` | needed for the chi-squared limit of `W` |

## Audit decision rules

| Audit | Reject when | Action on reject |
|---|---|---|
| paper (mean) | `\|t_stat\| > 1.96` (5% level), Bonferroni / BH across policies | flag bias, report sign and magnitude of `r_bar`, recalibrate or refuse level claim |
| CVaR | `W > 5.99` (chi-squared 2 df, 5%) | recalibrate or refuse level claim for `CVaR_α(π')` |

## Companion diagnostics (paper, alongside the audit)

| Symbol | Meaning | Gate |
|---|---|---|
| `ESS` | effective sample size fraction (IPS-side) | reported, no hard gate in Direct |
| `TTC` | target-typicality coverage: logger mass on target-typical regions | ≥ 0.70 |
| `CLE` | coverage-limited efficiency: precision floor under low overlap | reported |
| `A_B` | judge-space overlap between logger and target | reported |
| Hill index | tail-thickness diagnostic on importance ratios | reported |
| `Var_cal / Var_total` | calibration uncertainty share of total variance | reported |

## Quick conversion — what to read in the appendix

| In the .tex you see | Read it as |
|---|---|
| `\E_{\pi'}[\cdot]` | average over rows from policy π' |
| `\hat t` | `t_hat` |
| `\bar g` | `g_bar` |
| `\stoploss{t}{Y}` | `max(t - y, 0)` |
| `\Sscore` | `s` (judge score) |
| `q_\alpha(\pprime)` | `q_α(π')` |
| `\1\{Y \le \hat t\}` | `1[y ≤ t_hat]` (indicator: 1 if true, 0 otherwise) |
| `W_n` | `W` |
| `\chi^2_2` | chi-squared with 2 degrees of freedom |
| `(T1)` | mean transport assumption (see above) |
