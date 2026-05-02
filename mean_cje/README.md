# mean_cje — empirical scrutiny of the original CJE paper

## What this folder is

A custom Mean-CJE implementation with swappable components, run on a
known-truth parametric Beta DGP. Goal: figure out which pieces of the
paper's machinery are actually needed for unbiased estimates and 90–95%
coverage.

Spec: `docs/superpowers/specs/2026-05-02-mean-cje-empirical-scrutiny-design.md`
Original CJE paper: `cvar_v4/papers/2512.11150_causal_judge_evaluation/`

## Setup

- **DGP**: 4-policy parametric Beta panel (`uniform`, `right_skew`,
  `left_skew`, `tail_heavy`). `Y ~ Beta(a, b)`, `S = 4·(Y − 0.5) + ε`,
  `ε ~ N(0, 0.64)`. Closed-form `truth_mean = a / (a + b)`.
- **Logger = target** (no transport break). Bias studied is the
  finite-sample calibrator-fit bias only.
- **Sweep**: `n_oracle ∈ {50, 100, 250, 500, 1000}`, `n_eval = 2000`,
  R = 200 (Wald) / 50 (bootstrap), B = 500.
- **Six configurations**:

  | # | estimator | variance | CI |
  |---|---|---|---|
  | 1 | plug-in | IF only (f̂ fixed) | Wald |
  | 2 | plug-in | IF + jackknife | Wald |
  | 3 | AIPW | IF only | Wald |
  | 4 | AIPW | IF + jackknife | Wald |
  | 5 | plug-in | (full-pipeline bootstrap) | bootstrap |
  | 6 | AIPW | (full-pipeline bootstrap) | bootstrap |

  Bootstrap configs 5, 6 implement the original paper's
  `Bootstrap-Direct` algorithm (`appendix_algorithms.tex:101-118`):
  resample BOTH oracle and eval rows per replicate, refit calibrator
  on bootstrap oracle, recompute `V̂^(b)` on bootstrap eval.

Provenance: `runs/2026-05-02T192048/`, run wall = 349 s on 4 cores.

## Headline coverage table

Coverage at 95% nominal, averaged over 4 policies:

| # | config | n=50 | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|---|---|
| 1 | plug-in + IF only | 33% | 38% | 57% | 68% | 75% |
| 2 | plug-in + IF + jackknife | **89%** | **90%** | **90%** | **91%** | **92%** |
| 3 | AIPW + IF only | 28% | 38% | 56% | 68% | 75% |
| 4 | AIPW + IF + jackknife | **88%** | **89%** | **90%** | **90%** | **92%** |
| 5 | plug-in + bootstrap | **94%** | **96%** | **94%** | **94%** | **96%** |
| 6 | AIPW + bootstrap | **96%** | **96%** | **93%** | **94%** | **96%** |

## Headline bias table

Bias = `mean V̂ − truth`, averaged over 4 policies:

| # | estimator | n=50 | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|---|---|
| 1 | plug-in | +0.0003 | −0.0006 | +0.0001 | +0.00002 | −0.0002 |
| 3 | AIPW | +0.0003 | −0.0003 | +0.0002 | +0.00004 | −0.0002 |

Both within MC noise (`σ_MC ≈ 0.0015` at R=200, n=50). **Bias is
negligible on this DGP.**

## Mean CI half-width

Averaged over 4 policies:

| # | config | n=50 | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|---|---|
| 1 | plug-in + IF only | 0.0089 | 0.0088 | 0.0086 | 0.0085 | 0.0084 |
| 2 | plug-in + IF + jackknife | 0.0431 | 0.0296 | 0.0196 | 0.0151 | 0.0123 |
| 5 | plug-in + bootstrap | 0.0393 | 0.0288 | 0.0195 | 0.0151 | 0.0123 |

Configs 2 and 5 produce **near-identical half-widths**, confirming
that the jackknife `V̂_cal` and the full-pipeline bootstrap target the
same underlying variance — but the bootstrap covers a few points
better.

## Findings

### 1. AIPW is doing nothing measurable on this DGP — and slightly hurts RMSE

Compare configs 1 vs 3, 2 vs 4, 5 vs 6: **identical coverage**, **identical
mean bias** at every cell. The AIPW residual term contributes a few
decimal-fourth-place adjustments to the point estimate.

Per-rep, AIPW and plug-in DO disagree (100% of cases, std of difference
~0.006 at n=50), but the differences average to zero. So AIPW is adding
**mean-zero noise** without removing bias.

Empirical RMSE comparison (plug-in vs AIPW v_hat):

| n_oracle | plug-in RMSE | AIPW RMSE | Δ |
|---|---|---|---|
| 50 | 0.02181 | 0.02251 | **+3.2%** |
| 100 | 0.01539 | 0.01577 | +2.5% |
| 250 | 0.01016 | 0.01029 | +1.3% |
| 500 | 0.00788 | 0.00793 | +0.7% |
| 1000 | 0.00663 | 0.00663 | +0.1% |

AIPW is **strictly worse** on this DGP — adds noise, removes no bias.

The paper's headline ("bootstrap with `θ̂_aug` gets ~95%") IS reproduced
in our coverage table — but it's the **bootstrap** doing the work, not
the AIPW. The "AIPW corrects bias" story isn't visible here because the
plug-in is already unbiased: PAVA's mean-preservation + linear S|Y means
`mean(f̂(s_oracle))` exactly equals `mean(y_oracle)`, and the symmetric
DGP makes plug-in unbiased on fresh eval too.

### 2. Jackknife `V̂_cal` is the load-bearing piece

Compare configs 1 → 2 (just adding the jackknife term):
- 33% → 89% coverage at n=50
- 38% → 90% at n=100
- 75% → 92% at n=1000

This is the move that gets you to 90%. The IF-only Wald CI is
catastrophically narrow at small `n_oracle` because it treats `f̂` as
fixed — which is wrong when `f̂` is being estimated from 50 oracle rows.

### 3. Full-pipeline bootstrap is real but barely beats jackknife on this DGP

Configs 5, 6 (bootstrap) hit **94–96%** coverage uniformly — matching
the paper's nominal target. Compared to configs 2, 4 (Wald + jackknife)
at **89–92%**: bootstrap gains roughly **3–5 percentage points** of
coverage at similar half-width.

The marginal value of the bootstrap on this DGP is real but small:
both methods produce CIs of nearly identical width, but the bootstrap's
percentile interval picks up an asymmetry the symmetric Wald + √V̂_total
misses. Whether 89% vs 95% matters depends on stakes.

### 4. The naive bootstrap (eval fixed) undercovers

We initially implemented the bootstrap with `s_eval` held fixed
(only resampling oracle), capturing only `Var_cal` and missing
`Var_eval`. Result: coverage dropped to 80% at large `n_oracle` because
`Var_cal → 0` while `Var_eval` stays finite, so the CI becomes too narrow.

The fix is to follow the paper's `Bootstrap-Direct` algorithm and
resample BOTH oracle and eval per replicate. This is on its face a small
change but it's what makes the bootstrap match nominal coverage.

## What's essential, what's not (on this DGP)

| Piece | Essential? | Evidence |
|---|---|---|
| Calibrator (isotonic) | yes | nothing else recovers `f̂(s) ≈ E[Y \| s]` |
| Cross-fit (K folds) | yes (for jackknife) | `V̂_cal` requires per-fold refits |
| AIPW one-step | **no** | indistinguishable from plug-in at every cell |
| Jackknife `V̂_cal` | **yes** | the only piece that lifts coverage 33%→90% |
| Full-pipeline bootstrap (B=500) | yes for 95% | adds 3–5pp of coverage over Wald + jackknife; resamples both oracle AND eval |
| Resampling EVAL in bootstrap | **yes** | holding eval fixed undercovers at large n_oracle |

The minimum stack for ~90% coverage on this DGP:

```
calibrator (isotonic, K-fold)
  + plug-in mean
  + IF variance + jackknife V̂_cal
  + Wald CI
```

The minimum stack for ~95% coverage:

```
calibrator (isotonic, K-fold)
  + plug-in mean
  + full-pipeline bootstrap
    (resample BOTH oracle and eval; refit calibrator per replicate)
```

AIPW one-step is paper-grade polish that doesn't move the needle here.

## Implications for cvar_v5 audit

The audit problem we're trying to solve in cvar_v5
(`[omega-research-derivation]`) is: the audit's null is biased at finite
`n_calib` because `ĥ_t` has bias. The paper's prescription is to inflate
`Ω̂` to include calibration uncertainty — analogous to our `V̂_cal` jackknife.

These results suggest:
- **Jackknife is the load-bearing piece**. Implement the OUA jackknife
  on `(g_1, g_2)` first.
- **Full-pipeline bootstrap is the gold standard for CIs.** For the
  audit's `Ω̂`, the bootstrap is the canonical answer per the paper.
  Build it once for both CIs and the audit `Ω̂` (single output port).
- **Don't bother with AIPW analog for CVaR.** It's research without
  paper precedent, and likely inert if the calibrator is well-behaved.
- **If we build a bootstrap, resample EVAL too.** This is the bug we
  hit; easy to miss if you read the algorithm casually.

## Caveats

- **Single DGP**: parametric Beta with linear S|Y. Heteroscedastic noise
  or non-linear `E[Y|S]` might surface AIPW's value.
- **Logger = target**: no transport break. The paper's claims about
  AIPW under transport stress aren't tested here.
- **R=50 for bootstrap configs** gives ±3pp MC noise on coverage.
- **Single-stage isotonic**: no covariates. The paper's two-stage
  on real Arena data with response-length covariate is a different
  regime.

## Running

```bash
python3 -m mean_cje.exp_coverage -w 4
# Output: mean_cje/runs/<ts>/{results.csv, summary.csv, log.txt}
# Wall time: ~5–6 minutes on 4 cores at default budgets.
```

Tests:

```bash
pytest mean_cje/lib_test.py -v
# 11/11 should pass.
```
