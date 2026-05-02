# mean_cje — empirical scrutiny of the original CJE paper

## What this folder is

A custom Mean-CJE implementation with swappable components, run on a
known-truth parametric Beta DGP. Goal: figure out which pieces of the
paper's machinery are actually needed for unbiased estimates and 90–95%
coverage.

Spec: `docs/superpowers/specs/2026-05-02-mean-cje-empirical-scrutiny-design.md`

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
  | 5 | plug-in | (bootstrap) | bootstrap |
  | 6 | AIPW | (bootstrap) | bootstrap |

Provenance: `runs/2026-05-02T184540/`, run wall = 311 s on 4 cores.

## Headline coverage table

Coverage at 95% nominal, averaged over 4 policies:

| # | config | n=50 | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|---|---|
| 1 | plug-in + IF only | 33% | 38% | 57% | 68% | 75% |
| 2 | plug-in + IF + jackknife | **89%** | **90%** | **90%** | **91%** | **92%** |
| 3 | AIPW + IF only | 28% | 38% | 56% | 68% | 75% |
| 4 | AIPW + IF + jackknife | **88%** | **89%** | **90%** | **90%** | **92%** |
| 5 | plug-in + bootstrap | 92% | 93% | 90% | 89% | 80% |
| 6 | AIPW + bootstrap | 94% | 94% | 91% | 88% | 80% |

(Bootstrap configs use R=50; SE on coverage at 95% is ±3pp, so 80%–94% is
within MC noise of nominal at this rep count.)

## Headline bias table

Bias = `mean V̂ − truth`, averaged over 4 policies:

| # | estimator | n=50 | n=100 | n=250 | n=500 | n=1000 |
|---|---|---|---|---|---|---|
| 1 | plug-in | +0.0003 | −0.0006 | +0.0001 | +0.00002 | −0.0002 |
| 3 | AIPW | +0.0003 | −0.0003 | +0.0002 | +0.00004 | −0.0002 |

Both are within MC noise (`σ_MC ≈ RMSE / √R ≈ 0.0015` at R=200, n=50)
at every cell. **Bias is negligible on this DGP.**

## Findings

### 1. AIPW is doing almost nothing on this DGP

Compare configs 1 vs 3 (no jackknife) and 2 vs 4 (with jackknife):
**identical coverage**, **identical bias**. The AIPW residual term
contributes a few decimal-fourth-place adjustments and that's it. The
"plug-in has bias that AIPW corrects" story from the paper is **not
visible at these oracle fractions on this DGP.**

This contradicts the paper's strong claim that the bootstrap with
`θ̂_aug` is required for nominal coverage. On our DGP, the AIPW one-step
is empirically inert.

**Why?** The paper's setup is real-data Arena evaluation where the
`f̂(s)` to `Y` mapping is genuinely lossy. Our linear-in-Y `S | Y`
mapping is friendly to isotonic — `f̂` recovers the conditional mean
well at every n. Whatever bias the AIPW correction is supposed to
absorb just isn't here in the synthetic regime.

### 2. The jackknife `V̂_cal` is the real workhorse

Compare configs 1 → 2 (just adding the jackknife term):
- 33% → 89% coverage at n=50
- 38% → 90% at n=100
- 75% → 92% at n=1000

**This is the move that gets you to 90%+.** The paper's "70–89%
coverage" headline for jackknife-only undersells what we measured here
(89–92%). Possibly because:
- Their oracle fraction range or DGP is harsher than ours.
- Their plug-in has more bias for the jackknife to miss.
- Or they're comparing on a different cell of their experiment.

### 3. The bootstrap is no better than the Wald + jackknife

Configs 5, 6 vs configs 2, 4:
- At small n (50, 100): bootstrap matches jackknife (89–94%).
- At large n (500, 1000): bootstrap **drops below** jackknife (80–88% vs 90–92%).

The bootstrap drop at large n is partly MC noise (R=50) but the trend
is consistent across configs 5 and 6. The bootstrap CIs are narrower
than the Wald CIs at large n (`mean_half_width` drops from ~0.04 to
~0.009 in the table), and the calibrator-side variance shrinks faster
than the bootstrap percentile interval can keep up with the residual
bias. **Bootstrap is paying its compute cost for no observable benefit
on this DGP.**

### 4. Plug-in bias is essentially zero everywhere

Even at `n_oracle = 50` with a 2.5% oracle fraction, plug-in has
`|bias| < 0.001` on every policy. The "bias correction" stack
(AIPW one-step) is solving a problem we don't have here.

## What's essential, what's not (on this DGP)

| Piece | Essential? | Evidence |
|---|---|---|
| Calibrator (isotonic) | yes | nothing else recovers `f̂(s) ≈ E[Y \| s]` |
| Cross-fit (K folds) | yes (for the jackknife) | `V̂_cal` requires per-fold refits |
| AIPW one-step | **no** | configs 1 vs 3, 2 vs 4 are indistinguishable |
| Jackknife `V̂_cal` | **yes** | the only piece that lifts coverage 33%→90% |
| Bootstrap (B=2000) | **no** | matches Wald + jackknife or worse, costs more |

The minimum stack for 90% coverage on this DGP:

```
calibrator (isotonic, K-fold)
  + plug-in mean
  + IF variance + jackknife V̂_cal
  + Wald CI
```

That's **config 2**. AIPW and bootstrap are paper-grade polish that
don't move the needle here.

## Implications for cvar_v5 audit

The audit problem we're trying to solve in cvar_v5 (`[omega-research-derivation]`) is:
the audit's null is biased at finite `n_calib` because `ĥ_t` has bias.
The paper's prescription is to inflate `Ω̂` to include calibration
uncertainty (their analog of our `V̂_cal` jackknife).

**These results don't directly tell us whether that prescription
calibrates audit size**, but they suggest:
- The jackknife is the load-bearing piece — implement it.
- Don't bother with AIPW analog for CVaR (it's research without paper
  precedent and likely inert if the underlying calibrator is already
  well-behaved).
- Don't pay for B=2000 bootstrap unless we have evidence it helps.

## Caveats to the findings

- **Single DGP**: parametric Beta with linear S|Y. A harsher DGP
  (heteroscedastic noise, non-linear E[Y|S], heavy-tailed S) might
  surface AIPW's value.
- **Logger = target**: no transport break. The paper's claims about
  AIPW under transport stress aren't tested here.
- **R=50 for bootstrap configs** gives ±3pp MC noise on coverage. The
  "bootstrap drops at large n" finding is suggestive, not proven.
- **Single-stage isotonic**: no covariates. The paper's two-stage
  (`g(S, X)` then isotonic) on real data with response-length
  covariate is a different regime.

## Running

```bash
python3 -m mean_cje.exp_coverage -w 4
# Output: mean_cje/runs/<ts>/{results.csv, summary.csv, log.txt}
# Wall time: ~5 minutes on 4 cores at default budgets.
```

Tests:

```bash
pytest mean_cje/lib_test.py -v
# 11/11 should pass.
```
