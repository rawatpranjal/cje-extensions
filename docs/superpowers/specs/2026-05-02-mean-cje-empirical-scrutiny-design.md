# mean_cje — empirical scrutiny of the paper's machinery

**Date:** 2026-05-02
**Owner:** Pranjal
**Status:** design (approved scope)

## Context

The original CJE paper (Mean-CJE, arXiv 2512.11150) prescribes a stack:
isotonic calibrator + cross-fit + AIPW one-step (`θ̂_aug`) + jackknife `V̂_cal`
+ cluster bootstrap with B=2000. They report coverage 0% (naive) → 70–89%
(jackknife only) → ~95% (full stack).

We do not know which of these claims survives on a controlled DGP we
understand. Reading the paper does not tell us which pieces are
*structurally* required vs which are *empirically* tuned to their setup.
Before adopting any of this machinery for cvar_v5, we want to **measure
empirically which pieces are required to hit ~95% coverage and to remove
bias**, on a known-truth DGP we can dial.

Out of scope: audit size calibration. That's a separate question we'll
return to once we trust the inference layer.

## Goal

Build a minimal Mean-CJE implementation with **swappable** components, run
ablations on a known-truth DGP, and answer two concrete questions:

1. **What's the minimum set of pieces needed to make `bias(V̂) ≈ 0`?**
2. **What's the minimum set of pieces needed to hit 90–95% CI coverage?**

A piece is "essential" if dropping it pushes either metric off-target by
>5 percentage points (coverage) or by >2σ_MC (bias). Otherwise it's
optional.

## DGP

Reuse the parametric Beta panel from `cvar_v5/mc/dgp.py`:

```
Y ~ Beta(a_p, b_p)                              Y ∈ [0, 1]
S | Y = scale_p · (Y − 0.5) + ε,    ε ~ N(0, σ_p²)
```

Four policies (uniform, right_skew, left_skew, tail_heavy) with closed-form
`truth_mean = a / (a + b)`. **No transport break, no logger ≠ target
distinction** — the bias we're studying comes from the calibrator's
finite-sample fit at small oracle slices, which is what `θ̂_aug` is meant
to correct.

## Architecture

```
mean_cje/
├── CLAUDE.md                  scope rules (parallel to cvar_v5)
├── lib/
│   ├── __init__.py
│   ├── calibrator.py          isotonic on S, cross-fit (K folds)
│   ├── estimators.py          plug_in_mean(),  aipw_one_step()
│   └── variance.py            var_eval_if(),   var_cal_jackknife(),
│                              wald_ci(),       bootstrap_ci()
├── lib_test.py                code tests for the four library files
├── exp_coverage.py            ablation runner — writes runs/<ts>/
├── runs/                      per-experiment outputs (untracked)
└── README.md                  findings: which pieces matter
```

DGP imported from `cvar_v5.mc.dgp` (don't duplicate). The `mean_cje/`
folder lives parallel to `cvar_v5/`, doesn't depend on `cvar_v5/cvar_cje/`.

## Components (each independently swappable)

| Component | Variants |
|---|---|
| **Point estimator** | `plug_in`, `aipw_one_step` |
| **Variance estimator** | `if_only` (treats f̂ as fixed), `if_plus_jackknife` |
| **CI** | `wald` (uses variance estimate), `bootstrap` (cluster, B configurable) |

A *configuration* is a tuple `(estimator, variance, ci)`. The six
configurations we run are:

| # | estimator | variance | CI | what it is |
|---|---|---|---|---|
| 1 | plug-in | IF only (f̂ fixed) | Wald | naive baseline |
| 2 | plug-in | IF + jackknife | Wald | jackknife-only |
| 3 | AIPW | IF only | Wald | AIPW + naive variance |
| 4 | AIPW | IF + jackknife | Wald | AIPW + jackknife (paper's intermediate) |
| 5 | plug-in | — | bootstrap | bootstrap on plug-in |
| 6 | AIPW | — | bootstrap | **paper's gold standard** |

For Wald configurations: `V̂_total = V̂_eval` (config 1, 3) or
`V̂_eval + V̂_cal` (config 2, 4).
Bootstrap configs (5, 6) get their CI directly from percentiles —
no Wald variance assembly.

Coverage and bias are reported per configuration.

### Math contracts (per file)

**`calibrator.py`**

```
Monotone isotonic on S, cross-fit:
    f̂      :=  IsotonicRegression(increasing=True).fit(s_oracle, y_oracle)
    f̂^(−k) :=  same, fit on oracle \ fold_k
    f̂^(−i) :=  fold-out prediction at oracle row i (cross-fit)
```

**`estimators.py`**

```
plug_in_mean(s_eval, calibrator):
    return mean_i  f̂(s_eval_i)

aipw_one_step(s_eval, oracle_slice, calibrator):
    return mean_i f̂(s_eval_i)
         + mean_{j ∈ L} ( y_j − f̂^(−j)(s_oracle_j) )
```

**`variance.py`**

```
var_eval_if(s_eval, calibrator):
    φ_i := f̂(s_eval_i) − V̂
    return (1/n²) · Σ_i φ_i²

var_cal_jackknife(s_eval, oracle, calibrator_factory, K):
    For k = 1..K:  V̂^(−k) := estimator using f̂^(−k)
    V̄ := mean_k V̂^(−k)
    return ((K−1)/K) · Σ_k (V̂^(−k) − V̄)²

wald_ci(V̂, V̂_total, level=0.95):
    return V̂ ± 1.96 · √V̂_total

bootstrap_ci(samples, level=0.95):
    return percentile interval
```

`bootstrap_ci` is paired with a separate driver loop that resamples and
refits — kept in `exp_coverage.py`, not in the library, since it's
estimator-specific.

## Experiment

`exp_coverage.py` sweeps:

```
policy        ∈ {uniform, right_skew, left_skew, tail_heavy}
n_oracle      ∈ {50, 100, 250, 500, 1000}        (small → large)
n_eval         = 2000                              (fixed; eval-side variance small)
config        ∈ {1, 2, 3, 4, 5, 6}  (the six rows above)
R              = 200 (Wald configs) / 50 (bootstrap configs) reps per cell
B              = 500 bootstrap replicates (configs 5, 6)
```

Bootstrap reps cost B× a calibrator refit each; we cut R to 50 to keep
total wall-time tractable. Headline coverage interval at R=50 has half-width
~7 percentage points (1.96·√(.95·.05/50)·100), enough to distinguish
70% from 90% from 95%.

For each cell, report:

```
bias       =  mean_r V̂_r  −  truth_mean(p)
RMSE       =  √( mean_r (V̂_r − truth)² )
coverage   =  fraction of reps with truth ∈ CI_95%
mean_width =  mean CI half-width (for diagnostic)
```

Output to `runs/<ts>/results.csv`. A small `report.py` (or just the
notebook in `runs/<ts>/`) emits the headline table.

## Decision rule (what counts as a finding)

For each (config, n_oracle) cell, mark:

- **bias-OK** if `|bias| < 3 · σ_MC = 3 · std(V̂_r) / √R`
- **coverage-OK** if `0.90 ≤ coverage ≤ 0.97`

The output table answers two ablations:

1. Does **plug-in alone** ever achieve bias-OK at small oracle fractions?
   (We expect no — that's why AIPW exists.)
2. Does **AIPW + IF-only variance** achieve coverage-OK?
   (We expect ~70–89% per the paper. If our number is much different we
    want to know why.)

A piece is essential iff removing it (with all other pieces held) flips a
cell from OK to fail in ≥ 2 of 4 policies.

## Verification

1. `pytest mean_cje/lib_test.py` passes — all components have code-level
   identity / range / monotonicity tests.
2. `python -m mean_cje.exp_coverage` runs in < 90 min on 4 cores at the
   default settings (Wald: 4 configs × 4 policies × 5 oracles × R=200 = 16k
   reps, each ~50ms = ~14 min single-thread; Bootstrap: 2 configs × 4 ×
   5 × R=50 × B=500 = 100k bootstrap fits, each ~50ms = ~70 min on 4 cores).
3. `runs/<ts>/results.csv` contains all cells, no NaNs.
4. The headline table can be summarized in one paragraph: "naive CIs cover
   X%; jackknife alone covers Y%; AIPW + jackknife covers Z%; the paper
   reports 0/70–89/95%."

## Out of scope (do not implement)

- Two-stage calibrator (no covariates in this DGP).
- Weight stabilization (Direct mode only).
- TTC gate (IPS-only diagnostic).
- IF stacking (multi-estimator ensemble).
- Audit size calibration (separate question).
- Real-data adapter (Arena / HealthBench).
- AIPW for CVaR (no paper-derived form; that's research).

If any of these become relevant, they get their own spec.

## Files to create

| Path | Purpose |
|---|---|
| `mean_cje/CLAUDE.md` | scope rules (no IPS/DR, no audit, etc.) |
| `mean_cje/lib/__init__.py` | empty |
| `mean_cje/lib/calibrator.py` | isotonic + cross-fit |
| `mean_cje/lib/estimators.py` | plug-in, AIPW |
| `mean_cje/lib/variance.py` | IF, jackknife, Wald CI, percentile CI |
| `mean_cje/lib_test.py` | code tests for the library |
| `mean_cje/exp_coverage.py` | ablation runner CLI |
| `mean_cje/README.md` | findings (filled in after the run) |

## What we expect to learn (predictions to verify)

1. Plug-in V̂ has visible bias at small `n_oracle` (10–50). AIPW reduces it
   to MC-noise level.
2. Naive Wald CIs (treating f̂ as fixed) systematically undercover at
   small `n_oracle`.
3. Adding `V̂_cal` jackknife to the Wald CI lifts coverage to 70–89%.
4. Bootstrap on `θ̂_aug` lifts it to ~95%.
5. At `n_oracle = 1000+`, all variants converge to similar coverage —
   calibrator is essentially flat, so there's nothing for the corrections
   to fix.

If any of these predictions fails, that's a finding worth chasing.
