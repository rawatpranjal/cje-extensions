# Math notes for the Direct CJE estimators

Companion to `_estimator.py`. Four sections: (0) a plain-English overview
of the work, (1) why the stop-loss term is clipped at zero, (2) how
Mean-CJE and CVaR-CJE differ as inference machines, (3) the variance
taxonomy — what each SE in the codebase actually measures, and (4) the
α = 1 collapse and its regression tests.

## 0. Plain-English overview

The setup. We have five LLM policies (different prompts, different sampling
configs) and we want to grade their outputs on a benchmark of medical
questions. A perfect grader is expensive (a powerful model called per
response, per criterion). A cheap grader is fast but a bit lenient. We can
afford to run the perfect grader on only ~25% of the panel.

The trick. Use the cheap grader's score `S` everywhere. Train a
**translator** `f̂(s) ≈ E[Y | S=s]` from cheap to perfect on the 25% slice
where we have both. Apply the translator to the other 75% to estimate
each policy's true score under the perfect grader. This is what
"calibration" means here.

Two estimands. *Average grade* (Mean-CJE) and *worst-tenth-percentile
grade* (CVaR at α = 0.10). They tell different stories: a policy can have
a great average and a terrible tail — that's a deployment risk only the
tail estimand catches.

Two questions, not one. For each estimate, we ask both:
1. **Is the translator trustworthy on this policy?** — the **audit**. We
   compare predictions to actual oracle labels on a held-out part of the
   slice and check that residuals are zero on average.
2. **How precise is the estimate?** — the **uncertainty**. We measure how
   much the estimate would wobble under a different draw of the 25% slice.

Different audits and different uncertainties for the two estimands. The
α = 1 case (CVaR over the worst 100% — i.e. the average) is where these
two estimators must give exactly the same answer, by math. We use that
identity as a regression test on the entire inference stack.

## 1. Why `max(t − Y, 0)`

CVaR-CJE uses the Rockafellar–Uryasev saddle-point form (lower-tail variant):

$$
\mathrm{CVaR}_{\alpha}(Y) \;=\; \sup_{t}\Big[\,t \;-\; \tfrac{1}{\alpha}\,\mathbb{E}\bigl[(t-Y)_+\bigr]\,\Big].
$$

The **clip at zero** in `(t − Y)_+ = max(t − Y, 0)` is the entire point of
the construction. Without it,

$$
t - \mathbb{E}[t - Y] \;=\; t - (t - \mathbb{E}[Y]) \;=\; \mathbb{E}[Y],
$$

which is constant in t — the supremum is trivially $\mathbb{E}[Y]$ and t
carries no information about the threshold. The clipping introduces a
kink at $t = Y$. Then

$$
\mathbb{E}[(t - Y)_+] \;=\; F_Y(t)\cdot\bigl(t - \mathbb{E}[Y\mid Y\le t]\bigr),
$$

so the objective is concave in t with derivative

$$
\frac{\partial}{\partial t}\Bigl[t - \tfrac{1}{\alpha}\,\mathbb{E}[(t-Y)_+]\Bigr]
= 1 - \tfrac{1}{\alpha}\,F_Y(t).
$$

The FOC $F_Y(\hat t) = \alpha$ recovers the α-quantile $\hat t = \mathrm{VaR}_\alpha$, and
plugging it back yields the mean below threshold:

$$
\hat t - \tfrac{1}{\alpha}\,\mathbb{E}[(\hat t - Y)_+] \;=\; \mathbb{E}[Y \mid Y \le \hat t] \;=\; \mathrm{CVaR}_\alpha.
$$

At α = 1 the FOC is $F_Y(\hat t) = 1$, satisfied by any $\hat t \ge Y_{\max}$. There the
clip is non-binding ($t - Y \ge 0$ for every $Y$), so $\mathbb{E}[(t - Y)_+] = t - \mathbb{E}[Y]$,
and the objective collapses to $\mathbb{E}[Y]$ exactly. **This is the
algebraic root of the Mean-CJE / CVaR-CJE identity at α = 1.**

The Direct estimator implements the same supremum on a calibrated plug-in:

$$
\widehat{\mathrm{CVaR}}_\alpha(\pi') \;=\; \max_{t \in \mathcal{T}}\Big[\,t \;-\; \tfrac{1}{\alpha}\cdot \tfrac{1}{M} \sum_{j=1}^{M} \hat g_t(s_{\mathrm{eval},j})\,\Big],
$$

where $\hat g_t(s) \approx \mathbb{E}_{\pi_0}[(t - Y)_+ \mid S=s]$ is fit by HT-weighted
isotonic regression on the logger slice. The grid $\mathcal{T}$ must cover the
optimum; `make_t_grid` extends the upper bound past `max(y_train)` to keep
the α = 1 identity exact (see commit history).

## 2. Mean-CJE vs CVaR-CJE — the critical differences

|  | Mean-CJE | CVaR-CJE |
|---|---|---|
| **Estimand** | $\mathbb{E}_{\pi'}[Y]$ | $\mathrm{CVaR}_\alpha(Y) = \mathbb{E}_{\pi'}[Y \mid Y \le \mathrm{VaR}_\alpha]$ |
| **Calibrator** | $\hat f(s)\approx \mathbb{E}[Y\mid S]$ — `fit_isotonic_mean` (increasing) | $\hat g_t(s) \approx \mathbb{E}[(t-Y)_+\mid S]$ — `fit_isotonic_tail_loss` (decreasing) |
| **Threshold** | none | $\hat t$ from grid argmax of saddle-point objective |
| **Audit moment(s)** | one: $\varepsilon = Y - \hat f(S)$ | two: $g_1 = \mathbf{1}\{Y \le \hat t\} - \alpha$; $g_2 = (\hat t - Y)_+ - \hat g_{\hat t}(S)$ |
| **Audit decision** | one-sample t-test on $\varepsilon$ — `mean_transport_audit` | heuristic $\lvert\bar g_1\rvert, \lvert\bar g_2\rvert \le 0.05$ in production; bootstrap-Σ̂ Wald (2-df χ²) in `two_moment_wald_audit_xf` |
| **Bootstrap CI** | `bootstrap_mean_ci` (paired bootstrap on train; eval fixed) | `bootstrap_cvar_ci` (same design, additionally re-maximizes $\hat t$ per rep) |
| **Var_cal** | `jackknife_var_cal_mean` (K-fold delete-one-group on oracle slice) | `jackknife_var_cal` (same K-fold; refits calibrator + $\hat t$) |
| **Argmax variance** | none | non-zero — captured by re-maximizing $\hat t$ per bootstrap rep |
| **Sample-size scaling** | $\mathrm{Var} \propto 1/n$ | tail-effective: $\mathrm{Var} \propto 1/(\alpha n)$ in the slice |
| **Refusal semantics** | calibrator's average prediction is biased on $\pi'$ | tail mass at $\hat t$ wrong (g1) **or** stop-loss surface doesn't transport (g2) |
| **Independence of failures** | a CVaR pass does not certify the mean | a mean pass does not certify the CVaR |

The two audits are **not nested**. We have observed both pass-mean / fail-CVaR
and pass-CVaR / fail-mean cells in the n = 500 panel; gate each estimand by its
own audit (see `CLAUDE.md` four-cell matrix).

## 3. Variance taxonomy — what every SE in the codebase actually measures

There are two **independent** sources of uncertainty in a Direct CJE
report, and we have separate code paths to estimate each:

| family | answers the question | depends on | method |
|---|---|---|---|
| **estimator variance** | how much would V̂ wobble if we had drawn a different 25% slice for calibration? | calibrator-training data $(s_{\text{train}}, y_{\text{train}})$ | bootstrap or jackknife (varies the slice, refits the calibrator each time) |
| **audit variance** | given the calibrator, is the residual mean on the held-out audit slice statistically distinguishable from zero? | the audit slice $(s_{\text{audit}}, y_{\text{audit}})$ | analytical SE on residuals (t-test) **or** paired-bootstrap of the audit slice |

The two are orthogonal: a tight estimator CI says "we are confident in
this point estimate"; a clean audit says "the calibrator transports to
this policy". A policy can be precisely measured but unfairly graded
(narrow CI + audit fail), or noisily measured but fairly graded (wide CI
+ audit pass).

### Mean-CJE variance estimators

| function (file:line) | family | what it captures |
|---|---|---|
| `bootstrap_mean_ci` (`_estimator.py`) | estimator | resamples calibrator-training rows iid B times, refits f̂ each rep, percentile CI on `mean f̂(s_eval)` |
| `jackknife_var_cal_mean` | estimator | K systematic delete-one-group folds on the same data; alternative variance estimator targeting the same quantity |
| `mean_transport_audit` t-test | audit | analytical SE = `sd(ε) / √n_audit` where `ε = Y − f̂(S)` on the audit slice; calibrator treated as fixed |
| (paired audit bootstrap, inline) | audit | resamples audit rows with replacement, B reps, percentile SE on the residual mean |

`bootstrap_mean_ci` and `jackknife_var_cal_mean` should agree to within
~10–20 % at our slice sizes; small disagreements come from the iid-vs-K-fold
sampling distinction and from the smoothness penalty isotonic regression
imposes (flat regions don't move under data perturbation). The analytical
t-test SE and the paired-audit-bootstrap SE should agree to within ~1–2 %
(`B ≥ 500`, `n_audit ≥ 50`); the regression test
`test_prodalpha_audit_se_analytical_vs_bootstrap` locks the CVaR g2 analog
of this match at α = 0.10.

### CVaR-CJE variance estimators

| function (file:line) | family | what it captures |
|---|---|---|
| `bootstrap_cvar_ci` | estimator | resamples train, refits calibrator, **re-maximizes t̂** each rep — captures both calibrator and argmax variance |
| `jackknife_var_cal` | estimator | K-fold delete-one-group; refits calibrator and re-optimizes t̂ on each remaining fold |
| `cvar_audit_analytical_se` | audit | fixed-t̂, fixed-calibrator analytical SEs for `g1` and `g2` plus the off-diagonal cov; the closest analog to the mean t-test |
| (paired audit bootstrap, inline) | audit | resamples audit rows with t̂ fixed; gives the same SE as the analytical helper up to MC noise |
| `two_moment_wald_audit_xf` | audit | bootstrap-Σ̂ Wald with t̂ **re-maximized** per rep — adds argmax variance the analytical SE doesn't see |

The argmax wedge is the one place CVaR variance is structurally larger
than mean variance: re-maximizing t̂ inside each bootstrap rep injects
extra variation that the fixed-t̂ analytical formula doesn't capture. On
the n=500 panel at α=0.10 this inflates the production audit SE for g2
by ~18 % relative to the fixed-t̂ analytical SE; at α=1 the inflation is
~74 % (because the saddle-point objective is flat over a larger region
when α=1, so t̂_b can wander further).

### Combined Var_total = Var_cal + Var_audit

The MVP figure's whiskers are 95% intervals built from $\sigma_{\text{total}} = \sqrt{\mathrm{Var}_{\text{cal}} + \mathrm{Var}_{\text{audit}}}$, where:

- $\mathrm{Var}_{\text{cal}}$ is `var_eval` from `bootstrap_*_ci` (paired bootstrap of the calibrator-training rows)
- $\mathrm{Var}_{\text{audit}}$ is the audit-noise variance:
  - **Mean**: $(\mathrm{sd}(\varepsilon)/\sqrt{n_{\text{audit}}})^2$ where $\varepsilon = Y - \hat f(S)$ on the audit slice
  - **CVaR**: $\mathrm{se}_{g_2}^2$ from `cvar_audit_analytical_se` at the production $\hat t$

The **independence** assumption $\mathrm{Var}_{\text{cal}} \perp \mathrm{Var}_{\text{audit}}$ holds **by construction** in our pipeline: the calibration-training rows and audit-slice rows are disjoint (80/20 split of the logger oracle slice for `base`; different rows of the target slice for non-`base` policies). So adding the variances is honest, not pessimistic.

Empirically on the n=500 panel: the **mean** estimator's `Var_audit` is comparable in magnitude to its `Var_cal`, so $\sigma_{\text{total}}$ is roughly $\sqrt{2}\times \sigma_{\text{cal}}$ — the figure bars on the mean panel are visibly wider than what bootstrap-only would show. The **CVaR** estimator has `Var_cal` dominating (calibrator + argmax), so $\sigma_{\text{total}} \approx \sigma_{\text{cal}}$ and the figure bars barely move.

The pilot table now exports `cvar_se_total`, `mean_se_total` (in addition to the bootstrap-only `cvar_se_boot`, `mean_se_boot`) per row.

### Why we report bootstrap CIs on the figure, not the t-test SE

A figure CI should answer "where would V̂ land under a different slice?"
That's estimator variance — what `bootstrap_*_ci` measures. The audit SE
(t-test or analytical-CVaR) measures something different: a property of
the audit statistic, not of the estimate. Showing a t-test CI as if it
were the estimator CI would visually conflate the two and understate
uncertainty (audit SE is typically smaller than estimator SE, because the
calibrator-resampling component is missing).

## 4. The α = 1 collapse — code-level identities

When α = 1 and $\hat t \ge \max(y_{\mathrm{train}})$, every step of CVaR-CJE
reduces algebraically to the corresponding Mean-CJE step:

| at α = 1 | identity | tested in |
|---|---|---|
| **truth** | atom-split $\mathrm{CVaR}_1(Y) = \mathbb{E}[Y]$ | `test_alpha1_truth_equals_mean` |
| **point estimate** | $\widehat{\mathrm{CVaR}}_1 = (1/M)\sum_j \hat f(s_{\mathrm{eval},j}) = \widehat{\mathrm{Mean}}$ | `test_alpha1_estimator_equals_mean_cje` |
| **isotonic collapse** | $\hat g_{\hat t}(s) = \hat t - \hat f(s)$, max abs error ≤ 1e-12 | `test_alpha1_isotonic_collapse` |
| **g1 moment** | $\bar g_1 = 0$ exactly | `test_alpha1_g1_is_zero` |
| **g2 moment** | $\bar g_2 = -\bar\varepsilon$ exactly | `test_alpha1_g2_equals_neg_mean_residual` |
| **bootstrap CI** | per-rep cvar_b ≡ mean_b at α=1 | `test_alpha1_bootstrap_point_estimate_ci_agreement` |
| **g2 bootstrap CI** | bootstrap distribution of $\bar g_2$ matches $-\bar\varepsilon$ | `test_alpha1_bootstrap_g2_matches_mean_residual` |
| **calibration variance** | $\mathrm{Var}_{\mathrm{cal}}^{\mathrm{cvar}}(\alpha=1) = \mathrm{Var}_{\mathrm{cal}}^{\mathrm{mean}}$ | `test_alpha1_var_cal_agreement` |
| **audit p-value** (fixed $\hat t$) | bootstrap-Wald on g2 ≈ t-test on ε | `test_alpha1_audit_pvalue_agreement` |
| **grid bound** | $\hat t(\alpha=1) \ge \max(y_{\mathrm{train}})$ | `test_alpha1_t_hat_covers_y_train_max` |
| **legacy α** | low-α grid points unchanged after the α=1 fix | `test_low_alpha_grid_unchanged` |

Run all eleven with `python3 -m cvar_v4.healthbench_data.tests.test_alpha1_identity`.

## 5. The main derivation: paper math → code mapping

This section walks the CVaR-CJE pipeline end to end, in execution order, and
maps every paper equation to the function that implements it. Each step has
three blocks: **Paper math** (with equation labels and `file:line`
citations), **Pseudocode** (in Python-flavoured pseudo, mirroring
`_estimator.py` faithfully), and **Code** (function name and line range
in `_estimator.py`).

### 5.1 Notation and inputs

Symbols used throughout, fixed once.

| Symbol | Meaning | Code variable |
|---|---|---|
| $\Y \in [0,1]$ (or $\mathbb{R}$) | oracle label (HealthBench rubric score) | `y_train`, `y_audit` |
| $\Sscore$ | cheap-judge score | `s_train`, `s_audit`, `s_eval`, `s_eval_full` |
| $\X$ | optional covariates (e.g.\ response length) | folded into `s_train` upstream |
| $\pzero$ | logger / reference policy | `s_train`, `y_train` come from logger |
| $\ptarget$ | target policy being evaluated | `s_eval_full` are target's cheap scores |
| $\alpha \in (0,1]$ | tail fraction (worst-$\alpha$) | `alpha` |
| $\mathcal{T}$ | grid of candidate thresholds | `t_grid` from `make_t_grid` |
| $n_{\mathrm{train}}, n_{\mathrm{audit}}$ | oracle-slice / audit-slice sizes | `len(s_train)`, `len(s_audit)` |
| $f(\Sscore, \X)$ | mean calibrator $\approx \E_{\pzero}[\Y \mid \Sscore, \X]$ | output of `fit_isotonic_mean` |
| $\hat h_t(\Sscore, \X)$ | stop-loss calibrator $\approx \E_{\pzero}[(t-\Y)_+ \mid \Sscore, \X]$ | output of `fit_isotonic_tail_loss` (one per $t$) |
| $\hat t_\alpha$ | selected threshold | `t_hat` returned from `estimate_direct_cvar_isotonic` |
| $\bar g_1, \bar g_2$ | empirical audit moments | `gbar[0]`, `gbar[1]` from `_build_g_vector` |

Paper-side definitions: see `appendix_a_estimands.tex` lines 35–48 ($f_0$,
$h_{0,t}$, residuals $\varepsilon_f$, $u_t$).

### 5.2 Threshold-grid construction

**Paper math.** The grid $\mathcal{T}_n$ is required to (i) cover the
$\alpha$-quantile of $\Y$ under the target with positive density
(`appendix_a_estimands.tex:142-148`, `ass:regular-tail`), and (ii) extend
above $\max_i Y_i$ at $\alpha=1$ so the saddle-point optimum at $t \ge
Y_{\max}$ is reachable (the $\alpha=1$ collapse identity, see §4).

**Pseudocode.**
```python
def make_t_grid(y_train, alpha, grid_size=61):
    t_lo = quantile(y_train, max(0.001, alpha/5)) - 0.60
    t_hi_heuristic = quantile(y_train, min(0.60, alpha+0.45)) + 0.35
    t_hi_max = max(y_train) + 0.05
    base = linspace(t_lo, t_hi_heuristic, grid_size)
    if t_hi_max <= t_hi_heuristic:
        return base
    step = base[1] - base[0]
    n_extra = ceil((t_hi_max - t_hi_heuristic) / step)
    return concatenate([base, base[-1] + step * arange(1, n_extra+1)])
```

**Code.** `_estimator.py:42-66` (`make_t_grid`). The base linspace
preserves bit-identical low-$\alpha$ headlines from the legacy code; the
upward extension only adds points above the heuristic ceiling (zero
effect at low $\alpha$, decisive at $\alpha=1$).

### 5.3 Mean calibrator $\hat f$

**Paper math.** Reference-law conditional mean
$f_0(\Sscore, \X) = \E_{\pzero}[\Y \mid \mathcal{G}]$
(`appendix_a_estimands.tex:35-37`). The mean transport identity
$V(\ptarget) - V_f(\ptarget) = \E_\ptarget[\varepsilon_f]$
(`appendix_a_estimands.tex:65-69`) makes the target estimand
$V_f(\ptarget) = \E_\ptarget[f(\Sscore, \X)]$ the meaningful surrogate
for the mean (`ass:mean-transport`, lines 72–78).

**Pseudocode.**
```python
def fit_isotonic_mean(s_train, y_train, s_pred, sample_weight=None):
    order = argsort(s_train)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train[order], y_train[order], sample_weight=sample_weight[order])
    return iso.predict(s_pred)
```

**Code.** `_estimator.py:26-31` (`fit_isotonic_mean`). HT-weighted via
`sample_weight`. Out-of-range `s_pred` values are clipped to the fitted
range.

### 5.4 Stop-loss calibrator family $\{\hat h_t\}_{t \in \mathcal{T}}$

**Paper math.** For each $t$, the calibrator targets
$\hat h_t(\Sscore, \X) \approx \E_{\pzero}[(t-\Y)_+ \mid \Sscore, \X]$
(`method.tex:32-33`, `appendix_a_estimands.tex:38-40`). The shortfall
residual $u_t = (t-\Y)_+ - h_t(\Sscore, \X)$
(`appendix_a_estimands.tex:47-48`) must satisfy
`ass:shortfall-transport` (`appendix_a_estimands.tex:246-253`):
$\E_\ptarget[u_t] = 0$ for every $t \in \mathcal{T}$ near the oracle
optimizer. The fit is a **decreasing** isotonic regression: as $\Sscore$
increases, the conditional shortfall $\E[(t-\Y)_+ \mid \Sscore]$ should
decrease (better cheap scores → less shortfall). Cross-fitting over $K$
folds is required (`ass:nuisance`,
`appendix_a_estimands.tex:344-360`).

**Pseudocode.**
```python
def fit_isotonic_tail_loss(s_train, z_train, s_pred, sample_weight=None):
    # z_train = (t - y_train)_+ has been computed by the caller
    order = argsort(s_train)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(s_train[order], z_train[order], sample_weight=sample_weight[order])
    return iso.predict(s_pred)

# Caller computes z and loops over t (inlined in 5.5):
for t in t_grid:
    z_train = maximum(t - y_train, 0)
    h_hat_t_pred = fit_isotonic_tail_loss(s_train, z_train, s_pred,
                                           sample_weight=w_train)
```

**Code.** `_estimator.py:34-39` (`fit_isotonic_tail_loss`, single $t$).
The loop over $t$ lives in `estimate_direct_cvar_isotonic`, lines 87–94
— each iteration recomputes $z_{\mathrm{train}}$ and refits a fresh
isotonic. The decreasing direction is the only structural difference
from §5.3.

### 5.5 Direct CVaR-CJE point estimate

**Paper math.** The Rockafellar–Uryasev dual
$\CVaR_\alpha(\ptarget) = \sup_{t}[t - \alpha^{-1}\E_\ptarget(t-\Y)_+]$
(`method.tex:13-17`, `eq:stoploss-cvar`) is replaced by a finite-sample
surrogate (`method.tex:46-56`, `eq:direct-estimator`):

$$
\hat\Psi_{\alpha,\ptarget}(t) = t - \frac{1}{\alpha n}\sum_{i=1}^{n} \hat h_t(\Sscore_i, \X_i),
\qquad
\widehat{\CVaR}_{\alpha}^{\mathrm{direct}}(\ptarget) = \max_{t \in \mathcal{T}} \hat\Psi_{\alpha,\ptarget}(t),
\qquad
\hat t_\alpha = \arg\max_{t \in \mathcal{T}} \hat\Psi_{\alpha,\ptarget}(t).
$$

Identification under shortfall transport: see
`thm:formal-cvar-identification` in
`appendix_a_estimands.tex:255-273`.

**Pseudocode.**
```python
def estimate_direct_cvar_isotonic(s_train, y_train, s_eval, alpha,
                                    grid_size=61, sample_weight_train=None):
    t_grid = make_t_grid(y_train, alpha, grid_size)
    obj = empty(len(t_grid))
    for i, t in enumerate(t_grid):
        z_train = maximum(t - y_train, 0)
        pred_eval = fit_isotonic_tail_loss(
            s_train, z_train, s_eval, sample_weight=sample_weight_train)
        obj[i] = t - pred_eval.mean() / alpha
    best = argmax(obj)
    return obj[best], t_grid[best], t_grid, obj   # (cvar_hat, t_hat, grid, obj)
```

**Code.** `_estimator.py:69-97` (`estimate_direct_cvar_isotonic`). The
`s_eval` average is unweighted by design — the CVaR is over the
$\ptarget$ distribution, not the slice.

### 5.6 Direct mean-CJE point estimate (the $\alpha=1$ case)

**Paper math.** $V(\ptarget) = \E_\ptarget[\Y]$
(`appendix_a_estimands.tex:57-59`); under mean transport
$V_f(\ptarget) = V(\ptarget)$. At $\alpha=1$ this is exactly §5.5
applied with a different calibration target — see §4 above for the
chain of identities (`prop:formal-alpha-one`,
`appendix_d_alpha_one.tex:12-23`).

**Pseudocode.**
```python
def mean_point_estimate(s_train, y_train, s_eval, sample_weight_train=None):
    f_hat = fit_isotonic_mean(s_train, y_train, s_eval,
                               sample_weight=sample_weight_train)
    return f_hat.mean()
```

**Code.** No standalone wrapper — callers invoke `fit_isotonic_mean(...)`
directly and take the mean. Used inside `bootstrap_mean_ci` and
`pipeline_bootstrap_mean` (see §5.12).

### 5.7 Two-moment audit (point moments)

**Paper math.** At the selected threshold $\hat t_\alpha$, on a held-out
audit slice (`appendix_b_audit.tex:84-88`):

$$
\bar g_1 = \frac{1}{n_{\mathrm{audit}}} \sum_i \left[\one\{\Y_i \le \hat t_\alpha\} - \alpha\right],
\qquad
\bar g_2 = \frac{1}{n_{\mathrm{audit}}} \sum_i \left[(\hat t_\alpha - \Y_i)_+ - \hat h_{\hat t_\alpha}(\Sscore_i, \X_i)\right].
$$

Population versions and target-moment theorem:
`appendix_b_audit.tex:16-22` and `thm:formal-two-moment` (lines 23–33).
$\bar g_1$ is the tail-mass restriction; $\bar g_2$ is the
shortfall-transport restriction at $\hat t_\alpha$.

**Pseudocode.**
```python
def _build_g_vector(s_train, y_train, s_audit, y_audit, t0, alpha,
                    sample_weight_train=None, sample_weight_audit=None):
    z_train = maximum(t0 - y_train, 0)
    pred_audit = fit_isotonic_tail_loss(
        s_train, z_train, s_audit, sample_weight=sample_weight_train)
    g1 = (y_audit <= t0).astype(float) - alpha
    g2 = maximum(t0 - y_audit, 0) - pred_audit
    if sample_weight_audit is None:
        return array([g1.mean(), g2.mean()])
    w = sample_weight_audit
    return array([(w*g1).sum() / w.sum(), (w*g2).sum() / w.sum()])
```

**Code.** `_estimator.py:153-174` (`_build_g_vector`). Note: the
calibrator is fit fresh on `s_train, z_train(t0)` inside this function
— the same calibrator used at the same threshold in §5.5 must be reused
for the audit to test the right transport restriction.

### 5.8 Mean-transport audit (paired $t$-test on residuals)

**Paper math.** Test of $\E_\ptarget[\Y - f(\Sscore, \X)] = 0$ on the
audit slice (`ass:mean-transport`, `appendix_a_estimands.tex:72-78`).
This is the canonical one-sample $t$-test on residuals
$\varepsilon_i = Y_i - \hat f(\Sscore_i)$.

**Pseudocode.**
```python
def mean_transport_audit(s_train, y_train, s_audit, y_audit,
                         sample_weight_train=None, sample_weight_audit=None,
                         test_alpha=0.05):
    f_hat_audit = fit_isotonic_mean(s_train, y_train, s_audit,
                                      sample_weight=sample_weight_train)
    eps = y_audit - f_hat_audit
    if sample_weight_audit is None:
        t_stat, p_val = scipy.stats.ttest_1samp(eps, 0.0)
    else:
        w = sample_weight_audit
        mu = (w*eps).sum() / w.sum()
        var = (w*(eps - mu)**2).sum() / w.sum()
        n_eff = (w.sum())**2 / (w**2).sum()       # Kish ESS
        t_stat = mu / sqrt(var / n_eff)
        p_val = 2 * (1 - student_t_cdf(abs(t_stat), df=n_eff - 1))
    return {"residual_mean": mu, "t_stat": t_stat,
            "p_value": p_val, "reject": p_val < test_alpha}
```

**Code.** `_estimator.py:927-983` (`mean_transport_audit`).

### 5.9 Wald test (joint two-moment)

**Paper math.** $W_n = n \bar g_n^\top \hat\Omega^{-1} \bar g_n$ where
$\hat\Omega$ is a consistent estimator of the asymptotic covariance,
*including the first-order effect of fitting $\hat h_t$ and selecting
$\hat t_\alpha$* (`appendix_b_audit.tex:90-95`, `eq:formal-wald`). Under
`thm:formal-wald-limit` (lines 98–113), $W_n \rightsquigarrow \chi^2_2$.
Reject if $W_n > \chi^2_{2,1-\eta}$ (`method.tex:83-87`, `eq:gate`).

The covariance must propagate the $\hat h_t$-fit and $\hat t$-selection
randomness — that's why we estimate it from the bootstrap, not from a
fixed-$\hat t$ analytic formula. This is the appendix-(viii) fix: a
naive analytic $\hat\Omega$ over-rejects (~0.50 size at the truest
null).

**Pseudocode.**
```python
def two_moment_wald_audit_xf(s_train, y_train, s_audit, y_audit, t0, alpha,
                              B=100, fold_seed=0, wald_alpha=0.05, ...):
    gbar = _build_g_vector(s_train, y_train, s_audit, y_audit, t0, alpha, ...)
    g_per_boot = _bootstrap_g_vectors(s_train, y_train, s_audit, y_audit,
                                       alpha, B=B, fold_seed=fold_seed, ...)
    Sigma = cov(g_per_boot, rowvar=False, ddof=1)
    eps = max(1e-6, 1 / max(len(s_audit), 1))
    Sigma += eps * eye(2)                          # ridge
    W = gbar.T @ inv(Sigma) @ gbar
    p = 1 - chi2.cdf(W, df=2)
    return {"wald_stat": W, "p_value": p, "reject": p < wald_alpha,
            "mean_g1": gbar[0], "mean_g2": gbar[1], "t0": t0}

def _bootstrap_g_vectors(s_train, y_train, s_audit, y_audit, alpha, ...,
                          B=200, fold_seed=0):
    rng = default_rng(fold_seed)
    g = empty((B, 2))
    for b in range(B):
        idx_t = rng.integers(0, n_train, size=n_train)
        idx_a = rng.integers(0, n_audit, size=n_audit)
        # CRUCIAL: re-pick t̂ on bootstrap data
        _, t_b, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx_t], y_train[idx_t], s_audit[idx_a], alpha, ...)
        g[b] = _build_g_vector(s_train[idx_t], y_train[idx_t],
                                s_audit[idx_a], y_audit[idx_a], t_b, alpha)
    return g
```

**Code.** `_estimator.py:212-273` (`two_moment_wald_audit_xf`) +
`_estimator.py:177-209` (`_bootstrap_g_vectors`). Ridge stabilization
at line 257 (`eps = max(1e-6, 1/n_audit)`).

### 5.10 Single-moment audits ($g_1$ alone, $g_2$ alone)

**Paper math.** The four-cell readout of
`appendix_b_audit.tex:187-200+`: each moment failing alone points at a
specific knob. $g_1$ failure $\to$ wrong threshold; $g_2$ failure $\to$
calibrator off.

**Pseudocode.**
```python
def _single_moment_audit(s_train, y_train, s_audit, y_audit, t0, alpha,
                          moment_idx, variant_name, ...):
    gbar = _build_g_vector(...)
    g_per_boot = _bootstrap_g_vectors(...)            # same boot as 5.9
    g_scalar = gbar[moment_idx]
    var_scalar = var(g_per_boot[:, moment_idx], ddof=1)
    var_scalar = max(var_scalar, 1 / max(n_audit, 1))  # ridge
    W = g_scalar**2 / var_scalar
    p = 1 - chi2.cdf(W, df=1)
    return {"audit_variant": variant_name, "wald_stat": W, "p_value": p, ...}

g1_only_audit_xf = partial(_single_moment_audit, moment_idx=0,
                             variant_name="g1_only")
g2_only_audit_xf = partial(_single_moment_audit, moment_idx=1,
                             variant_name="g2_only")
```

**Code.** `_estimator.py:276-310` (`_single_moment_audit`); convenience
wrappers `g1_only_audit_xf` (lines 313–325) and `g2_only_audit_xf`
(lines 328–340). Registered in `AUDIT_VARIANTS` dict at line 343.

### 5.11 Full-pipeline bootstrap CI for CVaR

**Paper math.** Algorithm `alg:bootstrap` in
`appendix_c_bootstrap.tex:18-66`. Resample prompt clusters; refit the
calibrator family; **re-optimize $\hat t_\alpha$** inside each
replicate; recompute the estimate. The re-optimization is required
because the saddle-point objective trades off tail-mass against
shortfall (`appendix_c_bootstrap.tex:13-16`). 95% percentile CI from
$\{\hat V^{(b)}_\alpha\}_{b=1}^B$.

`pipeline_bootstrap_cvar` adds two practical extensions: (a)
`resample` flag selects which subsets (`train`, `eval`, `audit`) to
bootstrap, so the same engine handles the full pipeline ($\Var_{\rm
total}$), the calibrator-only mode ($\Var_{\rm cal}$, see §5.13), or
the eval-only mode; (b) the augmented estimator
$\widehat V_{\mathrm{aug}} = \widehat{\CVaR} + \bar g_2$ is computed
alongside the plug-in, since adding the bias-correction term gives
near-nominal coverage even at small oracle slices.

**Pseudocode.**
```python
def pipeline_bootstrap_cvar(s_train, y_train, s_eval_full, s_audit, y_audit,
                              alpha, resample=("train", "eval", "audit"),
                              B=500, seed=42, ci=0.95, ...):
    plug_point, t_hat_point, _, _ = estimate_direct_cvar_isotonic(
        s_train, y_train, s_eval_full, alpha, ...)
    gbar2_point = _gbar2(s_train, y_train, t_hat_point,
                          s_audit, y_audit, ...)
    aug_point = plug_point + gbar2_point

    rng = default_rng(seed)
    plug_boots, aug_boots, t_hat_boots = empty(B), empty(B), empty(B)
    for b in range(B):
        idx_t = rng.integers(0, n_t) if "train" in resample else arange(n_t)
        idx_e = rng.integers(0, n_e) if "eval"  in resample else arange(n_e)
        idx_a = rng.integers(0, n_a) if "audit" in resample else arange(n_a)
        plug_b, t_hat_b, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx_t], y_train[idx_t], s_eval_full[idx_e], alpha, ...)
        gbar2_b = _gbar2(s_train[idx_t], y_train[idx_t], t_hat_b,
                          s_audit[idx_a], y_audit[idx_a], ...)
        plug_boots[b] = plug_b
        aug_boots[b]  = plug_b + gbar2_b
        t_hat_boots[b] = t_hat_b

    return {
        "plug_boots": plug_boots, "aug_boots": aug_boots,
        "plug_point": plug_point, "aug_point": aug_point,
        "t_hat_boots": t_hat_boots, "t_hat_point": t_hat_point,
        "var_plug": var(plug_boots, ddof=1),
        "var_aug":  var(aug_boots,  ddof=1),
        "ci_plug": (quantile(plug_boots, (1-ci)/2),
                    quantile(plug_boots, 1-(1-ci)/2)),
        "ci_aug":  (quantile(aug_boots,  (1-ci)/2),
                    quantile(aug_boots,  1-(1-ci)/2)),
        "B": B, "resample": resample,
    }
```

**Code.** `_estimator.py:552-714` (`pipeline_bootstrap_cvar`). The
nested helper `_gbar2` lives at lines 613–623; the bootstrap loop is
lines 659–693. A simpler subset-mode wrapper `bootstrap_cvar_ci` lives
at lines 497–545.

### 5.12 Full-pipeline bootstrap CI for the mean

**Paper math.** Mirror of §5.11 with the mean calibrator $\hat f$ in
place of the family $\{\hat h_t\}$.

**Pseudocode.**
```python
def pipeline_bootstrap_mean(s_train, y_train, s_eval_full,
                              resample=("train", "eval"), B=500, seed=42, ...):
    point = fit_isotonic_mean(s_train, y_train, s_eval_full, ...).mean()
    rng = default_rng(seed)
    boots = empty(B)
    for b in range(B):
        idx_t = rng.integers(0, n_t) if "train" in resample else arange(n_t)
        idx_e = rng.integers(0, n_e) if "eval"  in resample else arange(n_e)
        f_hat_b = fit_isotonic_mean(
            s_train[idx_t], y_train[idx_t], s_eval_full[idx_e], ...)
        boots[b] = f_hat_b.mean()
    return {"boots": boots, "point": point,
            "var_eval": var(boots, ddof=1), ...}
```

**Code.** `_estimator.py:765-834` (`pipeline_bootstrap_mean`); simpler
wrapper `bootstrap_mean_ci` at lines 717–762.

**$\alpha=1$ identity.** With the same RNG seed,
`pipeline_bootstrap_cvar(..., alpha=1.0)` and
`pipeline_bootstrap_mean(...)` must produce identical per-replicate
draws — see §4 row "bootstrap CI" and
`test_alpha1_bootstrap_point_estimate_ci_agreement`.

### 5.13 Variance decomposition: $\Var_{\rm total} \approx \Var_{\rm cal} + \Var_{\rm audit}$

**Paper math.**
`appendix_c_bootstrap.tex:80-96` (`eq:var-total`):

$$\Var_{\mathrm{total}}(\hat V_\alpha) \approx \Var_{\mathrm{cal}}(\hat V_\alpha) + \Var_{\mathrm{audit}}(\hat V_\alpha).$$

Approximate independence holds because the calibration slice and the
audit slice are disjoint by construction
(`appendix_b_audit.tex:164-179`, the $80/20$ self-audit split when the
target equals the logger).

- $\Var_{\rm cal}$: how much $\hat V_\alpha$ moves under different draws
  of the oracle calibration slice. Estimated via delete-one-fold
  jackknife with Efron–Stein scaling.
- $\Var_{\rm audit}$: finite-sample noise of the audit-slice average of
  the calibrated shortfall. Either an analytical fixed-$\hat t$ SE or
  the bootstrap variance from `_bootstrap_g_vectors` resampling only
  the audit.

**Pseudocode (calibrator side, CVaR).**
```python
def jackknife_var_cal(s_train, y_train, s_eval, alpha, K=5,
                       sample_weight_train=None, fold_seed=0):
    rng = default_rng(fold_seed)
    perm = rng.permutation(len(s_train))
    folds = [perm[k::K] for k in range(K)]            # round-robin
    cvar_k = empty(K)
    for k, fold_idx in enumerate(folds):
        keep = setdiff1d(arange(len(s_train)), fold_idx)
        cvar_k[k], _, _, _ = estimate_direct_cvar_isotonic(
            s_train[keep], y_train[keep], s_eval, alpha,
            sample_weight_train=sample_weight_train[keep] if ... else None)
    return ((K-1)/K) * sum((cvar_k - cvar_k.mean())**2)
```

**Pseudocode (calibrator side, mean).** Identical structure with
`fit_isotonic_mean(..).mean()` in place of `estimate_direct_cvar_isotonic`.

**Pseudocode (audit side).**
```python
def cvar_audit_analytical_se(s_train, y_train, s_audit, y_audit, t0, alpha, ...):
    z_train = maximum(t0 - y_train, 0)
    pred_audit = fit_isotonic_tail_loss(s_train, z_train, s_audit, ...)
    g1 = (y_audit <= t0).astype(float) - alpha
    g2 = maximum(t0 - y_audit, 0) - pred_audit
    n_a = len(y_audit)
    return {
        "mean_g1": g1.mean(), "mean_g2": g2.mean(),
        "se_g1":   g1.std(ddof=1) / sqrt(n_a),
        "se_g2":   g2.std(ddof=1) / sqrt(n_a),
        "cov_g1g2": cov(g1, g2, ddof=1) / n_a,
        "n_audit": n_a, "t0": t0,
    }
```

**Code.**
- $\Var_{\rm cal}$ for CVaR: `_estimator.py:841-880`
  (`jackknife_var_cal`).
- $\Var_{\rm cal}$ for mean: `_estimator.py:883-920`
  (`jackknife_var_cal_mean`).
- Analytical $\Var_{\rm audit}$: `_estimator.py:432-490`
  (`cvar_audit_analytical_se`).
- The total envelope $\sigma_{\rm total} = \sqrt{\Var_{\rm cal} +
  \Var_{\rm audit}}$ used on the figure (\cref{fig:mvp}) is assembled
  by the caller; the estimator exports the components only.

**Concrete numbers.** From `appendix_c_bootstrap.tex:90-96` at the
$25\%$ oracle slice and $\alpha = 0.10$:
- `base`: $\Var_{\rm cal} = 5.15 \times 10^{-3}$, $\Var_{\rm audit} =
  2.07 \times 10^{-4}$, $\sigma_{\rm total} = 0.073$.
- `unhelpful`: $\Var_{\rm cal} = 1.76 \times 10^{-2}$, $\Var_{\rm audit} =
  1.75 \times 10^{-5}$ — calibrator-side dominates by ~10³, reflecting an
  unstable isotonic fit on a tail with very different shape from the
  logger.

### 5.14 The $\alpha=1$ identity as a regression test

**Paper math.** `prop:formal-alpha-one`,
`appendix_d_alpha_one.tex:12-23`: at $\alpha=1$ with $\Y \in [0,1]$,
$\CVaR_1(\ptarget) = V(\ptarget)$. The proof goes through
$h_1(\Sscore, \X) = 1 - f(\Sscore, \X)$ (`appendix_d_alpha_one.tex:53-58`),
which forces $g_2 = -\varepsilon_f$ (lines 62–64) and $g_1 = 0$ (line 71).

**Pseudocode.**
```python
def test_alpha1_bootstrap_point_estimate_ci_agreement(...):
    # same RNG seed in both calls
    cvar = pipeline_bootstrap_cvar(..., alpha=1.0, seed=42, B=200, ...)
    mean = pipeline_bootstrap_mean(..., seed=42, B=200, ...)
    assert max_abs(cvar["plug_boots"] - mean["boots"]) < 1e-12
    assert abs(cvar["plug_point"] - mean["point"]) < 1e-12
```

**Code.** Eleven such regression checks live in
`cvar_v4/healthbench_data/tests/test_alpha1_identity.py` (see §4 table
above for the full list). Each one anchors a specific identity from
`appendix_d_alpha_one.tex` to a code-level invariant. Run them with
`python3 -m cvar_v4.healthbench_data.tests.test_alpha1_identity`.

---

## Cross-references at a glance

| Pipeline stage | Paper section | Code function | Lines |
|---|---|---|---|
| Threshold grid | `app_a:142-148` | `make_t_grid` | 42–66 |
| Mean calibrator | `app_a:35-37, 72-78` | `fit_isotonic_mean` | 26–31 |
| Stop-loss family | `method:32-33`, `app_a:38-48,246-253` | `fit_isotonic_tail_loss` | 34–39 |
| Direct CVaR estimate | `method:46-56` (`eq:direct-estimator`) | `estimate_direct_cvar_isotonic` | 69–97 |
| Direct mean estimate | `app_a:57-59`, `app_d` | (caller uses `fit_isotonic_mean`) | — |
| Audit moments | `method:67-72` (`eq:audit`), `app_b:16-22` | `_build_g_vector` | 153–174 |
| Mean transport audit | `app_a:72-78` (`ass:mean-transport`) | `mean_transport_audit` | 927–983 |
| Joint Wald | `app_b:90-113` (`eq:formal-wald`, `thm:formal-wald-limit`) | `two_moment_wald_audit_xf` + `_bootstrap_g_vectors` | 212–273 + 177–209 |
| Single-moment audits | `app_b:187-200+` | `_single_moment_audit`, `g1_only_audit_xf`, `g2_only_audit_xf` | 276–340 |
| Heuristic audit (no boot) | `method:67-87` | `simple_cvar_audit` | 354–429 |
| Audit analytical SE | `app_c:67-96` (audit side) | `cvar_audit_analytical_se` | 432–490 |
| Pipeline bootstrap CI (CVaR) | `app_c:18-66` (`alg:bootstrap`) | `pipeline_bootstrap_cvar` (and `bootstrap_cvar_ci`) | 552–714 (497–545) |
| Pipeline bootstrap CI (mean) | `app_c:18-66` (mirror) | `pipeline_bootstrap_mean` (and `bootstrap_mean_ci`) | 765–834 (717–762) |
| $\Var_{\rm cal}$ (CVaR) | `app_c:67-96` (calibrator side) | `jackknife_var_cal` | 841–880 |
| $\Var_{\rm cal}$ (mean) | `app_c:67-96` (calibrator side) | `jackknife_var_cal_mean` | 883–920 |
| $\alpha=1$ regression tests | `app_d` (whole section) | `test_alpha1_identity.py` (eleven tests) | — |
