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
