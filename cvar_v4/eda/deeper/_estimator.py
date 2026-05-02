"""Self-contained Direct CVaR-CJE estimator + Wald audits.

Inlined from cvar_v3/workhorse.py. Pure numpy + scipy + sklearn.

This module supports HT-weighted calibration (sample_weight on the isotonic
fits) so the design-aware oracle slice in oracle_design.py composes cleanly
with the estimator. Three audit variants share the bootstrap-Σ̂ structure
that re-fits the calibrator and re-maximizes t̂ inside each rep, which
captures argmax variance:

  - two_moment_wald_audit_xf — joint Wald on (g1, g2). 2-df chi-square.
  - g1_only_audit_xf         — single-moment z² on tail-mass.    1-df chi-square.
  - g2_only_audit_xf         — single-moment z² on stop-loss res. 1-df chi-square.

Where:
  g1 = 1{y_audit ≤ t̂} − α                        (tail-mass moment)
  g2 = (t̂ − y_audit)_+ − ĝ_t̂(s_audit)           (stop-loss residual)
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_mean(s_train, y_train, s_pred, *, sample_weight=None):
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    sw = None if sample_weight is None else np.asarray(sample_weight)[order]
    iso.fit(s_train[order], y_train[order], sample_weight=sw)
    return iso.predict(s_pred)


def fit_isotonic_tail_loss(s_train, z_train, s_pred, *, sample_weight=None):
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    sw = None if sample_weight is None else np.asarray(sample_weight)[order]
    iso.fit(s_train[order], z_train[order], sample_weight=sw)
    return iso.predict(s_pred)


def fit_two_stage_tail_loss(
    s_train,
    length_train,
    z_train,
    s_pred,
    length_pred,
    *,
    sample_weight=None,
    ridge_alpha: float = 1.0,
):
    """Two-stage version of `fit_isotonic_tail_loss` for `+cov` CVaR-CJE.

    Stage 1: ridge regression `z ≈ a*S + b*S² + c*length + d*length² +
             e*S*length + intercept` on the training rows. Z = predicted z.
    Stage 2: ECDF transform of Z (rank-based on the training distribution),
             then mean-preserving isotonic regression of `z` vs ECDF(Z) with
             `increasing=False` (high cheap-S → high Y → low z = max(t−Y, 0)).

    Mirror of `fit_two_stage_calibrator` but with the target as the
    stop-loss `z = max(t − Y, 0)` rather than `Y`. Used in the `+cov` CVaR
    estimator path so the tail-loss calibrator picks up the same
    response-length covariate as the mean-side `+cov`.
    """
    from sklearn.linear_model import Ridge

    s_train = np.asarray(s_train, dtype=float)
    length_train = np.asarray(length_train, dtype=float)
    z_train = np.asarray(z_train, dtype=float)
    s_pred = np.asarray(s_pred, dtype=float)
    length_pred = np.asarray(length_pred, dtype=float)

    def _features(s, length):
        return np.column_stack([s, s ** 2, length, length ** 2, s * length])

    X_train = _features(s_train, length_train)
    X_pred = _features(s_pred, length_pred)

    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    stage1 = Ridge(alpha=ridge_alpha)
    stage1.fit(X_train, z_train, sample_weight=sw)
    z_index_train = stage1.predict(X_train)
    z_index_pred = stage1.predict(X_pred)

    sorted_z = np.sort(z_index_train)
    n = len(sorted_z)
    z_train_ecdf = np.searchsorted(sorted_z, z_index_train, side="right") / n
    z_pred_ecdf = np.searchsorted(sorted_z, z_index_pred, side="right") / n

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    order = np.argsort(z_train_ecdf)
    sw_ord = None if sw is None else sw[order]
    iso.fit(z_train_ecdf[order], z_train[order], sample_weight=sw_ord)
    return iso.predict(z_pred_ecdf)


def fit_two_stage_calibrator(
    s_train,
    length_train,
    y_train,
    s_pred,
    length_pred,
    *,
    sample_weight=None,
    ridge_alpha: float = 1.0,
):
    """Two-stage calibrator matching CJE's `direct+cov` form
    (arxiv 2512.11150 method.tex §15-30).

    Stage 1: ridge regression `Y ≈ a*S + b*S² + c*length + d*length² +
             e*S*length + intercept` on the oracle slice. Produces a 1D
             index Z(S, length) = predicted Y from stage 1.
    Stage 2: ECDF transform of Z (rank-based on training distribution),
             then mean-preserving isotonic regression of Y vs ECDF(Z).

    Returns predicted Y values for (s_pred, length_pred). When `length_train`
    and `length_pred` are both constant (e.g. all zeros), the calibrator
    degenerates into a near-equivalent of `fit_isotonic_mean` on S alone
    (ridge on S only → isotonic on Y vs ECDF(Z)).

    Required covariate per SPEC_SHEET §249. K-fold cross-fitting is NOT
    applied here; we mirror the existing single-fit pattern of
    `fit_isotonic_mean`. Variance is captured by the existing
    `jackknife_var_cal` machinery, which already does delete-one-fold on
    the calibrator.
    """
    from sklearn.linear_model import Ridge

    s_train = np.asarray(s_train, dtype=float)
    length_train = np.asarray(length_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    s_pred = np.asarray(s_pred, dtype=float)
    length_pred = np.asarray(length_pred, dtype=float)

    # Feature matrix: [S, S², length, length², S*length].
    def _features(s, length):
        return np.column_stack([s, s ** 2, length, length ** 2, s * length])

    X_train = _features(s_train, length_train)
    X_pred = _features(s_pred, length_pred)

    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    stage1 = Ridge(alpha=ridge_alpha)
    stage1.fit(X_train, y_train, sample_weight=sw)
    z_train = stage1.predict(X_train)
    z_pred = stage1.predict(X_pred)

    # ECDF transform of Z based on the training distribution.
    sorted_z = np.sort(z_train)
    n = len(sorted_z)
    z_train_ecdf = np.searchsorted(sorted_z, z_train, side="right") / n
    z_pred_ecdf = np.searchsorted(sorted_z, z_pred, side="right") / n

    # Stage 2: mean-preserving isotonic Y vs ECDF(Z).
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    order = np.argsort(z_train_ecdf)
    sw_ord = None if sw is None else sw[order]
    iso.fit(z_train_ecdf[order], y_train[order], sample_weight=sw_ord)
    return iso.predict(z_pred_ecdf)


def make_t_grid(y_train, alpha: float, grid_size: int = 61):
    """Threshold grid for the CVaR saddle-point.

    Lower bound: the α-tail of y_train minus a 0.60 buffer.
    Upper bound (heuristic): a quantile-plus-margin tuned for low-α tails.
    Upper extension: appends additional points up to max(y_train)+0.05 when
    the heuristic ceiling falls short — required for the α=1 identity
    `CVaR_α=1 = E[Y]` to hold exactly, since the saddle-point optimum sits
    at any t ≥ Y_max.

    The first `grid_size` points exactly reproduce the legacy linspace
    (so headline low-α results are bit-identical); the extension only
    adds points above `t_hi_heuristic`. Net effect at low α: zero.
    """
    y_train = np.asarray(y_train)
    t_lo = float(np.quantile(y_train, max(0.001, alpha / 5.0)) - 0.60)
    t_hi_heuristic = float(np.quantile(y_train, min(0.60, alpha + 0.45)) + 0.35)
    t_hi_max = float(y_train.max()) + 0.05
    base_grid = np.linspace(t_lo, t_hi_heuristic, grid_size)
    if t_hi_max <= t_hi_heuristic + 1e-12 or grid_size < 2:
        return base_grid
    step = float(base_grid[1] - base_grid[0])
    n_extra = int(np.ceil((t_hi_max - t_hi_heuristic) / step))
    extra = base_grid[-1] + np.arange(1, n_extra + 1) * step
    return np.concatenate([base_grid, extra])


def estimate_direct_cvar_isotonic(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    grid_size: int = 61,
    *,
    sample_weight_train: np.ndarray | None = None,
    length_train: np.ndarray | None = None,
    length_eval: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Direct CVaR estimator — grid search over stop-loss thresholds.

    The isotonic stop-loss calibrator is fit on s_train with HT weights
    (sample_weight_train, e.g. 1/π_i). The evaluation mean over s_eval is
    UNWEIGHTED — we want CVaR_α of the policy-π' distribution, not of the
    slice distribution.

    When `length_train` and `length_eval` are both provided, runs in `+cov`
    mode: the tail-loss calibrator becomes `fit_two_stage_tail_loss` with
    response_length as a covariate (CJE `direct+cov` for the tail estimand).
    Otherwise falls back to `fit_isotonic_tail_loss` (S-only).

    Returns (cvar_estimate, t_hat, t_grid, objective).
    """
    use_cov = length_train is not None and length_eval is not None
    t_grid = make_t_grid(y_train, alpha, grid_size)
    objective = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        z_train = np.maximum(t - y_train, 0.0)
        if use_cov:
            pred_eval = fit_two_stage_tail_loss(
                s_train, length_train, z_train, s_eval, length_eval,
                sample_weight=sample_weight_train,
            )
        else:
            pred_eval = fit_isotonic_tail_loss(
                s_train, z_train, s_eval,
                sample_weight=sample_weight_train,
            )
        objective[i] = float(t - pred_eval.mean() / alpha)
    best = int(np.argmax(objective))
    return float(objective[best]), float(t_grid[best]), t_grid, objective


def estimate_plugin_cvar_quantile(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    *,
    sample_weight_train: np.ndarray | None = None,
) -> float:
    """Plug-in CVaR_α via empirical α-tail of m̂(s_eval).

    Fits a single isotonic mean calibrator m̂(s) ≈ E[Y|s] on (s_train, y_train),
    evaluates on s_eval, and returns the empirical lower-α CVaR of the
    predicted vector. Biased toward the unconditional mean when Var(Y|S) > 0:
    the predicted distribution shrinks the tail relative to the true Y|S
    distribution.
    """
    m_hat = fit_isotonic_mean(
        s_train, y_train, s_eval, sample_weight=sample_weight_train
    )
    k = max(1, int(np.ceil(alpha * len(m_hat))))
    sorted_pred = np.sort(m_hat)
    return float(sorted_pred[:k].mean())


def estimate_plugin_cvar_ru_dual(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    grid_size: int = 61,
    *,
    sample_weight_train: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Plug-in CVaR_α via the Rockafellar–Uryasev dual on m̂.

    Fit m̂(s) ≈ E[Y|s] once, then maximize
        max_t [ t − α⁻¹ · mean_eval((t − m̂(s_eval))_+) ]
    over the same t-grid as estimate_direct_cvar_isotonic. Biased upward
    (CVaR overestimated, tail risk underestimated) by the Jensen gap on
    (t − ·)_+ at every t ≥ ess inf m̂.
    """
    m_eval = fit_isotonic_mean(
        s_train, y_train, s_eval, sample_weight=sample_weight_train
    )
    t_grid = make_t_grid(y_train, alpha, grid_size)
    objective = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        shortfall = np.maximum(t - m_eval, 0.0)
        objective[i] = float(t - shortfall.mean() / alpha)
    best = int(np.argmax(objective))
    return float(objective[best]), float(t_grid[best]), t_grid, objective


def _build_g_vector(
    s_train, y_train, s_audit, y_audit, t0, alpha,
    *, sample_weight_train=None, sample_weight_audit=None,
) -> np.ndarray:
    """Compute (g1_bar, g2_bar) at the supplied t0 with optional HT weights
    on the audit-side averages."""
    # Calibrator at t0 — fit on training (slice) with weights
    z_train_full = np.maximum(t0 - y_train, 0.0)
    pred_audit_full = fit_isotonic_tail_loss(
        s_train, z_train_full, s_audit,
        sample_weight=sample_weight_train,
    )
    # Audit-side moments. If sample_weight_audit is supplied, take HT-weighted
    # mean (mean of weight*x / mean of weight). Otherwise plain mean.
    g1 = (y_audit <= t0).astype(float) - alpha
    g2 = np.maximum(t0 - y_audit, 0.0) - pred_audit_full
    if sample_weight_audit is None:
        return np.array([g1.mean(), g2.mean()])
    w = np.asarray(sample_weight_audit, dtype=float)
    if w.sum() <= 0:
        return np.array([float("nan"), float("nan")])
    return np.array([(w * g1).sum() / w.sum(), (w * g2).sum() / w.sum()])


def _bootstrap_g_vectors(
    s_train, y_train, s_audit, y_audit, alpha,
    *, sample_weight_train=None, sample_weight_audit=None,
    B=200, fold_seed=0, grid_size=61,
) -> np.ndarray:
    """Paired bootstrap with t̂ re-maximization. Returns array of shape (B, 2).

    Both train and audit are resampled with replacement (preserving sample
    weights). t̂ is re-maximized inside each rep so Σ̂ captures argmax
    variance.
    """
    rng = np.random.default_rng(fold_seed)
    n_train = len(s_train)
    n_audit = len(s_audit)
    g_per_boot = np.empty((B, 2))
    for b in range(B):
        idx_t = rng.integers(0, n_train, size=n_train)
        idx_a = rng.integers(0, n_audit, size=n_audit)
        sw_t = None if sample_weight_train is None else np.asarray(sample_weight_train)[idx_t]
        sw_a = None if sample_weight_audit is None else np.asarray(sample_weight_audit)[idx_a]
        # Refit t̂ on bootstrap data (HT-weighted train, unweighted eval).
        _, t_b, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx_t], y_train[idx_t], s_audit[idx_a], alpha, grid_size,
            sample_weight_train=sw_t,
        )
        g_per_boot[b] = _build_g_vector(
            s_train[idx_t], y_train[idx_t],
            s_audit[idx_a], y_audit[idx_a],
            t_b, alpha,
            sample_weight_train=sw_t,
            sample_weight_audit=sw_a,
        )
    return g_per_boot


def two_moment_wald_audit_xf(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    t0: float,
    alpha: float,
    B: int = 100,
    fold_seed: int = 0,
    wald_alpha: float = 0.05,
    *,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_audit: np.ndarray | None = None,
) -> dict:
    """Joint two-moment Wald audit. 2-df chi-square.

    Tests the null E[(g₁, g₂)] = 0 jointly, where
        g₁_i = 1{y_i ≤ t̂} − α                            (tail-mass deviation)
        g₂_i = (t̂ − y_i)_+ − ĝ_t̂(s_i)                    (stop-loss residual)
    Σ̂ comes from a paired bootstrap of (train, audit) with t̂ re-maximized
    per rep — the appendix-(viii) fix. A naive Σ̂ that fixes t̂ at the
    point estimate over-rejects (~0.50 size at the truest null) because it
    ignores the variance from re-finding the argmax. Re-maximizing inside
    each bootstrap rep gives empirical size ~0.05 at the null.

    A small ridge (1/n_audit) is added to Σ̂ for numerical stability when
    g₂ is highly correlated with g₁ (which happens at small audit slices).
    Returns the Wald statistic, p-value, reject flag, and the per-moment
    means (so callers can report them alongside the verdict).
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)

    gbar = _build_g_vector(
        s_train, y_train, s_audit, y_audit, t0, alpha,
        sample_weight_train=sample_weight_train,
        sample_weight_audit=sample_weight_audit,
    )
    g_per_boot = _bootstrap_g_vectors(
        s_train, y_train, s_audit, y_audit, alpha,
        sample_weight_train=sample_weight_train,
        sample_weight_audit=sample_weight_audit,
        B=B, fold_seed=fold_seed,
    )
    Sigma_boot = np.cov(g_per_boot, rowvar=False, ddof=1)
    eps = max(1e-6, 1.0 / max(len(s_audit), 1))
    Sigma_boot = Sigma_boot + eps * np.eye(2)
    try:
        Sigma_inv = np.linalg.inv(Sigma_boot)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_boot)
    wald = float(gbar @ Sigma_inv @ gbar)
    p_value = float(1.0 - stats.chi2.cdf(wald, df=2))
    return {
        "audit_variant": "two_moment",
        "t0": float(t0),
        "wald_stat": wald,
        "p_value": p_value,
        "reject": bool(p_value < wald_alpha),
        "mean_g1": float(gbar[0]),
        "mean_g2": float(gbar[1]),
    }


def _single_moment_audit(
    s_train, y_train, s_audit, y_audit, t0, alpha,
    *, moment_idx: int, variant_name: str,
    B=100, fold_seed=0, wald_alpha=0.05,
    sample_weight_train=None, sample_weight_audit=None,
) -> dict:
    """Shared body for g1_only and g2_only audits. moment_idx ∈ {0, 1}."""
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)

    gbar = _build_g_vector(
        s_train, y_train, s_audit, y_audit, t0, alpha,
        sample_weight_train=sample_weight_train,
        sample_weight_audit=sample_weight_audit,
    )
    g_per_boot = _bootstrap_g_vectors(
        s_train, y_train, s_audit, y_audit, alpha,
        sample_weight_train=sample_weight_train,
        sample_weight_audit=sample_weight_audit,
        B=B, fold_seed=fold_seed,
    )
    g_scalar = float(gbar[moment_idx])
    var_scalar = float(np.var(g_per_boot[:, moment_idx], ddof=1))
    var_scalar = max(var_scalar, 1.0 / max(len(s_audit), 1))  # ridge for stability
    wald = g_scalar * g_scalar / var_scalar
    p_value = float(1.0 - stats.chi2.cdf(wald, df=1))
    return {
        "audit_variant": variant_name,
        "t0": float(t0),
        "wald_stat": float(wald),
        "p_value": p_value,
        "reject": bool(p_value < wald_alpha),
        "mean_g1": float(gbar[0]),
        "mean_g2": float(gbar[1]),
    }


def g1_only_audit_xf(
    s_train, y_train, s_audit, y_audit, t0: float, alpha: float,
    B: int = 100, fold_seed: int = 0, wald_alpha: float = 0.05,
    *, sample_weight_train=None, sample_weight_audit=None,
) -> dict:
    """Single-moment audit on g1 = 1{y ≤ t̂} − α (tail-mass)."""
    return _single_moment_audit(
        s_train, y_train, s_audit, y_audit, t0, alpha,
        moment_idx=0, variant_name="g1_only",
        B=B, fold_seed=fold_seed, wald_alpha=wald_alpha,
        sample_weight_train=sample_weight_train,
        sample_weight_audit=sample_weight_audit,
    )


def g2_only_audit_xf(
    s_train, y_train, s_audit, y_audit, t0: float, alpha: float,
    B: int = 100, fold_seed: int = 0, wald_alpha: float = 0.05,
    *, sample_weight_train=None, sample_weight_audit=None,
) -> dict:
    """Single-moment audit on g2 = (t̂ − y)_+ − ĝ_t̂(s) (stop-loss residual)."""
    return _single_moment_audit(
        s_train, y_train, s_audit, y_audit, t0, alpha,
        moment_idx=1, variant_name="g2_only",
        B=B, fold_seed=fold_seed, wald_alpha=wald_alpha,
        sample_weight_train=sample_weight_train,
        sample_weight_audit=sample_weight_audit,
    )


AUDIT_VARIANTS = {
    "two_moment": two_moment_wald_audit_xf,
    "g1_only": g1_only_audit_xf,
    "g2_only": g2_only_audit_xf,
}


# ---------------------------------------------------------------------------
# Simple CVaR audit — no bootstrap, no chi-square; just the moments
# ---------------------------------------------------------------------------

def simple_cvar_audit(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    s_eval_full: np.ndarray,           # FULL target cheap scores — used for t̂
    alpha: float,
    *,
    sample_weight_train: np.ndarray | None = None,
    grid_size: int = 61,
    moment_tol: float = 0.05,
    override: bool = False,
    length_train: np.ndarray | None = None,
    length_audit: np.ndarray | None = None,
    length_eval_full: np.ndarray | None = None,
) -> dict:
    """Audit-gated CVaR transport diagnostic. At the t̂ chosen on the FULL
    target eval distribution, compute the two transport moments on the
    audit slice and emit a verdict + a gated level claim.

    g1_i = 1{y_audit_i ≤ t̂} − α          (tail-mass deviation at threshold)
    g2_i = (t̂ − y_audit_i)_+ − ĝ_t̂(s_audit_i)   (stop-loss residual transport)

    The function ALWAYS returns the diagnostic point estimate `cvar_est`
    along with the moments. The `level` field is the gated level claim:
    it equals `cvar_est` only when the audit passes the heuristic
    `max(|g1|,|g2|) ≤ moment_tol`, and is `None` otherwise. Pass
    `override=True` to force the level claim through even when the audit
    flags — used for diagnostic comparisons against truth, never for
    headline reporting.

    Why two moments? The CVaR estimator's saddle-point representation
    integrates `(t̂ − Y)_+ / α + t̂` over the target. For the integral to
    converge to the true CVaR, the calibrator-implied t̂ must satisfy
    BOTH (a) the cheap distribution puts the right α-mass below it
    (g₁ → 0) AND (b) the calibrator reconstructs the shortfall above the
    threshold (g₂ → 0). Either failing breaks transport.

    Why t̂ on s_eval_full and NOT s_audit? The estimand is CVaR over the
    target policy distribution, not over the audit slice. Picking t̂ on
    the small audit slice tests a different functional (audit-slice CVaR)
    and gives small-sample-noisy thresholds. We re-use the production t̂
    so the moments tested are exactly the moments the estimator commits to.
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)
    s_eval_full = np.asarray(s_eval_full)
    use_cov = (length_train is not None and length_audit is not None
               and length_eval_full is not None)

    cvar_pt, t_hat, _, _ = estimate_direct_cvar_isotonic(
        s_train, y_train, s_eval_full, alpha, grid_size,
        sample_weight_train=sample_weight_train,
        length_train=length_train if use_cov else None,
        length_eval=length_eval_full if use_cov else None,
    )
    z_train = np.maximum(t_hat - y_train, 0.0)
    if use_cov:
        pred_audit = fit_two_stage_tail_loss(
            s_train, length_train, z_train, s_audit, length_audit,
            sample_weight=sample_weight_train,
        )
    else:
        pred_audit = fit_isotonic_tail_loss(
            s_train, z_train, s_audit, sample_weight=sample_weight_train,
        )
    g1 = (y_audit <= t_hat).astype(float) - alpha
    g2 = np.maximum(t_hat - y_audit, 0.0) - pred_audit
    mean_g1 = float(g1.mean()); mean_g2 = float(g2.mean())
    g1_flag = abs(mean_g1) > moment_tol
    g2_flag = abs(mean_g2) > moment_tol
    if g1_flag and g2_flag:
        verdict = "FLAG_BOTH"
    elif g1_flag:
        verdict = "FLAG_TAIL"
    elif g2_flag:
        verdict = "FLAG_TRANSPORT"
    else:
        verdict = "PASS"
    level = float(cvar_pt) if (verdict == "PASS" or override) else None
    return {
        "t_hat": float(t_hat),
        "cvar_est": float(cvar_pt),  # diagnostic point estimate, always set
        "level": level,              # gated level claim: None unless PASS or override
        "mean_g1": mean_g1,
        "mean_g2": mean_g2,
        "verdict": verdict,
        "n_audit": int(len(y_audit)),
    }


def cvar_audit_analytical_se(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    t0: float,
    alpha: float,
    *,
    sample_weight_train: np.ndarray | None = None,
) -> dict:
    """Fixed-t̂ analytical SE for the CVaR audit moments (g1, g2).

    The CVaR analog of `mean_transport_audit`'s t-test SE: treats both the
    calibrator (fit on s_train, y_train with HT weights) and threshold t0
    as fixed, computes per-row sample standard deviation on the audit
    slice and divides by √n_audit. Returns the (mean, SE) for g1 and g2
    plus the sample covariance.

    Use this when you need the audit-noise SE without paying for the
    bootstrap (and without picking up argmax variance from re-maximizing
    t̂). It is NOT a valid hypothesis test for the audit verdict — it
    underestimates Var when t̂ is itself estimated. Use it for:

      * Var_audit accounting in the honest envelope cvar_se_total =
        sqrt(Var_cal + Var_audit) — the calibrator-resampling variance
        Var_cal already absorbs t̂-resampling noise via bootstrap_cvar_ci,
        so adding the fixed-t̂ Var_audit avoids double-counting.
      * α=1 sanity checks — when CVaR collapses to the mean, this SE
        agrees with the mean-CJE Wald SE up to numerical precision.

    For the actual audit hypothesis test, use `two_moment_wald_audit_xf`.
    The inline version of this is what `test_alpha1_audit_pvalue_agreement`
    uses to verify the t-test/Wald collapse at α=1.
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)
    n_a = int(len(y_audit))
    z_train = np.maximum(t0 - y_train, 0.0)
    pred_audit = fit_isotonic_tail_loss(
        s_train, z_train, s_audit, sample_weight=sample_weight_train,
    )
    g1 = (y_audit <= t0).astype(float) - alpha
    g2 = np.maximum(t0 - y_audit, 0.0) - pred_audit
    mean_g1 = float(g1.mean()); mean_g2 = float(g2.mean())
    if n_a < 2:
        nan = float("nan")
        return {"mean_g1": mean_g1, "mean_g2": mean_g2,
                "se_g1": nan, "se_g2": nan,
                "cov_g1g2": nan, "n_audit": n_a, "t0": float(t0)}
    se_g1 = float(g1.std(ddof=1) / np.sqrt(n_a))
    se_g2 = float(g2.std(ddof=1) / np.sqrt(n_a))
    # cov_g1g2 is the off-diagonal of the analytical Σ̂ for the audit-mean
    # vector, i.e. sample-cov(g1, g2) / n_audit.
    cov = float(np.cov(g1, g2, ddof=1)[0, 1] / n_a)
    return {
        "mean_g1": mean_g1, "mean_g2": mean_g2,
        "se_g1": se_g1, "se_g2": se_g2,
        "cov_g1g2": cov, "n_audit": n_a, "t0": float(t0),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI on cvar_est (eval-only variance)
# ---------------------------------------------------------------------------

def bootstrap_cvar_ci(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    *,
    sample_weight_train: np.ndarray | None = None,
    B: int = 500,
    seed: int = 42,
    grid_size: int = 61,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap CI on the Direct CVaR-CJE point estimate.

    Eval set (s_eval) is fixed (we want CVaR over the policy distribution,
    not the slice distribution). Train set (s_train, y_train) is resampled
    iid; HT weights propagate via index. Calibrator is refit and t̂
    re-optimized inside each rep so the bootstrap captures both calibrator
    and threshold variance.
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_eval = np.asarray(s_eval)
    n = len(s_train)
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    sw_arr = None if sample_weight_train is None else np.asarray(sample_weight_train, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sw_b = None if sw_arr is None else sw_arr[idx]
        cvar_b, _, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx], y_train[idx], s_eval, alpha, grid_size,
            sample_weight_train=sw_b,
        )
        boots[b] = cvar_b
    point, _, _, _ = estimate_direct_cvar_isotonic(
        s_train, y_train, s_eval, alpha, grid_size,
        sample_weight_train=sw_arr,
    )
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    var_eval = float(np.var(boots, ddof=1))
    return {
        "point": float(point),
        "ci_lo": lo,
        "ci_hi": hi,
        "var_eval": var_eval,
        "B": B,
        "boots": boots,
    }


# ---------------------------------------------------------------------------
# Full-pipeline bootstrap with selectable resample sources
# ---------------------------------------------------------------------------

def pipeline_bootstrap_cvar(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval_full: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    alpha: float,
    *,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_audit: np.ndarray | None = None,
    resample: tuple[str, ...] = ("train", "eval", "audit"),
    B: int = 500,
    seed: int = 42,
    grid_size: int = 61,
    ci: float = 0.95,
    idx_train_per_b: np.ndarray | None = None,
    idx_eval_per_b: np.ndarray | None = None,
    idx_audit_per_b: np.ndarray | None = None,
) -> dict:
    """Bootstrap engine that resamples logger / target-eval / target-audit
    sets per ``resample``, refits the calibrator, re-maximizes t̂, and
    computes both plug-in V̂ and augmented V̂_aug per replicate.

    On each rep b:
      - if "train" in resample: idx_t ~ Unif(n_train); else use all rows.
      - if "eval"  in resample: idx_e ~ Unif(n_eval);  else use all rows.
      - if "audit" in resample: idx_a ~ Unif(n_audit); else use all rows.
      - Refit isotonic stop-loss on (s_train[idx_t], y_train[idx_t], w[idx_t])
        for the threshold grid; re-optimize t̂_b on s_eval_full[idx_e].
      - V_plug_b = sup_t [t − mean ĝ_t(s_eval_full[idx_e])/α]
      - ḡ2_b    = (HT-)mean of (t̂_b − y_audit[idx_a])_+ − ĝ_t̂_b(s_audit[idx_a])
      - V_aug_b = V_plug_b + ḡ2_b

    Point estimates are computed on the unresampled data with the same
    machinery, so the centre matches the production estimator path.

    Returns:
        {
          "plug_boots": (B,) array of plug-in bootstrap V̂_b,
          "aug_boots":  (B,) array of augmented bootstrap V̂_aug_b,
          "plug_point": float, "aug_point": float, "gbar2_point": float,
          "t_hat_point": float,
          "var_plug": float, "var_aug": float,
          "ci_plug": (lo, hi), "ci_aug": (lo, hi),
          "B": int, "resample": tuple,
        }
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_eval_full = np.asarray(s_eval_full)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)
    n_t, n_e, n_a = len(s_train), len(s_eval_full), len(s_audit)
    sw_t_arr = None if sample_weight_train is None else np.asarray(sample_weight_train, dtype=float)
    sw_a_arr = None if sample_weight_audit is None else np.asarray(sample_weight_audit, dtype=float)
    valid = {"train", "eval", "audit"}
    bad = set(resample) - valid
    if bad:
        raise ValueError(f"resample contains invalid keys {bad}; allowed {valid}")
    do_t = "train" in resample
    do_e = "eval" in resample
    do_a = "audit" in resample

    def _gbar2(s_tr, y_tr, sw_tr, t_hat, s_au, y_au, sw_au):
        z_tr = np.maximum(t_hat - y_tr, 0.0)
        pred_au = fit_isotonic_tail_loss(s_tr, z_tr, s_au, sample_weight=sw_tr)
        resid = np.maximum(t_hat - y_au, 0.0) - np.asarray(pred_au)
        if sw_au is None:
            return float(resid.mean())
        w = np.asarray(sw_au, dtype=float)
        sw_sum = w.sum()
        if sw_sum <= 0:
            return float("nan")
        return float((w * resid).sum() / sw_sum)

    # Point estimates on unresampled data
    plug_point, t_hat_point, _, _ = estimate_direct_cvar_isotonic(
        s_train, y_train, s_eval_full, alpha, grid_size,
        sample_weight_train=sw_t_arr,
    )
    gbar2_point = _gbar2(
        s_train, y_train, sw_t_arr, t_hat_point,
        s_audit, y_audit, sw_a_arr,
    )
    aug_point = float(plug_point + gbar2_point)

    # If explicit per-rep indices are provided, use them (coupled-RNG mode).
    # Otherwise, generate fresh draws inside the loop using `seed` + `resample`.
    use_explicit = (idx_train_per_b is not None
                    or idx_eval_per_b is not None
                    or idx_audit_per_b is not None)
    if use_explicit:
        if idx_train_per_b is None:
            idx_train_per_b = np.broadcast_to(np.arange(n_t), (B, n_t))
        if idx_eval_per_b is None:
            idx_eval_per_b = np.broadcast_to(np.arange(n_e), (B, n_e))
        if idx_audit_per_b is None:
            idx_audit_per_b = np.broadcast_to(np.arange(n_a), (B, n_a))
        if idx_train_per_b.shape != (B, n_t):
            raise ValueError(f"idx_train_per_b shape {idx_train_per_b.shape} != ({B}, {n_t})")
        if idx_eval_per_b.shape != (B, n_e):
            raise ValueError(f"idx_eval_per_b shape {idx_eval_per_b.shape} != ({B}, {n_e})")
        if idx_audit_per_b.shape != (B, n_a):
            raise ValueError(f"idx_audit_per_b shape {idx_audit_per_b.shape} != ({B}, {n_a})")

    rng = np.random.default_rng(seed)
    plug_boots = np.empty(B)
    aug_boots = np.empty(B)
    t_hat_boots = np.empty(B)
    for b in range(B):
        if use_explicit:
            idx_t = idx_train_per_b[b]
            idx_e = idx_eval_per_b[b]
            idx_a = idx_audit_per_b[b]
        else:
            if do_t:
                idx_t = rng.integers(0, n_t, size=n_t)
            else:
                idx_t = np.arange(n_t)
            if do_e:
                idx_e = rng.integers(0, n_e, size=n_e)
            else:
                idx_e = np.arange(n_e)
            if do_a and n_a > 0:
                idx_a = rng.integers(0, n_a, size=n_a)
            else:
                idx_a = np.arange(n_a)
        sw_t_b = None if sw_t_arr is None else sw_t_arr[idx_t]
        sw_a_b = None if sw_a_arr is None else sw_a_arr[idx_a]

        plug_b, t_hat_b, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx_t], y_train[idx_t], s_eval_full[idx_e], alpha, grid_size,
            sample_weight_train=sw_t_b,
        )
        if n_a > 0:
            gbar2_b = _gbar2(
                s_train[idx_t], y_train[idx_t], sw_t_b, t_hat_b,
                s_audit[idx_a], y_audit[idx_a], sw_a_b,
            )
        else:
            gbar2_b = float("nan")
        plug_boots[b] = float(plug_b)
        aug_boots[b] = float(plug_b + (gbar2_b if gbar2_b == gbar2_b else 0.0))
        t_hat_boots[b] = float(t_hat_b)

    var_plug = float(np.var(plug_boots, ddof=1)) if B > 1 else float("nan")
    var_aug = float(np.var(aug_boots, ddof=1)) if B > 1 else float("nan")
    qlo = (1 - ci) / 2
    qhi = 1 - qlo
    return {
        "plug_boots": plug_boots,
        "aug_boots": aug_boots,
        "t_hat_boots": t_hat_boots,
        "n_unique_t_hat": int(len(np.unique(t_hat_boots))),
        "plug_point": float(plug_point),
        "aug_point": aug_point,
        "gbar2_point": float(gbar2_point),
        "t_hat_point": float(t_hat_point),
        "var_plug": var_plug,
        "var_aug": var_aug,
        "ci_plug": (float(np.quantile(plug_boots, qlo)), float(np.quantile(plug_boots, qhi))),
        "ci_aug": (float(np.quantile(aug_boots, qlo)), float(np.quantile(aug_boots, qhi))),
        "B": int(B),
        "resample": tuple(resample),
    }


def bootstrap_mean_ci(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    *,
    sample_weight_train: np.ndarray | None = None,
    B: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap CI on Direct Mean-CJE: V̂_mean = mean f̂(s_eval).

    Mirrors `bootstrap_cvar_ci` exactly — same paired-bootstrap design, same
    RNG path, eval set fixed, train resampled iid with HT weights via index.
    With identical (seed, B, sample_weight_train, s_eval), the per-rep
    bootstrap of `bootstrap_mean_ci` equals the per-rep bootstrap of
    `bootstrap_cvar_ci(..., alpha=1.0)` to machine precision (since at α=1
    with the patched grid each rep's cvar_b reduces to mean f̂_b(s_eval)).
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_eval = np.asarray(s_eval)
    n = len(s_train)
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    sw_arr = None if sample_weight_train is None else np.asarray(sample_weight_train, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sw_b = None if sw_arr is None else sw_arr[idx]
        f_hat_b = fit_isotonic_mean(
            s_train[idx], y_train[idx], s_eval, sample_weight=sw_b,
        )
        boots[b] = float(f_hat_b.mean())
    point = float(fit_isotonic_mean(
        s_train, y_train, s_eval, sample_weight=sw_arr,
    ).mean())
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    var_eval = float(np.var(boots, ddof=1))
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "var_eval": var_eval,
        "B": B,
        "boots": boots,
    }


def pipeline_bootstrap_mean(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval_full: np.ndarray,
    *,
    sample_weight_train: np.ndarray | None = None,
    resample: tuple[str, ...] = ("train", "eval"),
    B: int = 500,
    seed: int = 42,
    ci: float = 0.95,
    idx_train_per_b: np.ndarray | None = None,
    idx_eval_per_b: np.ndarray | None = None,
) -> dict:
    """Full-pipeline bootstrap for Direct Mean-CJE: V̂_mean = mean f̂(s_eval).

    Mirrors pipeline_bootstrap_cvar's structure for the mean estimand. On each
    rep b: optionally resample train, optionally resample eval, refit isotonic
    f̂_b on the resampled train rows (HT-weighted), and compute
    V̂_mean^(b) = mean f̂_b(s_eval[idx_e^(b)]).

    With resample=("train",) and identical (seed, B), reduces to
    bootstrap_mean_ci exactly. With resample=("train","eval") it also
    captures the eval Monte-Carlo contribution — symmetric counterpart to
    pipeline_bootstrap_cvar's full-pipeline mode.

    Returns: {boots, point, var_eval, ci_lo, ci_hi, B, resample}.
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_eval_full = np.asarray(s_eval_full)
    n_t = len(s_train); n_e = len(s_eval_full)
    sw_t_arr = None if sample_weight_train is None else np.asarray(sample_weight_train, dtype=float)
    valid = {"train", "eval"}
    bad = set(resample) - valid
    if bad:
        raise ValueError(f"resample contains invalid keys {bad}; allowed {valid}")
    do_t = "train" in resample
    do_e = "eval" in resample

    use_explicit = (idx_train_per_b is not None or idx_eval_per_b is not None)
    if use_explicit:
        if idx_train_per_b is None:
            idx_train_per_b = np.broadcast_to(np.arange(n_t), (B, n_t))
        if idx_eval_per_b is None:
            idx_eval_per_b = np.broadcast_to(np.arange(n_e), (B, n_e))

    point = float(fit_isotonic_mean(
        s_train, y_train, s_eval_full, sample_weight=sw_t_arr,
    ).mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        if use_explicit:
            idx_t = idx_train_per_b[b]; idx_e = idx_eval_per_b[b]
        else:
            idx_t = rng.integers(0, n_t, size=n_t) if do_t else np.arange(n_t)
            idx_e = rng.integers(0, n_e, size=n_e) if do_e else np.arange(n_e)
        sw_t_b = None if sw_t_arr is None else sw_t_arr[idx_t]
        f_hat_b = fit_isotonic_mean(
            s_train[idx_t], y_train[idx_t], s_eval_full[idx_e],
            sample_weight=sw_t_b,
        )
        boots[b] = float(f_hat_b.mean())
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return {
        "boots": boots, "point": point,
        "ci_lo": lo, "ci_hi": hi,
        "var_eval": float(np.var(boots, ddof=1)) if B > 1 else float("nan"),
        "B": int(B), "resample": tuple(resample),
    }


# ---------------------------------------------------------------------------
# Calibration-aware variance via delete-one-fold jackknife on oracle slice
# ---------------------------------------------------------------------------

def jackknife_var_cal(
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    *,
    sample_weight_oracle: np.ndarray | None = None,
    K: int = 5,
    seed: int = 42,
    grid_size: int = 61,
) -> float:
    """Variance contribution from re-fitting the calibrator on K folds of
    the oracle slice with one fold removed each time.

    Standard delete-one-group jackknife formula (Efron–Stein):
        Var_cal = ((K-1)/K) * Σ_k (cvar_(-k) − cvar_mean)²

    HT weights propagate via mask. Returns scalar variance contribution.
    """
    s_oracle = np.asarray(s_oracle); y_oracle = np.asarray(y_oracle)
    s_eval = np.asarray(s_eval)
    n = len(s_oracle)
    if n < K + 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    for i, p in enumerate(perm):
        fold_id[p] = i % K
    sw_arr = None if sample_weight_oracle is None else np.asarray(sample_weight_oracle, dtype=float)
    estimates = np.empty(K)
    for k in range(K):
        mask = fold_id != k
        sw_k = None if sw_arr is None else sw_arr[mask]
        cvar_k, _, _, _ = estimate_direct_cvar_isotonic(
            s_oracle[mask], y_oracle[mask], s_eval, alpha, grid_size,
            sample_weight_train=sw_k,
        )
        estimates[k] = cvar_k
    return float(((K - 1) / K) * np.sum((estimates - estimates.mean()) ** 2))


def jackknife_var_cal_mean(
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    s_eval: np.ndarray,
    *,
    sample_weight_oracle: np.ndarray | None = None,
    K: int = 5,
    seed: int = 42,
) -> float:
    """Calibration-aware variance for Direct Mean-CJE.

    Mirrors `jackknife_var_cal` exactly (same K, same `seed` permutation,
    same round-robin fold assignment via `fold_id != k`), so with
    identical inputs and α=1 each held-out fold's cvar_(-k) reduces to
    mean f̂_(-k)(s_eval) and the two scalars agree to machine precision.

        Var_cal = ((K-1)/K) * Σ_k (mean_(-k) − mean_mean)²
    """
    s_oracle = np.asarray(s_oracle); y_oracle = np.asarray(y_oracle)
    s_eval = np.asarray(s_eval)
    n = len(s_oracle)
    if n < K + 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=int)
    for i, p in enumerate(perm):
        fold_id[p] = i % K
    sw_arr = None if sample_weight_oracle is None else np.asarray(sample_weight_oracle, dtype=float)
    estimates = np.empty(K)
    for k in range(K):
        mask = fold_id != k
        sw_k = None if sw_arr is None else sw_arr[mask]
        f_hat_k = fit_isotonic_mean(
            s_oracle[mask], y_oracle[mask], s_eval, sample_weight=sw_k,
        )
        estimates[k] = float(f_hat_k.mean())
    return float(((K - 1) / K) * np.sum((estimates - estimates.mean()) ** 2))


# ---------------------------------------------------------------------------
# Mean transport audit (per-π' test of E[Y − f̂(S)] = 0)
# ---------------------------------------------------------------------------

def mean_transport_audit(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    *,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_audit: np.ndarray | None = None,
    wald_alpha: float = 0.05,
    length_train: np.ndarray | None = None,
    length_audit: np.ndarray | None = None,
) -> dict:
    """One-sample t-test of E_{π'}[Y − f̂(S, length)] = 0 on the target policy's
    oracle slice. f̂ fit on the training oracle slice with HT weights, applied
    to the audit slice. Residuals ε = y_audit − f̂(s_audit).

    When `length_train` and `length_audit` are both provided, uses
    `fit_two_stage_calibrator` (CJE's `direct+cov` calibrator). Otherwise
    falls back to `fit_isotonic_mean` (S-only).

    If sample_weight_audit is None, plain one-sample t-test. Otherwise
    weighted-t with effective sample size n_eff = (Σw)² / Σw².
    """
    from scipy import stats
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)
    if length_train is not None and length_audit is not None:
        pred = fit_two_stage_calibrator(
            s_train, length_train, y_train, s_audit, length_audit,
            sample_weight=sample_weight_train,
        )
    else:
        pred = fit_isotonic_mean(s_train, y_train, s_audit, sample_weight=sample_weight_train)
    eps = y_audit - pred

    if sample_weight_audit is None:
        if len(eps) < 2:
            return {"residual_mean": float("nan"), "t_stat": float("nan"),
                    "p_value": float("nan"), "reject": False, "n_audit": int(len(eps))}
        t_res = stats.ttest_1samp(eps, 0.0)
        return {
            "residual_mean": float(eps.mean()),
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "reject": bool(t_res.pvalue < wald_alpha),
            "n_audit": int(len(eps)),
        }

    w = np.asarray(sample_weight_audit, dtype=float)
    sum_w = w.sum()
    if sum_w <= 0 or len(eps) < 2:
        return {"residual_mean": float("nan"), "t_stat": float("nan"),
                "p_value": float("nan"), "reject": False, "n_audit": int(len(eps))}
    mu = float((w * eps).sum() / sum_w)
    var = float((w * (eps - mu) ** 2).sum() / sum_w)
    n_eff = float(sum_w ** 2 / (w ** 2).sum())
    if var <= 0 or n_eff <= 1:
        return {"residual_mean": mu, "t_stat": float("nan"),
                "p_value": float("nan"), "reject": False, "n_audit": int(len(eps))}
    t_stat = mu / np.sqrt(var / n_eff)
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(1.0, n_eff - 1))))
    return {
        "residual_mean": mu,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "reject": bool(p_value < wald_alpha),
        "n_audit": int(len(eps)),
        "n_eff": n_eff,
    }
