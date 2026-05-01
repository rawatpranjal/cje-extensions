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
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Direct CVaR estimator — grid search over stop-loss thresholds.

    The isotonic stop-loss calibrator is fit on s_train with HT weights
    (sample_weight_train, e.g. 1/π_i). The evaluation mean over s_eval is
    UNWEIGHTED — we want CVaR_α of the policy-π' distribution, not of the
    slice distribution.

    Returns (cvar_estimate, t_hat, t_grid, objective).
    """
    t_grid = make_t_grid(y_train, alpha, grid_size)
    objective = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        z_train = np.maximum(t - y_train, 0.0)
        pred_eval = fit_isotonic_tail_loss(
            s_train, z_train, s_eval,
            sample_weight=sample_weight_train,
        )
        objective[i] = float(t - pred_eval.mean() / alpha)
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
    """Joint two-moment Wald audit. 2-df chi-square."""
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
) -> dict:
    """Minimal CVaR transport diagnostic. At the t̂ chosen on the FULL target
    eval distribution, compute the two transport moments on the audit slice.

    No bootstrap, no chi-square Wald, no joint vs single-moment variants.
    Just two numbers + a heuristic verdict left to the caller.

    g1_i = 1{y_audit_i ≤ t̂} − α          (tail-mass deviation at threshold)
    g2_i = (t̂ − y_audit_i)_+ − ĝ_t̂(s_audit_i)   (stop-loss residual transport)

    The threshold t̂ is selected on s_eval_full — the full target policy
    distribution — NOT on the small audit slice. Selecting t̂ on the audit
    slice tests a different statistical functional (issue #3 in the review).
    """
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)
    s_eval_full = np.asarray(s_eval_full)

    cvar_pt, t_hat, _, _ = estimate_direct_cvar_isotonic(
        s_train, y_train, s_eval_full, alpha, grid_size,
        sample_weight_train=sample_weight_train,
    )
    z_train = np.maximum(t_hat - y_train, 0.0)
    pred_audit = fit_isotonic_tail_loss(
        s_train, z_train, s_audit, sample_weight=sample_weight_train,
    )
    g1 = (y_audit <= t_hat).astype(float) - alpha
    g2 = np.maximum(t_hat - y_audit, 0.0) - pred_audit
    return {
        "t_hat": float(t_hat),
        "cvar_est": float(cvar_pt),
        "mean_g1": float(g1.mean()),
        "mean_g2": float(g2.mean()),
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
    t̂). The inline version of this is what `test_alpha1_audit_pvalue_agreement`
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
) -> dict:
    """One-sample t-test of E_{π'}[Y − f̂(S)] = 0 on the target policy's
    oracle slice. f̂ fit on (s_train, y_train) with HT weights, applied to
    s_audit. Residuals ε = y_audit − f̂(s_audit).

    If sample_weight_audit is None, plain one-sample t-test. Otherwise
    weighted-t with effective sample size n_eff = (Σw)² / Σw².
    """
    from scipy import stats
    s_train = np.asarray(s_train); y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit); y_audit = np.asarray(y_audit)
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
