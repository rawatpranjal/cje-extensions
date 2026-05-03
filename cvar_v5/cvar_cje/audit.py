"""
Two-moment Wald audit for the Direct CVaR-CJE estimator.

Math contract (paper: cvar_v4/sections/method.tex:67-94, eq:audit, eq:gate;
and appendix_c_audit.tex):

    On the AUDIT slice (held out from CALIB), evaluate two moments at the
    selected threshold t̂_α:

        g_1(O; t) = 1{Y ≤ t} − α                       (tail-mass deviation)
        g_2(O; t) = (t − Y)_+ − ĥ_t(s)                 (shortfall transport)

        ḡ = (mean g_1, mean g_2)  on AUDIT

    Wald statistic with χ²_2 limit:

        W_n = ḡᵀ · Var̂(ḡ)⁻¹ · ḡ

    Reject (REFUSE-Level) iff  W_n > χ²_{2, 1−η}.

    NOTE on units: the spec's expression `W_n = n_audit · ḡᵀ Ω̂⁻¹ ḡ` uses
    per-row covariance Ω̂ = n_audit · Var̂(ḡ). We work in `Var̂(ḡ)` directly
    (the covariance of the audit-slice mean) for parity across all four
    variants below; the resulting W_n is identical.

α=1 reduction:
    For Y ∈ [0,1] and t̂_1 = 1:
        g_1 ≡ 0       (since 1{Y ≤ 1} = 1 for all rows, minus α=1)
        g_2 = (1 − Y) − ĥ_1(s) = (1 − Y) − (1 − f̂(s)) = f̂(s) − Y
              (the negative mean-CJE residual)
    So at α=1 the two-moment audit reduces algebraically to the mean
    transport audit. Tested in audit_test.py::test_alpha_one_reduces_to_mean.

Ω̂ ESTIMATORS (research-time variants, selected via Config.omega_estimator):

    "analytical":
        Var̂(ḡ) = sample_cov_per_row(g_1, g_2) / n_audit, with t̂ FIXED.
        Cheap; underestimates variance because it captures only audit-side
        variance, missing the calibrator-fit nuisance.

    "analytical_oua":
        Var̂(ḡ) = analytical Ω̂_audit + jackknife V̂_cal on (g_1, g_2).
        Per-fold k: refit ĥ_t^(−k) (already cached in cg._folded), apply to
        AUDIT, recompute ḡ^(−k), accumulate
            V̂_cal_g = ((K-1)/K) · Σ_k (ḡ^(−k) - ḡ̄)(ḡ^(−k) - ḡ̄)ᵀ
        Cheap (K refits already cached). Captures the calibrator-fit
        nuisance the analytical variant misses. See `mean_cje/README.md`
        for evidence that this jackknife is the load-bearing piece for
        going from undercovering CIs (33%) to nominal coverage (~90%).
        Does NOT capture the argmax-on-grid nuisance.

    "boot_remax_ridge":
        Paired bootstrap of AUDIT rows (with replacement). For each rep,
        re-maximize t̂_b over the calibrator's grid using the bootstrapped
        AUDIT slice as the eval-stand-in (calibrator FIXED at the pooled fit).
        Compute g_b at t̂_b. Var̂(ḡ) = empirical covariance of {g_b} + ridge.
        Ridge = (1/n_audit) · I  — paper-canonical for numerical stability;
        empirical fact (memory: project_audit_xf_fix) is that the naive
        non-ridge variant over-rejects at ~0.50 size due to non-smooth argmax.
        At small n_audit the ridge dominates real moment variance and zero-
        powers the audit; tracked in TODO.md [omega-research].

    "boot_remax_no_ridge":
        As above, no ridge added. Tests whether ridge is the cause of the
        zero-power finding at small n_audit.

    "boot_fixed":
        Paired bootstrap of AUDIT rows; t̂ FIXED at the original (no re-max).
        Asymptotically equivalent to "analytical"; included for diagnostic
        parity.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

from .calibrator import CalibratorGrid
from .config import OmegaEstimator
from .schema import AuditVerdict, Slice


AuditMoments = Literal["g1g2", "g2_only"]


# ----- shared helpers ----------------------------------------------------------


def _t_index(t_grid: np.ndarray, t: float) -> int:
    """Locate t in the calibrator grid (require near-exact match)."""
    j = int(np.argmin(np.abs(t_grid - t)))
    if abs(float(t_grid[j]) - t) > 1e-12:
        raise ValueError(
            f"t_hat={t} is not on the calibrator grid; nearest is {t_grid[j]}"
        )
    return j


def _g_pair_fixed_t(
    s: np.ndarray,
    y: np.ndarray,
    h_t: np.ndarray,
    t: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row g_1 and g_2 vectors at a fixed threshold t, given pre-computed
    ĥ_t(s) values.
    """
    g1 = (y <= t).astype(np.float64) - alpha
    g2 = np.maximum(t - y, 0.0) - h_t
    return g1, g2


def _argmax_t_on_grid(
    H: np.ndarray, t_grid: np.ndarray, alpha: float
) -> tuple[int, float]:
    """
    Re-maximize t̂_α = argmax_t {t − (1/(α n)) Σ_i ĥ_t(s_i)} over the grid.

    H has shape (n, |T|). Returns (j_star, t_star).
    """
    psi = t_grid - H.mean(axis=0) / alpha
    j_star = int(np.argmax(psi))
    return j_star, float(t_grid[j_star])


# ----- Ω̂ estimators -----------------------------------------------------------


def _omega_analytical(
    s: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    t_idx: int,
    t_hat: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Var̂(ḡ) via the per-row sample covariance at fixed t̂.

    Returns (omega, g_bar).
    """
    h_t = H[:, t_idx]
    g1, g2 = _g_pair_fixed_t(s, y, h_t, t_hat, alpha)
    g_bar = np.array([g1.mean(), g2.mean()])
    cov_per_row = np.cov(np.vstack([g1, g2]), ddof=1)  # 2×2
    omega = cov_per_row / len(s)
    return omega, g_bar


def _omega_analytical_oua(
    s: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    t_idx: int,
    t_hat: float,
    alpha: float,
    calibrator: "CalibratorGrid",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical Ω̂_audit + delete-one-fold jackknife V̂_cal on (g_1, g_2).

    Math:
        Ω̂_audit  =  per-row sample cov of (g_1, g_2)  /  n_audit
        For each fold k = 1..K:
            ĥ_t^(−k)  :=  cg._folded[t_idx][k]      (already fit on CALIB \\ fold_k)
            ḡ^(−k)   :=  audit-slice mean of (g_1, g_2)  using ĥ_t^(−k)
        V̂_cal_g  =  ((K − 1) / K) · Σ_k (ḡ^(−k) − ḡ̄)(ḡ^(−k) − ḡ̄)ᵀ
        Ω̂        =  Ω̂_audit + V̂_cal_g

    Captures BOTH audit-side and calibrator-fit variance — the latter is
    what the original 4 variants miss. See `mean_cje/README.md` for the
    Mean-CJE evidence that this jackknife is the load-bearing piece.

    Returns (omega, g_bar). g_bar is computed at the original (pooled) ĥ_t.
    """
    omega_audit, g_bar = _omega_analytical(s, y, H, t_idx, t_hat, alpha)

    if calibrator.n_folds < 2:
        raise ValueError(
            "analytical_oua requires a cross-fit calibrator with K ≥ 2; "
            f"got n_folds={calibrator.n_folds}"
        )
    folded_at_t = calibrator._folded[t_idx]
    g_per_fold = np.empty((calibrator.n_folds, 2), dtype=np.float64)
    for k, ir_k in enumerate(folded_at_t):
        h_t_k = ir_k.predict(s)
        g1_k, g2_k = _g_pair_fixed_t(s, y, h_t_k, t_hat, alpha)
        g_per_fold[k, 0] = g1_k.mean()
        g_per_fold[k, 1] = g2_k.mean()

    g_bar_jk = g_per_fold.mean(axis=0)
    diffs = g_per_fold - g_bar_jk
    K = calibrator.n_folds
    v_cal = ((K - 1) / K) * (diffs.T @ diffs)
    return omega_audit + v_cal, g_bar


def _bootstrap_omega(
    s: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,                 # shape (n_audit, |T|), pre-computed once
    t_grid: np.ndarray,
    t_idx_orig: int,
    t_hat_orig: float,
    alpha: float,
    B: int,
    seed: int,
    remax: bool,
    ridge: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Paired bootstrap of AUDIT rows. Calibrator stays fixed (we precomputed
    H = ĥ_grid(s_audit) outside). t̂ is optionally re-maxed per rep using
    the bootstrapped audit as the eval-stand-in.

    Returns (omega, g_bar). g_bar is computed on the ORIGINAL (non-bootstrapped)
    audit at the ORIGINAL t̂ — the bootstrap is only used to estimate Var̂(ḡ).
    """
    n = len(s)
    rng = np.random.default_rng(seed)

    # Original ḡ at original t̂.
    h_t_orig = H[:, t_idx_orig]
    g1_o, g2_o = _g_pair_fixed_t(s, y, h_t_orig, t_hat_orig, alpha)
    g_bar = np.array([g1_o.mean(), g2_o.mean()])

    g_b = np.empty((B, 2), dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        s_b = s[idx]
        y_b = y[idx]
        H_b = H[idx, :]
        if remax:
            j_b, t_b = _argmax_t_on_grid(H_b, t_grid, alpha)
        else:
            j_b, t_b = t_idx_orig, t_hat_orig
        h_t_b = H_b[:, j_b]
        g1_b, g2_b = _g_pair_fixed_t(s_b, y_b, h_t_b, t_b, alpha)
        g_b[b, 0] = g1_b.mean()
        g_b[b, 1] = g2_b.mean()

    omega = np.cov(g_b, rowvar=False, ddof=1)
    if ridge:
        # Paper-canonical floor. Same units as Var̂(ḡ). At small n_audit this
        # dominates real moment variance — known issue, tracked in TODO.md
        # [omega-research].
        omega = omega + (1.0 / n) * np.eye(2)
    return omega, g_bar


# ----- public API --------------------------------------------------------------


def two_moment_wald_audit(
    audit_slice: Slice,
    calibrator: CalibratorGrid,
    t_hat: float,
    alpha: float,
    omega_estimator: OmegaEstimator,
    B: int,
    seed: int,
    eta: float = 0.05,
    moments: AuditMoments = "g1g2",
) -> AuditVerdict:
    """
    Joint two-moment Wald audit. χ²_2 limit (or χ²_1 if moments="g2_only").

    Args:
        moments:
            "g1g2"    (default) - the canonical 2-moment Wald, χ²_2.
            "g2_only" - drop g_1 (tail-mass deviation), test only the
                        shortfall transport residual g_2 against χ²_1.
                        Useful when audit-side argmax noise in g_1 is
                        inflating size.

    Returns an AuditVerdict.
    """
    if audit_slice.role != "audit":
        raise ValueError(
            f"audit_slice.role must be 'audit'; got {audit_slice.role!r}"
        )

    s = audit_slice.s()
    y = audit_slice.y()
    if len(s) == 0:
        raise ValueError("audit_slice is empty")
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(
            f"audit y must be in [0, 1]; got [{y.min()}, {y.max()}]"
        )

    t_grid = calibrator.t_grid
    t_idx = _t_index(t_grid, t_hat)
    H = calibrator.predict(s)  # shape (n_audit, |T|)

    if omega_estimator == "analytical":
        omega, g_bar = _omega_analytical(s, y, H, t_idx, t_hat, alpha)
    elif omega_estimator == "analytical_oua":
        omega, g_bar = _omega_analytical_oua(
            s, y, H, t_idx, t_hat, alpha, calibrator,
        )
    elif omega_estimator == "boot_remax_ridge":
        omega, g_bar = _bootstrap_omega(
            s, y, H, t_grid, t_idx, t_hat, alpha, B, seed,
            remax=True, ridge=True,
        )
    elif omega_estimator == "boot_remax_no_ridge":
        omega, g_bar = _bootstrap_omega(
            s, y, H, t_grid, t_idx, t_hat, alpha, B, seed,
            remax=True, ridge=False,
        )
    elif omega_estimator == "boot_remax_oua":
        # Layer jackknife V̂_cal on top of bootstrap-with-remax.
        # Captures audit-side variance + argmax nuisance (bootstrap)
        # AND calibrator-fit nuisance (jackknife). Closest analog to
        # mean_cje's "bootstrap + jackknife" stack.
        omega_boot, g_bar = _bootstrap_omega(
            s, y, H, t_grid, t_idx, t_hat, alpha, B, seed,
            remax=True, ridge=False,
        )
        # Reuse the analytical_oua jackknife to add V̂_cal_g.
        # We discard its g_bar (matches the bootstrap one) and take only V̂_cal.
        if calibrator.n_folds < 2:
            raise ValueError(
                "boot_remax_oua requires a cross-fit calibrator with K ≥ 2"
            )
        folded_at_t = calibrator._folded[t_idx]
        g_per_fold = np.empty((calibrator.n_folds, 2), dtype=np.float64)
        for k, ir_k in enumerate(folded_at_t):
            h_t_k = ir_k.predict(s)
            g1_k, g2_k = _g_pair_fixed_t(s, y, h_t_k, t_hat, alpha)
            g_per_fold[k, 0] = g1_k.mean()
            g_per_fold[k, 1] = g2_k.mean()
        diffs = g_per_fold - g_per_fold.mean(axis=0)
        K = calibrator.n_folds
        v_cal = ((K - 1) / K) * (diffs.T @ diffs)
        omega = omega_boot + v_cal
    elif omega_estimator == "boot_fixed":
        omega, g_bar = _bootstrap_omega(
            s, y, H, t_grid, t_idx, t_hat, alpha, B, seed,
            remax=False, ridge=False,
        )
    else:
        raise ValueError(f"unknown omega_estimator: {omega_estimator!r}")

    # Wald statistic.
    if moments == "g1g2":
        # 2-moment, χ²_2. pinv for safety against near-singular Ω̂.
        omega_inv = np.linalg.pinv(omega)
        W_n = float(g_bar @ omega_inv @ g_bar)
        df = 2
    elif moments == "g2_only":
        # 1-moment on g_2 alone. Σ̂_22 is the (1,1) entry of the 2x2 Ω̂.
        var_g2 = float(omega[1, 1])
        if var_g2 <= 0:
            raise ValueError(f"Σ̂_22 must be > 0 for g2_only audit; got {var_g2}")
        W_n = float(g_bar[1] ** 2 / var_g2)
        df = 1
    else:
        raise ValueError(f"unknown moments: {moments!r}")
    p_value = float(1.0 - stats.chi2.cdf(W_n, df=df))
    decision = "REFUSE-LEVEL" if p_value < eta else "PASS"

    return AuditVerdict(
        W_n=W_n,
        p_value=p_value,
        g1=float(g_bar[0]),
        g2=float(g_bar[1]),
        omega_estimator=omega_estimator,
        decision=decision,
    )
