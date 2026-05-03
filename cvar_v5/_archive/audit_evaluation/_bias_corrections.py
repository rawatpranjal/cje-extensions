"""
Three bias-correction methods for the audit moment vector ḡ.

Setup
-----
Under H_0 (population transport), the audit's null is `E[g(t*; h*)] = 0`,
where (h*, t*) is the population truth. At finite (n_calib, n_eval), the
realized estimators (ĥ, t̂) carry sampling fluctuation, so

    E[ ḡ_realized ]  =  E[ g(t̂; ĥ) ]  =:  ε  ≠  0.

The Wald statistic centered at zero, even with the correct Ω̂, sees a
non-central χ²_2 distribution with non-centrality

    λ  =  n_audit · ε^T Ω̂^{-1} ε.

Bias correction subtracts an estimate of ε from ḡ before forming W:

    ḡ_bc  :=  ḡ  −  ε̂.

Three estimators ε̂ are implemented, in order of increasing scope:

    BC-jk-cal   varies only ĥ across oracle folds (status quo, paper-canonical).
    BC-jk-full  varies BOTH ĥ AND t̂^(-k), per fold (NEW; needed to reach g_1).
    BC-boot     full-pipeline bootstrap; resamples calib + eval + audit (NEW).

All three are deterministic given a seed and an input RepResult.

Notation
--------
    K            number of oracle folds in the cross-fit calibrator
    ĥ            pooled calibrator (full-data fit)
    ĥ^(-k)       leave-one-fold-out calibrator (cal.folded[j][k])
    t̂            argmax_t [t − mean_eval(ĥ_t(s_e)) / α]   (pooled)
    t̂^(-k)       argmax_t [t − mean_eval(ĥ_t^(-k)(s_e)) / α]  (per fold)
    ḡ            (1/n_audit) Σ_i  g_i(t̂; ĥ)                          ∈ R²
    ḡ^(-k)_A     (1/n_audit) Σ_i  g_i(t̂; ĥ^(-k))                     (BC-jk-cal)
    ḡ^(-k)_B     (1/n_audit) Σ_i  g_i(t̂^(-k); ĥ^(-k))                (BC-jk-full)
    ḡ^(b)        bootstrap-rep audit moment from the full-pipeline bootstrap

Standard delete-one-fold jackknife bias-corrected estimator:
    ḡ_bc  =  K · ḡ  −  (K − 1) · mean_k( ḡ^(-k) )

Equivalently:
    ε̂_jk  =  (K − 1) · ( mean_k(ḡ^(-k)) − ḡ )
    ḡ_bc   =  ḡ  −  ε̂_jk
"""

from __future__ import annotations

import numpy as np

from ._harness import (
    Calibrator, RepResult, fit_calibrator, g_pair, saddle_argmax,
)


# ---------------- BC-jk-cal: jackknife on calibrator only -----------------


def bc_jk_cal(rep: RepResult) -> np.ndarray:
    """
    Bias-correct ḡ using the delete-one-fold jackknife with t̂ FIXED at
    the pooled t̂.

    Math
    ----
        ε̂_A   =  (K − 1) · ( mean_k(ḡ^(-k)_A)  −  ḡ )
        ḡ_bc_A =  ḡ  −  ε̂_A
                =  K · ḡ  −  (K − 1) · mean_k(ḡ^(-k)_A)

    Where ḡ^(-k)_A = (1/n_audit) Σ_i g_i(t̂_pooled; ĥ^(-k)) is computed at
    the original t̂. g_1 = 1{Y ≤ t̂_pooled} − α doesn't depend on ĥ, so

        ḡ^(-k)_A[0]  =  ḡ[0]    for every k

    and the bias correction is identically zero on g_1. Only g_2 gets
    corrected.

    Returns ḡ_bc as a 2-vector.
    """
    K = rep.g_per_fold_at_t_hat.shape[0]
    g_jk = rep.g_per_fold_at_t_hat.mean(axis=0)
    return K * rep.ḡ - (K - 1) * g_jk


# ---------------- BC-jk-full: jackknife on calibrator AND threshold -------


def bc_jk_full(rep: RepResult) -> np.ndarray:
    """
    Bias-correct ḡ using the delete-one-fold jackknife where BOTH ĥ AND t̂
    vary across folds.

    Math
    ----
        For each fold k = 1..K:
            ĥ^(-k)        cached on cal.folded
            t̂^(-k)        argmax_t [t − mean_eval(ĥ_t^(-k)(s_e)) / α]
            ḡ^(-k)_B  =  (1/n_audit) Σ_i  g_i(t̂^(-k); ĥ^(-k))

        ε̂_B   =  (K − 1) · ( mean_k(ḡ^(-k)_B) − ḡ )
        ḡ_bc_B =  K · ḡ  −  (K − 1) · mean_k(ḡ^(-k)_B)

    Why this differs from BC-jk-cal
    -------------------------------
    g_1 = 1{Y ≤ t̂} − α DOES depend on t̂. When t̂^(-k) shifts across folds,
    so does g_1^(-k). So mean_k(ḡ^(-k)_B[0]) ≠ ḡ[0] in general, giving a
    non-trivial bias correction on g_1. This is the piece that BC-jk-cal
    misses.

    Returns ḡ_bc as a 2-vector.
    """
    K = rep.g_per_fold_at_t_k.shape[0]
    g_jk = rep.g_per_fold_at_t_k.mean(axis=0)
    return K * rep.ḡ - (K - 1) * g_jk


# ---------------- BC-boot: full-pipeline bootstrap ------------------------


def bc_boot(
    rep: RepResult,
    t_grid: np.ndarray,
    alpha: float,
    K: int,
    B: int,
    seed: int,
) -> np.ndarray:
    """
    Bias-correct ḡ using a full-pipeline bootstrap of (CALIB, EVAL, AUDIT).

    Math
    ----
        For b = 1..B:
            idx_c^(b)   ~ Multinomial(n_calib, 1/n_calib · 1)
            idx_e^(b)   ~ Multinomial(n_eval,  1/n_eval  · 1)
            idx_a^(b)   ~ Multinomial(n_audit, 1/n_audit · 1)

            ĥ^(b)        := refit calibrator on (s_c[idx_c], y_c[idx_c])
            t̂^(b)        := argmax_t [t − mean_{eval^(b)}(ĥ^(b)_t(s_e^(b))) / α]
            ḡ^(b)        := (1/n_audit) Σ_i  g_i(t̂^(b); ĥ^(b))
                            with i ranging over the bootstrapped audit slice

        bias_boot   :=  mean_b(ḡ^(b))  −  ḡ
        ε̂_C          :=  bias_boot
        ḡ_bc_C       :=  ḡ  −  bias_boot
                      =  2 · ḡ  −  mean_b(ḡ^(b))

    Captures all three nuisances (N1 audit, N2 calibrator, N3 argmax)
    AND their cross-terms in a single integrated estimate. Cost: B
    calibrator refits per audit (vs K for the jackknife methods).

    Returns ḡ_bc as a 2-vector.
    """
    rng = np.random.default_rng(seed)
    n_c = len(rep.s_calib)
    n_e = len(rep.s_eval)
    n_a = len(rep.s_audit)
    g_b = np.empty((B, 2), dtype=np.float64)

    for b in range(B):
        # Resample each slice independently with replacement.
        idx_c = rng.integers(0, n_c, size=n_c)
        idx_e = rng.integers(0, n_e, size=n_e)
        idx_a = rng.integers(0, n_a, size=n_a)

        s_c_b = rep.s_calib[idx_c]
        y_c_b = rep.y_calib[idx_c]
        s_e_b = rep.s_eval[idx_e]
        s_a_b = rep.s_audit[idx_a]
        y_a_b = rep.y_audit[idx_a]

        # Refit calibrator on the bootstrapped CALIB rows.
        cal_b = fit_calibrator(s_c_b, y_c_b, t_grid, K, seed=seed + b)

        # Re-max t̂ on the bootstrapped EVAL with the bootstrapped calibrator.
        H_e_b = cal_b.predict(s_e_b)
        j_b, t_b = saddle_argmax(H_e_b, t_grid, alpha)

        # Compute ḡ^(b) on the bootstrapped AUDIT slice at (ĥ^(b), t̂^(b)).
        h_t_a_b = cal_b.predict(s_a_b)[:, j_b]
        g1_b, g2_b = g_pair(s_a_b, y_a_b, h_t_a_b, t_b, alpha)
        g_b[b] = [g1_b.mean(), g2_b.mean()]

    bias_boot = g_b.mean(axis=0) - rep.ḡ
    return rep.ḡ - bias_boot
