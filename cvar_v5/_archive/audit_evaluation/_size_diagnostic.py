"""
Combined size diagnostic: empirical and Jensen-correct predicted size.

Scope
-----
This module produces the operational headline (empirical size) and a
prediction from the (variance, center) inputs that, by construction,
should match empirical to MC noise when the framework is sound.

Math
----
Empirical size (the actual operational metric):

    W_M^(r)         =  ḡ_used_M^(r)^T  ·  (Ω̂_M^(r))^{-1}  ·  ḡ_used_M^(r)
    empirical_size  =  (1 / R) · Σ_r  1{W_M^(r) > χ²_{2, 1−η}}

Predicted size — per-rep Ω̂ Jensen-correct version:

    Goal: predict empirical_size from the joint distribution of
    (ḡ_used_M, Ω̂_M) under H_0, treating ḡ_used_M as approximately Gaussian
    with mean center_M and covariance true_var = Σ_full / n_audit.

    Naive (wrong):
        size_pred  =  Pr( ḡ̃^T (mean Ω̂_M)^{-1} ḡ̃ > c )
                       ḡ̃ ~ N(center_M, true_var)
        ↑ uses the AVERAGE Ω̂ in the inverse. By Jensen on the convex map
        Ω → Ω^{-1}, mean(Ω̂) inverted gives a LARGER quadratic form, but
        per-rep Ω̂ has tail realizations that reject more often. The naive
        prediction systematically under-estimates size.

    Per-rep Jensen-correct:
        For r_sim = 1..R_sim_outer:
            idx        ~ Uniform({1, ..., R})
            Ω̂_pick     :=  Ω̂_M^(idx)            (sampled from the EMPIRICAL
                                                  distribution of Ω̂_M, not its mean)
            for s = 1..R_sim_inner:
                ḡ̃_s    ~  N(center_M, true_var)
                W_s    :=  ḡ̃_s^T  Ω̂_pick^{-1}  ḡ̃_s
            rejections_outer[r_sim]  :=  mean_s 1{W_s > c}
        size_pred  :=  mean_{r_sim} rejections_outer[r_sim]

    Equivalent (and more efficient): vectorize.
        For each n_inner sample ḡ̃, draw a fresh Ω̂_pick. Compute W. The
        rejection rate over (ḡ̃, Ω̂_pick) jointly equals the per-rep
        nested-mean above by Fubini.

Pass criterion:
    | empirical_size − predicted_size |  <  size_match_tol  (default 0.02)

Notes
-----
- The "true_var" used here is Σ_full / n_audit, which is the actual
  Var(ḡ_used_M) on the DGP under H_0. If center_M ≠ 0 (bias not corrected),
  the prediction reflects how that center bias inflates W_n via
  non-central χ² mass.
- We use np.linalg.pinv for the per-pick inverse to be robust to small
  numerical deficiencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


_CHI2_CRIT = float(stats.chi2.ppf(0.95, df=2))


@dataclass
class SizeDiag:
    name: str
    bias_correction: str
    empirical_size: float
    predicted_size: float
    abs_error: float            # |empirical - predicted|
    passed: bool                # abs_error < size_match_tol
    R: int


def _empirical_size(
    omegas: list[np.ndarray], g_used: list[np.ndarray],
) -> float:
    """W_M^(r) = ḡ^T Ω̂^(r)^{-1} ḡ with pinv; size = mean rejection."""
    R = len(omegas)
    rejections = 0
    for r in range(R):
        omega_inv = np.linalg.pinv(omegas[r])
        W = float(g_used[r] @ omega_inv @ g_used[r])
        if W > _CHI2_CRIT:
            rejections += 1
    return rejections / R


def _predict_size_per_rep_omega(
    omegas: list[np.ndarray],
    center: np.ndarray,
    true_var: np.ndarray,
    n_sim: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Per-rep Ω̂ Jensen-correct predicted size.

    Math
    ----
        For each of n_sim simulated reps, jointly draw:
            ḡ̃   ~ N(center, true_var)
            idx ~ Uniform({1, ..., R})        (R = len(omegas))
            Ω̂   := omegas[idx]
            W   := ḡ̃^T Ω̂^{-1} ḡ̃

        size_pred := (1/n_sim) Σ 1{W > χ²_{2, 0.95}}.

    By Fubini, this equals the per-Ω̂_pick inner-mean averaged over outer
    Ω̂_pick draws. The vectorization pairs one ḡ̃ with one Ω̂_pick per
    simulated rep.
    """
    rng = np.random.default_rng(seed)
    R = len(omegas)
    try:
        L = np.linalg.cholesky(true_var)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(true_var + 1e-9 * np.eye(2))

    # Pre-compute per-omega inverses ONCE.
    omega_invs = np.stack([np.linalg.pinv(o) for o in omegas], axis=0)  # (R, 2, 2)

    # For n_sim simulated reps:
    #   ḡ̃ = z @ L^T + center,  z ~ N(0, I)
    #   pick idx ~ Uniform({0, ..., R-1})
    #   W = ḡ̃ · Ω̂_invs[idx] · ḡ̃
    z = rng.normal(size=(n_sim, 2))
    g_tilde = z @ L.T + center                                          # (n_sim, 2)
    idx = rng.integers(0, R, size=n_sim)                                # (n_sim,)
    picked_invs = omega_invs[idx]                                       # (n_sim, 2, 2)
    # W[r] = g_tilde[r] · picked_invs[r] · g_tilde[r]
    W = np.einsum("ri,rij,rj->r", g_tilde, picked_invs, g_tilde)
    return float((W > _CHI2_CRIT).mean())


def diagnose_size(
    name: str,
    omegas_per_rep: list[np.ndarray],
    g_used_per_rep: list[np.ndarray],
    *,
    bias_correction: str = "none",
    n_pred_sim: int = 50_000,
    pred_seed: int = 0,
    size_match_tol: float = 0.02,
) -> SizeDiag:
    """
    Compute the size-side diagnostic for one method.

    Steps
    -----
        1. empirical_size := mean over r of 1{W_M^(r) > χ²_{2, 0.95}}
                             with W_M^(r) using per-rep Ω̂_M^(r) and ḡ_used_M^(r).
        2. center  := mean_r(ḡ_used_M^(r))                      ∈ R²
        3. true_var := sample-cov_r(ḡ_used_M^(r))               ∈ R^{2×2}
                       computed FROM g_used_per_rep itself (NOT from raw ḡ),
                       so bias-correction-induced variance inflation is
                       captured automatically.
        4. predicted_size := per-rep-Ω̂ Jensen-correct prediction
                             (samples ḡ̃ ~ N(center, true_var) and picks
                              Ω̂ uniformly from omegas_per_rep, takes
                              fraction with W > χ²_crit).
        5. passed := |empirical_size − predicted_size| < size_match_tol.

    Why true_var from g_used (not raw ḡ)
    ------------------------------------
    Bias correction can amplify per-rep fluctuations. For bc_jk_full's
    formula ḡ_bc = K·ḡ − (K−1)·ḡ_jk, the corrected estimator has its own
    variance Var(ḡ_bc), which is generally NOT Var(ḡ). The size prediction
    must use Var(ḡ_used), measured from the same R reps as the empirical
    rejection rate. Otherwise we mismatch the sampling variance of the
    test statistic to what's being tested.

    Args:
        name              method label
        omegas_per_rep    list[R] of (2,2) per-rep Ω̂_M^(r)
        g_used_per_rep    list[R] of (2,) per-rep ḡ_used_M^(r)
        bias_correction   for traceability
        n_pred_sim        # simulated reps in the prediction
        pred_seed         seed for the prediction's RNG
        size_match_tol    pass tolerance |emp − pred|

    Returns: SizeDiag.
    """
    R = len(omegas_per_rep)
    emp = _empirical_size(omegas_per_rep, g_used_per_rep)

    g_arr = np.stack(g_used_per_rep, axis=0)              # (R, 2)
    center = g_arr.mean(axis=0)
    # true_var is the sampling variance of ḡ_used_M (a single rep's mean),
    # which is sample-cov / 1 (each row IS already a per-rep mean — not a
    # per-row mean within a rep). So no division by n_audit.
    true_var = np.cov(g_arr, rowvar=False, ddof=1)        # (2, 2)

    pred = _predict_size_per_rep_omega(
        omegas_per_rep, center, true_var,
        n_sim=n_pred_sim, seed=pred_seed,
    )

    abs_err = abs(emp - pred)
    return SizeDiag(
        name=name, bias_correction=bias_correction,
        empirical_size=emp, predicted_size=pred,
        abs_error=abs_err, passed=abs_err < size_match_tol, R=R,
    )
