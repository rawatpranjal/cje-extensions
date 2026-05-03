"""
Per-method 4-axis diagnostic from R MC reps.

For each method M, given:
    {Ω̂_M^(r)}_r        the method's Ω̂ at each rep
    {ḡ^(r)}_r           the realized ḡ at each rep
    Σ_full              empirical Var(ḡ) across reps (scaled by n_audit)
    ε                   empirical mean(ḡ) across reps
    Σ_oracle            closed-form variance under (h*, t*)

Compute and return:
    A: variance bias       mean(Ω̂_M) − Σ_full / n_audit
    B: variance dispersion frob_norm of std(Ω̂_M^(r)) per entry
    C: center bias for the test as used by M (depends on whether M
       applies bias correction; default no correction, ḡ_used = ḡ)
    D: empirical size      Pr(W_M > χ²_{2, 0.95})
    Predicted size from non-central χ² formula given (mean Ω̂_M, ε)

Bias correction layer:
    A method "+_bc" subtracts the jackknife-based bias estimate
    bias_jk = (K-1)·(mean(g_per_fold) − ḡ) per rep, giving
    ḡ_bc^(r) = K·ḡ^(r) − (K-1)·mean(g_per_fold^(r)).
    Center bias becomes mean(ḡ_bc^(r)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


_CHI2_CRIT = float(stats.chi2.ppf(0.95, df=2))   # 5.99146...


@dataclass
class MethodDiag:
    name: str
    var_bias: np.ndarray       # 2x2: mean(Ω̂_M) − Σ_full / n_audit
    var_dispersion: np.ndarray # 2x2 of per-entry std across reps
    center_bias: np.ndarray    # 2-vector: mean(ḡ_used)
    empirical_size: float
    predicted_size: float
    mean_omega: np.ndarray     # 2x2 mean of Ω̂_M^(r)


def _wald_n(g_used: np.ndarray, omega: np.ndarray) -> float:
    """Return W_n = ḡᵀ Ω̂⁻¹ ḡ with pinv for safety."""
    return float(g_used @ np.linalg.pinv(omega) @ g_used)


def _predict_size_noncentral(
    method_omega: np.ndarray, true_var: np.ndarray, center: np.ndarray,
    n_rep_sim: int = 50_000, seed: int = 0,
) -> float:
    """
    Predicted size: Pr( ḡ̃ᵀ method_omega⁻¹ ḡ̃ > 5.99 ),
       ḡ̃ ~ N(center, true_var).

    Two distinct matrices:
        method_omega  : the Ω̂ the method's Wald uses for its inverse.
                        Default: mean(Ω̂_M^(r)) across reps.
        true_var      : the true sampling variance of ḡ_used_M.
                        Default: Σ_full / n_audit (the actual MC-estimated
                        variance of the realized statistic).

    Setting method_omega = true_var collapses to "method has perfect
    variance estimate"; non-centrality drives the size in that case.
    Setting method_omega ≠ true_var captures the variance-misspecification
    contribution to size.
    """
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(true_var)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(true_var + 1e-9 * np.eye(2))
    z = rng.normal(size=(n_rep_sim, 2))
    g = z @ L.T + center
    omega_inv = np.linalg.pinv(method_omega)
    W = np.einsum("ri,ij,rj->r", g, omega_inv, g)
    return float((W > _CHI2_CRIT).mean())


def diagnose_method(
    name: str,
    omegas: list[np.ndarray],
    g_used: list[np.ndarray],
    sigma_full: np.ndarray,    # n_audit · sample-cov(ḡ)
    n_audit: int,
) -> MethodDiag:
    """
    Compute all four axes (A, B, C, D) plus predicted size for one method.

    Args:
        omegas:     list of Ω̂_M^(r) (each 2x2) across R reps
        g_used:     list of ḡ_used_M^(r) (each 2-vector) across same R reps
        sigma_full: empirical Var(ḡ) · n_audit (the "true" Ω, in our sense)
        n_audit:    audit size used in the simulation
    """
    omegas_arr = np.stack(omegas, axis=0)                # (R, 2, 2)
    g_arr = np.stack(g_used, axis=0)                     # (R, 2)
    R = omegas_arr.shape[0]

    mean_omega = omegas_arr.mean(axis=0)                 # (2,2)
    var_dispersion = omegas_arr.std(axis=0, ddof=1)      # (2,2)

    # A: variance bias  =  mean(Ω̂_M) − Σ_full / n_audit
    target_omega = sigma_full / n_audit                  # the per-mean variance Ω̂ should estimate
    var_bias = mean_omega - target_omega

    # C: center bias = mean(ḡ_used)
    center_bias = g_arr.mean(axis=0)

    # D: empirical size
    rejections = 0
    for r in range(R):
        W = _wald_n(g_arr[r], omegas_arr[r])
        if W > _CHI2_CRIT:
            rejections += 1
    empirical_size = rejections / R

    # Predicted size: ḡ ~ N(center_bias, sigma_full / n_audit), Wald uses mean_omega.
    true_var = sigma_full / n_audit
    predicted_size = _predict_size_noncentral(mean_omega, true_var, center_bias)

    return MethodDiag(
        name=name,
        var_bias=var_bias,
        var_dispersion=var_dispersion,
        center_bias=center_bias,
        empirical_size=empirical_size,
        predicted_size=predicted_size,
        mean_omega=mean_omega,
    )


def jackknife_bias_correct(g_realized: np.ndarray, g_per_fold: np.ndarray) -> np.ndarray:
    """
    Standard delete-one-fold jackknife bias correction:
        ḡ_bc = K · ḡ − (K − 1) · mean_k(ḡ^(−k))

    Args:
        g_realized: (2,)
        g_per_fold: (K, 2)
    """
    K = g_per_fold.shape[0]
    return K * g_realized - (K - 1) * g_per_fold.mean(axis=0)
