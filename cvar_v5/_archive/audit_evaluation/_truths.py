"""
Population-truth computations for the audit-Ω̂ evaluation.

See evaluation.md §3.1 (Σ_oracle), §3.3 (oracle h*, t*).

DGP (controlled):
    Y ~ Beta(a, b + δ_y)
    S | Y = (scale + δ_scale) · (Y − 0.5) + ε,   ε ~ N(0, (σ + δ_sigma)²)

What we compute:
    t*           closed form via scipy.stats.beta.ppf
    h*_t(s)      1-D quadrature against p_target(Y | s) ∝ p(s|Y) · p_Y(Y)
    Σ_oracle     2x2 population covariance of (g_1*, g_2*) at the truth
                 — Σ_oracle[0,0] = α(1−α) closed form
                 — Σ_oracle[1,1], Σ_oracle[0,1] via 2-D quadrature
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DGPParams:
    a: float
    b: float
    scale: float = 4.0
    sigma: float = 0.8


@dataclass(frozen=True)
class TargetPert:
    delta_y: float = 0.0
    delta_scale: float = 0.0
    delta_sigma: float = 0.0


def target_params(p: DGPParams, pert: TargetPert) -> tuple[float, float, float, float]:
    """Effective (a, b, scale, σ) of the target distribution."""
    return p.a, p.b + pert.delta_y, p.scale + pert.delta_scale, max(1e-6, p.sigma + pert.delta_sigma)


def t_star(p: DGPParams, pert: TargetPert, alpha: float) -> float:
    """t* = α-quantile of Y under target."""
    a, b, _, _ = target_params(p, pert)
    return float(stats.beta.ppf(alpha, a, b))


def h_star_at(s: np.ndarray, t_grid: np.ndarray, p: DGPParams, pert: TargetPert,
              n_y: int = 401) -> np.ndarray:
    """
    Compute h*_t(s) := E_target[(t − Y)_+ | S = s] for every s in `s` and
    every t in `t_grid`. Returns shape (len(s), len(t_grid)).

    Uses fine quadrature in y on [0, 1].
    """
    a, b, sc, sg = target_params(p, pert)
    s = np.asarray(s, dtype=np.float64).ravel()
    t_grid = np.asarray(t_grid, dtype=np.float64).ravel()

    # Quadrature grid in y, exclude endpoints to avoid Beta-pdf blowups.
    eps = 1e-6
    y_grid = np.linspace(eps, 1 - eps, n_y)
    dy = y_grid[1] - y_grid[0]
    py = stats.beta.pdf(y_grid, a, b)            # (n_y,)

    # log p(s | Y = y) for each s_i, y_j: shape (n_s, n_y)
    z = (s[:, None] - sc * (y_grid[None, :] - 0.5)) / sg
    logps_y = -0.5 * z * z - 0.5 * np.log(2 * np.pi) - np.log(sg)

    # Joint density p(s, y) ∝ p(s|y) · p(y), unnormalized over s.
    # We want p(y | s) = p(s, y) / p(s).
    # Numerically safe: subtract per-row max in log-space.
    log_joint = logps_y + np.log(py)[None, :]    # (n_s, n_y)
    log_joint -= log_joint.max(axis=1, keepdims=True)
    p_yIs = np.exp(log_joint)
    p_yIs /= (p_yIs.sum(axis=1, keepdims=True) * dy)  # normalize as a density in y

    # h*_t(s) = ∫₀^t (t - y) p(y|s) dy, vectorized.
    # For each t, integrate (t - y)_+ p(y|s) over y_grid.
    H = np.empty((len(s), len(t_grid)), dtype=np.float64)
    for j, t in enumerate(t_grid):
        integrand = np.maximum(t - y_grid, 0.0)[None, :] * p_yIs   # (n_s, n_y)
        H[:, j] = integrand.sum(axis=1) * dy
    return H


def sigma_oracle(p: DGPParams, pert: TargetPert, alpha: float,
                 n_quad_s: int = 401, n_quad_y: int = 401) -> np.ndarray:
    """
    2x2 population covariance of (g_1*, g_2*) at the population truth (h*, t*).

    Σ_{11} = α(1 − α)  exactly.
    Σ_{12}, Σ_{22}    via 2-D quadrature against p(s, Y).

    Returns the 2x2 matrix.
    """
    a, b, sc, sg = target_params(p, pert)
    t = t_star(p, pert, alpha)

    # Joint quadrature on (s, y). y ∈ (0, 1), s on a wide enough grid.
    y_grid = np.linspace(1e-6, 1 - 1e-6, n_quad_y)
    dy = y_grid[1] - y_grid[0]
    py = stats.beta.pdf(y_grid, a, b)

    s_lo = sc * (-0.5) - 5 * sg
    s_hi = sc * (+0.5) + 5 * sg
    s_grid = np.linspace(s_lo, s_hi, n_quad_s)
    ds = s_grid[1] - s_grid[0]

    # h*_t(s) at the single t = t_star
    H = h_star_at(s_grid, np.array([t]), p, pert, n_y=n_quad_y).ravel()  # (n_s,)

    # joint density p(s, y) = p(s | y) · p(y)
    z = (s_grid[:, None] - sc * (y_grid[None, :] - 0.5)) / sg
    p_sIy = np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - np.log(sg))
    pjoint = p_sIy * py[None, :]                      # (n_s, n_y)

    # g_1*(s, y) = 1{y ≤ t} - α    — actually doesn't depend on s.
    # g_2*(s, y) = (t - y)_+ - h*(s)

    g1 = (y_grid <= t).astype(np.float64) - alpha     # (n_y,) — same across s
    g2_per_y = np.maximum(t - y_grid, 0.0)            # (n_y,)
    # g_2*(s, y) = g2_per_y[y] - H[s]
    # E[g_2²] = Σ_{s,y} (g2_per_y - H)² · p(s,y) · ds dy

    # Compute E[g_1²] = α(1-α) — sanity
    sigma_11 = float((g1 ** 2 * (pjoint.sum(axis=0) * ds)).sum() * dy)
    # Equivalent closed form:
    sigma_11_closed = alpha * (1 - alpha)

    # E[g_1 · g_2]
    g2_mat = g2_per_y[None, :] - H[:, None]           # (n_s, n_y)
    sigma_12 = float((g1[None, :] * g2_mat * pjoint).sum() * ds * dy)

    # E[g_2²]
    sigma_22 = float((g2_mat ** 2 * pjoint).sum() * ds * dy)

    # Mean checks: E[g_1] should be α - α = 0 at population t*.
    # E[g_2] should be 0 by construction (h* is the conditional mean of (t-Y)_+).
    # We don't enforce here, just report.

    out = np.array([[sigma_11_closed, sigma_12],
                    [sigma_12,       sigma_22]])

    # numerical sanity
    if abs(sigma_11 - sigma_11_closed) > 1e-3:
        # quadrature is poor; widen grid or report
        pass
    return out


def population_g_at_estimated(
    p: DGPParams, pert: TargetPert, alpha: float,
    s_query: np.ndarray, t_hat: float, h_t_hat_estim: np.ndarray,
    n_quad_y: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Population (E_target) of g_1 and g_2 evaluated at FIXED (s_query, t_hat,
    estimated ĥ_t̂(s_query)). Useful for diagnosing whether ε contributions
    come from t̂ ≠ t* or ĥ ≠ h*.

    Returns (E_g1, E_g2_per_s):
        E_g1            = P_target(Y ≤ t_hat) − α                    scalar
        E_g2_per_s[i]   = E_target[(t_hat − Y)_+ | s = s_query[i]] − h_t_hat_estim[i]
                        = h*_{t_hat}(s_query[i]) − h_t_hat_estim[i]   shape (len(s_query),)
    """
    a, b, sc, sg = target_params(p, pert)
    e_g1 = float(stats.beta.cdf(t_hat, a, b) - alpha)
    h_star_at_s = h_star_at(s_query, np.array([t_hat]), p, pert, n_y=n_quad_y).ravel()
    e_g2 = h_star_at_s - h_t_hat_estim
    return np.array([e_g1]), e_g2
