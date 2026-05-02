"""Synthetic DGP for stop-loss vs. plug-in CVaR-CJE benchmark.

S ~ U[0, 1] (cheap-judge score)
Y = clip(0.2 + 0.6 * S + ε, 0, 1), ε ~ N(0, σ^2)

The σ knob controls Var(Y | S). σ = 0 forces the cheap surrogate to determine
Y deterministically; σ > 0 introduces conditional residual variance.
"""
from __future__ import annotations

import numpy as np


# Default seed for MC truth computations. bench.py passes this explicitly and
# uses the same value for both Sweep 1 and Sweep 2; exposed here so external
# callers see the same default.
DEFAULT_TRUTH_SEED = 42


def sample_panel(n: int, sigma: float, rng: np.random.Generator):
    """Return (s, y) of length n. Y is clipped to [0, 1]."""
    s = rng.uniform(0.0, 1.0, n)
    g = 0.2 + 0.6 * s
    eps = rng.normal(0.0, sigma, n) if sigma > 0 else np.zeros(n)
    y = np.clip(g + eps, 0.0, 1.0)
    return s, y


def _empirical_cvar(y: np.ndarray, alpha: float) -> float:
    n = len(y)
    k = max(1, int(np.ceil(alpha * n)))
    return float(np.partition(y, k)[:k].mean())


def true_cvar(
    alpha: float, sigma: float, *, n_mc: int = 10**7, seed: int = DEFAULT_TRUTH_SEED
) -> float:
    """MC reference CVaR_α(Y) under the marginal distribution of Y."""
    rng = np.random.default_rng(seed)
    _, y = sample_panel(n_mc, sigma, rng)
    return _empirical_cvar(y, alpha)


def true_cvar_with_se(
    alpha: float,
    sigma: float,
    *,
    n_mc: int = 10**7,
    n_batches: int = 10,
    seed: int = DEFAULT_TRUTH_SEED,
) -> tuple[float, float]:
    """Return (mean_truth, mc_se) using batch-means.

    Splits `n_mc` samples into `n_batches` independent batches; computes the
    empirical CVaR within each batch; reports the mean across batches and
    sd / sqrt(n_batches) as the Monte-Carlo standard error of the mean.
    """
    if n_batches < 2:
        raise ValueError("n_batches must be >= 2")
    n_per = n_mc // n_batches
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_batches)
    ests = np.empty(n_batches)
    for i, sd in enumerate(seeds):
        batch_rng = np.random.default_rng(int(sd))
        _, y = sample_panel(n_per, sigma, batch_rng)
        ests[i] = _empirical_cvar(y, alpha)
    mean = float(ests.mean())
    se = float(ests.std(ddof=1) / np.sqrt(n_batches))
    return mean, se


def true_cvar_sigma_zero_exact(alpha: float) -> float:
    """Closed-form CVaR_α(Y) at σ = 0 for this DGP.

    With σ = 0: Y = 0.2 + 0.6·S, S ~ U[0, 1], so Y ~ U[0.2, 0.8] (no clipping).
    The α-quantile is VaR_α = 0.2 + 0.6·α and CVaR_α = (1/α)·∫_0^α (0.2 + 0.6·u) du
    = 0.2 + 0.3·α.
    """
    return 0.2 + 0.3 * alpha
