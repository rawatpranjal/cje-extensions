"""
Parametric known-truth DGP for the Monte Carlo validation harness.

Math contract:
    For each policy p ∈ P, parameters (a_p, b_p, scale_p, σ_p) are fixed:

        Y ~ Beta(a_p, b_p)                                  Y ∈ [0, 1]
        S | Y = scale_p · (Y − 0.5) + ε,    ε ~ N(0, σ_p²)

    (Linear in Y so that E[S | Y] is monotone increasing — the isotonic
    calibrator can recover it. Heteroscedasticity / non-linearity could be
    swapped in later without breaking the truth formulas.)

    Closed-form truth (no simulation; one-dimensional Beta integrals):

        Mean(p)        = a / (a + b)
        CVaR_α(p)      = (a / (a + b)) · F_{Beta(a+1, b)}(q_α) / α
        where q_α = F⁻¹_{Beta(a, b)}(α)

    Derivation:
        ∫₀^q  y · f_{Beta(a,b)}(y) dy
            = (B(a+1,b) / B(a,b)) · ∫₀^q  f_{Beta(a+1,b)}(y) dy
            = (a / (a+b)) · F_{Beta(a+1,b)}(q)

    α=1 collapse for the truth:
        q_1 = 1, F_{Beta(a+1,b)}(1) = 1, so  CVaR_1 = a/(a+b) = Mean.
        Verified in dgp_test.py::test_truth_cvar_alpha_one_equals_mean.

    Transport-perturbation knob (for audit-power sweeps):
        sample(..., delta) draws Y ~ Beta(a, b + delta) instead of Beta(a, b).
        delta=0 → null (transport holds). delta>0 → heavier lower tail in
        the target slice than the calibrator was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats


@dataclass(frozen=True)
class PolicyParams:
    a: float
    b: float
    scale: float = 4.0
    sigma: float = 0.8


# Default policy panel: four diverse Y marginals.
DEFAULT_POLICIES: dict[str, PolicyParams] = {
    "uniform":     PolicyParams(a=1.0, b=1.0),    # Mean = 0.5,   CVaR_0.10 ≈ 0.05
    "right_skew":  PolicyParams(a=2.0, b=5.0),    # Mean ≈ 0.286, CVaR_0.10 ≈ 0.043
    "left_skew":   PolicyParams(a=5.0, b=2.0),    # Mean ≈ 0.714, CVaR_0.10 ≈ 0.343
    "tail_heavy":  PolicyParams(a=0.5, b=0.5),    # U-shaped, big lower tail
}


class DGP:
    """Container for per-policy parametric DGPs with closed-form truth."""

    def __init__(self, policies: dict[str, PolicyParams]) -> None:
        self.policies = dict(policies)

    # ----- truth ---------------------------------------------------------------

    def truth_mean(self, policy: str) -> float:
        p = self.policies[policy]
        return p.a / (p.a + p.b)

    def truth_cvar(self, policy: str, alpha: float) -> float:
        """
        CVaR_α(Y) = (a/(a+b)) · F_{Beta(a+1, b)}(q_α) / α,
        with  q_α = F⁻¹_{Beta(a, b)}(α).
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        p = self.policies[policy]
        if alpha == 1.0:
            return self.truth_mean(policy)
        q_alpha = stats.beta.ppf(alpha, p.a, p.b)
        partial = stats.beta.cdf(q_alpha, p.a + 1.0, p.b)
        return (p.a / (p.a + p.b)) * partial / alpha

    # ----- sampling ------------------------------------------------------------

    def sample(
        self,
        policy: str,
        n: int,
        with_oracle: bool,
        seed: int,
        delta: float = 0.0,
        prompt_id_prefix: str = "p",
    ) -> pl.DataFrame:
        """
        Draw n rows from the DGP under policy `policy`.

        Args:
            n:            number of rows.
            with_oracle:  if True, include `y` (oracle) column. EVAL slices
                          set this to False.
            seed:         RNG seed.
            delta:        transport perturbation. b → b + delta. delta=0 is
                          the null (no perturbation).
            prompt_id_prefix: for distinguishing CALIB vs AUDIT vs EVAL slices
                              when needed.
        """
        if policy not in self.policies:
            raise KeyError(f"unknown policy: {policy}")
        p = self.policies[policy]

        rng = np.random.default_rng(seed)
        y = rng.beta(p.a, p.b + delta, size=n)
        eps = rng.normal(0.0, p.sigma, size=n)
        s = p.scale * (y - 0.5) + eps

        cols: dict[str, object] = {
            "prompt_id": [f"{prompt_id_prefix}_{seed}_{i}" for i in range(n)],
            "s": s,
        }
        if with_oracle:
            cols["y"] = y
        return pl.DataFrame(cols)
