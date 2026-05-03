"""
Variance-only diagnostic for an audit method's Ω̂ estimator.

Scope
-----
Measures only how well a variance method's Ω̂ matches the true sampling
variance Σ_full / n_audit. Center-of-test is handled separately in
`_bias_diagnostic.py`.

A method is variance-OK when:
    1. mean(Ω̂_M)  ≈  Σ_full / n_audit              (low variance bias)
    2. per-rep dispersion is small relative to mean   (low instability)

Math
----
For R MC reps producing {Ω̂_M^(r)}_r=1..R, given Σ_full = n_audit · Var(ḡ):

    target           :=  Σ_full / n_audit
    mean_omega       :=  (1/R) Σ_r  Ω̂_M^(r)                       ∈ R^{2×2}
    var_bias         :=  mean_omega  −  target                    ∈ R^{2×2}
    dispersion[i,j]  :=  std_r( Ω̂_M^(r)[i,j] )                    ∈ R^{2×2}
    frob_rel_err     :=  ||var_bias||_F  /  ||target||_F            scalar
    spectral_ratio   :=  trace(mean_omega) / trace(target)          scalar

Pass criterion:
    frob_rel_err  <  threshold (default 0.10)
    spectral_ratio in [0.85, 1.20]
    (we report dispersion descriptively; large dispersion hurts size
     stability per rep but doesn't enter the pass test directly.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VarDiag:
    """One method's variance-side diagnostic."""
    name: str
    mean_omega: np.ndarray        # (2,2) average Ω̂_M^(r) across reps
    target: np.ndarray            # (2,2) Σ_full / n_audit (the truth)
    var_bias: np.ndarray          # (2,2) mean_omega - target
    dispersion: np.ndarray        # (2,2) per-entry std across reps
    frob_rel_err: float           # ||var_bias||_F / ||target||_F
    spectral_ratio: float         # trace(mean_omega) / trace(target)
    passed: bool
    R: int


def diagnose_variance(
    name: str,
    omegas_per_rep: list[np.ndarray],
    sigma_full: np.ndarray,
    n_audit: int,
    *,
    frob_threshold: float = 0.10,
    spectral_band: tuple[float, float] = (0.85, 1.20),
) -> VarDiag:
    """
    Compute the per-method variance-side diagnostic.

    Math
    ----
        target          =  Σ_full / n_audit
        mean_omega      =  (1/R) Σ_r Ω̂_M^(r)
        var_bias        =  mean_omega − target
        dispersion[i,j] =  std_r( Ω̂_M^(r)[i,j] )
        frob_rel_err    =  ||var_bias||_F / ||target||_F
        spectral_ratio  =  trace(mean_omega) / trace(target)

    Pass: frob_rel_err < frob_threshold AND
          spectral_ratio in spectral_band.

    Args:
        name              method label
        omegas_per_rep    list[R] of (2,2) per-rep Ω̂_M
        sigma_full        n_audit · sample-cov(ḡ) — the true variance estimand
        n_audit           audit slice size used in the simulation
        frob_threshold    pass tolerance on ||var_bias||_F / ||target||_F
        spectral_band     pass tolerance on tr(mean_omega) / tr(target)

    Returns: VarDiag.
    """
    omegas = np.stack(omegas_per_rep, axis=0)                       # (R, 2, 2)
    R = omegas.shape[0]
    mean_omega = omegas.mean(axis=0)                                # (2, 2)
    dispersion = omegas.std(axis=0, ddof=1)                         # (2, 2)
    target = sigma_full / n_audit                                   # (2, 2)
    var_bias = mean_omega - target

    frob_target = float(np.linalg.norm(target, ord="fro"))
    frob_err = float(np.linalg.norm(var_bias, ord="fro")) / max(frob_target, 1e-30)
    spec = float(np.trace(mean_omega)) / max(float(np.trace(target)), 1e-30)
    in_band = spectral_band[0] <= spec <= spectral_band[1]
    passed = (frob_err < frob_threshold) and in_band
    return VarDiag(
        name=name,
        mean_omega=mean_omega,
        target=target,
        var_bias=var_bias,
        dispersion=dispersion,
        frob_rel_err=frob_err,
        spectral_ratio=spec,
        passed=passed,
        R=R,
    )
