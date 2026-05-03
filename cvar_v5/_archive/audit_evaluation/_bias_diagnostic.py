"""
Bias-only diagnostic for an audit method's center-of-test.

Scope
-----
This module measures *only* the center bias of ḡ_used_M (the moment
vector the method's Wald uses). It does not look at variance or size.

A method is bias-OK when its center is statistically indistinguishable
from zero. Variance failure is diagnosed in `_variance_diagnostic.py`;
combined size is in `_size_diagnostic.py`.

Math
----
Across R MC reps under H_0:

    ḡ_used_M^(r)             the (possibly bias-corrected) audit moment for method M at rep r
    center_M     :=  mean_r ( ḡ_used_M^(r) )                 ∈ R²
    se_M[c]      :=  std_r( ḡ_used_M^(r)[c] ) / sqrt(R)      per-component MC SE
    z_M[c]       :=  center_M[c] / max(se_M[c], eps)         z-score per component

A method passes the bias diagnostic iff
    | z_M[c] |  ≤  3   for both components c ∈ {0, 1}.

(This corresponds to roughly the 99.7% MC band; with R=300 reps and
typical std ≈ 0.05, that's a center-bias tolerance of ~0.0086 per
component.)

Naming
------
Methods named here look like "<variance>_<bias_correction>" where:
    variance        ∈ {analytical, boot_remax, boot_remax_ridge,
                       analytical_oua, boot_remax_oua}
    bias_correction ∈ {none, bc_jk_cal, bc_jk_full, bc_boot}

The variance choice is irrelevant for the bias diagnostic — bias is
measured on ḡ_used_M, which is determined by the bias correction alone.
We still tag the method with the variance label for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BiasDiag:
    """One method's center-of-test diagnostic."""
    name: str                    # e.g. "analytical_oua + bc_jk_full"
    bias_correction: str         # e.g. "bc_jk_full" (or "none")
    center: np.ndarray           # mean_r(ḡ_used_M^(r)) — shape (2,)
    se: np.ndarray               # per-component MC standard error — shape (2,)
    z: np.ndarray                # center / se — shape (2,)
    passed: bool                 # max(|z|) ≤ 3
    R: int                       # number of MC reps used


def diagnose_bias(
    name: str,
    g_used_per_rep: list[np.ndarray],
    *,
    bias_correction: str = "none",
    z_threshold: float = 3.0,
) -> BiasDiag:
    """
    Compute the per-method center-of-test diagnostic.

    Math
    ----
        center  =  (1/R) Σ_r  ḡ_used_M^(r)
        se[c]   =  std_{r}( ḡ_used_M^(r)[c] ) / sqrt(R)         (component-wise)
        z[c]    =  center[c] / max(se[c], 1e-30)
        passed  =  max_c |z[c]| ≤ z_threshold                   (default 3)

    Args:
        name              human-readable method label, e.g. "analytical+bc_jk_full"
        g_used_per_rep    list of length R, each item shape (2,)
        bias_correction   the bias-correction tag; only for traceability
        z_threshold       passes iff max(|z|) ≤ threshold (default 3 ≈ 99.7% MC band)

    Returns: BiasDiag.
    """
    g_arr = np.stack(g_used_per_rep, axis=0)               # (R, 2)
    R = g_arr.shape[0]
    center = g_arr.mean(axis=0)                            # (2,)
    se = g_arr.std(axis=0, ddof=1) / np.sqrt(R)            # (2,)
    z = center / np.maximum(se, 1e-30)                     # (2,)
    passed = bool(np.all(np.abs(z) <= z_threshold))
    return BiasDiag(
        name=name, bias_correction=bias_correction,
        center=center, se=se, z=z, passed=passed, R=R,
    )
