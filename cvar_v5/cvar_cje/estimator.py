"""
Direct CVaR-CJE estimator (saddle-point form).

Math contract (paper: cvar_v4/sections/method.tex:25-65, eq:direct-estimator):

    Lower-tail CVaR at level α ∈ (0, 1]:
        CVaR_α(Y) = sup_{t ∈ R} { t − (1/α) · E_p[(t − Y)_+] }

    Direct CVaR-CJE estimate, on fresh target-policy responses (EVAL slice)
    with cheap-judge scores s_i:
        Ψ̂_α(t) = t − (1/(α · n_eval)) · Σ_i ĥ_t(s_i)
        ĈVaR_α^direct = max_{t ∈ T} Ψ̂_α(t)
        t̂_α          = argmax_{t ∈ T} Ψ̂_α(t)

    where ĥ_t is the stop-loss isotonic calibrator (cvar_cje/calibrator.py).

α=1 collapse (paper method.tex:115-121):

    For Y ∈ [0,1] with grid T containing t = 1:
        argmax_t Ψ̂_1(t) is reached on the plateau t ≥ max_i f̂(s_i),
        where f̂(s) = 1 − ĥ_1(s).
        Since max_i f̂(s_i) ≤ 1, the right-endpoint t = 1 is always on the
        plateau and `t̂_1 = 1`.
        Ψ̂_1(1) = 1 − (1/n) Σ_i ĥ_1(s_i)
              = 1 − (1/n) Σ_i (1 − f̂(s_i))
              = (1/n) Σ_i f̂(s_i)
              = Mean-CJE.

    Numerical equality of `estimate_direct_cvar(α=1)` and an independently-
    computed `mean(IsotonicRegression(increasing=True).fit(s_calib, y_calib)
    .predict(s_eval))` is enforced by estimator_test.py::test_alpha_one_collapse.
"""

from __future__ import annotations

import numpy as np

from .calibrator import CalibratorGrid
from .schema import EstimateResult, Slice


def estimate_direct_cvar(
    eval_slice: Slice,
    calibrator: CalibratorGrid,
    alpha: float,
) -> EstimateResult:
    """
    Compute the Direct CVaR-CJE estimate at level α.

    The grid T is the calibrator's own t_grid; this is the same grid used to
    fit the per-t calibrators, ensuring `t̂_α ∈ T`.

    Args:
        eval_slice: target-policy fresh draws. Must have an `s` column.
        calibrator: fitted CalibratorGrid (pooled fits used here).
        alpha:      level α ∈ (0, 1].

    Returns:
        EstimateResult(alpha, value, threshold).
    """
    if eval_slice.role != "eval":
        raise ValueError(f"eval_slice.role must be 'eval'; got {eval_slice.role!r}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

    s_eval = eval_slice.s()
    n = len(s_eval)
    if n == 0:
        raise ValueError("eval_slice is empty")

    t_grid = calibrator.t_grid                   # shape (|T|,)
    H = calibrator.predict(s_eval)               # shape (n_eval, |T|)
    H_mean = H.mean(axis=0)                      # shape (|T|,)

    # Ψ̂_α(t) = t − (1/α) · mean_i ĥ_t(s_i)
    psi = t_grid - H_mean / alpha                # shape (|T|,)

    j_star = int(np.argmax(psi))
    return EstimateResult(
        alpha=float(alpha),
        value=float(psi[j_star]),
        threshold=float(t_grid[j_star]),
    )
