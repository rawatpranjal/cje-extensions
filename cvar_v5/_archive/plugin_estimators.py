"""
Plug-in CVaR estimators (RESEARCH).

Investigated 2026-05-02 as part of the comparison study against the Direct
saddle-point estimator. Anchored in `cvar_v5/TODO.md::[plugin-quantile]` and
`[plugin-ru-dual]`.

Both plug-ins reduce the calibrator's output to the *mean reward*
    m̂(s) := E_p[Y | s]
via the PAV reflection identity at t=1 (paper method.tex:115-121):
    ĥ_1(s) = 1 − f̂(s)         where f̂ := iso_increasing(s, Y)
so
    m̂(s) = 1 − ĥ_1(s).

The two plug-ins differ in how they convert m̂(s_eval) into a CVaR scalar.

plugin_quantile (lower-tail CVaR_α on m̂):
    sort m̂(s_eval) ascending → take the bottom k = ⌊α·n_eval⌋ values
        → ĈVaR := mean of those bottom k values.
    Reported threshold is the empirical α-quantile q̂_α.

plugin_ru_dual (Rockafellar–Uryasev dual evaluated on m̂):
    Ψ̃_α(t) := t − (1/(α · n_eval)) · Σ_i (t − m̂(s_i))_+
    ĈVaR := max_{t ∈ T} Ψ̃_α(t)
    Reported threshold is the argmax t̂.

Why "plug-in" can be statistically inferior to Direct
------------------------------------------------------
The Direct calibrator estimates ĥ_t(s) = E[(t−Y)_+ | s] directly. By
Jensen on the convex map (t − y)_+,
    E[(t − Y)_+ | s]  ≥  (t − E[Y | s])_+  =  (t − m̂(s))_+
so plug-in RU-dual on m̂ systematically *underestimates* the conditional
shortfall, which biases the lower-tail CVaR upward. Plug-in quantile loses
the within-stratum variability of Y | s entirely (it only sees m̂).

This loss of variance information may or may not be paid back by faster
convergence (m̂ is one functional rather than a grid). Settled empirically
by `_archive/estimator_comparison_sweep.py`.

Math invariants (verified by archived plugin_estimators_test.py):
    1. α=1 collapse: both plug-ins return the same Mean-CJE value as Direct
       to ≤ 1e-9.
    2. monotone non-decreasing in α (the bottom-α tail is non-shrinking).
    3. for Y ∈ [0, 1]: estimate ∈ [0, 1].
    4. for Y ≡ c (constant):  estimate = c.
"""

from __future__ import annotations

import numpy as np

from cvar_v5.cvar_cje.calibrator import CalibratorGrid
from cvar_v5.cvar_cje.schema import EstimateResult, Slice


def _mean_reward_from_calibrator(
    cg: CalibratorGrid, s_eval: np.ndarray,
) -> np.ndarray:
    """
    m̂(s_eval) := 1 − ĥ_1(s_eval), using PAV reflection at t=1.

    Requires the calibrator's t_grid to contain 1.0 as its last entry; this
    is the convention used by `Config.grid_lo=0, grid_hi=1` and by every
    estimator in production.
    """
    t_grid = cg.t_grid
    if abs(t_grid[-1] - 1.0) > 1e-12:
        raise ValueError(
            f"plug-in estimators require t_grid to end at 1.0; got {t_grid[-1]}"
        )
    H = cg.predict(s_eval)               # (n_eval, |T|)
    return 1.0 - H[:, -1]                # m̂ = 1 − ĥ_1


def plugin_quantile_cvar(
    eval_slice: Slice,
    calibrator: CalibratorGrid,
    alpha: float,
) -> EstimateResult:
    """
    Plug-in lower-tail CVaR via empirical α-quantile of the calibrated mean
    reward `m̂(s_eval)`.

    Returns the standard tail-mean estimator on the (1-D) sample
    {m̂(s_eval_i)}: average of the bottom ⌊α·n⌋ values; q̂_α reported as
    the threshold.
    """
    if eval_slice.role != "eval":
        raise ValueError(f"eval_slice.role must be 'eval'; got {eval_slice.role!r}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

    s_eval = eval_slice.s()
    n = len(s_eval)
    if n == 0:
        raise ValueError("eval_slice is empty")

    m_hat = _mean_reward_from_calibrator(calibrator, s_eval)
    m_sorted = np.sort(m_hat)
    k = max(1, int(np.floor(alpha * n)))
    cvar = float(m_sorted[:k].mean())
    q_alpha = float(m_sorted[k - 1])

    return EstimateResult(alpha=float(alpha), value=cvar, threshold=q_alpha)


def plugin_ru_dual_cvar(
    eval_slice: Slice,
    calibrator: CalibratorGrid,
    alpha: float,
) -> EstimateResult:
    """
    Plug-in Rockafellar–Uryasev dual evaluated on the calibrated mean reward.

    Same saddle-point form as Direct, but the per-row shortfall is computed
    from `m̂(s_i) = 1 − ĥ_1(s_i)` rather than from the joint shortfall
    calibrator `ĥ_t(s_i)`.

        Ψ̃_α(t)  :=  t  −  (1/(α · n_eval)) · Σ_i (t − m̂(s_i))_+
        ĈVaR    :=  max_{t ∈ T}  Ψ̃_α(t)
        t̂       :=  argmax over the calibrator's t_grid

    By construction this collapses to Mean-CJE at α=1 (the (1 − m̂)_+ term
    becomes 1 − m̂ since m̂ ∈ [0,1]).
    """
    if eval_slice.role != "eval":
        raise ValueError(f"eval_slice.role must be 'eval'; got {eval_slice.role!r}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

    s_eval = eval_slice.s()
    n = len(s_eval)
    if n == 0:
        raise ValueError("eval_slice is empty")

    m_hat = _mean_reward_from_calibrator(calibrator, s_eval)
    t_grid = calibrator.t_grid

    shortfall = np.maximum(t_grid[None, :] - m_hat[:, None], 0.0)  # (n, |T|)
    psi = t_grid - shortfall.mean(axis=0) / alpha                  # (|T|,)

    j_star = int(np.argmax(psi))
    return EstimateResult(
        alpha=float(alpha),
        value=float(psi[j_star]),
        threshold=float(t_grid[j_star]),
    )
