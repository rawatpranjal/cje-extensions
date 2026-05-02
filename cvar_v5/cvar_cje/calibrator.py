"""
Stop-loss isotonic calibrator grid.

Math contract (paper: cvar_v4/sections/method.tex:33-46, eq:direct-estimator):

    For each t ∈ T, fit
        ĥ_t(s) ≈ E_p0[(t − Y)_+ | s]
    on the logger oracle slice (CALIB), with the constraint that ĥ_t is
    monotone DECREASING in s (paper appendix_e:107).

Grid construction (`make_t_grid`):

    For lower-tail CVaR_α, the saddle objective Ψ_α(t) = t − E[(t−Y)_+]/α
    has curvature −p_Y(t)/α. At small α the optimum t* = q_α sits in
    [0, 0.1] for typical Y, where the curvature is amplified by 1/α. A
    uniform grid spacing 1/60 ≈ 0.017 is too coarse there: at α=0.01 on
    Y ~ Uniform, the population grid argmax is biased downward by ~0.002
    (verified empirically; see TODO.md commit history).

    `tail_dense` (default) places 21 of 61 points in [0, 0.1] and 41 in
    [0.1, 1]. Same |T|=61 as a uniform grid → same calibrator-fit cost,
    but eliminates the small-α discretization bias on uniform-like Y.

    `uniform` is the legacy linspace(grid_lo, grid_hi, grid_size).

    Implementation:
        z_i(t) = max(t − Y_i, 0)
        ĥ_t = IsotonicRegression(increasing=False).fit(s_CALIB, z(t))

    A separate fit per t. The dependence on t is non-trivial: the conditional
    expectation E[(t − Y)_+ | s] is a different function of s at every t.

α=1 / right-boundary identity (regression test in calibrator_test.py):

    For Y ∈ [0,1] and t = 1:  z = max(1 − Y, 0) = 1 − Y  (since Y ≤ 1).
    By the PAV reflection identity (PAV is exact in float64 and commutes
    with target negation),
        iso_decreasing(s, 1 − Y) ≡ 1 − iso_increasing(s, Y).
    So at t = 1:
        ĥ_1(s) = 1 − f̂(s)        where  f̂ := iso_increasing(s, Y).
    This is what makes the α=1 estimator collapse to Mean-CJE numerically.

Cross-fit:
    With K-fold cross-fit, two collections of fits are produced per t:
      - POOLED:  fit on the full CALIB slice; used to score EVAL and AUDIT.
      - PER-FOLD: K fits, fold k trained on CALIB \\ fold_k.
                  predict_oof returns, for each CALIB row i in fold k,
                  the prediction of the calibrator that did NOT see row i.

    The MVP estimator only consumes POOLED predictions. PER-FOLD is wired in
    for future jackknife/CI work (TODO.md [var-cal-jackknife]) — it costs
    O(K) sklearn fits which is sub-second at MVP scale.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression


GridKind = Literal["uniform", "tail_dense"]


def make_t_grid(
    grid_lo: float = 0.0,
    grid_hi: float = 1.0,
    grid_size: int = 61,
    grid_kind: GridKind = "tail_dense",
    *,
    tail_dense_breakpoint: float = 0.1,
    tail_dense_fraction: float = 21 / 61,
) -> np.ndarray:
    """
    Build the saddle-point threshold grid T.

    Args:
        grid_lo, grid_hi: endpoints. Y ∈ [0, 1] ⇒ use 0.0, 1.0.
        grid_size: total |T|. Calibrator cost is linear in |T|.
        grid_kind:
            "tail_dense" (default): tail_dense_fraction of points uniformly in
                [grid_lo, grid_lo + tail_dense_breakpoint·(grid_hi-grid_lo)],
                the rest uniformly above. Same |T| as uniform; better at small α.
            "uniform": linspace(grid_lo, grid_hi, grid_size).

    Both kinds include grid_lo and grid_hi as endpoints (so α=1 collapse and
    Y=0 boundary cases stay deterministic).
    """
    if grid_kind == "uniform":
        return np.linspace(grid_lo, grid_hi, grid_size)
    if grid_kind == "tail_dense":
        if not (0.0 < tail_dense_fraction < 1.0):
            raise ValueError(f"tail_dense_fraction must be in (0,1); got {tail_dense_fraction}")
        if not (grid_lo < grid_lo + tail_dense_breakpoint * (grid_hi - grid_lo) < grid_hi):
            raise ValueError(
                f"tail_dense_breakpoint·(grid_hi-grid_lo) must produce a "
                f"breakpoint strictly inside [grid_lo, grid_hi]"
            )
        bp = grid_lo + tail_dense_breakpoint * (grid_hi - grid_lo)
        n_below = int(round(grid_size * tail_dense_fraction))
        n_above = grid_size - n_below + 1   # +1 for the shared bp endpoint
        return np.unique(np.concatenate([
            np.linspace(grid_lo, bp, n_below),
            np.linspace(bp, grid_hi, n_above),
        ]))
    raise ValueError(f"unknown grid_kind {grid_kind!r}")


class CalibratorGrid:
    """
    Container for the per-t isotonic fits over a fixed grid T.

    Use:
        cg = fit_calibrator_grid(s_calib, y_calib, t_grid, fold_id=fold_id, K=K)
        H_eval  = cg.predict(s_eval)               # shape (n_eval, |T|)
        H_audit = cg.predict(s_audit)              # shape (n_audit, |T|)
        H_oof   = cg.predict_oof(s_calib, fold_id) # shape (n_calib, |T|)
    """

    def __init__(
        self,
        t_grid: np.ndarray,
        pooled: list[IsotonicRegression],
        folded: list[list[IsotonicRegression]] | None,
        n_folds: int,
    ) -> None:
        self._t_grid = np.asarray(t_grid, dtype=np.float64)
        self._pooled = pooled                                # len = |T|
        self._folded = folded                                # None or list[|T|][K]
        self._n_folds = n_folds                              # 0 if no cross-fit

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    @property
    def n_folds(self) -> int:
        return self._n_folds

    def predict(self, s: np.ndarray) -> np.ndarray:
        """
        Pooled predictions ĥ_t(s) for every t ∈ T.

        Returns shape (len(s), |T|).
        """
        s = np.asarray(s, dtype=np.float64).ravel()
        out = np.empty((len(s), len(self._t_grid)), dtype=np.float64)
        for j, ir in enumerate(self._pooled):
            out[:, j] = ir.predict(s)
        return out

    def predict_oof(self, s: np.ndarray, fold_id: np.ndarray) -> np.ndarray:
        """
        Out-of-fold prediction: row i (with fold_id[i] = k) is scored by the
        calibrator trained on folds other than k.

        Requires cross-fit calibrators. `s` and `fold_id` must align with the
        CALIB rows that produced these folds; this is the caller's invariant.

        Returns shape (len(s), |T|).
        """
        if self._folded is None:
            raise RuntimeError(
                "predict_oof requires cross-fit calibrators; pass fold_id to "
                "fit_calibrator_grid."
            )
        s = np.asarray(s, dtype=np.float64).ravel()
        fold_id = np.asarray(fold_id, dtype=np.int64).ravel()
        if len(fold_id) != len(s):
            raise ValueError("len(fold_id) must equal len(s)")

        out = np.empty((len(s), len(self._t_grid)), dtype=np.float64)
        for k in range(self._n_folds):
            mask = fold_id == k
            if not mask.any():
                continue
            s_k = s[mask]
            for j, fold_fits in enumerate(self._folded):
                out[mask, j] = fold_fits[k].predict(s_k)
        return out


def fit_calibrator_grid(
    s_calib: np.ndarray,
    y_calib: np.ndarray,
    t_grid: np.ndarray,
    fold_id: np.ndarray | None = None,
    K: int | None = None,
) -> CalibratorGrid:
    """
    Fit the stop-loss isotonic calibrator at every t ∈ t_grid.

    Args:
        s_calib: shape (n_calib,) cheap-judge scores on CALIB.
        y_calib: shape (n_calib,) oracle labels on CALIB. Must be in [0, 1].
        t_grid:  shape (|T|,)     threshold grid (typically linspace(0, 1, 61)).
        fold_id: shape (n_calib,) optional, fold assignment in [0, K).
        K:       number of folds (required if fold_id provided).

    Returns:
        CalibratorGrid with pooled fits (always) and per-fold fits (if fold_id).
    """
    s = np.asarray(s_calib, dtype=np.float64).ravel()
    y = np.asarray(y_calib, dtype=np.float64).ravel()
    t_arr = np.asarray(t_grid, dtype=np.float64).ravel()

    if len(s) != len(y):
        raise ValueError("s_calib and y_calib must have the same length")
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(f"y_calib must be in [0, 1]; got [{y.min()}, {y.max()}]")
    if len(t_arr) < 1:
        raise ValueError("t_grid must have at least 1 entry")

    cross_fit = fold_id is not None
    if cross_fit:
        if K is None:
            raise ValueError("K must be provided when fold_id is given")
        fold_id = np.asarray(fold_id, dtype=np.int64).ravel()
        if len(fold_id) != len(s):
            raise ValueError("fold_id must align with s_calib")
        if fold_id.min() < 0 or fold_id.max() >= K:
            raise ValueError(f"fold_id values must be in [0, {K}); got "
                             f"[{fold_id.min()}, {fold_id.max()}]")

    pooled: list[IsotonicRegression] = []
    folded: list[list[IsotonicRegression]] | None = [] if cross_fit else None

    for t in t_arr:
        z = np.maximum(t - y, 0.0)

        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        ir.fit(s, z)
        pooled.append(ir)

        if cross_fit:
            fold_fits: list[IsotonicRegression] = []
            for k in range(K):
                train_mask = fold_id != k
                ir_k = IsotonicRegression(increasing=False, out_of_bounds="clip")
                ir_k.fit(s[train_mask], z[train_mask])
                fold_fits.append(ir_k)
            folded.append(fold_fits)

    return CalibratorGrid(
        t_grid=t_arr,
        pooled=pooled,
        folded=folded,
        n_folds=K if cross_fit else 0,
    )
