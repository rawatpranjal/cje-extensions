"""
Alternative calibrator implementations (ARCHIVED).

Investigated 2026-05-02 as part of `TODO.md::[joint-calibrator]`. None
beat the per-t isotonic baseline at our scale. Preserved here for
reference; see `../README.md` for context.

Each `fit_*` returns a `CalibratorGrid`-compatible object exposing:
    .t_grid : np.ndarray
    .n_folds: int   (always 0; alt methods don't implement cross-fit)
    .predict(s): np.ndarray of shape (len(s), |T|)

Math invariants (verified by archived calibrator_alt_test.py):
    1. monotone non-increasing in s
    2. monotone non-decreasing in t
    3. 1-Lipschitz in t:  0 ≤ ĝ(s, t') - ĝ(s, t) ≤ t' - t  for t' > t
    4. bounded:  ĝ(s, t) ∈ [0, t]  for Y ∈ [0, 1] and t ∈ [0, 1]
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.isotonic import IsotonicRegression

from cvar_v5.cvar_cje.calibrator import CalibratorGrid, fit_calibrator_grid


def _reproject_row(H: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Per-row, force ĥ_t to be non-decreasing in t (PAV), 1-Lipschitz in t,
    and bounded in [0, t]. Used by all archived alt methods after their
    initial prediction step.
    """
    T = t_grid.size
    out = np.empty_like(H)
    for i in range(H.shape[0]):
        row = H[i]
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        row_mono = ir.fit_transform(np.arange(T), row)
        dts = np.diff(t_grid)
        diffs = np.diff(row_mono)
        diffs_clipped = np.minimum(diffs, dts)
        row_proj = np.concatenate([row_mono[:1], row_mono[:1] + np.cumsum(diffs_clipped)])
        row_proj = np.maximum(row_proj, 0.0)
        row_proj = np.minimum(row_proj, t_grid)
        out[i] = row_proj
    return out


# ----- smoothed_per_t ---------------------------------------------------------


class _SmoothedCalibratorGrid:
    def __init__(self, base_cg: CalibratorGrid, window: int = 3) -> None:
        if window < 1 or window % 2 == 0:
            raise ValueError(f"window must be a positive odd int; got {window}")
        self._base = base_cg
        self._window = window

    @property
    def t_grid(self) -> np.ndarray:
        return self._base.t_grid

    @property
    def n_folds(self) -> int:
        return 0

    def predict(self, s: np.ndarray) -> np.ndarray:
        H = self._base.predict(s)
        H_smooth = uniform_filter1d(H, size=self._window, axis=1, mode="nearest")
        return _reproject_row(H_smooth, self._base.t_grid)


def fit_smoothed_per_t(
    s_calib: np.ndarray,
    y_calib: np.ndarray,
    t_grid: np.ndarray,
    *,
    window: int = 3,
) -> _SmoothedCalibratorGrid:
    """Fit per-t isotonic, then smooth predictions across t (window-w MA)."""
    base = fit_calibrator_grid(s_calib, y_calib, t_grid)
    return _SmoothedCalibratorGrid(base, window=window)


# ----- distribution_regression ------------------------------------------------


class _DistributionRegressionCalibratorGrid:
    def __init__(
        self,
        t_grid: np.ndarray,
        y_grid: np.ndarray,
        regressors_per_y: list[IsotonicRegression],
    ) -> None:
        self._t_grid = np.asarray(t_grid, dtype=np.float64)
        self._y_grid = np.asarray(y_grid, dtype=np.float64)
        self._regressors = regressors_per_y

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    @property
    def n_folds(self) -> int:
        return 0

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64).ravel()
        n_s = len(s)
        n_y = len(self._y_grid)

        F_pred = np.empty((n_s, n_y), dtype=np.float64)
        for q, reg in enumerate(self._regressors):
            F_pred[:, q] = reg.predict(s)

        for m in range(n_s):
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            F_pred[m] = ir.fit_transform(self._y_grid, F_pred[m])
        F_pred = np.clip(F_pred, 0.0, 1.0)

        T = len(self._t_grid)
        H = np.empty((n_s, T), dtype=np.float64)
        dF = np.diff(F_pred, axis=1)
        y_dF = self._y_grid[:-1] * dF
        cum_y_dF = np.concatenate(
            [np.zeros((n_s, 1)), np.cumsum(y_dF, axis=1)], axis=1
        )

        for j, t in enumerate(self._t_grid):
            F_at_t = np.array([
                float(np.interp(t, self._y_grid, F_pred[m])) for m in range(n_s)
            ])
            E_at_t = np.array([
                float(np.interp(t, self._y_grid, cum_y_dF[m])) for m in range(n_s)
            ])
            H[:, j] = t * F_at_t - E_at_t

        return _reproject_row(H, self._t_grid)


def fit_distribution_regression(
    s_calib: np.ndarray,
    y_calib: np.ndarray,
    t_grid: np.ndarray,
    *,
    n_y_grid: int = 21,
) -> _DistributionRegressionCalibratorGrid:
    """
    Distribution regression. Per y_q in a grid:
        F̂(y_q | s) := IsotonicRegression(decreasing).fit(s, 1{Y ≤ y_q})
    Then ĝ(s, t) = t · F̂(t|s) - sum_q y_q · ΔF̂_q(s) by partial expectation.
    """
    s = np.asarray(s_calib, dtype=np.float64).ravel()
    y = np.asarray(y_calib, dtype=np.float64).ravel()
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(f"y_calib must be in [0, 1]; got [{y.min()}, {y.max()}]")
    t_arr = np.asarray(t_grid, dtype=np.float64).ravel()
    y_grid = np.linspace(0.0, 1.0, n_y_grid)

    regressors: list[IsotonicRegression] = []
    for y_q in y_grid:
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        ir.fit(s, (y <= y_q).astype(np.float64))
        regressors.append(ir)

    return _DistributionRegressionCalibratorGrid(
        t_grid=t_arr, y_grid=y_grid, regressors_per_y=regressors,
    )


# ----- bivariate_isotonic_pooled ---------------------------------------------


class _BivariateIsotonicCalibratorGrid:
    def __init__(
        self,
        t_grid: np.ndarray,
        per_t_predictors: list[IsotonicRegression],
    ) -> None:
        self._t_grid = np.asarray(t_grid, dtype=np.float64)
        self._predictors = per_t_predictors

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    @property
    def n_folds(self) -> int:
        return 0

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64).ravel()
        out = np.empty((len(s), len(self._t_grid)), dtype=np.float64)
        for j, ir in enumerate(self._predictors):
            out[:, j] = ir.predict(s)
        return _reproject_row(out, self._t_grid)


def fit_bivariate_isotonic_pooled(
    s_calib: np.ndarray,
    y_calib: np.ndarray,
    t_grid: np.ndarray,
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> _BivariateIsotonicCalibratorGrid:
    """
    Alternating-projection PAV on the pooled (s, t) → z = (t-Y)_+ matrix.
    Constraints: non-increasing in s, non-decreasing in t, 1-Lipschitz in t.
    """
    s = np.asarray(s_calib, dtype=np.float64).ravel()
    y = np.asarray(y_calib, dtype=np.float64).ravel()
    t_arr = np.asarray(t_grid, dtype=np.float64).ravel()
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(f"y_calib must be in [0, 1]; got [{y.min()}, {y.max()}]")

    sort_idx = np.argsort(s)
    s_sorted = s[sort_idx]
    y_sorted = y[sort_idx]
    n = len(s)
    T = len(t_arr)
    dts = np.diff(t_arr)

    G = np.maximum(t_arr[None, :] - y_sorted[:, None], 0.0)

    for _ in range(max_iter):
        G_prev = G.copy()
        for k in range(T):
            ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
            G[:, k] = ir.fit_transform(s_sorted, G[:, k])
        for i in range(n):
            row = G[i]
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            row_mono = ir.fit_transform(np.arange(T), row)
            diffs = np.diff(row_mono)
            diffs_clipped = np.minimum(diffs, dts)
            row_proj = np.concatenate([row_mono[:1], row_mono[:1] + np.cumsum(diffs_clipped)])
            row_proj = np.maximum(row_proj, 0.0)
            row_proj = np.minimum(row_proj, t_arr)
            G[i] = row_proj
        if float(np.max(np.abs(G - G_prev))) < tol:
            break

    predictors: list[IsotonicRegression] = []
    for k in range(T):
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        ir.fit(s_sorted, G[:, k])
        predictors.append(ir)

    return _BivariateIsotonicCalibratorGrid(t_grid=t_arr, per_t_predictors=predictors)


# ----- gbm_monotone (LightGBM with monotone constraints) ----------------------


class _GBMMonotoneCalibratorGrid:
    def __init__(self, t_grid: np.ndarray, gbm: object) -> None:
        self._t_grid = np.asarray(t_grid, dtype=np.float64)
        self._gbm = gbm

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    @property
    def n_folds(self) -> int:
        return 0

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64).ravel()
        n_s = len(s)
        T = len(self._t_grid)
        S_rep = np.repeat(s, T)
        T_rep = np.tile(self._t_grid, n_s)
        X = np.column_stack([S_rep, T_rep])
        H = self._gbm.predict(X).reshape(n_s, T)
        return _reproject_row(H, self._t_grid)


def fit_gbm_monotone(
    s_calib: np.ndarray,
    y_calib: np.ndarray,
    t_grid: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
) -> _GBMMonotoneCalibratorGrid:
    """LightGBM regressor over pooled (s, t) → z, with monotone_constraints=[-1, +1]."""
    import lightgbm as lgb

    s = np.asarray(s_calib, dtype=np.float64).ravel()
    y = np.asarray(y_calib, dtype=np.float64).ravel()
    t_arr = np.asarray(t_grid, dtype=np.float64).ravel()
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(f"y_calib must be in [0, 1]; got [{y.min()}, {y.max()}]")

    n = len(s)
    T = len(t_arr)
    S_rep = np.repeat(s, T)
    T_rep = np.tile(t_arr, n)
    Y_rep = np.repeat(y, T)
    Z = np.maximum(T_rep - Y_rep, 0.0)
    X = np.column_stack([S_rep, T_rep])

    gbm = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        monotone_constraints=[-1, +1],
        verbose=-1,
    )
    gbm.fit(X, Z)
    return _GBMMonotoneCalibratorGrid(t_grid=t_arr, gbm=gbm)
