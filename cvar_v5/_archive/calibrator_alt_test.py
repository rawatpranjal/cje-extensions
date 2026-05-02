"""
Tests for the archived calibrator_alt methods. Not collected by pytest at
the v5 level (see ../conftest.py).

To run deliberately:
    cd cvar_v5/_archive && python -m pytest calibrator_alt_test.py
"""

from __future__ import annotations

import numpy as np
import pytest

from cvar_v5._archive.calibrator_alt import (
    fit_bivariate_isotonic_pooled,
    fit_distribution_regression,
    fit_gbm_monotone,
    fit_smoothed_per_t,
)
from cvar_v5.mc.dgp import DEFAULT_POLICIES, DGP


def _toy_data(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
    y = np.clip(y, 0.0, 1.0)
    return s, y


@pytest.fixture(params=[
    "smoothed_per_t",
    "distribution_regression",
    "bivariate_isotonic_pooled",
    "gbm_monotone",
])
def alt_method(request):
    return request.param


def _fit(method: str, s, y, t_grid):
    if method == "smoothed_per_t":
        return fit_smoothed_per_t(s, y, t_grid)
    if method == "distribution_regression":
        return fit_distribution_regression(s, y, t_grid)
    if method == "bivariate_isotonic_pooled":
        return fit_bivariate_isotonic_pooled(s, y, t_grid, max_iter=5)
    if method == "gbm_monotone":
        return fit_gbm_monotone(s, y, t_grid, n_estimators=80)
    raise ValueError(method)


def test_alt_calibrator_monotone_decreasing_in_s(alt_method) -> None:
    s, y = _toy_data(n=400)
    t_grid = np.linspace(0.0, 1.0, 21)
    cg = _fit(alt_method, s, y, t_grid)
    s_query = np.linspace(s.min(), s.max(), 50)
    H = cg.predict(s_query)
    assert (np.diff(H, axis=0) <= 1e-10).all()


def test_alt_calibrator_monotone_and_lipschitz_in_t(alt_method) -> None:
    s, y = _toy_data(n=400)
    t_grid = np.linspace(0.0, 1.0, 21)
    cg = _fit(alt_method, s, y, t_grid)
    s_query = np.linspace(s.min(), s.max(), 50)
    H = cg.predict(s_query)
    diffs = np.diff(H, axis=1)
    dts = np.diff(t_grid)
    assert (diffs >= -1e-10).all()
    assert (diffs <= dts[None, :] + 1e-10).all()


def test_alt_calibrator_predictions_in_zero_to_t(alt_method) -> None:
    s, y = _toy_data(n=400)
    t_grid = np.linspace(0.0, 1.0, 21)
    cg = _fit(alt_method, s, y, t_grid)
    s_query = np.linspace(s.min() - 0.5, s.max() + 0.5, 50)
    H = cg.predict(s_query)
    for j, t in enumerate(t_grid):
        h = H[:, j]
        assert (h >= -1e-10).all()
        assert (h <= t + 1e-10).all()


def test_statistical_alt_calibrator_l2_gap_shrinks_with_n() -> None:
    dgp = DGP(DEFAULT_POLICIES)
    t_grid = np.linspace(0.0, 1.0, 61)
    test_t_indices = [int(np.argmin(abs(t_grid - t))) for t in (0.10, 0.30, 0.50, 1.00)]
    for method in (
        "smoothed_per_t",
        "distribution_regression",
        "bivariate_isotonic_pooled",
        "gbm_monotone",
    ):
        for policy in ("uniform", "left_skew"):
            df_huge = dgp.sample(policy, n=50_000, with_oracle=True, seed=99999)
            cg_huge = _fit(method, df_huge["s"].to_numpy(), df_huge["y"].to_numpy(), t_grid)
            df_500 = dgp.sample(policy, n=500, with_oracle=True, seed=42)
            cg_500 = _fit(method, df_500["s"].to_numpy(), df_500["y"].to_numpy(), t_grid)
            df_5000 = dgp.sample(policy, n=5_000, with_oracle=True, seed=43)
            cg_5000 = _fit(method, df_5000["s"].to_numpy(), df_5000["y"].to_numpy(), t_grid)
            s_test = dgp.sample(policy, n=500, with_oracle=False, seed=12345)["s"].to_numpy()
            H_huge = cg_huge.predict(s_test)
            H_500 = cg_500.predict(s_test)
            H_5000 = cg_5000.predict(s_test)
            avg_500 = float(np.mean([
                np.sqrt(np.mean((H_500[:, j] - H_huge[:, j]) ** 2))
                for j in test_t_indices
            ]))
            avg_5000 = float(np.mean([
                np.sqrt(np.mean((H_5000[:, j] - H_huge[:, j]) ** 2))
                for j in test_t_indices
            ]))
            assert avg_5000 < avg_500
