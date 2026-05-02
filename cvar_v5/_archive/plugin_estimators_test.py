"""
Tests for the archived plug-in CVaR estimators. Not collected by pytest at
the v5 level (see ../conftest.py).

To run deliberately:
    cd cvar_v5/_archive && python -m pytest plugin_estimators_test.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.isotonic import IsotonicRegression

from cvar_v5._archive.plugin_estimators import (
    plugin_quantile_cvar,
    plugin_ru_dual_cvar,
)
from cvar_v5.cvar_cje.calibrator import fit_calibrator_grid
from cvar_v5.cvar_cje.schema import Slice


PLUGIN_FNS = [plugin_quantile_cvar, plugin_ru_dual_cvar]


def _toy_split(n_calib: int = 400, n_eval: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)

    def draw(n: int):
        s = rng.normal(size=n)
        y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
        y = np.clip(y, 0.0, 1.0)
        return s, y

    s_calib, y_calib = draw(n_calib)
    s_eval, _ = draw(n_eval)
    return s_calib, y_calib, s_eval


def _eval_slice(s_eval: np.ndarray) -> Slice:
    df = pl.DataFrame({
        "prompt_id": [f"p{i}" for i in range(len(s_eval))],
        "s": s_eval,
    })
    return Slice(df=df, role="eval")


@pytest.mark.parametrize("plugin", PLUGIN_FNS)
def test_alpha_one_collapse_to_mean_cje(plugin) -> None:
    """
    Both plug-ins must equal Mean-CJE at α=1 to ≤ 1e-9, by the same identity
    Direct uses (PAV reflection: m̂(s) = 1 − ĥ_1(s)).
    """
    s_calib, y_calib, s_eval = _toy_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    res = plugin(_eval_slice(s_eval), cg, alpha=1.0)

    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s_calib, y_calib)
    v_mean = float(np.mean(f_inc.predict(s_eval)))

    assert abs(res.value - v_mean) <= 1e-9, (
        f"{plugin.__name__} α=1 collapse failed: ĈVaR_1={res.value!r}, "
        f"Mean-CJE={v_mean!r}, |Δ|={abs(res.value - v_mean):.3e}"
    )


@pytest.mark.parametrize("plugin", PLUGIN_FNS)
def test_monotone_in_alpha(plugin) -> None:
    s_calib, y_calib, s_eval = _toy_split(n_calib=600, n_eval=800)
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)
    eval_s = _eval_slice(s_eval)

    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    values = [plugin(eval_s, cg, a).value for a in alphas]

    diffs = np.diff(values)
    assert (diffs >= -1e-9).all(), (
        f"{plugin.__name__}: should be non-decreasing in α; got {values}"
    )


@pytest.mark.parametrize("plugin", PLUGIN_FNS)
def test_value_in_zero_one(plugin) -> None:
    """Y ∈ [0, 1] ⇒ m̂(s) ∈ [0, 1] ⇒ ĈVaR ∈ [0, 1]."""
    s_calib, y_calib, s_eval = _toy_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    for alpha in (0.05, 0.10, 0.25, 0.50, 1.0):
        v = plugin(_eval_slice(s_eval), cg, alpha).value
        assert -1e-9 <= v <= 1.0 + 1e-9, (
            f"{plugin.__name__} α={alpha}: value {v} outside [0, 1]"
        )


@pytest.mark.parametrize("plugin", PLUGIN_FNS)
@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_constant_y_returns_constant(plugin, c: float) -> None:
    """Constant Y ≡ c ⇒ truth_cvar = c ⇒ both plug-ins return c at any α."""
    rng = np.random.default_rng(0)
    n_calib, n_eval = 200, 300
    s_calib = rng.normal(size=n_calib)
    s_eval = rng.normal(size=n_eval)
    y_calib = np.full(n_calib, c, dtype=np.float64)
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    for alpha in (0.05, 0.10, 0.25, 0.50, 1.0):
        v = plugin(_eval_slice(s_eval), cg, alpha).value
        assert abs(v - c) <= 1e-9, (
            f"{plugin.__name__} constant Y={c}, α={alpha}: ĈVaR={v}, "
            f"|Δ|={abs(v - c):.3e}"
        )


@pytest.mark.parametrize("plugin", PLUGIN_FNS)
def test_alpha_out_of_range_rejected(plugin) -> None:
    s_calib, y_calib, s_eval = _toy_split(n_calib=50, n_eval=50)
    cg = fit_calibrator_grid(s_calib, y_calib, np.linspace(0, 1, 11))
    eval_s = _eval_slice(s_eval)
    with pytest.raises(ValueError, match="alpha must be in"):
        plugin(eval_s, cg, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        plugin(eval_s, cg, alpha=1.5)
