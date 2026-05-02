"""Code tests for mean_cje.lib. No statistical tests at this level."""

from __future__ import annotations

import numpy as np
import pytest

from mean_cje.lib.calibrator import Calibrator


def _toy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
    y = np.clip(y, 0.0, 1.0)
    return s, y


def test_calibrator_predict_is_monotone_increasing() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    s_query = np.linspace(s.min(), s.max(), 50)
    pred = cal.predict(s_query)
    assert (np.diff(pred) >= -1e-12).all()


def test_calibrator_pooled_mean_equals_oracle_mean() -> None:
    """PAVA preserves the slice mean exactly."""
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    pooled_pred = cal.predict(s)
    assert abs(pooled_pred.mean() - y.mean()) <= 1e-12


def test_predict_oof_shape_and_finite() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    oof = cal.predict_oof(s)
    assert oof.shape == s.shape
    assert np.isfinite(oof).all()


def test_predict_oof_differs_from_pooled() -> None:
    """OOF predictions should differ from pooled at most rows."""
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    pooled = cal.predict(s)
    oof = cal.predict_oof(s)
    assert (np.abs(pooled - oof) > 0).mean() > 0.5
