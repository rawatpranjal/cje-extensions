"""
Isotonic calibrator with K-fold cross-fit.

Math contract:
    f̂      = IsotonicRegression(increasing=True).fit(s_oracle, y_oracle)
    f̂^(−k) = same, fit on oracle rows where fold_id ≠ k

The pooled fit is used for predict() (used by EVAL averages).
The K fold-out fits are used for predict_oof() (used in the AIPW residual
term and in the jackknife).

PAVA preserves the training-slice mean exactly:
    mean_i  f̂(s_oracle_i)  ==  mean_i  y_oracle_i.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self) -> None:
        self._pooled: IsotonicRegression | None = None
        self._folded: list[IsotonicRegression] = []
        self._fold_id: np.ndarray | None = None
        self._s_train: np.ndarray | None = None
        self._K: int | None = None

    def fit(
        self,
        s_oracle: np.ndarray,
        y_oracle: np.ndarray,
        K: int = 5,
        seed: int = 0,
    ) -> "Calibrator":
        s = np.asarray(s_oracle, dtype=np.float64).ravel()
        y = np.asarray(y_oracle, dtype=np.float64).ravel()
        if len(s) != len(y):
            raise ValueError("s and y must have the same length")
        if K < 2:
            raise ValueError(f"K must be >= 2; got {K}")

        rng = np.random.default_rng(seed)
        fold_id = rng.integers(0, K, size=len(s))

        pooled = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s, y)
        folded: list[IsotonicRegression] = []
        for k in range(K):
            train = fold_id != k
            ir_k = IsotonicRegression(increasing=True, out_of_bounds="clip")
            ir_k.fit(s[train], y[train])
            folded.append(ir_k)

        self._pooled = pooled
        self._folded = folded
        self._fold_id = fold_id
        self._s_train = s
        self._K = K
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        if self._pooled is None:
            raise RuntimeError("Calibrator not fit")
        return self._pooled.predict(np.asarray(s, dtype=np.float64).ravel())

    def predict_oof(self, s_oracle: np.ndarray) -> np.ndarray:
        """For each row i with fold_id[i] = k, predict using f̂^(−k)."""
        if not self._folded:
            raise RuntimeError("Calibrator not fit")
        s = np.asarray(s_oracle, dtype=np.float64).ravel()
        if self._s_train is None or len(s) != len(self._s_train):
            raise ValueError("predict_oof requires the same s used in fit()")
        out = np.empty_like(s)
        for k in range(self._K):
            mask = self._fold_id == k
            if mask.any():
                out[mask] = self._folded[k].predict(s[mask])
        return out

    def refit_excluding_fold(
        self, s_oracle: np.ndarray, y_oracle: np.ndarray, k: int,
    ) -> IsotonicRegression:
        """For the V̂_cal jackknife: refit on oracle \\ fold_k. Returns a fresh fit."""
        if self._fold_id is None:
            raise RuntimeError("Calibrator not fit")
        s = np.asarray(s_oracle, dtype=np.float64).ravel()
        y = np.asarray(y_oracle, dtype=np.float64).ravel()
        train = self._fold_id != k
        return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s[train], y[train])

    @property
    def K(self) -> int:
        if self._K is None:
            raise RuntimeError("Calibrator not fit")
        return self._K

    @property
    def fold_id(self) -> np.ndarray:
        if self._fold_id is None:
            raise RuntimeError("Calibrator not fit")
        return self._fold_id
