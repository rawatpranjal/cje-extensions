"""
Variance components and CIs.

Math contracts:

V̂_eval (IF):
    φ_i  =  f̂(s_eval_i) − V̂
    V̂_eval  =  (1/n_eval²) · Σ_i  φ_i²

V̂_cal (jackknife, paper appendix_notation:140):
    For k = 1..K:
        f̂^(−k) := refit calibrator omitting fold k of the oracle
        V̂^(−k) := estimator(eval, f̂^(−k))   (matches the chosen estimator)
    V̄  := mean_k V̂^(−k)
    V̂_cal := ((K−1)/K) · Σ_k (V̂^(−k) − V̄)²

Wald CI:
    V̂  ±  1.96 · √V̂_total      with V̂_total = V̂_eval + V̂_cal

Bootstrap CIs are estimator-specific; live in this file as
bootstrap_ci_plugin and bootstrap_ci_aipw.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .calibrator import Calibrator
from .estimators import plug_in_mean, aipw_one_step


EstimatorName = Literal["plug_in", "aipw"]


def var_eval_if(s_eval: np.ndarray, calibrator: Calibrator) -> float:
    """V̂_eval via per-row centered IF: (1/n²) Σ φ_i² with φ_i = f̂(s_i) − V̂."""
    pred = calibrator.predict(s_eval)
    v_hat = pred.mean()
    phi = pred - v_hat
    n = len(s_eval)
    return float((phi ** 2).sum() / (n * n))


def _eval_estimator(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    cal_full: Calibrator,
    estimator: EstimatorName,
) -> float:
    if estimator == "plug_in":
        return plug_in_mean(s_eval, cal_full)
    elif estimator == "aipw":
        return aipw_one_step(s_eval, s_oracle, y_oracle, cal_full)
    else:
        raise ValueError(f"unknown estimator {estimator!r}")


def var_cal_jackknife(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    calibrator: Calibrator,
    *,
    estimator: EstimatorName,
) -> float:
    """
    Delete-one-oracle-fold jackknife on the chosen estimator.

    For each k, we use the same refit_excluding_fold mechanism the calibrator
    already supports. Each refit is then wrapped in a tiny shim Calibrator
    so the estimator sees a well-formed object.
    """
    K = calibrator.K
    estimates: list[float] = []
    for k in range(K):
        ir_k = calibrator.refit_excluding_fold(s_oracle, y_oracle, k)

        class _ShimCal(Calibrator):
            def __init__(self, base: Calibrator, ir_pooled) -> None:
                super().__init__()
                self._pooled = ir_pooled
                self._folded = base._folded
                self._fold_id = base._fold_id
                self._s_train = base._s_train
                self._K = base._K

        shim = _ShimCal(calibrator, ir_k)
        estimates.append(_eval_estimator(
            s_eval, s_oracle, y_oracle, shim, estimator,
        ))
    arr = np.asarray(estimates)
    v_bar = arr.mean()
    return float((K - 1) / K * ((arr - v_bar) ** 2).sum())


def wald_ci(v_hat: float, v_total: float, level: float = 0.95) -> tuple[float, float]:
    """V̂ ± z · √V̂_total."""
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level}")
    if v_total < 0:
        raise ValueError(f"v_total must be ≥ 0; got {v_total}")
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(level, 2))
    if z is None:
        from scipy import stats
        z = float(stats.norm.ppf(1 - (1 - level) / 2))
    half = z * np.sqrt(v_total)
    return float(v_hat - half), float(v_hat + half)


def bootstrap_ci_plugin(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    *,
    B: int,
    K: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """
    Full-pipeline bootstrap on the plug-in V̂ (paper appendix_d_bootstrap.tex).
    Each rep:
        - resample ORACLE rows with replacement (size n_oracle)
        - resample EVAL rows with replacement  (size n_eval)
        - refit calibrator on bootstrap oracle
        - V̂^(b) = (1/n_eval) Σ f̂^(b)(s_eval_b_i)

    Resampling EVAL captures eval-side variance; resampling ORACLE
    captures calibrator-fit variance. Holding EVAL fixed (as we did
    initially) captures only Var_cal and undercovers at large n_oracle.
    """
    rng = np.random.default_rng(seed)
    n_o = len(s_oracle)
    n_e = len(s_eval)
    samples = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx_o = rng.integers(0, n_o, size=n_o)
        idx_e = rng.integers(0, n_e, size=n_e)
        s_b = s_oracle[idx_o]
        y_b = y_oracle[idx_o]
        s_eval_b = s_eval[idx_e]
        cal_b = Calibrator().fit(s_b, y_b, K=K, seed=seed + b)
        samples[b] = plug_in_mean(s_eval_b, cal_b)
    lo = float(np.percentile(samples, 100 * (1 - level) / 2))
    hi = float(np.percentile(samples, 100 * (1 - (1 - level) / 2)))
    return lo, hi


def bootstrap_ci_aipw(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    *,
    B: int,
    K: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """
    Full-pipeline bootstrap on θ̂_aug. Resamples both ORACLE and EVAL.

    Per replicate, θ̂_aug uses the fresh oracle (for the residual term)
    and the fresh eval (for the plug-in term).
    """
    rng = np.random.default_rng(seed)
    n_o = len(s_oracle)
    n_e = len(s_eval)
    samples = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx_o = rng.integers(0, n_o, size=n_o)
        idx_e = rng.integers(0, n_e, size=n_e)
        s_b = s_oracle[idx_o]
        y_b = y_oracle[idx_o]
        s_eval_b = s_eval[idx_e]
        cal_b = Calibrator().fit(s_b, y_b, K=K, seed=seed + b)
        samples[b] = aipw_one_step(s_eval_b, s_b, y_b, cal_b)
    lo = float(np.percentile(samples, 100 * (1 - level) / 2))
    hi = float(np.percentile(samples, 100 * (1 - (1 - level) / 2)))
    return lo, hi
