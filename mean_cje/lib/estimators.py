"""
Two estimators for E[Y]:

    plug_in_mean(s_eval, cal):
        V̂  =  (1/n_eval) · Σ  f̂(s_eval_i)

    aipw_one_step(s_eval, s_oracle, y_oracle, cal):
        θ̂_aug  =  (1/n_eval) · Σ  f̂(s_eval_i)
                + (1/|L|) · Σ_{j ∈ L}  ( y_oracle_j − f̂^(−j)(s_oracle_j) )
                                         ↑ cross-fit prediction at row j
"""

from __future__ import annotations

import numpy as np

from .calibrator import Calibrator


def plug_in_mean(s_eval: np.ndarray, calibrator: Calibrator) -> float:
    return float(calibrator.predict(s_eval).mean())


def aipw_one_step(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    calibrator: Calibrator,
    *,
    calibrator_pooled: Calibrator | None = None,
) -> float:
    """
    AIPW one-step. `calibrator` provides predict_oof on the oracle slice;
    `calibrator_pooled` (defaults to `calibrator`) provides the pooled fit
    used for the EVAL term.
    """
    pooled = calibrator_pooled or calibrator
    plug = pooled.predict(s_eval).mean()
    oof_residual = (y_oracle - calibrator.predict_oof(s_oracle)).mean()
    return float(plug + oof_residual)
