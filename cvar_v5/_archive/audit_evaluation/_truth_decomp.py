"""
Ground-truth variance decomposition of Σ_full into N1 + N2 + N3.

Goal
----
On a known DGP, compute four population quantities at the SAME
(n_calib, n_audit, n_eval) used by the main eval:

    V_audit_only   :  Var(ḡ) when only AUDIT varies (CALIB and EVAL pinned)
                      → captures N1 (audit-side sampling)

    V_calib_only   :  Var(ḡ) when only CALIB varies (AUDIT and EVAL pinned)
                      → captures N2 (calibrator-fit nuisance)

    V_eval_only    :  Var(ḡ) when only EVAL varies (CALIB and AUDIT pinned)
                      → captures N3 (argmax-on-grid nuisance)

    Σ_full          :  Var(ḡ) when ALL THREE slices vary jointly
                       → captures everything including cross-terms

Math
----
Each component is estimated by R MC reps with the corresponding pinning:

    V_audit_only   ≈  n_audit · sample-cov_r ( ḡ^(r) | one (calib,eval) realization,
                                                       audit re-sampled per rep )

    V_calib_only   ≈  n_audit · sample-cov_r ( ḡ^(r) | one (audit,eval) realization,
                                                       calib re-sampled per rep )

    V_eval_only    ≈  n_audit · sample-cov_r ( ḡ^(r) | one (calib,audit) realization,
                                                       eval re-sampled per rep )

    Σ_full         ≈  n_audit · sample-cov_r ( ḡ^(r) | all three vary jointly )

If the three nuisances are uncorrelated under the DGP (which they
approximately are — the slices are drawn independently), then
    Σ_full  ≈  V_audit_only  +  V_calib_only  +  V_eval_only  +  cross
where `cross` captures the higher-order interactions (e.g., t̂'s
fluctuation when ĥ shifts). The S4 sanity probe verifies cross is
small (< 20% of Σ_full's trace) on uniform Y.

This module is the first-class version of S4's ablation. The output is
useful as ground truth against which to measure each Ω̂ method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._harness import (
    fit_calibrator, sample, sample_logger, saddle_argmax, g_pair,
)
from ._sanity import _empirical_sigma_full_and_eps
from ._truths import DGPParams, TargetPert


@dataclass
class TruthDecomp:
    """Ground-truth variance decomposition output."""
    V_audit_only: np.ndarray     # (2,2) — N1 contribution
    V_calib_only: np.ndarray     # (2,2) — N2 contribution
    V_eval_only:  np.ndarray     # (2,2) — N3 contribution
    Sigma_full:   np.ndarray     # (2,2) — joint, including cross-terms
    V_predicted:  np.ndarray     # (2,2) — V_audit + V_calib + V_eval (additive prediction)
    cross_residual: np.ndarray   # (2,2) — Sigma_full − V_predicted (cross-term magnitude)
    R: int                       # MC reps used per component
    # Diagnostics:
    trace_audit: float
    trace_calib: float
    trace_eval:  float
    trace_full:  float
    trace_predicted: float
    rel_err_predicted: float     # |trace_full - trace_predicted| / trace_full


def _g_from_one_realization(s_a, y_a, s_e, cal, t_grid, alpha):
    """
    Standard pipeline: predict on EVAL, re-max t̂, evaluate g on AUDIT.
    Returns ḡ as a 2-vector.
    """
    H_e = cal.predict(s_e)
    j_star, t_star = saddle_argmax(H_e, t_grid, alpha)
    h_t_a = cal.predict(s_a)[:, j_star]
    g1, g2 = g_pair(s_a, y_a, h_t_a, t_star, alpha)
    return np.array([g1.mean(), g2.mean()])


def compute_truth_decomposition(
    p: DGPParams, pert: TargetPert, alpha: float,
    n_calib: int, n_audit: int, n_eval: int,
    K: int, t_grid: np.ndarray, B: int, R: int, seed_base: int,
) -> TruthDecomp:
    """
    Compute the ground-truth variance decomposition.

    Math
    ----
        Σ_full := n_audit · sample-cov_r(ḡ^(r))
                  where ḡ^(r) is from the FULL pipeline (all three slices
                  re-sampled per r).

        V_audit_only := n_audit · sample-cov_r(ḡ^(r))
                        with one fixed (calib, eval) realization,
                        AUDIT re-sampled per r.

        V_calib_only := n_audit · sample-cov_r(ḡ^(r))
                        with one fixed (audit, eval) realization,
                        CALIB re-sampled per r.

        V_eval_only := n_audit · sample-cov_r(ḡ^(r))
                       with one fixed (calib, audit) realization,
                       EVAL re-sampled per r.

        V_predicted := V_audit_only + V_calib_only + V_eval_only
        cross_residual := Σ_full − V_predicted

    The pinned realizations are drawn independently across the four
    branches, so the V_audit/V_calib/V_eval estimates are not
    deterministically related to Σ_full. Independence is a feature,
    not a bug — we want to see the pure nuisance contribution from each.

    Cost: 4 · R reps. Embarrassingly parallel.
    """
    # ---- (1) Σ_full from the full-pipeline MC --------------------------------
    _, sigma_full, _, _ = _empirical_sigma_full_and_eps(
        p, pert, alpha, n_calib, n_audit, n_eval,
        t_grid, K, B, R, seed_base,
    )

    # ---- (2) V_audit_only: pin (calib, eval), vary AUDIT ---------------------
    s_c_pin, y_c_pin = sample_logger(p, n_calib, seed=seed_base + 100)
    cal_pin = fit_calibrator(s_c_pin, y_c_pin, t_grid, K, seed=seed_base + 100)
    s_e_pin, _ = sample(p, pert, n_eval, seed=seed_base + 101, with_oracle=False)
    H_e_pin = cal_pin.predict(s_e_pin)
    j_pin, t_pin = saddle_argmax(H_e_pin, t_grid, alpha)

    g_audit_only = []
    for r in range(R):
        s_a, y_a = sample(p, pert, n_audit, seed=seed_base + 200 + 31 * r,
                          with_oracle=True)
        h_t_a = cal_pin.predict(s_a)[:, j_pin]
        g1, g2 = g_pair(s_a, y_a, h_t_a, t_pin, alpha)
        g_audit_only.append([g1.mean(), g2.mean()])
    V_audit = n_audit * np.cov(np.array(g_audit_only), rowvar=False, ddof=1)

    # ---- (3) V_calib_only: pin (audit, eval), vary CALIB ---------------------
    s_a_pin2, y_a_pin2 = sample(p, pert, n_audit, seed=seed_base + 300,
                                with_oracle=True)
    s_e_pin2, _ = sample(p, pert, n_eval, seed=seed_base + 301,
                         with_oracle=False)

    g_calib_only = []
    for r in range(R):
        s_c_r, y_c_r = sample_logger(p, n_calib, seed=seed_base + 400 + 71 * r)
        cal_r = fit_calibrator(s_c_r, y_c_r, t_grid, K,
                               seed=seed_base + 400 + 71 * r)
        # Same pipeline: predict on the pinned EVAL, re-max t̂_r, score
        # on the pinned AUDIT at t̂_r.
        H_e_r = cal_r.predict(s_e_pin2)
        j_r, t_r = saddle_argmax(H_e_r, t_grid, alpha)
        h_t_a_r = cal_r.predict(s_a_pin2)[:, j_r]
        g1, g2 = g_pair(s_a_pin2, y_a_pin2, h_t_a_r, t_r, alpha)
        g_calib_only.append([g1.mean(), g2.mean()])
    V_calib = n_audit * np.cov(np.array(g_calib_only), rowvar=False, ddof=1)

    # ---- (4) V_eval_only: pin (calib, audit), vary EVAL ----------------------
    s_c_pin2, y_c_pin2 = sample_logger(p, n_calib, seed=seed_base + 500)
    cal_pin2 = fit_calibrator(s_c_pin2, y_c_pin2, t_grid, K,
                              seed=seed_base + 500)

    g_eval_only = []
    for r in range(R):
        s_e_r, _ = sample(p, pert, n_eval, seed=seed_base + 600 + 71 * r,
                          with_oracle=False)
        H_e_r = cal_pin2.predict(s_e_r)
        j_r, t_r = saddle_argmax(H_e_r, t_grid, alpha)
        h_t_a_r = cal_pin2.predict(s_a_pin2)[:, j_r]
        g1, g2 = g_pair(s_a_pin2, y_a_pin2, h_t_a_r, t_r, alpha)
        g_eval_only.append([g1.mean(), g2.mean()])
    V_eval = n_audit * np.cov(np.array(g_eval_only), rowvar=False, ddof=1)

    # ---- Summary ------------------------------------------------------------
    V_predicted = V_audit + V_calib + V_eval
    cross_residual = sigma_full - V_predicted
    trace_full = float(np.trace(sigma_full))
    trace_pred = float(np.trace(V_predicted))
    rel_err = abs(trace_pred - trace_full) / max(trace_full, 1e-30)

    return TruthDecomp(
        V_audit_only=V_audit,
        V_calib_only=V_calib,
        V_eval_only=V_eval,
        Sigma_full=sigma_full,
        V_predicted=V_predicted,
        cross_residual=cross_residual,
        R=R,
        trace_audit=float(np.trace(V_audit)),
        trace_calib=float(np.trace(V_calib)),
        trace_eval=float(np.trace(V_eval)),
        trace_full=trace_full,
        trace_predicted=trace_pred,
        rel_err_predicted=rel_err,
    )
