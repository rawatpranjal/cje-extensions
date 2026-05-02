"""Inner Monte Carlo step — runs the full cvar_v4 pipeline once.

Mirrors cvar_v4/healthbench_data/analyze.py:step5_oracle_calibrated_designed
but with synthetic (S, Y) panels generated from the DGP instead of read from
disk, so per-cell ground-truth Mean and CVaR are known.

Per rep b:
  1. Sample n_total (S, Y) pairs from the LOGGER (base) DGP.
  2. Sample n_total (S, Y) pairs from the TARGET (eval_policy) DGP, optionally
     mis-specified via (delta, perturbation) and m_override.
  3. Apply oracle_design.select_slice on the logger panel → s_train, y_train,
     w_train (HT 1/π weights).
  4. Apply oracle_design.select_slice on the target panel → s_audit, y_audit,
     w_audit.
  5. Direct CVaR-CJE: estimate_direct_cvar_isotonic on (s_train, y_train,
     s_eval=s_target_full).  bootstrap_cvar_ci for Var_eval.
     jackknife_var_cal for Var_cal.  Total CI = est ± 1.96 √(Var_eval + Var_cal).
  6. Direct Mean-CJE: fit_isotonic_mean → mean_est on s_target_full.
     bootstrap_mean_ci for Var_eval.  jackknife_var_cal_mean for Var_cal.
  7. CVaR audit: two_moment_wald_audit_xf on (train slice, audit slice, t̂).
  8. Mean audit: mean_transport_audit on (train slice, audit slice).

Returns one dict (one CSV row).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# Estimator stack (no modifications)
from cvar_v4.eda.deeper._estimator import (
    estimate_direct_cvar_isotonic,
    fit_isotonic_mean,
    bootstrap_cvar_ci,
    bootstrap_mean_ci,
    jackknife_var_cal,
    jackknife_var_cal_mean,
    two_moment_wald_audit_xf,
    mean_transport_audit,
)

# Designed oracle slice
from cvar_v4.healthbench_data.oracle_design import select_slice

from .dgp import PolicyDGP, sample_synthetic


@dataclass(frozen=True)
class Cell:
    cell_kind: str          # "coverage" | "power"
    calib_policy: str       # always "base" for v4
    eval_policy: str
    alpha: float
    delta: float
    perturbation: str       # "none" | "uniform" | "tail"
    n_total: int            # size of each panel (logger + target)
    oracle_coverage: float
    design: str             # "uniform" | "floor_tail" | "floor_tail_band"


def _slice_to_arrays(panel_s: np.ndarray, panel_y: np.ndarray,
                     rows: list[dict], slices) -> tuple:
    """Convert a list[DesignedSlice] back to (s_sel, y_sel, w_sel)."""
    sel_mask = np.array([s.selected for s in slices])
    sel_pi = np.array([s.pi for s in slices])
    if sel_mask.sum() == 0:
        return None  # caller handles
    s_sel = panel_s[sel_mask]
    y_sel = panel_y[sel_mask]
    w_sel = 1.0 / sel_pi[sel_mask]
    return s_sel, y_sel, w_sel


def run_one(
    cell: Cell,
    mc_seed: int,
    dgps: dict[str, PolicyDGP],
    *,
    truths: Optional[dict[tuple[str, float], tuple[float, float]]] = None,
    q_low_by_alpha: Optional[dict[float, float]] = None,
    B_ci: int = 200,
    B_audit: int = 80,
    K_jackknife: int = 5,
    grid_size: int = 31,
) -> dict:
    """Run one MC replicate. Returns a flat row dict for CSV output.

    Args:
      cell: Cell describing the configuration.
      mc_seed: Per-rep seed; used for sampling AND for the slice selection
          AND for both bootstraps (so the rep is fully reproducible).
      dgps: dict of fitted PolicyDGPs. Must include "base" and cell.eval_policy.
      truths: optional pre-computed {(eval_policy, alpha): (mean_truth, cvar_truth)}.
          If None, truths are computed inline (slow — use the runner's cache).
      q_low_by_alpha: pre-computed q_α(Y_base) for tail perturbations.
      B_ci, B_audit, K_jackknife, grid_size: tuning knobs.
    """
    rng = np.random.default_rng(mc_seed)
    t0 = time.time()

    base_dgp = dgps[cell.calib_policy]
    target_dgp = dgps[cell.eval_policy]

    # --- 1. logger panel
    s_log, y_log = sample_synthetic(base_dgp, cell.n_total, rng)

    # --- 2. target panel
    # m_override=base ensures the calibrator's m̂_base transports cleanly when
    # delta=0 — the only mis-specification source is delta × perturbation.
    m_override = base_dgp if cell.eval_policy != cell.calib_policy else None
    q_low = (q_low_by_alpha or {}).get(cell.alpha) if cell.perturbation == "tail" else None
    s_tgt, y_tgt = sample_synthetic(
        target_dgp, cell.n_total, rng,
        delta=cell.delta, perturbation=cell.perturbation,
        q_low_threshold=q_low, m_override=m_override,
    )

    # --- 3. designed slice on logger (calibration training panel)
    log_rows = [
        {"prompt_id": f"log_{i}", "policy": cell.calib_policy,
         "cheap_score": float(s_log[i]), "oracle_score": float(y_log[i])}
        for i in range(cell.n_total)
    ]
    log_slices = select_slice(
        log_rows, design=cell.design, coverage=cell.oracle_coverage,
        alpha=cell.alpha, seed=mc_seed,
    )
    res_log = _slice_to_arrays(s_log, y_log, log_rows, log_slices)
    if res_log is None or sum(s.selected for s in log_slices) < 5:
        return _empty_row(cell, mc_seed, t0, reason="log_slice_too_small")
    s_train, y_train, w_train = res_log

    # --- 4. designed slice on target (audit panel)
    tgt_rows = [
        {"prompt_id": f"tgt_{i}", "policy": cell.eval_policy,
         "cheap_score": float(s_tgt[i]), "oracle_score": float(y_tgt[i])}
        for i in range(cell.n_total)
    ]
    tgt_slices = select_slice(
        tgt_rows, design=cell.design, coverage=cell.oracle_coverage,
        alpha=cell.alpha, seed=mc_seed + 1,
    )
    res_tgt = _slice_to_arrays(s_tgt, y_tgt, tgt_rows, tgt_slices)
    if res_tgt is None or sum(s.selected for s in tgt_slices) < 3:
        return _empty_row(cell, mc_seed, t0, reason="audit_slice_too_small")
    s_audit, y_audit, w_audit = res_tgt

    # --- 5. CVaR estimator + CI (calibration-aware)
    try:
        cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(
            s_train, y_train, s_tgt, cell.alpha, grid_size,
            sample_weight_train=w_train,
        )
        cvar_ci = bootstrap_cvar_ci(
            s_train, y_train, s_tgt, cell.alpha,
            sample_weight_train=w_train, B=B_ci, seed=mc_seed,
            grid_size=grid_size,
        )
        cvar_var_cal = jackknife_var_cal(
            s_train, y_train, s_tgt, cell.alpha,
            sample_weight_oracle=w_train, K=K_jackknife, seed=mc_seed,
            grid_size=grid_size,
        )
        cvar_var_eval = cvar_ci["var_eval"]
        cvar_var_total = float(cvar_var_eval + (cvar_var_cal if math.isfinite(cvar_var_cal) else 0.0))
        cvar_hw = 1.96 * math.sqrt(max(cvar_var_total, 0.0))
        cvar_ci_lo_total = float(cvar_est - cvar_hw)
        cvar_ci_hi_total = float(cvar_est + cvar_hw)

        # CVaR audit at t̂
        cvar_audit = two_moment_wald_audit_xf(
            s_train, y_train, s_audit, y_audit, t_hat, cell.alpha,
            B=B_audit, fold_seed=mc_seed,
            sample_weight_train=w_train, sample_weight_audit=w_audit,
        )
    except Exception as e:  # noqa: BLE001
        return _empty_row(cell, mc_seed, t0, reason=f"cvar_path:{type(e).__name__}")

    # --- 6. Mean estimator + CI (calibration-aware)
    try:
        f_hat_pred = fit_isotonic_mean(s_train, y_train, s_tgt, sample_weight=w_train)
        mean_est = float(f_hat_pred.mean())
        mean_ci = bootstrap_mean_ci(
            s_train, y_train, s_tgt,
            sample_weight_train=w_train, B=B_ci, seed=mc_seed,
        )
        mean_var_cal = jackknife_var_cal_mean(
            s_train, y_train, s_tgt,
            sample_weight_oracle=w_train, K=K_jackknife, seed=mc_seed,
        )
        mean_var_eval = mean_ci["var_eval"]
        mean_var_total = float(mean_var_eval + (mean_var_cal if math.isfinite(mean_var_cal) else 0.0))
        mean_hw = 1.96 * math.sqrt(max(mean_var_total, 0.0))
        mean_ci_lo_total = float(mean_est - mean_hw)
        mean_ci_hi_total = float(mean_est + mean_hw)

        mean_audit = mean_transport_audit(
            s_train, y_train, s_audit, y_audit,
            sample_weight_train=w_train, sample_weight_audit=w_audit,
        )
    except Exception as e:  # noqa: BLE001
        return _empty_row(cell, mc_seed, t0, reason=f"mean_path:{type(e).__name__}")

    # --- 7. truths (precomputed cache, fall back to inline if missing)
    if truths is not None and (cell.eval_policy, cell.alpha) in truths:
        m_truth, c_truth = truths[(cell.eval_policy, cell.alpha)]
    else:
        from .dgp import cvar_truth as _ct, mean_truth as _mt
        m_truth = _mt(target_dgp, n_truth=200_000)
        c_truth = _ct(target_dgp, cell.alpha, n_truth=200_000)

    return {
        **asdict(cell),
        "mc_seed": mc_seed,
        "n_log_slice": int(sum(s.selected for s in log_slices)),
        "n_audit_slice": int(sum(s.selected for s in tgt_slices)),
        # Truth
        "mean_truth": float(m_truth),
        "cvar_truth": float(c_truth),
        # Mean estimand
        "mean_est": mean_est,
        "mean_var_eval": float(mean_var_eval),
        "mean_var_cal": float(mean_var_cal),
        "mean_var_total": mean_var_total,
        "mean_ci_lo": float(mean_ci["ci_lo"]),
        "mean_ci_hi": float(mean_ci["ci_hi"]),
        "mean_ci_lo_total": mean_ci_lo_total,
        "mean_ci_hi_total": mean_ci_hi_total,
        "mean_covers": bool(mean_ci_lo_total <= m_truth <= mean_ci_hi_total),
        "mean_covers_eval": bool(mean_ci["ci_lo"] <= m_truth <= mean_ci["ci_hi"]),
        "mean_audit_p": float(mean_audit["p_value"]),
        "mean_audit_reject": bool(mean_audit["reject"]),
        "mean_residual": float(mean_audit["residual_mean"]),
        # CVaR estimand
        "cvar_est": float(cvar_est),
        "t_hat": float(t_hat),
        "cvar_var_eval": float(cvar_var_eval),
        "cvar_var_cal": float(cvar_var_cal),
        "cvar_var_total": cvar_var_total,
        "cvar_ci_lo": float(cvar_ci["ci_lo"]),
        "cvar_ci_hi": float(cvar_ci["ci_hi"]),
        "cvar_ci_lo_total": cvar_ci_lo_total,
        "cvar_ci_hi_total": cvar_ci_hi_total,
        "cvar_covers": bool(cvar_ci_lo_total <= c_truth <= cvar_ci_hi_total),
        "cvar_covers_eval": bool(cvar_ci["ci_lo"] <= c_truth <= cvar_ci["ci_hi"]),
        "cvar_audit_p": float(cvar_audit["p_value"]),
        "cvar_audit_reject": bool(cvar_audit["reject"]),
        "cvar_audit_g1": float(cvar_audit.get("mean_g1", float("nan"))),
        "cvar_audit_g2": float(cvar_audit.get("mean_g2", float("nan"))),
        # Bookkeeping
        "abs_err_mean": float(abs(mean_est - m_truth)),
        "abs_err_cvar": float(abs(cvar_est - c_truth)),
        "runtime_sec": float(time.time() - t0),
        "skip_reason": "",
    }


def _empty_row(cell: Cell, mc_seed: int, t0: float, *, reason: str) -> dict:
    """Failure stub — schema-compatible with success rows."""
    nan = float("nan")
    return {
        **asdict(cell),
        "mc_seed": mc_seed,
        "n_log_slice": 0,
        "n_audit_slice": 0,
        "mean_truth": nan, "cvar_truth": nan,
        "mean_est": nan, "mean_var_eval": nan, "mean_var_cal": nan, "mean_var_total": nan,
        "mean_ci_lo": nan, "mean_ci_hi": nan,
        "mean_ci_lo_total": nan, "mean_ci_hi_total": nan,
        "mean_covers": False, "mean_covers_eval": False,
        "mean_audit_p": nan, "mean_audit_reject": False, "mean_residual": nan,
        "cvar_est": nan, "t_hat": nan,
        "cvar_var_eval": nan, "cvar_var_cal": nan, "cvar_var_total": nan,
        "cvar_ci_lo": nan, "cvar_ci_hi": nan,
        "cvar_ci_lo_total": nan, "cvar_ci_hi_total": nan,
        "cvar_covers": False, "cvar_covers_eval": False,
        "cvar_audit_p": nan, "cvar_audit_reject": False,
        "cvar_audit_g1": nan, "cvar_audit_g2": nan,
        "abs_err_mean": nan, "abs_err_cvar": nan,
        "runtime_sec": float(time.time() - t0),
        "skip_reason": reason,
    }
