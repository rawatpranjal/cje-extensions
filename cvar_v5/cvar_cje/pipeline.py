"""
End-to-end orchestrator for the Direct CVaR-CJE estimator + audit.

Wires together:
    1. Input validation (Y ∈ [0,1], required columns).
    2. partition_oracle  → (CALIB, AUDIT, CrossFitFolds)   [audit-holdout disjoint]
    3. fit_calibrator_grid on CALIB with K-fold cross-fit.
    4. estimate_direct_cvar on EVAL → EstimateResult.
    5. two_moment_wald_audit on AUDIT at t̂_α → AuditVerdict.

This module contains no math beyond input validation. All math contracts
live in calibrator.py / estimator.py / audit.py.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._crossfit import partition_oracle
from .audit import two_moment_wald_audit
from .calibrator import fit_calibrator_grid, make_t_grid
from .config import Config
from .estimator import estimate_direct_cvar
from .schema import AuditVerdict, EstimateResult, Slice


def _validate_oracle(oracle: pl.DataFrame) -> None:
    required = {"prompt_id", "s", "y"}
    missing = required - set(oracle.columns)
    if missing:
        raise ValueError(f"oracle DataFrame missing columns: {sorted(missing)}")
    y = oracle["y"].to_numpy()
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError(f"oracle y must be in [0, 1]; got [{y.min()}, {y.max()}]")
    if oracle["prompt_id"].n_unique() != oracle.height:
        raise ValueError(
            "oracle DataFrame must have unique prompt_id (one row per prompt)"
        )


def _validate_eval(eval_df: pl.DataFrame) -> None:
    required = {"prompt_id", "s"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval DataFrame missing columns: {sorted(missing)}")


def run_pipeline(
    oracle_rows: pl.DataFrame,
    eval_rows: pl.DataFrame,
    cfg: Config,
) -> tuple[EstimateResult, AuditVerdict]:
    """
    Run the full Direct CVaR-CJE pipeline at α = cfg.alpha.

    Args:
        oracle_rows: target-policy rows with oracle labels. Required columns:
            prompt_id (unique), s (cheap-judge score), y (oracle label, [0,1]).
        eval_rows:   target-policy fresh draws. Required columns:
            prompt_id, s. (No y required.)
        cfg:         Config (paper-default values; advanced flags disabled).

    Returns:
        (EstimateResult, AuditVerdict).
    """
    # --- (1) validate ---------------------------------------------------------
    _validate_oracle(oracle_rows)
    _validate_eval(eval_rows)

    # --- (2) partition oracle into CALIB ⫫ AUDIT -----------------------------
    calib_slice, audit_slice, folds = partition_oracle(
        oracle_rows, K=cfg.K, seed=cfg.seed
    )

    # Runtime invariant: prompt_id sets disjoint.
    calib_ids = set(calib_slice.prompt_ids())
    audit_ids = set(audit_slice.prompt_ids())
    overlap = calib_ids & audit_ids
    if overlap:
        raise AssertionError(
            f"audit-holdout discipline violated: "
            f"{len(overlap)} prompt_ids in both CALIB and AUDIT"
        )

    # --- (3) build grid + fit calibrator on CALIB with cross-fit -------------
    t_grid = make_t_grid(cfg.grid_lo, cfg.grid_hi, cfg.grid_size, cfg.grid_kind)
    cg = fit_calibrator_grid(
        s_calib=calib_slice.s(),
        y_calib=calib_slice.y(),
        t_grid=t_grid,
        fold_id=folds.fold_id,
        K=cfg.K,
    )

    # --- (4) estimate on EVAL -------------------------------------------------
    eval_slice = Slice(df=eval_rows, role="eval")
    estimate = estimate_direct_cvar(eval_slice, cg, alpha=cfg.alpha)

    # --- (5) audit on AUDIT at t̂_α -------------------------------------------
    verdict = two_moment_wald_audit(
        audit_slice=audit_slice,
        calibrator=cg,
        t_hat=estimate.threshold,
        alpha=cfg.alpha,
        omega_estimator=cfg.omega_estimator,
        B=cfg.audit_B,
        seed=cfg.seed,
        eta=cfg.audit_eta,
    )

    return estimate, verdict
