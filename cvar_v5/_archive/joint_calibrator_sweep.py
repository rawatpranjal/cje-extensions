"""
Joint-calibrator comparison sweep (ARCHIVED).

Compared four alt methods against the per-t isotonic baseline.
Outcome: no method met the bar (`var_calib_ratio < 0.7` on ≥ 3 of 4
policies AND `rmse_ratio ≤ 1.05` AND `time_ratio ≤ 5`). See
`cvar_v5/TODO.md::[joint-calibrator]`.

Preserved here for reference. Not part of the production CLI.
To run deliberately:
    cd cvar_v5/_archive && python -m cvar_v5._archive.joint_calibrator_sweep
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from cvar_v5._archive.calibrator_alt import (
    fit_bivariate_isotonic_pooled,
    fit_distribution_regression,
    fit_gbm_monotone,
    fit_smoothed_per_t,
)
from cvar_v5.cvar_cje.calibrator import fit_calibrator_grid
from cvar_v5.cvar_cje.estimator import estimate_direct_cvar
from cvar_v5.cvar_cje.schema import Slice
from cvar_v5.mc import _io
from cvar_v5.mc.dgp import DEFAULT_POLICIES, DGP


LOG = logging.getLogger("cvar_v5.mc")


def _per_t_isotonic_factory(s, y, t_grid):
    return fit_calibrator_grid(s, y, t_grid)


def _smoothed_per_t_factory(s, y, t_grid):
    return fit_smoothed_per_t(s, y, t_grid, window=3)


def _distribution_regression_factory(s, y, t_grid):
    return fit_distribution_regression(s, y, t_grid, n_y_grid=21)


def _bivariate_isotonic_factory(s, y, t_grid):
    return fit_bivariate_isotonic_pooled(s, y, t_grid, max_iter=5)


def _gbm_monotone_factory(s, y, t_grid):
    return fit_gbm_monotone(s, y, t_grid, n_estimators=100)


METHODS: dict[str, Callable] = {
    "per_t_isotonic":             _per_t_isotonic_factory,
    "smoothed_per_t":             _smoothed_per_t_factory,
    "distribution_regression":    _distribution_regression_factory,
    "bivariate_isotonic_pooled":  _bivariate_isotonic_factory,
    "gbm_monotone":               _gbm_monotone_factory,
}


_ACC_T_VALUES = (0.10, 0.30, 0.50, 1.00)
_N_ORACLE = 600
_N_EVAL = 1000
_N_HUGE = 50_000
_R = 20
_ALPHA = 0.10
_T_GRID = np.linspace(0.0, 1.0, 61)


def _run_cell(
    method, policy, rep, cg_huge, eval_frozen_slice, s_test,
    H_huge_at_panel, panel_t_indices,
):
    dgp = DGP(DEFAULT_POLICIES)
    factory = METHODS[method]
    seed = 5003 * rep + 41

    oracle = dgp.sample(policy, n=_N_ORACLE, with_oracle=True, seed=seed)
    eval_df = dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=seed + 1)

    s_calib = oracle["s"].to_numpy()
    y_calib = oracle["y"].to_numpy()

    t0 = time.perf_counter()
    cg_real = factory(s_calib, y_calib, _T_GRID)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = cg_real.predict(s_test)
    pred_time = time.perf_counter() - t0

    eval_slice = Slice(df=eval_df, role="eval")
    est_real = estimate_direct_cvar(eval_slice, cg_real, _ALPHA).value
    est_eval = estimate_direct_cvar(eval_slice, cg_huge, _ALPHA).value
    est_cal = estimate_direct_cvar(eval_frozen_slice, cg_real, _ALPHA).value

    H_real = cg_real.predict(s_test)
    l2_per_t = [
        float(np.sqrt(np.mean((H_real[:, j] - H_huge_at_panel[:, idx]) ** 2)))
        for idx, j in enumerate(panel_t_indices)
    ]
    bias_per_t = [
        float(abs(np.mean(H_real[:, j] - H_huge_at_panel[:, idx])))
        for idx, j in enumerate(panel_t_indices)
    ]

    return {
        "method": method, "policy": policy, "rep": rep,
        "est_real": est_real, "est_eval_only": est_eval, "est_calib_only": est_cal,
        "truth": dgp.truth_cvar(policy, _ALPHA),
        "fit_time_ms": fit_time * 1000.0, "pred_time_ms": pred_time * 1000.0,
        "l2_gap_avg": float(np.mean(l2_per_t)),
        "bias_func_avg": float(np.mean(bias_per_t)),
    }


def run_sweep(out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = _io.make_run_dir("joint_calibrator")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    _io.setup_logging(run_dir=out_dir)
    LOG.info("[ARCHIVED] joint-calibrator sweep")

    panel_t_indices = [int(np.argmin(abs(_T_GRID - t))) for t in _ACC_T_VALUES]
    rows = []
    t_start = time.time()

    for policy in DEFAULT_POLICIES:
        dgp = DGP(DEFAULT_POLICIES)
        df_huge = dgp.sample(policy, n=_N_HUGE, with_oracle=True, seed=99999)
        s_huge = df_huge["s"].to_numpy()
        y_huge = df_huge["y"].to_numpy()
        s_test = dgp.sample(policy, n=500, with_oracle=False, seed=12345)["s"].to_numpy()
        eval_frozen_df = dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=88888)
        eval_frozen_slice = Slice(df=eval_frozen_df, role="eval")

        for method in METHODS:
            cg_huge = METHODS[method](s_huge, y_huge, _T_GRID)
            H_huge = cg_huge.predict(s_test)
            H_huge_at_panel = H_huge[:, panel_t_indices]

            for rep in range(_R):
                rows.append(_run_cell(
                    method, policy, rep, cg_huge, eval_frozen_slice,
                    s_test, H_huge_at_panel, panel_t_indices,
                ))
            LOG.info("  %s on %s: R=%d done (elapsed %.1fs)",
                     method, policy, _R, time.time() - t_start)

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "results.csv")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_sweep(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
