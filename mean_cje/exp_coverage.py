"""
Ablation runner. Six configurations × 4 policies × 5 oracle sizes.
Outputs to mean_cje/runs/<ts>/results.csv with one row per cell.

Configurations (see spec):
    1 plug-in   + IF only         + Wald
    2 plug-in   + IF + jackknife  + Wald
    3 AIPW      + IF only         + Wald
    4 AIPW      + IF + jackknife  + Wald
    5 plug-in   + bootstrap
    6 AIPW      + bootstrap

For each cell: bias, RMSE, coverage at 95%, mean CI half-width.

CLI:
    python -m mean_cje.exp_coverage -w 4
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from cvar_v5.mc.dgp import DEFAULT_POLICIES, DGP
from .lib.calibrator import Calibrator
from .lib.estimators import plug_in_mean, aipw_one_step
from .lib.variance import (
    var_eval_if, var_cal_jackknife, wald_ci,
    bootstrap_ci_plugin, bootstrap_ci_aipw,
)


LOG = logging.getLogger("mean_cje")


# Sweep parameters
_POLICIES = list(DEFAULT_POLICIES.keys())
_N_ORACLES = (50, 100, 250, 500, 1000)
_N_EVAL = 2000
_R_WALD = 200
_R_BOOT = 50
_B = 500
_K = 5
_LEVEL = 0.95


CONFIGS = [
    {"id": 1, "estimator": "plug_in", "variance": "if",      "ci": "wald"},
    {"id": 2, "estimator": "plug_in", "variance": "if+jack", "ci": "wald"},
    {"id": 3, "estimator": "aipw",    "variance": "if",      "ci": "wald"},
    {"id": 4, "estimator": "aipw",    "variance": "if+jack", "ci": "wald"},
    {"id": 5, "estimator": "plug_in", "variance": "boot",    "ci": "bootstrap"},
    {"id": 6, "estimator": "aipw",    "variance": "boot",    "ci": "bootstrap"},
]


def _make_run_dir() -> Path:
    base = Path(__file__).parent / "runs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out = base / ts
    n = 2
    while out.exists():
        out = base / f"{ts}_{n}"
        n += 1
    out.mkdir()
    return out


def _setup_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "log.txt", mode="w"),
        ],
    )


def _one_rep(config: dict, policy: str, n_oracle: int, rep: int) -> dict:
    """One (config, policy, n_oracle, rep) cell. Returns one row."""
    dgp = DGP(DEFAULT_POLICIES)
    seed = 7919 * rep + 31
    oracle = dgp.sample(policy, n=n_oracle, with_oracle=True, seed=seed)
    eval_df = dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=seed + 1)
    s_oracle = oracle["s"].to_numpy()
    y_oracle = oracle["y"].to_numpy()
    s_eval = eval_df["s"].to_numpy()

    cal = Calibrator().fit(s_oracle, y_oracle, K=_K, seed=seed)

    # Point estimate
    if config["estimator"] == "plug_in":
        v_hat = plug_in_mean(s_eval, cal)
    else:
        v_hat = aipw_one_step(s_eval, s_oracle, y_oracle, cal)

    # CI
    if config["ci"] == "wald":
        v_eval = var_eval_if(s_eval, cal)
        v_cal = (
            var_cal_jackknife(s_eval, s_oracle, y_oracle, cal,
                              estimator=config["estimator"])
            if config["variance"] == "if+jack" else 0.0
        )
        lo, hi = wald_ci(v_hat, v_eval + v_cal, level=_LEVEL)
    else:
        if config["estimator"] == "plug_in":
            lo, hi = bootstrap_ci_plugin(s_eval, s_oracle, y_oracle,
                                         B=_B, K=_K, seed=seed, level=_LEVEL)
        else:
            lo, hi = bootstrap_ci_aipw(s_eval, s_oracle, y_oracle,
                                       B=_B, K=_K, seed=seed, level=_LEVEL)

    truth = dgp.truth_mean(policy)
    return {
        "config_id": config["id"],
        "estimator": config["estimator"],
        "variance": config["variance"],
        "ci": config["ci"],
        "policy": policy,
        "n_oracle": n_oracle,
        "rep": rep,
        "v_hat": float(v_hat),
        "truth": float(truth),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def _cells_to_run() -> list[tuple]:
    cells = []
    for config in CONFIGS:
        R = _R_BOOT if config["ci"] == "bootstrap" else _R_WALD
        for policy, n_oracle in product(_POLICIES, _N_ORACLES):
            for rep in range(R):
                cells.append((config, policy, n_oracle, rep))
    return cells


def _run_cell(args: tuple) -> dict:
    return _one_rep(*args)


def _aggregate(rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.group_by(["config_id", "estimator", "variance", "ci", "policy", "n_oracle"])
        .agg([
            (pl.col("v_hat") - pl.col("truth")).mean().alias("bias"),
            ((pl.col("v_hat") - pl.col("truth")) ** 2).mean().sqrt().alias("rmse"),
            ((pl.col("ci_lo") <= pl.col("truth")) & (pl.col("truth") <= pl.col("ci_hi")))
                .mean().alias("coverage"),
            ((pl.col("ci_hi") - pl.col("ci_lo")) / 2).mean().alias("mean_half_width"),
            pl.len().alias("R"),
        ])
        .sort(["config_id", "policy", "n_oracle"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="mean_cje coverage ablation")
    parser.add_argument("-w", "--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or _make_run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)
    LOG.info("mean_cje coverage ablation; out_dir=%s, workers=%d",
             out_dir, args.n_workers)

    cells = _cells_to_run()
    LOG.info("total cells: %d (across 6 configs × 4 policies × 5 oracle sizes)",
             len(cells))

    t0 = time.time()
    rows: list[dict] = []
    if args.n_workers <= 1:
        for i, c in enumerate(cells):
            rows.append(_run_cell(c))
            if (i + 1) % max(1, len(cells) // 20) == 0:
                LOG.info("  %d/%d (%.1f%%, %.1fs)", i+1, len(cells),
                         100*(i+1)/len(cells), time.time()-t0)
    else:
        ctx = get_context("fork")
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_cell, cells, chunksize=4)):
                rows.append(row)
                if (i + 1) % max(1, len(cells) // 20) == 0:
                    LOG.info("  %d/%d (%.1f%%, %.1fs)", i+1, len(cells),
                             100*(i+1)/len(cells), time.time()-t0)

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "results.csv")
    summary = _aggregate(df)
    summary.write_csv(out_dir / "summary.csv")
    LOG.info("wrote results.csv and summary.csv in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
