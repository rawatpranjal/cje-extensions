"""Outer Monte Carlo loop on the semi-synthetic HealthBench DGP.

Output: cvar_v4/eda/deeper/mc_validation/results_mc.csv

Modes:
  --smoke   (default): 20 reps × ~10 cells, B_ci=80, B_audit=40, ≈ 4 min
  --medium             : 100 reps × full cell grid, B_ci=200, B_audit=80
  --full               : 200 reps × full cell grid, B_ci=500, B_audit=120

Cells:
  coverage : delta=0, all (calib=base, eval=π') × n × oracle_coverage × α × design.
             Each cell records null-case Mean and CVaR coverage + RMSE +
             audit size (false-positive rate at the null).
  power    : delta sweep on calib=base, eval=clone with perturbation ∈ {tail, uniform}.
             Records audit power and (less importantly) coverage degradation.

CSV row schema: see pipeline_step.run_one return value.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from .dgp import (
    fit_healthbench_dgp, cvar_truth, mean_truth, q_lower_tail_threshold, PolicyDGP,
)
from .pipeline_step import Cell, run_one


# Mode configurations
SMOKE  = dict(mc_reps=20,  B_ci=80,  B_audit=40,  K_jackknife=5, grid_size=31)
# MEDIUM tuned to a ~45–60 min wall-clock on an 8-core laptop. Wider coverage
# tables come from FULL on a cloud box.
MEDIUM = dict(mc_reps=60,  B_ci=120, B_audit=60,  K_jackknife=5, grid_size=31)
FULL   = dict(mc_reps=200, B_ci=500, B_audit=120, K_jackknife=5, grid_size=31)


CALIB_POLICY = "base"
TARGET_POLICIES = ("base", "clone", "premium", "parallel_universe_prompt", "unhelpful", "risky")


def build_smoke_cells() -> list[Cell]:
    """Smoke: minimal grid for plumbing checks."""
    cells: list[Cell] = []
    # Self-consistency null
    cells.append(Cell("coverage", "base", "base", 0.10, 0.0, "none", 250, 0.25, "uniform"))
    # One cell per target at fixed knobs
    for tgt in ("clone", "unhelpful", "risky"):
        cells.append(Cell("coverage", "base", tgt, 0.10, 0.0, "none", 250, 0.25, "uniform"))
    # Two power cells
    for d in (0.0, 0.10):
        cells.append(Cell("power", "base", "clone", 0.10, d, "tail", 250, 0.25, "uniform"))
    # One floor_tail_band check
    cells.append(Cell("coverage", "base", "clone", 0.10, 0.0, "none", 250, 0.25, "floor_tail_band"))
    return cells


def build_medium_cells() -> list[Cell]:
    """Medium: full coverage grid + power sweep, but fewer (n, cov) levels.

    96 coverage cells + 10 power cells = 106 cells.
    """
    cells: list[Cell] = []
    # Coverage: 6 policies × 2 n × 2 cov × 2 α × 2 design
    for tgt in TARGET_POLICIES:
        for n in (250, 500):
            for cov in (0.10, 0.25):
                for alpha in (0.10, 0.30):
                    for design in ("uniform", "floor_tail_band"):
                        cells.append(Cell(
                            "coverage", "base", tgt,
                            alpha, 0.0, "none",
                            n, cov, design,
                        ))
    # Power: clone target only, fixed knobs, sweep delta × perturbation
    for d in (0.0, 0.025, 0.05, 0.10, 0.20):
        for pert in ("tail", "uniform"):
            cells.append(Cell(
                "power", "base", "clone",
                0.10, d, pert,
                500, 0.25, "uniform",
            ))
    return cells


def build_full_cells() -> list[Cell]:
    """Full: medium grid extended with (n=1000) and α=0.05 cells."""
    cells = build_medium_cells()
    # Add α=0.05 (deeper tail) at n=500, cov=0.25
    for tgt in TARGET_POLICIES:
        for design in ("uniform", "floor_tail_band"):
            cells.append(Cell(
                "coverage", "base", tgt,
                0.05, 0.0, "none",
                500, 0.25, design,
            ))
    # Add n=1000 cells at α=0.10, cov=0.25, uniform
    for tgt in TARGET_POLICIES:
        cells.append(Cell(
            "coverage", "base", tgt,
            0.10, 0.0, "none",
            1000, 0.25, "uniform",
        ))
    return cells


# Module-level state for fork-pool workers (copy-on-write — no pickle).
_DGPS: Optional[dict[str, PolicyDGP]] = None
_TRUTHS: Optional[dict[tuple[str, float], tuple[float, float]]] = None
_Q_LOW: Optional[dict[float, float]] = None
_CFG: dict = SMOKE


def _worker(args: tuple[Cell, int]) -> dict:
    cell, mc_seed = args
    return run_one(
        cell, mc_seed, _DGPS,
        truths=_TRUTHS, q_low_by_alpha=_Q_LOW,
        B_ci=_CFG["B_ci"], B_audit=_CFG["B_audit"],
        K_jackknife=_CFG["K_jackknife"], grid_size=_CFG["grid_size"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="(default) 20 reps × small cell set")
    mode.add_argument("--medium", action="store_true", help="100 reps × full cell grid")
    mode.add_argument("--full", action="store_true", help="200 reps × extended grid")
    parser.add_argument("--n-workers", type=int, default=min(8, mp.cpu_count()),
                        help="multiprocessing fork-pool size")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "results_mc.csv",
    )
    args = parser.parse_args()

    if args.full:
        cfg, cells, mode_name = FULL, build_full_cells(), "FULL"
    elif args.medium:
        cfg, cells, mode_name = MEDIUM, build_medium_cells(), "MEDIUM"
    else:
        cfg, cells, mode_name = SMOKE, build_smoke_cells(), "SMOKE"

    print(f"Mode: {mode_name}")
    print(f"  mc_reps={cfg['mc_reps']}  B_ci={cfg['B_ci']}  B_audit={cfg['B_audit']}  "
          f"workers={args.n_workers}")
    print(f"  cells={len(cells)}  total_outer_reps={len(cells) * cfg['mc_reps']}")

    print("Fitting HealthBench DGP...")
    global _DGPS, _TRUTHS, _Q_LOW, _CFG
    _DGPS = fit_healthbench_dgp()
    missing = [p for p in TARGET_POLICIES if p not in _DGPS]
    if missing:
        print(f"  WARNING: missing policies {missing} — skipping cells that need them")
        cells = [c for c in cells if c.eval_policy not in missing]
    _CFG = cfg

    # Pre-compute per-cell truths and q_low thresholds.
    print("Computing per-cell truths (n_truth=200_000)...")
    _Q_LOW = {a: q_lower_tail_threshold(_DGPS[CALIB_POLICY], a)
              for a in {c.alpha for c in cells}}
    _TRUTHS = {}
    needed = {(c.eval_policy, c.alpha) for c in cells}
    for tgt, alpha in needed:
        _TRUTHS[(tgt, alpha)] = (
            mean_truth(_DGPS[tgt], n_truth=200_000),
            cvar_truth(_DGPS[tgt], alpha, n_truth=200_000),
        )
    print(f"  cached {len(_TRUTHS)} (policy, α) pairs.")

    # Build task list
    tasks: list[tuple[Cell, int]] = []
    for ci, cell in enumerate(cells):
        for r in range(cfg["mc_reps"]):
            mc_seed = ci * 100_000 + r
            tasks.append((cell, mc_seed))
    print(f"Total tasks: {len(tasks)}")

    t0 = time.time()
    rows: list[dict] = []
    if args.n_workers <= 1:
        for i, t in enumerate(tasks):
            rows.append(_worker(t))
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(tasks) - (i + 1)) / rate if rate > 0 else 0.0
                print(f"  {i+1:5d}/{len(tasks)}  elapsed={el:6.1f}s  ETA={eta:6.1f}s")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
                rows.append(row)
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    el = time.time() - t0
                    rate = (i + 1) / el
                    eta = (len(tasks) - (i + 1)) / rate if rate > 0 else 0.0
                    print(f"  {i+1:5d}/{len(tasks)}  elapsed={el:6.1f}s  ETA={eta:6.1f}s")

    df = pl.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

    # Quick stdout summary
    valid = df.filter(pl.col("skip_reason") == "")
    if len(valid) == 0:
        print("WARNING: zero successful rows — check skip_reason column")
        return
    summary = (
        valid.group_by(["cell_kind", "eval_policy", "alpha", "delta", "perturbation"])
        .agg([
            pl.col("mean_covers").mean().alias("mean_cov"),
            pl.col("cvar_covers").mean().alias("cvar_cov"),
            pl.col("mean_audit_reject").mean().alias("mean_reject"),
            pl.col("cvar_audit_reject").mean().alias("cvar_reject"),
            pl.col("abs_err_mean").mean().alias("rmse_mean_proxy"),
            pl.col("abs_err_cvar").mean().alias("rmse_cvar_proxy"),
            pl.len().alias("n_reps"),
        ])
        .sort(["cell_kind", "eval_policy", "alpha", "delta"])
    )
    print("\nSummary (per-cell pooled across n/cov/design):")
    print(summary.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
