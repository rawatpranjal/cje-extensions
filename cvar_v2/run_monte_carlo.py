"""Outer Monte Carlo loop on the semi-synthetic Arena DGP.

Cells (smoke = default; --full enables the wider sweep):

  size_diagnostic : calib=base, eval=base, δ=0  — measures empirical audit size
                    under the truest null available (matches the test in
                    cvar/tests_dgp.py). Sanity for nominal-vs-empirical gap.

  power_curve     : calib=base, eval=clone, δ ∈ {0, 0.02, 0.05, 0.10, 0.20},
                    perturbation="tail". Reject rate vs δ. δ=0 here also
                    contributes a per-policy size estimate (with natural
                    Y-marginal mis-spec on top, distinct from base→base).

  scaling         : calib=base, eval=clone, δ=0.05, n_eval ∈ {500, 2500},
                    perturbation="tail". Power vs sample size at fixed δ.

CSV columns:
  cell_kind, calib_policy, eval_policy, alpha, delta, perturbation,
  n_eval, n_oracle, mc_seed, cvar_truth, cvar_est, cvar_ci_lo,
  cvar_ci_hi, ci_covers_truth, audit_p, audit_reject, t_hat, runtime_sec

Run:
  python3.11 cvar/run_monte_carlo.py [--full] [--n-workers N] [--out cvar/results_mc.csv]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

sys.path.insert(0, "cvar")
from dgp import (  # noqa: E402
    fit_arena_dgp, sample_synthetic, cvar_truth, q_lower_tail_threshold,
)
from workhorse import (  # noqa: E402
    estimate_direct_cvar_isotonic, two_moment_wald_audit,
    two_moment_wald_audit_xf, cluster_bootstrap_cvar,
)

DATA = Path.home() / "Dropbox" / "cvar-cje-data" / "cje-arena-experiments" / "data"

# Smoke vs medium vs full configuration
SMOKE  = dict(mc_reps=30,  B_inner=80,  n_eval_main=2000)
MEDIUM = dict(mc_reps=100, B_inner=150, n_eval_main=2000)
FULL   = dict(mc_reps=200, B_inner=500, n_eval_main=5000)


@dataclass(frozen=True)
class Cell:
    cell_kind: str
    calib_policy: str
    eval_policy: str
    alpha: float
    delta: float
    perturbation: str
    n_eval: int
    n_oracle: int


def build_smoke_cells(cfg: dict) -> list[Cell]:
    """8 cells — fast scaffold check."""
    n_eval = cfg["n_eval_main"]
    n_oracle = n_eval // 4
    cells: list[Cell] = []

    # 1. base→base size diagnostic (one cell)
    cells.append(Cell("size_diagnostic", "base", "base", 0.10, 0.0, "none", n_eval, n_oracle))

    # 2. power curve (5 δ values, clone)
    for d in (0.0, 0.02, 0.05, 0.10, 0.20):
        cells.append(Cell("power_curve", "base", "clone", 0.10, d, "tail", n_eval, n_oracle))

    # 3. scaling (2 n values at δ=0.05, clone)
    for n in (500, 2500):
        cells.append(Cell("scaling", "base", "clone", 0.10, 0.05, "tail", n, max(50, n // 4)))

    return cells


def build_medium_cells(cfg: dict) -> list[Cell]:
    """25 cells: power curve × 4 targets, size diagnostic, scaling on
    {clone, unhelpful}. α=0.10 only. ~25–40 min wall-clock at 6 workers
    with cross-fit audit. Scope: enough to fill the targets table in
    `cvar/power_targets.md`."""
    n_eval = cfg["n_eval_main"]
    n_oracle = n_eval // 4
    targets = ("clone", "premium", "parallel_universe_prompt", "unhelpful")
    cells: list[Cell] = []

    # 1. base→base size diagnostic
    cells.append(Cell("size_diagnostic", "base", "base", 0.10, 0.0, "none", n_eval, n_oracle))

    # 2. power curve × 4 targets × 5 δ
    for tgt in targets:
        for d in (0.0, 0.02, 0.05, 0.10, 0.20):
            cells.append(Cell("power_curve", "base", tgt, 0.10, d, "tail", n_eval, n_oracle))

    # 3. scaling: clone + unhelpful at 2 n values
    for tgt in ("clone", "unhelpful"):
        for n in (500, 2500):
            cells.append(Cell("scaling", "base", tgt, 0.10, 0.05, "tail", n, max(50, n // 4)))

    return cells


def build_full_cells(cfg: dict) -> list[Cell]:
    """Full sweep: power curve × 4 targets, scaling × 4 targets, multi-α."""
    n_eval = cfg["n_eval_main"]
    n_oracle = n_eval // 4
    targets = ("clone", "premium", "parallel_universe_prompt", "unhelpful")
    cells: list[Cell] = []

    # base→base size diagnostic — multi-α
    for a in (0.05, 0.10, 0.20):
        cells.append(Cell("size_diagnostic", "base", "base", a, 0.0, "none", n_eval, n_oracle))

    # power curve, all targets, multi-α
    for tgt in targets:
        for a in (0.05, 0.10, 0.20):
            for d in (0.0, 0.02, 0.05, 0.10, 0.20):
                cells.append(Cell("power_curve", "base", tgt, a, d, "tail", n_eval, n_oracle))

    # scaling, all targets, α=0.10 only
    for tgt in targets:
        for n in (500, 1000, 2500, 5000):
            cells.append(Cell("scaling", "base", tgt, 0.10, 0.05, "tail", n, max(50, n // 4)))

    return cells


# Module-level globals populated in the parent before forking the pool.
# With fork-context multiprocessing, workers inherit these via copy-on-write
# and don't need any pickle round-trip.
_DGPS: Optional[dict] = None
_Q_LOW_BY_ALPHA: Optional[dict[float, float]] = None
_B_INNER: int = 100
_GRID_SIZE: int = 31


def _run_one(arg: tuple[Cell, int, float]) -> dict:
    cell, mc_seed, truth = arg
    rng = np.random.default_rng(mc_seed)
    t0 = time.time()
    s_calib, y_calib = sample_synthetic(_DGPS[cell.calib_policy], cell.n_oracle, rng)

    # m_override = base for the eval policy when calib_policy == "base"
    # (ensures the only mis-specification source is δ × perturbation, not the
    # natural m_p heterogeneity across policies).
    m_override = _DGPS[cell.calib_policy] if cell.eval_policy != cell.calib_policy else None
    q_low = _Q_LOW_BY_ALPHA[cell.alpha] if cell.perturbation == "tail" else None

    s_eval, y_eval = sample_synthetic(
        _DGPS[cell.eval_policy], cell.n_eval, rng,
        delta=cell.delta, perturbation=cell.perturbation,
        q_low_threshold=q_low, m_override=m_override,
    )

    cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(
        s_calib, y_calib, s_eval, cell.alpha, _GRID_SIZE,
    )
    ci_lo, ci_hi, _ = cluster_bootstrap_cvar(
        s_calib, y_calib, s_eval,
        eval_cluster=np.arange(cell.n_eval),
        train_cluster=np.arange(cell.n_oracle),
        alpha=cell.alpha, grid_size=_GRID_SIZE, B=_B_INNER, seed=mc_seed,
    )
    # Run BOTH audits per replicate so we can directly compare the
    # naive Σ̂ to the cross-fit Σ̂ on the same data.
    audit_naive = two_moment_wald_audit(
        s_calib, y_calib, s_eval, y_eval, t_hat, cell.alpha,
    )
    audit_xf = two_moment_wald_audit_xf(
        s_calib, y_calib, s_eval, y_eval, t_hat, cell.alpha,
        B=80, fold_seed=mc_seed,
    )

    return {
        **asdict(cell),
        "mc_seed": mc_seed,
        "cvar_truth": truth,
        "cvar_est": cvar_est,
        "cvar_ci_lo": ci_lo,
        "cvar_ci_hi": ci_hi,
        "ci_covers_truth": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi)
                                and ci_lo <= truth <= ci_hi),
        "audit_p": audit_naive["p_value"],
        "audit_reject": audit_naive["reject"],
        "audit_p_xf": audit_xf["p_value"],
        "audit_reject_xf": audit_xf["reject"],
        "t_hat": t_hat,
        "runtime_sec": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full",   action="store_true", help="Run full sweep")
    mode.add_argument("--medium", action="store_true", help="Run medium sweep (4 targets, 100 reps)")
    parser.add_argument("--n-workers", type=int, default=min(8, mp.cpu_count()),
                        help="multiprocessing pool size")
    parser.add_argument("--out", type=Path, default=Path("cvar/results_mc.csv"))
    args = parser.parse_args()

    if args.full:
        cfg, cells, mode_name = FULL, build_full_cells(FULL), "FULL"
    elif args.medium:
        cfg, cells, mode_name = MEDIUM, build_medium_cells(MEDIUM), "MEDIUM"
    else:
        cfg, cells, mode_name = SMOKE, build_smoke_cells(SMOKE), "SMOKE"

    print(f"Mode: {mode_name}")
    print(f"  mc_reps={cfg['mc_reps']}  B_inner={cfg['B_inner']}  workers={args.n_workers}")
    print(f"  cells={len(cells)}  total_outer_reps={len(cells) * cfg['mc_reps']}")

    print("Fitting Arena DGP...")
    global _DGPS, _Q_LOW_BY_ALPHA, _B_INNER, _GRID_SIZE
    _DGPS = fit_arena_dgp(DATA)
    _Q_LOW_BY_ALPHA = {a: q_lower_tail_threshold(_DGPS["base"], a) for a in (0.05, 0.10, 0.20)}
    _B_INNER = cfg["B_inner"]
    _GRID_SIZE = 31

    # Pre-compute truth per cell (depends on eval_dgp + alpha; deterministic)
    cell_truths: dict[Cell, float] = {}
    for cell in cells:
        cell_truths[cell] = cvar_truth(_DGPS[cell.eval_policy], cell.alpha, n_truth=200_000)
    print("Per-cell truths cached.")

    tasks: list[tuple[Cell, int, float]] = []
    for ci, cell in enumerate(cells):
        for r in range(cfg["mc_reps"]):
            mc_seed = ci * 100_000 + r
            tasks.append((cell, mc_seed, cell_truths[cell]))

    print(f"Total tasks: {len(tasks)}.  Spawning fork-pool ({args.n_workers} workers)...")
    t0 = time.time()
    rows: list[dict] = []
    if args.n_workers <= 1:
        # Serial path — easier to debug and small enough at smoke scale.
        for i, t in enumerate(tasks):
            rows.append(_run_one(t))
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(tasks) - (i + 1)) / rate
                print(f"  {i+1:4d}/{len(tasks)}  elapsed={el:6.1f}s  ETA={eta:6.1f}s")
    else:
        # Fork context: workers inherit the parent's _DGPS via copy-on-write,
        # so no pickle round-trip and no spawn-import storm.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=4)):
                rows.append(row)
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    el = time.time() - t0
                    rate = (i + 1) / el
                    eta = (len(tasks) - (i + 1)) / rate
                    print(f"  {i+1:4d}/{len(tasks)}  elapsed={el:6.1f}s  ETA={eta:6.1f}s")

    df = pl.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

    # Stdout summary: size, coverage, power-at-δ=0.10
    summary = (
        df.group_by(["cell_kind", "eval_policy", "delta", "alpha"])
        .agg([
            pl.col("audit_reject").mean().alias("reject_rate"),
            pl.col("ci_covers_truth").mean().alias("ci_coverage"),
            (pl.col("cvar_est") - pl.col("cvar_truth")).abs().mean().alias("mean_abs_err"),
            pl.len().alias("n"),
        ])
        .sort(["cell_kind", "eval_policy", "alpha", "delta"])
    )
    print("\nSummary:")
    print(summary.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
