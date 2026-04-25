"""Scale n on the synthetic DGP — does CVaR resolve clone vs premium?

Real Arena is fixed at n=5,000 fresh draws per policy. At that n the
estimator gives clone and premium near-identical CVaR estimates at deep
alpha levels even though the underlying truths differ. This script asks
at what n the gap becomes resolvable, by sampling from the per-policy
DGPs at progressively larger n.

The DGP preserves each policy's empirical Y marginal exactly, so the
clone-vs-premium CVaR truth gap is identical to the Arena gap. The
synthetic part is how S is generated from Y.

Cloud-friendly. Uses fork-pool multiprocessing and detects vCPU count.

Usage (laptop):
    python3.11 cvar/n_sweep_synthetic.py
Usage (cloud, fully parallel on a 64-vCPU box):
    bash cvar/run_n_sweep_cloud.sh

Outputs `cvar/results_n_sweep.csv` with one row per (policy, n, alpha,
seed) and prints a small "do they separate" summary.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "cvar")
from dgp import fit_arena_dgp, sample_synthetic, cvar_truth  # noqa: E402
from workhorse import (  # noqa: E402
    estimate_direct_cvar_isotonic,
    cluster_bootstrap_cvar,
)

POLICIES = ("clone", "premium")
ALPHAS = (0.005, 0.01, 0.05)
N_EVALS = (5_000, 25_000, 100_000, 500_000)
N_ORACLE = 1_250  # ~realistic Arena oracle slice (25% of 5,000)
GRID_SIZE = 61

# Set in main(); fork-inherited by workers.
_DGPS: dict | None = None
_B: int = 200


@dataclass(frozen=True)
class Task:
    policy: str
    n_eval: int
    alpha: float
    seed: int


def _run_one(task: Task) -> dict:
    assert _DGPS is not None
    rng = np.random.default_rng(task.seed * 100_003 + hash(task.policy) % 10_000)
    s_o, y_o = sample_synthetic(_DGPS["base"], N_ORACLE, rng)
    s_e, _ = sample_synthetic(_DGPS[task.policy], task.n_eval, rng)

    t0 = time.time()
    est, t_hat, _, _ = estimate_direct_cvar_isotonic(
        s_o, y_o, s_e, task.alpha, grid_size=GRID_SIZE,
    )
    train_cluster = np.arange(len(s_o))
    eval_cluster = np.arange(len(s_e))
    ci_lo, ci_hi, n_fail = cluster_bootstrap_cvar(
        s_o, y_o, s_e, eval_cluster, train_cluster,
        task.alpha, GRID_SIZE, _B, seed=task.seed * 7 + 13,
    )
    dt = time.time() - t0
    return {
        "policy": task.policy,
        "n_eval": task.n_eval,
        "alpha": task.alpha,
        "seed": task.seed,
        "cvar_est": float(est),
        "t_hat": float(t_hat),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_boot_fail": int(n_fail),
        "runtime_sec": float(dt),
    }


def main() -> int:
    global _DGPS, _B
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=mp.cpu_count(),
                    help="Number of fork-pool workers. Default: detected vCPU count.")
    ap.add_argument("--seeds", type=int, default=20,
                    help="Number of MC seeds per cell. Default: 20.")
    ap.add_argument("--b", type=int, default=200,
                    help="Bootstrap reps per CI. Default: 200.")
    ap.add_argument("--data", type=Path,
                    default=Path.home() / "Dropbox" / "cvar-cje-data"
                    / "cje-arena-experiments" / "data",
                    help="Path to Arena data directory.")
    ap.add_argument("--out", type=Path, default=Path("cvar/results_n_sweep.csv"),
                    help="Output CSV path.")
    args = ap.parse_args()
    _B = args.b

    print(f"n-sweep synthetic: policies={POLICIES} alphas={ALPHAS} "
          f"n_evals={N_EVALS} seeds={args.seeds} n_oracle={N_ORACLE} B={args.b} "
          f"workers={args.n_workers}")
    print(f"Fitting Arena DGPs from {args.data} ...")
    _DGPS = fit_arena_dgp(args.data)

    print("Population CVaR truths (one-shot at n_truth=200,000):")
    truths: dict[tuple[str, float], float] = {}
    for pol in POLICIES:
        for a in ALPHAS:
            truths[(pol, a)] = cvar_truth(_DGPS[pol], a, n_truth=200_000)
    header = "  policy".ljust(12) + "".join(f"{f'α={a}':>10}" for a in ALPHAS)
    print(header)
    for pol in POLICIES:
        line = f"  {pol:<10}" + "".join(f"{truths[(pol, a)]:>10.4f}" for a in ALPHAS)
        print(line)
    for a in ALPHAS:
        gap = truths[("premium", a)] - truths[("clone", a)]
        print(f"  gap@α={a}: {gap:+.4f}")

    tasks = [
        Task(pol, n, a, s)
        for pol in POLICIES
        for n in N_EVALS
        for a in ALPHAS
        for s in range(args.seeds)
    ]
    print(f"\nTotal tasks: {len(tasks)}. Spawning fork-pool ({args.n_workers} workers)...")

    t0 = time.time()
    rows: list[dict] = []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.n_workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(tasks) - (i + 1)) / rate
                print(f"  {i+1:4d}/{len(tasks)}  elapsed={el:6.1f}s  ETA={eta:6.1f}s",
                      flush=True)

    df = pl.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out)
    print(f"\nWrote {len(df)} rows to {args.out} in {time.time()-t0:.1f}s.\n")

    n_nan = df.filter(pl.col("cvar_est").is_nan() | pl.col("ci_lo").is_nan()).height
    if n_nan:
        print(f"WARNING: {n_nan} rows with NaN estimates or CIs.")

    print("Per-cell summary (median across seeds):")
    print(f"{'policy':<10} {'α':>6} {'n':>8} {'truth':>8} {'est':>8} "
          f"{'CI_lo':>8} {'CI_hi':>8} {'hw':>7}")
    for pol in POLICIES:
        for a in ALPHAS:
            for n in N_EVALS:
                sub = df.filter(
                    (pl.col("policy") == pol)
                    & (pl.col("alpha") == a)
                    & (pl.col("n_eval") == n)
                )
                est = sub["cvar_est"].median()
                lo = sub["ci_lo"].median()
                hi = sub["ci_hi"].median()
                hw = (hi - lo) / 2.0
                tr = truths[(pol, a)]
                print(f"{pol:<10} {a:>6.3f} {n:>8d} {tr:>8.4f} {est:>8.4f} "
                      f"{lo:>8.4f} {hi:>8.4f} {hw:>7.4f}")

    print("\nClone vs premium separation table:")
    print(f"{'α':>6} {'n':>8} {'clone est [CI]':<26} {'premium est [CI]':<26} {'CIs disjoint?':>14}")
    for a in ALPHAS:
        for n in N_EVALS:
            cl = df.filter((pl.col("policy") == "clone") & (pl.col("alpha") == a) & (pl.col("n_eval") == n))
            pr = df.filter((pl.col("policy") == "premium") & (pl.col("alpha") == a) & (pl.col("n_eval") == n))
            cl_lo, cl_hi = cl["ci_lo"].median(), cl["ci_hi"].median()
            pr_lo, pr_hi = pr["ci_lo"].median(), pr["ci_hi"].median()
            cl_est = cl["cvar_est"].median()
            pr_est = pr["cvar_est"].median()
            disjoint = (cl_hi < pr_lo) or (pr_hi < cl_lo)
            cls = f"{cl_est:.4f} [{cl_lo:.4f},{cl_hi:.4f}]"
            prs = f"{pr_est:.4f} [{pr_lo:.4f},{pr_hi:.4f}]"
            print(f"{a:>6.3f} {n:>8d} {cls:<26} {prs:<26} {'YES' if disjoint else 'no':>14}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
