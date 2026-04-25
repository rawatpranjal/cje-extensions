"""Run Direct Mean + Direct CVaR-CJE on the authors' Arena data.

For each of the 4 target policies, runs the Mean via `cje-eval`'s
`CalibratedDirectEstimator` and the CVaR via our workhorse, for α ∈
{0.05, 0.10, 0.20}, across `N_SEEDS` oracle-fold seeds.

Writes `cvar/results_arena.csv` (long format, one row per policy × seed × alpha).

Pass conditions (per the plan):
  - 4 × N_SEEDS × 3 = (4 * N_SEEDS * 3) rows, no NaNs in estimates or CIs.
  - No bootstrap with > 10% failure rate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, "cvar")
from workhorse import estimate_all_policies  # noqa: E402

DATA_ROOT = Path.home() / "Dropbox" / "cvar-cje-data" / "cje-arena-experiments" / "data"
ORACLE_COVERAGE = 0.25
ALPHAS = (0.01, 0.05, 0.10, 0.20)
N_SEEDS = 20
B = 500  # profiled: bootstrap CIs stable from ~200 reps; 500 gives comfortable headroom and ~30 min total runtime
OUT_CSV = Path("cvar/results_arena.csv")


def main() -> int:
    t0 = time.time()
    print(f"Arena CVaR run: coverage={ORACLE_COVERAGE}, αs={ALPHAS}, seeds={N_SEEDS}, B={B}")
    print(f"Data: {DATA_ROOT}")

    all_rows: list[dict] = []
    for seed in range(N_SEEDS):
        t_seed = time.time()
        results = estimate_all_policies(
            data_root=DATA_ROOT,
            oracle_coverage=ORACLE_COVERAGE,
            alphas=ALPHAS,
            B=B,
            seed=seed,
            verbose=False,
        )
        for pe in results:
            all_rows.extend(pe.to_rows())
        dt = time.time() - t_seed
        print(f"  seed {seed:2d} done in {dt:.1f}s  (elapsed {time.time()-t0:.1f}s)")

    df = pl.DataFrame(all_rows)
    df.write_csv(OUT_CSV)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print(f"Total runtime: {time.time()-t0:.1f}s")

    # Pass checks
    expected = 4 * N_SEEDS * len(ALPHAS)
    if len(df) != expected:
        print(f"❌ FAIL: expected {expected} rows, got {len(df)}")
        return 1
    n_nan_mean = df.filter(pl.col("mean").is_nan() | pl.col("mean").is_null()).height
    n_nan_cvar = df.filter(pl.col("cvar").is_nan() | pl.col("cvar").is_null()).height
    n_nan_cvar_ci = df.filter(
        pl.col("cvar_ci_lo").is_nan() | pl.col("cvar_ci_lo").is_null()
    ).height
    max_boot_fail = df["n_bootstrap_failures"].max()
    if n_nan_mean or n_nan_cvar or n_nan_cvar_ci:
        print(f"❌ FAIL: NaNs — mean={n_nan_mean}, cvar={n_nan_cvar}, cvar_ci={n_nan_cvar_ci}")
        return 1
    if max_boot_fail is not None and max_boot_fail > B // 10:
        print(f"❌ FAIL: max bootstrap failures {max_boot_fail} > {B//10}")
        return 1

    print(
        f"✓ Pass: {len(df)} rows, no NaNs, max bootstrap failures per run = {max_boot_fail}"
    )

    # Quick summary: median (2.5%, 97.5%) across seeds, per policy × alpha
    agg = (
        df.group_by(["policy", "alpha"])
        .agg(
            [
                pl.col("mean").median().alias("mean_median"),
                pl.col("cvar").median().alias("cvar_median"),
                pl.col("cvar").quantile(0.025).alias("cvar_p025"),
                pl.col("cvar").quantile(0.975).alias("cvar_p975"),
                pl.col("cvar_empirical_truth").median().alias("cvar_truth_median"),
                pl.col("audit_p_value").median().alias("audit_p_median"),
                pl.col("audit_reject").mean().alias("audit_reject_rate"),
            ]
        )
        .sort(["alpha", "policy"])
    )
    print("\nAggregated summary:")
    print(agg.to_pandas().to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
