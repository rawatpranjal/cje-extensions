"""Print a comparison table across alphas from cvar_v3/results_arena.csv.

For each (target policy, alpha) cell, aggregates the 20 seeds and reports:
  - median Mean estimate
  - Mean truth (full-oracle empirical mean of fresh draws)
  - median CVaR estimate
  - CVaR truth
  - 95% CI half-width (median across seeds)
  - audit p-value (median) and reject rate
"""
from __future__ import annotations
import sys
from pathlib import Path
import polars as pl

CSV = Path("cvar_v3/results_arena.csv")


def main() -> int:
    if not CSV.exists():
        print(f"FAIL: {CSV} not found.", file=sys.stderr)
        return 1
    df = pl.read_csv(CSV)
    alphas = sorted(df["alpha"].unique().to_list())
    n_seeds = df["seed"].n_unique()

    agg = (
        df.group_by(["policy", "alpha"])
        .agg([
            pl.col("mean").median().alias("mean_med"),
            pl.col("oracle_truth").first().alias("mean_truth"),
            pl.col("cvar").median().alias("cvar_med"),
            pl.col("cvar_empirical_truth").first().alias("cvar_truth"),
            ((pl.col("cvar_ci_hi") - pl.col("cvar_ci_lo")) / 2.0).median().alias("cvar_hw"),
            pl.col("audit_p_value").median().alias("audit_p_med"),
            pl.col("audit_reject").mean().alias("reject_rate"),
        ])
        .sort(["policy", "alpha"])
    )

    print(f"Arena CVaR sweep across α — {n_seeds} seeds, n_eval=5000.")
    print()
    print(f"{'policy':<28} {'α':>5}  {'Ĉ mean':>8} {'V truth':>8}  {'Ĉ CVaR':>9} {'CVaR tr':>9} {'CI hw':>7}  {'p_med':>7} {'reject':>7}")
    print("-" * 110)
    last = None
    for r in agg.iter_rows(named=True):
        if last is not None and r["policy"] != last:
            print()
        last = r["policy"]
        print(
            f"{r['policy']:<28} {r['alpha']:>5.2f}  "
            f"{r['mean_med']:>8.4f} {r['mean_truth']:>8.4f}  "
            f"{r['cvar_med']:>9.4f} {r['cvar_truth']:>9.4f} "
            f"{r['cvar_hw']:>7.4f}  "
            f"{r['audit_p_med']:>7.3g} {r['reject_rate']:>7.0%}"
        )

    print()
    print("Pairs to inspect (similar Mean truth, different CVaR truth):")
    truths = (
        df.group_by(["policy", "alpha"])
        .agg([pl.col("oracle_truth").first().alias("mean_truth"),
              pl.col("cvar_empirical_truth").first().alias("cvar_truth")])
    )
    for a in alphas:
        sub = truths.filter(pl.col("alpha") == a).sort("policy")
        rows = list(sub.iter_rows(named=True))
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                a_mean = abs(rows[i]["mean_truth"] - rows[j]["mean_truth"])
                a_cvar = abs(rows[i]["cvar_truth"] - rows[j]["cvar_truth"])
                if a_mean < 0.02 and a_cvar > 0.01:
                    print(
                        f"  α={a:.2f}: {rows[i]['policy']} vs {rows[j]['policy']}  "
                        f"|ΔV|={a_mean:.4f}  |ΔCVaR|={a_cvar:.4f}"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
