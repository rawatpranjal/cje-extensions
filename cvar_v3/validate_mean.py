"""Blocking benchmark: reproduce the authors' Direct Mean CJE results.

Runs `workhorse.estimate_all_policies` over a small set of seeds at
oracle_coverage=0.25 (headline paper setting) and compares the across-seed
median Mean estimate per policy to the oracle ground truth from
full-oracle fresh draws.

Multiple seeds are required because in the authors' protocol the 25%-oracle
slice varies with the experiment seed (`random.seed(spec.seed)` at
`base.py:100`). Single-seed CIs occasionally miss truth by ≤0.001 even when
the calibrator is paper-faithful; the across-seed median is the right gate.

Pass condition:
  - For all non-adversarial policies (clone, premium, parallel_universe_prompt),
    |median(mean) - oracle_truth| ≤ 0.01 AND the across-seed P2.5/P97.5
    range of estimates contains the oracle truth.
  - `unhelpful` is exempt (catastrophic shift).

Writes `cvar_v3/validate_mean_report.md` with the comparison table.
Exits non-zero if the benchmark fails — this blocks downstream steps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "cvar")
from workhorse import estimate_all_policies  # noqa: E402

DATA_ROOT = Path.home() / "Dropbox" / "cvar-cje-data" / "cje-arena-experiments" / "data"
ORACLE_COVERAGE = 0.25
SEEDS = (0, 1, 2, 3, 4)
TOLERANCE = 0.01
ADVERSARIAL = {"unhelpful"}


def main() -> int:
    print(
        f"Validating Direct Mean CJE against oracle truth "
        f"(coverage={ORACLE_COVERAGE}, seeds={list(SEEDS)})"
    )
    print(f"Data: {DATA_ROOT}")

    # Per-seed results. Each call uses mask_seed=seed (authors' protocol).
    rows: list[dict] = []
    for s in SEEDS:
        out = estimate_all_policies(
            data_root=DATA_ROOT,
            oracle_coverage=ORACLE_COVERAGE,
            alphas=(0.10,),
            B=20,  # tiny — Mean is what we gate on; CVaR CIs aren't used here
            seed=s,
            verbose=False,
        )
        for pe in out:
            rows.append(
                {
                    "policy": pe.policy,
                    "seed": s,
                    "mean": pe.mean,
                    "ci_lo": pe.mean_ci_lo,
                    "ci_hi": pe.mean_ci_hi,
                    "oracle_truth": pe.oracle_truth,
                }
            )

    df = pl.DataFrame(rows)

    # Aggregate across seeds: median + slice variability.
    agg = (
        df.group_by("policy")
        .agg(
            [
                pl.col("mean").median().alias("mean_median"),
                pl.col("mean").min().alias("mean_min"),
                pl.col("mean").max().alias("mean_max"),
                pl.col("oracle_truth").first().alias("oracle_truth"),
            ]
        )
        .sort("policy")
    )

    summary = []
    for r in agg.iter_rows(named=True):
        delta = abs(r["mean_median"] - r["oracle_truth"])
        slice_contains = r["mean_min"] <= r["oracle_truth"] <= r["mean_max"]
        adversarial = r["policy"] in ADVERSARIAL
        if adversarial:
            passes = True
            note = "exempt (catastrophic shift)"
        else:
            passes = (delta <= TOLERANCE) and slice_contains
            note = ""
        summary.append(
            {
                "policy": r["policy"],
                "median_mean": r["mean_median"],
                "slice_lo": r["mean_min"],
                "slice_hi": r["mean_max"],
                "oracle_truth": r["oracle_truth"],
                "abs_delta": delta,
                "slice_contains_truth": slice_contains,
                "pass": passes,
                "note": note,
            }
        )

    summary_df = pl.DataFrame(summary)
    print("\n" + summary_df.to_pandas().to_string(index=False))

    all_pass = all(r["pass"] for r in summary)

    # Report
    report_path = Path("cvar_v3/validate_mean_report.md")
    lines = [
        "# Validate Mean CJE — blocking benchmark report",
        "",
        f"- Oracle coverage: {ORACLE_COVERAGE}",
        f"- Seeds: {list(SEEDS)} (each seed varies fold assignment AND oracle slice, per `base.py:100`)",
        "- Estimator: `CalibratedDirectEstimator` (`cje-eval==0.2.10`)",
        f"- Tolerance: |median(Mean) − oracle truth| ≤ {TOLERANCE} AND across-seed [min, max] ⊇ oracle truth",
        f"- Adversarial policies exempt: {sorted(ADVERSARIAL)}",
        "",
        "| Policy | Median Mean | Across-seed [min, max] | Oracle truth | \\|Δ\\| | range ⊇ truth | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| `{r['policy']}` | {r['median_mean']:.4f} | "
            f"[{r['slice_lo']:.4f}, {r['slice_hi']:.4f}] | "
            f"{r['oracle_truth']:.4f} | "
            f"{r['abs_delta']:.4f} | "
            f"{'✓' if r['slice_contains_truth'] else '✗'} | "
            f"{'✓' if r['pass'] else '✗'}"
            + (f" _({r['note']})_" if r['note'] else "")
            + " |"
        )
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")

    if not all_pass:
        print("\n❌ BENCHMARK FAILED — do not proceed to CVaR run.")
        return 1
    print("\n✓ Benchmark passed. Safe to proceed to cvar_v3/run_arena.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
