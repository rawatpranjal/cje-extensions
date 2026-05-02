"""
Aggregate `results_mc.csv` into `mc_validation.md` (markdown only — no LaTeX).

Math contract (aggregations over R replicates):
    bias_{p,α,ω}    = mean_r(estimate)  −  truth_cvar(p, α)
    rmse_{p,α,ω}    = sqrt( mean_r((estimate − truth_cvar)²) )
    abs_error_p99   = 99th percentile of |estimate − truth| (diagnostic)
    size_{p,α,ω}    = mean_r( decision == "REFUSE-LEVEL" )      at δ=0
    power_{p,α,ω,δ} = mean_r( decision == "REFUSE-LEVEL" )      at δ>0

G1 check (α=1 collapse identity):
    max over (policy, rep, ω) of  |estimate − truth_mean|  at α=1.
    Spec gate: ≤ 1e-9.

The MC harness records `truth_cvar(α=1) = truth_mean` (exact algebraic
identity in mc.dgp), so abs_error at α=1 is the α=1 collapse error directly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from . import _io


LOG = logging.getLogger("cvar_v5.mc")


def _fmt(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def _fmt_sci(x: float) -> str:
    return f"{x:.2e}" if np.isfinite(x) else "—"


def _bias_rmse_table(df: pl.DataFrame) -> str:
    """Per (policy, α): bias and RMSE of the point estimate (averaged across ω,
    since ω only affects the audit, not the estimate)."""
    g = (
        df.filter(pl.col("delta") == 0.0)
        .group_by(["policy", "alpha"])
        .agg([
            pl.col("estimate").mean().alias("mean_est"),
            pl.col("truth_cvar").first().alias("truth"),
            (pl.col("estimate") - pl.col("truth_cvar")).pow(2).mean().sqrt().alias("rmse"),
            pl.col("abs_error").max().alias("max_abs_err"),
            pl.len().alias("n_rows"),
        ])
        .sort(["policy", "alpha"])
    )
    lines = [
        "| policy | α | mean(est) | truth | bias | RMSE | max\\|err\\| |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in g.iter_rows(named=True):
        bias = row["mean_est"] - row["truth"]
        lines.append(
            f"| {row['policy']} | {row['alpha']:.2f} "
            f"| {_fmt(row['mean_est'])} | {_fmt(row['truth'])} "
            f"| {_fmt(bias)} | {_fmt(row['rmse'])} | {_fmt_sci(row['max_abs_err'])} |"
        )
    return "\n".join(lines)


def _audit_size_table(df: pl.DataFrame) -> str:
    """Per (policy, α, ω) at δ=0: rejection rate (size)."""
    g = (
        df.filter(pl.col("delta") == 0.0)
        .group_by(["policy", "alpha", "omega_estimator"])
        .agg([
            (pl.col("audit_decision") == "REFUSE-LEVEL").mean().alias("size"),
            pl.col("n_audit").first().alias("n_audit"),
        ])
        .sort(["policy", "alpha", "omega_estimator"])
    )
    lines = [
        "| policy | α | Ω̂ | n_audit | rejection rate (size) |",
        "|---|---|---|---|---|",
    ]
    for row in g.iter_rows(named=True):
        lines.append(
            f"| {row['policy']} | {row['alpha']:.2f} | {row['omega_estimator']} "
            f"| {row['n_audit']} | {_fmt(row['size'], 3)} |"
        )
    return "\n".join(lines)


def _omega_ranking_table(df: pl.DataFrame) -> str:
    """
    Per Ω̂ variant, summarize:
        mean_size:   mean rejection rate at δ=0, averaged across (policy, α).
                     Target: ≈ η = 0.05.
        size_dev:    |mean_size − 0.05|. Smaller is better.
        mean_power:  mean rejection rate at δ>0, averaged across (policy, α, δ).
                     Higher is better — but only meaningful if size is calibrated.

    The "winner" is the variant with size_dev close to zero AND high mean_power.
    A variant with low size_dev but very low mean_power is conservative
    (under-powered); a variant with high mean_power and high size_dev is
    over-rejecting (Type-I inflation, not real power).
    """
    nominal_eta = 0.05

    null = (
        df.filter(pl.col("delta") == 0.0)
        .group_by("omega_estimator")
        .agg([(pl.col("audit_decision") == "REFUSE-LEVEL").mean().alias("mean_size")])
    )
    pwr = (
        df.filter(pl.col("delta") > 0.0)
        .group_by("omega_estimator")
        .agg([(pl.col("audit_decision") == "REFUSE-LEVEL").mean().alias("mean_power")])
    )
    if pwr.height == 0:
        joined = null.with_columns(pl.lit(float("nan")).alias("mean_power"))
    else:
        joined = null.join(pwr, on="omega_estimator", how="left")

    joined = joined.with_columns(
        (pl.col("mean_size") - nominal_eta).abs().alias("size_dev")
    ).sort("size_dev")

    lines = [
        f"_Target size: η = {nominal_eta}. Higher mean_power better, but only if size_dev is small._",
        "",
        "| rank | Ω̂ | mean_size (target 0.05) | size_dev | mean_power (δ>0) |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(joined.iter_rows(named=True), start=1):
        lines.append(
            f"| {i} | {row['omega_estimator']} | "
            f"{_fmt(row['mean_size'], 3)} | {_fmt(row['size_dev'], 3)} | "
            f"{_fmt(row['mean_power'], 3)} |"
        )
    return "\n".join(lines)


def _audit_power_table(df: pl.DataFrame) -> str:
    """Per (policy, α, ω, δ>0): rejection rate (power)."""
    if df.filter(pl.col("delta") > 0.0).height == 0:
        return "_(no δ>0 cells in this run.)_"

    g = (
        df.filter(pl.col("delta") > 0.0)
        .group_by(["policy", "alpha", "omega_estimator", "delta"])
        .agg([
            (pl.col("audit_decision") == "REFUSE-LEVEL").mean().alias("power"),
        ])
        .sort(["policy", "alpha", "omega_estimator", "delta"])
    )
    lines = [
        "| policy | α | Ω̂ | δ | rejection rate (power) |",
        "|---|---|---|---|---|",
    ]
    for row in g.iter_rows(named=True):
        lines.append(
            f"| {row['policy']} | {row['alpha']:.2f} | {row['omega_estimator']} "
            f"| {row['delta']:.2f} | {_fmt(row['power'], 3)} |"
        )
    return "\n".join(lines)


def _g1_check(df: pl.DataFrame) -> tuple[float, str]:
    """
    G1 (α=1 collapse): max |estimate − inline_mean_reference| at α=1.

    The inline reference is a separate IsotonicRegression(increasing=True)
    fit on (s_calib, y_calib) averaged over EVAL — see runner.run_one_cell.
    By the PAV reflection identity this must equal `estimate` at α=1 to ≤ 1e-9.

    NOT compared to DGP truth_mean: that comparison conflates the structural
    α=1 identity (code-level, ≤ 1e-9) with the MC sampling error of finite
    calibration / eval slices (O(1/√n)).

    Returns (max_error, formatted_section).
    """
    a1 = df.filter(pl.col("alpha") == 1.0)
    if a1.height == 0:
        return float("nan"), "_(no α=1 cells in this run.)_"
    err = (a1["estimate"] - a1["mean_ref"]).abs()
    max_err = float(err.max())
    section = [
        f"- α=1 cells: **{a1.height}**",
        f"- max |ĈVaR_1 − inline mean reference|: **{_fmt_sci(max_err)}**",
        f"- spec gate G1: ≤ 1e-9 → "
        + ("**PASS** ✓" if max_err <= 1e-9 else f"**FAIL** ✗ (got {_fmt_sci(max_err)})"),
        "",
        "_The inline reference is a separate `IsotonicRegression(increasing=True).fit"
        "(s_calib, y_calib)` averaged over EVAL — see `runner.run_one_cell`. By the "
        "PAV reflection identity (`calibrator.py`), this must equal the saddle-point "
        "estimate at α=1 to numerical precision._",
    ]
    return max_err, "\n".join(section)


def make_report(run_dir: Path | None = None) -> Path:
    """
    Read `<run_dir>/results_mc.csv` and write `<run_dir>/mc_validation.md`.

    If `run_dir` is None, uses `_io.latest_run_dir()`.

    Returns the path to the written markdown report.
    """
    if run_dir is None:
        run_dir = _io.latest_run_dir()

    # Logging setup is idempotent. If the runner already configured a file
    # handler, we reuse it; otherwise we attach to <run_dir>/log.txt so the
    # report's own output is captured next to the data.
    if not logging.getLogger().handlers:
        _io.setup_logging(run_dir=run_dir)

    results_path = run_dir / "results_mc.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} does not exist; run `python -m cvar_v5.mc.runner` first"
        )

    df = pl.read_csv(results_path)
    g1_max_err, g1_section = _g1_check(df)

    parts = [
        "# cvar_v5 — MC validation report",
        "",
        f"Generated by `python -m cvar_v5.mc.report`. Data from `{results_path.name}`.",
        f"Run dir: `{run_dir}`",
        "",
        "## G1 — α=1 collapse identity",
        "",
        g1_section,
        "",
        "## Bias and RMSE of the Direct CVaR estimate (δ=0, all Ω̂ averaged)",
        "",
        _bias_rmse_table(df),
        "",
        "## Ω̂ ranking (canonical-estimator selection)",
        "",
        _omega_ranking_table(df),
        "",
        "## Audit size (rejection rate at δ=0)",
        "",
        _audit_size_table(df),
        "",
        "## Audit power (rejection rate at δ>0)",
        "",
        _audit_power_table(df),
        "",
        "## Run summary",
        "",
        f"- rows: {df.height}",
        f"- policies: {sorted(df['policy'].unique().to_list())}",
        f"- α values: {sorted(df['alpha'].unique().to_list())}",
        f"- δ values: {sorted(df['delta'].unique().to_list())}",
        f"- Ω̂ estimators: {sorted(df['omega_estimator'].unique().to_list())}",
        "",
    ]
    out_path = run_dir / "mc_validation.md"
    out_path.write_text("\n".join(parts))
    LOG.info("wrote %s", out_path)
    LOG.info("G1 max error: %.3e", g1_max_err)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="cvar_v5 MC report generator")
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="run directory under cvar_v5/mc/runs/ (default: latest)",
    )
    args = parser.parse_args()
    make_report(run_dir=args.run_dir)


if __name__ == "__main__":
    main()
