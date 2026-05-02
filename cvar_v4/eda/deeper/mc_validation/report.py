"""Aggregate results_mc.csv → mc_validation.md + power-curve PNG.

Sections of the report:
  1. Mean coverage & RMSE table per (eval_policy × n × oracle_coverage × α × design)
  2. CVaR coverage & RMSE table, same axes
  3. Audit size at the null (mean + CVaR audit) per policy
  4. Audit power vs δ × perturbation
  5. Theory-vs-observed gate

Coverage uses the calibration-aware total CI (Var_eval + Var_cal). The
"_eval"-only percentile-bootstrap CI is also reported so the appendix can
show the gap that motivates Var_cal.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results_mc.csv"
DEFAULT_MD = HERE / "mc_validation.md"
DEFAULT_FIG = HERE / "figs" / "mc_validation_power_curve.png"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def fmt_rate(p: float, k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_ci(k, n)
    return f"{p:.3f} [{lo:.2f},{hi:.2f}]"


def section_coverage(df: pl.DataFrame, estimand: str, alpha_filter: float | None = None) -> str:
    """Coverage / RMSE / half-width table for `estimand` ∈ {'mean','cvar'}."""
    cov_col = f"{estimand}_covers"
    cov_eval_col = f"{estimand}_covers_eval"
    err_col = f"abs_err_{estimand}"
    var_total_col = f"{estimand}_var_total"
    var_cal_col = f"{estimand}_var_cal"
    var_eval_col = f"{estimand}_var_eval"
    truth_col = f"{estimand}_truth"
    est_col = f"{estimand}_est"

    sub = df.filter(
        (pl.col("cell_kind") == "coverage")
        & (pl.col("skip_reason") == "")
    )
    if alpha_filter is not None:
        sub = sub.filter(pl.col("alpha") == alpha_filter)
    if len(sub) == 0:
        return f"_(no coverage cells for {estimand})_\n\n"

    grp = (
        sub.group_by(["eval_policy", "n_total", "oracle_coverage", "alpha", "design"])
        .agg([
            pl.col(cov_col).mean().alias("cov_total"),
            pl.col(cov_col).sum().alias("cov_total_n"),
            pl.col(cov_eval_col).mean().alias("cov_eval"),
            pl.col(err_col).pow(2).mean().sqrt().alias("rmse"),
            (pl.col(est_col) - pl.col(truth_col)).mean().alias("bias"),
            (pl.col(var_cal_col) / pl.col(var_total_col)).mean().alias("cal_share"),
            ((pl.col(var_total_col).clip(lower_bound=0)).sqrt() * 1.96).mean().alias("hw_total"),
            ((pl.col(var_eval_col).clip(lower_bound=0)).sqrt() * 1.96).mean().alias("hw_eval"),
            pl.len().alias("n_reps"),
        ])
        .sort(["eval_policy", "n_total", "oracle_coverage", "alpha", "design"])
    )

    out = (
        "| policy | n | cov | α | design | n_reps | cov_total | cov_eval | rmse | bias | hw_total | hw_eval | Vc/Vt |\n"
        "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rng = grp.iter_rows(named=True)
    for r in rng:
        n_reps = int(r["n_reps"])
        cov_t_str = fmt_rate(float(r["cov_total"]), int(r["cov_total_n"]), n_reps)
        out += (
            f"| {r['eval_policy']:24} | {r['n_total']:>4} | {r['oracle_coverage']:.2f} "
            f"| {r['alpha']:.2f} | {r['design']} | {n_reps} "
            f"| {cov_t_str} | {r['cov_eval']:.3f} "
            f"| {r['rmse']:.3f} | {r['bias']:+.3f} "
            f"| {r['hw_total']:.3f} | {r['hw_eval']:.3f} "
            f"| {r['cal_share']:.2f} |\n"
        )
    return out + "\n"


def section_audit_size(df: pl.DataFrame) -> str:
    sub = df.filter(
        (pl.col("cell_kind") == "coverage")
        & (pl.col("skip_reason") == "")
    )
    if len(sub) == 0:
        return "_(no rows)_\n\n"
    grp = (
        sub.group_by(["eval_policy", "alpha", "n_total", "oracle_coverage", "design"])
        .agg([
            pl.col("mean_audit_reject").mean().alias("mean_size"),
            pl.col("mean_audit_reject").sum().alias("mean_size_n"),
            pl.col("cvar_audit_reject").mean().alias("cvar_size"),
            pl.col("cvar_audit_reject").sum().alias("cvar_size_n"),
            pl.len().alias("n_reps"),
        ])
        .sort(["eval_policy", "alpha", "n_total", "oracle_coverage", "design"])
    )
    out = (
        "| policy | α | n | cov | design | n_reps | mean reject | cvar reject |\n"
        "|---|--:|--:|--:|---|--:|--:|--:|\n"
    )
    for r in grp.iter_rows(named=True):
        n_reps = int(r["n_reps"])
        out += (
            f"| {r['eval_policy']:24} | {r['alpha']:.2f} | {r['n_total']:>4} | "
            f"{r['oracle_coverage']:.2f} | {r['design']} | {n_reps} | "
            f"{fmt_rate(float(r['mean_size']), int(r['mean_size_n']), n_reps)} | "
            f"{fmt_rate(float(r['cvar_size']), int(r['cvar_size_n']), n_reps)} |\n"
        )
    return out + "\n"


def section_audit_power(df: pl.DataFrame) -> str:
    sub = df.filter(
        (pl.col("cell_kind") == "power")
        & (pl.col("skip_reason") == "")
    )
    if len(sub) == 0:
        return "_(no power cells)_\n\n"
    grp = (
        sub.group_by(["eval_policy", "alpha", "perturbation", "delta"])
        .agg([
            pl.col("mean_audit_reject").mean().alias("mean_pow"),
            pl.col("mean_audit_reject").sum().alias("mean_pow_n"),
            pl.col("cvar_audit_reject").mean().alias("cvar_pow"),
            pl.col("cvar_audit_reject").sum().alias("cvar_pow_n"),
            pl.col("cvar_covers").mean().alias("cvar_cov"),
            pl.col("mean_covers").mean().alias("mean_cov"),
            pl.len().alias("n_reps"),
        ])
        .sort(["eval_policy", "alpha", "perturbation", "delta"])
    )
    out = (
        "| target | α | perturb | δ | n_reps | mean reject | cvar reject | mean cov | cvar cov |\n"
        "|---|--:|---|--:|--:|--:|--:|--:|--:|\n"
    )
    for r in grp.iter_rows(named=True):
        n_reps = int(r["n_reps"])
        out += (
            f"| {r['eval_policy']} | {r['alpha']:.2f} | {r['perturbation']} | {r['delta']:.3f} "
            f"| {n_reps} | "
            f"{fmt_rate(float(r['mean_pow']), int(r['mean_pow_n']), n_reps)} | "
            f"{fmt_rate(float(r['cvar_pow']), int(r['cvar_pow_n']), n_reps)} | "
            f"{r['mean_cov']:.3f} | {r['cvar_cov']:.3f} |\n"
        )
    return out + "\n"


def section_gates(df: pl.DataFrame) -> str:
    """Theory-vs-observed gate. Pooled across coverage cells (δ=0)."""
    null_df = df.filter(
        (pl.col("cell_kind") == "coverage") & (pl.col("skip_reason") == "")
    )
    self_validation = null_df.filter(pl.col("eval_policy") == "base")

    def pooled(df_, col):
        if len(df_) == 0:
            return float("nan"), 0, 0
        k = int(df_[col].sum())
        n = len(df_)
        return k / n, k, n

    rows = []
    for label, d in [("self-validation (base→base)", self_validation),
                     ("all coverage cells (pooled)", null_df)]:
        mc_mean_cov, k_mc, n_mc = pooled(d, "mean_covers")
        mc_cvar_cov, k_cc, n_cc = pooled(d, "cvar_covers")
        mc_mean_size, k_ms, n_ms = pooled(d, "mean_audit_reject")
        mc_cvar_size, k_cs, n_cs = pooled(d, "cvar_audit_reject")
        rows.append((label, mc_mean_cov, k_mc, n_mc,
                     mc_cvar_cov, k_cc, n_cc,
                     mc_mean_size, k_ms, n_ms,
                     mc_cvar_size, k_cs, n_cs))

    out = (
        "| pool | mean cov (≥0.93) | cvar cov (≥0.93) | mean size (≤0.10) | cvar size (≤0.10) |\n"
        "|---|--:|--:|--:|--:|\n"
    )
    for r in rows:
        (label, mcov, kmc, nmc, ccov, kcc, ncc,
         msize, kms, nms, csize, kcs, ncs) = r
        def gate(v, target, kind):
            if not (v == v):
                return "n/a"
            if kind == "geq":
                return f"{v:.3f} {'PASS' if v >= target else 'FAIL'}"
            return f"{v:.3f} {'PASS' if v <= target else 'FAIL'}"
        out += (
            f"| {label} | {gate(mcov, 0.93, 'geq')} | {gate(ccov, 0.93, 'geq')} "
            f"| {gate(msize, 0.10, 'leq')} | {gate(csize, 0.10, 'leq')} |\n"
        )

    # Power cells: monotonicity check
    power_df = df.filter((pl.col("cell_kind") == "power") & (pl.col("skip_reason") == ""))
    if len(power_df) > 0:
        out += "\n**Power monotonicity (by perturbation):**\n\n"
        for pert in ["tail", "uniform"]:
            d = power_df.filter(pl.col("perturbation") == pert)
            if len(d) == 0:
                continue
            agg = (d.group_by("delta")
                    .agg(pl.col("cvar_audit_reject").mean().alias("rate"))
                    .sort("delta"))
            ds = agg["delta"].to_list()
            rs = agg["rate"].to_list()
            mono = all(rs[i] <= rs[i + 1] for i in range(len(rs) - 1))
            out += f"- perturbation={pert}: " + ", ".join(
                f"δ={d:.3f}→{r:.3f}" for d, r in zip(ds, rs)
            ) + f"  ({'monotone' if mono else 'NOT monotone'})\n"
    return out + "\n"


def make_power_figure(df: pl.DataFrame, out_path: Path) -> None:
    """Render audit power curves (mean & cvar audit, both perturbations)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib unavailable; skipping {out_path}")
        return
    sub = df.filter((pl.col("cell_kind") == "power") & (pl.col("skip_reason") == ""))
    if len(sub) == 0:
        return
    grp = (sub.group_by(["perturbation", "delta"])
           .agg([
               pl.col("mean_audit_reject").mean().alias("mean_rej"),
               pl.col("cvar_audit_reject").mean().alias("cvar_rej"),
               pl.len().alias("n"),
           ])
           .sort(["perturbation", "delta"]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for pert in ["tail", "uniform"]:
        d = grp.filter(pl.col("perturbation") == pert)
        if len(d) == 0:
            continue
        deltas = d["delta"].to_numpy()
        ax[0].plot(deltas, d["mean_rej"].to_numpy(), marker="o", label=pert)
        ax[1].plot(deltas, d["cvar_rej"].to_numpy(), marker="o", label=pert)
    for a, title in [(ax[0], "Mean transport audit"), (ax[1], "CVaR transport audit")]:
        a.axhline(0.05, color="gray", linestyle=":", linewidth=0.8)
        a.set_xlabel(r"$\delta$ (perturbation magnitude)")
        a.set_title(title)
        a.set_ylim(-0.02, 1.02)
        a.legend(title="perturbation", fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Reject rate")
    fig.suptitle("Audit power vs. δ — calib=base, eval=clone, n=500, oracle=0.25, α=0.10")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {out_path}")


def latex_coverage_headline(df: pl.DataFrame) -> str:
    """Compact LaTeX table: per-policy coverage + RMSE at the headline cell.

    Headline cell: n_total=500, oracle_coverage=0.25, α=0.10, design=uniform,
    cell_kind=coverage. One row per eval_policy.
    """
    sub = df.filter(
        (pl.col("cell_kind") == "coverage")
        & (pl.col("skip_reason") == "")
        & (pl.col("n_total") == 500)
        & (pl.col("oracle_coverage") == 0.25)
        & (pl.col("alpha") == 0.10)
        & (pl.col("design") == "uniform")
    )
    if len(sub) == 0:
        # fall back to whatever cell has the most reps
        sub = df.filter(
            (pl.col("cell_kind") == "coverage")
            & (pl.col("skip_reason") == "")
            & (pl.col("design") == "uniform")
        )
    grp = (sub.group_by("eval_policy")
           .agg([
               pl.col("mean_covers").mean().alias("mean_cov"),
               pl.col("cvar_covers").mean().alias("cvar_cov"),
               pl.col("mean_audit_reject").mean().alias("mean_size"),
               pl.col("cvar_audit_reject").mean().alias("cvar_size"),
               pl.col("abs_err_mean").pow(2).mean().sqrt().alias("rmse_mean"),
               pl.col("abs_err_cvar").pow(2).mean().sqrt().alias("rmse_cvar"),
               pl.len().alias("n_reps"),
           ])
           .sort("eval_policy"))
    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{\textbf{Monte Carlo headline results.} Per-policy 95\%-CI"
        r" coverage of the calibration-aware total CI, RMSE, and audit"
        r" reject rate at the null. Cell: $n{=}500$, oracle coverage 0.25,"
        r" $\alpha{=}0.10$, uniform design, $\delta{=}0$. Calibrator $\hat m_{\text{base}}$"
        r" with $m_{\text{override}}{=}\hat m_{\text{base}}$, so the only mis-specification"
        r" source is per-policy heterogeneity in $Y$ marginals and noise"
        r" $\sigma(Y)$.}",
        r"\label{tab:mc-validation-headline}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"target $\pi'$ & $n_{\text{rep}}$ & "
        r"Mean cov.\ & CVaR cov.\ & "
        r"RMSE Mean & RMSE CVaR & Mean rej.\ \\",
        r"\midrule",
    ]
    for r in grp.iter_rows(named=True):
        name = r["eval_policy"].replace("_", r"\_")
        lines.append(
            f"\\texttt{{{name}}} & {int(r['n_reps'])} & "
            f"{r['mean_cov']:.2f} & {r['cvar_cov']:.2f} & "
            f"{r['rmse_mean']:.3f} & {r['rmse_cvar']:.3f} & "
            f"{r['mean_size']:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def latex_audit_power(df: pl.DataFrame) -> str:
    """Compact LaTeX table for audit power (calib=base, eval=clone)."""
    sub = df.filter((pl.col("cell_kind") == "power") & (pl.col("skip_reason") == ""))
    if len(sub) == 0:
        return ""
    grp = (sub.group_by(["perturbation", "delta"])
           .agg([
               pl.col("mean_audit_reject").mean().alias("mean_rej"),
               pl.col("cvar_audit_reject").mean().alias("cvar_rej"),
               pl.len().alias("n"),
           ])
           .sort(["perturbation", "delta"]))
    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{\textbf{Audit power vs.\ perturbation magnitude $\delta$.}"
        r" Calib $=$ base, eval $=$ clone, $n{=}500$, oracle 0.25, $\alpha{=}0.10$,"
        r" uniform design. Mean and CVaR transport audits run on the same data per replicate.}",
        r"\label{tab:mc-validation-power}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"perturbation & $\delta$ & Mean reject & CVaR reject \\",
        r"\midrule",
    ]
    for r in grp.iter_rows(named=True):
        lines.append(
            f"{r['perturbation']} & {r['delta']:.3f} & "
            f"{r['mean_rej']:.2f} & {r['cvar_rej']:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--tex-coverage", type=Path,
                        default=Path(__file__).resolve().parents[4]
                        / "arxiv_source" / "tables" / "mc_validation_coverage.tex")
    parser.add_argument("--tex-power", type=Path,
                        default=Path(__file__).resolve().parents[4]
                        / "arxiv_source" / "tables" / "mc_validation_audit_power.tex")
    parser.add_argument("--tex-fig", type=Path,
                        default=Path(__file__).resolve().parents[4]
                        / "arxiv_source" / "figs" / "mc_validation_power_curve.png")
    args = parser.parse_args()

    df = pl.read_csv(args.csv)
    n_total = len(df)
    n_skip = int((df["skip_reason"] != "").sum())
    print(f"Loaded {n_total} rows ({n_skip} skipped)")

    parts: list[str] = []
    parts.append("# Monte Carlo Validation: results\n\n")
    parts.append(
        f"_{n_total} rows ({n_skip} skipped); semi-synthetic HealthBench DGP; "
        "estimator/audit/CI machinery from `cvar_v4/eda/deeper/_estimator.py`._\n\n"
    )

    parts.append("## 1. Mean-CJE coverage & RMSE (δ=0 cells)\n\n")
    parts.append(section_coverage(df, "mean"))

    parts.append("## 2. CVaR-CJE coverage & RMSE (δ=0 cells)\n\n")
    parts.append(section_coverage(df, "cvar"))

    parts.append("## 3. Audit size at the null\n\n")
    parts.append(section_audit_size(df))

    parts.append("## 4. Audit power vs δ\n\n")
    parts.append(section_audit_power(df))

    parts.append("## 5. Theory-vs-observed gate\n\n")
    parts.append(section_gates(df))

    args.out.write_text("".join(parts))
    print(f"Wrote report: {args.out}")
    make_power_figure(df, args.fig)

    # LaTeX outputs for the appendix
    args.tex_coverage.parent.mkdir(parents=True, exist_ok=True)
    args.tex_coverage.write_text(latex_coverage_headline(df))
    print(f"Wrote LaTeX: {args.tex_coverage}")
    tex_power = latex_audit_power(df)
    if tex_power:
        args.tex_power.parent.mkdir(parents=True, exist_ok=True)
        args.tex_power.write_text(tex_power)
        print(f"Wrote LaTeX: {args.tex_power}")
    # Also save the figure to arxiv_source/figs/
    if args.tex_fig != args.fig:
        make_power_figure(df, args.tex_fig)


if __name__ == "__main__":
    main()
