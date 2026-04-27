"""Regenerate cvar_v3/numbers.tex from results CSVs and config constants.

Single source of truth for every number in cvar_v3/extension_appendix.tex.
Re-running cvar_v3/run_arena.py, cvar_v3/run_monte_carlo.py, cvar_v3/n_sweep_synthetic.py,
or cvar_v3/audit_chi2_calibration_mc.py and then this script auto-updates the
appendix on the next pdflatex.

Inputs (CSV):
  - cvar_v3/results_arena.csv          (run_arena.py)             -> §6 main table, alpha-sweep
  - cvar_v3/audit_chi2_W_values.csv    (audit_chi2_calibration_mc) -> §A.1 statistics
  - cvar_v3/results_mc.csv             (run_monte_carlo.py)        -> §A.3 reject and coverage
  - cvar_v3/results_n_sweep.csv        (n_sweep_synthetic.py)      -> §A.4 scaling

Output:
  - cvar_v3/numbers.tex (a flat list of \\newcommand{\\<name>}{<value>} lines)

§A.2 (moment isolation) numbers are currently hardcoded here because
audit_failure_modes_mc.py prints to stdout instead of writing a CSV. When that
script is updated to write cvar_v3/results_failure_modes.csv, replace the
hardcoded values with a CSV read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

CVAR_DIR = Path(__file__).parent
ARENA_CSV = CVAR_DIR / "results_arena.csv"
WALD_CSV = CVAR_DIR / "audit_chi2_W_values.csv"
MC_CSV = CVAR_DIR / "results_mc.csv"
NSWEEP_CSV = CVAR_DIR / "results_n_sweep.csv"
OUT = CVAR_DIR / "numbers.tex"

POLICY_TAG = {
    "parallel_universe_prompt": "Parallel",
    "premium": "Premium",
    "clone": "Clone",
    "unhelpful": "Unhelpful",
    "base": "Base",
}
ALPHA_TAG = {0.20: "Twenty", 0.10: "Ten", 0.05: "Five", 0.01: "One", 0.005: "Halfpct"}
N_TAG = {500: "FiveHundred", 1000: "OneK", 2000: "TwoK", 2500: "TwoFiveK",
         5000: "FiveK", 25000: "TwentyFiveK", 100000: "HundredK", 500000: "FiveHundredK"}


def _cmd(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _f2(x: float) -> str:
    return f"{x:.2f}"


def _pct(x: float) -> str:
    return f"{int(round(x * 100))}\\%"


def _pval(p: float) -> str:
    if p < 1e-7:
        return "{<}10^{-8}"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.2f}"


def _signed(x: float) -> str:
    return f"{x:+.3f}"


def _config_macros() -> list[str]:
    """Configuration constants. Sourced from run_arena.py, run_monte_carlo.py,
    audit_chi2_calibration_mc.py, audit_failure_modes_mc.py, n_sweep_synthetic.py."""
    return [
        "% --- Arena run config (cvar_v3/run_arena.py) ---",
        _cmd("nFreshDraws", "5{,}000"),
        _cmd("oracleCovPct", "25\\%"),
        _cmd("nFolds", "5"),
        _cmd("nSeeds", "20"),
        _cmd("bootB", "500"),
        _cmd("auditB", "80"),
        _cmd("gridN", "61"),
        "",
        "% --- Wald-null MC config (§A.1; cvar_v3/audit_chi2_calibration_mc.py) ---",
        _cmd("waldNumOuter", "300"),
        _cmd("waldNumInner", "80"),
        _cmd("waldNaSmall", "1{,}000"),
        _cmd("waldNaLarge", "2{,}500"),
        "",
        "% --- Moment-isolation MC config (§A.2; cvar_v3/audit_failure_modes_mc.py) ---",
        _cmd("modeNumReps", "200"),
        _cmd("modeNa", "2{,}000"),
        "",
        "% --- Semi-synthetic MC config (§A.3; cvar_v3/run_monte_carlo.py) ---",
        _cmd("mcNumOuter", "2{,}500"),
        _cmd("mcNumCells", "25"),
        _cmd("mcNumPerCell", "100"),
        _cmd("mcNa", "2{,}000"),
        "",
        "% --- A.4 n-sweep config (cvar_v3/n_sweep_synthetic.py) ---",
        _cmd("nSweepBoot", "200"),
        _cmd("nSweepOracle", "1{,}250"),
    ]


def _arena_macros() -> list[str]:
    df = pl.read_csv(ARENA_CSV)
    out: list[str] = ["% --- §6 Arena main table at α=0.10 (median across seeds) ---"]
    head = df.filter(pl.col("alpha") == 0.10)
    agg = head.group_by("policy").agg([
        pl.col("mean").median().alias("mean"),
        pl.col("mean_ci_lo").median().alias("mean_lo"),
        pl.col("mean_ci_hi").median().alias("mean_hi"),
        pl.col("oracle_truth").median().alias("mean_truth"),
        pl.col("cvar").median().alias("cvar"),
        pl.col("cvar_ci_lo").median().alias("cvar_lo"),
        pl.col("cvar_ci_hi").median().alias("cvar_hi"),
        pl.col("cvar_empirical_truth").median().alias("cvar_truth"),
        pl.col("audit_p_value").median().alias("audit_p"),
        pl.col("audit_reject").mean().alias("reject_rate"),
    ])
    rows = {r["policy"]: r for r in agg.iter_rows(named=True)}
    for pol in ["parallel_universe_prompt", "premium", "clone", "unhelpful"]:
        r = rows[pol]
        tag = POLICY_TAG[pol]
        out.append(_cmd(f"arena{tag}Mean", _f3(r["mean"])))
        out.append(_cmd(f"arena{tag}MeanLo", _f3(r["mean_lo"])))
        out.append(_cmd(f"arena{tag}MeanHi", _f3(r["mean_hi"])))
        out.append(_cmd(f"arena{tag}MeanTruth", _f3(r["mean_truth"])))
        out.append(_cmd(f"arena{tag}Cvar", _f3(r["cvar"])))
        out.append(_cmd(f"arena{tag}CvarLo", _f3(r["cvar_lo"])))
        out.append(_cmd(f"arena{tag}CvarHi", _f3(r["cvar_hi"])))
        out.append(_cmd(f"arena{tag}CvarTruth", _f3(r["cvar_truth"])))
        out.append(_cmd(f"arena{tag}AuditP", f"${_pval(r['audit_p'])}$"))
        out.append(_cmd(f"arena{tag}RejectPct", _pct(r["reject_rate"])))
    out.append("")
    out.append("% --- §6 alpha-sweep table (median across seeds, α ∈ {0.20,0.10,0.05,0.01}) ---")
    sweep = df.group_by(["policy", "alpha"]).agg([
        pl.col("cvar").median().alias("cvar"),
        pl.col("cvar_empirical_truth").median().alias("cvar_truth"),
        pl.col("oracle_truth").median().alias("mean_truth"),
    ])
    sweep_rows = {(r["policy"], r["alpha"]): r for r in sweep.iter_rows(named=True)}
    for pol in ["clone", "premium", "parallel_universe_prompt", "unhelpful"]:
        tag = POLICY_TAG[pol]
        # Use any alpha row for mean truth (it's the same — oracle full-mean per policy)
        out.append(_cmd(f"sweep{tag}MeanTruth", _f3(sweep_rows[(pol, 0.10)]["mean_truth"])))
        for a in [0.20, 0.10, 0.05, 0.01]:
            r = sweep_rows[(pol, a)]
            atag = ALPHA_TAG[a]
            out.append(_cmd(f"sweep{tag}{atag}Est", _f3(r["cvar"])))
            out.append(_cmd(f"sweep{tag}{atag}Truth", _f3(r["cvar_truth"])))
    return out


def _wald_null_macros() -> list[str]:
    df = pl.read_csv(WALD_CSV).filter(pl.col("audit") == "xf")
    out: list[str] = ["% --- §A.1 Wald-null statistics (xf audit; n_a∈{1000,2500}) ---"]
    chi2_2 = stats.chi2(df=2)
    for n_eval, tag in [(1000, "OneK"), (2500, "TwoFiveK")]:
        sub = df.filter(pl.col("n_eval") == n_eval)
        W = np.asarray(sub["W"].to_list())
        ks_stat, ks_p = stats.kstest(W, chi2_2.cdf)
        out.append(_cmd(f"waldMean{tag}", _f2(float(np.mean(W)))))
        out.append(_cmd(f"waldMedian{tag}", f"{float(np.median(W)):.3f}"))
        out.append(_cmd(f"waldVar{tag}", _f2(float(np.var(W, ddof=1)))))
        out.append(_cmd(f"waldNinetyFifth{tag}", _f2(float(np.percentile(W, 95)))))
        out.append(_cmd(f"waldKS{tag}", f"{float(ks_stat):.3f}"))
        out.append(_cmd(f"waldKSp{tag}", f"{float(ks_p):.3f}"))
        rej = float(np.mean(W > chi2_2.ppf(0.95)))
        out.append(_cmd(f"waldReject{tag}", f"{rej:.3f}"))
    return out


def _moment_isolation_macros() -> list[str]:
    """§A.2 numbers — read from cvar_v3/results_failure_modes.csv if present;
    otherwise fall back to hardcoded last-known values."""
    csv = CVAR_DIR / "results_failure_modes.csv"
    cell_to_macro = {
        "baseline":   "modeBaseline",
        "modeA_t02":  "modeAOne",
        "modeA_t05":  "modeATwo",
        "modeB_g005": "modeBOne",
        "modeB_g010": "modeBTwo",
    }
    if csv.exists():
        df = pl.read_csv(csv)
        out = ["% --- §A.2 moment-isolation table (from results_failure_modes.csv) ---"]
        rows = {r["cell_id"]: r for r in df.iter_rows(named=True)}
        for cid, prefix in cell_to_macro.items():
            r = rows[cid]
            out.append(_cmd(f"{prefix}GOne", _signed(r["mean_g1"])))
            out.append(_cmd(f"{prefix}GTwo", _signed(r["mean_g2"])))
            out.append(_cmd(f"{prefix}Reject", _f2(r["reject_rate"])))
        return out
    # Fallback (last manually-verified values).
    return [
        "% --- §A.2 moment-isolation table (HARDCODED; CSV not found) ---",
        _cmd("modeBaselineGOne", "+0.002"),
        _cmd("modeBaselineGTwo", "+0.000"),
        _cmd("modeBaselineReject", "0.08"),
        _cmd("modeAOneGOne", "+0.013"),
        _cmd("modeAOneGTwo", "+0.000"),
        _cmd("modeAOneReject", "0.22"),
        _cmd("modeATwoGOne", "+0.029"),
        _cmd("modeATwoGTwo", "+0.000"),
        _cmd("modeATwoReject", "0.68"),
        _cmd("modeBOneGOne", "+0.002"),
        _cmd("modeBOneGTwo", "-0.005"),
        _cmd("modeBOneReject", "0.87"),
        _cmd("modeBTwoGOne", "+0.002"),
        _cmd("modeBTwoGTwo", "-0.010"),
        _cmd("modeBTwoReject", "1.00"),
    ]


def _mc_a3_macros() -> list[str]:
    df = pl.read_csv(MC_CSV)
    out: list[str] = ["% --- §A.3 reject rate at n_eval=2000, by (eval_policy, delta) ---"]
    pc = df.filter((pl.col("cell_kind") == "power_curve") & (pl.col("n_eval") == 2000))
    rej = pc.group_by(["eval_policy", "delta"]).agg(
        pl.col("audit_reject_xf").mean().alias("rej")
    )
    rej_rows = {(r["eval_policy"], r["delta"]): r["rej"] for r in rej.iter_rows(named=True)}
    delta_tag = {0.0: "DeltaZero", 0.02: "DeltaTwo", 0.05: "DeltaFive",
                 0.10: "DeltaTen", 0.20: "DeltaTwenty"}
    for pol in ["clone", "premium", "parallel_universe_prompt", "unhelpful"]:
        tag = POLICY_TAG[pol]
        for d in [0.0, 0.02, 0.05, 0.10, 0.20]:
            out.append(_cmd(f"mcReject{tag}{delta_tag[d]}", _f2(rej_rows[(pol, d)])))
    out.append("")
    out.append("% --- §A.3 CI coverage at delta=0 (cleanest null) ---")
    cov_src = df.filter(
        pl.col("cell_kind").is_in(["power_curve", "size_diagnostic"])
        & (pl.col("n_eval") == 2000) & (pl.col("delta") == 0.0)
    )
    cov_agg = cov_src.group_by("eval_policy").agg([
        pl.col("ci_covers_truth").mean().alias("cov_all"),
        (pl.col("ci_covers_truth").filter(~pl.col("audit_reject_xf"))).mean().alias("cov_accept"),
        (~pl.col("audit_reject_xf")).sum().alias("n_accept"),
        pl.len().alias("n_total"),
    ])
    cov_rows = {r["eval_policy"]: r for r in cov_agg.iter_rows(named=True)}
    for pol in ["base", "clone", "premium", "parallel_universe_prompt", "unhelpful"]:
        if pol not in cov_rows:
            continue
        r = cov_rows[pol]
        tag = POLICY_TAG[pol]
        out.append(_cmd(f"mcCovAll{tag}", _f2(r["cov_all"])))
        out.append(_cmd(f"mcCovAccept{tag}", _f2(r["cov_accept"])))
        out.append(_cmd(f"mcNAccept{tag}", str(int(r["n_accept"]))))
        out.append(_cmd(f"mcNTotal{tag}", str(int(r["n_total"]))))
    return out


def _nsweep_macros() -> list[str]:
    df = pl.read_csv(NSWEEP_CSV)
    out: list[str] = ["% --- §A.4 n-sweep at fixed n_oracle=1250 (median across seeds) ---"]
    agg = df.group_by(["policy", "alpha", "n_eval"]).agg([
        pl.col("cvar_est").median().alias("est"),
        pl.col("ci_lo").median().alias("lo"),
        pl.col("ci_hi").median().alias("hi"),
    ])
    rows = {(r["policy"], r["alpha"], r["n_eval"]): r for r in agg.iter_rows(named=True)}
    # Per-policy estimates and CI at each n
    for pol in ["clone", "premium"]:
        tag = POLICY_TAG[pol]
        for a in [0.05, 0.01, 0.005]:
            atag = ALPHA_TAG[a]
            for n in [5000, 25000, 100000, 500000]:
                ntag = N_TAG[n]
                r = rows[(pol, a, n)]
                out.append(_cmd(f"nsweep{tag}{atag}{ntag}Est", _f3(r["est"])))
            r = rows[(pol, a, 500000)]
            out.append(_cmd(f"nsweep{tag}{atag}FiveHundredKLo", _f3(r["lo"])))
            out.append(_cmd(f"nsweep{tag}{atag}FiveHundredKHi", _f3(r["hi"])))
    # Truth gap (premium - clone) per alpha
    out.append("")
    out.append("% --- §A.4 truth gap (premium − clone CVaR truth) per α ---")
    # Truth gap is reported in the appendix narrative; we need population CVaR_α from
    # the DGP for that. We infer it from cvar_empirical_truth in results_arena.csv at the
    # closest available α — but A.4 uses α∈{0.05,0.01,0.005} and arena α∈{0.01,0.05,...}.
    # The paper's reported gaps are computed in n_sweep_synthetic.py against the
    # population CVaR via cvar_truth(); n_sweep_synthetic prints them at runtime and
    # they are stable across seeds. Hardcode them with a note.
    out.append("% Source: n_sweep_synthetic.py prints these at runtime as ")
    out.append("% 'gap@α=...' (see cvar_v3/results_n_sweep.log).")
    out.append(_cmd("nsweepTruthGapFive", "+0.012"))
    out.append(_cmd("nsweepTruthGapOne", "+0.018"))
    out.append(_cmd("nsweepTruthGapHalfpct", "+0.037"))
    return out


def main() -> int:
    sections = [
        "% AUTO-GENERATED by cvar_v3/regenerate_macros.py — do not edit by hand.",
        "% Re-run that script to refresh from results CSVs.",
        "",
        *_config_macros(),
        "",
        *_arena_macros(),
        "",
        *_wald_null_macros(),
        "",
        *_moment_isolation_macros(),
        "",
        *_mc_a3_macros(),
        "",
        *_nsweep_macros(),
        "",
    ]
    OUT.write_text("\n".join(sections))
    n_macros = sum(1 for s in sections if s.startswith("\\newcommand"))
    print(f"Wrote {n_macros} macros to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
