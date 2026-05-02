"""Minimal CVaR-CJE pilot table — uniform oracle slice, atom-split truth,
single transport audit (mean_g1, mean_g2).

No CIs, no variance decomposition, no design comparison, no audit-variant grid.

Usage:
    python -m cvar_v4.healthbench_data.cvar_pilot_table --alpha 0.10 --coverage 0.25 --seed 42

Outputs:
    pilot_table_alpha_{α}_cov_{cov}_seed_{seed}.{jsonl, md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import step5_oracle_calibrated_uniform
from .analyses.variance_breakdown import _assemble_cell, _master_indices
from ..eda.deeper._estimator import pipeline_bootstrap_cvar, pipeline_bootstrap_mean


OUT_DIR = Path(__file__).parent / "writeup" / "data"

PIPELINE_BOOTSTRAP_B = 500
PIPELINE_BOOTSTRAP_SEED_OFFSET = 7  # avoid colliding with the slice/audit RNG


def _augment_with_pipeline_ci(
    rows: list[dict], *, coverage: float, alpha: float, seed: int,
    B: int = PIPELINE_BOOTSTRAP_B,
) -> None:
    """Add full-pipeline-bootstrap percentile CIs to each row in place.

    Resamples (train, eval, audit) for CVaR and (train, eval) for mean,
    refits the isotonic calibrator and re-optimizes t̂ inside every
    replicate. Coupled per-rep indices ensure the CVaR and mean bootstraps
    share train/eval draws — same RNG path the figure caption now claims.

    Adds the following keys per row:
        cvar_pipeline_ci_lo, cvar_pipeline_ci_hi, cvar_pipeline_var
        mean_pipeline_ci_lo, mean_pipeline_ci_hi, mean_pipeline_var
    """
    log, per_policy = _assemble_cell(coverage, alpha, seed)
    boot_seed = seed + PIPELINE_BOOTSTRAP_SEED_OFFSET
    for r in rows:
        pol = per_policy.get(r["policy"])
        if pol is None or pol["n_audit"] < 3:
            continue
        master = _master_indices(
            n_train=int(len(log["s_train"])), n_eval=pol["n_eval"],
            n_audit=pol["n_audit"], B=B, seed=boot_seed,
        )
        cvar_full = pipeline_bootstrap_cvar(
            s_train=log["s_train"], y_train=log["y_train"],
            s_eval_full=pol["s_eval_full"],
            s_audit=pol["s_audit"], y_audit=pol["y_audit"],
            alpha=alpha,
            sample_weight_train=log["w_train"],
            sample_weight_audit=pol["w_audit"],
            resample=("train", "eval", "audit"),
            B=B, seed=boot_seed,
            idx_train_per_b=master["boot"]["train"],
            idx_eval_per_b=master["boot"]["eval"],
            idx_audit_per_b=master["boot"]["audit"],
        )
        mean_full = pipeline_bootstrap_mean(
            s_train=log["s_train"], y_train=log["y_train"],
            s_eval_full=pol["s_eval_full"],
            sample_weight_train=log["w_train"],
            resample=("train", "eval"),
            B=B, seed=boot_seed,
            idx_train_per_b=master["boot"]["train"],
            idx_eval_per_b=master["boot"]["eval"],
        )
        r["cvar_pipeline_ci_lo"] = float(cvar_full["ci_plug"][0])
        r["cvar_pipeline_ci_hi"] = float(cvar_full["ci_plug"][1])
        r["cvar_pipeline_var"] = float(cvar_full["var_plug"])
        r["mean_pipeline_ci_lo"] = float(mean_full["ci_lo"])
        r["mean_pipeline_ci_hi"] = float(mean_full["ci_hi"])
        r["mean_pipeline_var"] = float(mean_full["var_eval"])
        r["pipeline_bootstrap_B"] = int(B)
        r["pipeline_bootstrap_seed"] = int(boot_seed)


def _output_paths(alpha: float, coverage: float, seed: int) -> tuple[Path, Path]:
    suffix = f"alpha_{alpha:.2f}_cov_{coverage:.2f}_seed_{seed}"
    # Strip trailing zeros in the numeric parts for readability
    suffix = suffix.replace("0.00", "0").replace("0.10", "0.1").replace("0.20", "0.2")
    suffix = suffix.replace("0.30", "0.3").replace("0.50", "0.5")
    return (
        OUT_DIR / f"pilot_table_{suffix}.jsonl",
        OUT_DIR / f"pilot_table_{suffix}.md",
    )


def run(*, coverage: float, alpha: float, seed: int, verbose: bool = True) -> list[dict]:
    """Programmatic entry point: produce pilot table for one (α, coverage, seed)
    and write JSONL + MD to writeup/data/. Returns the rows list.
    """
    rows = step5_oracle_calibrated_uniform(
        coverage=coverage, alpha=alpha, seed=seed, verbose=verbose,
    )
    if rows:
        _augment_with_pipeline_ci(
            rows, coverage=coverage, alpha=alpha, seed=seed,
        )
    jsonl_out, md_out = _output_paths(alpha, coverage, seed)
    write_jsonl(rows, jsonl_out)
    write_md(rows, md_out, alpha=alpha, coverage=coverage, seed=seed)
    if verbose:
        print(f"\n[cvar-pilot] wrote {len(rows)} rows to {jsonl_out}")
        print(f"[cvar-pilot] summary at {md_out}")
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_md(rows: list[dict], path: Path,
             alpha: float, coverage: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# Pilot table — α={alpha} cov={coverage} seed={seed}\n\nNo results.\n")
        return
    n_total = rows[0]["n_total"]
    lines = [
        f"# CVaR-CJE pilot table",
        f"",
        f"- **α** = `{alpha}`",
        f"- **coverage** = `{coverage}` (uniform oracle slice)",
        f"- **seed** = `{seed}`",
        f"- **n_total per policy** = `{n_total}`",
        f"- **truth** = atom-split CVaR_α on full oracle panel",
        f"- **t̂** = optimized on FULL target cheap-score distribution (not on audit slice)",
        f"- **verdict heuristic** = PASS if both \\|mean_g1\\|, \\|mean_g2\\| ≤ 0.05; otherwise tag offending moment",
        f"",
        f"| policy | n_slice | CVaR_hat | truth | error | audit_only (95% CI) | t̂ | t* | gap | n_oracle_eq | cost× | mean_g1 | mean_g2 | verdict |",
        f"|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        ao_lo = r.get("cvar_audit_only_ci_lo", float("nan"))
        ao_hi = r.get("cvar_audit_only_ci_hi", float("nan"))
        ao_pt = r.get("cvar_audit_only", float("nan"))
        t_hat = r.get("t_hat", float("nan"))
        t_star = r.get("t_star", float("nan"))
        gap = t_hat - t_star
        n_eq = r.get("n_oracle_equiv", -1)
        cx = r.get("cost_ratio_x", float("nan"))
        lines.append(
            f"| {r['policy']} | {r['n_slice']} | "
            f"{r['cvar_est']:+.3f} | {r['full_oracle_truth']:+.3f} | "
            f"{r['abs_error']:.3f} | "
            f"{ao_pt:+.3f} [{ao_lo:+.3f}, {ao_hi:+.3f}] | "
            f"{t_hat:+.3f} | {t_star:+.3f} | {gap:+.3f} | "
            f"{n_eq} | {cx:.2f}× | "
            f"{r['mean_g1']:+.3f} | {r['mean_g2']:+.3f} | "
            f"{r['verdict']} |"
        )
    lines.append("")
    lines.append(f"**Notes**:")
    lines.append(f"- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; "
                 f"the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).")
    lines.append(f"- `audit_only` = empirical CVaR_α on Y_audit alone (no calibrator). 95% CI is a 1-D percentile bootstrap of n_audit oracle rows (B=2000). The skeptic baseline.")
    lines.append(f"- `t̂` = saddle-point threshold from the FULL cheap target distribution; `t*` = atom-split α-quantile of full oracle Y. `gap = t̂ − t*` is a mechanism diagnostic.")
    lines.append(f"- `n_oracle_eq` = pure-oracle rows needed to match Direct's variance. `cost× = Var(audit_only) / Var(Direct) = n_oracle_eq / n_audit`. `>1` means Direct is more efficient than just averaging the audit slice; `<1` means Direct's calibrator-resampling variance exceeds the gain. Honest envelope uses Var_cal + Var_audit for Direct.")
    lines.append(f"- mean_g1 = mean of `1{{y_audit ≤ t̂}} − α` (tail-mass deviation).")
    lines.append(f"- mean_g2 = mean of `(t̂ − y_audit)_+ − ĝ_t̂(s_audit)` (stop-loss residual transport).")
    lines.append(f"- Verdict is a heuristic flag, NOT a hypothesis test. For a formal audit, multi-seed + bootstrap-Σ̂ are needed.")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(coverage=args.coverage, alpha=args.alpha, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
