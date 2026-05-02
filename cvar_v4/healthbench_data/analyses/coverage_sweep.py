"""Coverage sweep: how often does each CI variant cover the full-oracle truth?

The CJE paper's headline justification for V_aug + bootstrap is *coverage*,
not point-estimate accuracy. Naive plug-in achieves 0% coverage of nominal
95% CIs; V_aug + bootstrap achieves ~95% (arxiv 2512.11150 experiments §Q1).
We've surfaced V_aug and Var_total (jackknife) on our HealthBench panel but
haven't measured coverage. This script does that.

For each (policy, alpha, ci_variant), runs N seeds with different oracle
slice draws and computes the empirical fraction of seeds whose CI covers
`full_oracle_truth`. Wilson 95% intervals on the coverage rates handle the
small-N seed count.

CI variants compared:
  - cvar_eval_only   : (cvar_ci_lo, cvar_ci_hi) — eval-only percentile bootstrap (B=200)
  - cvar_var_total   : cvar_est ± 1.96·sqrt(var_cal + var_audit) — calibration-aware envelope
  - cvar_aug_var_tot : cvar_est_aug ± 1.96·sqrt(var_cal + var_audit) — V_aug center, same envelope
  - mean_eval_only   : mean-side analog
  - mean_var_total   : mean-side analog with Var_cal + Var_audit
  - mean_aug_var_tot : mean V_aug center

No API spend; pure compute on the existing graded panel.

Usage:
    python -m cvar_v4.healthbench_data.analyses.coverage_sweep
    python -m cvar_v4.healthbench_data.analyses.coverage_sweep --n-seeds 30 --alphas 0.10 0.20
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..analyze import step5_oracle_calibrated_uniform


WRITEUP_DIR = Path(__file__).resolve().parent.parent / "writeup" / "data"
OUT_JSON = WRITEUP_DIR / "coverage_sweep.json"
OUT_MD = WRITEUP_DIR / "coverage_sweep.md"


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion k/n. Stable at small n and at 0 / n
    boundary (binomial normal-approximation breaks there)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _row_ci_variants(row: dict) -> dict[str, tuple[float, float, float]]:
    """For a single pilot-table row, return {variant_name: (lo, hi, center)}.

    `center` is included so we can compute coverage for V_aug-centered CIs
    by shifting the envelope to the aug point.
    """
    variants: dict[str, tuple[float, float, float]] = {}

    # CVaR variants
    cvar_est = row.get("cvar_est")
    cvar_est_aug = row.get("cvar_est_aug")
    cvar_est_cov = row.get("cvar_est_cov")
    cvar_est_aug_cov = row.get("cvar_est_aug_cov")
    cvar_var_cal = row.get("cvar_var_cal")
    cvar_var_audit = row.get("cvar_var_audit")
    cvar_se_total = row.get("cvar_se_total")
    cvar_ci_lo = row.get("cvar_ci_lo")
    cvar_ci_hi = row.get("cvar_ci_hi")

    if cvar_est is not None:
        if cvar_ci_lo is not None and cvar_ci_hi is not None:
            variants["cvar_eval_only"] = (float(cvar_ci_lo), float(cvar_ci_hi), float(cvar_est))
        if cvar_se_total is not None and cvar_se_total == cvar_se_total:
            half = 1.96 * float(cvar_se_total)
            variants["cvar_var_total"] = (cvar_est - half, cvar_est + half, cvar_est)
            if cvar_est_aug is not None and cvar_est_aug == cvar_est_aug:
                variants["cvar_aug_var_total"] = (
                    cvar_est_aug - half, cvar_est_aug + half, cvar_est_aug,
                )
            if cvar_est_cov is not None and cvar_est_cov == cvar_est_cov:
                variants["cvar_cov_var_total"] = (
                    cvar_est_cov - half, cvar_est_cov + half, cvar_est_cov,
                )
            if cvar_est_aug_cov is not None and cvar_est_aug_cov == cvar_est_aug_cov:
                variants["cvar_aug_cov_var_total"] = (
                    cvar_est_aug_cov - half, cvar_est_aug_cov + half, cvar_est_aug_cov,
                )

    # Mean variants
    mean_cje_est = row.get("mean_cje_est")
    mean_cje_aug = row.get("mean_cje_aug")
    mean_cje_est_cov = row.get("mean_cje_est_cov")
    mean_se_total = row.get("mean_se_total")
    mean_ci_lo = row.get("mean_ci_lo")
    mean_ci_hi = row.get("mean_ci_hi")

    if mean_cje_est is not None:
        if mean_ci_lo is not None and mean_ci_hi is not None:
            variants["mean_eval_only"] = (
                float(mean_ci_lo), float(mean_ci_hi), float(mean_cje_est),
            )
        if mean_se_total is not None and mean_se_total == mean_se_total:
            half = 1.96 * float(mean_se_total)
            variants["mean_var_total"] = (
                mean_cje_est - half, mean_cje_est + half, mean_cje_est,
            )
            if mean_cje_aug is not None and mean_cje_aug == mean_cje_aug:
                variants["mean_aug_var_total"] = (
                    mean_cje_aug - half, mean_cje_aug + half, mean_cje_aug,
                )
            if mean_cje_est_cov is not None and mean_cje_est_cov == mean_cje_est_cov:
                variants["mean_cov_var_total"] = (
                    mean_cje_est_cov - half, mean_cje_est_cov + half, mean_cje_est_cov,
                )

    return variants


def _truth_for_variant(row: dict, variant_name: str) -> float | None:
    """For each variant, what's the right `truth` to compare against?

    CVaR variants → full_oracle_truth (CVaR_α on the full oracle panel).
    Mean variants → mean_Y (mean of full oracle panel).
    """
    if variant_name.startswith("cvar"):
        t = row.get("full_oracle_truth")
    else:
        t = row.get("mean_Y")
    return None if t is None else float(t)


def run(
    n_seeds: int = 20,
    alphas: tuple[float, ...] = (0.10, 0.20),
    coverage: float = 0.25,
    seed_base: int = 1000,
    verbose: bool = True,
) -> dict:
    # results: nested dict {(policy, alpha): {variant: [(covered_bool, ci_lo, ci_hi, center, truth, est), ...]}}
    per_cell: dict[tuple[str, float], dict[str, list]] = {}

    for s_offset in range(n_seeds):
        seed = seed_base + s_offset
        for alpha in alphas:
            try:
                rows = step5_oracle_calibrated_uniform(
                    coverage=coverage, alpha=alpha, seed=seed, verbose=False,
                )
            except Exception as e:
                if verbose:
                    print(f"[coverage_sweep] seed={seed} α={alpha}: ERROR {e}")
                continue
            for r in rows:
                key = (r["policy"], float(alpha))
                per_cell.setdefault(key, {})
                variants = _row_ci_variants(r)
                for vname, (lo, hi, center) in variants.items():
                    truth = _truth_for_variant(r, vname)
                    if truth is None or lo != lo or hi != hi:
                        continue
                    covered = bool(lo <= truth <= hi)
                    per_cell[key].setdefault(vname, []).append({
                        "seed": seed,
                        "covered": covered,
                        "lo": float(lo), "hi": float(hi), "center": float(center),
                        "truth": float(truth),
                    })
        if verbose and (s_offset + 1) % 5 == 0:
            print(f"[coverage_sweep] {s_offset + 1}/{n_seeds} seeds done")

    # Aggregate to coverage rates
    results = []
    for (policy, alpha), variants in sorted(per_cell.items()):
        for vname, rec_list in sorted(variants.items()):
            n = len(rec_list)
            k = sum(1 for rec in rec_list if rec["covered"])
            wilson_lo, wilson_hi = _wilson_interval(k, n)
            results.append({
                "policy": policy,
                "alpha": alpha,
                "variant": vname,
                "n_seeds": n,
                "n_covered": k,
                "coverage": k / n if n > 0 else float("nan"),
                "wilson_lo": wilson_lo,
                "wilson_hi": wilson_hi,
                "records": rec_list,
            })

    payload = {
        "n_seeds_requested": n_seeds,
        "alphas": list(alphas),
        "coverage_design": coverage,
        "seed_base": seed_base,
        "rows": results,
    }
    return payload


def write_outputs(payload: dict) -> None:
    WRITEUP_DIR.mkdir(parents=True, exist_ok=True)

    # Strip the per-record details for the headline JSON; keep them in a
    # sibling file in case someone wants to debug.
    headline_rows = []
    for r in payload["rows"]:
        slim = {k: v for k, v in r.items() if k != "records"}
        headline_rows.append(slim)
    headline = {
        "n_seeds_requested": payload["n_seeds_requested"],
        "alphas": payload["alphas"],
        "coverage_design": payload["coverage_design"],
        "seed_base": payload["seed_base"],
        "rows": headline_rows,
    }
    OUT_JSON.write_text(json.dumps(headline, indent=2))

    # Markdown summary table — one block per α, rows = (policy, variant), cols = (n, coverage, Wilson 95%).
    lines = [
        "# Coverage sweep on existing HealthBench panel\n",
        f"`n_seeds = {payload['n_seeds_requested']}`, oracle coverage = `{payload['coverage_design']}`, seed_base = `{payload['seed_base']}`.\n",
        "Per-variant coverage rate is the fraction of seeds whose 95% CI covers the full-oracle truth.\n",
        "Wilson 95% interval on the coverage rate (handles small-N).\n",
    ]
    for alpha in payload["alphas"]:
        lines.append(f"\n## α = {alpha}\n")
        lines.append("| policy | variant | n | covered | rate | Wilson 95% |")
        lines.append("|---|---|---|---|---|---|")
        for r in payload["rows"]:
            if r["alpha"] != alpha:
                continue
            cov_str = f"{r['coverage']:.3f}" if r["coverage"] == r["coverage"] else "NA"
            wil_str = f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]"
            lines.append(
                f"| {r['policy']} | {r['variant']} | {r['n_seeds']} | "
                f"{r['n_covered']} | {cov_str} | {wil_str} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()
    payload = run(
        n_seeds=args.n_seeds,
        alphas=tuple(args.alphas),
        coverage=args.coverage,
        seed_base=args.seed_base,
    )
    write_outputs(payload)
    print(f"[coverage_sweep] wrote {OUT_JSON} and {OUT_MD}")
    print(f"[coverage_sweep] {len(payload['rows'])} (policy, alpha, variant) rows")


if __name__ == "__main__":
    main()
