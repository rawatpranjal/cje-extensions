"""End-to-end orchestrator. Runs every diagnostic and prints a summary.

Use after the data pipeline has produced data/responses/ and judge_outputs/.
This script is read-only on those directories; it writes only to writeup/.

Usage:
    python -m cvar_v4.healthbench_data.analyses.run_all \
        [--alpha 0.10] [--alpha-robust 0.20] [--coverage 0.25] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys

from . import (audit_drilldown, audit_truth, base_clone_forensics,
                budget_curve, mean_vs_cvar, reliability, tail_cdf,
                tail_composition, threshold_gap, variance_breakdown)
from ._common import WRITEUP_DATA_DIR, WRITEUP_FIG_DIR, panel_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Headline tail level")
    ap.add_argument("--alpha-robust", type=float, default=0.20, dest="alpha_robust",
                    help="Robustness tail level")
    ap.add_argument("--coverage", type=float, default=0.25,
                    help="Oracle slice coverage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--err-threshold", type=float, default=0.10,
                    dest="err_threshold",
                    help="Threshold for the audit-truth-table cross-tabulation")
    ap.add_argument("--variance-B", type=int, default=500, dest="variance_B",
                    help="Bootstrap reps for variance_breakdown (default 500)")
    args = ap.parse_args()

    n = panel_size()
    print(f"[run_all] panel size detected: n_0 = {n}")
    print(f"[run_all] alpha = {args.alpha} (headline), {args.alpha_robust} (robustness)")
    print(f"[run_all] coverage = {args.coverage}, seed = {args.seed}")
    print()

    if n < 5:
        print("[run_all] ERROR: panel is empty or very small. Run pipeline.py first.",
              file=sys.stderr)
        sys.exit(1)

    summary = {}

    # 1. Headline pilot table at the two alpha values
    print("=" * 60)
    print("Pilot tables (cvar_pilot_table)")
    print("=" * 60)
    from ..cvar_pilot_table import run as run_pilot_table
    for a in (args.alpha, args.alpha_robust):
        rows = run_pilot_table(coverage=args.coverage, alpha=a, seed=args.seed,
                                verbose=False)
        if rows:
            mean_err_direct = sum(r["abs_error"] for r in rows) / len(rows)
            mean_err_cheap = sum(abs(r["cheap_only_cvar"] - r["full_oracle_truth"])
                                  for r in rows) / len(rows)
            print(f"  α={a}: mean Direct |err|={mean_err_direct:.3f}  "
                  f"mean cheap-only |err|={mean_err_cheap:.3f}")
            summary[f"pilot_alpha_{a}"] = {
                "mean_err_direct": mean_err_direct,
                "mean_err_cheap_only": mean_err_cheap,
                "verdicts": {r["policy"]: r["verdict"] for r in rows},
            }

    # 2. The four general diagnostics
    print()
    print("=" * 60)
    print("General diagnostics")
    print("=" * 60)
    print()

    print("[1/5] Oracle-budget curve")
    bc = budget_curve.compute(alpha=args.alpha, seed=args.seed)
    budget_curve.plot(bc, budget_curve.fig_path("budget_curve.pdf"))
    budget_curve.write_json("budget_curve.json", bc)

    print("[2/5] Mean versus CVaR scatter")
    mvc = mean_vs_cvar.compute(alpha=args.alpha, coverage=args.coverage, seed=args.seed)
    mean_vs_cvar.plot(mvc, mean_vs_cvar.fig_path("mean_vs_cvar.pdf"))
    mean_vs_cvar.write_json("mean_vs_cvar.json", mvc)

    print("[3/5] Audit truth table")
    at = audit_truth.compute(alphas=(args.alpha, args.alpha_robust),
                              coverage=args.coverage, seed=args.seed,
                              err_threshold=args.err_threshold)
    audit_truth.write_json("audit_truth.json", at)
    print(f"  pass_low={at['counts']['pass_low']}  pass_high={at['counts']['pass_high']}  "
          f"fail_low={at['counts']['fail_low']}  fail_high={at['counts']['fail_high']}")

    print("[4/5] Tail composition")
    tc = tail_composition.compute(alpha=args.alpha)
    tail_composition.write_json("tail_composition.json", tc)

    # Tail-zoom CDF figure: cheap-S vs oracle-Y vs CJE-calibrated f̂(S),
    # zoomed on the bottom 20% for parallel and unhelpful. Visual proof
    # that the calibrator reconstructs the tail.
    print("[5/5] Tail-zoom CDF figure")
    tcdf = tail_cdf.compute(alpha=args.alpha, coverage=args.coverage, seed=args.seed)
    tail_cdf.plot(tcdf, tail_cdf.fig_path("tail_cdf.pdf"))
    tail_cdf.write_json("tail_cdf.json", tcdf)

    # 3. Three deep dives
    print()
    print("=" * 60)
    print("Targeted deep dives")
    print("=" * 60)
    print()

    print("[1/4] Audit failure drill-down (binomial p-values)")
    ad = audit_drilldown.compute(alphas=(args.alpha, args.alpha_robust),
                                   coverage=args.coverage, seed=args.seed)
    audit_drilldown.write_json("audit_drilldown.json", ad)
    real_flags = sum(1 for r in ad["rows"]
                     if r["verdict"] != "PASS" and r["binomial_p"] < 0.05)
    artifact_flags = sum(1 for r in ad["rows"]
                          if r["verdict"] != "PASS" and r["binomial_p"] >= 0.05)
    print(f"  flagged cells: {real_flags} statistically real, "
          f"{artifact_flags} likely small-sample artifact")

    print("[2/4] Base versus clone tail forensics")
    bc_for = base_clone_forensics.compute(alpha=args.alpha)
    base_clone_forensics.write_json("base_clone.json", bc_for)
    print(f"  common in both bottoms: {len(bc_for['common_in_both_bottoms'])}; "
          f"clone-only: {len(bc_for['clone_only'])}")

    print("[3/4] Cheap-judge calibration in the tail")
    rel = reliability.compute()
    reliability.write_json("reliability.json", rel)
    print(f"  bottom-20%: gap={rel['bottom_20_pct']['gap']:+.3f}, "
          f"corr={rel['bottom_20_pct']['pearson_corr']:+.3f}")

    # Threshold gap t̂ vs t*. Mechanism diagnostic — explains how the
    # estimator can pass the audit even when the calibrator's cutoff is
    # off by a finite gap (stop-loss residual silently corrects via g₂).
    print("[4/4] Threshold gap t̂ vs t*")
    tg = threshold_gap.compute(alpha=args.alpha, coverage=args.coverage,
                                seed=args.seed)
    threshold_gap.write_json("threshold_gap.json", tg)
    threshold_gap.write_md("threshold_gap.md", tg)
    max_gap = max((abs(r["gap"]) for r in tg["rows"]), default=float("nan"))
    print(f"  max |t̂ − t*| = {max_gap:.3f}")

    print()
    print("=" * 60)
    print(f"Variance breakdown (full-pipeline bootstrap, B={args.variance_B})")
    print("=" * 60)
    vb = variance_breakdown.compute(
        coverage=args.coverage,
        alphas=(args.alpha, args.alpha_robust),
        B=args.variance_B, seed=args.seed, verbose=False,
    )
    from ._common import WRITEUP_DATA_DIR as _VBD
    (_VBD / "variance_breakdown.json").write_text(
        __import__("json").dumps(vb, indent=2, default=float)
    )
    (_VBD / "variance_breakdown.md").write_text(variance_breakdown.render_markdown(vb))
    print(f"  rows: {len(vb['rows'])} (5 policies × {len((args.alpha, args.alpha_robust))} alphas × 2 estimators)")

    # 4. Final summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"JSON artifacts: {WRITEUP_DATA_DIR}")
    print(f"Figures: budget_curve.pdf, mean_vs_cvar.pdf, tail_cdf.pdf in {WRITEUP_FIG_DIR}")
    print()
    print("To recompile the writeup with new figures:")
    print(f"  cd cvar_v4/healthbench_data/writeup && pdflatex pilot.tex")


if __name__ == "__main__":
    main()
