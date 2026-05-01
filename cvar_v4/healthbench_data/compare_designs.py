"""Compare 3 oracle-slice designs × 3 audit variants on the current panel.

Runs the parameterized step5 for every (design, audit) cell, writes results
to results_design_comparison.jsonl + .md.

Example:
    python3 -m cvar_v4.healthbench_data.compare_designs \\
        --coverage 0.25 --alpha 0.10 --seed 42

Output columns per row:
    policy, design, audit_variant, coverage, alpha, seed,
    n_total, n_slice, cvar_est, t_hat, full_oracle_truth, abs_error,
    audit_p, audit_reject, mean_g1, mean_g2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import step5_oracle_calibrated_designed
from .oracle_design import VALID_DESIGNS

try:
    from eda.deeper._estimator import AUDIT_VARIANTS
except Exception:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eda.deeper._estimator import AUDIT_VARIANTS


OUT_DIR = Path(__file__).parent


def _output_paths(alpha: float) -> tuple[Path, Path]:
    """Per-alpha output paths so multiple α values can coexist on disk."""
    suffix = f"alpha_{alpha:.2f}".rstrip("0").rstrip(".")
    return (
        OUT_DIR / f"results_design_comparison_{suffix}.jsonl",
        OUT_DIR / f"results_design_comparison_{suffix}.md",
    )


def run_grid(*, coverage: float, alpha: float, seed: int,
             B: int = 200, B_ci: int = 500, K_jackknife: int = 5) -> list[dict]:
    rows: list[dict] = []
    for design in VALID_DESIGNS:
        for audit_variant in AUDIT_VARIANTS:
            print(f"\n--- design={design} audit={audit_variant} ---")
            rows.extend(step5_oracle_calibrated_designed(
                design=design, audit_variant=audit_variant,
                coverage=coverage, alpha=alpha, seed=seed,
                B=B, B_ci=B_ci, K_jackknife=K_jackknife, verbose=True,
            ))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _md_table_per_policy(rows: list[dict]) -> str:
    """One block per policy with two views:
       1. CI table — calibration-aware CIs and Var_cal share per design (mean audit is per-policy)
       2. Audit table — design × audit_variant rejection grid
    """
    out: list[str] = []
    by_policy: dict[str, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    for policy, prows in by_policy.items():
        out.append(f"### {policy}\n")
        if not prows:
            continue
        truth = prows[0]["full_oracle_truth"]
        n_total = prows[0]["n_total"]
        out.append(f"Full-oracle CVaR_α truth = `{truth:+.3f}` (n_total={n_total})\n")

        # View 1: CI + var_cal_share + mean audit per design (these don't depend on audit_variant)
        designs = sorted({r["design"] for r in prows})
        out.append("**CI + variance decomposition + mean audit (per design):**\n")
        out.append("| design | n_slice | est | 95% CI (cal-aware) | \\|err\\| | in_CI | Var_cal/Var_total | mean_audit_p | mean_residual |")
        out.append("|---|---:|---:|---|---:|:--:|---:|---:|---:|")
        for d in designs:
            # All rows for this (policy, design) share these values; pick one
            m = next((r for r in prows if r["design"] == d), None)
            if m is None:
                continue
            in_ci = "✓" if m.get("in_ci_total") else "✗"
            m_flag = "🔥" if m.get("mean_audit_reject") else ""
            out.append(
                f"| {d} | {m['n_slice']} | {m['cvar_est']:+.3f} | "
                f"[{m.get('ci_lo_total', float('nan')):+.3f}, {m.get('ci_hi_total', float('nan')):+.3f}] | "
                f"{m['abs_error']:.3f} | {in_ci} | {m.get('var_cal_share', float('nan')):.2f} | "
                f"{m.get('mean_audit_p', float('nan')):.3f} {m_flag} | {m.get('mean_residual', float('nan')):+.3f} |"
            )
        out.append("")

        # View 2: audit_variant × design rejection grid
        audits = sorted({r["audit_variant"] for r in prows})
        out.append("**CVaR audit p-values (per design × audit_variant):**\n")
        header = "| design \\ audit | " + " | ".join(audits) + " |"
        sep = "|" + "---|" * (len(audits) + 1)
        out.append(header)
        out.append(sep)
        for d in designs:
            cells = []
            for a in audits:
                m = next((r for r in prows if r["design"] == d and r["audit_variant"] == a), None)
                if m is None:
                    cells.append("—")
                else:
                    flag = "🔥" if m["audit_reject"] else "✅"
                    cells.append(f"p={m['audit_p']:.3f} {flag}")
            out.append(f"| {d} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def _md_summary(rows: list[dict], coverage: float, alpha: float) -> str:
    """Aggregate by (design, audit): mean abs_error, mean reject rate, etc."""
    out: list[str] = []
    out.append("## Aggregate summary")
    out.append("")
    out.append(f"Across all policies. coverage = {coverage}, α = {alpha}.\n")
    out.append("| design | audit | mean \\|err\\| | reject_rate | n_cells |")
    out.append("|---|---|---:|---:|---:|")
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["design"], r["audit_variant"]), []).append(r)
    for (d, a), cells in sorted(by_key.items()):
        if not cells:
            continue
        mae = sum(c["abs_error"] for c in cells) / len(cells)
        rej = sum(int(c["audit_reject"]) for c in cells) / len(cells)
        out.append(f"| {d} | {a} | {mae:.3f} | {rej:.2f} | {len(cells)} |")
    return "\n".join(out)


def write_md(rows: list[dict], path: Path, coverage: float, alpha: float, seed: int) -> None:
    head = (
        f"# Design × audit comparison\n\n"
        f"coverage = `{coverage}`, α = `{alpha}`, seed = `{seed}`. "
        f"Estimator: HT-weighted isotonic Direct CVaR-CJE. "
        f"Audit Σ̂: paired bootstrap with t̂ re-maximization (B=200). "
        f"Full-oracle truth = CVaR_α computed on every row's oracle label.\n\n"
    )
    body = _md_table_per_policy(rows)
    summary = _md_summary(rows, coverage, alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + body + "\n\n" + summary + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.25,
                    help="Target oracle slice coverage (e.g. 0.25)")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Lower-tail CVaR level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=200, help="Bootstrap reps for audit Σ̂")
    ap.add_argument("--B-ci", type=int, default=500, dest="B_ci",
                    help="Bootstrap reps for cvar_est CI (default 500)")
    ap.add_argument("--K-jackknife", type=int, default=5, dest="K_jackknife",
                    help="Folds for Var_cal jackknife (default 5)")
    args = ap.parse_args()

    rows = run_grid(coverage=args.coverage, alpha=args.alpha, seed=args.seed,
                    B=args.B, B_ci=args.B_ci, K_jackknife=args.K_jackknife)
    jsonl_out, md_out = _output_paths(args.alpha)
    write_jsonl(rows, jsonl_out)
    write_md(rows, md_out, coverage=args.coverage, alpha=args.alpha, seed=args.seed)
    print(f"\n[compare-designs] wrote {len(rows)} rows to {jsonl_out}")
    print(f"[compare-designs] wrote summary to {md_out}")


if __name__ == "__main__":
    main()
