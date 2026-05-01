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


OUT_DIR = Path(__file__).parent / "writeup" / "data"


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
        f"| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |",
        f"|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['n_slice']} | "
            f"{r['cvar_est']:+.3f} | {r['full_oracle_truth']:+.3f} | "
            f"{r['abs_error']:.3f} | {r['mean_g1']:+.3f} | {r['mean_g2']:+.3f} | "
            f"{r['verdict']} |"
        )
    lines.append("")
    lines.append(f"**Notes**:")
    lines.append(f"- Atom-split CVaR_α averages exactly α·n units of mass on the sorted-tail; "
                 f"the older naive `mean(y[y ≤ quantile_α])` over-averaged on ties (relevant for HealthBench's tied-zero floor).")
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
