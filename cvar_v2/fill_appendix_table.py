"""Auto-fill the empirical-results table in cvar/extension_appendix.tex
from cvar/results_arena.csv.

Replaces the placeholder block (between the two `% AUTOFILL` markers) with
LaTeX rows generated from the run results at α=0.10.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import polars as pl

TEX = Path("cvar/extension_appendix.tex")
CSV = Path("cvar/results_arena.csv")
ALPHA = 0.10
ROW_ORDER = ["parallel_universe_prompt", "premium", "clone", "unhelpful"]


def main() -> int:
    df = pl.read_csv(CSV).filter(pl.col("alpha") == ALPHA)
    agg = df.group_by("policy").agg([
        pl.col("mean").median().alias("mean"),
        pl.col("mean_ci_lo").median().alias("mean_ci_lo"),
        pl.col("mean_ci_hi").median().alias("mean_ci_hi"),
        pl.col("oracle_truth").median().alias("mean_truth"),
        pl.col("cvar").median().alias("cvar"),
        pl.col("cvar_ci_lo").median().alias("cvar_ci_lo"),
        pl.col("cvar_ci_hi").median().alias("cvar_ci_hi"),
        pl.col("cvar_empirical_truth").median().alias("cvar_truth"),
        pl.col("audit_p_value").median().alias("audit_p"),
        pl.col("audit_reject").mean().alias("reject_rate"),
    ])
    rows = {r["policy"]: r for r in agg.iter_rows(named=True)}

    lines = []
    for p in ROW_ORDER:
        r = rows[p]
        # Escape underscores for LaTeX
        ptex = "\\texttt{" + p.replace("_", "\\_") + "}"
        reject_pct = f"{int(round(r['reject_rate'] * 100))}\\%"
        lines.append(
            f"{ptex:<48} & {r['mean']:.3f} [{r['mean_ci_lo']:.3f}, {r['mean_ci_hi']:.3f}] & "
            f"{r['mean_truth']:.3f} & "
            f"{r['cvar']:.3f} [{r['cvar_ci_lo']:.3f}, {r['cvar_ci_hi']:.3f}] & "
            f"{r['cvar_truth']:.3f} & "
            f"{r['audit_p']:.3g} & "
            f"{reject_pct} \\\\"
        )
    body = "\n".join(lines)

    tex = TEX.read_text()
    new_block = (
        "% AUTOFILL-START\n"
        + body + "\n"
        + "% AUTOFILL-END"
    )
    # Use a callable replacement so backslashes in `new_block` (e.g. \texttt{})
    # aren't interpreted by re.sub as escape sequences (\t -> TAB, etc).
    pat = re.compile(r"% AUTOFILL-START.*?% AUTOFILL-END", re.DOTALL)
    if pat.search(tex):
        tex = pat.sub(lambda _m: new_block, tex)
    else:
        # First-time fill: replace the manual TBD block.
        old = re.compile(
            r"\\texttt\{parallel\\?_universe\\?_prompt\}.*?\\texttt\{unhelpful\}.*?\\\\\n",
            re.DOTALL,
        )
        if not old.search(tex):
            print("Could not find table body to replace. Aborting.", file=sys.stderr)
            return 1
        tex = old.sub(lambda _m: new_block + "\n", tex)
    TEX.write_text(tex)
    print(f"Updated {TEX} with {len(ROW_ORDER)} rows from {CSV}.")
    print("\nFilled rows:")
    for ln in lines:
        print(" ", ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
