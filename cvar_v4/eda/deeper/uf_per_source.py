"""UltraFeedback: per-source × per-model (mean, CVaR_0.05) — does the
same-mean-different-tail pattern hold WITHIN a prompt source?

If yes, source-stratified analysis works as a clean audit covariate. If no
(e.g., the pattern is mostly between-source heterogeneity), we'd want to
re-stratify by something else.

Output: cvar_v4/eda/deeper/uf_per_source.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cvar_v4.eda.datasets.ultrafeedback import _load, _filter, SPEC


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def main() -> None:
    print("[load] UltraFeedback...", flush=True)
    df = _filter(_load(SPEC))
    print(f"[load] {df.height:,} rows", flush=True)

    y = "overall_score"
    # Top sources by row count
    src_counts = df.group_by("source").agg(pl.len().alias("n")).sort("n", descending=True).head(6)
    top_sources = [r["source"] for r in src_counts.iter_rows(named=True)]
    print(f"[sources] top 6: {top_sources}", flush=True)

    # Build per-source per-model table
    out = ["# UltraFeedback — per-source × per-model (mean, CVaR_0.05)\n"]
    out.append(f"_n={df.height:,} (instruction × completion) cells; Y = `{y}` (range 1..10); "
               f"6 largest prompt sources × 17 models. CVaR_0.05 = mean of bottom 5% per (source, model) cell._\n")

    out.append("## Cross-source overall (sanity)\n")
    overall_table = []
    for src, g in df.group_by("source"):
        sname = src[0] if isinstance(src, tuple) else src
        if sname not in top_sources: continue
        arr = g[y].drop_nulls().to_numpy()
        overall_table.append({
            "source": sname,
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "cvar_05": cvar_alpha(arr, 0.05),
        })
    overall_table.sort(key=lambda r: -r["n"])
    out.append("| source | n | mean | CVaR_0.05 |")
    out.append("|---|---|---|---|")
    for r in overall_table:
        out.append(f"| {r['source']} | {r['n']:,} | {r['mean']:.3f} | {r['cvar_05']:.3f} |")
    out.append("")

    # Per-source per-model
    out.append("## Per-source per-model decomposition\n")
    out.append("For each top source, ranked by mean. **Bold** rows have CVaR_0.05 ≥ 1.0 different "
               "from the source's median model — they're potential same-mean-different-tail candidates.\n")

    same_mean_diff_cvar_pairs_within_source = []

    for src in top_sources:
        sub = df.filter(pl.col("source") == src)
        rows = []
        for mod, g in sub.group_by("model"):
            mname = mod[0] if isinstance(mod, tuple) else mod
            arr = g[y].drop_nulls().to_numpy()
            if arr.size < 200:
                continue
            rows.append({
                "model": mname,
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "cvar_05": cvar_alpha(arr, 0.05),
            })
        rows.sort(key=lambda r: r["mean"])
        if not rows: continue

        # Find within-source same-mean-different-tail pairs.
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                a, b = rows[i], rows[j]
                dmean = abs(a["mean"] - b["mean"])
                dcvar = abs(a["cvar_05"] - b["cvar_05"])
                if dmean <= 0.2 and dcvar >= 0.5:
                    same_mean_diff_cvar_pairs_within_source.append({
                        "source": src, "a": a, "b": b, "dmean": dmean, "dcvar": dcvar,
                        "ratio": dcvar / max(dmean, 0.01),
                    })

        median_cvar = float(np.median([r["cvar_05"] for r in rows]))
        out.append(f"### source = `{src}`  (n = {sum(r['n'] for r in rows):,})\n")
        out.append("| model | n | mean | CVaR_0.05 | Δmean from median | ΔCVaR from median |")
        out.append("|---|---|---|---|---|---|")
        median_mean = float(np.median([r["mean"] for r in rows]))
        for r in rows:
            dm = r["mean"] - median_mean
            dc = r["cvar_05"] - median_cvar
            bold = "**" if abs(dc) >= 1.0 else ""
            out.append(f"| {bold}`{r['model']}`{bold} | {r['n']:,} | {r['mean']:.3f} | {r['cvar_05']:.3f} | {dm:+.3f} | {dc:+.3f} |")
        out.append("")

    # Within-source same-mean-different-tail pairs
    out.append("## Within-source same-mean-different-tail pairs\n")
    out.append("Pairs of (model_A, model_B) within the *same* prompt source where |Δmean| ≤ 0.2 and |ΔCVaR_0.05| ≥ 0.5. "
               "These are the cleanest CVaR-CJE demonstrations because the prompt distribution is held fixed — "
               "the tail difference reflects *policy*, not *prompt*.\n")
    same_mean_diff_cvar_pairs_within_source.sort(key=lambda p: -p["ratio"])
    if same_mean_diff_cvar_pairs_within_source:
        out.append("| source | model_A | model_B | mean_A | CVaR_A | mean_B | CVaR_B | Δmean | ΔCVaR | ratio |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for p in same_mean_diff_cvar_pairs_within_source[:30]:
            out.append(
                f"| {p['source']} | `{p['a']['model']}` | `{p['b']['model']}` | "
                f"{p['a']['mean']:.3f} | {p['a']['cvar_05']:.3f} | "
                f"{p['b']['mean']:.3f} | {p['b']['cvar_05']:.3f} | "
                f"{p['dmean']:.3f} | {p['dcvar']:.3f} | **{p['ratio']:.1f}×** |"
            )
    else:
        out.append("_(none)_")
    out.append("")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "uf_per_source.md"
    out_path.write_text(text)
    print(f"[write] {out_path}  ({len(text):,} chars; "
          f"{len(same_mean_diff_cvar_pairs_within_source)} within-source same-mean-diff-tail pairs)", flush=True)


if __name__ == "__main__":
    main()
