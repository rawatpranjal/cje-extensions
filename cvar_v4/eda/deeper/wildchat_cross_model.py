"""WildChat-4.8M shuffled-stream cross-model analysis.

The shuffled-stream re-pull (`ds.shuffle(buffer=10K, seed=42)` in
`datasets/wildchat_48m.py`) gave us 5 distinct models in the 100K sample:
gpt-3.5-turbo-0301, gpt-4o-2024-08-06, gpt-4-0314, gpt-4o-mini, o1-mini.

We compute per-model (mean, CVaR_0.05) on max_detoxify_toxicity and look
for cross-model same-mean-different-tail pairs. Note: lower max_detox
toxicity is *better* (less toxic). For CVaR semantics consistent with HH
and UltraFeedback (lower-tail = worst behavior), we flip sign:
Y = -max_detoxify_toxicity. Then bottom 5% = most toxic continuations.

Output: cvar_v4/eda/deeper/wildchat_cross_model.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Re-load the section markdown's row data: simpler to re-run the loader
from cvar_v4.eda.datasets.wildchat_48m import _load, _filter, SPEC


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def main() -> None:
    print("[load] WildChat-4.8M (streaming, shuffled, 100K)...", flush=True)
    df = _filter(_load(SPEC))
    print(f"[load] {df.height:,} rows, {df['model'].n_unique()} models", flush=True)

    # Y = -max_detoxify_toxicity (so lower = more toxic; CVaR_0.05 = mean of worst 5%)
    import polars as pl
    df = df.with_columns((-pl.col("max_detoxify_toxicity")).alias("Y_safe"))

    rows = []
    for mod, g in df.group_by("model"):
        mname = mod[0] if isinstance(mod, tuple) else mod
        arr = g["Y_safe"].drop_nulls().to_numpy()
        if arr.size < 200: continue
        rows.append({
            "model": mname,
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "q_05": float(np.quantile(arr, 0.05)),
            "cvar_05": cvar_alpha(arr, 0.05),
            "cvar_01": cvar_alpha(arr, 0.01),
        })
    rows.sort(key=lambda r: r["mean"])

    out = ["# WildChat-4.8M (shuffled stream) — cross-model (mean, CVaR_0.05)\n"]
    out.append(f"_n = {df.height:,} (conversation, model) cells from a shuffled 100K-row stream. "
               f"Y_safe = -max_detoxify_toxicity (higher = safer; lower-tail = worst-toxicity continuations). "
               f"Models: {df['model'].n_unique()} distinct after shuffle._\n")

    out.append("## Per-model statistics\n")
    out.append("| model | n | mean | std | q_0.05 | CVaR_0.05 | CVaR_0.01 |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(f"| `{r['model']}` | {r['n']:,} | {r['mean']:+.5f} | {r['std']:.5f} | "
                   f"{r['q_05']:+.5f} | {r['cvar_05']:+.5f} | {r['cvar_01']:+.5f} |")
    out.append("")

    # Same-mean-different-tail pairs (looser threshold because Y range is ~0.1).
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            dmean = abs(a["mean"] - b["mean"])
            dcvar = abs(a["cvar_05"] - b["cvar_05"])
            # Y range here is ~ [-0.1, 0]; "small" Δmean = 0.001; "large" ΔCVaR = 0.005
            if dmean <= 0.001 and dcvar >= 0.003:
                pairs.append({"a": a, "b": b, "dmean": dmean, "dcvar": dcvar,
                              "ratio": dcvar / max(dmean, 1e-6)})
    pairs.sort(key=lambda p: -p["ratio"])

    out.append("## Cross-model same-mean-different-tail pairs\n")
    out.append("Criterion (Y range ≈ 0.1): |Δmean| ≤ 0.001 AND |ΔCVaR_0.05| ≥ 0.003 (3× the mean threshold).\n")
    if pairs:
        out.append("| model_A | model_B | Δmean | ΔCVaR_0.05 | ratio |")
        out.append("|---|---|---|---|---|")
        for p in pairs[:10]:
            out.append(f"| `{p['a']['model']}` | `{p['b']['model']}` | {p['dmean']:.5f} | {p['dcvar']:.5f} | **{p['ratio']:.0f}×** |")
    else:
        out.append("_(no pairs at this threshold)_")
    out.append("")

    # Toxic flag note
    out.append("## Caveat — toxic subset is absent from this stream\n")
    out.append("Even with `ds.shuffle(buffer=10000, seed=42)` the streamed sample contains 0 conversations "
               "with `toxic=True`. The WildChat-4.8M release likely partitions toxic conversations into a "
               "separate file the streaming loader doesn't reach by default — the 100% non-toxic finding here "
               "is consistent with the dataset card's structure (3.2M non-toxic + 1.5M toxic split into "
               "different parquet shards). To pull representative toxic conversations we'd need to switch "
               "shards explicitly via `data_files='toxic/*.parquet'` or similar. **For audit-discriminative "
               "covariate work we should use this stream's `language` and `country` cardinality (70 / 191) "
               "as effective stratifiers, not `toxic`.**")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "wildchat_cross_model.md"
    out_path.write_text(text)
    print(f"[write] {out_path}  ({len(text):,} chars; {len(pairs)} cross-model pairs)", flush=True)


if __name__ == "__main__":
    main()
