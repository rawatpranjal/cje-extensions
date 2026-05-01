"""HH red-team-attempts: cellwise (mean, CVaR_0.05, CVaR_0.01) with bootstrap CIs,
plus enumeration of all same-mean-different-tail pairs.

Output: cvar_v4/eda/deeper/hh_pairwise.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

# repo root path so cvar_v3 imports work
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cvar_v4.eda.datasets.hh_red_team import _load, _filter, SPEC


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def bootstrap_stat(arr: np.ndarray, fn, B: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Returns (point, ci_lo, ci_hi) at 95%."""
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots[b] = fn(arr[idx])
    point = float(fn(arr))
    return point, float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))


def main() -> None:
    print("[load] HH red-team-attempts...", flush=True)
    df = _filter(_load(SPEC))
    print(f"[load] {df.height:,} rows, {df['policy_id'].n_unique()} cells", flush=True)

    y_col = "min_harmlessness_score_transcript"
    cells = []
    for pid_tup, g in df.group_by("policy_id"):
        pid = pid_tup[0] if isinstance(pid_tup, tuple) else pid_tup
        arr = g[y_col].to_numpy()
        if arr.size < 200:
            continue
        mean_pt, mean_lo, mean_hi = bootstrap_stat(arr, np.mean, B=1000, seed=42)
        c05_pt, c05_lo, c05_hi = bootstrap_stat(arr, lambda a: cvar_alpha(a, 0.05), B=1000, seed=42)
        c01_pt, c01_lo, c01_hi = bootstrap_stat(arr, lambda a: cvar_alpha(a, 0.01), B=1000, seed=42)
        q05 = float(np.quantile(arr, 0.05))
        cells.append({
            "policy": pid,
            "n": int(arr.size),
            "mean": mean_pt, "mean_lo": mean_lo, "mean_hi": mean_hi,
            "q_05": q05,
            "cvar_05": c05_pt, "cvar_05_lo": c05_lo, "cvar_05_hi": c05_hi,
            "cvar_01": c01_pt, "cvar_01_lo": c01_lo, "cvar_01_hi": c01_hi,
        })

    cells.sort(key=lambda r: r["mean"])
    print(f"[stats] computed {len(cells)} cells", flush=True)

    # Find same-mean / different-tail pairs.
    # Criterion: 95% CIs overlap on mean (max_lo < min_hi) AND don't overlap on CVaR_0.05.
    pairs = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            mean_overlap = max(a["mean_lo"], b["mean_lo"]) < min(a["mean_hi"], b["mean_hi"])
            cvar_overlap = max(a["cvar_05_lo"], b["cvar_05_lo"]) < min(a["cvar_05_hi"], b["cvar_05_hi"])
            if mean_overlap and not cvar_overlap:
                dmean = abs(a["mean"] - b["mean"])
                dcvar = abs(a["cvar_05"] - b["cvar_05"])
                pairs.append({
                    "a": a, "b": b, "dmean": dmean, "dcvar": dcvar,
                    "ratio": dcvar / max(dmean, 0.01),
                })
    pairs.sort(key=lambda p: -p["ratio"])
    print(f"[pairs] found {len(pairs)} same-mean / different-tail pairs", flush=True)

    # Render markdown.
    out = ["# HH red-team-attempts — pairwise (mean, CVaR_0.05, CVaR_0.01) with 95% bootstrap CIs\n"]
    out.append(f"_n={df.height:,} transcripts; 12 (model_type | num_params) cells; B=1,000 bootstrap reps; Y = `min_harmlessness_score_transcript`_\n")
    out.append("## Cellwise statistics (sorted by mean ascending — π0 candidates at top)\n")
    out.append("| policy | n | mean [95% CI] | q_0.05 | CVaR_0.05 [95% CI] | CVaR_0.01 [95% CI] |")
    out.append("|---|---|---|---|---|---|")
    for c in cells:
        out.append(
            f"| `{c['policy']}` | {c['n']:,} | "
            f"{c['mean']:+.3f} [{c['mean_lo']:+.3f}, {c['mean_hi']:+.3f}] | "
            f"{c['q_05']:+.3f} | "
            f"{c['cvar_05']:+.3f} [{c['cvar_05_lo']:+.3f}, {c['cvar_05_hi']:+.3f}] | "
            f"{c['cvar_01']:+.3f} [{c['cvar_01_lo']:+.3f}, {c['cvar_01_hi']:+.3f}] |"
        )
    out.append("")

    out.append("## Same-mean / different-tail pairs\n")
    out.append("Criterion: 95% mean CIs overlap (means statistically indistinguishable) **AND** 95% CVaR_0.05 CIs do NOT overlap (tails clearly distinguishable). These are the cells where mean-CJE would tie but CVaR-CJE would resolve.\n")
    if pairs:
        out.append("| A | B | Δmean | ΔCVaR_0.05 | ratio | A.mean / A.CVaR / B.mean / B.CVaR |")
        out.append("|---|---|---|---|---|---|")
        for p in pairs[:25]:
            a, b = p["a"], p["b"]
            out.append(
                f"| `{a['policy']}` | `{b['policy']}` | "
                f"{p['dmean']:.3f} | {p['dcvar']:.3f} | **{p['ratio']:.1f}×** | "
                f"{a['mean']:+.3f} / {a['cvar_05']:+.3f} / {b['mean']:+.3f} / {b['cvar_05']:+.3f} |"
            )
    else:
        out.append("_(none — every pair is either ranked equivalent on both axes or differs on both)_")
    out.append("")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "hh_pairwise.md"
    out_path.write_text(text)
    print(f"[write] {out_path}  ({len(text):,} chars, {len(pairs)} pairs)", flush=True)


if __name__ == "__main__":
    main()
