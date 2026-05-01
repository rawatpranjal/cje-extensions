"""RealToxicityPrompts: challenging-stratified analysis.

The dataset_options.md doc highlights `challenging` (a ~1.2% subset that
consistently elicited toxic continuations across multiple models) as the
canonical audit-positive subset. We verify:
1. Tail concentration: how much of the bottom-1% Y mass lives in the
   challenging subset vs the rest?
2. Per-stratum CVaR shift: is the same-distribution-different-tail pattern
   visible *within* the toxicity Y distribution conditional on challenging?
3. Audit-discriminativeness: how much does conditioning on `challenging`
   change CVaR_0.05?

We frame Y two ways:
- Y_safe = 1 - continuation_toxicity (lower tail = most toxic continuations)
- Y_tox = continuation_toxicity (upper tail = most toxic; we'll flip sign for CVaR semantics)

Output: cvar_v4/eda/deeper/rtp_challenging.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cvar_v4.eda.datasets.real_toxicity_prompts import _load, _filter, SPEC


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def main() -> None:
    print("[load] RealToxicityPrompts...", flush=True)
    df = _filter(_load(SPEC))
    print(f"[load] {df.height:,} rows", flush=True)

    # We want Y where lower = worse. Let Y_safe = 1 - continuation_toxicity.
    df = df.with_columns((1 - pl.col("continuation_toxicity")).alias("Y_safe"))

    out = ["# RealToxicityPrompts — `challenging` subset stratified analysis\n"]
    out.append(f"_n = {df.height:,} (prompt, continuation) pairs. "
               f"Y_safe = 1 - continuation_toxicity (so lower tail = most toxic continuations). "
               f"`challenging` boolean flag identifies prompts that consistently elicited toxic continuations._\n")

    # Subset sizes
    n_chall = int(df.filter(pl.col("challenging")).height)
    n_other = int(df.filter(~pl.col("challenging")).height)
    print(f"[subsets] challenging={n_chall:,}, other={n_other:,}", flush=True)

    out.append("## Subset sizes\n")
    out.append(f"- `challenging = True`: **{n_chall:,}** rows ({n_chall/df.height*100:.1f}%)")
    out.append(f"- `challenging = False`: **{n_other:,}** rows ({n_other/df.height*100:.1f}%)")
    out.append("")

    # Y_safe distribution per subset
    out.append("## Y_safe = 1 - toxicity per subset\n")
    out.append("| subset | n | mean | std | q_0.05 | q_0.01 | CVaR_0.05 | CVaR_0.01 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for label, mask_expr in [("challenging=True", pl.col("challenging")), ("challenging=False", ~pl.col("challenging")), ("ALL", pl.lit(True))]:
        sub = df.filter(mask_expr)
        arr = sub["Y_safe"].drop_nulls().to_numpy()
        out.append(
            f"| {label} | {arr.size:,} | {arr.mean():.4f} | {arr.std():.4f} | "
            f"{np.quantile(arr, 0.05):.4f} | {np.quantile(arr, 0.01):.4f} | "
            f"{cvar_alpha(arr, 0.05):.4f} | {cvar_alpha(arr, 0.01):.4f} |"
        )
    out.append("")

    # Where does the bottom-1% mass live?
    arr_all = df["Y_safe"].drop_nulls().to_numpy()
    threshold = np.quantile(arr_all, 0.01)
    bottom_1pct_mask = df["Y_safe"] <= threshold
    n_bot = int(bottom_1pct_mask.sum())
    n_bot_in_chall = int((bottom_1pct_mask & df["challenging"]).sum())
    n_bot_in_other = n_bot - n_bot_in_chall
    chall_share_in_bottom = n_bot_in_chall / max(n_bot, 1) * 100
    chall_share_overall = n_chall / df.height * 100

    out.append("## Tail concentration in `challenging` subset\n")
    out.append(f"At Y_safe ≤ q_0.01 = {threshold:.4f} (the global 1% threshold):")
    out.append(f"- {n_bot:,} rows total in this tail")
    out.append(f"- {n_bot_in_chall:,} of them ({chall_share_in_bottom:.1f}%) are in `challenging=True`")
    out.append(f"- vs `challenging=True` is {chall_share_overall:.1f}% of the full corpus")
    out.append(f"- **Concentration ratio**: {chall_share_in_bottom / max(chall_share_overall, 0.01):.2f}× — "
               f"the challenging subset is {chall_share_in_bottom / max(chall_share_overall, 0.01):.1f}× over-represented in the worst-1% tail.")
    out.append("")

    # Same for bottom 5%
    threshold5 = np.quantile(arr_all, 0.05)
    bottom_5pct_mask = df["Y_safe"] <= threshold5
    n_bot5 = int(bottom_5pct_mask.sum())
    n_bot5_chall = int((bottom_5pct_mask & df["challenging"]).sum())
    chall_share_bot5 = n_bot5_chall / max(n_bot5, 1) * 100

    out.append(f"At Y_safe ≤ q_0.05 = {threshold5:.4f} (the 5% threshold):")
    out.append(f"- {n_bot5:,} rows; {n_bot5_chall:,} ({chall_share_bot5:.1f}%) are challenging — "
               f"concentration ratio {chall_share_bot5 / max(chall_share_overall, 0.01):.2f}×.")
    out.append("")

    # The CVaR-CJE motivation: same-mean-different-tail across `challenging` subsets
    out.append("## CVaR-CJE motivation: π_base policy on full corpus vs π_audit-restricted\n")
    out.append("Suppose we evaluate two hypothetical policies whose responses we'd generate fresh:\n")
    out.append("- π_uniform: applies same generation strategy to every prompt → its tail toxicity matches the dataset's overall pattern")
    out.append("- π_challenging-aware: routes `challenging` prompts to a stricter refusal policy, others to π_uniform\n")
    out.append(f"On the full {df.height:,}-row corpus the two policies' *mean* Y_safe will be near-identical "
               f"(challenging is only {chall_share_overall:.1f}% of mass). But their *CVaR_0.05* will diverge "
               f"by approximately the per-subset gap in `Y_safe @ q_0.05` between challenging and other subsets:")
    chall_q05 = float(np.quantile(df.filter(pl.col("challenging"))["Y_safe"].to_numpy(), 0.05))
    other_q05 = float(np.quantile(df.filter(~pl.col("challenging"))["Y_safe"].to_numpy(), 0.05))
    out.append(f"\n- Y_safe q_0.05 conditional on `challenging=True`: **{chall_q05:.4f}**")
    out.append(f"- Y_safe q_0.05 conditional on `challenging=False`: **{other_q05:.4f}**")
    out.append(f"- gap: **{abs(chall_q05 - other_q05):.4f}**")
    out.append("")
    out.append("This is exactly the 'same prompts, different response strategies → similar mean, different tail' "
               "story that motivates the method — and the `challenging` flag IS the audit covariate that surfaces it.")
    out.append("")

    # The audit-rejection prediction: a calibrator trained on challenging=False's S→Y won't transport to challenging=True.
    out.append("## Audit prediction\n")
    out.append("Per the framework, calibrate on the `challenging=False` subset (the body of the data, ~98.8%) and run "
               "the audit on `challenging=True`. The audit should fire because:")
    out.append("- The Y conditional distribution under challenging-True is shifted left of challenging-False")
    out.append("- The S→Y relationship (Perspective toxicity → 1 - Perspective toxicity) is mechanically similar but the realized values differ")
    out.append("- An honest CVaR-CJE pipeline must refuse to certify the level claim on the challenging stratum and report a stratum-conditional CVaR instead.")
    out.append("")
    out.append("This is the cleanest *within-dataset* audit-discriminativeness demo we have — **a single boolean covariate that the audit must flag**.")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "rtp_challenging.md"
    out_path.write_text(text)
    print(f"[write] {out_path}  ({len(text):,} chars)", flush=True)


if __name__ == "__main__":
    main()
