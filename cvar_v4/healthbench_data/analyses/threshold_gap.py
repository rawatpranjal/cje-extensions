"""Threshold gap diagnostic: t̂ (saddle-point cheap-derived) vs t* (oracle truth).

Why this diagnostic exists
--------------------------
The Direct CVaR-CJE estimator picks `t̂` as the argmax of its saddle-point
objective on the FULL cheap target distribution. The truth `t*` is the
atom-split α-quantile of the full-oracle Y panel.

Two things can happen:

1. **Cutoff agreement** — `|t̂ − t*|` is small. The calibrator found the
   right tail boundary; CVaR-CJE works because it's averaging shortfalls
   below an essentially correct threshold.

2. **Stop-loss correction** — `|t̂ − t*|` is non-trivial, BUT the audit
   passes and `abs_error` is still small. The calibrator picked the
   wrong cutoff, but the stop-loss residual `(t̂ − Y)_+ − ĝ_t̂(S)`
   silently corrects via the audit's g₂ moment (because the saddle-point
   derivative at the wrong t̂ trades off tail-mass against shortfall and
   reaches the same CVaR value).

This is a **mechanism diagnostic** — it explains *how* the estimator can
pass the audit even when the calibrator's cutoff is off by a finite gap.
It does NOT gate any claim; the audit's g₁ + g₂ tests still do that.

Outputs
-------
- `writeup/data/threshold_gap.json` — programmatic
- `writeup/data/threshold_gap.md`   — readable mini-table

Usage
-----
    python -m cvar_v4.healthbench_data.analyses.threshold_gap [--alpha 0.10] \
        [--coverage 0.25] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..analyze import step5_oracle_calibrated_uniform
from ._common import LABELS, WRITEUP_DATA_DIR, write_json


def compute(*, alpha: float = 0.10, coverage: float = 0.25, seed: int = 42) -> dict:
    """Reuse step5 — every row already exposes t_hat and t_star."""
    rows = step5_oracle_calibrated_uniform(
        coverage=coverage, alpha=alpha, seed=seed, verbose=False,
    )
    out_rows = []
    for r in rows:
        t_hat = float(r["t_hat"])
        t_star = float(r["t_star"])
        out_rows.append({
            "policy": r["policy"],
            "n_total": int(r["n_total"]),
            "n_slice": int(r["n_slice"]),
            "t_hat": t_hat,
            "t_star": t_star,
            "gap": t_hat - t_star,
            "abs_error": float(r["abs_error"]),
            "verdict": r["verdict"],
            # Include the moment values so the reader can spot the
            # stop-loss-correction case (|gap| > 0.05 AND |g_2| small AND
            # abs_error small).
            "mean_g1": float(r["mean_g1"]),
            "mean_g2": float(r["mean_g2"]),
        })
    return {
        "alpha": float(alpha), "coverage": float(coverage), "seed": int(seed),
        "rows": out_rows,
    }


def write_md(name: str, payload: dict) -> Path:
    """Write a readable markdown table to writeup/data/{name}."""
    WRITEUP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = WRITEUP_DATA_DIR / name
    if not name.endswith(".md"):
        path = path.with_suffix(".md")
    rows = payload["rows"]
    if not rows:
        path.write_text("# Threshold gap\n\nNo rows.\n")
        return path
    lines = [
        f"# Threshold gap t̂ vs t*",
        "",
        f"- **α** = `{payload['alpha']}`",
        f"- **coverage** = `{payload['coverage']}`",
        f"- **seed** = `{payload['seed']}`",
        f"- `t̂` = saddle-point threshold from FULL cheap target distribution",
        f"- `t*` = atom-split α-quantile of full oracle Y panel (truth)",
        f"- `gap = t̂ − t*` — small = direct cutoff agreement; non-trivial gap with low |error| = stop-loss correction via g₂",
        "",
        f"| policy | n_audit | t̂ | t* | gap | |error| | mean_g1 | mean_g2 | verdict |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {LABELS.get(r['policy'], r['policy'])} | "
            f"{r['n_slice']} | "
            f"{r['t_hat']:+.3f} | {r['t_star']:+.3f} | {r['gap']:+.3f} | "
            f"{r['abs_error']:.3f} | "
            f"{r['mean_g1']:+.3f} | {r['mean_g2']:+.3f} | "
            f"{r['verdict']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = compute(alpha=args.alpha, coverage=args.coverage, seed=args.seed)
    json_path = write_json("threshold_gap.json", payload)
    md_path = write_md("threshold_gap.md", payload)
    print(f"[threshold_gap] wrote {json_path}")
    print(f"[threshold_gap] wrote {md_path}")


if __name__ == "__main__":
    main()
