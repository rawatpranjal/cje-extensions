"""Cheap-judge calibration in the tail.

Bin pooled (cheap, oracle) pairs by cheap-score decile and report the
oracle mean and gap per bin. Tells us whether the cheap judge has signal
where CVaR lives (the lower part of the distribution).

Outputs:
    writeup/data/reliability.json
"""
from __future__ import annotations

import argparse

import numpy as np

from ._common import all_pairs_pooled, panel_size, write_json


def compute(n_bins: int = 10) -> dict:
    rows = all_pairs_pooled()
    cheap = np.array([r[2] for r in rows])
    oracle = np.array([r[3] for r in rows])

    quantile_edges = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(cheap, quantile_edges)

    bins = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = ((cheap >= lo) & (cheap < hi)) if k < n_bins - 1 else \
               ((cheap >= lo) & (cheap <= hi))
        if mask.sum() == 0:
            continue
        bins.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "cheap_mean": float(cheap[mask].mean()),
            "oracle_mean": float(oracle[mask].mean()),
            "gap_cheap_minus_oracle": float(cheap[mask].mean() - oracle[mask].mean()),
            "oracle_sd": float(oracle[mask].std()),
        })

    # Bottom-20% summary statistic for the writeup's reliability paragraph
    bottom_cutoff = float(np.quantile(cheap, 0.20))
    bot_mask = cheap <= bottom_cutoff
    bottom = {
        "cutoff_cheap_S": bottom_cutoff,
        "n": int(bot_mask.sum()),
        "cheap_mean": float(cheap[bot_mask].mean()),
        "oracle_mean": float(oracle[bot_mask].mean()),
        "gap": float(cheap[bot_mask].mean() - oracle[bot_mask].mean()),
        "pearson_corr": float(np.corrcoef(cheap[bot_mask], oracle[bot_mask])[0, 1])
                          if bot_mask.sum() >= 2 else float("nan"),
    }

    return {
        "n_bins": n_bins,
        "panel_size": panel_size(),
        "n_pooled_rows": len(rows),
        "bins": bins,
        "bottom_20_pct": bottom,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()
    payload = compute(n_bins=args.n_bins)
    path = write_json("reliability.json", payload)
    print(f"[reliability] {'cheap S bin':<22} {'n':>5} {'cheap mean':>11} "
          f"{'oracle mean':>13} {'gap':>9}")
    for b in payload["bins"]:
        label = f"[{b['lo']:+.3f}, {b['hi']:+.3f})"
        print(f"[reliability] {label:<22} {b['n']:>5} {b['cheap_mean']:>+11.3f} "
              f"{b['oracle_mean']:>+13.3f} {b['gap_cheap_minus_oracle']:>+9.3f}")
    bot = payload["bottom_20_pct"]
    print(f"[reliability] bottom 20%: n={bot['n']}, gap={bot['gap']:+.3f}, "
          f"corr={bot['pearson_corr']:+.3f}")
    print(f"[reliability] wrote {path}")


if __name__ == "__main__":
    main()
