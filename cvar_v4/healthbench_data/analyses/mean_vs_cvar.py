"""Mean versus CVaR scatter: per-policy (mean Y, CVaR_α) with Direct overlay.

Outputs:
    writeup/data/mean_vs_cvar.json
    writeup/mean_vs_cvar.pdf
"""
from __future__ import annotations

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as ml
import matplotlib.pyplot as plt

from ..analyze import step5_oracle_calibrated_uniform
from ._common import (COLORS, LABELS, MARKERS, POLICIES, fig_path,
                       panel_size, write_json)


def compute(alpha: float = 0.10, coverage: float = 0.25, seed: int = 42) -> dict:
    rows = step5_oracle_calibrated_uniform(
        coverage=coverage, alpha=alpha, seed=seed, verbose=False,
    )
    return {
        "alpha": alpha,
        "coverage": coverage,
        "seed": seed,
        "panel_size": panel_size(),
        "rows": rows,
    }


def plot(payload: dict, out_path):
    rows = payload["rows"]
    alpha = payload["alpha"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for r in rows:
        p = r["policy"]
        ax.scatter(r["mean_Y"], r["full_oracle_truth"], marker=MARKERS[p],
                   color=COLORS[p], s=70, edgecolors='black', linewidth=0.5, zorder=3)
        ax.scatter(r["mean_Y"], r["cvar_est"], marker=MARKERS[p],
                   facecolor='none', edgecolors=COLORS[p], s=70, linewidth=1.5, zorder=2)
        ax.plot([r["mean_Y"], r["mean_Y"]],
                [r["full_oracle_truth"], r["cvar_est"]],
                color=COLORS[p], linewidth=0.5, alpha=0.5)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"mean oracle score $\overline{Y}_{\pi'}$", fontsize=10)
    ax.set_ylabel(rf"$\mathrm{{CVaR}}_{{{alpha}}}$", fontsize=10)
    ax.grid(alpha=0.3)

    truth_handle = ml.Line2D([], [], marker='o', color='gray', markerfacecolor='gray',
                              linestyle='', markersize=7, label='full-oracle truth')
    direct_handle = ml.Line2D([], [], marker='o', color='gray', markerfacecolor='none',
                               markeredgecolor='gray', linestyle='', markersize=7,
                               label='Direct estimate')
    policy_handles = [ml.Line2D([], [], marker=MARKERS[p], color=COLORS[p], linestyle='',
                                  markersize=7, label=LABELS[p]) for p in POLICIES]
    leg1 = ax.legend(handles=[truth_handle, direct_handle], loc='lower right',
                      fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=policy_handles, loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_title(rf"Mean vs $\mathrm{{CVaR}}_{{{alpha}}}$ across policies", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = compute(alpha=args.alpha, coverage=args.coverage, seed=args.seed)
    json_path = write_json("mean_vs_cvar.json", payload)
    pdf_path = fig_path("mean_vs_cvar.pdf")
    plot(payload, pdf_path)
    print(f"[mean_vs_cvar] wrote {json_path}")
    print(f"[mean_vs_cvar] wrote {pdf_path}")


if __name__ == "__main__":
    main()
