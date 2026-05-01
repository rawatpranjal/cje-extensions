"""Oracle-budget curve: |err| vs oracle coverage at fixed alpha.

Re-runs the Direct CVaR-CJE estimator at multiple coverage levels and plots
how absolute error against full-oracle truth changes with the oracle budget.

Outputs:
    writeup/data/budget_curve.json
    writeup/budget_curve.pdf

Usage:
    python -m cvar_v4.healthbench_data.analyses.budget_curve [--alpha 0.10] [--seed 42]
"""
from __future__ import annotations

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..analyze import step5_oracle_calibrated_uniform
from ._common import (COLORS, LABELS, MARKERS, POLICIES, fig_path,
                       panel_size, write_json)


DEFAULT_COVERAGES = (0.10, 0.25, 0.50, 1.00)


def compute(alpha: float = 0.10, seed: int = 42, coverages=DEFAULT_COVERAGES) -> dict:
    """Run the estimator at each coverage level. Returns a dict of results
    keyed by (policy, coverage) suitable for plotting."""
    rows_by_cov: dict[float, list[dict]] = {}
    for cov in coverages:
        rows = step5_oracle_calibrated_uniform(
            coverage=cov, alpha=alpha, seed=seed, verbose=False,
        )
        rows_by_cov[cov] = rows
    return {
        "alpha": alpha,
        "seed": seed,
        "coverages": list(coverages),
        "panel_size": panel_size(),
        "rows": [
            {**r, "coverage_target": cov}
            for cov, rs in rows_by_cov.items() for r in rs
        ],
    }


def plot(payload: dict, out_path):
    coverages = payload["coverages"]
    rows = payload["rows"]
    alpha = payload["alpha"]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for p in POLICIES:
        xs, ys = [], []
        for cov in coverages:
            cell = next((r for r in rows
                         if r["policy"] == p and abs(r["coverage_target"] - cov) < 1e-6),
                        None)
            if cell is None:
                continue
            xs.append(cov)
            ys.append(cell["abs_error"])
        if xs:
            ax.plot(xs, ys, marker=MARKERS[p], color=COLORS[p], label=LABELS[p],
                    linewidth=1.5, markersize=6)
    ax.set_xlabel("oracle coverage", fontsize=10)
    ax.set_ylabel(r"$|\widehat{V} - \mathrm{CVaR}_{\alpha}|$", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(coverages)
    ax.set_xticklabels([f"{c:.2f}" for c in coverages])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title(rf"Direct CVaR-CJE error vs oracle coverage  ($\alpha = {alpha}$)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = compute(alpha=args.alpha, seed=args.seed)
    json_path = write_json("budget_curve.json", payload)
    pdf_path = fig_path("budget_curve.pdf")
    plot(payload, pdf_path)
    print(f"[budget_curve] wrote {json_path}")
    print(f"[budget_curve] wrote {pdf_path}")


if __name__ == "__main__":
    main()
