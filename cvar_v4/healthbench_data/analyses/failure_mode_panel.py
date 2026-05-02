"""Failure-mode panel: how the audit's behavior varies with sample size.

Runs the headline α=0.10 + α=0.20 audit at three subsample sizes drawn
from the same n=500 panel — small (n=100), medium (n=250), large (n=500)
— and emits a small-multiples figure showing per-cell audit verdicts and
|ḡ_1| against the heuristic 0.05 threshold.

Why HEURISTIC_THR = 0.05:
    The audit verdict heuristic |ḡ_1| ≤ 0.05 is calibrated to the
    α-tolerance of typical CVaR claims. A 5-percentage-point deviation
    in tail-mass at the threshold corresponds to roughly half of α=0.10
    or a quarter of α=0.20 — small enough that the resulting CVaR bias
    is bounded by a few percentage points of Y. This is NOT a hypothesis
    test (the formal test is `two_moment_wald_audit_xf`); it's a fast
    triage threshold. Cells that flag here can be re-checked formally.

The non-monotone-in-n story:
    The heuristic has per-row sensitivity 1/n_audit. At small n_audit a
    SINGLE row crossing t̂ moves |ḡ_1| past 0.05 even when the underlying
    tail-mass is exactly α; at large n_audit only structural transport
    failures cross 0.05. Net result: a cell can flag at n=100, relax to
    PASS at n=250, and only flag again at n=500 if the failure is real
    (not heuristic noise). This panel makes that pattern legible by
    putting the same five policies × two alphas at three sample sizes
    side-by-side.

Outputs:
    writeup/data/failure_mode_panel.json
    writeup/failure_mode_panel.{pdf,png}

Usage:
    python -m cvar_v4.healthbench_data.analyses.failure_mode_panel
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analyze import step5_oracle_calibrated_uniform
from ._common import WRITEUP_DATA_DIR, WRITEUP_FIG_DIR


SIZES = [
    ("small",  100),
    ("medium", 250),
    ("large",  500),  # full panel
]
ALPHAS = (0.10, 0.20)
HEURISTIC_THR = 0.05

POLICY_DISPLAY = {
    "base": "base",
    "clone": "clone",
    "premium": "premium",
    "parallel_universe_prompt": "parallel",
    "unhelpful": "unhelpful",
    "risky": "risky",
}


def compute(coverage: float = 0.25, seed: int = 42) -> dict:
    """For each (size, α) pair, run step5 and collect per-policy audit."""
    out = {"sizes": [], "alphas": list(ALPHAS), "rows": []}
    for label, n in SIZES:
        n_max = None if n >= 500 else n
        for alpha in ALPHAS:
            rows = step5_oracle_calibrated_uniform(
                coverage=coverage, alpha=alpha, seed=seed,
                verbose=False, n_max=n_max,
            )
            for r in rows:
                out["rows"].append({
                    "size_label": label, "n_total": r["n_total"],
                    "alpha": alpha,
                    "policy": r["policy"],
                    "n_audit": r["n_slice"],
                    "abs_error": r["abs_error"],
                    "mean_g1": r["mean_g1"],
                    "mean_g2": r["mean_g2"],
                    "verdict": r["verdict"],
                })
        out["sizes"].append({"label": label, "n_total": n})
    return out


def write_json(name: str, payload: dict) -> Path:
    out_path = WRITEUP_DATA_DIR / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def fig_path(name: str) -> Path:
    p = WRITEUP_FIG_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(payload: dict, out_path: Path) -> None:
    """Two-row figure (one per α). x-axis: |ḡ_1|. y-axis: policy.
    Three n-levels per cell, colored by verdict."""
    policies_order = list(POLICY_DISPLAY.keys())
    n_pol = len(policies_order)
    sizes = payload["sizes"]
    n_sizes = len(sizes)

    pass_color = "#1f78b4"
    flag_color = "#d95f02"

    # 2 rows (alphas) × 1 col, sharing y-axis
    fig, axes = plt.subplots(
        len(ALPHAS), 1, figsize=(10.5, 5.0),
        gridspec_kw={"hspace": 0.35},
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    # Row offsets within each policy: small/medium/large stacked vertically
    n_offsets = np.linspace(-0.30, 0.30, n_sizes)

    for ax, alpha in zip(axes, ALPHAS):
        ax.set_facecolor("#fbfbf8")
        # Threshold line
        ax.axvline(HEURISTIC_THR, color="#888888", lw=1.0, ls="--", zorder=0)
        ax.axvline(-HEURISTIC_THR, color="#888888", lw=1.0, ls="--", zorder=0)
        ax.axvspan(-HEURISTIC_THR, HEURISTIC_THR,
                   color="#e8efe8", alpha=0.5, zorder=0)
        ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)

        for pi, pol in enumerate(policies_order):
            for si, sz in enumerate(sizes):
                rows = [r for r in payload["rows"]
                        if r["alpha"] == alpha and r["policy"] == pol
                        and r["size_label"] == sz["label"]]
                if not rows:
                    continue
                r = rows[0]
                color = pass_color if r["verdict"] == "PASS" else flag_color
                y = pi + n_offsets[si]
                ax.scatter(r["mean_g1"], y, s=80, marker="o",
                           facecolor=color, edgecolor="white",
                           linewidth=1.0, zorder=5)
                # tiny n_audit annotation right of marker
                ax.text(r["mean_g1"] + 0.005, y,
                        f"n={r['n_audit']}", fontsize=7.5,
                        va="center", ha="left", color="#555555")

        ax.set_yticks(range(n_pol))
        ax.set_yticklabels([POLICY_DISPLAY[p] for p in policies_order],
                           fontsize=10)
        ax.set_ylim(-0.7, n_pol - 0.3)
        ax.invert_yaxis()
        ax.set_title(rf"$\alpha = {alpha:.2f}$", fontsize=11,
                     weight="bold", loc="left", pad=4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(axis="x", labelsize=9.5)

    axes[-1].set_xlabel(r"audit moment $\bar g_1 = \mathbb{E}[\mathbf{1}\{Y \leq \hat t\}] - \alpha$",
                        fontsize=10.5)

    # Shared legend at bottom
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=pass_color,
               markeredgecolor="white", markersize=9, label="audit PASS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=flag_color,
               markeredgecolor="white", markersize=9, label="audit FLAG"),
        Line2D([0], [0], color="#888888", ls="--", lw=1.0,
               label=r"heuristic $\pm 0.05$ threshold"),
    ]
    # n-levels legend (separate)
    n_legend = "    ".join(
        f"{lbl} = n_total {sz['n_total']}"
        for lbl, sz in zip(["small", "medium", "large"], sizes)
    )
    fig.legend(handles=legend_items, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncols=3,
               frameon=False, fontsize=9)
    fig.text(0.5, -0.02, f"three vertically-stacked points per row: {n_legend}",
             ha="center", va="top", fontsize=8.5, color="#666666",
             transform=fig.transFigure)
    fig.suptitle(
        "Audit behavior across sample sizes: small-n flags relax at large n",
        fontsize=12.5, weight="bold", y=0.99,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = compute(coverage=args.coverage, seed=args.seed)
    write_json("failure_mode_panel.json", payload)
    plot(payload, fig_path("failure_mode_panel.pdf"))
    # Concise console summary
    print(f"[failure_mode_panel] {len(payload['rows'])} rows across "
          f"{len(SIZES)} sizes × {len(ALPHAS)} alphas × 5 policies")
    flagged_by_size = {}
    for r in payload["rows"]:
        if r["verdict"] != "PASS":
            flagged_by_size.setdefault(r["size_label"], []).append(
                f"{r['policy']}@α={r['alpha']:.2f}"
            )
    for lbl, _ in SIZES:
        flags = flagged_by_size.get(lbl, [])
        print(f"  {lbl} (n_total={[s['n_total'] for s in payload['sizes'] if s['label']==lbl][0]}): "
              f"{len(flags)} flagged — {', '.join(flags) if flags else '(none)'}")


if __name__ == "__main__":
    main()
