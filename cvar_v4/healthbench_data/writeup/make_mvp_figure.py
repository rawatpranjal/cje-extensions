"""Two-panel MVP summary built from the real n=N pilot data.

Reads writeup/data/pilot_table_alpha_0.1_cov_0.25_seed_42.jsonl and produces
mvp_one_graph.{png,pdf}. Left panel = Mean estimand; right panel = CVaR
estimand. Each panel shows cheap-only baseline → Direct CJE → full-oracle
truth, colored by that panel's own transport audit verdict.

Usage:
    python -m cvar_v4.healthbench_data.writeup.make_mvp_figure
or:
    cd cvar_v4/healthbench_data/writeup && python make_mvp_figure.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


WRITEUP_DIR = Path(__file__).resolve().parent
DATA_PATH = WRITEUP_DIR / "data" / "pilot_table_alpha_0.1_cov_0.25_seed_42.jsonl"


# Display-name override; the JSONL stores "parallel_universe_prompt"
LABELS = {
    "base": "base",
    "clone": "clone",
    "premium": "premium",
    "parallel_universe_prompt": "parallel",
    "unhelpful": "unhelpful",
}


def load_rows() -> list[dict]:
    out = []
    for line in DATA_PATH.open():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({
            "policy": LABELS.get(r["policy"], r["policy"]),
            # Mean panel
            "mean_cheap": r["cheap_only_mean"],
            "mean_direct": r["mean_cje_est"],
            "mean_truth": r["mean_Y"],
            "mean_gate": "pass" if r["mean_verdict"] == "PASS" else "flag",
            "mean_p": r["mean_audit_p"],
            "mean_resid": r["mean_audit_residual"],
            "mean_ci_lo": r.get("mean_ci_lo"),
            "mean_ci_hi": r.get("mean_ci_hi"),
            # CVaR panel
            "cvar_cheap": r["cheap_only_cvar"],
            "cvar_direct": r["cvar_est"],
            "cvar_truth": r["full_oracle_truth"],
            "cvar_gate": "pass" if r["verdict"] == "PASS" else "flag",
            "cvar_ci_lo": r.get("cvar_ci_lo"),
            "cvar_ci_hi": r.get("cvar_ci_hi"),
            "g1": r["mean_g1"],
            "audit_n": r["n_slice"],
            "n_total": r.get("n_total", 0),
        })
    # Sort by mean truth descending (best on top — same ordering as before)
    out.sort(key=lambda x: -x["mean_truth"])
    return out


def marker_size(audit_n: int) -> float:
    return 70 + 2.5 * audit_n


PASS_COLOR = "#1f78b4"
FLAG_COLOR = "#d95f02"
CHEAP_COLOR = "#b8c3d6"
TRUTH_COLOR = "black"


def draw_panel(ax, rows, y_positions, *, kind: str) -> None:
    """kind: 'mean' or 'cvar'. Pulls the right fields from each row."""
    for row, y in zip(rows, y_positions):
        cheap = row[f"{kind}_cheap"]
        direct = row[f"{kind}_direct"]
        truth = row[f"{kind}_truth"]
        gate = row[f"{kind}_gate"]
        ci_lo = row.get(f"{kind}_ci_lo")
        ci_hi = row.get(f"{kind}_ci_hi")
        gate_color = PASS_COLOR if gate == "pass" else FLAG_COLOR

        # Backbone line connecting cheap → direct → truth
        ax.plot([cheap, direct, truth], [y, y, y],
                color="#c8c8c8", lw=2.0, zorder=1)
        # Highlight cheap → direct movement
        ax.plot([cheap, direct], [y, y],
                color="#7a9cc6", lw=4.0, alpha=0.45, zorder=2)

        # 95% bootstrap CI whiskers around the Direct estimate, drawn behind
        # the marker so the colored circle stays visually dominant.
        if ci_lo is not None and ci_hi is not None:
            ax.plot([ci_lo, ci_hi], [y, y],
                    color=gate_color, lw=1.4, alpha=0.55, zorder=3)
            # End caps
            cap_h = 0.13
            for x in (ci_lo, ci_hi):
                ax.plot([x, x], [y - cap_h, y + cap_h],
                        color=gate_color, lw=1.4, alpha=0.55, zorder=3)

        # Cheap-only (pale triangle)
        ax.scatter(cheap, y, s=120, marker="^",
                   facecolor=CHEAP_COLOR, edgecolor="#75849a",
                   linewidth=0.8, zorder=5)
        # Direct (colored circle, size scales with audit slice for CVaR;
        # mean panel uses fixed size since the audit is a t-test on residuals,
        # not a tail count)
        size = marker_size(row["audit_n"]) if kind == "cvar" else 150
        ax.scatter(direct, y, s=size, marker="o",
                   facecolor=gate_color, edgecolor="white",
                   linewidth=1.5, zorder=6)
        # Truth (black diamond)
        ax.scatter(truth, y, s=115, marker="D",
                   facecolor=TRUTH_COLOR, edgecolor="white",
                   linewidth=0.8, zorder=7)

        # Right-edge audit-status label
        if kind == "cvar":
            label = (f"PASS  n={row['audit_n']}" if gate == "pass"
                     else f"FLAG  g1={row['g1']:+.2f}")
        else:
            label = (f"PASS  p={row['mean_p']:.2f}" if gate == "pass"
                     else f"FLAG  p={row['mean_p']:.3f}")
        ax.text(0.985, y, label, va="center", ha="right",
                fontsize=9.5, color=gate_color, fontweight="bold",
                transform=ax.get_yaxis_transform())


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No rows in {DATA_PATH}")

    fig, (ax_m, ax_c) = plt.subplots(
        1, 2, figsize=(12.0, 5.4),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.05},
        sharey=True,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.85, bottom=0.20)
    fig.patch.set_facecolor("white")
    for ax in (ax_m, ax_c):
        ax.set_facecolor("#fbfbf8")

    y_positions = list(range(len(rows)))[::-1]

    # Draw both panels
    draw_panel(ax_m, rows, y_positions, kind="mean")
    draw_panel(ax_c, rows, y_positions, kind="cvar")

    # Per-panel x-limits — auto-fit with margin so the right-edge labels fit
    for ax, kind in ((ax_m, "mean"), (ax_c, "cvar")):
        xs = []
        for r in rows:
            xs += [r[f"{kind}_cheap"], r[f"{kind}_direct"], r[f"{kind}_truth"]]
        xmin = min(xs) - 0.10
        xmax = max(xs) + 0.30  # extra room for the right-edge label
        ax.set_xlim(xmin, xmax)
        ax.axvline(0, color="#d0d0d0", lw=1.0, ls="--", zorder=0)
        ax.grid(axis="x", color="#e2e2e2", lw=0.8)
        ax.tick_params(axis="x", labelsize=10.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")

    # Y-axis: only the left panel shows policy labels (sharey)
    ax_m.set_ylim(-0.9, len(rows) - 0.25)
    ax_m.set_yticks(y_positions)
    ax_m.set_yticklabels([r["policy"] for r in rows], fontsize=12)
    ax_m.tick_params(axis="y", length=0)

    # Per-panel titles + shared x-axis label
    ax_m.set_title("Mean estimand: $\\mathbb{E}[Y]$", fontsize=12.5,
                   weight="bold", pad=10, color="#333333")
    ax_c.set_title("CVaR estimand: $\\mathrm{CVaR}_{0.10}(Y)$",
                   fontsize=12.5, weight="bold", pad=10, color="#333333")
    fig.text(0.5, 0.10,
             "Score value: higher is better; left side is worse",
             ha="center", va="center", fontsize=11.5, color="#444444")

    # Suptitle
    n_total = rows[0].get("n_total", 0) if rows else 0
    fig.suptitle(
        "Mean and CVaR transport audits gate two different estimands "
        f"(n={n_total} HealthBench, single seed)",
        fontsize=13.5, weight="bold", y=0.96,
    )

    # Footer
    fig.text(
        0.99, 0.01,
        f"alpha = 0.10 (CVaR panel). Mean audit is t-test on residuals; "
        "CVaR audit is two-moment heuristic.",
        ha="right", va="bottom", fontsize=8.5, color="#666666",
    )

    # Shared legend at bottom
    legend_items = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=CHEAP_COLOR,
               markeredgecolor="#75849a", markersize=9,
               label="cheap-only baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PASS_COLOR,
               markeredgecolor="white", markersize=10,
               label="Direct CJE, audit pass"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FLAG_COLOR,
               markeredgecolor="white", markersize=10,
               label="Direct CJE, audit flag"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black",
               markeredgecolor="white", markersize=8,
               label="full-oracle truth"),
    ]
    fig.legend(
        handles=legend_items, loc="lower center",
        bbox_to_anchor=(0.5, 0.02), ncols=4, frameon=True,
        facecolor="white", edgecolor="#d6d6d6", fontsize=10,
    )

    for ext in ("png", "pdf"):
        fig.savefig(WRITEUP_DIR / f"mvp_one_graph.{ext}", dpi=220)
    print(f"wrote {WRITEUP_DIR / 'mvp_one_graph.pdf'}")


if __name__ == "__main__":
    main()
