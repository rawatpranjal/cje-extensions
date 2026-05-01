"""Create a mock one-figure MVP summary for audit-gated CVaR-CJE.

The values are illustrative and pilot-like. This is a visual design sketch, not
an analysis artifact.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT_DIR = Path(__file__).resolve().parent


ROWS = [
    {
        "policy": "premium",
        "mean": 0.44,
        "cheap": -0.02,
        "direct": -0.05,
        "truth": -0.12,
        "gate": "pass",
        "audit_n": 26,
        "g1": 0.02,
        "g2": -0.01,
    },
    {
        "policy": "base",
        "mean": 0.33,
        "cheap": -0.10,
        "direct": -0.14,
        "truth": -0.18,
        "gate": "flag",
        "audit_n": 5,
        "g1": -0.10,
        "g2": 0.00,
    },
    {
        "policy": "clone",
        "mean": 0.32,
        "cheap": -0.10,
        "direct": -0.17,
        "truth": -0.27,
        "gate": "pass",
        "audit_n": 26,
        "g1": -0.02,
        "g2": 0.01,
    },
    {
        "policy": "parallel",
        "mean": 0.13,
        "cheap": -0.17,
        "direct": -0.31,
        "truth": -0.32,
        "gate": "flag",
        "audit_n": 26,
        "g1": 0.13,
        "g2": 0.01,
    },
    {
        "policy": "unhelpful",
        "mean": -0.11,
        "cheap": -0.38,
        "direct": -0.52,
        "truth": -0.50,
        "gate": "flag",
        "audit_n": 26,
        "g1": -0.06,
        "g2": 0.02,
    },
]


def marker_size(audit_n: int) -> float:
    return 70 + 4.0 * audit_n


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbf8")

    y_positions = list(range(len(ROWS)))[::-1]
    pass_color = "#1f78b4"
    flag_color = "#d95f02"
    cheap_color = "#b8c3d6"
    mean_color = "#8f8f8f"

    for row, y in zip(ROWS, y_positions):
        gate_color = pass_color if row["gate"] == "pass" else flag_color

        xs = [row["cheap"], row["direct"], row["truth"]]
        ax.plot(
            xs,
            [y, y, y],
            color="#c8c8c8",
            lw=2.0,
            zorder=1,
        )
        ax.plot(
            [row["cheap"], row["direct"]],
            [y, y],
            color="#7a9cc6",
            lw=4.0,
            alpha=0.45,
            zorder=2,
        )

        ax.scatter(
            row["mean"],
            y + 0.18,
            s=70,
            marker="o",
            facecolor=mean_color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(
            row["cheap"],
            y,
            s=120,
            marker="^",
            facecolor=cheap_color,
            edgecolor="#75849a",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            row["direct"],
            y,
            s=marker_size(row["audit_n"]),
            marker="o",
            facecolor=gate_color,
            edgecolor="white",
            linewidth=1.5,
            zorder=6,
        )
        ax.scatter(
            row["truth"],
            y,
            s=115,
            marker="D",
            facecolor="black",
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
        )

        label = (
            f"PASS  n_audit={row['audit_n']}"
            if row["gate"] == "pass"
            else f"FLAG  g1={row['g1']:+.2f}"
        )
        ax.text(
            0.53,
            y,
            label,
            va="center",
            ha="left",
            fontsize=10.5,
            color=gate_color,
            fontweight="bold",
        )

    ax.annotate(
        "Mean says base and clone are tied\n(gray dots nearly overlap)",
        xy=(0.325, y_positions[1] + 0.18),
        xytext=(0.05, y_positions[0] + 0.55),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        fontsize=10.5,
        color="#333333",
        ha="left",
        va="center",
    )
    ax.annotate(
        "Tail separates them:\ntruth differs by about 0.09",
        xy=(-0.27, y_positions[2]),
        xytext=(-0.57, y_positions[1] - 0.35),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        fontsize=10.5,
        color="#333333",
        ha="left",
        va="center",
    )
    ax.annotate(
        "Calibration pulls cheap-only tail\nscores toward oracle truth",
        xy=(-0.305, y_positions[3]),
        xytext=(-0.56, y_positions[3] + 0.65),
        arrowprops=dict(arrowstyle="->", color="#4b6f9e", lw=1.2),
        fontsize=10.5,
        color="#2d496c",
        ha="left",
        va="center",
    )
    ax.annotate(
        "Flagged Direct points are shown,\nbut not claimable as levels",
        xy=(-0.52, y_positions[4]),
        xytext=(-0.33, y_positions[4] - 0.55),
        arrowprops=dict(arrowstyle="->", color=flag_color, lw=1.2),
        fontsize=10.5,
        color=flag_color,
        ha="left",
        va="center",
    )

    ax.text(
        -0.63,
        y_positions[0] + 0.6,
        "MVP read:\n1. Mean hides deployment tail risk\n2. Cheap judge is lenient in the tail\n3. Direct CVaR-CJE improves levels\n4. Audit decides what can be claimed",
        fontsize=10.3,
        ha="left",
        va="top",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#d6d6d6"),
    )

    ax.axvline(0, color="#d0d0d0", lw=1.0, ls="--", zorder=0)
    ax.grid(axis="x", color="#e2e2e2", lw=0.8)
    ax.set_xlim(-0.65, 0.69)
    ax.set_ylim(-0.9, len(ROWS) - 0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["policy"] for row in ROWS], fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("Score value: higher is better; left side is worse lower-tail performance", fontsize=12)
    ax.set_title(
        "Mean Rankings Hide Tail Risk; Audit-Gated CVaR-CJE Recovers and Refuses",
        fontsize=14,
        weight="bold",
        pad=14,
    )
    ax.text(
        0.99,
        0.01,
        "Mock data for figure design only. alpha = 0.10.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#666666",
    )

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=mean_color, markeredgecolor="white", markersize=8, label="mean oracle score"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=cheap_color, markeredgecolor="#75849a", markersize=9, label="cheap-only CVaR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=pass_color, markeredgecolor="white", markersize=10, label="Direct CVaR-CJE, audit pass"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=flag_color, markeredgecolor="white", markersize=10, label="Direct CVaR-CJE, audit flag"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black", markeredgecolor="white", markersize=8, label="full-oracle truth"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=5,
        frameon=True,
        facecolor="white",
        edgecolor="#d6d6d6",
        fontsize=9.5,
    )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"mvp_one_graph_mock.{ext}", dpi=220)


if __name__ == "__main__":
    main()
