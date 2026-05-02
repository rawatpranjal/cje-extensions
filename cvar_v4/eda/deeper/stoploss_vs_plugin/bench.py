"""Benchmark: stop-loss calibrator vs. plug-in E[Y|S,X] for CVaR-CJE.

Sweeps σ × α × n_train × seeds. Reports bias and RMSE for three estimators:
  (A) stop-loss:         paper's threshold-indexed shortfall calibrator
  (B1) plug-in quantile: empirical α-tail of m̂(s_eval)
  (B2) plug-in RU dual:  RU dual on m̂

Outputs (written next to this script):
  - sweep_v1.csv  / sweep_v1.md   (default `main()`: 4-σ, 200-seed)
  - sweep_deep.csv / sweep_deep.pdf / sweep_deep.png (`main_deep()`: 11-σ, 500-seed)
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from cvar_v4.eda.deeper._estimator import (
    estimate_direct_cvar_isotonic,
    estimate_plugin_cvar_quantile,
    estimate_plugin_cvar_ru_dual,
)
from cvar_v4.eda.deeper.stoploss_vs_plugin.dgp import sample_panel, true_cvar


SIGMAS = (0.01, 0.05, 0.10, 0.20)
ALPHAS = (0.05, 0.10, 0.25)
NS_TRAIN = (500, 5000)
N_EVAL = 5000
N_SEEDS = 200
N_MC_TRUTH = 5_000_000


def _run_cell(sigma, alpha, n_train, seed):
    rng = np.random.default_rng(seed)
    s_tr, y_tr = sample_panel(n_train, sigma, rng)
    s_ev, _ = sample_panel(N_EVAL, sigma, rng)
    sl, _, _, _ = estimate_direct_cvar_isotonic(s_tr, y_tr, s_ev, alpha)
    pq = estimate_plugin_cvar_quantile(s_tr, y_tr, s_ev, alpha)
    pr, _, _, _ = estimate_plugin_cvar_ru_dual(s_tr, y_tr, s_ev, alpha)
    return sl, pq, pr


def main():
    here = Path(__file__).parent
    csv_path = here / "sweep_v1.csv"
    md_path = here / "sweep_v1.md"

    truths = {}
    print("Computing MC truths...")
    for sigma in SIGMAS:
        for alpha in ALPHAS:
            truths[(sigma, alpha)] = true_cvar(alpha, sigma, n_mc=N_MC_TRUTH, seed=42)
            print(f"  sigma={sigma}, alpha={alpha}: truth={truths[(sigma, alpha)]:.4f}")

    rows = []
    t0 = time.time()
    for sigma in SIGMAS:
        for alpha in ALPHAS:
            for n_train in NS_TRAIN:
                truth = truths[(sigma, alpha)]
                est_sl = np.empty(N_SEEDS)
                est_pq = np.empty(N_SEEDS)
                est_pr = np.empty(N_SEEDS)
                for s in range(N_SEEDS):
                    seed = (
                        SIGMAS.index(sigma) * 10_000
                        + ALPHAS.index(alpha) * 1000
                        + NS_TRAIN.index(n_train) * 100
                        + s
                    )
                    est_sl[s], est_pq[s], est_pr[s] = _run_cell(sigma, alpha, n_train, seed)
                bias_sl = float(est_sl.mean() - truth)
                bias_pq = float(est_pq.mean() - truth)
                bias_pr = float(est_pr.mean() - truth)
                rmse_sl = float(np.sqrt(np.mean((est_sl - truth) ** 2)))
                rmse_pq = float(np.sqrt(np.mean((est_pq - truth) ** 2)))
                rmse_pr = float(np.sqrt(np.mean((est_pr - truth) ** 2)))
                row = {
                    "sigma": sigma,
                    "alpha": alpha,
                    "n_train": n_train,
                    "truth": truth,
                    "bias_stoploss": bias_sl,
                    "bias_plugin_qt": bias_pq,
                    "bias_plugin_ru": bias_pr,
                    "rmse_stoploss": rmse_sl,
                    "rmse_plugin_qt": rmse_pq,
                    "rmse_plugin_ru": rmse_pr,
                }
                rows.append(row)
                elapsed = time.time() - t0
                print(
                    f"  sigma={sigma} alpha={alpha} n={n_train}  "
                    f"truth={truth:.4f}  "
                    f"bias_sl={bias_sl:+.4f}  bias_pq={bias_pq:+.4f}  bias_pr={bias_pr:+.4f}  "
                    f"[{elapsed:.0f}s]"
                )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown(md_path, rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


def _write_markdown(md_path: Path, rows: list[dict]):
    lines = []
    lines.append("# Stop-loss calibrator vs. plug-in E[Y|S,X]: benchmark\n")
    lines.append(
        "DGP: `S ~ U[0,1]`, `Y = clip(0.2 + 0.6*S + N(0, sigma^2), 0, 1)`. "
        f"Truth: MC with `n_mc={N_MC_TRUTH:,}` per (sigma, alpha). "
        f"Estimators averaged over `{N_SEEDS}` seeds with `n_eval={N_EVAL}`.\n"
    )
    lines.append("**Estimators**\n")
    lines.append("- **stop-loss**: paper's threshold-indexed isotonic shortfall calibrator + RU max")
    lines.append("- **plug-in qt**: fit `m̂(s)`, take empirical CVaR_α of `{m̂(s_eval)}`")
    lines.append("- **plug-in RU**: fit `m̂(s)`, then `max_t [t − α⁻¹ · mean((t − m̂)+)]`\n")
    lines.append("## Bias (estimate − truth, signed; positive = tail risk underestimated)\n")
    lines.append("| sigma | alpha | n_train | truth | bias stop-loss | bias plug-in qt | bias plug-in RU |")
    lines.append("|------:|------:|--------:|------:|---------------:|----------------:|----------------:|")
    for r in rows:
        lines.append(
            f"| {r['sigma']} | {r['alpha']} | {r['n_train']} | {r['truth']:.4f} | "
            f"{r['bias_stoploss']:+.4f} | {r['bias_plugin_qt']:+.4f} | {r['bias_plugin_ru']:+.4f} |"
        )
    lines.append("\n## RMSE\n")
    lines.append("| sigma | alpha | n_train | truth | RMSE stop-loss | RMSE plug-in qt | RMSE plug-in RU |")
    lines.append("|------:|------:|--------:|------:|---------------:|----------------:|----------------:|")
    for r in rows:
        lines.append(
            f"| {r['sigma']} | {r['alpha']} | {r['n_train']} | {r['truth']:.4f} | "
            f"{r['rmse_stoploss']:.4f} | {r['rmse_plugin_qt']:.4f} | {r['rmse_plugin_ru']:.4f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines))


SIGMAS_DEEP = tuple(round(0.02 * i, 4) for i in range(11))   # 0.00, 0.02, …, 0.20
ALPHAS_DEEP = (0.05, 0.10, 0.25)
N_TRAIN_DEEP = 5000
N_EVAL_DEEP = 5000
N_SEEDS_DEEP = 500
N_MC_TRUTH_DEEP = 10_000_000


def _run_seeds(sigma: float, alpha: float, n_seeds: int, seed_offset: int):
    est_sl = np.empty(n_seeds)
    est_pr = np.empty(n_seeds)
    for s in range(n_seeds):
        rng = np.random.default_rng(seed_offset + s)
        s_tr, y_tr = sample_panel(N_TRAIN_DEEP, sigma, rng)
        s_ev, _ = sample_panel(N_EVAL_DEEP, sigma, rng)
        sl, _, _, _ = estimate_direct_cvar_isotonic(s_tr, y_tr, s_ev, alpha)
        pr, _, _, _ = estimate_plugin_cvar_ru_dual(s_tr, y_tr, s_ev, alpha)
        est_sl[s] = sl
        est_pr[s] = pr
    return est_sl, est_pr


def main_deep():
    """Finer σ grid + more seeds + single 1×3 figure summary."""
    here = Path(__file__).parent
    csv_path = here / "sweep_deep.csv"
    pdf_path = here / "sweep_deep.pdf"

    print("Computing MC truths (deep)...")
    truths = {}
    for sigma in SIGMAS_DEEP:
        for alpha in ALPHAS_DEEP:
            truths[(sigma, alpha)] = true_cvar(alpha, sigma, n_mc=N_MC_TRUTH_DEEP, seed=42)

    rows = []
    # arrays for the figure: shape (n_alpha, n_sigma)
    mean_sl = np.zeros((len(ALPHAS_DEEP), len(SIGMAS_DEEP)))
    p05_sl = np.zeros_like(mean_sl)
    p95_sl = np.zeros_like(mean_sl)
    mean_pr = np.zeros_like(mean_sl)
    p05_pr = np.zeros_like(mean_sl)
    p95_pr = np.zeros_like(mean_sl)
    truth_arr = np.zeros_like(mean_sl)

    t0 = time.time()
    for ai, alpha in enumerate(ALPHAS_DEEP):
        for si, sigma in enumerate(SIGMAS_DEEP):
            truth = truths[(sigma, alpha)]
            seed_offset = ai * 1_000_000 + si * 1_000
            est_sl, est_pr = _run_seeds(sigma, alpha, N_SEEDS_DEEP, seed_offset)
            mean_sl[ai, si] = est_sl.mean()
            p05_sl[ai, si] = np.quantile(est_sl, 0.05)
            p95_sl[ai, si] = np.quantile(est_sl, 0.95)
            mean_pr[ai, si] = est_pr.mean()
            p05_pr[ai, si] = np.quantile(est_pr, 0.05)
            p95_pr[ai, si] = np.quantile(est_pr, 0.95)
            truth_arr[ai, si] = truth
            rows.append({
                "sigma": sigma,
                "alpha": alpha,
                "n_train": N_TRAIN_DEEP,
                "truth": truth,
                "mean_stoploss": float(est_sl.mean()),
                "p05_stoploss": float(np.quantile(est_sl, 0.05)),
                "p95_stoploss": float(np.quantile(est_sl, 0.95)),
                "bias_stoploss": float(est_sl.mean() - truth),
                "rmse_stoploss": float(np.sqrt(np.mean((est_sl - truth) ** 2))),
                "mean_plugin_ru": float(est_pr.mean()),
                "p05_plugin_ru": float(np.quantile(est_pr, 0.05)),
                "p95_plugin_ru": float(np.quantile(est_pr, 0.95)),
                "bias_plugin_ru": float(est_pr.mean() - truth),
                "rmse_plugin_ru": float(np.sqrt(np.mean((est_pr - truth) ** 2))),
            })
            elapsed = time.time() - t0
            print(
                f"  alpha={alpha} sigma={sigma}  truth={truth:.4f}  "
                f"sl={est_sl.mean():.4f} (bias {est_sl.mean()-truth:+.4f})  "
                f"pr={est_pr.mean():.4f} (bias {est_pr.mean()-truth:+.4f})  "
                f"[{elapsed:.0f}s]"
            )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    _plot_deep(
        np.array(SIGMAS_DEEP), ALPHAS_DEEP,
        truth_arr, mean_sl, p05_sl, p95_sl, mean_pr, p05_pr, p95_pr,
        pdf_path,
    )
    print(f"Wrote {pdf_path}")


def _plot_deep(sigmas, alphas, truth, mean_sl, p05_sl, p95_sl, mean_pr, p05_pr, p95_pr, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(alphas), figsize=(11.0, 3.6), sharex=True)
    if len(alphas) == 1:
        axes = [axes]

    for ai, (alpha, ax) in enumerate(zip(alphas, axes)):
        ax.fill_between(sigmas, p05_pr[ai], p95_pr[ai],
                        color="tab:orange", alpha=0.18, linewidth=0)
        ax.fill_between(sigmas, p05_sl[ai], p95_sl[ai],
                        color="tab:blue", alpha=0.18, linewidth=0)
        ax.plot(sigmas, mean_pr[ai], color="tab:orange", lw=1.6,
                label=r"plug-in $\hat m(S)$")
        ax.plot(sigmas, mean_sl[ai], color="tab:blue", lw=1.6,
                label="stop-loss (paper)")
        ax.plot(sigmas, truth[ai], color="black", lw=2.0, ls="--",
                label="truth")
        ax.set_title(rf"$\alpha = {alpha}$", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if ai == 0:
            ax.set_ylabel(r"CVaR$_\alpha$ estimate", fontsize=10)
        ax.set_xlabel(r"residual SD $\sigma$", fontsize=10)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        "CVaR-CJE under increasing residual noise (n_train=5000, n_seeds=500)",
        fontsize=10, y=1.10,
    )
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    # Also write a PNG for markdown embedding
    png_path = pdf_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    import sys
    if "--deep" in sys.argv:
        main_deep()
    else:
        main()
