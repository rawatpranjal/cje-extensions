"""Tail-zoom CDF figure: cheap-S vs oracle-Y vs CJE-calibrated f̂(S).

Why this diagnostic exists
--------------------------
CVaR is a single number — readers can't see the *shape* of the tail from
a table. A reviewer who is suspicious of the calibration story wants to
SEE that the cheap judge's bottom-20% mass sits to the right of the
oracle's (cheap is too lenient on the tail) and that the CJE-calibrated
prediction f̂(S) lies on top of the oracle ECDF (the calibrator
reconstructs the tail correctly).

Two side-by-side panels, one per stress-test policy:

* `parallel_universe_prompt` — the policy that flagged in the n=100
  pilot. Subtle drift from base.
* `unhelpful` — the worst-tail policy in the panel. Gross drift.

If the calibrator handles both, the reader can extrapolate: it works in
the easy and hard regimes.

Three ECDFs per panel
---------------------
1. Cheap-S ECDF over `s_target` (full target distribution, n=500).
2. Oracle-Y ECDF over `y_target_full` (the truth, n=500).
3. CJE-calibrated ECDF over `f̂(s_target)` where f̂ is the isotonic mean
   fit on the 25% logger oracle slice. This shows what the calibrated
   prediction *looks like* across the whole target — pre-CVaR, pre-saddle.

X-axis is clipped to `[joint_min, 1.05 · q_20(oracle)]` so the figure
zooms on the bottom 20% of mass; y-axis runs 0 → 0.30 to keep the tail
crossing visible without being dominated by the body of the distribution.

Outputs
-------
- `writeup/data/tail_cdf.json`
- `writeup/tail_cdf.pdf`

Usage
-----
    python -m cvar_v4.healthbench_data.analyses.tail_cdf [--alpha 0.10] \
        [--coverage 0.25] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analyze import (_atom_split_quantile, _load_judge_scores, _logger_panel,
                       _policy_panel)
from ..oracle_design import select_slice
from ..policies import logger_policy
from ._common import COLORS, LABELS, fig_path, write_json

# Try to import the isotonic-mean fitter from the estimator module.
# If unavailable, the module will surface the import error at compute time.
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from eda.deeper._estimator import fit_isotonic_mean  # noqa: E402
except Exception:  # pragma: no cover
    fit_isotonic_mean = None  # type: ignore


# Default policies to plot. parallel = subtle drift; unhelpful = gross drift;
# risky = mean-vs-CVaR divergence (good mean, bad tail).
DEFAULT_POLICIES = ("parallel_universe_prompt", "unhelpful", "risky")


def _ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_x, F(x)) for the empirical CDF — step function points."""
    arr = np.asarray(arr)
    n = arr.size
    x = np.sort(arr)
    F = np.arange(1, n + 1) / n
    return x, F


def _fit_calibrator(coverage: float, alpha: float, seed: int):
    """Fit the isotonic mean calibrator on the logger oracle slice with
    the SAME slice/split logic as `step5_oracle_calibrated_uniform`.

    Returns (s_train, y_train, w_train) — the calibration data — or None
    if the panel is too small.
    """
    if fit_isotonic_mean is None:
        raise RuntimeError("fit_isotonic_mean not importable; check eda.deeper._estimator")
    log_panel = _logger_panel()
    if log_panel is None:
        return None
    s_log_full, y_log_full = log_panel
    logger = logger_policy()
    log_pids = sorted(set(_load_judge_scores(logger.name, "cheap"))
                      & set(_load_judge_scores(logger.name, "oracle")))
    log_rows = [
        {"prompt_id": pid, "policy": logger.name, "cheap_score": float(s_log_full[i]),
         "oracle_score": float(y_log_full[i])}
        for i, pid in enumerate(log_pids)
    ]
    log_slice = select_slice(
        log_rows, design="uniform", coverage=coverage, alpha=alpha, seed=seed,
    )
    sel_mask = np.array([s.selected for s in log_slice])
    sel_pi = np.array([s.pi for s in log_slice])
    if sel_mask.sum() < 5:
        return None
    sel_idx = np.where(sel_mask)[0]
    n_sel = int(len(sel_idx))
    # Same 80/20 calibration/audit split RNG seed as step5 — keeps the
    # train set identical so the calibrator we plot matches the one in
    # the pilot table.
    rng_split = np.random.default_rng(seed + 991)
    perm = rng_split.permutation(n_sel)
    n_cal = max(3, int(round(0.8 * n_sel)))
    cal_idx_in_sel = perm[:n_cal]
    cal_global_idx = sel_idx[cal_idx_in_sel]
    s_train = s_log_full[cal_global_idx]
    y_train = y_log_full[cal_global_idx]
    w_train = 1.0 / sel_pi[cal_global_idx]
    return s_train, y_train, w_train


def compute(*, alpha: float = 0.10, coverage: float = 0.25, seed: int = 42,
            policies=DEFAULT_POLICIES) -> dict:
    """For each policy, return the three ECDFs and a few summary numbers."""
    cal = _fit_calibrator(coverage=coverage, alpha=alpha, seed=seed)
    if cal is None:
        return {"alpha": alpha, "coverage": coverage, "seed": seed,
                "policies": [], "error": "logger panel too small"}
    s_train, y_train, w_train = cal

    panels = []
    for p in policies:
        panel = _policy_panel(p)
        if panel is None:
            continue
        _, s_target, y_target_full = panel
        # f̂(s_target): the calibrator's predicted Y for every cheap score
        # in the full target distribution. This is the CJE-calibrated
        # quantity that the saddle-point estimator integrates over.
        y_pred = np.asarray(fit_isotonic_mean(
            s_train, y_train, s_target, sample_weight=w_train,
        ))
        x_cheap, F_cheap = _ecdf(s_target)
        x_oracle, F_oracle = _ecdf(y_target_full)
        x_cje, F_cje = _ecdf(y_pred)
        # Atom-split α-quantile thresholds for the vertical guides.
        t_star = _atom_split_quantile(y_target_full, alpha)   # truth
        t_cheap = _atom_split_quantile(s_target, alpha)       # cheap baseline
        panels.append({
            "policy": p,
            "x_cheap": x_cheap.tolist(), "F_cheap": F_cheap.tolist(),
            "x_oracle": x_oracle.tolist(), "F_oracle": F_oracle.tolist(),
            "x_cje": x_cje.tolist(), "F_cje": F_cje.tolist(),
            "t_star": float(t_star),
            "t_cheap": float(t_cheap),
            "n_target": int(s_target.size),
        })
    return {
        "alpha": float(alpha), "coverage": float(coverage), "seed": int(seed),
        "policies": panels,
    }


def _step_curve(x: list[float], F: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert sorted points into a step-style curve for matplotlib's
    `where='post'` plotter — prepend min-1 with F=0 so the curve starts
    on the floor."""
    x = np.asarray(x); F = np.asarray(F)
    if x.size == 0:
        return x, F
    return np.r_[x[0] - 1e-9, x], np.r_[0.0, F]


def plot(payload: dict, out_path: Path) -> None:
    panels = payload["policies"]
    alpha = payload["alpha"]
    if not panels:
        print("[tail_cdf] nothing to plot")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.6),
                              sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        # Three step ECDFs.
        for key_x, key_F, color, label, lw in [
            ("x_cheap",  "F_cheap",  "#1f77b4", "cheap-S",          1.6),
            ("x_oracle", "F_oracle", "#d62728", "oracle-Y (truth)", 1.8),
            ("x_cje",    "F_cje",    "#2ca02c", r"CJE-calibrated $\widehat{f}(S)$", 1.8),
        ]:
            x, F = _step_curve(panel[key_x], panel[key_F])
            ax.step(x, F, where="post", color=color, label=label, linewidth=lw)
        # Vertical guides at the truth and the cheap-baseline α-quantile.
        ax.axvline(panel["t_star"], color="#d62728", linestyle="--",
                    linewidth=1.0, alpha=0.6,
                    label=rf"$t^*$ = {panel['t_star']:+.2f}")
        ax.axvline(panel["t_cheap"], color="#1f77b4", linestyle=":",
                    linewidth=1.0, alpha=0.6,
                    label=rf"cheap $t$ = {panel['t_cheap']:+.2f}")
        ax.axhline(alpha, color="black", linestyle="-",
                    linewidth=0.8, alpha=0.4)
        # Zoom: x covers [joint_min, 1.05·t* + headroom]; y to 0.30 (well
        # above α=0.10 / α=0.20 so the crossing is visible).
        joint_min = min(min(panel["x_cheap"]), min(panel["x_oracle"]),
                        min(panel["x_cje"]))
        x_max = panel["t_star"] + 0.20  # 20-pt headroom past the truth
        ax.set_xlim(joint_min - 0.02, x_max)
        ax.set_ylim(0.0, 0.30)
        ax.set_xlabel("score", fontsize=10)
        ax.set_title(rf"{LABELS.get(panel['policy'], panel['policy'])}  "
                      rf"($n = {panel['n_target']}$)", fontsize=10)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("ECDF", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(rf"Bottom-20\% tail CDFs at $\alpha = {alpha}$", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = compute(alpha=args.alpha, coverage=args.coverage, seed=args.seed)
    json_path = write_json("tail_cdf.json", payload)
    pdf_path = fig_path("tail_cdf.pdf")
    plot(payload, pdf_path)
    print(f"[tail_cdf] wrote {json_path}")
    print(f"[tail_cdf] wrote {pdf_path}")


if __name__ == "__main__":
    main()
