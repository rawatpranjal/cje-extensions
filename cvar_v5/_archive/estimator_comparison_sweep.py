"""
Estimator comparison sweep — Direct CVaR-CJE vs plug-in variants.

Background
----------
The Direct saddle-point estimator joint-calibrates the conditional shortfall
`E[(t-Y)_+ | s]`. Two classical alternatives are tracked in TODO.md as
deferred MVP:

    plugin_quantile : empirical α-quantile + tail mean over m̂(s_eval)
    plugin_ru_dual  : Rockafellar-Uryasev dual on m̂(s_eval), where
                      m̂(s) = 1 - ĥ_1(s)  (PAV reflection)

Both plug-ins reduce the calibrator to its mean reward `m̂` and lose the
conditional-shortfall information that Direct captures. By Jensen,
plug-in RU systematically underestimates conditional shortfall.

This sweep settles the comparison empirically on the parametric Beta panel.

Decision
--------
Per `cvar_v5/.../work-in-v4-reactive-peacock.md`:

    If for any plug-in p:
        rmse(p) / rmse(direct) < 0.95 on >= 2 of 4 policies, at α=0.10,
        AND var_calib(p) / var_calib(direct) < 0.95 on those policies:
            → flag p as a candidate; update TODO.md::[plugin-*] with
              comparison evidence; propose a follow-up to bring it into
              paper-canonical scope.
    Else:
            → close [plugin-quantile] and [plugin-ru-dual] in TODO.md as
              "comparison done; no headroom over Direct on the parametric
              panel". Preserve archived implementations for reference.

Outputs
-------
mc/runs/<ts>_estimator_comparison/:
    results.csv       per-rep raw rows
    summary.csv       (estimator, policy, alpha) → bias, rmse, var_real,
                                                   var_calib, ratios vs Direct
    figures/
        bias_comparison.png       grouped bar, |bias| per (policy, α)
        rmse_comparison.png       grouped bar, RMSE per (policy, α)
        var_calib_comparison.png  grouped bar, var_calib per (policy, α)
    report.md         headline numbers + decision verdict

CLI:
    python -m cvar_v5._archive.estimator_comparison_sweep -w 4
"""

from __future__ import annotations

import argparse
import logging
import time
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from cvar_v5._archive.plugin_estimators import (
    plugin_quantile_cvar,
    plugin_ru_dual_cvar,
)
from cvar_v5.cvar_cje._crossfit import partition_oracle
from cvar_v5.cvar_cje.calibrator import fit_calibrator_grid
from cvar_v5.cvar_cje.estimator import estimate_direct_cvar
from cvar_v5.cvar_cje.schema import Slice
from cvar_v5.mc import _io
from cvar_v5.mc.dgp import DEFAULT_POLICIES, DGP


LOG = logging.getLogger("cvar_v5.mc")


# Sweep parameters
_N_ORACLE = 600
_N_EVAL = 1000
_R = 20
_ALPHAS = (0.05, 0.10, 0.20)
_K = 5
_T_GRID = np.linspace(0.0, 1.0, 61)
_HEADLINE_ALPHA = 0.10


# (label, callable). Direct first so plug-in ratios are "vs Direct".
ESTIMATORS: dict[str, callable] = {
    "direct_saddle":   estimate_direct_cvar,
    "plugin_quantile": plugin_quantile_cvar,
    "plugin_ru_dual":  plugin_ru_dual_cvar,
}


def _run_cell(args: tuple) -> list[dict]:
    """
    One (policy, rep) cell. For each rep:
      - draws oracle + REAL eval slice
      - fits one CalibratorGrid (shared across estimators)
      - evaluates all 3 estimators on (real_eval, cg)        → est_real
      - evaluates all 3 estimators on (eval_frozen, cg)      → est_calib_only

    Returns one row per (estimator, alpha).
    """
    policy, rep, eval_frozen_df = args
    dgp = DGP(DEFAULT_POLICIES)
    seed = 9001 * rep + 17

    oracle = dgp.sample(policy, n=_N_ORACLE, with_oracle=True, seed=seed)
    eval_real_df = dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=seed + 1)

    calib_slice, _, folds = partition_oracle(oracle, K=_K, seed=seed)
    cg = fit_calibrator_grid(
        calib_slice.s(), calib_slice.y(), _T_GRID,
        fold_id=folds.fold_id, K=_K,
    )

    eval_real = Slice(df=eval_real_df, role="eval")
    eval_frozen = Slice(df=eval_frozen_df, role="eval")

    rows: list[dict] = []
    for est_name, est_fn in ESTIMATORS.items():
        for alpha in _ALPHAS:
            est_real = float(est_fn(eval_real, cg, alpha).value)
            est_calib_only = float(est_fn(eval_frozen, cg, alpha).value)
            truth = dgp.truth_cvar(policy, alpha)
            rows.append({
                "estimator": est_name,
                "policy": policy,
                "alpha": alpha,
                "rep": rep,
                "est_real": est_real,
                "est_calib_only": est_calib_only,
                "truth": float(truth),
            })
    return rows


def run_sweep(n_workers: int = 1, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = _io.make_run_dir("estimator_comparison")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    _io.setup_logging(run_dir=out_dir)

    LOG.info("[ARCHIVED] estimator-comparison sweep")
    LOG.info(
        "policies=%s, alphas=%s, R=%d, n_oracle=%d, n_eval=%d, K=%d, "
        "estimators=%s, workers=%d",
        list(DEFAULT_POLICIES.keys()), list(_ALPHAS), _R, _N_ORACLE, _N_EVAL,
        _K, list(ESTIMATORS.keys()), n_workers,
    )

    # One frozen eval slice per policy (shared across reps for var_calib).
    dgp = DGP(DEFAULT_POLICIES)
    eval_frozen_by_policy: dict[str, pl.DataFrame] = {
        policy: dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=88888)
        for policy in DEFAULT_POLICIES
    }

    cells = [
        (policy, rep, eval_frozen_by_policy[policy])
        for policy, rep in product(DEFAULT_POLICIES.keys(), range(_R))
    ]

    t0 = time.time()
    rows: list[dict] = []
    if n_workers <= 1:
        for i, c in enumerate(cells):
            rows.extend(_run_cell(c))
            if (i + 1) % max(1, len(cells) // 10) == 0 or i == len(cells) - 1:
                LOG.info("  %d/%d cells (%5.1f%%, %.1fs)",
                         i + 1, len(cells), 100*(i+1)/len(cells), time.time()-t0)
    else:
        ctx = get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for i, cell_rows in enumerate(pool.imap_unordered(_run_cell, cells, chunksize=1)):
                rows.extend(cell_rows)
                if (i + 1) % max(1, len(cells) // 10) == 0 or i == len(cells) - 1:
                    LOG.info("  %d/%d cells (%5.1f%%, %.1fs)",
                             i + 1, len(cells), 100*(i+1)/len(cells), time.time()-t0)

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "results.csv")

    summary = _aggregate(df)
    summary.write_csv(out_dir / "summary.csv")

    _make_figures(summary, out_dir)
    _write_report(summary, out_dir)
    LOG.info("done in %.1fs → %s", time.time() - t0, out_dir)
    return out_dir


def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Per (estimator, policy, alpha), reduce reps to:
        bias       = mean(est_real) - truth
        rmse       = sqrt(mean( (est_real - truth)^2 ))
        var_real   = var(est_real)
        var_calib  = var(est_calib_only)

    Then add ratios vs the Direct estimator within each (policy, alpha).
    """
    summary = (
        df.group_by(["estimator", "policy", "alpha"])
        .agg([
            (pl.col("est_real") - pl.col("truth")).mean().alias("bias"),
            ((pl.col("est_real") - pl.col("truth")) ** 2).mean().sqrt().alias("rmse"),
            pl.col("est_real").var().alias("var_real"),
            pl.col("est_calib_only").var().alias("var_calib"),
            pl.col("truth").first().alias("truth"),
            pl.len().alias("R"),
        ])
        .sort(["policy", "alpha", "estimator"])
    )

    direct = (
        summary.filter(pl.col("estimator") == "direct_saddle")
        .select([
            "policy", "alpha",
            pl.col("rmse").alias("rmse_direct"),
            pl.col("var_calib").alias("var_calib_direct"),
        ])
    )
    summary = (
        summary.join(direct, on=["policy", "alpha"], how="left")
        .with_columns([
            (pl.col("rmse") / pl.col("rmse_direct")).alias("rmse_ratio"),
            (pl.col("var_calib") / pl.col("var_calib_direct")).alias("var_calib_ratio"),
        ])
        .drop(["rmse_direct", "var_calib_direct"])
    )
    return summary


def _make_figures(summary: pl.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimators = list(ESTIMATORS.keys())
    policies = list(DEFAULT_POLICIES.keys())
    cells = [(p, a) for p in policies for a in _ALPHAS]
    cell_labels = [f"{p}\nα={a}" for (p, a) in cells]
    x = np.arange(len(cells))
    width = 0.27

    for metric, ylabel, title in [
        ("bias",       "|bias|",     "absolute bias by estimator"),
        ("rmse",       "RMSE",       "RMSE by estimator"),
        ("var_calib",  "var_calib",  "calibrator-side variance by estimator"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 5))
        for j, est in enumerate(estimators):
            sub = summary.filter(pl.col("estimator") == est).sort(["policy", "alpha"])
            vals = sub[metric].to_numpy()
            if metric == "bias":
                vals = np.abs(vals)
            ax.bar(x + (j - 1) * width, vals, width, label=est)
        ax.set_xticks(x)
        ax.set_xticklabels(cell_labels, rotation=0, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title + " (lower is better)")
        ax.legend()
        if metric != "bias":
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / f"{metric}_comparison.png", dpi=120)
        plt.close(fig)


def _write_report(summary: pl.DataFrame, out_dir: Path) -> None:
    lines = [
        "# estimator_comparison sweep — report",
        "",
        f"Run dir: `{out_dir}`",
        "",
        f"Setting: n_oracle={_N_ORACLE}, n_eval={_N_EVAL}, R={_R}, "
        f"K={_K}, alphas={list(_ALPHAS)}, all 4 policies. Same calibrator "
        f"shared across estimators per rep — only the reduction differs.",
        "",
        "## Per-cell numbers",
        "",
        "| policy | α | estimator | bias | RMSE | rmse_ratio | var_calib | var_calib_ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['policy']} | {row['alpha']:.2f} | {row['estimator']} | "
            f"{row['bias']:+.5f} | {row['rmse']:.5f} | {row['rmse_ratio']:.3f} | "
            f"{row['var_calib']:.2e} | {row['var_calib_ratio']:.3f} |"
        )

    # Decision rule at headline α
    lines.extend([
        "",
        f"## Decision (at α = {_HEADLINE_ALPHA})",
        "",
        "Per the plan: a plug-in is flagged as a candidate iff",
        "`rmse_ratio < 0.95` AND `var_calib_ratio < 0.95` on >= 2 of 4 policies.",
        "",
        "| estimator | policies with both ratios < 0.95 |",
        "|---|---|",
    ])
    headline = summary.filter(pl.col("alpha") == _HEADLINE_ALPHA)
    for est in ESTIMATORS:
        if est == "direct_saddle":
            continue
        sub = headline.filter(pl.col("estimator") == est)
        winning = []
        for row in sub.iter_rows(named=True):
            if row["rmse_ratio"] < 0.95 and row["var_calib_ratio"] < 0.95:
                winning.append(row["policy"])
        verdict = ", ".join(winning) if winning else "(none)"
        lines.append(f"| {est} | {verdict} ({len(winning)}/4) |")

    flagged = []
    for est in ESTIMATORS:
        if est == "direct_saddle":
            continue
        sub = headline.filter(pl.col("estimator") == est)
        n_winning = sum(
            1 for row in sub.iter_rows(named=True)
            if row["rmse_ratio"] < 0.95 and row["var_calib_ratio"] < 0.95
        )
        if n_winning >= 2:
            flagged.append(est)

    lines.extend([
        "",
        "**Verdict**: " + (
            f"{', '.join(flagged)} flagged as candidate(s) — update "
            f"TODO.md::[plugin-*] with comparison evidence and propose "
            f"follow-up to bring into paper-canonical scope."
            if flagged
            else "Direct saddle-point dominates. Close `[plugin-quantile]` "
                 "and `[plugin-ru-dual]` in TODO.md as 'comparison done; no "
                 "headroom over Direct on the parametric panel'. Preserve "
                 "archived implementations for reference."
        ),
    ])

    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="cvar_v5 estimator-comparison sweep")
    parser.add_argument("-w", "--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_sweep(n_workers=args.n_workers, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
