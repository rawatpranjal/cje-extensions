"""
Audit-size n_audit sweep — closes (or escalates) `[omega-research-n-audit]`.

Background
----------
The omega_sweep run at n_oracle=600 (n_audit ≈ 104) found that NO Ω̂ variant
calibrates to the nominal η = 0.05:

    boot_remax_ridge:    size = 0.000   (zero-power, ridge dominates)
    boot_remax_no_ridge: size = 0.170
    analytical:          size = 0.185
    boot_fixed:          size = 0.190

Hypothesis: at larger n_audit, size calibrates. This sweep tests it.

Setup
-----
For each n_oracle in {1500, 3000, 6000} (giving n_audit ≈ {250, 500, 1000}
under K=5 hash partition):

    For each policy in DEFAULT_POLICIES:
      For each Ω̂ variant in {analytical, boot_remax_ridge,
                              boot_remax_no_ridge, boot_fixed}:
        Run R = 50 reps at α = 0.10, δ = 0 (null DGP).
        size = mean over reps of (audit_decision == REFUSE).

Decision
--------
Look at  size  per (n_audit, Ω̂)  averaged over policies:

  if  any Ω̂ gets size_dev = |size − 0.05| < 0.05 at some n_audit:
      LOCK that Ω̂ as Config.omega_estimator default.
      Close `[omega-research-n-audit]` with the n threshold.

  if  no Ω̂ calibrates even at n_audit = 1000:
      Close `[omega-research-n-audit]` as "doesn't help"; escalate to
      `[omega-research-derivation]` (re-derive Ω̂).

Outputs
-------
mc/runs/<ts>_omega_n_audit/:
    results.csv     per-rep raw rows
    summary.csv     (n_oracle, policy, omega) → size
    figures/
        size_vs_n_audit.png   line plot per Ω̂; horizontal η=0.05 reference
    report.md       headline numbers + decision verdict

CLI:
    python -m cvar_v5.mc.omega_n_audit_sweep -w 4
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

from . import _io
from .dgp import DEFAULT_POLICIES, DGP
from ..cvar_cje._crossfit import partition_oracle
from ..cvar_cje.audit import two_moment_wald_audit
from ..cvar_cje.calibrator import fit_calibrator_grid
from ..cvar_cje.config import OmegaEstimator
from ..cvar_cje.estimator import estimate_direct_cvar
from ..cvar_cje.schema import Slice


LOG = logging.getLogger("cvar_v5.mc")


# Sweep parameters
_N_ORACLE_VALUES = (1500, 3000, 6000)
_N_EVAL = 1000
_AUDIT_B = 500
_R = 50
_ALPHA = 0.10
_K = 5
_OMEGA_VARIANTS: tuple[OmegaEstimator, ...] = (
    "analytical",
    "analytical_oua",
    "boot_remax_ridge",
    "boot_remax_no_ridge",
    "boot_remax_oua",
    "boot_fixed",
)
_NOMINAL_ETA = 0.05


def _run_cell(args: tuple) -> list[dict]:
    """One (n_oracle, policy, rep) cell. Sweeps Ω̂ variants internally."""
    n_oracle, policy, rep = args
    dgp = DGP(DEFAULT_POLICIES)
    seed = 7000 * rep + 13

    oracle = dgp.sample(policy, n=n_oracle, with_oracle=True, seed=seed)
    eval_df = dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=seed + 1)

    calib_slice, audit_slice, folds = partition_oracle(oracle, K=_K, seed=seed)
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(
        calib_slice.s(), calib_slice.y(), t_grid,
        fold_id=folds.fold_id, K=_K,
    )
    estimate = estimate_direct_cvar(
        Slice(df=eval_df, role="eval"), cg, alpha=_ALPHA,
    )

    rows: list[dict] = []
    for omega in _OMEGA_VARIANTS:
        verdict = two_moment_wald_audit(
            audit_slice, cg, estimate.threshold, _ALPHA,
            omega_estimator=omega, B=_AUDIT_B, seed=seed,
        )
        rows.append({
            "n_oracle": n_oracle,
            "n_audit": audit_slice.n(),
            "policy": policy,
            "rep": rep,
            "omega_estimator": omega,
            "audit_W": verdict.W_n,
            "audit_p": verdict.p_value,
            "audit_decision": verdict.decision,
        })
    return rows


def run_sweep(n_workers: int = 1, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = _io.make_run_dir("omega_n_audit")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    _io.setup_logging(run_dir=out_dir)

    cells = list(product(
        _N_ORACLE_VALUES,
        list(DEFAULT_POLICIES.keys()),
        range(_R),
    ))
    LOG.info(
        "omega_n_audit sweep: n_oracle ∈ %s, policies=%s, R=%d, "
        "alpha=%.2f, audit_B=%d, omegas=%s, workers=%d",
        list(_N_ORACLE_VALUES), list(DEFAULT_POLICIES.keys()), _R, _ALPHA,
        _AUDIT_B, list(_OMEGA_VARIANTS), n_workers,
    )

    t0 = time.time()
    rows: list[dict] = []
    if n_workers <= 1:
        for i, c in enumerate(cells):
            rows.extend(_run_cell(c))
            if (i + 1) % max(1, len(cells) // 20) == 0 or i == len(cells) - 1:
                LOG.info("  %d/%d cells (%5.1f%%, %.1fs)",
                         i + 1, len(cells), 100*(i+1)/len(cells), time.time()-t0)
    else:
        ctx = get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for i, cell_rows in enumerate(pool.imap_unordered(_run_cell, cells, chunksize=1)):
                rows.extend(cell_rows)
                if (i + 1) % max(1, len(cells) // 20) == 0 or i == len(cells) - 1:
                    LOG.info("  %d/%d cells (%5.1f%%, %.1fs)",
                             i + 1, len(cells), 100*(i+1)/len(cells), time.time()-t0)

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "results.csv")

    # Aggregate over policies and reps; n_audit varies a few per cell due to
    # hash-partition discreteness, so we just record the median.
    summary = (
        df.group_by(["n_oracle", "omega_estimator"])
        .agg([
            (pl.col("audit_decision") == "REFUSE-LEVEL").mean().alias("mean_size"),
            pl.col("n_audit").median().alias("n_audit_median"),
        ])
        .with_columns((pl.col("mean_size") - _NOMINAL_ETA).abs().alias("size_dev"))
        .sort(["n_oracle", "omega_estimator"])
    )
    summary.write_csv(out_dir / "summary.csv")

    _make_figure(summary, out_dir)
    _write_report(summary, out_dir)
    LOG.info("done in %.1fs", time.time() - t0)
    return out_dir


def _make_figure(summary: pl.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for omega in _OMEGA_VARIANTS:
        sub = summary.filter(pl.col("omega_estimator") == omega).sort("n_oracle")
        ax.plot(
            sub["n_audit_median"].to_numpy(),
            sub["mean_size"].to_numpy(),
            marker="o", label=omega,
        )
    ax.axhline(_NOMINAL_ETA, color="k", linestyle="--",
               linewidth=1, label=f"nominal η={_NOMINAL_ETA}")
    ax.set_xscale("log")
    ax.set_xlabel("n_audit (median)")
    ax.set_ylabel("rejection rate at δ=0 (size)")
    ax.set_title("Audit size vs n_audit, averaged over policies × reps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "size_vs_n_audit.png", dpi=120)
    plt.close(fig)


def _write_report(summary: pl.DataFrame, out_dir: Path) -> None:
    lines = [
        "# omega_n_audit sweep — report",
        "",
        f"Run dir: `{out_dir}`",
        "",
        f"Setting: n_oracle ∈ {list(_N_ORACLE_VALUES)}, R={_R}, α={_ALPHA}, "
        f"audit_B={_AUDIT_B}, K={_K}, all 4 policies. Target size η={_NOMINAL_ETA}.",
        "",
        "## Mean size by (n_audit, Ω̂), averaged over policies × reps",
        "",
        "| n_oracle | n_audit (median) | Ω̂ | mean_size | size_dev |",
        "|---|---|---|---|---|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['n_oracle']} | {int(row['n_audit_median'])} | "
            f"{row['omega_estimator']} | {row['mean_size']:.3f} | "
            f"{row['size_dev']:.3f} |"
        )

    largest_n = max(_N_ORACLE_VALUES)
    lines.extend([
        "",
        f"## Decision (at largest n_oracle = {largest_n})",
        "",
        "| Ω̂ | mean_size | size_dev | calibrates (size_dev < 0.05)? |",
        "|---|---|---|---|",
    ])
    largest = summary.filter(pl.col("n_oracle") == largest_n).sort("size_dev")
    any_calibrates = False
    for row in largest.iter_rows(named=True):
        ok = row["size_dev"] < 0.05
        any_calibrates = any_calibrates or ok
        lines.append(
            f"| {row['omega_estimator']} | {row['mean_size']:.3f} | "
            f"{row['size_dev']:.3f} | {'YES' if ok else 'no'} |"
        )

    if any_calibrates:
        lines.extend([
            "",
            "**Result**: at least one Ω̂ calibrates at the largest n_audit. "
            "Lock the best as `Config.omega_estimator` default; close `[omega-research-n-audit]`.",
        ])
    else:
        lines.extend([
            "",
            "**Result**: NO Ω̂ calibrates even at the largest n_audit. "
            "`[omega-research-n-audit]` closes as 'increasing n doesn't help'. "
            "Escalate to sub-anchor `[omega-research-derivation]` "
            "(re-derive Ω̂ structurally).",
        ])

    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="cvar_v5 omega n_audit sweep")
    parser.add_argument("-w", "--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_sweep(n_workers=args.n_workers, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
