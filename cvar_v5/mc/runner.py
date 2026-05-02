"""
Monte Carlo outer loop for the cvar_v5 estimator + audit.

Math contract:
    For each cell (policy p, level α, replicate r, Ω̂-estimator ω, perturbation δ):
        seed = seed_base + 1000 · r
        oracle ~ DGP(p)            n_oracle rows, with Y       (prefix "O", seed)
        eval   ~ DGP(p, delta=δ)   n_eval rows,    no Y         (prefix "E", seed+1)
        partition oracle → CALIB, AUDIT  (hash bucket K-fold + holdout)
        fit calibrator on CALIB (K-fold cross-fit pooled fits)
        estimate = direct_cvar(eval, cg, α)     → ĈVaR_α, t̂_α
        verdict_ω = two_moment_wald_audit(audit, cg, t̂_α, α, ω, B)
        record (policy, α, r, ω, δ, ĈVaR, t̂, truth_cvar, abs_error, W, p, decision)

    Aggregates (computed in mc/report.py, not here):
        bias_{p,α,ω}   = mean_r(ĈVaR) − truth_cvar(p, α)
        rmse_{p,α,ω}   = sqrt(mean_r((ĈVaR − truth)²))
        size_{p,α,ω}   = mean_r(decision == REFUSE)  at δ=0
        power_{p,α,ω}(δ) = mean_r(decision == REFUSE) at δ>0

Mode parameters (smoke/medium/full):
                       smoke    medium     full
    n_oracle             300      600     1000
    n_eval               500     1000     2000
    audit_B              100      500     2000
    R (reps per cell)      5       60      200
    alphas         {.10,1.0}  {.10,.20,1.0}  {.10,.20,1.0}
    deltas              {0}      {0}     {0, 0.5, 1.0, 2.0}

The α=1 cell is always included (G1 collapse gate).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ..cvar_cje._crossfit import partition_oracle
from ..cvar_cje.audit import two_moment_wald_audit
from ..cvar_cje.calibrator import fit_calibrator_grid, make_t_grid
from ..cvar_cje.config import Config, OmegaEstimator
from ..cvar_cje.estimator import estimate_direct_cvar
from ..cvar_cje.schema import Slice
from . import _io
from .dgp import DEFAULT_POLICIES, DGP


LOG = logging.getLogger("cvar_v5.mc")


Mode = Literal["smoke", "medium", "full", "omega_sweep"]

OMEGA_ESTIMATORS: tuple[OmegaEstimator, ...] = (
    "analytical",
    "boot_remax_ridge",
    "boot_remax_no_ridge",
    "boot_fixed",
)


@dataclass(frozen=True)
class ModeParams:
    n_oracle: int
    n_eval: int
    audit_B: int
    R: int
    alphas: tuple[float, ...]
    deltas: tuple[float, ...]


def mode_params(mode: Mode) -> ModeParams:
    if mode == "smoke":
        return ModeParams(n_oracle=300, n_eval=500, audit_B=100, R=5,
                          alphas=(0.10, 1.0), deltas=(0.0,))
    if mode == "medium":
        return ModeParams(n_oracle=600, n_eval=1000, audit_B=500, R=60,
                          alphas=(0.10, 0.20, 1.0), deltas=(0.0,))
    if mode == "full":
        return ModeParams(n_oracle=1000, n_eval=2000, audit_B=2000, R=200,
                          alphas=(0.10, 0.20, 1.0), deltas=(0.0, 0.5, 1.0, 2.0))
    if mode == "omega_sweep":
        # Dedicated Ω̂ comparison sweep — closes the [omega-research] TODO.
        # α=0.10 only (audit info is at α<1; g_1 ≡ 0 at α=1).
        # Finer δ grid for power curves; R=100 is enough to estimate a
        # rejection-rate proportion to ~0.022 SE.
        return ModeParams(n_oracle=600, n_eval=1000, audit_B=500, R=100,
                          alphas=(0.10,),
                          deltas=(0.0, 0.1, 0.3, 0.5, 1.0))
    raise ValueError(f"unknown mode: {mode!r}")


@dataclass(frozen=True)
class Cell:
    policy: str
    alpha: float
    rep: int
    delta: float


def list_cells(policies: list[str], params: ModeParams) -> list[Cell]:
    return [
        Cell(policy=p, alpha=a, rep=r, delta=d)
        for p, a, r, d in product(policies, params.alphas, range(params.R), params.deltas)
    ]


def run_one_cell(
    cell: Cell,
    params: ModeParams,
    cfg_template: Config,
    dgp_policies: dict,
    seed_base: int,
) -> list[dict]:
    """
    Run one cell. Sweeps Ω̂-estimators within the cell so the calibrator is
    fit once and reused across all four audits.
    """
    dgp = DGP(dgp_policies)
    seed = seed_base + 1000 * cell.rep

    oracle = dgp.sample(
        cell.policy, n=params.n_oracle, with_oracle=True,
        seed=seed, delta=0.0, prompt_id_prefix="O",
    )
    eval_df = dgp.sample(
        cell.policy, n=params.n_eval, with_oracle=False,
        seed=seed + 1, delta=cell.delta, prompt_id_prefix="E",
    )

    calib_slice, audit_slice, folds = partition_oracle(
        oracle, K=cfg_template.K, seed=seed
    )
    t_grid = make_t_grid(
        cfg_template.grid_lo, cfg_template.grid_hi, cfg_template.grid_size,
        cfg_template.grid_kind,
    )
    cg = fit_calibrator_grid(
        s_calib=calib_slice.s(),
        y_calib=calib_slice.y(),
        t_grid=t_grid,
        fold_id=folds.fold_id,
        K=cfg_template.K,
    )

    estimate = estimate_direct_cvar(
        Slice(df=eval_df, role="eval"), cg, alpha=cell.alpha
    )
    truth_cvar = dgp.truth_cvar(cell.policy, cell.alpha)
    truth_mean = dgp.truth_mean(cell.policy)

    # Inline Mean-CJE reference for the α=1 collapse identity check.
    # Independent of the saddle-point path: a separate increasing isotonic fit
    # on (s_calib, y_calib), averaged over the EVAL slice. By the PAV reflection
    # identity (calibrator.py), this MUST equal estimate.value at α=1 to ≤ 1e-9.
    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        calib_slice.s(), calib_slice.y()
    )
    mean_ref = float(np.mean(f_inc.predict(eval_df["s"].to_numpy())))

    out: list[dict] = []
    for omega in OMEGA_ESTIMATORS:
        verdict = two_moment_wald_audit(
            audit_slice=audit_slice,
            calibrator=cg,
            t_hat=estimate.threshold,
            alpha=cell.alpha,
            omega_estimator=omega,
            B=params.audit_B,
            seed=seed,
            eta=cfg_template.audit_eta,
        )
        out.append({
            "policy": cell.policy,
            "alpha": cell.alpha,
            "rep": cell.rep,
            "delta": cell.delta,
            "omega_estimator": omega,
            "n_calib": calib_slice.n(),
            "n_audit": audit_slice.n(),
            "n_eval": params.n_eval,
            "estimate": estimate.value,
            "mean_ref": mean_ref,
            "t_hat": estimate.threshold,
            "truth_cvar": truth_cvar,
            "truth_mean": truth_mean,
            "abs_error": abs(estimate.value - truth_cvar),
            "alpha1_collapse_err": abs(estimate.value - mean_ref) if cell.alpha == 1.0 else 0.0,
            "audit_W": verdict.W_n,
            "audit_p": verdict.p_value,
            "audit_g1": verdict.g1,
            "audit_g2": verdict.g2,
            "audit_decision": verdict.decision,
        })
    return out


def _worker(args: tuple) -> list[dict]:
    cell, params, cfg_template, dgp_policies, seed_base = args
    return run_one_cell(cell, params, cfg_template, dgp_policies, seed_base)


def run_mc(
    mode: Mode = "smoke",
    n_workers: int = 1,
    seed_base: int = 0,
    out_dir: Path | None = None,
) -> tuple[pl.DataFrame, Path]:
    """
    Run the MC sweep in `mode` with `n_workers` parallel processes.

    Outputs land in a timestamped run directory:
        <out_dir>/results_mc.csv
        <out_dir>/run_config.json
        <out_dir>/log.txt

    If `out_dir` is None, a fresh dir is created under `cvar_v5/mc/runs/`.

    Returns (results_dataframe, run_dir).
    """
    params = mode_params(mode)
    cfg_template = Config(audit_B=params.audit_B)  # alpha used per-cell, not from cfg
    policies = list(DEFAULT_POLICIES.keys())
    cells = list_cells(policies, params)

    if out_dir is None:
        out_dir = _io.make_run_dir(mode)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    _io.setup_logging(run_dir=out_dir)
    _io.serialize_run_config(
        out_dir, mode=mode, n_workers=n_workers, seed_base=seed_base,
        cfg=cfg_template, mode_params=params,
    )

    LOG.info(
        "mode=%s  cells=%d  R=%d  policies=%s  alphas=%s  deltas=%s  "
        "omega=%s  workers=%d  out_dir=%s",
        mode, len(cells), params.R, policies,
        params.alphas, params.deltas, list(OMEGA_ESTIMATORS),
        n_workers, out_dir,
    )

    t0 = time.time()
    args_list = [
        (c, params, cfg_template, DEFAULT_POLICIES, seed_base) for c in cells
    ]

    rows: list[dict] = []
    if n_workers <= 1:
        for i, args in enumerate(args_list):
            rows.extend(_worker(args))
            if (i + 1) % max(1, len(cells) // 20) == 0 or i == len(cells) - 1:
                pct = 100 * (i + 1) / len(cells)
                elapsed = time.time() - t0
                LOG.info("%d/%d cells (%5.1f%%, %6.1fs)",
                         i + 1, len(cells), pct, elapsed)
    else:
        ctx = get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for i, cell_rows in enumerate(pool.imap_unordered(_worker, args_list, chunksize=1)):
                rows.extend(cell_rows)
                if (i + 1) % max(1, len(cells) // 20) == 0 or i == len(cells) - 1:
                    pct = 100 * (i + 1) / len(cells)
                    elapsed = time.time() - t0
                    LOG.info("%d/%d cells (%5.1f%%, %6.1fs)",
                             i + 1, len(cells), pct, elapsed)

    df = pl.DataFrame(rows)
    elapsed = time.time() - t0
    LOG.info("done. %d rows in %.1fs.", len(df), elapsed)

    csv_path = out_dir / "results_mc.csv"
    df.write_csv(csv_path)
    LOG.info("wrote %s", csv_path)
    return df, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="cvar_v5 MC validation runner")
    parser.add_argument("--smoke", action="store_const", const="smoke",
                        dest="mode", help="quick sanity run")
    parser.add_argument("--medium", action="store_const", const="medium",
                        dest="mode", help="moderate run")
    parser.add_argument("--full", action="store_const", const="full",
                        dest="mode", help="full validation run")
    parser.add_argument("--omega-sweep", action="store_const", const="omega_sweep",
                        dest="mode",
                        help="Ω̂ comparison sweep (size + power vs δ-perturbations)")
    parser.add_argument("-w", "--n-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override the auto-generated run dir under mc/runs/")
    args = parser.parse_args()
    mode: Mode = args.mode or "smoke"
    run_mc(
        mode=mode,
        n_workers=args.n_workers,
        seed_base=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
