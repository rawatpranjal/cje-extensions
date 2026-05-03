"""
Audit-Ω̂ evaluation runner — clean separation between bias and variance.

Order of operations:
    1. Run sanity probes S1–S9 (test the test). If any FAIL, log warning.
    2. Run R reps under H_0 → collect per-method Ω̂_M and per-bias-correction
       ḡ_used vectors. Compute Σ_full empirically.
    3. Compute Σ_oracle (closed form / quadrature) for cross-check (truth.csv).
    4. Run three diagnostics, each writing its own CSV:
        - bias_report.csv         : per (variance, bias_correction) pair, the
                                    center of ḡ_used and its z-score.
        - variance_report.csv     : per variance method only, |Ω̂ − target|.
        - size_report.csv         : full cross of (variance × bias correction):
                                    empirical and per-rep-Ω̂ predicted size.

CLI:
    python -m cvar_v5._archive.audit_evaluation.run_evaluation [--R 300] [--B 100]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from ._bias_corrections import bc_jk_cal, bc_jk_full, bc_boot
from ._bias_diagnostic import diagnose_bias
from ._harness import run_one_rep
from ._sanity import run_all_probes
from ._size_diagnostic import diagnose_size
from ._truth_decomp import compute_truth_decomposition
from ._truths import DGPParams, TargetPert, sigma_oracle, t_star
from ._variance_diagnostic import diagnose_variance
from ._variance_methods_extra import boot_v_cal_oua, full_pipeline_boot


LOG = logging.getLogger(__name__)


# Variance-side methods (Ω̂ estimators). The first 5 are computed inside
# run_one_rep; the last 2 are computed by the runner after the rep returns
# (they need B_cal / B_full and consume the raw rep arrays).
_VAR_METHODS = [
    "analytical", "boot_remax", "boot_remax_ridge",
    "analytical_oua", "boot_remax_oua",
    "boot_v_cal_oua",            # NEW (additive: V_audit + bootstrap V_cal)
    "full_pipeline_boot",        # NEW (integrated; captures cross-terms)
]

# Bias-correction labels.
_BC_LABELS = ["none", "bc_jk_cal", "bc_jk_full", "bc_boot"]


def _make_run_dir() -> Path:
    base = Path(__file__).resolve().parents[2] / "mc" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out = base / f"{ts}_audit_evaluation"
    n = 2
    while out.exists():
        out = base / f"{ts}_audit_evaluation_{n}"
        n += 1
    out.mkdir()
    return out


def _setup_logging(run_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "log.txt", mode="w"),
        ],
    )


def _apply_bc(label: str, rep, t_grid, alpha, K, B_boot, seed):
    """Dispatch from a bias-correction label to the corresponding ḡ_used."""
    if label == "none":
        return rep.ḡ
    if label == "bc_jk_cal":
        return bc_jk_cal(rep)
    if label == "bc_jk_full":
        return bc_jk_full(rep)
    if label == "bc_boot":
        return bc_boot(rep, t_grid, alpha, K, B_boot, seed=seed)
    raise ValueError(f"unknown bias correction label: {label!r}")


def _run_one_rep_worker(payload: dict) -> dict:
    """
    Worker for parallel pool. Takes a fully-populated payload dict and returns
    a results dict with:
        - per-method Ω̂s (5 from rep.omegas + 2 NEW computed here)
        - per-bc ḡ_used vectors
        - raw ḡ

    All inputs are pickle-safe (numpy arrays, primitives, dataclasses).
    """
    p = payload["p"]
    pert = payload["pert"]
    alpha = payload["alpha"]
    n_calib = payload["n_calib"]
    n_audit = payload["n_audit"]
    n_eval = payload["n_eval"]
    t_grid = payload["t_grid"]
    K = payload["K"]
    B = payload["B"]
    B_boot_bc = payload["B_boot_bc"]
    B_cal_var = payload["B_cal_var"]
    B_full_var = payload["B_full_var"]
    seed = payload["seed"]
    bc_labels = payload["bc_labels"]

    rep = run_one_rep(
        p, pert, alpha, n_calib, n_audit, n_eval, t_grid, K, B, seed,
    )
    # Existing 5 variance methods are on rep.omegas already.
    omegas = dict(rep.omegas)
    # The two NEW variance methods consume the raw arrays + their own bootstrap
    # budgets. Compute them here so the worker stays self-contained.
    omegas["boot_v_cal_oua"] = boot_v_cal_oua(
        rep, t_grid, alpha, K, B_cal_var, seed=seed + 11,
    )
    omegas["full_pipeline_boot"] = full_pipeline_boot(
        rep, t_grid, alpha, K, B_full_var, seed=seed + 13,
    )

    g_used = {
        bc: _apply_bc(bc, rep, t_grid, alpha, K, B_boot_bc, seed=seed + 5)
        for bc in bc_labels
    }
    return {
        "omegas": omegas,
        "g_used": g_used,
        "g_realized": rep.ḡ,
    }


def main():
    parser = argparse.ArgumentParser(description="audit-Ω̂ evaluation harness")
    parser.add_argument("--R", type=int, default=300,
                        help="MC reps for Σ_full / per-method axes")
    parser.add_argument("--n-calib", type=int, default=600)
    parser.add_argument("--n-audit", type=int, default=250)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--B", type=int, default=100,
                        help="Bootstrap reps INSIDE per-method Ω̂ (boot_remax variants)")
    parser.add_argument("--B-boot-bc", type=int, default=80,
                        help="Bootstrap reps for bc_boot's bias estimate")
    parser.add_argument("--seed-base", type=int, default=20260503)
    parser.add_argument("--policy", default="uniform",
                        choices=["uniform", "right_skew", "left_skew", "tail_heavy"],
                        help="DGP policy. Beta(a, b) parameters per the cvar_v5 panel.")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip sanity probes S1-9 (only run main eval)")
    parser.add_argument("--skip-truth-decomp", action="store_true",
                        help="skip ground-truth decomposition (4·R extra reps)")
    parser.add_argument("-w", "--n-workers", type=int, default=1,
                        help="MC reps run in parallel (multiprocessing fork pool)")
    parser.add_argument("--B-cal-var", type=int, default=80,
                        help="Bootstrap reps for boot_v_cal_oua (per audit verdict)")
    parser.add_argument("--B-full-var", type=int, default=80,
                        help="Bootstrap reps for full_pipeline_boot (per audit verdict)")
    args = parser.parse_args()

    out_dir = _make_run_dir()
    _setup_logging(out_dir)
    LOG.info("audit-Ω̂ evaluation; out_dir=%s", out_dir)
    LOG.info("R=%d, n_calib=%d, n_audit=%d, n_eval=%d, B=%d, B_boot_bc=%d",
             args.R, args.n_calib, args.n_audit, args.n_eval, args.B, args.B_boot_bc)

    # Map policy name to DGPParams (mirrors cvar_v5/mc/dgp.py panel).
    _POLICY_PARAMS = {
        "uniform":    DGPParams(a=1.0, b=1.0),
        "right_skew": DGPParams(a=2.0, b=5.0),
        "left_skew":  DGPParams(a=5.0, b=2.0),
        "tail_heavy": DGPParams(a=0.5, b=0.5),
    }
    p = _POLICY_PARAMS[args.policy]
    pert = TargetPert()
    t_grid = np.linspace(0.0, 1.0, 121)
    LOG.info("policy = %s; DGPParams = %s", args.policy, p)

    # ---------------- Sanity probes (test the test) ----------------
    if not args.skip_sanity:
        LOG.info("running sanity probes S1–S9...")
        probes = run_all_probes(
            p, args.alpha, args.n_calib, args.n_audit, args.n_eval,
            t_grid, args.K, args.B, R=args.R, seed_base=args.seed_base,
        )
        sanity_rows = []
        for pr in probes:
            sanity_rows.append({
                "name": pr.name, "passed": pr.passed, "detail": pr.detail,
            })
            LOG.info("  %s  -- %s  -- %s",
                     "PASS" if pr.passed else "FAIL", pr.name, pr.detail)
        pl.DataFrame(sanity_rows).write_csv(out_dir / "sanity.csv")
        all_passed = all(r["passed"] for r in sanity_rows)
        if not all_passed:
            LOG.warning("Some sanity probes FAILED. Continuing; trust eval results with caution.")

    # ---------------- Truth (Σ_oracle, t*, ground-truth decomp) -----
    LOG.info("computing population truths (Σ_oracle, t*)...")
    Sigma_oracle = sigma_oracle(p, pert, args.alpha)
    t_pop = t_star(p, pert, args.alpha)
    LOG.info("Σ_oracle = %s, t* = %.5f", Sigma_oracle.tolist(), t_pop)

    if not args.skip_truth_decomp:
        LOG.info("computing ground-truth variance decomposition (4·R reps)...")
        td = compute_truth_decomposition(
            p, pert, args.alpha, args.n_calib, args.n_audit, args.n_eval,
            args.K, t_grid, args.B, R=args.R, seed_base=args.seed_base + 50000,
        )
        LOG.info(
            "  trace(Σ_full)=%.5f  V_audit=%.5f (%.0f%%)  V_calib=%.5f (%.0f%%)  "
            "V_eval=%.5f (%.0f%%)  cross_residual=%.5f (%.0f%%)",
            td.trace_full,
            td.trace_audit, 100 * td.trace_audit / td.trace_full,
            td.trace_calib, 100 * td.trace_calib / td.trace_full,
            td.trace_eval,  100 * td.trace_eval  / td.trace_full,
            td.trace_full - td.trace_predicted,
            100 * (td.trace_full - td.trace_predicted) / td.trace_full,
        )
        # Persist the decomposition for downstream analysis.
        pl.DataFrame([{
            "trace_full":    td.trace_full,
            "trace_audit":   td.trace_audit,
            "trace_calib":   td.trace_calib,
            "trace_eval":    td.trace_eval,
            "trace_predicted": td.trace_predicted,
            "rel_err_predicted": td.rel_err_predicted,
            "Sigma_full_11":  float(td.Sigma_full[0, 0]),
            "Sigma_full_22":  float(td.Sigma_full[1, 1]),
            "V_audit_11":     float(td.V_audit_only[0, 0]),
            "V_audit_22":     float(td.V_audit_only[1, 1]),
            "V_calib_11":     float(td.V_calib_only[0, 0]),
            "V_calib_22":     float(td.V_calib_only[1, 1]),
            "V_eval_11":      float(td.V_eval_only[0, 0]),
            "V_eval_22":      float(td.V_eval_only[1, 1]),
            "cross_11":       float(td.cross_residual[0, 0]),
            "cross_22":       float(td.cross_residual[1, 1]),
        }]).write_csv(out_dir / "truth_decomp.csv")

    # ---------------- Run R reps and collect everything ------------
    LOG.info("running R=%d MC reps under H_0 (workers=%d)...",
             args.R, args.n_workers)
    omegas: dict = {m: [] for m in _VAR_METHODS}
    g_used: dict = {bc: [] for bc in _BC_LABELS}
    g_realized: list = []

    payloads = [
        {
            "p": p, "pert": pert, "alpha": args.alpha,
            "n_calib": args.n_calib, "n_audit": args.n_audit, "n_eval": args.n_eval,
            "t_grid": t_grid, "K": args.K, "B": args.B,
            "B_boot_bc": args.B_boot_bc,
            "B_cal_var": args.B_cal_var,
            "B_full_var": args.B_full_var,
            "seed": args.seed_base + 100 + 9007 * r,
            "bc_labels": _BC_LABELS,
        }
        for r in range(args.R)
    ]

    import time
    t0 = time.time()

    if args.n_workers <= 1:
        # Single-threaded path.
        for r, payload in enumerate(payloads):
            res = _run_one_rep_worker(payload)
            for m in _VAR_METHODS:
                omegas[m].append(res["omegas"][m])
            g_realized.append(res["g_realized"])
            for bc in _BC_LABELS:
                g_used[bc].append(res["g_used"][bc])
            if (r + 1) % max(1, args.R // 10) == 0:
                LOG.info("  rep %d / %d  (%.1fs)", r + 1, args.R, time.time() - t0)
    else:
        # Multiprocessing fork pool. Order of results is irrelevant for
        # the per-method aggregate stats; we use imap_unordered for
        # efficiency.
        ctx = get_context("fork")
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(
                _run_one_rep_worker, payloads, chunksize=1,
            )):
                for m in _VAR_METHODS:
                    omegas[m].append(res["omegas"][m])
                g_realized.append(res["g_realized"])
                for bc in _BC_LABELS:
                    g_used[bc].append(res["g_used"][bc])
                if (i + 1) % max(1, args.R // 10) == 0:
                    LOG.info("  rep %d / %d  (%.1fs)", i + 1, args.R, time.time() - t0)

    g_arr = np.stack(g_realized, axis=0)
    sigma_full = args.n_audit * np.cov(g_arr, rowvar=False, ddof=1)
    eps = g_arr.mean(axis=0)
    LOG.info("ε (raw center)  = %s", eps.tolist())
    LOG.info("Σ_full / n_audit = %s", (sigma_full / args.n_audit).tolist())
    LOG.info("Σ_oracle         = %s", Sigma_oracle.tolist())

    # ---------------- (1) Bias diagnostic per (variance, bc) -----
    # Bias depends only on bc (not on variance method), but we list all
    # variance × bc rows for traceability and so the size table joins cleanly.
    LOG.info("computing bias diagnostic per (variance, bc) pair...")
    bias_rows = []
    for bc in _BC_LABELS:
        bd = diagnose_bias(f"bc={bc}", g_used[bc], bias_correction=bc)
        # one row per variance method × bc combination
        for m in _VAR_METHODS:
            bias_rows.append({
                "variance": m,
                "bias_correction": bc,
                "center_g1": float(bd.center[0]),
                "center_g2": float(bd.center[1]),
                "se_g1": float(bd.se[0]),
                "se_g2": float(bd.se[1]),
                "z_g1": float(bd.z[0]),
                "z_g2": float(bd.z[1]),
                "passed": bd.passed,
            })
        LOG.info("  bias bc=%-12s center=(%+.5f, %+.5f) z=(%+.2f, %+.2f) passed=%s",
                 bc, bd.center[0], bd.center[1], bd.z[0], bd.z[1], bd.passed)
    pl.DataFrame(bias_rows).write_csv(out_dir / "bias_report.csv")

    # ---------------- (2) Variance diagnostic per variance method --
    LOG.info("computing variance diagnostic per variance method...")
    var_rows = []
    for m in _VAR_METHODS:
        vd = diagnose_variance(m, omegas[m], sigma_full, args.n_audit)
        var_rows.append({
            "variance": m,
            "frob_rel_err": vd.frob_rel_err,
            "spectral_ratio": vd.spectral_ratio,
            "var_bias_11": float(vd.var_bias[0, 0]),
            "var_bias_22": float(vd.var_bias[1, 1]),
            "var_bias_12": float(vd.var_bias[0, 1]),
            "dispersion_11": float(vd.dispersion[0, 0]),
            "dispersion_22": float(vd.dispersion[1, 1]),
            "passed": vd.passed,
        })
        LOG.info("  variance %-22s frob_rel_err=%.3f spectral_ratio=%.3f passed=%s",
                 m, vd.frob_rel_err, vd.spectral_ratio, vd.passed)
    pl.DataFrame(var_rows).write_csv(out_dir / "variance_report.csv")

    # ---------------- (3) Size diagnostic per (variance × bc) -----
    LOG.info("computing size diagnostic per (variance × bias_correction) pair...")
    size_rows = []
    for m in _VAR_METHODS:
        for bc in _BC_LABELS:
            sd = diagnose_size(
                f"{m}+{bc}", omegas[m], g_used[bc],
                bias_correction=bc, pred_seed=hash((m, bc)) & 0xFFFF_FFFF,
            )
            size_rows.append({
                "variance": m,
                "bias_correction": bc,
                "empirical_size": sd.empirical_size,
                "predicted_size": sd.predicted_size,
                "abs_error": sd.abs_error,
                "size_dev": abs(sd.empirical_size - 0.05),
                "passed": sd.passed,
            })
            LOG.info("  size %-22s+%-12s emp=%.3f pred=%.3f |Δ|=%.3f size_dev=%.3f passed=%s",
                     m, bc, sd.empirical_size, sd.predicted_size,
                     sd.abs_error, abs(sd.empirical_size - 0.05), sd.passed)
    pl.DataFrame(size_rows).write_csv(out_dir / "size_report.csv")

    # Provenance / truth values
    pl.DataFrame([{
        "Sigma_oracle_11": float(Sigma_oracle[0, 0]),
        "Sigma_oracle_22": float(Sigma_oracle[1, 1]),
        "Sigma_oracle_12": float(Sigma_oracle[0, 1]),
        "Sigma_full_11": float(sigma_full[0, 0]),
        "Sigma_full_22": float(sigma_full[1, 1]),
        "Sigma_full_12": float(sigma_full[0, 1]),
        "eps_g1": float(eps[0]),
        "eps_g2": float(eps[1]),
        "t_star": t_pop,
        "alpha": args.alpha,
        "n_calib": args.n_calib,
        "n_audit": args.n_audit,
        "n_eval": args.n_eval,
        "R": args.R,
        "B": args.B,
    }]).write_csv(out_dir / "truth.csv")

    LOG.info("done. outputs in %s", out_dir)


if __name__ == "__main__":
    main()
