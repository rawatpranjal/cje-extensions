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
from pathlib import Path

import numpy as np
import polars as pl

from ._bias_corrections import bc_jk_cal, bc_jk_full, bc_boot
from ._bias_diagnostic import diagnose_bias
from ._harness import run_one_rep
from ._sanity import run_all_probes, _empirical_sigma_full_and_eps
from ._size_diagnostic import diagnose_size
from ._truths import DGPParams, TargetPert, sigma_oracle, t_star
from ._variance_diagnostic import diagnose_variance


LOG = logging.getLogger(__name__)


# Variance-side methods (Ω̂ estimators).
_VAR_METHODS = [
    "analytical", "boot_remax", "boot_remax_ridge",
    "analytical_oua", "boot_remax_oua",
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
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip sanity probes S1-9 (only run main eval)")
    args = parser.parse_args()

    out_dir = _make_run_dir()
    _setup_logging(out_dir)
    LOG.info("audit-Ω̂ evaluation; out_dir=%s", out_dir)
    LOG.info("R=%d, n_calib=%d, n_audit=%d, n_eval=%d, B=%d, B_boot_bc=%d",
             args.R, args.n_calib, args.n_audit, args.n_eval, args.B, args.B_boot_bc)

    p = DGPParams(a=1.0, b=1.0)
    pert = TargetPert()
    t_grid = np.linspace(0.0, 1.0, 121)

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

    # ---------------- Truth (Σ_oracle, t*) -------------------------
    LOG.info("computing population truths (Σ_oracle, t*)...")
    Sigma_oracle = sigma_oracle(p, pert, args.alpha)
    t_pop = t_star(p, pert, args.alpha)
    LOG.info("Σ_oracle = %s, t* = %.5f", Sigma_oracle.tolist(), t_pop)

    # ---------------- Run R reps and collect everything ------------
    LOG.info("running R=%d MC reps under H_0...", args.R)
    omegas: dict = {m: [] for m in _VAR_METHODS}
    g_used: dict = {bc: [] for bc in _BC_LABELS}
    g_realized: list = []

    for r in range(args.R):
        seed = args.seed_base + 100 + 9007 * r
        rep = run_one_rep(
            p, pert, args.alpha, args.n_calib, args.n_audit, args.n_eval,
            t_grid, args.K, args.B, seed,
        )
        for m in _VAR_METHODS:
            omegas[m].append(rep.omegas[m])
        g_realized.append(rep.ḡ)
        for bc in _BC_LABELS:
            g_used[bc].append(_apply_bc(bc, rep, t_grid, args.alpha,
                                         args.K, args.B_boot_bc, seed=seed + 5))
        if (r + 1) % max(1, args.R // 10) == 0:
            LOG.info("  rep %d / %d", r + 1, args.R)

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
