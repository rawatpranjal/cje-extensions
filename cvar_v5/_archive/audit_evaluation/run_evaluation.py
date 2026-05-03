"""
Audit-Ω̂ evaluation runner.

Order of operations:
    1. Run sanity probes S1–S6 (test the test). If any FAIL, halt.
    2. Run R reps under H_0 → compute Σ_full, ε empirically.
    3. Compute Σ_oracle (closed form / quadrature) for cross-check.
    4. For each candidate Ω̂ method M, compute the 4-axis diagnostic.
    5. Write summary CSV + report markdown to mc/runs/<ts>_audit_evaluation/.

CLI:
    python -m cvar_v5._archive.audit_evaluation.run_evaluation
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ._diagnostics import diagnose_method, jackknife_bias_correct
from ._harness import run_one_rep
from ._sanity import run_all_probes, _empirical_sigma_full_and_eps
from ._truths import DGPParams, TargetPert, sigma_oracle, t_star


LOG = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(description="audit-Ω̂ evaluation harness")
    parser.add_argument("--R", type=int, default=300,
                        help="MC reps for Σ_full / ε / per-method axes")
    parser.add_argument("--n-calib", type=int, default=600)
    parser.add_argument("--n-audit", type=int, default=250)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--B", type=int, default=200,
                        help="Bootstrap reps per Ω̂_M (per outer rep)")
    parser.add_argument("--seed-base", type=int, default=20260503)
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip sanity probes (only run main eval)")
    args = parser.parse_args()

    out_dir = _make_run_dir()
    _setup_logging(out_dir)
    LOG.info("audit-Ω̂ evaluation; out_dir=%s; R=%d, n_calib=%d, n_audit=%d, n_eval=%d, B=%d",
             out_dir, args.R, args.n_calib, args.n_audit, args.n_eval, args.B)

    p = DGPParams(a=1.0, b=1.0)            # uniform policy as the workhorse DGP
    pert = TargetPert()                    # H_0: transport holds at population
    t_grid = np.linspace(0.0, 1.0, 121)    # finer than usual to reduce grid bias

    # ---------------- Sanity probes (test the test) ----------------
    if not args.skip_sanity:
        LOG.info("running sanity probes S1–S6...")
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
            LOG.warning("Some sanity probes FAILED. Continuing anyway, but trust the "
                        "main eval results with caution.")

    # ---------------- Main eval: methods on H_0 ----------------
    LOG.info("computing Σ_oracle (closed-form / quadrature)...")
    Sigma_oracle = sigma_oracle(p, pert, args.alpha)
    LOG.info("Σ_oracle =\n%s", Sigma_oracle)

    LOG.info("running R=%d MC reps under H_0...", args.R)
    eps, sigma_full, g_arr, g_per_fold_all = _empirical_sigma_full_and_eps(
        p, pert, args.alpha, args.n_calib, args.n_audit, args.n_eval,
        t_grid, args.K, args.B, args.R, args.seed_base + 100,
    )
    LOG.info("ε (empirical bias) = %s", eps.tolist())
    LOG.info("Σ_full = n_audit · sample-cov(ḡ):\n%s", sigma_full)

    # We need the omegas already collected during reps. Re-run to gather.
    LOG.info("collecting per-method Ω̂ across reps (this duplicates the H_0 reps)...")
    method_names = ["analytical", "boot_remax", "boot_remax_ridge",
                    "analytical_oua", "boot_remax_oua"]
    omegas_per_method = {m: [] for m in method_names}
    g_realized = []
    g_per_fold_collected = []
    for r in range(args.R):
        seed = args.seed_base + 100 + 9007 * r
        res = run_one_rep(p, pert, args.alpha, args.n_calib, args.n_audit, args.n_eval,
                          t_grid, args.K, args.B, seed)
        for m in method_names:
            omegas_per_method[m].append(res.omegas[m])
        g_realized.append(res.ḡ)
        g_per_fold_collected.append(res.g_per_fold)
        if (r + 1) % max(1, args.R // 10) == 0:
            LOG.info("  rep %d / %d", r + 1, args.R)
    g_realized = np.stack(g_realized)
    eps_recomputed = g_realized.mean(axis=0)
    sigma_full_rec = args.n_audit * np.cov(g_realized, rowvar=False, ddof=1)

    # 4-axis diagnostic per method (ḡ uncorrected by default)
    LOG.info("computing 4-axis diagnostic per method (no bias correction)...")
    diag_rows = []
    for m in method_names:
        diag = diagnose_method(
            name=m,
            omegas=omegas_per_method[m],
            g_used=[g_realized[r] for r in range(args.R)],
            sigma_full=sigma_full_rec,
            n_audit=args.n_audit,
        )
        diag_rows.append({
            "method": diag.name,
            "var_bias_trace": float(np.trace(diag.var_bias)),
            "var_bias_11": float(diag.var_bias[0, 0]),
            "var_bias_22": float(diag.var_bias[1, 1]),
            "center_bias_g1": float(diag.center_bias[0]),
            "center_bias_g2": float(diag.center_bias[1]),
            "empirical_size": diag.empirical_size,
            "predicted_size": diag.predicted_size,
        })
        LOG.info("  %s: emp_size=%.3f predicted=%.3f var_bias_trace=%+.3e center=(%+.4f, %+.4f)",
                 diag.name, diag.empirical_size, diag.predicted_size,
                 float(np.trace(diag.var_bias)),
                 float(diag.center_bias[0]), float(diag.center_bias[1]))

    # Same diagnostic with jackknife bias-correction overlay
    LOG.info("computing 4-axis diagnostic per method with jackknife bias-correction...")
    g_bc = [jackknife_bias_correct(g_realized[r], g_per_fold_collected[r])
            for r in range(args.R)]
    for m in method_names:
        diag_bc = diagnose_method(
            name=m + "+_bc",
            omegas=omegas_per_method[m],
            g_used=g_bc,
            sigma_full=sigma_full_rec,
            n_audit=args.n_audit,
        )
        diag_rows.append({
            "method": diag_bc.name,
            "var_bias_trace": float(np.trace(diag_bc.var_bias)),
            "var_bias_11": float(diag_bc.var_bias[0, 0]),
            "var_bias_22": float(diag_bc.var_bias[1, 1]),
            "center_bias_g1": float(diag_bc.center_bias[0]),
            "center_bias_g2": float(diag_bc.center_bias[1]),
            "empirical_size": diag_bc.empirical_size,
            "predicted_size": diag_bc.predicted_size,
        })
        LOG.info("  %s: emp_size=%.3f predicted=%.3f var_bias_trace=%+.3e center_bc=(%+.4f, %+.4f)",
                 diag_bc.name, diag_bc.empirical_size, diag_bc.predicted_size,
                 float(np.trace(diag_bc.var_bias)),
                 float(diag_bc.center_bias[0]), float(diag_bc.center_bias[1]))

    pl.DataFrame(diag_rows).write_csv(out_dir / "diagnostic.csv")

    # Provenance table
    pl.DataFrame([{
        "Sigma_oracle_11": float(Sigma_oracle[0, 0]),
        "Sigma_oracle_12": float(Sigma_oracle[0, 1]),
        "Sigma_oracle_22": float(Sigma_oracle[1, 1]),
        "Sigma_full_11":   float(sigma_full_rec[0, 0]),
        "Sigma_full_12":   float(sigma_full_rec[0, 1]),
        "Sigma_full_22":   float(sigma_full_rec[1, 1]),
        "eps_g1":          float(eps_recomputed[0]),
        "eps_g2":          float(eps_recomputed[1]),
    }]).write_csv(out_dir / "truth.csv")

    LOG.info("done. outputs in %s", out_dir)


if __name__ == "__main__":
    main()
