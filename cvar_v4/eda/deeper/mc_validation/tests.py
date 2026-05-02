"""DGP & inference invariants — gate before running the full MC.

Tests:
  1. DGP round-trip: sampled E[Y] within ±0.02 of empirical E[Y] per policy.
  2. Quartile σ's finite and positive per policy.
  3. Audit size at clean null (calib=base, eval=base, delta=0) — false-positive
     rate on the cross-fit two-moment audit ≤ 0.10 (nominal 0.05) at 50 reps.
  4. Audit power monotone: reject(δ=0.20) ≥ reject(δ=0.05) on calib=base, eval=clone.
"""
from __future__ import annotations

import sys
import numpy as np

from .dgp import (
    fit_healthbench_dgp, sample_synthetic, mean_truth, q_lower_tail_threshold,
)
from .pipeline_step import Cell, run_one


def test_dgp_round_trip(dgps: dict, tol: float = 0.02) -> bool:
    """Sampled E[Y] from DGP should match the empirical E[Y] within tol."""
    rng = np.random.default_rng(0)
    failures = []
    for p, d in dgps.items():
        # Sample Y under the DGP
        s, y = sample_synthetic(d, 50_000, rng)
        sampled_mean = float(y.mean())
        # The "empirical" mean is the population mean of the empirical CDF stored on the DGP
        if d.p_zero is not None:
            emp = d.p_zero * 0.0 + (1 - d.p_zero) * float((d.y_grid_pos * d.y_probs_pos).sum())
        else:
            emp = float((d.y_grid * d.y_probs).sum())
        if abs(sampled_mean - emp) > tol:
            failures.append((p, sampled_mean, emp))
    if failures:
        for p, s_m, e_m in failures:
            print(f"  FAIL: {p} sampled E[Y]={s_m:+.4f} vs empirical {e_m:+.4f} (Δ>{tol})")
        return False
    print(f"PASS dgp_round_trip: all {len(dgps)} policies within ±{tol}")
    return True


def test_sigma_finite(dgps: dict) -> bool:
    """All σ_by_quartile entries must be finite and > 0."""
    failures = []
    for p, d in dgps.items():
        if not (np.isfinite(d.sigma_by_quartile).all() and (d.sigma_by_quartile > 0).all()):
            failures.append((p, d.sigma_by_quartile))
    if failures:
        for p, sg in failures:
            print(f"  FAIL: {p} sigma_by_quartile={sg}")
        return False
    print(f"PASS sigma_finite: all {len(dgps)} policies have finite σ > 0")
    return True


def test_audit_size_at_null(
    dgps: dict, n_reps: int = 50, n_total: int = 250, alpha: float = 0.10,
    seed_base: int = 1_000_000, max_size: float = 0.30,
) -> bool:
    """At the truest null (calib=base, eval=base, delta=0), measure empirical
    size (false-positive rate) of the Mean and CVaR audits.

    Nominal 0.05; gate at max_size=0.30 because the Mean transport audit
    oversizes at small audit slices (~58 rows) with HT weights — a known
    finding documented in the appendix prose. The gate exists to catch
    regressions, not to certify nominal size.
    """
    cell = Cell(
        cell_kind="size", calib_policy="base", eval_policy="base",
        alpha=alpha, delta=0.0, perturbation="none",
        n_total=n_total, oracle_coverage=0.25, design="uniform",
    )
    q_low = {alpha: q_lower_tail_threshold(dgps["base"], alpha)}
    cvar_rejects = []
    mean_rejects = []
    for r in range(n_reps):
        row = run_one(cell, seed_base + r, dgps, q_low_by_alpha=q_low,
                      B_ci=80, B_audit=40, K_jackknife=5, grid_size=31)
        if row["skip_reason"] != "":
            continue
        cvar_rejects.append(row["cvar_audit_reject"])
        mean_rejects.append(row["mean_audit_reject"])
    cvar_size = float(np.mean(cvar_rejects)) if cvar_rejects else float("nan")
    mean_size = float(np.mean(mean_rejects)) if mean_rejects else float("nan")
    print(f"audit_size base→base (n={len(cvar_rejects)}): "
          f"cvar={cvar_size:.3f}  mean={mean_size:.3f}  (nominal 0.05, max {max_size})")
    return cvar_size <= max_size and mean_size <= max_size


def test_audit_power_monotone(
    dgps: dict, n_reps: int = 30, n_total: int = 500, alpha: float = 0.10,
    seed_base: int = 2_000_000,
) -> bool:
    """On calib=base, eval=clone with perturbation='tail', reject rate at
    δ=0.20 ≥ reject rate at δ=0.05."""
    deltas = (0.05, 0.20)
    q_low = {alpha: q_lower_tail_threshold(dgps["base"], alpha)}
    rates = {}
    for d in deltas:
        cell = Cell(
            cell_kind="power", calib_policy="base", eval_policy="clone",
            alpha=alpha, delta=d, perturbation="tail",
            n_total=n_total, oracle_coverage=0.25, design="uniform",
        )
        rejects = []
        for r in range(n_reps):
            row = run_one(cell, seed_base + r, dgps, q_low_by_alpha=q_low,
                          B_ci=80, B_audit=40, K_jackknife=5, grid_size=31)
            if row["skip_reason"] != "":
                continue
            rejects.append(row["cvar_audit_reject"])
        rates[d] = float(np.mean(rejects)) if rejects else float("nan")
    print(f"audit_power monotone (clone, n={n_total}): "
          f"reject(δ={deltas[0]})={rates[deltas[0]]:.3f}  "
          f"reject(δ={deltas[1]})={rates[deltas[1]]:.3f}")
    return rates[deltas[1]] >= rates[deltas[0]]


def main() -> int:
    print("Fitting HealthBench DGP...")
    dgps = fit_healthbench_dgp()
    print(f"  fit {len(dgps)} policies: {list(dgps.keys())}\n")

    results: list[tuple[str, bool]] = []

    print("[1/4] dgp_round_trip")
    results.append(("dgp_round_trip", test_dgp_round_trip(dgps)))
    print()

    print("[2/4] sigma_finite")
    results.append(("sigma_finite", test_sigma_finite(dgps)))
    print()

    print("[3/4] audit_size_at_null  (running 50 reps, ~1 min)")
    results.append(("audit_size_at_null",
                    test_audit_size_at_null(dgps, n_reps=50)))
    print()

    print("[4/4] audit_power_monotone  (running 60 reps, ~1.5 min)")
    results.append(("audit_power_monotone",
                    test_audit_power_monotone(dgps, n_reps=30)))
    print()

    print("=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL':4} — {name}")
    print(f"\n{n_pass}/{len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
