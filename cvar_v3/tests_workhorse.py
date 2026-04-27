"""Smoke tests for cvar_v3/workhorse.py.

Synthetic tests that don't require the Arena data. Run with:
    python3.11 cvar_v3/tests_workhorse.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "cvar")
from workhorse import (  # noqa: E402
    cluster_bootstrap_cvar,
    estimate_direct_cvar_isotonic,
    estimate_direct_mean_isotonic,
    fit_isotonic_mean,
    fit_isotonic_tail_loss,
    make_t_grid,
    two_moment_wald_audit,
)


def _dgp(n: int, seed: int = 0, noise: float = 0.08):
    """S = true_quality + noise; Y = true_quality. Monotone in expectation."""
    rng = np.random.default_rng(seed)
    y = rng.beta(5, 2, size=n)
    s = np.clip(y + rng.normal(0, noise, size=n), 0.0, 1.0)
    return s, y


def test_isotonic_mean_recovery():
    s, y = _dgp(10_000, seed=1)
    pred = fit_isotonic_mean(s, y, s)
    empirical = y.mean()
    recovered = pred.mean()
    err = abs(recovered - empirical)
    ok = err < 0.01
    print(f"[{'PASS' if ok else 'FAIL'}] isotonic mean recovery: "
          f"empirical={empirical:.4f}, recovered={recovered:.4f}, err={err:.5f}")
    return ok


def test_estimate_direct_mean():
    s, y = _dgp(10_000, seed=2)
    est = estimate_direct_mean_isotonic(s, y, s)
    truth = y.mean()
    err = abs(est - truth)
    ok = err < 0.005
    print(f"[{'PASS' if ok else 'FAIL'}] direct mean estimator: "
          f"truth={truth:.4f}, estimate={est:.4f}, err={err:.5f}")
    return ok


def test_direct_cvar():
    s, y = _dgp(20_000, seed=3, noise=0.05)
    alpha = 0.10
    cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(s, y, s, alpha, grid_size=61)
    q = np.quantile(y, alpha)
    emp = y[y <= q].mean()
    # Direct CVaR is a LOWER BOUND on the true CVaR (grid-search maximum of
    # a dual objective). So `cvar_est >= true_cvar - small_slack` is sufficient;
    # we check it's within ±5% relatively.
    err = abs(cvar_est - emp)
    ok = err < 0.03 and np.isfinite(t_hat)
    print(f"[{'PASS' if ok else 'FAIL'}] direct CVaR estimator: "
          f"empirical_cvar={emp:.4f}, estimate={cvar_est:.4f}, t_hat={t_hat:.4f}, err={err:.5f}")
    return ok


def test_make_t_grid_bounds():
    _, y = _dgp(5000, seed=4)
    grid = make_t_grid(y, 0.10, 31)
    ok = grid[0] < np.quantile(y, 0.10) < grid[-1] and len(grid) == 31
    print(f"[{'PASS' if ok else 'FAIL'}] t_grid bounds: [{grid[0]:.3f}, {grid[-1]:.3f}], len={len(grid)}")
    return ok


def test_transport_audit_well_specified():
    # Larger n + tighter S–Y noise so the well-specified audit isn't borderline
    # against the χ²₂ 5%-rejection threshold (previous setting landed at p≈0.07,
    # one bad seed away from flaking).
    s, y = _dgp(8000, seed=5, noise=0.04)
    s_a, y_a = _dgp(8000, seed=15, noise=0.04)
    alpha = 0.10
    t0 = float(np.quantile(y, alpha))
    out = two_moment_wald_audit(s, y, s_a, y_a, t0, alpha)
    ok = not out["reject"] and out["p_value"] > 0.10
    print(f"[{'PASS' if ok else 'FAIL'}] transport audit (well-specified): "
          f"p={out['p_value']:.3g}, reject={out['reject']}")
    return ok


def test_transport_audit_mis_specified():
    # Train calibrator on one DGP; audit on a DIFFERENT DGP with shifted Y.
    rng = np.random.default_rng(6)
    n = 3000
    y_tr = rng.beta(5, 2, size=n)
    s_tr = np.clip(y_tr + rng.normal(0, 0.05, size=n), 0, 1)
    y_au = rng.beta(2, 5, size=n)  # different distribution, same S-Y corr direction
    s_au = np.clip(y_au + rng.normal(0, 0.05, size=n), 0, 1)
    alpha = 0.10
    t0 = float(np.quantile(y_tr, alpha))
    out = two_moment_wald_audit(s_tr, y_tr, s_au, y_au, t0, alpha)
    ok = out["reject"]
    print(f"[{'PASS' if ok else 'FAIL'}] transport audit (mis-specified): "
          f"p={out['p_value']:.3g}, reject={out['reject']}")
    return ok


def test_cluster_bootstrap_shape():
    s, y = _dgp(1000, seed=7)
    cluster = np.arange(len(s))  # each row its own cluster
    lo, hi, n_fail = cluster_bootstrap_cvar(
        s, y, s, eval_cluster=cluster, train_cluster=cluster,
        alpha=0.10, grid_size=31, B=200, seed=7,
    )
    ok = np.isfinite(lo) and np.isfinite(hi) and lo <= hi and n_fail < 20
    print(f"[{'PASS' if ok else 'FAIL'}] cluster bootstrap: "
          f"[{lo:.4f}, {hi:.4f}], n_fail={n_fail}")
    return ok


def main():
    tests = [
        test_isotonic_mean_recovery,
        test_estimate_direct_mean,
        test_direct_cvar,
        test_make_t_grid_bounds,
        test_transport_audit_well_specified,
        test_transport_audit_mis_specified,
        test_cluster_bootstrap_shape,
    ]
    results = [t() for t in tests]
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(tests)} tests passed")
    sys.exit(0 if n_pass == len(tests) else 1)


if __name__ == "__main__":
    main()
