"""
Test: does cross-fitting INSIDE the bootstrap of full_pipeline_boot
matter for the variance estimate?

Setup
-----
Two variants of full_pipeline_boot, run on the SAME outer reps and
the SAME bootstrap RNG seeds:

    K_in_boot = 5    : refit cross-fitted calibrator (K=5) per bootstrap rep
                       (the current full_pipeline_boot)

    K_in_boot = 1    : refit ONLY the pooled isotonic per t in T_grid
                       (no fold partition, no per-fold fits)
                       Cheaper by ~K×.

Comparison
----------
Per outer rep r, both variants produce a 2x2 Ω̂_M^(r). We compare:
    1.  ratio of trace : trace(Ω̂_K=5) / trace(Ω̂_K=1)  per rep
    2.  ratio of Frobenius norms
    3.  size on the same set of ḡ^(r) values

If K=1 is statistically indistinguishable from K=5 inside the bootstrap,
we can drop cross-fit and 5× the bootstrap loop.

Math (K=1 variant)
------------------
For b = 1..B_full:
    idx_c^(b)  ~ Multinomial(n_calib, 1/n_calib · 1)
    ĥ_t^(b)   :=  IsotonicRegression(decreasing).fit(s_c[idx_c^(b)], z_t)
                  for each t in T_grid       — POOLED ONLY, no fold partition
    (rest of the loop identical: re-max t̂^(b), recompute ḡ^(b))

Ω̂_K=1 := sample-cov_b(ḡ^(b)).

This is what the cvar_v4 paper algorithm 1 actually does — it doesn't
specify cross-fitting inside the bootstrap.

Run
---
    python -m cvar_v5._archive.audit_evaluation._compare_kfold_in_boot
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from ._harness import (
    Calibrator, fit_calibrator, g_pair, run_one_rep, saddle_argmax,
)
from ._truths import DGPParams, TargetPert


_CHI2_CRIT = float(stats.chi2.ppf(0.95, df=2))


def _fit_pooled_only(s_calib, y_calib, t_grid):
    """
    Fit ONLY the pooled isotonic per t (no cross-fit folds).

    Returns a list of IsotonicRegression objects, one per t in t_grid.
    Cheaper than fit_calibrator by ~K× because we skip the K per-fold fits.
    """
    pooled = []
    for t in t_grid:
        z = np.maximum(t - y_calib, 0.0)
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            s_calib, z,
        )
        pooled.append(ir)
    return pooled


def _predict_pooled(pooled_list, s):
    s = np.asarray(s).ravel()
    out = np.empty((len(s), len(pooled_list)), dtype=np.float64)
    for j, ir in enumerate(pooled_list):
        out[:, j] = ir.predict(s)
    return out


def full_pipeline_boot_k1(
    rep, t_grid, alpha, B_full, seed,
):
    """
    Same as full_pipeline_boot but uses K=1 (pooled only) inside the
    bootstrap. Math identical except ĥ^(b) is the pooled fit, not a
    cross-fitted Calibrator.
    """
    n_c = len(rep.s_calib)
    n_e = len(rep.s_eval)
    n_a = len(rep.s_audit)
    rng = np.random.default_rng(seed)
    g_b_per_rep = np.empty((B_full, 2), dtype=np.float64)

    for b in range(B_full):
        idx_c = rng.integers(0, n_c, size=n_c)
        idx_e = rng.integers(0, n_e, size=n_e)
        idx_a = rng.integers(0, n_a, size=n_a)

        s_c_b = rep.s_calib[idx_c]
        y_c_b = rep.y_calib[idx_c]
        s_e_b = rep.s_eval[idx_e]
        s_a_b = rep.s_audit[idx_a]
        y_a_b = rep.y_audit[idx_a]

        # Pooled-only fit (no folds).
        pooled_b = _fit_pooled_only(s_c_b, y_c_b, t_grid)

        H_e_b = _predict_pooled(pooled_b, s_e_b)
        j_b, t_b = saddle_argmax(H_e_b, t_grid, alpha)

        H_a_b = _predict_pooled(pooled_b, s_a_b)
        h_t_a_b = H_a_b[:, j_b]
        g1_b, g2_b = g_pair(s_a_b, y_a_b, h_t_a_b, t_b, alpha)
        g_b_per_rep[b] = [g1_b.mean(), g2_b.mean()]

    return np.cov(g_b_per_rep, rowvar=False, ddof=1)


def main():
    p = DGPParams(a=1.0, b=1.0)
    pert = TargetPert()
    alpha = 0.10
    n_calib, n_audit, n_eval = 600, 250, 1000
    K = 5
    B = 100  # for inner Ω̂_audit bootstrap (boot_remax)
    B_full = 80
    R = 60   # outer reps (cheap; we want a precise comparison, not new info)
    t_grid = np.linspace(0.0, 1.0, 121)

    omegas_k5, omegas_k1 = [], []
    g_realized = []
    t0 = time.time()
    for r in range(R):
        seed = 20260503 + 9007 * r
        rep = run_one_rep(p, pert, alpha, n_calib, n_audit, n_eval,
                          t_grid, K, B, seed)
        # Use the SAME bootstrap seed for both variants.
        from ._variance_methods_extra import full_pipeline_boot
        om_k5 = full_pipeline_boot(rep, t_grid, alpha, K, B_full, seed=seed + 13)
        om_k1 = full_pipeline_boot_k1(rep, t_grid, alpha, B_full, seed=seed + 13)
        omegas_k5.append(om_k5)
        omegas_k1.append(om_k1)
        g_realized.append(rep.ḡ)
        if (r + 1) % max(1, R // 6) == 0:
            print(f"  rep {r+1}/{R}  ({time.time()-t0:.1f}s)")

    omegas_k5 = np.stack(omegas_k5, axis=0)
    omegas_k1 = np.stack(omegas_k1, axis=0)
    g_arr = np.stack(g_realized, axis=0)

    # ---- Σ_full reference (the truth) ----
    sigma_full = n_audit * np.cov(g_arr, rowvar=False, ddof=1)

    # ---- Per-rep ratios ----
    print()
    print(f"R = {R}")
    print(f"trace(Σ_full / n_audit) (truth proxy) = {np.trace(sigma_full)/n_audit:.5e}")
    print()

    print("Per-rep mean Ω̂ vs truth:")
    print(f"  trace(mean Ω̂_K=5) = {np.trace(omegas_k5.mean(axis=0)):.5e}")
    print(f"  trace(mean Ω̂_K=1) = {np.trace(omegas_k1.mean(axis=0)):.5e}")
    print(f"  trace(Σ_full)/n_audit = {np.trace(sigma_full)/n_audit:.5e}")
    print()

    # ---- Per-rep ratio: K=1 / K=5 ----
    trace_k5 = np.trace(omegas_k5, axis1=1, axis2=2)
    trace_k1 = np.trace(omegas_k1, axis1=1, axis2=2)
    ratios = trace_k1 / np.maximum(trace_k5, 1e-30)
    print(f"Per-rep ratio  trace(Ω̂_K=1) / trace(Ω̂_K=5):")
    print(f"  mean = {ratios.mean():.4f}")
    print(f"  std  = {ratios.std(ddof=1):.4f}")
    print(f"  q05/q50/q95 = {np.quantile(ratios, 0.05):.4f} / "
          f"{np.quantile(ratios, 0.50):.4f} / "
          f"{np.quantile(ratios, 0.95):.4f}")
    print()

    # ---- Diagonal entries individually ----
    print("Per-rep ratio of [g_1, g_1] entry  K=1 / K=5:")
    r11 = omegas_k1[:, 0, 0] / np.maximum(omegas_k5[:, 0, 0], 1e-30)
    print(f"  mean = {r11.mean():.4f}, std = {r11.std(ddof=1):.4f}")
    print()
    print("Per-rep ratio of [g_2, g_2] entry  K=1 / K=5:")
    r22 = omegas_k1[:, 1, 1] / np.maximum(omegas_k5[:, 1, 1], 1e-30)
    print(f"  mean = {r22.mean():.4f}, std = {r22.std(ddof=1):.4f}")
    print()

    # ---- Empirical size with each ----
    rejects_k5 = sum(
        1 for r in range(R)
        if g_realized[r] @ np.linalg.pinv(omegas_k5[r]) @ g_realized[r] > _CHI2_CRIT
    )
    rejects_k1 = sum(
        1 for r in range(R)
        if g_realized[r] @ np.linalg.pinv(omegas_k1[r]) @ g_realized[r] > _CHI2_CRIT
    )
    print(f"Empirical size at η=0.05:")
    print(f"  K=5 inside boot:  {rejects_k5/R:.3f}  ({rejects_k5}/{R})")
    print(f"  K=1 inside boot:  {rejects_k1/R:.3f}  ({rejects_k1}/{R})")


if __name__ == "__main__":
    main()
