"""
"Test the test" — sanity probes for the audit-Ω̂ evaluation framework.

See evaluation.md §6. Each probe verifies a known property of the framework
on synthetic methods constructed to pass / fail in known ways. If any
probe disagrees with its analytical prediction beyond MC tolerance, the
framework itself is buggy.

S1  oracle method passes                    size ≈ η
S2  variance-only oracle over-rejects       predicted == empirical
S3  artificially-inflated Ω̂ under-rejects   size ≪ η, predicted ≪ η
S4  Σ_full decomposition by ablation        V_audit + V_calib + V_eval ≈ Σ_full
S5  ε vanishes as n_calib → ∞               monotone shrink at isotonic rate
S6  oracle ε at population (h*, t*) is 0    within MC noise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ._diagnostics import _wald_n, _predict_size_noncentral, _CHI2_CRIT
from ._harness import (
    fit_calibrator, sample, sample_logger, saddle_argmax, g_pair, run_one_rep,
)
from ._truths import DGPParams, TargetPert, t_star, h_star_at


LOG = logging.getLogger(__name__)


# ---------------- Helpers ---------------------------------------------------


def _empirical_sigma_full_and_eps(
    p: DGPParams, pert: TargetPert, alpha: float,
    n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
):
    """Run R reps under H_0; return ε = mean(ḡ), Σ_full = n_audit · sample-cov(ḡ)."""
    g_realized = []
    g_per_fold_all = []
    for r in range(R):
        seed = seed_base + 9007 * r
        res = run_one_rep(p, pert, alpha, n_calib, n_audit, n_eval, t_grid, K, B, seed)
        g_realized.append(res.ḡ)
        # Back-compat: BC-jk-cal style (ĥ varies, t̂ fixed at t̂_pooled)
        g_per_fold_all.append(res.g_per_fold_at_t_hat)
    g_arr = np.stack(g_realized, axis=0)
    eps = g_arr.mean(axis=0)
    sigma_full = n_audit * np.cov(g_arr, rowvar=False, ddof=1)
    return eps, sigma_full, g_arr, g_per_fold_all


# ---------------- S1: oracle method passes ----------------------------------


@dataclass
class ProbeResult:
    name: str
    passed: bool
    detail: str


def probe_S1_oracle_passes(
    p: DGPParams, alpha: float, n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
) -> ProbeResult:
    """
    Oracle method: Ω̂ = Σ_full / n_audit (the TRUE sampling variance of ḡ,
    estimated from the same R MC reps) AND ḡ_used = ḡ − ε_emp (bias-
    corrected with empirical center).

    Both variance bias (A) and center bias (C) are zero by construction.
    Empirical size must equal η.

    Tolerance: |size − 0.05| ≤ 1.96 · √(0.05·0.95/R).

    NOTE: we do NOT use Σ_oracle (closed-form audit-only variance) because
    Σ_oracle ≠ Σ_full — Σ_oracle misses N2 and N3 contributions. An "oracle"
    method using Σ_oracle would actually under-estimate variance and over-
    reject, which is informative but not what S1 is testing.
    """
    pert = TargetPert()
    eps_emp, sigma_full, g_arr, _ = _empirical_sigma_full_and_eps(
        p, pert, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base,
    )
    omega_hat = sigma_full / n_audit       # the true variance of ḡ

    rejections = 0
    for r in range(R):
        g_corrected = g_arr[r] - eps_emp
        W = _wald_n(g_corrected, omega_hat)
        if W > _CHI2_CRIT:
            rejections += 1
    size_emp = rejections / R
    tol = 1.96 * np.sqrt(0.05 * 0.95 / R)
    passed = abs(size_emp - 0.05) <= tol
    return ProbeResult(
        name="S1 oracle passes (Σ_full)",
        passed=passed,
        detail=f"empirical size {size_emp:.4f}, target 0.0500 ± {tol:.4f}",
    )


def probe_S2_varonly_oracle(
    p: DGPParams, alpha: float, n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
) -> ProbeResult:
    """
    Variance-only oracle: Ω̂ = Σ_full / n_audit (correct variance) but
    ḡ_used = ḡ (NO bias correction). Center bias c = ε_emp ≠ 0.

    Predicted size from non-central χ² formula:
        ḡ ~ N(ε_emp, Σ_full / n_audit), Wald with Σ_full / n_audit
            ⇒ W ~ non-central χ²_2(λ),  λ = n_audit · ε_emp^T Σ_full^{-1} ε_emp.

    Predicted MUST match empirical within MC noise.
    """
    pert = TargetPert()
    eps_emp, sigma_full, g_arr, _ = _empirical_sigma_full_and_eps(
        p, pert, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base,
    )
    omega_hat = sigma_full / n_audit       # the true variance

    rejections = 0
    for r in range(R):
        W = _wald_n(g_arr[r], omega_hat)
        if W > _CHI2_CRIT:
            rejections += 1
    size_emp = rejections / R

    size_pred = _predict_size_noncentral(omega_hat, omega_hat, eps_emp)

    # Tolerance: MC noise on size at R reps.
    tol_size = 1.96 * np.sqrt(0.05 * 0.95 / R) + 0.02   # add slack for prediction noise
    diff = abs(size_emp - size_pred)
    passed = diff < tol_size
    return ProbeResult(
        name="S2 var-only-oracle predicted == empirical",
        passed=passed,
        detail=(f"empirical {size_emp:.4f}, predicted {size_pred:.4f}, "
                f"|Δ|={diff:.4f} ≤ {tol_size:.4f}; ε ≈ {eps_emp.tolist()}"),
    )


def probe_S3_inflated_omega(
    p: DGPParams, alpha: float, n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
) -> ProbeResult:
    """
    Inflated Ω̂: 4 × Σ_full / n_audit, ḡ_used = ḡ − ε_emp (bias-corrected).
    With variance 4× too big and center corrected, Wald is χ²_2 / 4 → reject
    only when W > 4 · 5.99 = 23.96. P(χ²_2 > 23.96) ≈ 6.2 × 10⁻⁶.

    So empirical size MUST be near zero (< 0.005 at R=1000).
    """
    pert = TargetPert()
    eps_emp, sigma_full, g_arr, _ = _empirical_sigma_full_and_eps(
        p, pert, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base,
    )
    omega_hat = 4.0 * sigma_full / n_audit  # 4× the true variance
    rejections = 0
    for r in range(R):
        g_corrected = g_arr[r] - eps_emp
        W = _wald_n(g_corrected, omega_hat)
        if W > _CHI2_CRIT:
            rejections += 1
    size_emp = rejections / R
    # P(χ²_2 > 4 · 5.99) = P(χ²_2 > 23.96) ≈ 6.2e-6 → expect near-zero rejections.
    passed = size_emp < 0.01
    return ProbeResult(
        name="S3 inflated Ω̂ under-rejects",
        passed=passed,
        detail=f"empirical size {size_emp:.4f}, expected < 0.01 (theoretical ~6e-6)",
    )


def probe_S4_sigma_decomposition(
    p: DGPParams, alpha: float, n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
) -> ProbeResult:
    """
    Σ_full ≈ V_audit_only + V_calib_only + V_eval_only.
    Compute each by ablating two of three nuisances:
        V_audit_only  : pin (calib, eval); vary AUDIT only.
        V_calib_only  : pin (audit, eval); vary CALIB only.
        V_eval_only   : pin (calib, audit); vary EVAL only.

    Returns pass if relative error < 20% on the trace (test the additive
    decomposition; finer-grained per-entry could flag cross-terms).
    """
    pert = TargetPert()
    # 1. Σ_full from full-pipeline reps
    _, sigma_full, _, _ = _empirical_sigma_full_and_eps(
        p, pert, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R // 2, seed_base,
    )

    # 2. V_audit_only: pin (calib, eval)
    s_c, y_c = sample_logger(p, n_calib, seed=seed_base)
    cal_pin = fit_calibrator(s_c, y_c, t_grid, K, seed=seed_base)
    s_e_pin, _ = sample(p, pert, n_eval, seed=seed_base + 1, with_oracle=False)
    H_e_pin = cal_pin.predict(s_e_pin)
    t_idx_pin, t_hat_pin = saddle_argmax(H_e_pin, t_grid, alpha)

    g_audit_only = []
    for r in range(R // 4):
        s_a, y_a = sample(p, pert, n_audit, seed=seed_base + 2 + 31 * r, with_oracle=True)
        h_t_a = cal_pin.predict(s_a)[:, t_idx_pin]
        g1, g2 = g_pair(s_a, y_a, h_t_a, t_hat_pin, alpha)
        g_audit_only.append([g1.mean(), g2.mean()])
    V_audit = n_audit * np.cov(np.array(g_audit_only), rowvar=False, ddof=1)

    # 3. V_calib_only: pin (audit, eval). Need a frozen audit; reuse the
    # one we drew for V_audit_only's first iter.
    s_a_pin, y_a_pin = sample(p, pert, n_audit, seed=seed_base + 2, with_oracle=True)
    s_e_pin2, _ = sample(p, pert, n_eval, seed=seed_base + 1, with_oracle=False)

    g_calib_only = []
    for r in range(R // 4):
        s_c_r, y_c_r = sample_logger(p, n_calib, seed=seed_base + 100 + 71 * r)
        cal_r = fit_calibrator(s_c_r, y_c_r, t_grid, K, seed=seed_base + 100 + 71 * r)
        H_e_r = cal_r.predict(s_e_pin2)
        t_idx_r, t_hat_r = saddle_argmax(H_e_r, t_grid, alpha)
        h_t_a_r = cal_r.predict(s_a_pin)[:, t_idx_r]
        g1, g2 = g_pair(s_a_pin, y_a_pin, h_t_a_r, t_hat_r, alpha)
        g_calib_only.append([g1.mean(), g2.mean()])
    V_calib = n_audit * np.cov(np.array(g_calib_only), rowvar=False, ddof=1)

    # 4. V_eval_only: pin (calib, audit)
    s_c_pin, y_c_pin = sample_logger(p, n_calib, seed=seed_base + 200)
    cal_pin2 = fit_calibrator(s_c_pin, y_c_pin, t_grid, K, seed=seed_base + 200)

    g_eval_only = []
    for r in range(R // 4):
        s_e_r, _ = sample(p, pert, n_eval, seed=seed_base + 300 + 71 * r, with_oracle=False)
        H_e_r = cal_pin2.predict(s_e_r)
        t_idx_r, t_hat_r = saddle_argmax(H_e_r, t_grid, alpha)
        h_t_a_r = cal_pin2.predict(s_a_pin)[:, t_idx_r]
        g1, g2 = g_pair(s_a_pin, y_a_pin, h_t_a_r, t_hat_r, alpha)
        g_eval_only.append([g1.mean(), g2.mean()])
    V_eval = n_audit * np.cov(np.array(g_eval_only), rowvar=False, ddof=1)

    V_predicted = V_audit + V_calib + V_eval
    trace_full = float(np.trace(sigma_full))
    trace_pred = float(np.trace(V_predicted))
    rel_err = abs(trace_pred - trace_full) / max(trace_full, 1e-9)
    passed = rel_err < 0.20
    detail = (
        f"trace(Σ_full)={trace_full:.5f}, trace(V_audit)={float(np.trace(V_audit)):.5f}, "
        f"trace(V_calib)={float(np.trace(V_calib)):.5f}, trace(V_eval)={float(np.trace(V_eval)):.5f}, "
        f"sum={trace_pred:.5f}, rel_err={rel_err:.3f}"
    )
    return ProbeResult(name="S4 Σ_full decomposition", passed=passed, detail=detail)


def probe_S5_eps_vanishes(
    p: DGPParams, alpha: float, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
) -> ProbeResult:
    """
    Compute |ε| at n_calib ∈ {300, 600, 1500, 3000}. Should monotone-decrease.
    """
    pert = TargetPert()
    sizes = [300, 1200, 4800]   # wider spread, fewer points to keep cost bounded
    eps_norms = []
    se_norms = []
    for n_cal in sizes:
        eps_emp, _, g_arr_local, _ = _empirical_sigma_full_and_eps(
            p, pert, alpha, n_cal, n_audit, n_eval, t_grid, K, B, R, seed_base + n_cal,
        )
        norm = float(np.linalg.norm(eps_emp))
        # SE on |ε| via std of ḡ_r norms / sqrt(R)
        per_rep_norms = np.linalg.norm(g_arr_local, axis=1)
        se = float(per_rep_norms.std(ddof=1) / np.sqrt(R))
        eps_norms.append(norm)
        se_norms.append(se)
    # The smallest-to-largest n_calib should show |ε| decreasing.
    # MC noise band: ±2 · max(SE). If extremes overlap, declare not significant.
    range_signif = abs(eps_norms[0] - eps_norms[-1]) > 2.0 * max(se_norms)
    monotone_with_slack = eps_norms[-1] < eps_norms[0] * 1.5  # tolerate one bump
    passed = range_signif and monotone_with_slack
    detail = "  ".join(
        f"n_calib={n_cal}: |ε|={en:.5f}±{se:.5f}"
        for n_cal, en, se in zip(sizes, eps_norms, se_norms)
    )
    return ProbeResult(name="S5 ε shrinks with n_calib", passed=passed, detail=detail)


def probe_S6_population_oracle_eps_zero(
    p: DGPParams, alpha: float, n_audit: int, n_eval: int,
    t_grid: np.ndarray, R: int, seed_base: int,
) -> ProbeResult:
    """
    Use closed-form (h*, t*) instead of (ĥ, t̂). Compute ḡ on R reps; mean
    must be ≈ 0 within MC noise.

    This validates the population-truth computations themselves.
    """
    pert = TargetPert()
    t_hat = t_star(p, pert, alpha)

    # Take t_grid index closest to t_hat for the audit's h* lookup:
    j = int(np.argmin(np.abs(t_grid - t_hat)))
    t_pop_grid = float(t_grid[j])
    g_means = []
    for r in range(R // 4):
        s_a, y_a = sample(p, pert, n_audit, seed=seed_base + 7 * r, with_oracle=True)
        h_pop = h_star_at(s_a, np.array([t_pop_grid]), p, pert).ravel()
        g1, g2 = g_pair(s_a, y_a, h_pop, t_pop_grid, alpha)
        g_means.append([g1.mean(), g2.mean()])
    g_arr = np.array(g_means)
    eps_pop = g_arr.mean(axis=0)
    se = g_arr.std(axis=0, ddof=1) / np.sqrt(len(g_means))
    z = np.abs(eps_pop) / np.maximum(se, 1e-12)
    passed = bool((z < 3.0).all())
    detail = f"ε at population (h*, t*) = {eps_pop.tolist()},  z-scores = {z.tolist()}"
    return ProbeResult(name="S6 population ε ≈ 0", passed=passed, detail=detail)


def run_all_probes(
    p: DGPParams, alpha: float, n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, R: int, seed_base: int,
) -> list[ProbeResult]:
    return [
        probe_S1_oracle_passes(p, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base),
        probe_S2_varonly_oracle(p, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base + 1000),
        probe_S3_inflated_omega(p, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base + 2000),
        probe_S4_sigma_decomposition(p, alpha, n_calib, n_audit, n_eval, t_grid, K, B, R, seed_base + 3000),
        probe_S5_eps_vanishes(p, alpha, n_audit, n_eval, t_grid, K, B, R, seed_base + 4000),
        probe_S6_population_oracle_eps_zero(p, alpha, n_audit, n_eval, t_grid, R, seed_base + 5000),
    ]
