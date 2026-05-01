"""Regression tests for the α=1 identity.

At α=1, CVaR_α(Y) = E[Y]. The Direct CVaR-CJE estimator should reduce to
Direct Mean-CJE algebraically: when t̂ ≥ Y_max, every (t̂ − Y) is positive,
the isotonic stop-loss ĝ_t̂(s) collapses to (t̂ − f̂(s)), and the
saddle-point becomes E[f̂(s_eval)] — exactly the Direct Mean estimate.

Tests run against real n=500 HealthBench data:
  1. Atom-split CVaR_1(Y) == E[Y] exactly
  2. cvar_est(α=1) == mean_cje_est at machine precision
  3. mean_g2(α=1) == −mean_audit_residual at machine precision
  4. mean_g1(α=1) == 0 exactly (when t̂ covers Y_max)
  5. t̂(α=1) ≥ max(y_train) for every policy (grid extension works)
  6. Bootstrap CI on g2(α=1) matches bootstrap CI on −mean_residual

If any test breaks, suspect a regression in:
  - eda/deeper/_estimator.py: make_t_grid, fit_isotonic_*, simple_cvar_audit,
    mean_transport_audit, estimate_direct_cvar_isotonic, _bootstrap_g_vectors
  - healthbench_data/analyze.py: step5_oracle_calibrated_uniform

Run as a script (no pytest needed):
    python -m cvar_v4.healthbench_data.tests.test_alpha1_identity

Run via pytest:
    pytest cvar_v4/healthbench_data/tests/test_alpha1_identity.py
"""
from __future__ import annotations

import sys

import numpy as np

from ..analyze import (
    _load_judge_scores, _logger_panel, _policy_panel, logger_policy,
    cvar_alpha, step5_oracle_calibrated_uniform,
)
from ..oracle_design import select_slice
from ...eda.deeper._estimator import (
    _bootstrap_g_vectors,
    bootstrap_cvar_ci,
    bootstrap_mean_ci,
    cvar_audit_analytical_se,
    estimate_direct_cvar_isotonic,
    fit_isotonic_mean,
    fit_isotonic_tail_loss,
    jackknife_var_cal,
    jackknife_var_cal_mean,
    make_t_grid,
    mean_transport_audit,
    two_moment_wald_audit_xf,
)


# Tight tolerance for algebraic identities; loose for bootstrap MC error.
TOL_EXACT = 1e-12
TOL_BOOT_CI_WIDTH_RATIO = 0.05    # bootstrap CI widths within 5%
TOL_BOOT_CORR = 0.99              # per-rep correlation for g2 vs -ε


def _build_panels():
    """Replicate the slice + cal/audit split used by step5_oracle_calibrated_uniform.
    Returns (s_train, y_train, w_train) and a function that yields per-policy
    (s_target, y_target).
    """
    s_log_full, y_log_full = _logger_panel()
    logger = logger_policy()
    log_pids = sorted(
        set(_load_judge_scores(logger.name, "cheap"))
        & set(_load_judge_scores(logger.name, "oracle"))
    )
    log_rows = [
        {"prompt_id": pid, "policy": logger.name,
         "cheap_score": float(s_log_full[i]), "oracle_score": float(y_log_full[i])}
        for i, pid in enumerate(log_pids)
    ]
    log_slice = select_slice(log_rows, design="uniform", coverage=0.25,
                              alpha=1.0, seed=42)
    sel_mask = np.array([s.selected for s in log_slice])
    sel_pi = np.array([s.pi for s in log_slice])
    sel_idx = np.where(sel_mask)[0]
    n_sel = len(sel_idx)
    rng = np.random.default_rng(42 + 991)
    perm = rng.permutation(n_sel)
    n_cal = max(3, int(round(0.8 * n_sel)))
    cal_idx = sel_idx[perm[:n_cal]]
    s_train = s_log_full[cal_idx]
    y_train = y_log_full[cal_idx]
    w_train = 1.0 / sel_pi[cal_idx]

    def get_target(name: str):
        pids, s_target, y_target = _policy_panel(name)
        return s_target, y_target

    return s_train, y_train, w_train, get_target


def test_alpha1_truth_equals_mean():
    """Atom-split CVaR_1(Y) must equal E[Y] for every policy."""
    rows = step5_oracle_calibrated_uniform(coverage=0.25, alpha=1.0,
                                            seed=42, verbose=False)
    assert rows, "no rows produced"
    for r in rows:
        assert abs(r["full_oracle_truth"] - r["mean_Y"]) < TOL_EXACT, (
            f"{r['policy']}: atom-split CVaR_1 = {r['full_oracle_truth']} "
            f"!= mean_Y = {r['mean_Y']}"
        )


def test_alpha1_estimator_equals_mean_cje():
    """cvar_est at α=1 must equal mean_cje_est at machine precision."""
    rows = step5_oracle_calibrated_uniform(coverage=0.25, alpha=1.0,
                                            seed=42, verbose=False)
    for r in rows:
        diff = abs(r["cvar_est"] - r["mean_cje_est"])
        assert diff < TOL_EXACT, (
            f"{r['policy']}: cvar_est={r['cvar_est']} mean_cje_est={r['mean_cje_est']} "
            f"diff={diff:.2e}"
        )


def test_alpha1_g2_equals_neg_mean_residual():
    """mean_g2 at α=1 must equal -mean_audit_residual at machine precision.

    Identity: when t̂ ≥ Y_max(audit), (t̂-Y)_+ = t̂-Y for every audit row,
    and ĝ_t̂(s) = t̂ - f̂(s), so g2 = -(Y - f̂(s)) = -ε.
    """
    rows = step5_oracle_calibrated_uniform(coverage=0.25, alpha=1.0,
                                            seed=42, verbose=False)
    for r in rows:
        diff = abs(r["mean_g2"] - (-r["mean_audit_residual"]))
        assert diff < TOL_EXACT, (
            f"{r['policy']}: mean_g2={r['mean_g2']} "
            f"-mean_residual={-r['mean_audit_residual']} diff={diff:.2e}"
        )


def test_alpha1_g1_is_zero():
    """mean_g1 = 1{Y≤t̂} - 1 = 0 when every audit row satisfies Y ≤ t̂."""
    rows = step5_oracle_calibrated_uniform(coverage=0.25, alpha=1.0,
                                            seed=42, verbose=False)
    for r in rows:
        assert abs(r["mean_g1"]) < TOL_EXACT, (
            f"{r['policy']}: mean_g1={r['mean_g1']} != 0 "
            f"(t̂={r['t_hat']:+.4f}; some audit row has Y > t̂)"
        )


def test_alpha1_t_hat_covers_y_train_max():
    """Grid extension must produce t̂ ≥ max(y_train) at α=1."""
    s_train, y_train, w_train, _ = _build_panels()
    grid = make_t_grid(y_train, alpha=1.0)
    y_train_max = float(y_train.max())
    assert grid.max() >= y_train_max + 1e-6, (
        f"grid_max={grid.max():+.4f} does not exceed y_train_max={y_train_max:+.4f}; "
        f"the α=1 saddle-point optimum cannot be reached."
    )


def test_low_alpha_grid_unchanged():
    """For α=0.10, the first grid_size points must reproduce the legacy linspace
    exactly. Headline production numbers depend on this.
    """
    _, y_train, _, _ = _build_panels()
    legacy_t_lo = float(np.quantile(y_train, max(0.001, 0.10 / 5.0)) - 0.60)
    legacy_t_hi = float(np.quantile(y_train, min(0.60, 0.10 + 0.45)) + 0.35)
    legacy = np.linspace(legacy_t_lo, legacy_t_hi, 61)
    new = make_t_grid(y_train, alpha=0.10, grid_size=61)
    np.testing.assert_allclose(new[:61], legacy, rtol=0, atol=TOL_EXACT)


def test_alpha1_bootstrap_g2_matches_mean_residual():
    """Bootstrap CI on g2 at α=1 should match bootstrap CI on -mean_residual,
    using the same paired bootstrap design.
    """
    s_train, y_train, w_train, get_target = _build_panels()
    s_target, y_target = get_target("premium")  # highest Y_max stress test
    B = 200
    seed = 42

    g_boot = _bootstrap_g_vectors(
        s_train, y_train, s_target, y_target, alpha=1.0,
        sample_weight_train=w_train, sample_weight_audit=None,
        B=B, fold_seed=seed,
    )
    g2_boot = g_boot[:, 1]

    # Recompute mean residual on each bootstrap rep with the same RNG path
    rng = np.random.default_rng(seed)
    n_t, n_a = len(s_train), len(s_target)
    eps_boot = np.empty(B)
    for b in range(B):
        idx_t = rng.integers(0, n_t, size=n_t)
        idx_a = rng.integers(0, n_a, size=n_a)
        f_hat = fit_isotonic_mean(
            s_train[idx_t], y_train[idx_t], s_target[idx_a],
            sample_weight=w_train[idx_t],
        )
        eps_boot[b] = float((y_target[idx_a] - f_hat).mean())

    corr = float(np.corrcoef(g2_boot, -eps_boot)[0, 1])
    assert corr > TOL_BOOT_CORR, (
        f"per-rep correlation cor(g2, -ε) = {corr:.4f} below {TOL_BOOT_CORR}"
    )

    g2_lo, g2_hi = np.percentile(g2_boot, [2.5, 97.5])
    e_lo, e_hi = np.percentile(-eps_boot, [2.5, 97.5])
    g2_w = g2_hi - g2_lo
    e_w = e_hi - e_lo
    rel = abs(g2_w - e_w) / max(g2_w, e_w, 1e-12)
    assert rel < TOL_BOOT_CI_WIDTH_RATIO, (
        f"bootstrap CI width disagreement: g2={g2_w:.4f}, -ε={e_w:.4f}, "
        f"rel={rel:.4f} > {TOL_BOOT_CI_WIDTH_RATIO}"
    )


def test_alpha1_isotonic_collapse():
    """Algebraic identity: at any t > max(y_train), the isotonic of (t - y) on s
    equals (t - isotonic of y on s), to machine precision. This is the core
    reason the saddle-point reduces to Mean-CJE at α=1.
    """
    s_train, y_train, w_train, get_target = _build_panels()
    _, s_target = get_target("premium")[0], get_target("premium")[0]
    t = float(y_train.max()) + 1e-9
    z = np.maximum(t - y_train, 0.0)
    g_hat = fit_isotonic_tail_loss(s_train, z, s_target, sample_weight=w_train)
    f_hat = fit_isotonic_mean(s_train, y_train, s_target, sample_weight=w_train)
    implied = t - f_hat
    err = float(np.abs(g_hat - implied).max())
    assert err < TOL_EXACT, (
        f"max |g_hat(s) - (t - f_hat(s))| = {err:.2e} at t = y_max + ε"
    )


def test_alpha1_bootstrap_point_estimate_ci_agreement():
    """bootstrap_cvar_ci(α=1) and bootstrap_mean_ci must produce per-rep
    identical bootstrap distributions of the point estimate (max |Δ| < 1e-12)
    when given identical (seed, B, sample_weight, s_eval).
    """
    s_train, y_train, w_train, get_target = _build_panels()
    s_target, _ = get_target("premium")
    B = 200
    seed = 42
    cv = bootstrap_cvar_ci(
        s_train, y_train, s_target, alpha=1.0,
        sample_weight_train=w_train, B=B, seed=seed,
    )
    mn = bootstrap_mean_ci(
        s_train, y_train, s_target,
        sample_weight_train=w_train, B=B, seed=seed,
    )
    diff = float(np.abs(cv["boots"] - mn["boots"]).max())
    assert diff < TOL_EXACT, (
        f"max |Δ| per rep = {diff:.2e} (cvar α=1 bootstrap vs mean bootstrap)"
    )
    point_diff = abs(cv["point"] - mn["point"])
    assert point_diff < TOL_EXACT, f"point estimate diff = {point_diff:.2e}"
    ci_diff = max(abs(cv["ci_lo"] - mn["ci_lo"]), abs(cv["ci_hi"] - mn["ci_hi"]))
    assert ci_diff < TOL_EXACT, f"95% CI diff = {ci_diff:.2e}"


def test_alpha1_var_cal_agreement():
    """jackknife_var_cal(α=1) and jackknife_var_cal_mean must produce
    identical Var_cal scalars to machine precision (same K folds via the
    same seeded permutation, same fold-id rule).
    """
    s_train, y_train, w_train, get_target = _build_panels()
    s_target, _ = get_target("premium")
    K, seed = 5, 42
    v_cvar = jackknife_var_cal(
        s_train, y_train, s_target, alpha=1.0,
        sample_weight_oracle=w_train, K=K, seed=seed,
    )
    v_mean = jackknife_var_cal_mean(
        s_train, y_train, s_target,
        sample_weight_oracle=w_train, K=K, seed=seed,
    )
    diff = abs(v_cvar - v_mean)
    assert diff < TOL_EXACT, (
        f"Var_cal disagreement: cvar={v_cvar:.6e}, mean={v_mean:.6e}, "
        f"|Δ|={diff:.2e}"
    )


def test_alpha1_audit_pvalue_agreement():
    """At α=1 with t̂ fixed at y_train_max+ε (so the saddle-point identity
    g2 ≡ −ε holds row-by-row), a paired-bootstrap Wald on g2 should match
    the mean t-test on residuals to within Monte Carlo noise.

    Notably this is NOT what `two_moment_wald_audit_xf` does — that
    function re-maximizes t̂ per bootstrap rep, which adds argmax
    variance the mean t-test doesn't see. So we run an inline fixed-t̂
    bootstrap here to make the comparison apples-to-apples.
    """
    from scipy import stats as scipy_stats
    s_train, y_train, w_train, get_target = _build_panels()
    s_target, y_target = get_target("premium")
    n_full = len(s_target)
    rng_audit = np.random.default_rng(123)
    a_idx = rng_audit.choice(n_full, size=min(132, n_full), replace=False)
    s_audit = s_target[a_idx]
    y_audit = y_target[a_idx]

    t_hat = float(y_train.max()) + 0.05  # ensures z_train has no clipping
    B, seed = 600, 42

    # Inline fixed-t̂ bootstrap of g2-bar
    rng = np.random.default_rng(seed)
    n_t, n_a = len(s_train), len(s_audit)
    g2_boot = np.empty(B)
    eps_boot = np.empty(B)
    for b in range(B):
        idx_t = rng.integers(0, n_t, size=n_t)
        idx_a = rng.integers(0, n_a, size=n_a)
        sw_b = w_train[idx_t]
        z_b = np.maximum(t_hat - y_train[idx_t], 0.0)
        g_pred = fit_isotonic_tail_loss(
            s_train[idx_t], z_b, s_audit[idx_a], sample_weight=sw_b,
        )
        g2_b = np.maximum(t_hat - y_audit[idx_a], 0.0) - g_pred
        g2_boot[b] = float(g2_b.mean())
        f_pred = fit_isotonic_mean(
            s_train[idx_t], y_train[idx_t], s_audit[idx_a], sample_weight=sw_b,
        )
        eps_boot[b] = float((y_audit[idx_a] - f_pred).mean())

    # The two bootstrap distributions should be sign-flipped images of
    # each other (g2 = -ε per row, hence per-rep). Verify identity.
    per_rep_diff = float(np.abs(g2_boot + eps_boot).max())
    assert per_rep_diff < 1e-10, (
        f"per-rep g2 + ε disagreement = {per_rep_diff:.2e} (expected ≈0)"
    )

    # Wald-style two-sided z² test using the fixed-t̂ bootstrap variance
    g2_bar_point = float(g2_boot.mean())  # plug-in of the bootstrap mean
    se_g2 = float(g2_boot.std(ddof=1))
    z2 = (g2_bar_point / se_g2) ** 2 if se_g2 > 0 else 0.0
    p_wald = float(1.0 - scipy_stats.chi2.cdf(z2, df=1))

    mean_aud = mean_transport_audit(
        s_train, y_train, s_audit, y_audit,
        sample_weight_train=w_train,
    )
    p_t = mean_aud["p_value"]

    # Allow modest MC tolerance (B=600). Both should land in the same
    # significance regime.
    diff = abs(p_wald - p_t)
    assert diff < 0.05, (
        f"audit p-value disagreement at fixed t̂: bootstrap-Wald={p_wald:.4f}, "
        f"mean t-test={p_t:.4f}, |Δ|={diff:.4f}"
    )


# ---------------------------------------------------------------------------
# Production-α (α = 0.10) variance-estimator agreement tests
# ---------------------------------------------------------------------------

# These tests run at the production headline α and check that the variance
# estimators we have available are not wildly inconsistent. The α=1 tests
# above lock the *math* (Mean ≡ CVaR collapse). These tests check that the
# *numerical methods* — bootstrap vs jackknife for the estimator, analytical
# vs bootstrap for the audit — agree well enough to use interchangeably in
# the writeup.

PROD_ALPHA = 0.10
TOL_PROD_ESTIMATOR_RATIO = 0.25     # bootstrap-vs-jackknife within 25%
TOL_PROD_AUDIT_RATIO = 0.10         # analytical-vs-bootstrap audit SE within 10%


def test_prodalpha_bootstrap_jackknife_estimator_agreement():
    """At α = 0.10, bootstrap SE and jackknife SE for the CVaR-CJE estimator
    should agree within 25% on a stress policy (premium). They target the
    same calibrator-induced variance from different angles, so 5–20% gap
    is expected; >25% would suggest one method is mis-specified.
    """
    s_train, y_train, w_train, get_target = _build_panels()
    s_target, _ = get_target("premium")
    cv = bootstrap_cvar_ci(
        s_train, y_train, s_target, alpha=PROD_ALPHA,
        sample_weight_train=w_train, B=500, seed=42,
    )
    var_jk = jackknife_var_cal(
        s_train, y_train, s_target, alpha=PROD_ALPHA,
        sample_weight_oracle=w_train, K=5, seed=42,
    )
    se_boot = float(np.sqrt(cv["var_eval"]))
    se_jk = float(np.sqrt(var_jk))
    if min(se_boot, se_jk) <= 0:
        raise AssertionError(f"degenerate SEs: boot={se_boot}, jk={se_jk}")
    rel = abs(se_boot - se_jk) / max(se_boot, se_jk)
    assert rel < TOL_PROD_ESTIMATOR_RATIO, (
        f"estimator SE disagreement at α={PROD_ALPHA}: "
        f"bootstrap={se_boot:.4f}, jackknife={se_jk:.4f}, "
        f"rel={rel:.3f} > {TOL_PROD_ESTIMATOR_RATIO}"
    )


def test_prodalpha_audit_se_analytical_vs_bootstrap():
    """At α = 0.10 with t̂ fixed at the production saddle-point optimum,
    analytical SE for g2 (`cvar_audit_analytical_se`) should agree with
    paired audit-bootstrap SE within 10%. This is the analog to B1 ≈ B2
    for the mean audit (t-test ≈ paired audit bootstrap).
    """
    s_train, y_train, w_train, get_target = _build_panels()
    s_target, y_target = get_target("premium")
    n_full = len(s_target)
    rng_audit = np.random.default_rng(123)
    a_idx = rng_audit.choice(n_full, size=132, replace=False)
    s_audit = s_target[a_idx]
    y_audit = y_target[a_idx]
    # Pin t̂ at the production optimum
    _, t_hat_prod, _, _ = estimate_direct_cvar_isotonic(
        s_train, y_train, s_target, PROD_ALPHA, 61,
        sample_weight_train=w_train,
    )
    an = cvar_audit_analytical_se(
        s_train, y_train, s_audit, y_audit,
        t0=t_hat_prod, alpha=PROD_ALPHA, sample_weight_train=w_train,
    )
    # Paired audit bootstrap with t̂ fixed and calibrator fixed
    z_train = np.maximum(t_hat_prod - y_train, 0.0)
    g_pred_audit = fit_isotonic_tail_loss(
        s_train, z_train, s_audit, sample_weight=w_train,
    )
    g2 = np.maximum(t_hat_prod - y_audit, 0.0) - g_pred_audit
    rng = np.random.default_rng(42)
    n_a = len(s_audit)
    g2_boots = np.empty(500)
    for b in range(500):
        idx = rng.integers(0, n_a, size=n_a)
        g2_boots[b] = float(g2[idx].mean())
    se_g2_boot = float(g2_boots.std(ddof=1))
    se_g2_an = an["se_g2"]
    rel = abs(se_g2_an - se_g2_boot) / max(se_g2_an, se_g2_boot, 1e-12)
    assert rel < TOL_PROD_AUDIT_RATIO, (
        f"audit g2 SE disagreement at α={PROD_ALPHA}, fixed t̂={t_hat_prod:+.4f}: "
        f"analytical={se_g2_an:.4f}, bootstrap={se_g2_boot:.4f}, "
        f"rel={rel:.3f} > {TOL_PROD_AUDIT_RATIO}"
    )


# ---------------------------------------------------------------------------
# Script runner: collect all test_* functions and report pass/fail
# ---------------------------------------------------------------------------
def _all_tests():
    return [
        ("truth = E[Y]", test_alpha1_truth_equals_mean),
        ("estimator = mean_cje", test_alpha1_estimator_equals_mean_cje),
        ("g2 = -mean_residual", test_alpha1_g2_equals_neg_mean_residual),
        ("g1 = 0", test_alpha1_g1_is_zero),
        ("t̂ ≥ y_train_max", test_alpha1_t_hat_covers_y_train_max),
        ("low-α grid unchanged", test_low_alpha_grid_unchanged),
        ("bootstrap CI agreement (g2 vs -ε)", test_alpha1_bootstrap_g2_matches_mean_residual),
        ("isotonic collapse identity", test_alpha1_isotonic_collapse),
        ("bootstrap point-estimate CI agreement", test_alpha1_bootstrap_point_estimate_ci_agreement),
        ("Var_cal agreement (CVaR vs Mean jackknife)", test_alpha1_var_cal_agreement),
        ("audit p-value agreement (Wald vs t-test)", test_alpha1_audit_pvalue_agreement),
        ("prod α: bootstrap vs jackknife estimator SE", test_prodalpha_bootstrap_jackknife_estimator_agreement),
        ("prod α: analytical vs bootstrap audit g2 SE", test_prodalpha_audit_se_analytical_vs_bootstrap),
    ]


def main() -> int:
    results = []
    for name, fn in _all_tests():
        try:
            fn()
            results.append((name, True, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))

    width = max(len(name) for name, _, _ in results)
    print(f"α=1 identity regression suite (n=500 HealthBench, seed=42)")
    print("-" * (width + 12))
    n_pass = 0
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<{width}}  {status}")
        if not ok:
            print(f"  {' ' * width}  {msg}")
        else:
            n_pass += 1
    print("-" * (width + 12))
    print(f"  {n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
