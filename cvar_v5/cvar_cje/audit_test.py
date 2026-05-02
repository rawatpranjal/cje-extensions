"""
Tests for cvar_cje.audit.

Most tests below are code tests: algebraic identities, validation, branch
coverage, and deterministic bootstrap behavior. The explicitly named
`test_statistical_*` tests run small repeated-DGP checks of audit size, power,
and Wald-statistic movement.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from ..mc.dgp import DEFAULT_POLICIES, DGP
from .audit import two_moment_wald_audit
from .calibrator import fit_calibrator_grid
from .estimator import estimate_direct_cvar
from .schema import Slice


def _toy_oracle_split(n_total: int = 600, seed: int = 0):
    """
    Two disjoint slices simulating CALIB and AUDIT from the same DGP, plus
    independently drawn EVAL.
    """
    rng = np.random.default_rng(seed)

    def draw(n: int):
        s = rng.normal(size=n)
        y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
        y = np.clip(y, 0.0, 1.0)
        return s, y

    s_calib, y_calib = draw(n_total)
    s_audit, y_audit = draw(n_total // 2)
    return s_calib, y_calib, s_audit, y_audit


def _audit_slice(s: np.ndarray, y: np.ndarray) -> Slice:
    df = pl.DataFrame({
        "prompt_id": [f"a{i}" for i in range(len(s))],
        "s": s,
        "y": y,
    })
    return Slice(df=df, role="audit")


def _statistical_dgp_audit(
    dgp: DGP,
    policy: str,
    delta: float,
    omega_estimator: str,
    seed: int,
    B: int = 120,
):
    """Fit on δ=0 CALIB, then audit a δ-shifted target slice."""
    alpha = 0.10
    t_grid = np.linspace(0.0, 1.0, 61)

    calib = dgp.sample(
        policy, n=500, with_oracle=True, seed=seed,
        delta=0.0, prompt_id_prefix="C",
    )
    audit = dgp.sample(
        policy, n=300, with_oracle=True, seed=seed + 1,
        delta=delta, prompt_id_prefix="A",
    )
    eval_df = dgp.sample(
        policy, n=800, with_oracle=False, seed=seed + 2,
        delta=delta, prompt_id_prefix="E",
    )

    cg = fit_calibrator_grid(
        calib["s"].to_numpy(),
        calib["y"].to_numpy(),
        t_grid,
    )
    estimate = estimate_direct_cvar(Slice(df=eval_df, role="eval"), cg, alpha)
    return two_moment_wald_audit(
        Slice(df=audit, role="audit"),
        cg,
        t_hat=estimate.threshold,
        alpha=alpha,
        omega_estimator=omega_estimator,  # type: ignore[arg-type]
        B=B,
        seed=seed,
    )


def test_alpha_one_reduces_to_mean() -> None:
    """
    At α=1 with t̂_1 = 1 and Y ∈ [0,1]:
        g_1 ≡ 0   element-wise
        g_2 == f̂(s) - Y   element-wise
    where f̂ is the increasing iso fit on (s_calib, y_calib).
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    # Run audit at α=1, t̂_1 = 1.0.
    verdict = two_moment_wald_audit(
        _audit_slice(s_audit, y_audit),
        cg,
        t_hat=1.0,
        alpha=1.0,
        omega_estimator="analytical",
        B=1,            # not used by analytical
        seed=0,
    )
    # g_1 ≡ 0 element-wise, so its mean is exactly 0.
    assert verdict.g1 == 0.0, f"g_1 should be exactly zero at α=1; got {verdict.g1}"

    # g_2 should equal mean(f̂(s_audit) - y_audit) where f̂ is iso_increasing.
    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s_calib, y_calib)
    expected_g2 = float(np.mean(f_inc.predict(s_audit) - y_audit))
    np.testing.assert_allclose(verdict.g2, expected_g2, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "estimator",
    [
        "analytical",
        "analytical_oua",
        "boot_remax_ridge",
        "boot_remax_no_ridge",
        "boot_fixed",
    ],
)
def test_omega_returns_finite_positive_W(estimator: str) -> None:
    """
    Sanity: each Ω̂ estimator returns a finite, non-negative W_n. Catches NaN
    leaks (e.g. divide-by-zero in a singular Ω̂) and sign errors.

    This is intentionally weak. The strong analytical-Ω̂ check is in
    test_omega_analytical_matches_closed_form_at_alpha_one. Inter-Ω̂
    discrimination is in test_omega_variants_give_different_W.

    Cross-fit calibrator (K=5) is used so analytical_oua's per-fold refit
    machinery is available.
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    fold_id = np.random.default_rng(0).integers(0, 5, size=len(s_calib))
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid, fold_id=fold_id, K=5)

    verdict = two_moment_wald_audit(
        _audit_slice(s_audit, y_audit),
        cg, t_hat=0.5, alpha=0.10,
        omega_estimator=estimator,  # type: ignore[arg-type]
        B=80, seed=42,
    )
    assert np.isfinite(verdict.W_n)
    assert verdict.W_n >= 0.0


def test_omega_analytical_oua_dominates_analytical_in_variance() -> None:
    """
    analytical_oua = analytical + V̂_cal_g, where V̂_cal_g is PSD by
    construction (sum of outer products). So Ω̂_oua ≥ Ω̂_analytical entrywise
    on the diagonal, and W_oua ≤ W_analytical when ḡ is the same.

    Catches sign errors in the V̂_cal accumulation.
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    fold_id = np.random.default_rng(0).integers(0, 5, size=len(s_calib))
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid, fold_id=fold_id, K=5)

    v_an = two_moment_wald_audit(
        _audit_slice(s_audit, y_audit),
        cg, t_hat=0.5, alpha=0.10,
        omega_estimator="analytical", B=80, seed=42,
    )
    v_oua = two_moment_wald_audit(
        _audit_slice(s_audit, y_audit),
        cg, t_hat=0.5, alpha=0.10,
        omega_estimator="analytical_oua", B=80, seed=42,
    )

    # ḡ should match exactly (oua reports the original ḡ).
    assert abs(v_an.g1 - v_oua.g1) <= 1e-12
    assert abs(v_an.g2 - v_oua.g2) <= 1e-12

    # Ω̂_oua ≥ Ω̂_analytical (entrywise on diagonal) ⇒ W_oua ≤ W_an.
    # (Tolerance 1e-9 to absorb pinv numerical noise.)
    assert v_oua.W_n <= v_an.W_n + 1e-9, (
        f"V̂_cal_g should be PSD; expected W_oua ≤ W_an. "
        f"Got W_an={v_an.W_n}, W_oua={v_oua.W_n}"
    )


def test_omega_analytical_oua_requires_cross_fit() -> None:
    """
    analytical_oua needs the per-fold calibrators to compute V̂_cal_g.
    A non-cross-fit calibrator (n_folds=0) must raise.
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg_no_fold = fit_calibrator_grid(s_calib, y_calib, t_grid)
    assert cg_no_fold.n_folds == 0

    with pytest.raises(ValueError, match="cross-fit calibrator"):
        two_moment_wald_audit(
            _audit_slice(s_audit, y_audit),
            cg_no_fold, t_hat=0.5, alpha=0.10,
            omega_estimator="analytical_oua", B=80, seed=42,
        )


def test_omega_analytical_matches_closed_form_at_alpha_one() -> None:
    """
    At α=1 with t̂_α = 1.0 and Y ∈ [0,1]:
        g_1 ≡ 0  (degenerate)
        g_2 = ĥ_1(s) − Y    [equivalently, (1 − Y) − ĥ_1(s); we use the
                              reflection identity ĥ_1(s) = 1 − f̂(s) and
                              note this equals f̂(s) − Y up to a sign]

    Var̂(ḡ) is rank-1: top-left, off-diagonals all zero; bottom-right
    = sample_var(g_2) / n_audit. With pinv on a rank-1 PSD matrix, the
    Wald reduces to a one-dimensional t² statistic:

        W_n = n_audit · ḡ_2² / sample_var(g_2)

    We compute this expected value from (s_calib, y_calib, s_audit, y_audit)
    by hand using IsotonicRegression(increasing=True), then assert the
    analytical-Ω̂ audit returns the same number to floating-point precision.

    RED-verified by mutation: changing `omega = cov_per_row / len(s)` to
    `omega = cov_per_row` (forgetting the /n) inflates W_n by a factor of
    n_audit (~150 here) — far outside any reasonable rtol.
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split(n_total=600, seed=3)
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    v = two_moment_wald_audit(
        _audit_slice(s_audit, y_audit),
        cg, t_hat=1.0, alpha=1.0,
        omega_estimator="analytical", B=1, seed=0,
    )

    # Closed-form expected W_n.
    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip") \
        .fit(s_calib, y_calib)
    # At α=1, t̂=1, Y∈[0,1]: per audit.py math contract, g_2 = f̂(s) − Y.
    g2 = f_inc.predict(s_audit) - y_audit
    n_audit = len(s_audit)
    expected_W = n_audit * (g2.mean() ** 2) / g2.var(ddof=1)

    np.testing.assert_allclose(v.W_n, expected_W, rtol=1e-8, atol=0.0)


def test_decision_flips_at_chi2_threshold() -> None:
    """
    Decision must respect the χ²_{2, 0.95} = 5.991 cutoff. Two ground-truth
    cases — one with a well-specified DGP (small W expected, must PASS) and
    one with a heavily transport-broken DGP (large W expected, must REFUSE).

    The threshold value 5.991 is HARD-CODED, not recomputed from chi2.ppf —
    so a bug that swapped df=2 for df=1 (threshold 3.841) would not silently
    pass via the assertion using the same buggy formula.

    RED: this test would fail if the audit always returned the same decision
    (constant PASS or constant REFUSE), or if df were swapped to df=1 AND the
    W of one of the two cases happened to fall in the (3.841, 5.991) gap.
    """
    chi2_2_95 = 5.991  # χ²_{2, 1−0.05}, hard-coded

    # ---- Case PASS: well-specified data, W expected small ------------------
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split(n_total=600, seed=0)
    cg = fit_calibrator_grid(s_calib, y_calib, np.linspace(0.0, 1.0, 61))
    v_pass = two_moment_wald_audit(
        _audit_slice(s_audit, y_audit), cg,
        t_hat=0.5, alpha=0.50, omega_estimator="analytical",
        B=1, seed=0, eta=0.05,
    )
    assert v_pass.W_n < chi2_2_95, (
        f"well-specified data gave W_n={v_pass.W_n:.3f}; expected < 5.991. "
        f"Either setup is wrong or audit is over-rejecting at the null."
    )
    assert v_pass.decision == "PASS", (
        f"W_n={v_pass.W_n:.3f} < 5.991 should yield PASS; got {v_pass.decision!r}"
    )

    # ---- Case REFUSE: Y shifted by 0.40 in audit, W expected large ----------
    rng = np.random.default_rng(7)
    s_c = rng.normal(size=600)
    y_c = np.clip(1.0 / (1.0 + np.exp(-s_c)) + 0.05 * rng.normal(size=600), 0.0, 1.0)
    s_a = rng.normal(size=400)
    y_a = np.clip(
        1.0 / (1.0 + np.exp(-s_a)) + 0.05 * rng.normal(size=400) - 0.40,
        0.0, 1.0,
    )
    cg_r = fit_calibrator_grid(s_c, y_c, np.linspace(0.0, 1.0, 61))
    v_ref = two_moment_wald_audit(
        _audit_slice(s_a, y_a), cg_r,
        t_hat=0.5, alpha=0.10, omega_estimator="analytical",
        B=1, seed=7, eta=0.05,
    )
    assert v_ref.W_n > chi2_2_95, (
        f"transport-broken data gave W_n={v_ref.W_n:.3f}; expected > 5.991. "
        f"Audit may be under-powered."
    )
    assert v_ref.decision == "REFUSE-LEVEL", (
        f"W_n={v_ref.W_n:.3f} > 5.991 should yield REFUSE-LEVEL; "
        f"got {v_ref.decision!r}"
    )


def test_t_hat_must_be_on_grid() -> None:
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split(n_total=80)
    t_grid = np.linspace(0.0, 1.0, 11)  # spacing 0.1
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    with pytest.raises(ValueError, match="not on the calibrator grid"):
        two_moment_wald_audit(
            _audit_slice(s_audit, y_audit),
            cg,
            t_hat=0.123,  # off-grid
            alpha=0.10,
            omega_estimator="analytical",
            B=1,
            seed=0,
        )


def test_audit_rejects_when_transport_breaks() -> None:
    """
    The audit's whole purpose: when CALIB and AUDIT come from materially
    different conditional distributions, at least one Ω̂ variant must REFUSE.

    Construction:
      - CALIB Y ~ logit(s) + small noise            (calibrator fits this)
      - AUDIT Y' = clip(Y_orig - 0.30, 0, 1)        (target distribution
        systematically lower for the same s — calibrator over-predicts)
      - At t_hat=0.5, audit's g_2 = mean[(0.5 - Y')_+] - mean[ĥ_0.5(s)] is
        biased AWAY from zero. The Wald statistic should grow large, at least
        on the analytical Ω̂ which doesn't suffer the small-n_audit ridge issue.

    RED-verified by mutation: replacing `decision = "REFUSE-LEVEL" if p_value
    < eta else "PASS"` with `decision = "PASS"` makes this test fail with
    decisions=['PASS', 'PASS', 'PASS', 'PASS'].
    """
    rng = np.random.default_rng(0)
    n_calib, n_audit = 600, 300

    def draw(n: int):
        s = rng.normal(size=n)
        y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
        y = np.clip(y, 0.0, 1.0)
        return s, y

    s_calib, y_calib = draw(n_calib)
    s_audit, y_audit_orig = draw(n_audit)
    y_audit_shifted = np.clip(y_audit_orig - 0.30, 0.0, 1.0)

    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    decisions = []
    for omega in ("analytical", "boot_remax_ridge", "boot_remax_no_ridge", "boot_fixed"):
        v = two_moment_wald_audit(
            _audit_slice(s_audit, y_audit_shifted),
            cg, t_hat=0.5, alpha=0.10,
            omega_estimator=omega, B=200, seed=0,
        )
        decisions.append(v.decision)

    assert "REFUSE-LEVEL" in decisions, (
        f"audit failed to detect a clear transport break (Y shifted by 0.30); "
        f"all four Ω̂ variants returned PASS. decisions={decisions}"
    )


def test_omega_variants_give_different_W() -> None:
    """
    The four Ω̂ variants must produce DIFFERENT W_n values on the same input.
    If a refactor accidentally routes all four to one code path, this test
    catches it.

    RED-verified by mutation: replacing the if/elif dispatch in
    `two_moment_wald_audit` so all branches call `_omega_analytical` makes
    the assertion fail (all four W values become identical).
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    W_by_omega: dict[str, float] = {}
    for omega in ("analytical", "boot_remax_ridge", "boot_remax_no_ridge", "boot_fixed"):
        v = two_moment_wald_audit(
            _audit_slice(s_audit, y_audit),
            cg, t_hat=0.5, alpha=0.10,
            omega_estimator=omega, B=200, seed=42,
        )
        W_by_omega[omega] = v.W_n

    distinct = {round(w, 6) for w in W_by_omega.values()}
    assert len(distinct) >= 3, (
        f"Ω̂ variants collapsed to {len(distinct)} distinct W values; "
        f"expected ≥ 3 distinct (allowing two boot variants to coincide by "
        f"chance is fine). W_by_omega={W_by_omega}"
    )


def test_bootstrap_audit_is_reproducible() -> None:
    """
    Bootstrap-based Ω̂ estimators must be deterministic given a fixed seed.
    Same audit data + same seed → byte-identical W_n.

    RED-verified by mutation: replacing `rng = np.random.default_rng(seed)`
    with `rng = np.random.default_rng()` (no seed) makes this test fail —
    different runs produce different W_n.
    """
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    for omega in ("boot_remax_ridge", "boot_remax_no_ridge", "boot_fixed"):
        kwargs = dict(
            audit_slice=_audit_slice(s_audit, y_audit),
            calibrator=cg, t_hat=0.5, alpha=0.10,
            omega_estimator=omega, B=200, seed=7,
        )
        v1 = two_moment_wald_audit(**kwargs)  # type: ignore[arg-type]
        v2 = two_moment_wald_audit(**kwargs)  # type: ignore[arg-type]
        assert v1.W_n == v2.W_n, (
            f"{omega}: bootstrap not reproducible. "
            f"W_n run1={v1.W_n!r} W_n run2={v2.W_n!r}"
        )
        assert v1.p_value == v2.p_value


def test_statistical_audit_size_at_null_default_boot_remax_ridge() -> None:
    """
    Statistical: under δ=0, the default boot_remax_ridge audit should not
    wildly over-reject.

    This encodes the current Ω̂ state honestly: the ridge variant is
    conservative at this smoke scale, while analytical/no-ridge variants are
    diagnostics rather than default validation gates.
    """
    dgp = DGP(DEFAULT_POLICIES)
    policies = ("uniform", "left_skew")
    R = 16

    rejects = 0
    total = 0
    for policy_offset, policy in enumerate(policies):
        for r in range(R):
            verdict = _statistical_dgp_audit(
                dgp,
                policy=policy,
                delta=0.0,
                omega_estimator="boot_remax_ridge",
                seed=7003 * r + 911 * policy_offset,
            )
            rejects += int(verdict.decision == "REFUSE-LEVEL")
            total += 1

    rejection_rate = rejects / total
    assert rejection_rate <= 0.25, (
        f"default boot_remax_ridge over-rejects at the null: "
        f"rate={rejection_rate:.3f} over {total} fixed-seed reps"
    )


def test_statistical_audit_power_under_transport_break() -> None:
    """
    Statistical: under a clear δ=1 transport break, at least one diagnostic
    Ω̂ variant should reject materially more often than at δ=0.

    The default ridge variant is deliberately excluded from the power gate
    because the current known issue is conservativeness, not a locked Ω̂
    winner.
    """
    dgp = DGP(DEFAULT_POLICIES)
    policies = ("uniform", "left_skew")
    omegas = ("analytical", "boot_remax_no_ridge", "boot_fixed")
    R = 16
    rates: dict[tuple[str, float], float] = {}

    for omega in omegas:
        for delta in (0.0, 1.0):
            rejects = 0
            total = 0
            for policy_offset, policy in enumerate(policies):
                for r in range(R):
                    verdict = _statistical_dgp_audit(
                        dgp,
                        policy=policy,
                        delta=delta,
                        omega_estimator=omega,
                        seed=7003 * r + 911 * policy_offset,
                    )
                    rejects += int(verdict.decision == "REFUSE-LEVEL")
                    total += 1
            rates[(omega, delta)] = rejects / total

    null_best = max(rates[(omega, 0.0)] for omega in omegas)
    break_best = max(rates[(omega, 1.0)] for omega in omegas)

    assert break_best >= 0.35, (
        f"no diagnostic Ω̂ variant has material power at δ=1. "
        f"rates={rates}"
    )
    assert break_best >= null_best + 0.15, (
        f"transport break did not raise rejection materially. rates={rates}"
    )


def test_statistical_audit_wald_moves_with_break_strength() -> None:
    """
    Statistical: with analytical Ω̂, stronger transport breaks should raise
    W_n, lower p-values, and increase REFUSE decisions on fixed DGP draws.
    """
    dgp = DGP(DEFAULT_POLICIES)
    R = 12
    summaries: dict[float, tuple[float, float, float]] = {}

    for delta in (0.0, 1.0, 2.0):
        W_values: list[float] = []
        p_values: list[float] = []
        rejects = 0
        for r in range(R):
            verdict = _statistical_dgp_audit(
                dgp,
                policy="uniform",
                delta=delta,
                omega_estimator="analytical",
                seed=9001 * r + 5,
            )
            W_values.append(verdict.W_n)
            p_values.append(verdict.p_value)
            rejects += int(verdict.decision == "REFUSE-LEVEL")
        summaries[delta] = (
            float(np.mean(W_values)),
            float(np.median(p_values)),
            rejects / R,
        )

    mean_W0, median_p0, reject0 = summaries[0.0]
    mean_W2, median_p2, reject2 = summaries[2.0]
    assert mean_W2 > mean_W0 + 3.0, f"W_n did not increase enough: {summaries}"
    assert median_p2 < median_p0, f"p-values did not move down: {summaries}"
    assert reject2 > reject0, f"REFUSE rate did not increase: {summaries}"


def test_wrong_role_rejected() -> None:
    s_calib, y_calib, s_audit, y_audit = _toy_oracle_split(n_total=80)
    t_grid = np.linspace(0.0, 1.0, 11)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    df = pl.DataFrame({"prompt_id": ["p0"], "s": [0.0], "y": [0.5]})
    bad = Slice(df=df, role="calib")
    with pytest.raises(ValueError, match="must be 'audit'"):
        two_moment_wald_audit(
            bad, cg, t_hat=0.5, alpha=0.10,
            omega_estimator="analytical", B=1, seed=0,
        )
