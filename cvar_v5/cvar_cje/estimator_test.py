"""
Tests for cvar_cje.estimator.

The α=1 collapse identity is the load-bearing test of this module: at α=1 the
saddle-point estimator must equal an independently-derived Mean-CJE reference
to numerical precision. The reference is computed inline (Q4b option ii) using
a separate IsotonicRegression(increasing=True) fit on (s_calib, y_calib).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.isotonic import IsotonicRegression

from ._crossfit import partition_oracle
from .calibrator import fit_calibrator_grid
from .estimator import estimate_direct_cvar
from .schema import Slice
from ..mc.dgp import DEFAULT_POLICIES, DGP


def _toy_split(n_calib: int = 400, n_eval: int = 600, seed: int = 0):
    """Two independent draws from the same monotone DGP."""
    rng = np.random.default_rng(seed)

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        s = rng.normal(size=n)
        y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
        y = np.clip(y, 0.0, 1.0)
        return s, y

    s_calib, y_calib = draw(n_calib)
    s_eval, _ = draw(n_eval)
    return s_calib, y_calib, s_eval


def _eval_slice(s_eval: np.ndarray) -> Slice:
    df = pl.DataFrame({
        "prompt_id": [f"p{i}" for i in range(len(s_eval))],
        "s": s_eval,
    })
    return Slice(df=df, role="eval")


def test_alpha_one_collapse() -> None:
    """
    ĈVaR_1 (direct, saddle-point) == Mean-CJE reference (inline, independent fit).

    Tolerance ≤ 1e-9 per spec §2 Bar G1.
    """
    s_calib, y_calib, s_eval = _toy_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    # Direct CVaR-CJE at α=1.
    res = estimate_direct_cvar(_eval_slice(s_eval), cg, alpha=1.0)
    v_cvar = res.value

    # Independent reference: increasing iso on (s_calib, y_calib), mean over EVAL.
    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s_calib, y_calib)
    v_mean = float(np.mean(f_inc.predict(s_eval)))

    assert abs(v_cvar - v_mean) <= 1e-9, (
        f"α=1 collapse failed: ĈVaR_1={v_cvar!r}, Mean-CJE={v_mean!r}, "
        f"|Δ|={abs(v_cvar - v_mean):.3e} > 1e-9"
    )


def test_grid_boundary_t1_at_alpha_one() -> None:
    """
    At α=1 with T = linspace(0, 1, 61) and Y ∈ [0,1], `t̂_1 == 1.0` deterministically:
    the plateau t ≥ max_i f̂(s_i) ≤ 1 always reaches the right endpoint.
    """
    s_calib, y_calib, s_eval = _toy_split()
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    res = estimate_direct_cvar(_eval_slice(s_eval), cg, alpha=1.0)
    assert res.threshold == 1.0


def test_monotone_in_alpha() -> None:
    """
    On a monotone DGP, lower α (more pessimistic tail) → smaller estimate.
    """
    s_calib, y_calib, s_eval = _toy_split(n_calib=600, n_eval=800)
    t_grid = np.linspace(0.0, 1.0, 61)
    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)
    eval_s = _eval_slice(s_eval)

    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    values = [estimate_direct_cvar(eval_s, cg, a).value for a in alphas]

    diffs = np.diff(values)
    assert (diffs >= -1e-9).all(), (
        f"CVaR estimate should be non-decreasing in α; got values={values}"
    )


def test_alpha_out_of_range_rejected() -> None:
    s_calib, y_calib, s_eval = _toy_split(n_calib=50, n_eval=50)
    cg = fit_calibrator_grid(s_calib, y_calib, np.linspace(0, 1, 11))
    eval_s = _eval_slice(s_eval)

    with pytest.raises(ValueError, match="alpha must be in"):
        estimate_direct_cvar(eval_s, cg, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        estimate_direct_cvar(eval_s, cg, alpha=1.5)


def test_wrong_slice_role_rejected() -> None:
    s_calib, y_calib, s_eval = _toy_split(n_calib=50, n_eval=50)
    cg = fit_calibrator_grid(s_calib, y_calib, np.linspace(0, 1, 11))

    df = pl.DataFrame({"prompt_id": ["p0"], "s": [0.0]})
    bad = Slice(df=df, role="calib")
    with pytest.raises(ValueError, match="must be 'eval'"):
        estimate_direct_cvar(bad, cg, alpha=0.1)


# -----------------------------------------------------------------------------
# Degeneracy code tests: pathological-Y / pathological-s structures where the
# estimator's output is determined exactly (≤ 1e-9 tolerance, no MC).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_constant_y_returns_constant(c: float) -> None:
    """
    Code: when Y ≡ c (a degenerate point mass), CVaR_α(Y) = c at any α ∈ (0, 1].

    Math
    -----
    With y_i = c for all i,
        z_i(t) = max(t − c, 0)  is constant in i for every t,
    so the isotonic fit returns ĥ_t(s) ≡ max(t − c, 0). Then
        Ψ̂_α(t) = t − (1/α) · max(t − c, 0)
               = { t           if t ≤ c
                 { c + (1 − 1/α)(t − c)   if t > c
    Since 1/α ≥ 1, the second branch is non-increasing in t past c. The argmax
    over T ⊇ {c} is at t = c with value c.

    Coverage: c ∈ {0, 0.5, 1.0}, all three on the grid linspace(0, 1, 61).
    """
    rng = np.random.default_rng(0)
    n_calib, n_eval = 200, 300
    s_calib = rng.normal(size=n_calib)
    s_eval = rng.normal(size=n_eval)
    y_calib = np.full(n_calib, c, dtype=np.float64)
    t_grid = np.linspace(0.0, 1.0, 61)

    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    for alpha in (0.05, 0.10, 0.25, 0.50, 1.0):
        res = estimate_direct_cvar(_eval_slice(s_eval), cg, alpha)
        assert abs(res.value - c) <= 1e-9, (
            f"constant Y={c}, α={alpha}: ĈVaR={res.value!r}, expected {c}, "
            f"|Δ|={abs(res.value - c):.3e}"
        )


def test_constant_s_collapses_to_empirical_cvar() -> None:
    """
    Code: when s is constant (s_i = c0 for all i in CALIB), the calibrator
    has no covariate signal — IsotonicRegression collapses to a single fit
    point at x=c0 with value mean(z), and `out_of_bounds="clip"` returns that
    constant on every EVAL row.

    The Direct estimate must then equal the saddle-point empirical CVaR_α of
    Y on the calibration sample, computed independently:

        Ψ_α^emp(t) = t − (1/α) · (1/n_calib) Σ_i max(t − Y_i, 0)
        ĈVaR_α^emp = max_{t ∈ T} Ψ_α^emp(t)

    Tolerance ≤ 1e-9. Catches a mistaken s-pooling that breaks when `s`
    carries no information.
    """
    rng = np.random.default_rng(1)
    n_calib, n_eval = 200, 400
    s_calib = np.zeros(n_calib)
    s_eval = rng.normal(size=n_eval)
    y_calib = rng.beta(2.0, 5.0, size=n_calib)
    t_grid = np.linspace(0.0, 1.0, 61)

    cg = fit_calibrator_grid(s_calib, y_calib, t_grid)

    for alpha in (0.05, 0.10, 0.25, 0.50, 1.0):
        res = estimate_direct_cvar(_eval_slice(s_eval), cg, alpha)

        # Independent reference: same saddle-point form, evaluated on the
        # empirical Y_calib distribution directly (no calibrator).
        z = np.maximum(t_grid[None, :] - y_calib[:, None], 0.0)  # (n_calib, |T|)
        psi_ref = t_grid - z.mean(axis=0) / alpha
        ref_value = float(psi_ref.max())

        assert abs(res.value - ref_value) <= 1e-9, (
            f"α={alpha}: ĈVaR={res.value!r}, empirical Y CVaR={ref_value!r}, "
            f"|Δ|={abs(res.value - ref_value):.3e}"
        )


def test_estimate_invariant_under_s_affine_transform() -> None:
    """
    Code: isotonic regression depends on s only through its rank order.
    Applying the SAME monotone-affine map s → a·s + b (a > 0) to both the
    CALIB and the EVAL slice must leave the estimate unchanged exactly.

    Two transformations cover both shift and scale axes.
    """
    s_calib, y_calib, s_eval = _toy_split(n_calib=400, n_eval=400, seed=2)
    t_grid = np.linspace(0.0, 1.0, 61)

    cg_base = fit_calibrator_grid(s_calib, y_calib, t_grid)
    res_base = estimate_direct_cvar(_eval_slice(s_eval), cg_base, alpha=0.10)

    for a, b in [(1.0, 100.0), (5.0, -3.0)]:
        cg_t = fit_calibrator_grid(a * s_calib + b, y_calib, t_grid)
        res_t = estimate_direct_cvar(
            _eval_slice(a * s_eval + b), cg_t, alpha=0.10,
        )
        assert abs(res_t.value - res_base.value) <= 1e-9, (
            f"s → {a}·s + {b}: value drifted {abs(res_t.value - res_base.value):.3e}"
        )
        assert res_t.threshold == res_base.threshold, (
            f"s → {a}·s + {b}: t̂ changed from {res_base.threshold} to {res_t.threshold}"
        )


# =============================================================================
# Statistical tests
# =============================================================================
#
# These exercise the full DGP → calibrator → estimator path on the parametric
# Beta panel. Each runs a small MC loop (R = 12-20) and asserts a property
# under MC-noise tolerances.
#
# The audit is bypassed throughout (we call estimate_direct_cvar directly) —
# audit-statistical tests live in audit_test.py.
#
# Naming convention: every statistical test starts with `test_statistical_`.
# Run only these via:    pytest cvar_v5/cvar_cje/estimator_test.py -k statistical
# Skip them via:         pytest cvar_v5/cvar_cje/estimator_test.py -k 'not statistical'


# RMSE ceilings anchored to the MEDIUM run (R=60, n_oracle=600, n_eval=1000) at
# 2026-05-02T102005_medium, with ~2.2× headroom over observed values.
_RMSE_CEILING_AT_MEDIUM_SCALE: dict[tuple[str, float], float] = {
    ("uniform",    0.10): 0.015,   # MEDIUM RMSE: 0.0068
    ("right_skew", 0.10): 0.015,   # MEDIUM RMSE: 0.0063
    ("left_skew",  0.10): 0.030,   # MEDIUM RMSE: 0.0134
    ("tail_heavy", 0.10): 0.007,   # MEDIUM RMSE: 0.0028
    ("uniform",    0.20): 0.020,   # MEDIUM RMSE: 0.0091
    ("right_skew", 0.20): 0.012,   # MEDIUM RMSE: 0.0056
    ("left_skew",  0.20): 0.025,   # MEDIUM RMSE: 0.0104
    ("tail_heavy", 0.20): 0.015,   # MEDIUM RMSE: 0.0066
}


def _run_estimator(
    dgp: DGP, policy: str, alpha: float,
    n_oracle: int, n_eval: int, seed: int,
    *, calibrator_override=None, eval_override=None,
) -> float:
    """
    One-rep harness: sample → calibrator → estimate, no audit.

    If `calibrator_override` is provided, it is used in place of fitting on
    the rep's oracle slice (CALIB-FROZEN path).

    If `eval_override` is provided, it is used in place of the rep's eval
    slice (EVAL-FROZEN path).

    Returns the scalar estimate value.
    """
    t_grid = np.linspace(0.0, 1.0, 61)

    if calibrator_override is None:
        oracle = dgp.sample(policy, n=n_oracle, with_oracle=True, seed=seed)
        calib_slice, _, folds = partition_oracle(oracle, K=5, seed=seed)
        cg = fit_calibrator_grid(
            calib_slice.s(), calib_slice.y(), t_grid,
            fold_id=folds.fold_id, K=5,
        )
    else:
        cg = calibrator_override

    if eval_override is None:
        eval_df = dgp.sample(policy, n=n_eval, with_oracle=False, seed=seed + 1)
        eval_slice = Slice(df=eval_df, role="eval")
    else:
        eval_slice = eval_override

    return estimate_direct_cvar(eval_slice, cg, alpha).value


@pytest.mark.parametrize(
    ("policy", "alpha"),
    [(p, a) for p in ("uniform", "right_skew", "left_skew", "tail_heavy")
     for a in (0.10, 0.20)],
)
def test_statistical_estimator_bias_rmse_on_known_dgp(
    policy: str, alpha: float,
) -> None:
    """
    Statistical: at MEDIUM scale (n_oracle=600, n_eval=1000) over R=16 reps,
    the estimator recovers truth_cvar(p, α) with bias indistinguishable from
    MC noise (3σ) and RMSE within the ceiling anchored to MEDIUM-run evidence.

    Math
    -----
        For r = 0..R-1:
            est_r  =  estimate_direct_cvar( pipeline(seed_r),  α )

        bias    =  mean_r(est_r)  -  truth_cvar(p, α)
        sigma_MC=  std_r(est_r) / sqrt(R)
        RMSE    =  sqrt( mean_r ( (est_r - truth_cvar(p, α))^2 ) )

    Gates:  |bias| / sigma_MC <= 3      AND      RMSE <= ε_{p, α}

    Coverage: 4 panel policies × α ∈ {0.10, 0.20} = 8 cells.
    """
    dgp = DGP(DEFAULT_POLICIES)
    rmse_ceiling = _RMSE_CEILING_AT_MEDIUM_SCALE[(policy, alpha)]
    R = 16

    estimates: list[float] = []
    for r in range(R):
        seed = 5003 * r + 41
        estimates.append(_run_estimator(
            dgp, policy, alpha, n_oracle=600, n_eval=1000, seed=seed,
        ))

    arr = np.asarray(estimates)
    truth = dgp.truth_cvar(policy, alpha)
    bias = float(arr.mean() - truth)
    se_mc = float(arr.std(ddof=1) / np.sqrt(R))
    rmse = float(np.sqrt(np.mean((arr - truth) ** 2)))

    assert abs(bias) <= 3.0 * se_mc, (
        f"{policy} α={alpha}: estimator bias too large. "
        f"mean={arr.mean():.5f}, truth={truth:.5f}, bias={bias:+.5f}, "
        f"σ_MC={se_mc:.5f}, |bias|/σ_MC={abs(bias)/se_mc:.2f}"
    )
    assert rmse <= rmse_ceiling, (
        f"{policy} α={alpha}: RMSE={rmse:.5f} exceeds ceiling {rmse_ceiling:.5f} "
        f"(MEDIUM baseline ≈ {rmse_ceiling/2.2:.4f})"
    )


def test_statistical_estimator_with_oracle_calibrator_is_unbiased() -> None:
    """
    Statistical: when the calibrator is replaced by a near-truth proxy
    (fit on n_huge=50,000), the saddle-point estimator's bias is
    indistinguishable from MC noise.

    Why this test
    -------------
    The full pipeline mixes calibrator-side noise and eval-side noise. This
    test ISOLATES the eval side by removing calibrator variance: ĥ_t is
    fixed across all reps at a near-truth value, so the only randomness in
    `est_r` comes from the eval slice.

    Math
    -----
        cg_huge  =  fit_calibrator_grid( sample(p, n=50_000) )    # h*-proxy

        For r = 0..R-1:
            eval_r  =  sample(p, n=1000, seed=10000 + r)
            est_r   =  estimate_direct_cvar( eval_r, cg_huge, α )

        var(est_r) here is var_eval (no calib variance).
        bias = mean_r(est_r) - truth_cvar(p, α) is the
               argmax-and-finite-eval bias only.

    Gate: |bias| / sigma_MC <= 3.

    What a failure here would mean
    -------------------------------
    The estimator math (saddle-point + argmax-on-grid) is wrong, OR n_eval
    is so small that finite-grid bias is detectable. With n_huge=50k the
    calibrator is essentially truth; this isolates estimator-side issues.
    """
    dgp = DGP(DEFAULT_POLICIES)
    policy = "uniform"
    alpha = 0.10
    R = 20
    t_grid = np.linspace(0.0, 1.0, 61)

    huge = dgp.sample(policy, n=50_000, with_oracle=True, seed=99999)
    cg_huge = fit_calibrator_grid(
        huge["s"].to_numpy(), huge["y"].to_numpy(), t_grid,
    )

    estimates: list[float] = []
    for r in range(R):
        estimates.append(_run_estimator(
            dgp, policy, alpha, n_oracle=0, n_eval=1000, seed=10000 + r,
            calibrator_override=cg_huge,
        ))

    arr = np.asarray(estimates)
    truth = dgp.truth_cvar(policy, alpha)
    bias = float(arr.mean() - truth)
    se_mc = float(arr.std(ddof=1) / np.sqrt(R))

    assert abs(bias) <= 3.0 * se_mc, (
        f"With h*-proxy calibrator (calibrator variance removed), the "
        f"estimator's bias is {bias:+.5f} = {abs(bias)/se_mc:.2f}σ_MC > 3σ. "
        f"Either saddle-point or argmax-on-grid mis-implemented, or n_eval "
        f"too small for the argmax-on-finite-grid bias to vanish."
    )


def _grid_var_decomposition(
    estimator_fn, calibrators: list, evals: list, alpha: float,
) -> tuple[float, float, float, np.ndarray]:
    """
    Reusable variance-decomposition primitive.

    Build the R_cal × R_eval grid  ψ_ij = estimator_fn(evals[j], calibrators[i], alpha)
    and return three population variances (ddof=0) computed from the SAME grid:

        var_real   :=  pop-variance of all R_cal · R_eval cells
        var_eval   :=  mean over rows of (within-row pop-variance)
        var_calib  :=  pop-variance of (per-row mean)

    By the algebraic ANOVA identity on any matrix of numbers,

        var_real  =  var_eval  +  var_calib    (exact, modulo float)

    is enforced regardless of the value of R_cal, R_eval, or the underlying
    estimator. ddof=0 is mandatory — sample variances (ddof=1) carry a
    finite-R correction that breaks exactness.

    Use this anytime you want a noise-free decomposition for an estimator on
    a known DGP. Plug-in estimators, future calibrator variants, real-data
    adapters: same primitive.

    Returns:
        (var_real, var_eval, var_calib, psi_grid)
    """
    R_cal = len(calibrators)
    R_eval = len(evals)
    psi = np.zeros((R_cal, R_eval), dtype=np.float64)
    for i, cg in enumerate(calibrators):
        for j, ev in enumerate(evals):
            psi[i, j] = estimator_fn(ev, cg, alpha).value

    var_real = float(psi.var(ddof=0))
    var_eval = float(psi.var(axis=1, ddof=0).mean())
    var_calib = float(psi.mean(axis=1).var(ddof=0))
    return var_real, var_eval, var_calib, psi


def test_statistical_var_decomposition_holds() -> None:
    """
    Statistical: empirical check that  var_real == var_eval + var_calib
    via the grid-design primitive `_grid_var_decomposition`.

    Math
    -----
    For any R_cal × R_eval matrix of estimates ψ_{ij} = Ψ̂(cal_i, eval_j),
    the ANOVA decomposition holds exactly:

        Σ_{ij} (ψ_ij − ψ_..)²  =  Σ_{ij} (ψ_ij − ψ_i.)²
                              +  R_eval · Σ_i (ψ_i. − ψ_..)²

    Dividing by R_cal · R_eval translates to the population (ddof=0)
    law-of-total-variance:

        pop_var_{ij}( ψ_ij )  =  mean_i[ pop_var_j( ψ_ij ) ]
                             +  pop_var_i[ mean_j( ψ_ij ) ]

    No noise, no covariance correction, no minimum R required.

    Gate: |sum / total − 1| ≤ 1e-9   (numerical floor only)

    What a failure here would mean
    -------------------------------
    A bug in `_grid_var_decomposition` (wrong axis, wrong ddof, wrong
    aggregation), OR a non-deterministic estimator (rerunning it on the
    same input gives different values, breaking the matrix structure).
    Either of those is a real bug, not a statistical artifact.

    Headline number printed in the assertion message: var_calib /
    (var_eval + var_calib) — fraction of total variance attributable to
    the calibrator at MEDIUM scale.
    """
    dgp = DGP(DEFAULT_POLICIES)
    policy = "uniform"
    alpha = 0.10
    R_cal = R_eval = 12
    n_oracle, n_eval = 600, 1000
    t_grid = np.linspace(0.0, 1.0, 61)

    calibrators = []
    for i in range(R_cal):
        seed = 7000 * i + 13
        oracle = dgp.sample(policy, n=n_oracle, with_oracle=True, seed=seed)
        calib_slice, _, folds = partition_oracle(oracle, K=5, seed=seed)
        calibrators.append(fit_calibrator_grid(
            calib_slice.s(), calib_slice.y(), t_grid,
            fold_id=folds.fold_id, K=5,
        ))

    evals = [
        Slice(
            df=dgp.sample(policy, n=n_eval, with_oracle=False, seed=20000 + j),
            role="eval",
        )
        for j in range(R_eval)
    ]

    var_real, var_eval, var_calib, _ = _grid_var_decomposition(
        estimate_direct_cvar, calibrators, evals, alpha,
    )
    sum_decomp = var_eval + var_calib
    residual = abs(sum_decomp - var_real)
    rel_residual = residual / max(var_real, 1e-30)

    assert rel_residual <= 1e-9, (
        f"Variance decomposition broke. "
        f"var_real={var_real:.6e}, var_eval={var_eval:.6e}, "
        f"var_calib={var_calib:.6e}, sum={sum_decomp:.6e}, "
        f"residual={residual:.3e}, rel_residual={rel_residual:.3e}. "
        f"This is an ANOVA identity on the grid — must hold to float precision."
    )

    # Sanity: both terms positive (catches degenerate ddof / axis bugs).
    assert var_eval > 0.0 and var_calib > 0.0, (
        f"Expected both decomposition terms > 0; got var_eval={var_eval}, "
        f"var_calib={var_calib}."
    )


def test_statistical_estimator_rmse_scales_with_n_eval() -> None:
    """
    Statistical: with the calibrator pinned at an h*-proxy (n_huge=50k), the
    estimator's RMSE vs truth_cvar should fall as ~ 1/√n_eval — the only
    randomness left is the sample mean over n_eval rows of ĥ_{t̂}(s).

    Math
    -----
        Var[ Ψ̂_α(t̂) | h* fixed ]  =  Var_eval[ ĥ_{t̂}(s) ] / (α^2 · n_eval)
                                  ~  C / n_eval
    so  log(RMSE) ~ -0.5 · log(n_eval) + const.

    Gate (band tuned to MC noise at R=20):
        slope of log(RMSE) on log(n_eval)  ∈  [-0.7, -0.3]

    What a failure here would mean
    -------------------------------
    Either the var-decomposition prediction is wrong (n_eval-side variance
    is not 1/n), or the calibrator is leaking calib-side noise (e.g., the
    pooled override path is mistakenly refitting), or the saddle-point /
    argmax form is not the well-known O(1/√n) sample-mean estimator at fixed h.
    """
    dgp = DGP(DEFAULT_POLICIES)
    policy = "uniform"
    alpha = 0.10
    R = 20
    t_grid = np.linspace(0.0, 1.0, 61)

    huge = dgp.sample(policy, n=50_000, with_oracle=True, seed=99999)
    cg_huge = fit_calibrator_grid(
        huge["s"].to_numpy(), huge["y"].to_numpy(), t_grid,
    )

    truth = dgp.truth_cvar(policy, alpha)
    n_eval_values = (250, 500, 1000, 2000)
    rmses: list[float] = []

    for n_eval in n_eval_values:
        estimates = []
        for r in range(R):
            estimates.append(_run_estimator(
                dgp, policy, alpha, n_oracle=0, n_eval=n_eval, seed=20000 + r,
                calibrator_override=cg_huge,
            ))
        arr = np.asarray(estimates)
        rmses.append(float(np.sqrt(np.mean((arr - truth) ** 2))))

    log_n = np.log(np.asarray(n_eval_values, dtype=np.float64))
    log_rmse = np.log(np.asarray(rmses, dtype=np.float64))
    slope, _ = np.polyfit(log_n, log_rmse, 1)

    assert -0.7 <= slope <= -0.3, (
        f"RMSE-vs-n_eval slope {slope:.3f} not in [-0.7, -0.3]. "
        f"n_eval={list(n_eval_values)}, rmse={[f'{r:.4f}' for r in rmses]}. "
        f"Theoretical slope is -0.5 (O(1/√n))."
    )


def test_statistical_estimator_rmse_scales_with_n_oracle() -> None:
    """
    Statistical: with the eval slice pinned, the estimator's RMSE vs
    truth_cvar should fall at the isotonic L₂ rate as n_oracle grows.

    Math
    -----
    For a Lipschitz target h*_t, the per-t isotonic estimator has L₂ rate
    O(n_oracle^{-1/3}) (Brunk 1958, Yang 1999). The plug-in functional
    `mean_eval(ĥ_t(·))` inherits a rate between n^{-1/3} (variance-limited)
    and n^{-2/3} (bias-limited), depending on which term dominates at the
    scale being measured.

    Empirically on the `uniform` policy at α = 0.10 with n_eval = 1000
    pinned, the slope sits near -0.37 across n_oracle ∈ {300, 600, 1500,
    3000}.

    Gate (band tuned to MC noise at R=20):
        slope of log(RMSE) on log(n_oracle)  ∈  [-0.55, -0.15]

    Lower bound -0.15: a slope shallower than this means the calibrator
    isn't converging — almost certainly a bug in fold partitioning, fit,
    or the override path.
    Upper bound -0.55: faster than n^{-1/2} would be inconsistent with
    isotonic rates; signals an inadvertent variance reduction (e.g.,
    pooling across reps).
    """
    dgp = DGP(DEFAULT_POLICIES)
    policy = "uniform"
    alpha = 0.10
    R = 20
    t_grid = np.linspace(0.0, 1.0, 61)

    eval_frozen_df = dgp.sample(policy, n=1000, with_oracle=False, seed=88888)
    eval_frozen = Slice(df=eval_frozen_df, role="eval")

    truth = dgp.truth_cvar(policy, alpha)
    n_oracle_values = (300, 600, 1500, 3000)
    rmses: list[float] = []

    for n_oracle in n_oracle_values:
        estimates = []
        for r in range(R):
            seed = 7000 * r + 13
            estimates.append(_run_estimator(
                dgp, policy, alpha, n_oracle=n_oracle, n_eval=1000, seed=seed,
                eval_override=eval_frozen,
            ))
        arr = np.asarray(estimates)
        rmses.append(float(np.sqrt(np.mean((arr - truth) ** 2))))

    log_n = np.log(np.asarray(n_oracle_values, dtype=np.float64))
    log_rmse = np.log(np.asarray(rmses, dtype=np.float64))
    slope, _ = np.polyfit(log_n, log_rmse, 1)

    assert -0.55 <= slope <= -0.15, (
        f"RMSE-vs-n_oracle slope {slope:.3f} not in [-0.55, -0.15]. "
        f"n_oracle={list(n_oracle_values)}, rmse={[f'{r:.4f}' for r in rmses]}. "
        f"Isotonic theory predicts slope between -1/3 (variance-limited) "
        f"and -2/3 (bias-limited)."
    )


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_statistical_estimator_extreme_alpha(alpha: float) -> None:
    """
    Statistical: at small α the saddle-point divides by a small denominator,
    so any noise in `mean ĥ_t / α` is amplified by 1/α. Confirm that on the
    `uniform` policy the estimator stays finite and unbiased (bias within
    3·σ_MC) for α ∈ {0.05, 0.10, 0.20}.

    Math
    -----
    `Ψ̂_α(t) = t − (1/α) · mean_i ĥ_t(s_i)`. The 1/α factor magnifies
    any sampling error in the eval-slice mean of the calibrator output.

    Gate
    -----
        |bias| / σ_MC ≤ 3      AND       RMSE ≤ 0.05  (sanity ceiling)

    What a failure here would mean
    -------------------------------
    Either (a) the estimator returns a NaN / inf at small α (e.g., division
    by α inside an unguarded reciprocal that gets fed an empty mean), or
    (b) the variance scaling with 1/α^2 is much worse than predicted at
    small α — pointing at a numerical-stability issue around argmax-on-grid
    when Ψ̂(t) is very flat.

    Coverage: α=0.05 (paper headline), α=0.10 (paper default), α=0.20.
    """
    dgp = DGP(DEFAULT_POLICIES)
    policy = "uniform"
    R = 15

    estimates: list[float] = []
    for r in range(R):
        seed = 5003 * r + 41
        estimates.append(_run_estimator(
            dgp, policy, alpha, n_oracle=600, n_eval=1000, seed=seed,
        ))

    arr = np.asarray(estimates)
    assert np.isfinite(arr).all(), (
        f"α={alpha}: non-finite estimates encountered: {arr.tolist()}"
    )

    truth = dgp.truth_cvar(policy, alpha)
    bias = float(arr.mean() - truth)
    se_mc = float(arr.std(ddof=1) / np.sqrt(R))
    rmse = float(np.sqrt(np.mean((arr - truth) ** 2)))

    assert rmse <= 0.05, (
        f"α={alpha} on uniform: RMSE={rmse:.5f} > 0.05 sanity ceiling. "
        f"Estimates: {arr.tolist()}"
    )
    assert abs(bias) <= 3.0 * se_mc, (
        f"α={alpha} on uniform: bias={bias:+.5f}, σ_MC={se_mc:.5f}, "
        f"|bias|/σ_MC={abs(bias)/se_mc:.2f} > 3."
    )
