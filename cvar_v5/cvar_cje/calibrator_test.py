"""
Tests for cvar_cje.calibrator.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from .calibrator import fit_calibrator_grid, make_t_grid


def _toy_data(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Monotone DGP: Y = sigmoid(s) + small noise, clipped to [0,1]."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
    y = np.clip(y, 0.0, 1.0)
    return s, y


def test_monotone_decreasing_in_s() -> None:
    """ĥ_t is non-increasing in s for every t."""
    s, y = _toy_data()
    t_grid = np.linspace(0.0, 1.0, 11)
    cg = fit_calibrator_grid(s, y, t_grid)

    s_query = np.linspace(s.min(), s.max(), 50)
    H = cg.predict(s_query)  # (50, |T|)

    diffs = np.diff(H, axis=0)
    # All diffs should be ≤ 0 (with tiny float tolerance for plateau equality).
    assert (diffs <= 1e-12).all(), "ĥ_t should be non-increasing in s for every t"


def test_pav_reflection_identity_at_t1() -> None:
    """
    For Y ∈ [0,1] and t = 1:
        iso_decreasing(s, 1−Y) ≡ 1 − iso_increasing(s, Y)
    This is the foundation of the α=1 collapse to Mean-CJE.
    """
    s, y = _toy_data()
    t_grid = np.array([1.0])  # single grid point at the right endpoint
    cg = fit_calibrator_grid(s, y, t_grid)

    # Reference fit: increasing isotonic on (s, Y).
    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s, y)

    s_query = np.linspace(s.min(), s.max(), 100)
    h_dec = cg.predict(s_query)[:, 0]   # ĥ_1(s)
    f_pred = f_inc.predict(s_query)     # f̂(s)

    # Should match to numerical precision.
    np.testing.assert_allclose(h_dec, 1.0 - f_pred, atol=1e-12, rtol=0)


def test_oof_uses_correct_fold() -> None:
    """
    Direct check: for each fold k, the OOF prediction on fold-k rows must equal
    the prediction of an isotonic fit explicitly trained on (CALIB \\ fold_k).
    """
    s, y = _toy_data(n=300)
    K = 5
    rng = np.random.default_rng(2)
    fold_id = rng.integers(0, K, size=len(s))

    t = 0.4
    t_grid = np.array([t])
    cg = fit_calibrator_grid(s, y, t_grid, fold_id=fold_id, K=K)
    oof_pred = cg.predict_oof(s, fold_id)[:, 0]

    z = np.maximum(t - y, 0.0)
    for k in range(K):
        train_mask = fold_id != k
        ir_k = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            s[train_mask], z[train_mask]
        )
        ref = ir_k.predict(s[fold_id == k])
        np.testing.assert_allclose(oof_pred[fold_id == k], ref, atol=1e-12, rtol=0)


def test_calibrator_at_t_below_min_y_predicts_zero() -> None:
    """
    z_i(t) = max(t − Y_i, 0). When t ≤ min(Y), z_i = 0 for every row, so the
    isotonic fit is the zero function. Predict at any s must return 0 exactly.

    RED-verified by reasoning: a mutation that drops the `np.maximum(..., 0)`
    clipping (e.g., uses `t - y` directly) would produce negative targets and
    a non-zero fit, failing this test.
    """
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    y = rng.uniform(0.3, 0.9, size=200)  # Y bounded away from 0
    t_grid = np.array([0.0, 0.1, 0.2])    # all below min(y) ≈ 0.3
    cg = fit_calibrator_grid(s, y, t_grid)

    H = cg.predict(s)  # shape (200, 3)
    np.testing.assert_array_equal(H, 0.0)


def test_calibrator_above_max_y_shifts_by_dt() -> None:
    """
    For t > t' > max(Y): z_i(t) − z_i(t') = t − t' for every row (no clipping).
    Isotonic regression is shift-equivariant: iso_dec(s, z + c) = iso_dec(s, z) + c.
    Therefore ĥ_t(s) − ĥ_t'(s) ≡ t − t' for all s, exactly.

    This is a structural invariant of the calibrator's behaviour above max(Y).
    Predicates dependent on the random sample (mean, range, etc.) are not
    reliable for finite-n isotonic fits even when Y ⊥ s in the population —
    PAV will pick up sample noise. The shift-equivariance test, by contrast,
    is exact regardless of sample.

    RED-verified by reasoning: a mutation that drops `np.maximum(t - y, 0)`
    in favour of `y - t` (sign flip) would make z become 'y - t' for both
    thresholds, and the shift property would invert to t' - t. Test fails.
    """
    rng = np.random.default_rng(0)
    n = 300
    s = rng.normal(size=n)
    y = rng.uniform(0.0, 0.7, size=n)   # max(y) ≤ 0.7

    t1, t2 = 0.85, 1.0                   # both > max(y)
    cg = fit_calibrator_grid(s, y, np.array([t1, t2]))

    s_query = np.linspace(s.min(), s.max(), 50)
    H = cg.predict(s_query)              # shape (50, 2)

    diff = H[:, 1] - H[:, 0]             # ĥ_{t2}(s) - ĥ_{t1}(s)
    np.testing.assert_allclose(diff, t2 - t1, atol=1e-10, rtol=0)


def test_calibrator_monotone_and_lipschitz_in_t() -> None:
    """
    Math contract (1-Lipschitz in t):

        For Y ≥ 0 and t' > t:
            (t' - Y)_+  -  (t - Y)_+   ∈   [ 0,  t' - t ]

        So a consistent calibrator should satisfy, pointwise in s:
            0  ≤  ĥ_{t'}(s)  -  ĥ_t(s)   ≤   t' - t

    The lower bound is structural monotonicity (z = max(t - Y, 0) is non-
    decreasing in t). The upper bound is the unit Lipschitz constant of the
    target function in t.

    Tests both ends across the full grid.
    """
    s, y = _toy_data(n=400)
    t_grid = np.linspace(0.0, 1.0, 21)
    cg = fit_calibrator_grid(s, y, t_grid)

    s_query = np.linspace(s.min(), s.max(), 50)
    H = cg.predict(s_query)                 # shape (50, 21)
    diffs = np.diff(H, axis=1)              # ĥ_{t_{j+1}}(s) − ĥ_{t_j}(s)
    dts = np.diff(t_grid)                   # t_{j+1} − t_j   (constant)

    assert (diffs >= -1e-12).all(), (
        f"ĥ_t must be non-decreasing in t; "
        f"saw min diff = {float(diffs.min()):.4e}"
    )
    assert (diffs <= dts[None, :] + 1e-12).all(), (
        f"ĥ_t must be 1-Lipschitz in t (diff ≤ dt); "
        f"saw max excess = {float((diffs - dts[None, :]).max()):.4e}"
    )


def test_calibrator_predictions_in_zero_to_t() -> None:
    """
    Math contract (boundedness):

        For Y ∈ [0, 1] and t ∈ [0, 1]:
            z_i = (t - Y_i)_+   ∈   [ 0,  t ]

        Isotonic regression returns weighted averages of z within monotone
        groups, so its outputs lie in [min(z), max(z)] ⊂ [0, t].

    Verified at every grid t value, on s queries that include points
    outside the training range (out_of_bounds=clip).
    """
    s, y = _toy_data(n=400)
    t_grid = np.linspace(0.0, 1.0, 21)
    cg = fit_calibrator_grid(s, y, t_grid)

    s_query = np.linspace(s.min() - 0.5, s.max() + 0.5, 50)   # extrapolation included
    H = cg.predict(s_query)                                    # (50, 21)

    for j, t in enumerate(t_grid):
        h = H[:, j]
        assert (h >= -1e-12).all(), (
            f"ĥ_{t:.2f} below zero: min = {float(h.min()):.6f}"
        )
        assert (h <= t + 1e-12).all(), (
            f"ĥ_{t:.2f} exceeds t: max = {float(h.max()):.6f} > {t:.6f}"
        )


def test_calibrator_pooled_predict_is_kfold_independent() -> None:
    """
    Math / API contract:

        CalibratorGrid.predict() uses the POOLED fit (over all of CALIB).
        The pooled fit is computed identically regardless of fold_id; it
        sees the entire (s, y) sample. K-fold structure only affects
        per-fold fits, accessed via predict_oof().

        Therefore  predict(s)  is byte-identical  with or without folds.

    Why we test this
    ----------------
    The K-fold infrastructure exists in CalibratorGrid for future jackknife /
    OUA work (see TODO.md). It is NOT used by the production estimator path
    today — `estimate_direct_cvar` only calls `predict()`. This test pins the
    contract: a future change that accidentally routed pooled predictions
    through fold-k fits would break here.
    """
    s, y = _toy_data(n=400)
    t_grid = np.linspace(0.0, 1.0, 21)

    cg_no_folds = fit_calibrator_grid(s, y, t_grid)

    K = 5
    fold_id = np.arange(len(s)) % K
    cg_with_folds = fit_calibrator_grid(s, y, t_grid, fold_id=fold_id.astype(np.int64), K=K)

    s_query = np.linspace(s.min(), s.max(), 50)
    np.testing.assert_array_equal(
        cg_no_folds.predict(s_query),
        cg_with_folds.predict(s_query),
        err_msg="predict() output diverged between no-fold and K-fold fits — "
                "the pooled fit must be K-independent",
    )


def test_statistical_calibrator_l2_gap_shrinks_with_n() -> None:
    """
    Statistical: as n_calib grows, the calibrator's average L2 functional gap
    to a near-truth proxy shrinks.

    Math
    -----
    For a fitted calibrator ĥ_t and a "truth proxy" h_t*, the per-t L2 gap on
    a fresh sample {s_1, ..., s_M} from the target policy is:

        L2_gap_t  =  sqrt( (1/M) * sum_m  ( ĥ_t(s_m) - h_t*(s_m) )^2 )

    Aggregating across a panel of t values gives a single scalar:

        L2_gap_bar  =  mean_t  L2_gap_t

    A consistent isotonic estimator of h_t*(s) = E[(t - Y)_+ | s] satisfies:

        L2_gap_bar(n_calib)  →  0   as  n_calib  →  ∞

    So at two n_calib values n1 < n2, we expect L2_gap_bar(n2) < L2_gap_bar(n1).
    Per-t comparisons can fail on degenerate t (where h_t* ≈ 0 for all s, so
    the gap is dominated by sample noise), so we aggregate.

    Construction
    ------------
      h_t* proxy:    calibrator fitted on n_huge = 50,000 samples.
                     Its own residual error is ~1/sqrt(n_huge) ~ 5e-3.
      Test fits:     n_calib in {500, 5000}.
      Evaluation:    500 fresh s values; t panel = {0.10, 0.30, 0.50, 1.00}.

    RED: a mutation that disabled the isotonic fit (returned constant) would
    give L2_gap that does not shrink with n.
    """
    from ..mc.dgp import DGP, DEFAULT_POLICIES

    dgp = DGP(DEFAULT_POLICIES)
    t_grid = np.linspace(0.0, 1.0, 61)
    test_t_indices = [int(np.argmin(abs(t_grid - t))) for t in (0.10, 0.30, 0.50, 1.00)]

    for policy in ("uniform", "left_skew"):
        df_huge = dgp.sample(policy, n=50_000, with_oracle=True, seed=99999)
        cg_huge = fit_calibrator_grid(
            df_huge["s"].to_numpy(), df_huge["y"].to_numpy(), t_grid,
        )

        df_500 = dgp.sample(policy, n=500, with_oracle=True, seed=42)
        cg_500 = fit_calibrator_grid(
            df_500["s"].to_numpy(), df_500["y"].to_numpy(), t_grid,
        )

        df_5000 = dgp.sample(policy, n=5_000, with_oracle=True, seed=43)
        cg_5000 = fit_calibrator_grid(
            df_5000["s"].to_numpy(), df_5000["y"].to_numpy(), t_grid,
        )

        s_test = dgp.sample(policy, n=500, with_oracle=False, seed=12345)["s"].to_numpy()

        H_huge = cg_huge.predict(s_test)
        H_500  = cg_500.predict(s_test)
        H_5000 = cg_5000.predict(s_test)

        gaps_500 = [
            float(np.sqrt(np.mean((H_500[:, j]  - H_huge[:, j]) ** 2)))
            for j in test_t_indices
        ]
        gaps_5000 = [
            float(np.sqrt(np.mean((H_5000[:, j] - H_huge[:, j]) ** 2)))
            for j in test_t_indices
        ]

        avg_500  = float(np.mean(gaps_500))
        avg_5000 = float(np.mean(gaps_5000))

        assert avg_5000 < avg_500, (
            f"{policy}: average L2 gap across t did NOT shrink with n. "
            f"avg@n=500={avg_500:.4f}, avg@n=5000={avg_5000:.4f}. "
            f"Per-t gaps@500={['%.4f'%g for g in gaps_500]}, "
            f"per-t gaps@5000={['%.4f'%g for g in gaps_5000]}"
        )


def test_statistical_calibrator_integrated_bias_shrinks_with_n() -> None:
    """
    Statistical: as n_calib grows, the calibrator's integrated bias

        bias_func_t  =  mean_m  ( ĥ_t(s_m) - h_t*(s_m) )

    shrinks (in absolute value), aggregated across a t panel.

    Why this is the load-bearing metric
    -----------------------------------
    bias_func_t propagates DIRECTLY into the estimator's bias at fixed t:

        E[Psi_hat_α(t)] - Psi_α(t; h*)  =  -(1/α) * bias_func_t

    L2 gap (the prior test) bounds |bias_func_t| but is strictly weaker —
    a calibrator can have small L2 yet large signed bias if errors cancel
    pointwise but reinforce on average. This test catches that case.

    Construction parallels test_statistical_calibrator_l2_gap_shrinks_with_n
    (same h*-proxy, same s_test, same t panel).
    """
    from ..mc.dgp import DGP, DEFAULT_POLICIES

    dgp = DGP(DEFAULT_POLICIES)
    t_grid = np.linspace(0.0, 1.0, 61)
    test_t_indices = [int(np.argmin(abs(t_grid - t))) for t in (0.10, 0.30, 0.50, 1.00)]

    for policy in ("uniform", "left_skew"):
        df_huge = dgp.sample(policy, n=50_000, with_oracle=True, seed=99999)
        cg_huge = fit_calibrator_grid(
            df_huge["s"].to_numpy(), df_huge["y"].to_numpy(), t_grid,
        )

        df_500 = dgp.sample(policy, n=500, with_oracle=True, seed=42)
        cg_500 = fit_calibrator_grid(
            df_500["s"].to_numpy(), df_500["y"].to_numpy(), t_grid,
        )

        df_5000 = dgp.sample(policy, n=5_000, with_oracle=True, seed=43)
        cg_5000 = fit_calibrator_grid(
            df_5000["s"].to_numpy(), df_5000["y"].to_numpy(), t_grid,
        )

        s_test = dgp.sample(policy, n=500, with_oracle=False, seed=12345)["s"].to_numpy()

        H_huge = cg_huge.predict(s_test)
        H_500 = cg_500.predict(s_test)
        H_5000 = cg_5000.predict(s_test)

        biases_500 = [
            float(abs(np.mean(H_500[:, j] - H_huge[:, j])))
            for j in test_t_indices
        ]
        biases_5000 = [
            float(abs(np.mean(H_5000[:, j] - H_huge[:, j])))
            for j in test_t_indices
        ]

        avg_500 = float(np.mean(biases_500))
        avg_5000 = float(np.mean(biases_5000))

        assert avg_5000 < avg_500, (
            f"{policy}: average |bias_func_t| did NOT shrink with n. "
            f"avg@n=500={avg_500:.4f}, avg@n=5000={avg_5000:.4f}. "
            f"per-t @500={['%.4f' % b for b in biases_500]}, "
            f"per-t @5000={['%.4f' % b for b in biases_5000]}"
        )


def test_statistical_calibrator_l2_rate_matches_isotonic_theory() -> None:
    """
    Statistical: log(L2 gap) vs log(n_calib) has slope < -0.20.

    Theory: isotonic regression of a Lipschitz conditional expectation has
    rate `n^{-1/3}`, so the log-log slope should be ≈ -0.33. We assert the
    slope is more negative than -0.20 — a permissive bound that passes for
    healthy calibrators but fails if the rate is wrong (e.g., calibrator
    plateaus or shrinks too slowly with n).

    Construction
    ------------
    Fit calibrators at n_calib in {500, 1500, 5000, 15000}. For each:
        L2_avg(n)  =  mean_t  L2_t(n)
                    where t panel = {0.10, 0.30, 0.50, 1.00}.
    Linear-regress  log L2_avg  on  log n_calib;  assert slope < -0.20.

    What this catches
    -----------------
    A calibrator that's "lucky at one n" but doesn't actually converge —
    the prior `l2_gap_shrinks_with_n` test only checks two points; the
    rate test fits a slope across four points and is the asymptotic check.
    """
    from ..mc.dgp import DGP, DEFAULT_POLICIES

    dgp = DGP(DEFAULT_POLICIES)
    t_grid = np.linspace(0.0, 1.0, 61)
    test_t_indices = [int(np.argmin(abs(t_grid - t))) for t in (0.10, 0.30, 0.50, 1.00)]
    n_calibs = (500, 1500, 5000, 15000)

    for policy in ("uniform", "left_skew"):
        df_huge = dgp.sample(policy, n=50_000, with_oracle=True, seed=99999)
        cg_huge = fit_calibrator_grid(
            df_huge["s"].to_numpy(), df_huge["y"].to_numpy(), t_grid,
        )

        s_test = dgp.sample(policy, n=500, with_oracle=False, seed=12345)["s"].to_numpy()
        H_huge = cg_huge.predict(s_test)

        avg_l2_by_n: dict[int, float] = {}
        for n in n_calibs:
            df = dgp.sample(policy, n=n, with_oracle=True, seed=2000 + n)
            cg = fit_calibrator_grid(
                df["s"].to_numpy(), df["y"].to_numpy(), t_grid,
            )
            H = cg.predict(s_test)
            l2_per_t = [
                float(np.sqrt(np.mean((H[:, j] - H_huge[:, j]) ** 2)))
                for j in test_t_indices
            ]
            avg_l2_by_n[n] = float(np.mean(l2_per_t))

        log_n = np.log(np.array(n_calibs, dtype=np.float64))
        log_l2 = np.log(np.array([avg_l2_by_n[n] for n in n_calibs], dtype=np.float64))
        slope = float(np.polyfit(log_n, log_l2, 1)[0])

        assert slope < -0.20, (
            f"{policy}: L2 convergence rate slope = {slope:.3f} >= -0.20. "
            f"Calibrator is converging too slowly (theory: ~-0.33 for isotonic). "
            f"avg_l2_by_n = {dict((k, round(v, 4)) for k, v in avg_l2_by_n.items())}"
        )


def test_y_out_of_range_rejected() -> None:
    s = np.linspace(-2, 2, 10)
    y_bad = np.linspace(-0.1, 1.1, 10)  # outside [0,1]
    with pytest.raises(ValueError, match="must be in"):
        fit_calibrator_grid(s, y_bad, np.array([0.5]))


def test_fold_id_requires_K() -> None:
    s, y = _toy_data(n=50)
    fold_id = np.zeros(50, dtype=np.int64)
    with pytest.raises(ValueError, match="K must be provided"):
        fit_calibrator_grid(s, y, np.array([0.3, 0.5]), fold_id=fold_id)


# -----------------------------------------------------------------------------
# make_t_grid: grid factory used by Config-driven callers (pipeline, runner).
# -----------------------------------------------------------------------------


def test_make_t_grid_uniform_matches_linspace() -> None:
    """`uniform` kind must reproduce the legacy np.linspace exactly."""
    grid = make_t_grid(0.0, 1.0, 61, grid_kind="uniform")
    expected = np.linspace(0.0, 1.0, 61)
    assert grid.shape == expected.shape
    assert np.allclose(grid, expected)
    assert grid[0] == 0.0 and grid[-1] == 1.0


def test_make_t_grid_tail_dense_endpoints_and_size() -> None:
    """Tail-dense grid spans [grid_lo, grid_hi] and has |T| == grid_size."""
    grid = make_t_grid(0.0, 1.0, 61, grid_kind="tail_dense")
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert len(grid) == 61
    # Strictly increasing.
    assert (np.diff(grid) > 0).all()


def test_make_t_grid_tail_dense_concentrates_below_breakpoint() -> None:
    """At least 30% of points (~21 of 61) sit in [0, 0.1] for the default."""
    grid = make_t_grid(0.0, 1.0, 61, grid_kind="tail_dense")
    n_below = int((grid <= 0.1 + 1e-12).sum())
    assert n_below >= 21, f"expected ≥21 points in [0, 0.1]; got {n_below}"
    # And uniform reference puts only 7 there.
    uniform = make_t_grid(0.0, 1.0, 61, grid_kind="uniform")
    assert n_below > int((uniform <= 0.1 + 1e-12).sum())


def test_make_t_grid_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown grid_kind"):
        make_t_grid(0.0, 1.0, 61, grid_kind="bogus")  # type: ignore[arg-type]


def test_make_t_grid_alpha_one_collapse_unchanged() -> None:
    """
    Both grid kinds must include the right endpoint t=1 so the α=1 collapse
    identity (t̂_1 == 1) holds. This is the load-bearing structural property.
    """
    for kind in ("uniform", "tail_dense"):
        grid = make_t_grid(0.0, 1.0, 61, grid_kind=kind)
        assert grid[-1] == 1.0, f"{kind}: missing right endpoint"
