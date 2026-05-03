"""
Variance methods that go beyond the existing five (analytical, boot_remax,
boot_remax_ridge, analytical_oua, boot_remax_oua) implemented in
production audit.py.

The two methods here address gaps the existing methods can't close:

    boot_v_cal_oua    : analytical V_audit + bootstrap V_cal_g.
                        Replaces the JACKKNIFE V̂_cal_g (in analytical_oua)
                        with a bootstrap V̂_cal_g_boot. Jackknife
                        under-estimates V_calib for non-smooth (isotonic)
                        calibrators (Efron 1992); bootstrap captures it
                        properly. Captures N1 + N2 (additive).

    full_pipeline_boot: refit ĥ + re-max t̂ + recompute ḡ in each bootstrap
                        rep; Ω̂ = sample-cov_b(ḡ^(b)). The cvar_v4 paper's
                        prescription. Captures N1 + N2 + N3 + cross-terms
                        in a single integrated estimate.

Notation
--------
    n_audit, n_calib, n_eval     slice sizes
    K                            cross-fit folds in calibrator
    t̂                            saddle-point optimum from EVAL with pooled ĥ
    g_pair(s, y, h_t, t, α)      per-row (g_1i, g_2i) at threshold t with
                                 calibrator predictions h_t

Both methods consume `RepResult` (which now exposes the raw arrays
s_calib, y_calib, s_audit, y_audit, s_eval) and return a 2x2 Ω̂.
"""

from __future__ import annotations

import numpy as np

from ._harness import (
    RepResult, fit_calibrator, g_pair, omega_analytical, saddle_argmax,
)


def boot_v_cal_oua(
    rep: RepResult,
    t_grid: np.ndarray,
    alpha: float,
    K: int,
    B_cal: int,
    seed: int,
) -> np.ndarray:
    """
    Analytical Ω̂_audit + bootstrap V̂_cal_g (replacing jackknife).

    Math
    ----
        Ω̂_audit  =  (1 / (n_audit − 1)) · Σ_i  (g_i − ḡ)(g_i − ḡ)ᵀ  /  n_audit
                    (the analytical per-row cov, fixed ĥ and t̂)

        For b = 1..B_cal:
            idx_b      ~ Multinomial(n_calib, 1/n_calib · 1)
            ĥ^(b)      := refit isotonic on (s_c[idx_b], y_c[idx_b]) per t in T_grid
            h_t̂^(b)(s_audit) := apply at the ORIGINAL t̂ (no re-max)
            g_i^(b)    := (1{Y_i ≤ t̂} − α,  (t̂ − Y_i)_+ − h_t̂^(b)(s_i))
            ḡ^(b)      := (1 / n_audit) · Σ_i  g_i^(b)             ∈ R²

        V̂_cal_g_boot  =  (1 / (B_cal − 1)) · Σ_b (ḡ^(b) − ḡ̄_boot)(ḡ^(b) − ḡ̄_boot)ᵀ
                          ∈ R^{2×2}

        Ω̂_boot_v_cal_oua  =  Ω̂_audit  +  V̂_cal_g_boot

    What it captures
    ----------------
    Same shape as analytical_oua:  audit-side conditional variance (N1)
    plus calibrator-fit variance (N2). The improvement is that bootstrap
    V̂_cal_g_boot estimates Var_calib(ḡ) correctly even on non-smooth
    isotonic calibrators, where the jackknife V̂_cal_g is downward-biased.

    What it misses
    --------------
    Argmax-on-grid (N3) and N2×N3 cross-terms. By construction t̂ is
    fixed at the original value; the bootstrap doesn't re-max.

    Cost
    ----
    B_cal · |T_grid| isotonic fits per audit verdict. With K=5 cross-fit
    in the original calibrator, the bootstrap calibrator IS NOT cross-fit
    (B_cal samples is enough variation; cross-fitting inside the bootstrap
    loop would multiply cost without statistical gain at this stage).
    """
    n_calib = len(rep.s_calib)
    rng = np.random.default_rng(seed)

    # ḡ_audit and Ω̂_audit at the original (ĥ, t̂)
    s_a = rep.s_audit
    y_a = rep.y_audit
    g_b_per_rep = np.empty((B_cal, 2), dtype=np.float64)

    for b in range(B_cal):
        idx_b = rng.integers(0, n_calib, size=n_calib)
        s_c_b = rep.s_calib[idx_b]
        y_c_b = rep.y_calib[idx_b]

        # Refit calibrator on bootstrapped CALIB. K=1 (no cross-fit needed
        # for the bootstrap variance estimate — we just want ĥ^(b) at the
        # original t̂).
        cal_b = fit_calibrator(s_c_b, y_c_b, t_grid, K=K, seed=seed + b)

        # Apply at the ORIGINAL t̂ — find its index in t_grid.
        # t_grid is fixed; rep.t_hat is one of its values.
        t_idx = int(np.argmin(np.abs(t_grid - rep.t_hat)))
        h_t_audit_b = cal_b.predict(s_a)[:, t_idx]

        g1_b, g2_b = g_pair(s_a, y_a, h_t_audit_b, rep.t_hat, alpha)
        g_b_per_rep[b] = [g1_b.mean(), g2_b.mean()]

    # Bootstrap variance of ḡ across calibrator resamples.
    v_cal_boot = np.cov(g_b_per_rep, rowvar=False, ddof=1)

    # Reconstruct ḡ_audit and analytical Ω̂_audit from (s_a, y_a, ĥ_pooled).
    # We don't have a stored Ω̂_analytical on the rep, but rep.omegas['analytical']
    # IS exactly what we want.
    omega_audit = rep.omegas["analytical"]

    return omega_audit + v_cal_boot


def full_pipeline_boot(
    rep: RepResult,
    t_grid: np.ndarray,
    alpha: float,
    K: int,
    B_full: int,
    seed: int,
) -> np.ndarray:
    """
    Full-pipeline cluster bootstrap as a variance estimator.

    Math
    ----
        For b = 1..B_full:
            idx_c^(b)  ~ Multinomial(n_calib, 1/n_calib · 1)
            idx_e^(b)  ~ Multinomial(n_eval,  1/n_eval  · 1)
            idx_a^(b)  ~ Multinomial(n_audit, 1/n_audit · 1)

            s_c^(b), y_c^(b)  := rep.s_calib[idx_c^(b)],  rep.y_calib[idx_c^(b)]
            s_e^(b)            := rep.s_eval[idx_e^(b)]
            s_a^(b), y_a^(b)  := rep.s_audit[idx_a^(b)],  rep.y_audit[idx_a^(b)]

            ĥ^(b)        := refit isotonic on (s_c^(b), y_c^(b)) per t in T_grid
            t̂^(b)        := argmax_t [t − mean_{eval^(b)}(ĥ^(b)_t(s_e^(b))) / α]
            h_t̂^(b)(s_a^(b)) := bootstrap-calibrator predictions at t̂^(b) on bootstrap audit

            g_i^(b)      := (1{y_a^(b)_i ≤ t̂^(b)} − α,
                             (t̂^(b) − y_a^(b)_i)_+ − h_t̂^(b)(s_a^(b)_i))
            ḡ^(b)        := (1 / n_audit^(b)) · Σ_i  g_i^(b)         ∈ R²

        Ω̂_full_pipeline_boot  =  (1/(B_full − 1)) · Σ_b (ḡ^(b) − ḡ̄_b)(ḡ^(b) − ḡ̄_b)ᵀ

    What it captures
    ----------------
    All three nuisances jointly INCLUDING cross-terms:
        N1: each rep resamples AUDIT
        N2: each rep refits ĥ on resampled CALIB
        N3: each rep re-maxes t̂ on resampled EVAL
        cross: by construction (the bootstrap covariance integrates over
               the joint randomness)

    This is the cvar_v4 paper's prescription:
    `cvar_v4/papers/2512.11150_causal_judge_evaluation/sections/appendix_algorithms.tex:101-118`.

    Distinct from `bc_boot`
    -----------------------
    `bc_boot` uses `mean_b(ḡ^(b))` as a BIAS estimator; the bias-corrected
    ḡ_bc = ḡ − bias_boot is then plugged into a standard Wald with some
    other Ω̂. `full_pipeline_boot` uses `Var_b(ḡ^(b))` as the VARIANCE
    estimator and does not touch the center. The same loop produces both
    quantities; they are conceptually orthogonal layers of the audit
    machinery.

    Cost
    ----
    B_full · |T_grid| isotonic fits per audit verdict, plus B_full saddle
    argmaxes on the bootstrap eval slice. At B_full=80, n_calib=600,
    |T_grid|=121, this is about 30 sec per rep on one core.
    """
    n_c = len(rep.s_calib)
    n_e = len(rep.s_eval)
    n_a = len(rep.s_audit)
    rng = np.random.default_rng(seed)
    g_b_per_rep = np.empty((B_full, 2), dtype=np.float64)

    for b in range(B_full):
        # Resample each slice independently with replacement.
        idx_c = rng.integers(0, n_c, size=n_c)
        idx_e = rng.integers(0, n_e, size=n_e)
        idx_a = rng.integers(0, n_a, size=n_a)

        s_c_b = rep.s_calib[idx_c]
        y_c_b = rep.y_calib[idx_c]
        s_e_b = rep.s_eval[idx_e]
        s_a_b = rep.s_audit[idx_a]
        y_a_b = rep.y_audit[idx_a]

        # Refit calibrator on bootstrap CALIB.
        cal_b = fit_calibrator(s_c_b, y_c_b, t_grid, K=K, seed=seed + b)

        # Re-max t̂^(b) on bootstrap EVAL with bootstrap calibrator.
        H_e_b = cal_b.predict(s_e_b)
        j_b, t_b = saddle_argmax(H_e_b, t_grid, alpha)

        # Compute ḡ^(b) on bootstrap AUDIT at (ĥ^(b), t̂^(b)).
        h_t_a_b = cal_b.predict(s_a_b)[:, j_b]
        g1_b, g2_b = g_pair(s_a_b, y_a_b, h_t_a_b, t_b, alpha)
        g_b_per_rep[b] = [g1_b.mean(), g2_b.mean()]

    return np.cov(g_b_per_rep, rowvar=False, ddof=1)
