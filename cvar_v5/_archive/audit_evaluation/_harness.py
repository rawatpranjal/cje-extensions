"""
Per-rep pipeline + per-method Ω̂ collection.

For each MC rep r:
    draw (calib^(r), eval^(r), audit^(r)) under H_0
    fit ĥ^(r), compute t̂^(r)
    compute ḡ^(r) at the original (ĥ^(r), t̂^(r))
    for each method M, compute Ω̂_M^(r) and ḡ_used_M^(r) (post bias-correction)

Methods covered:
    - "analytical"          : Σ̂_per_row / n_audit
    - "boot_remax"          : audit-row bootstrap with re-max, no ridge
    - "boot_remax_ridge"    : above + λI with λ = 1/n_audit
    - "analytical_oua"      : analytical + jackknife V_cal_g (additive)
    - "boot_remax_oua"      : boot_remax + jackknife V_cal_g (additive)
    - "full_pipeline_boot"  : refit cal + remax + recompute g, single integrated cov
    - "oracle_var"          : Σ_oracle / n_audit  (synthetic, sanity probe)
    - "oracle_full_var"     : Σ_full / n_audit    (synthetic, sanity probe; uses Σ_full)
    - "inflated_oracle"     : 4 · Σ_full / n_audit (synthetic, sanity probe)

ḡ_used (per method):
    - default: ḡ                                   (no bias correction)
    - method "+_bc": ḡ − bias_jk                    (jackknife bias correction)
    - synthetic "_oracle_bc": ḡ − ε_oracle           (uses oracle ε from MC)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ._truths import DGPParams, TargetPert, target_params, t_star


# ---------------- DGP sampling ----------------------------------------------


def sample(p: DGPParams, pert: TargetPert, n: int, seed: int, with_oracle: bool):
    rng = np.random.default_rng(seed)
    a, b, sc, sg = target_params(p, pert)
    y = rng.beta(a, b, size=n)
    s = sc * (y - 0.5) + rng.normal(0.0, sg, size=n)
    return s, (y if with_oracle else None)


def sample_logger(p: DGPParams, n: int, seed: int):
    """Logger always uses unperturbed params (logger == base policy)."""
    return sample(p, TargetPert(), n, seed, with_oracle=True)


# ---------------- Calibrator (cross-fit, K folds) ---------------------------


@dataclass(frozen=True)
class Calibrator:
    t_grid: np.ndarray
    pooled: list                # list[|T|] of IsotonicRegression
    folded: list                # list[|T|][K]
    fold_id: np.ndarray         # (n_calib,) values in [0, K)
    K: int

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s).ravel()
        out = np.empty((len(s), len(self.t_grid)), dtype=np.float64)
        for j, ir in enumerate(self.pooled):
            out[:, j] = ir.predict(s)
        return out


def fit_calibrator(s_calib, y_calib, t_grid, K, seed) -> Calibrator:
    rng = np.random.default_rng(seed)
    fold_id = rng.integers(0, K, size=len(s_calib))
    pooled = []
    folded = [[] for _ in range(len(t_grid))]
    for j, t in enumerate(t_grid):
        z = np.maximum(t - y_calib, 0.0)
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(s_calib, z)
        pooled.append(ir)
        for k in range(K):
            mask = fold_id != k
            ir_k = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(s_calib[mask], z[mask])
            folded[j].append(ir_k)
    return Calibrator(t_grid=t_grid, pooled=pooled, folded=folded, fold_id=fold_id, K=K)


# ---------------- Estimator + g moments ------------------------------------


def saddle_argmax(H: np.ndarray, t_grid: np.ndarray, alpha: float) -> tuple[int, float]:
    psi = t_grid - H.mean(axis=0) / alpha
    j = int(np.argmax(psi))
    return j, float(t_grid[j])


def g_pair(s, y, h_t, t, alpha):
    g1 = (y <= t).astype(np.float64) - alpha
    g2 = np.maximum(t - y, 0.0) - h_t
    return g1, g2


# ---------------- Per-method Ω̂ computers -----------------------------------


def omega_analytical(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    cov = np.cov(np.vstack([g1, g2]), ddof=1)
    return cov / len(g1)


def omega_boot_remax(s, y, H, t_grid, t_idx, t_hat, alpha, B, seed, ridge):
    rng = np.random.default_rng(seed)
    n = len(s)
    g_b = np.empty((B, 2))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        s_b = s[idx]; y_b = y[idx]; H_b = H[idx, :]
        j_b, t_b = saddle_argmax(H_b, t_grid, alpha)
        g1_b, g2_b = g_pair(s_b, y_b, H_b[:, j_b], t_b, alpha)
        g_b[b] = [g1_b.mean(), g2_b.mean()]
    omega = np.cov(g_b, rowvar=False, ddof=1)
    if ridge:
        omega = omega + (1.0 / n) * np.eye(2)
    return omega


def v_cal_jackknife_g(cal: Calibrator, s_audit, y_audit, t_idx, t_hat, alpha) -> np.ndarray:
    """V̂_cal_g via leave-one-fold-out calibrators at fixed t̂."""
    K = cal.K
    g_per_fold = np.empty((K, 2))
    for k, ir_k in enumerate(cal.folded[t_idx]):
        h_t_k = ir_k.predict(s_audit)
        g1_k, g2_k = g_pair(s_audit, y_audit, h_t_k, t_hat, alpha)
        g_per_fold[k] = [g1_k.mean(), g2_k.mean()]
    diffs = g_per_fold - g_per_fold.mean(axis=0)
    return ((K - 1) / K) * (diffs.T @ diffs), g_per_fold


def omega_full_pipeline_boot(p: DGPParams, pert: TargetPert, alpha, n_calib, n_audit,
                             n_eval, K, t_grid, B, seed):
    """
    Single integrated bootstrap: for each rep, refit ĥ on bootstrap CALIB
    rows, re-max t̂ on bootstrap EVAL, recompute ḡ on bootstrap AUDIT.
    Returns the empirical covariance.
    """
    # Need access to the original (calib, eval, audit) — passed from caller.
    raise NotImplementedError("full_pipeline_boot requires sample arrays; see harness driver")


# ---------------- Single-rep pipeline + all-methods compute -----------------


@dataclass
class RepResult:
    """
    Output of one MC rep of the audit pipeline.

    Math contract:
        ḡ            =  (1/n_audit) · Σ_i  g_i( t̂ ; ĥ )            ∈ R²
        t̂            =  argmax over T_grid of [ t − mean_eval(ĥ_t(s_e)) / α ]

        Per-fold quantities (for K-fold cross-fit calibrator):
            ĥ^(-k)         calibrator with fold k held out (cached on cal.folded)
            t̂^(-k)         argmax over T_grid of [ t − mean_eval( ĥ_t^(-k)(s_e) ) / α ]
            ḡ^(-k)_at_t̂   = (1/n_audit) · Σ_i  g_i( t̂ ; ĥ^(-k) )      (varies ĥ only)
            ḡ^(-k)_at_t̂_k = (1/n_audit) · Σ_i  g_i( t̂^(-k) ; ĥ^(-k) ) (varies ĥ AND t̂)

        Raw arrays are kept so bootstrap-based bias correctors can resample
        them without re-drawing from the DGP each rep (the raw draw IS the rep).

    Fields:
        ḡ                   the realized audit moment vector
        t_hat               saddle-point optimum on EVAL with pooled ĥ
        omegas              per-method Ω̂ estimates (variance-side)
        g_per_fold_at_t_hat ḡ^(-k) computed at the original t̂ (for BC-jk-cal)
        g_per_fold_at_t_k   ḡ^(-k) computed at fold-specific t̂^(-k) (for BC-jk-full)
        t_hat_per_fold      t̂^(-k), one per fold k
        s_calib, y_calib    raw CALIB rows (for BC-boot)
        s_eval              raw EVAL rows (for BC-boot)
        s_audit, y_audit    raw AUDIT rows (for BC-boot)
    """

    ḡ: np.ndarray                       # (2,)
    t_hat: float
    omegas: dict                        # method_name -> Ω̂ (2x2)
    g_per_fold_at_t_hat: np.ndarray     # (K, 2) — bias correction A: ĥ varies, t̂ fixed
    g_per_fold_at_t_k: np.ndarray       # (K, 2) — bias correction B: ĥ AND t̂ vary
    t_hat_per_fold: np.ndarray          # (K,)   — fold-specific t̂^(-k)
    # Raw arrays for bootstrap-based bias correctors (BC-boot):
    s_calib: np.ndarray                 # (n_calib,)
    y_calib: np.ndarray                 # (n_calib,)
    s_eval: np.ndarray                  # (n_eval,)
    s_audit: np.ndarray                 # (n_audit,)
    y_audit: np.ndarray                 # (n_audit,)
    # The first column of g_per_fold_at_t_hat will equal ḡ[0] in every entry,
    # since g_1 = 1{Y ≤ t̂} − α does not depend on ĥ.


def _per_fold_quantities(
    cal: Calibrator, s_audit: np.ndarray, y_audit: np.ndarray,
    s_eval: np.ndarray, t_grid: np.ndarray, t_idx_pooled: int, t_hat_pooled: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each oracle fold k = 1..K, compute:
        h_t_k_at_audit                          ĥ_t^(-k)(s_audit) for all t in T_grid
        t̂^(-k)  := argmax  [ t − mean_e ĥ_t^(-k)(s_e) / α ]   (re-max on EVAL with ĥ^(-k))
        ḡ^(-k) at t̂_pooled                    g_i evaluated at t̂_pooled, ĥ^(-k)
        ḡ^(-k) at t̂^(-k)                      g_i evaluated at t̂^(-k),    ĥ^(-k)

    Returns:
        g_per_fold_at_t_hat  (K, 2) — g_1 column equals (1{Y≤t̂}−α).mean() for every k
        g_per_fold_at_t_k    (K, 2) — g_1 column varies with t̂^(-k)
        t_hat_per_fold       (K,)   — t̂^(-k) values
    """
    K = cal.K
    g_at_t_hat = np.empty((K, 2), dtype=np.float64)
    g_at_t_k = np.empty((K, 2), dtype=np.float64)
    t_hat_per_fold = np.empty(K, dtype=np.float64)

    for k in range(K):
        # ĥ_t^(-k) on every t in T_grid — needed for argmax on EVAL.
        # Each cal.folded[j][k] is the fitted IsotonicRegression for the j-th t,
        # k-th held-out fold.
        H_eval_k = np.empty((len(s_eval), len(t_grid)), dtype=np.float64)
        H_audit_k = np.empty((len(s_audit), len(t_grid)), dtype=np.float64)
        for j in range(len(t_grid)):
            ir_jk = cal.folded[j][k]
            H_eval_k[:, j] = ir_jk.predict(s_eval)
            H_audit_k[:, j] = ir_jk.predict(s_audit)

        # ḡ^(-k) at the original t̂ (BC-jk-cal style, varies ĥ only)
        h_t_audit_k = H_audit_k[:, t_idx_pooled]
        g1_a, g2_a = g_pair(s_audit, y_audit, h_t_audit_k, t_hat_pooled, alpha)
        g_at_t_hat[k] = [g1_a.mean(), g2_a.mean()]

        # ḡ^(-k) at fold-specific t̂^(-k) (BC-jk-full style, varies both)
        # t̂^(-k) := argmax  t − (1/(α n_eval)) Σ ĥ_t^(-k)(s_e)
        psi_k = t_grid - H_eval_k.mean(axis=0) / alpha
        j_k = int(np.argmax(psi_k))
        t_hat_k = float(t_grid[j_k])
        h_t_audit_at_k = H_audit_k[:, j_k]
        g1_b, g2_b = g_pair(s_audit, y_audit, h_t_audit_at_k, t_hat_k, alpha)
        g_at_t_k[k] = [g1_b.mean(), g2_b.mean()]
        t_hat_per_fold[k] = t_hat_k

    return g_at_t_hat, g_at_t_k, t_hat_per_fold


def run_one_rep(
    p: DGPParams, pert: TargetPert, alpha: float,
    n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, seed: int,
) -> RepResult:
    """
    One pipeline rep.

    Slices:
        CALIB ← logger (always unperturbed; calibrator is fit here).
        EVAL  ← target (with pert).  Drives t̂ via the saddle objective.
        AUDIT ← target (with pert).  Y observed; where g_i moments are evaluated.

    Computed:
        ĥ_t        pooled IsotonicRegression(decreasing) per t in T_grid
        t̂          argmax_t  [ t − mean_eval(ĥ_t(s_e)) / α ]
        ḡ          (1/n_audit) Σ_i  g_i(t̂; ĥ)
        Ω̂_method   five Ω̂ estimators (analytical, boot_remax, ridge, oua-variants)
        ḡ^(-k) at t̂_pooled       per-fold (BC-jk-cal)
        ḡ^(-k) at fold-specific t̂^(-k)  per-fold (BC-jk-full)
        Raw arrays kept for BC-boot.
    """
    s_c, y_c = sample_logger(p, n_calib, seed=seed)
    cal = fit_calibrator(s_c, y_c, t_grid, K, seed=seed)

    s_e, _ = sample(p, pert, n_eval, seed=seed + 1, with_oracle=False)
    H_e = cal.predict(s_e)
    t_idx, t_hat = saddle_argmax(H_e, t_grid, alpha)

    s_a, y_a = sample(p, pert, n_audit, seed=seed + 2, with_oracle=True)
    H_a = cal.predict(s_a)
    h_t_a = H_a[:, t_idx]
    g1, g2 = g_pair(s_a, y_a, h_t_a, t_hat, alpha)
    ḡ = np.array([g1.mean(), g2.mean()])

    # Variance-side Ω̂ estimators (orthogonal to bias correction).
    omegas: dict = {}
    omegas["analytical"] = omega_analytical(g1, g2)
    omegas["boot_remax"] = omega_boot_remax(
        s_a, y_a, H_a, t_grid, t_idx, t_hat, alpha, B, seed=seed + 3, ridge=False,
    )
    omegas["boot_remax_ridge"] = omega_boot_remax(
        s_a, y_a, H_a, t_grid, t_idx, t_hat, alpha, B, seed=seed + 3, ridge=True,
    )
    v_cal, _ = v_cal_jackknife_g(cal, s_a, y_a, t_idx, t_hat, alpha)
    omegas["analytical_oua"] = omegas["analytical"] + v_cal
    omegas["boot_remax_oua"] = omegas["boot_remax"] + v_cal
    # The two NEW variance methods from _variance_methods_extra are NOT
    # computed here — they require knowing B_cal / B_full at call time
    # and consume the raw arrays. The caller (run_evaluation) computes
    # them after run_one_rep returns, using the raw arrays exposed below.

    # Per-fold quantities for BC-jk-cal and BC-jk-full.
    g_at_t_hat, g_at_t_k, t_per_fold = _per_fold_quantities(
        cal, s_a, y_a, s_e, t_grid, t_idx, t_hat, alpha,
    )

    return RepResult(
        ḡ=ḡ, t_hat=t_hat, omegas=omegas,
        g_per_fold_at_t_hat=g_at_t_hat,
        g_per_fold_at_t_k=g_at_t_k,
        t_hat_per_fold=t_per_fold,
        s_calib=s_c, y_calib=y_c,
        s_eval=s_e,
        s_audit=s_a, y_audit=y_a,
    )
