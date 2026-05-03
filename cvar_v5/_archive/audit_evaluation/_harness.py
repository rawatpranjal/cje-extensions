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
    ḡ: np.ndarray                  # (2,)
    t_hat: float
    omegas: dict                   # method_name -> Ω̂ (2x2)
    g_per_fold: np.ndarray         # (K, 2) for jackknife-bias-correction layer


def run_one_rep(
    p: DGPParams, pert: TargetPert, alpha: float,
    n_calib: int, n_audit: int, n_eval: int,
    t_grid: np.ndarray, K: int, B: int, seed: int,
) -> RepResult:
    """
    One pipeline rep:
        CALIB ← logger (always unperturbed; we test against pert in target).
        EVAL  ← target (with pert).
        AUDIT ← target (with pert).

    Returns ḡ, t̂, all Ω̂_method, plus per-fold ḡ^(-k) for downstream
    bias-correction.
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

    omegas: dict = {}

    omegas["analytical"] = omega_analytical(g1, g2)

    omegas["boot_remax"] = omega_boot_remax(
        s_a, y_a, H_a, t_grid, t_idx, t_hat, alpha, B, seed=seed + 3, ridge=False,
    )
    omegas["boot_remax_ridge"] = omega_boot_remax(
        s_a, y_a, H_a, t_grid, t_idx, t_hat, alpha, B, seed=seed + 3, ridge=True,
    )

    v_cal, g_per_fold = v_cal_jackknife_g(cal, s_a, y_a, t_idx, t_hat, alpha)
    omegas["analytical_oua"] = omegas["analytical"] + v_cal
    omegas["boot_remax_oua"] = omegas["boot_remax"] + v_cal

    return RepResult(
        ḡ=ḡ, t_hat=t_hat, omegas=omegas, g_per_fold=g_per_fold,
    )
