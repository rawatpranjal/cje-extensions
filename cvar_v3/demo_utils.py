# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

# ---------------------------------------------------------------------------
# Estimation knobs
# ---------------------------------------------------------------------------
SEED = 123                  # master seed for full reproducibility
ALPHA = 0.10                # CVaR tail level (worst 10%)
GRID_SIZE = 61              # threshold grid points for CVaR
TRUTH_N = 400_000           # large sample for ground truth
N_CALIB = 40_000            # calibration sample size
N_EVAL = 60_000             # evaluation sample size
REPLICATIONS = 30          # Monte Carlo replications for estimation
VALIDATION_FRAC = 0.20      # train/val split within calibration sample

# ---------------------------------------------------------------------------
# Audit knobs
# ---------------------------------------------------------------------------
N_PROBE_TOTAL = 2_400       # total audit oracle budget per variant
PILOT_FRAC = 0.50           # fraction for pilot vs audit split
WALD_ALPHA = 0.05           # significance level for Wald test
AUDIT_REPLICATIONS = 30    # Monte Carlo replications for audit

# ---------------------------------------------------------------------------
# Sensitivity sweep knobs
# ---------------------------------------------------------------------------
ALPHA_GRID = (0.05, 0.10, 0.20)
ORACLE_BUDGET_GRID = (600, 1_200, 2_400, 4_800)
N_CALIB_GRID = (500, 1_000, 2_000, 5_000, 10_000, 40_000)
SENS_REPLICATIONS = 15      # reduced reps per sensitivity cell

# ---------------------------------------------------------------------------
# Plot knobs
# ---------------------------------------------------------------------------
SAMPLE_PLOT_N = 80_000      # max samples for distribution plots
BINS = 70                   # histogram bins
SCORE_DECILES = 10          # number of decile bins
FIG_WIDE = (14, 10)
FIG_STD = (8, 5)
DPI = 150

# ---------------------------------------------------------------------------
# Audit variant names
# ---------------------------------------------------------------------------
AUDIT_VARIANTS = ("stable", "mild_shift", "hard_shift", "fooled_judge", "weird_monotone")

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    # Policy A
    a_base_intercept: float
    a_base_slope: float
    a_cat_base: float
    a_cat_extra: float
    a_cat_shift: float
    a_cat_slope: float
    a_cat_y_intercept: float
    a_cat_y_slope: float
    a_cat_noise: float
    # Policy B
    b_base_intercept: float
    b_base_slope: float
    b_noise: float
    # Judge
    a_judge_noise: float
    a_judge_shift_strength: float
    b_judge_noise: float
    # Misc
    x_nonlin_strength: float = 0.0
    hard_case_bonus: float = 0.0
    hard_case_threshold: float = 0.60

SCENARIOS = {
    "clear_tail_tradeoff": ScenarioConfig(
        name="clear_tail_tradeoff",
        description="A wins on mean, B wins on lower-tail CVaR. Main demo regime.",
        a_base_intercept=10.40, a_base_slope=0.18,
        a_cat_base=0.05, a_cat_extra=0.10, a_cat_shift=0.10, a_cat_slope=2.5,
        a_cat_y_intercept=0.55, a_cat_y_slope=0.30, a_cat_noise=0.40,
        b_base_intercept=8.45, b_base_slope=0.34, b_noise=0.42,
        a_judge_noise=0.80, a_judge_shift_strength=0.75, b_judge_noise=0.58,
        x_nonlin_strength=0.15,
    ),
    "no_tradeoff": ScenarioConfig(
        name="no_tradeoff",
        description="A dominates on both mean and CVaR. Placebo regime.",
        a_base_intercept=9.10, a_base_slope=0.16,
        a_cat_base=0.01, a_cat_extra=-0.50, a_cat_shift=0.50, a_cat_slope=2.0,
        a_cat_y_intercept=2.50, a_cat_y_slope=0.20, a_cat_noise=0.35,
        b_base_intercept=7.80, b_base_slope=0.30, b_noise=0.45,
        a_judge_noise=0.72, a_judge_shift_strength=0.65, b_judge_noise=0.60,
        x_nonlin_strength=0.10,
    ),
    "hard_small_gap": ScenarioConfig(
        name="hard_small_gap",
        description="Smaller mean/CVaR gaps, noisier calibration. Selection is materially harder.",
        a_base_intercept=9.60, a_base_slope=0.16,
        a_cat_base=0.04, a_cat_extra=0.06, a_cat_shift=0.10, a_cat_slope=2.8,
        a_cat_y_intercept=0.55, a_cat_y_slope=0.35, a_cat_noise=0.50,
        b_base_intercept=8.70, b_base_slope=0.25, b_noise=0.48,
        a_judge_noise=0.95, a_judge_shift_strength=0.85, b_judge_noise=0.70,
        x_nonlin_strength=0.20, hard_case_bonus=0.15,
    ),
    "weird_nonlinear": ScenarioConfig(
        name="weird_nonlinear",
        description="Highly nonlinear judge, heterogeneous issue mix. Stress-tests isotonic assumption.",
        a_base_intercept=10.10, a_base_slope=0.12,
        a_cat_base=0.04, a_cat_extra=0.12, a_cat_shift=-0.02, a_cat_slope=3.0,
        a_cat_y_intercept=0.45, a_cat_y_slope=0.40, a_cat_noise=0.45,
        b_base_intercept=8.30, b_base_slope=0.28, b_noise=0.45,
        a_judge_noise=0.88, a_judge_shift_strength=0.95, b_judge_noise=0.66,
        x_nonlin_strength=0.35, hard_case_bonus=0.25,
    ),
    "knife_edge": ScenarioConfig(
        name="knife_edge",
        description="Catastrophe rate near alpha. Threshold selection becomes fragile.",
        a_base_intercept=9.90, a_base_slope=0.14,
        a_cat_base=0.04, a_cat_extra=-0.05, a_cat_shift=0.05, a_cat_slope=2.8,
        a_cat_y_intercept=0.60, a_cat_y_slope=0.25, a_cat_noise=0.38,
        b_base_intercept=8.55, b_base_slope=0.30, b_noise=0.44,
        a_judge_noise=0.82, a_judge_shift_strength=0.70, b_judge_noise=0.60,
        x_nonlin_strength=0.12,
    ),
    "reversed_judge": ScenarioConfig(
        name="reversed_judge",
        description="Judge quality varies: good for B, noisy for A. Tests asymmetric calibration.",
        a_base_intercept=10.20, a_base_slope=0.20,
        a_cat_base=0.05, a_cat_extra=0.08, a_cat_shift=0.05, a_cat_slope=2.5,
        a_cat_y_intercept=0.50, a_cat_y_slope=0.30, a_cat_noise=0.42,
        b_base_intercept=8.50, b_base_slope=0.32, b_noise=0.40,
        a_judge_noise=1.20, a_judge_shift_strength=0.60, b_judge_noise=0.45,
        x_nonlin_strength=0.10,
    ),
}

print("Scenarios:", list(SCENARIOS.keys()))
print(f"Estimation: {REPLICATIONS} reps, n_calib={N_CALIB}, n_eval={N_EVAL}, alpha={ALPHA}")
print(f"Audit: {AUDIT_REPLICATIONS} reps, n_probe={N_PROBE_TOTAL}, {len(AUDIT_VARIANTS)} variants")
print(f"Sensitivity: {SENS_REPLICATIONS} reps per cell")


# ===================================================================
# UTILITIES
# ===================================================================

def child_seed(seed, *keys):
    # Deterministic child seed using hashlib (stable across Python sessions).
    # Python's built-in hash() is randomized by default (PYTHONHASHSEED),
    # so we use md5 for reproducibility across runs.
    import hashlib
    h = str(seed)
    for key in keys:
        h = hashlib.md5(f'{h}:{key}'.encode()).hexdigest()
    return int(h, 16) % (2**32 - 1)

def make_rng(seed, *keys):
    return np.random.default_rng(child_seed(seed, *keys))

def safe_corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def score_decile_summary(score, value, n_bins=10):
    order = np.argsort(score)
    s_sorted, v_sorted = score[order], value[order]
    bins = np.array_split(np.arange(len(s_sorted)), n_bins)
    return pd.DataFrame({
        "score_decile_center": [float(s_sorted[idx].mean()) for idx in bins],
        "mean_target": [float(v_sorted[idx].mean()) for idx in bins],
    })

def issue_type_from_x(x):
    if x < -0.40: return "routine question"
    if x < 0.20:  return "billing dispute"
    if x < 0.65:  return "refund / exception case"
    return "fraud / identity case"

def chi2_df2_cdf(x):
    # Closed-form CDF for Chi-square(df=2)
    return 1.0 - math.exp(-x / 2.0)


# ===================================================================
# DATA GENERATING PROCESS
# ===================================================================

def _latent_features(n, rng):
    # Richer latent structure: severity, hard-case indicator, heteroskedasticity.
    x = rng.uniform(-1.0, 1.0, size=n)
    u = rng.normal(0.0, 1.0, size=n)
    severity = 0.5 * (x + 1.0) + 0.2 * np.tanh(u)
    hard_case = (severity > 0.70).astype(int)
    hetero = 0.85 + 0.25 * (severity - severity.mean())
    return {"x": x, "u": u, "severity": severity, "hard_case": hard_case, "hetero": hetero}

def _judge_transform(y, x, scenario, noise_sd, rng):
    # Monotone but nonlinear judge distortion.
    shifted = y + scenario.a_judge_shift_strength * np.tanh((y - 7.4) / 1.5)
    nonlinear = scenario.x_nonlin_strength * np.sin(np.pi * x)
    noise = rng.normal(0.0, noise_sd, size=len(y))
    return shifted + nonlinear + noise

def sample_policy(policy, n, rng, scenario, variant='base', with_meta=False):
    # Sample synthetic data for a policy under a named scenario and variant.
    # Returns DataFrame if with_meta=True, else tuple (x, s, y).
    feats = _latent_features(n, rng)
    x, severity, hard_case, hetero = feats["x"], feats["severity"], feats["hard_case"], feats["hetero"]

    if policy == "A":
        base = (scenario.a_base_intercept + scenario.a_base_slope * x
                + scenario.x_nonlin_strength * (x**2 - 0.35)
                + scenario.hard_case_bonus * hard_case)

        # Catastrophe probability -- designed so mass is well above alpha
        logits = (np.log(scenario.a_cat_base / max(1e-8, 1 - scenario.a_cat_base))
                  + scenario.a_cat_extra
                  + scenario.a_cat_slope * (x - scenario.a_cat_shift)
                  + 0.8 * hard_case)
        base_cat = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.001, 0.90)

        if variant in {"base", "stable"}:
            p_cat = base_cat
        elif variant == "mild_shift":
            p_cat = np.clip(base_cat + 0.04, 0.001, 0.95)
        elif variant == "hard_shift":
            p_cat = np.clip(base_cat + 0.06 + 0.05 * hard_case, 0.001, 0.98)
        elif variant == "fooled_judge":
            p_cat = np.clip(base_cat + 0.03, 0.001, 0.95)
        elif variant == "weird_monotone":
            p_cat = np.clip(base_cat + 0.02 * np.sin(np.pi * x) + 0.03 * hard_case, 0.001, 0.95)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        catastrophic = rng.binomial(1, p_cat, size=n)
        y = np.where(
            catastrophic == 1,
            scenario.a_cat_y_intercept + scenario.a_cat_y_slope * x
            + 0.18 * np.sin(2 * np.pi * severity) + rng.normal(0.0, scenario.a_cat_noise, size=n),
            base + rng.normal(0.0, 0.42 * hetero, size=n),
        )
        s = _judge_transform(y, x, scenario, scenario.a_judge_noise, rng)

        # Variant-specific judge distortions
        if variant == "fooled_judge":
            s = s + 2.4 * catastrophic + 0.8 * hard_case
        elif variant == "weird_monotone":
            s = s + 0.65 * np.tanh((x - 0.3) * 3.0) - 0.25 * (y < 2.0)
        elif variant == "hard_shift":
            s = s + rng.normal(0.0, 0.35, size=n)

    elif policy == "B":
        y = (scenario.b_base_intercept + scenario.b_base_slope * x
             + 0.10 * np.sin(1.8 * np.pi * x)
             + rng.normal(0.0, scenario.b_noise * hetero, size=n))
        s = y + 0.15 * np.tanh((y - 8.0) / 1.2) + rng.normal(0.0, scenario.b_judge_noise, size=n)
        catastrophic = np.zeros(n, dtype=int)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    if not with_meta:
        return x, s, y

    issue_type = np.array([issue_type_from_x(float(xx)) for xx in x])
    scenario_text = np.array([
        f"{it}; confident but wrong explanation" if (cat == 1 and variant == "fooled_judge")
        else f"{it}; brittle failure on a hard case" if cat == 1
        else f"{it}; handled normally"
        for it, cat in zip(issue_type, catastrophic)
    ])
    return pd.DataFrame({
        "policy": policy, "variant": variant, "X": x, "severity": severity,
        "hard_case": hard_case, "S": s, "Y": y, "catastrophic": catastrophic.astype(int),
        "issue_type": issue_type, "scenario": scenario_text,
    })


# ===================================================================
# CALIBRATORS AND ESTIMATORS
# ===================================================================

def fit_isotonic_mean(s_train, y_train, s_pred):
    # Fit increasing isotonic regression: m(s) ~ E[Y|S=s]
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train[order], y_train[order])
    return iso.predict(s_pred)

def fit_isotonic_tail_loss(s_train, z_train, s_pred):
    # Fit decreasing isotonic regression: m_t(s) ~ E[(t-Y)_+|S=s]
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(s_train[order], z_train[order])
    return iso.predict(s_pred)

def estimate_direct_mean_isotonic(s_train, y_train, s_eval):
    return float(fit_isotonic_mean(s_train, y_train, s_eval).mean())

def make_t_grid(y_train, alpha, grid_size):
    # Threshold grid anchored to training quantiles.
    t_lo = float(np.quantile(y_train, max(0.001, alpha / 5.0)) - 0.60)
    t_hi = float(np.quantile(y_train, min(0.60, alpha + 0.45)) + 0.35)
    return np.linspace(t_lo, t_hi, grid_size)

def estimate_direct_cvar_isotonic(s_train, y_train, s_eval, alpha, grid_size, return_curve=False):
    # Grid-search CVaR estimator via stop-loss calibration.
    t_grid = make_t_grid(y_train, alpha, grid_size)
    objective = []
    for t in t_grid:
        z_train = np.maximum(t - y_train, 0.0)
        pred_eval = fit_isotonic_tail_loss(s_train, z_train, s_eval)
        objective.append(float(t - pred_eval.mean() / alpha))
    objective = np.asarray(objective)
    best_idx = int(np.argmax(objective))
    est, t_hat = float(objective[best_idx]), float(t_grid[best_idx])
    if return_curve:
        return est, t_hat, t_grid, objective
    return est, t_hat

def mean_estimator_with_diagnostics(s_train, y_train, s_val, y_val, s_eval):
    # Mean estimator with validation-set diagnostics.
    pred_val = fit_isotonic_mean(s_train, y_train, s_val)
    pred_eval = fit_isotonic_mean(s_train, y_train, s_eval)
    est = float(pred_eval.mean())
    xcal = np.column_stack([np.ones_like(pred_val), pred_val])
    beta = np.linalg.lstsq(xcal, y_val, rcond=None)[0]
    return est, {
        "val_mae": float(np.mean(np.abs(pred_val - y_val))),
        "val_rmse": float(np.sqrt(np.mean((pred_val - y_val) ** 2))),
        "val_bias": float(np.mean(pred_val - y_val)),
        "val_corr": safe_corr(pred_val, y_val),
        "val_calibration_intercept": float(beta[0]),
        "val_calibration_slope": float(beta[1]),
    }

def cvar_estimator_with_diagnostics(s_train, y_train, s_val, y_val, s_eval, alpha, grid_size):
    # CVaR estimator with grid-level diagnostics.
    est, t_hat, t_grid, objective = estimate_direct_cvar_isotonic(
        s_train, y_train, s_eval, alpha, grid_size, return_curve=True)
    val_mae, val_rmse, val_bias, val_corr = [], [], [], []
    for t in t_grid:
        z_train = np.maximum(t - y_train, 0.0)
        z_val = np.maximum(t - y_val, 0.0)
        pred_val = fit_isotonic_tail_loss(s_train, z_train, s_val)
        val_mae.append(float(np.mean(np.abs(pred_val - z_val))))
        val_rmse.append(float(np.sqrt(np.mean((pred_val - z_val) ** 2))))
        val_bias.append(float(np.mean(pred_val - z_val)))
        val_corr.append(safe_corr(pred_val, z_val))
    best_idx = int(np.argmax(objective))
    sorted_obj = np.sort(objective)
    diag = {
        "t_hat": float(t_hat),
        "objective_margin": float(sorted_obj[-1] - sorted_obj[-2]) if len(sorted_obj) >= 2 else float("nan"),
        "val_mae_grid_avg": float(np.mean(val_mae)),
        "val_rmse_grid_avg": float(np.mean(val_rmse)),
        "val_bias_grid_avg": float(np.mean(val_bias)),
        "val_corr_grid_avg": float(np.nanmean(val_corr)),
        "val_mae_at_t_hat": float(val_mae[best_idx]),
        "val_rmse_at_t_hat": float(val_rmse[best_idx]),
        "val_bias_at_t_hat": float(val_bias[best_idx]),
        "val_corr_at_t_hat": float(val_corr[best_idx]),
    }
    return est, diag, t_grid, objective


# ===================================================================
# TRUTH, DIAGNOSTICS, AND AUDIT HELPERS
# ===================================================================

def compute_truth(policy_df, alpha):
    y = policy_df["Y"].to_numpy()
    q_alpha = float(np.quantile(y, alpha))
    tail = y[y <= q_alpha]
    cvar_alpha = float(tail.mean()) if len(tail) > 0 else float("nan")
    return {
        "true_mean": float(y.mean()),
        "true_q_alpha": q_alpha,
        "true_cvar_alpha": cvar_alpha,
        "catastrophe_rate_y_lt_2": float(np.mean(y < 2.0)),
    }

def compute_pre_diagnostics(policy_df, alpha, n_deciles):
    x, s, y = policy_df["X"].to_numpy(), policy_df["S"].to_numpy(), policy_df["Y"].to_numpy()
    q_alpha = float(np.quantile(y, alpha))
    z_q = np.maximum(q_alpha - y, 0.0)
    xmat = np.column_stack([np.ones_like(x), x, x**2, x**3])
    beta_y = np.linalg.lstsq(xmat, y, rcond=None)[0]
    beta_z = np.linalg.lstsq(xmat, z_q, rcond=None)[0]
    y_hat_x, z_hat_x = xmat @ beta_y, xmat @ beta_z
    dec_y = score_decile_summary(s, y, n_bins=n_deciles)
    dec_y["target_type"] = "Y"
    dec_z = score_decile_summary(s, z_q, n_bins=n_deciles)
    dec_z["target_type"] = "tail_loss_at_true_q"
    dec = pd.concat([dec_y, dec_z], ignore_index=True)
    pre = {
        "corr_S_Y": safe_corr(s, y),
        "corr_S_tail_loss": safe_corr(s, z_q),
        "r2_X_for_Y": float(1 - np.sum((y - y_hat_x)**2) / np.sum((y - y.mean())**2)),
        "r2_X_for_tail_loss": float(1 - np.sum((z_q - z_hat_x)**2) / np.sum((z_q - z_q.mean())**2)),
        "mono_violations_Y": int(np.sum(np.diff(dec_y["mean_target"].to_numpy()) < 0)),
        "mono_violations_tail": int(np.sum(np.diff(dec_z["mean_target"].to_numpy()) > 0)),
    }
    return pre, dec

def split_train_val(s, y, rng, validation_frac):
    idx = np.arange(len(s))
    rng.shuffle(idx)
    split = int((1.0 - validation_frac) * len(idx))
    return s[idx[:split]], y[idx[:split]], s[idx[split:]], y[idx[split:]]

def two_moment_wald_audit(s_train, y_train, audit_df, t0, alpha, wald_alpha):
    # Two-moment Wald test at threshold t0.
    y_a, s_a = audit_df["Y"].to_numpy(), audit_df["S"].to_numpy()
    g1 = (y_a <= t0).astype(float) - alpha
    z_train = np.maximum(t0 - y_train, 0.0)
    mhat = fit_isotonic_tail_loss(s_train, z_train, s_a)
    g2 = np.maximum(t0 - y_a, 0.0) - mhat
    g = np.column_stack([g1, g2])
    gbar = g.mean(axis=0)
    shat = np.cov(g, rowvar=False, ddof=1)
    try:
        sinv = np.linalg.inv(shat)
    except np.linalg.LinAlgError:
        sinv = np.linalg.pinv(shat)
    wald = float(len(audit_df) * gbar.T @ sinv @ gbar)
    p_value = float(1.0 - chi2_df2_cdf(wald))
    se1 = np.std(g1, ddof=1) / math.sqrt(len(g1))
    se2 = np.std(g2, ddof=1) / math.sqrt(len(g2))
    return {
        "t0": float(t0), "n_audit": len(audit_df),
        "mean_g1": float(gbar[0]), "mean_g2": float(gbar[1]),
        "z_g1": float(gbar[0] / se1) if se1 > 1e-12 else float("nan"),
        "z_g2": float(gbar[1] / se2) if se2 > 1e-12 else float("nan"),
        "wald_stat": wald, "p_value": p_value,
        "reject": bool(p_value < wald_alpha),
    }

def threshold_sensitivity_check(s_train, y_train, audit_df, t_grid, t_hat, alpha, wald_alpha, tsens_width=1):
    # Sensitivity check over a narrow band around pilot-selected threshold.
    hat_idx = int(np.argmin(np.abs(t_grid - t_hat)))
    lo, hi = max(0, hat_idx - tsens_width), min(len(t_grid), hat_idx + tsens_width + 1)
    rows = []
    for j in range(lo, hi):
        row = two_moment_wald_audit(s_train, y_train, audit_df, float(t_grid[j]), alpha, wald_alpha)
        row["t_grid_index"] = j
        row["is_pilot_t_hat"] = bool(j == hat_idx)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["tsens_reject_any"] = bool(out["reject"].any())
    return out

def compute_tail_residuals_fixed_t(s_train, y_train, probe_df, t_fixed):
    # Compute residuals: oracle_tail_loss - calibrated_tail_loss.
    z_train = np.maximum(t_fixed - y_train, 0.0)
    pred = fit_isotonic_tail_loss(s_train, z_train, probe_df["S"].to_numpy())
    oracle = np.maximum(t_fixed - probe_df["Y"].to_numpy(), 0.0)
    out = probe_df.copy()
    out["t_fixed"] = float(t_fixed)
    out["oracle_tail_loss"] = oracle
    out["calibrated_tail_loss"] = pred
    out["residual"] = oracle - pred
    out["judge_overestimated_safety"] = (out["residual"] > 0).astype(int)
    return out.sort_values("residual", ascending=False).reset_index(drop=True)

def residuals_by_score_decile(df, n_bins=10):
    order = np.argsort(df["S"].to_numpy())
    tmp = df.iloc[order].reset_index(drop=True)
    bins = np.array_split(np.arange(len(tmp)), n_bins)
    return pd.DataFrame({
        "score_decile_center": [float(tmp.loc[idx, "S"].mean()) for idx in bins],
        "mean_residual": [float(tmp.loc[idx, "residual"].mean()) for idx in bins],
        "mean_oracle_tail_loss": [float(tmp.loc[idx, "oracle_tail_loss"].mean()) for idx in bins],
        "mean_calibrated_tail_loss": [float(tmp.loc[idx, "calibrated_tail_loss"].mean()) for idx in bins],
    })

print("Core functions defined.")
