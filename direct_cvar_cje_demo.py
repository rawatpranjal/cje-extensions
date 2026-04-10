#!/usr/bin/env python3
"""
Direct Mean CJE and Direct CVaR-CJE demo script.

This script is a cleaned, reproducible rewrite of the original notebook. It keeps
its core methodology intentionally narrow:

1. Direct-only evaluation.
2. Plain isotonic calibration.
3. Mean target and lower-tail CVaR target.
4. Pilot / audit split for the tail transport audit.

What is added relative to the notebook is structural rather than conceptual:

- a compact theory block at the top,
- one configuration section with all major knobs,
- richer synthetic regimes and harder DGPs,
- larger Monte Carlo sweeps and sensitivity runs,
- grid plots and wide summary tables,
- tqdm progress bars,
- deterministic seeding,
- commented code throughout.

No HTML output is produced. The script saves CSV tables and PNG figures.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm


# ======================================================================================
# SECTION 0. THEORY
# ======================================================================================
#
# Estimands
# ---------
# For a target policy pi, let Y^pi be the trusted oracle score and S^pi the cheap judge.
# The mean target is
#     V(pi) = E[Y^pi].
#
# The lower-tail CVaR target at level alpha is
#     CVaR_alpha(Y^pi)
#         = sup_t { t - (1/alpha) E[(t - Y^pi)_+] }.
#
# Direct mean CJE
# ---------------
# Fit an isotonic calibrator m(s) ≈ E[Y | S = s] on a calibration slice containing
# both S and Y. Then estimate the policy value on a fresh target slice using judge
# scores alone:
#     V_hat_dir(pi) = mean_i m_hat(S_i^pi).
#
# Direct CVaR-CJE
# ---------------
# For each threshold t, define the stop-loss transform Z_t = (t - Y)_+.
# Fit an isotonic calibrator m_t(s) ≈ E[Z_t | S = s]. Then estimate
#     psi_hat_pi(t) = t - (1/alpha) mean_i m_hat_t(S_i^pi),
# and choose the maximizing threshold t_hat over a fixed grid.
#
# Pilot / audit transport logic
# -----------------------------
# To keep the audit aligned with the estimator, we:
# 1) fit the calibrator family on a source slice,
# 2) use a target pilot slice to choose the target-specific t_hat,
# 3) use a disjoint target audit slice for the formal test.
#
# The audit moments at the pilot-chosen t_hat are:
#     g1(Y; t_hat) = 1{Y <= t_hat} - alpha,
#     g2(S, Y; t_hat) = (t_hat - Y)_+ - m_hat_{t_hat}(S).
#
# The first checks whether t_hat behaves like the lower-tail cutoff on the target.
# The second checks whether the stop-loss calibration transports at that threshold.
#
# Caveat
# ------
# This script is still a demo. The audit uses a straightforward Wald statistic and
# does not propagate first-stage calibrator uncertainty. That is useful for stress
# testing and communication, but it is not presented as a fully valid inferential
# procedure.
#
# ======================================================================================


# ======================================================================================
# SECTION 1. CONFIGURATION
# ======================================================================================


@dataclass(frozen=True)
class GlobalConfig:
    """Top-level reproducibility and output settings."""

    seed: int = 123
    out_dir: str = "demo_outputs"
    save_tables: bool = True
    save_figures: bool = True
    show_figures: bool = False
    csv_float_format: str = "%.6f"


@dataclass(frozen=True)
class EstimationConfig:
    """Core estimation knobs.

    All of these are safe to tune without changing the methodology.
    """

    alpha: float = 0.10
    grid_size: int = 61
    truth_n: int = 400_000
    n_calib: int = 40_000
    n_eval: int = 60_000
    replications: int = 100
    validation_frac: float = 0.20


@dataclass(frozen=True)
class AuditConfig:
    """Knobs for the pilot / audit transport section."""

    n_probe_total: int = 2_400
    pilot_frac: float = 0.50
    wald_alpha: float = 0.05
    audit_replications: int = 100
    band_width: int = 1  # used only in a sensitivity audit around the pilot-selected t_hat


@dataclass(frozen=True)
class SweepConfig:
    """Optional sensitivity sweeps.

    These keep the same methodology but vary one or two knobs at a time.
    """

    alpha_grid: Tuple[float, ...] = (0.05, 0.10, 0.20)
    oracle_budget_grid: Tuple[int, ...] = (600, 1_200, 2_400, 4_800)
    replications_per_sensitivity: int = 50


@dataclass(frozen=True)
class PlotConfig:
    """Pure presentation knobs."""

    sample_plot_n_per_policy: int = 80_000
    bins: int = 70
    score_deciles: int = 10
    figsize_wide: Tuple[int, int] = (14, 10)
    figsize_standard: Tuple[int, int] = (8, 5)
    dpi: int = 150


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario-specific DGP knobs.

    The goal is not to make the DGP realistic in every detail. The goal is to make
    the estimand and the transport issues legible.
    """

    name: str
    description: str
    # Policy A parameters.
    a_base_intercept: float
    a_base_slope: float
    a_cat_base: float
    a_cat_extra: float
    a_cat_shift: float
    a_cat_slope: float
    a_cat_y_intercept: float
    a_cat_y_slope: float
    a_cat_noise: float
    # Policy B parameters.
    b_base_intercept: float
    b_base_slope: float
    b_noise: float
    # Judge parameters.
    a_judge_noise: float
    a_judge_shift_strength: float
    b_judge_noise: float
    # Misc.
    x_nonlin_strength: float = 0.0
    hard_case_bonus: float = 0.0
    hard_case_threshold: float = 0.60


@dataclass(frozen=True)
class FullConfig:
    global_cfg: GlobalConfig = field(default_factory=GlobalConfig)
    est_cfg: EstimationConfig = field(default_factory=EstimationConfig)
    audit_cfg: AuditConfig = field(default_factory=AuditConfig)
    sweep_cfg: SweepConfig = field(default_factory=SweepConfig)
    plot_cfg: PlotConfig = field(default_factory=PlotConfig)


DEFAULT_SCENARIOS: Dict[str, ScenarioConfig] = {
    # Clear and convincing: A wins on mean, B wins on lower-tail CVaR.
    "clear_tail_tradeoff": ScenarioConfig(
        name="clear_tail_tradeoff",
        description="Policy A is slightly better on average but much worse in the lower tail. This is the main demo regime.",
        a_base_intercept=9.30,
        a_base_slope=0.18,
        a_cat_base=0.08,
        a_cat_extra=0.14,
        a_cat_shift=0.02,
        a_cat_slope=4.8,
        a_cat_y_intercept=0.55,
        a_cat_y_slope=0.30,
        a_cat_noise=0.40,
        b_base_intercept=8.45,
        b_base_slope=0.34,
        b_noise=0.42,
        a_judge_noise=0.80,
        a_judge_shift_strength=0.75,
        b_judge_noise=0.58,
        x_nonlin_strength=0.15,
        hard_case_bonus=0.0,
    ),
    # No tradeoff: A should win on both mean and CVaR. Useful placebo.
    "no_tradeoff": ScenarioConfig(
        name="no_tradeoff",
        description="Policy A dominates both on mean and on lower-tail CVaR. This is a placebo regime.",
        a_base_intercept=9.10,
        a_base_slope=0.16,
        a_cat_base=0.03,
        a_cat_extra=0.05,
        a_cat_shift=0.05,
        a_cat_slope=4.0,
        a_cat_y_intercept=0.90,
        a_cat_y_slope=0.20,
        a_cat_noise=0.35,
        b_base_intercept=8.35,
        b_base_slope=0.30,
        b_noise=0.45,
        a_judge_noise=0.72,
        a_judge_shift_strength=0.65,
        b_judge_noise=0.60,
        x_nonlin_strength=0.10,
        hard_case_bonus=0.0,
    ),
    # Smaller gaps and harder calibration.
    "hard_small_gap": ScenarioConfig(
        name="hard_small_gap",
        description="Mean and CVaR gaps are smaller, calibration is noisier, and selection is materially harder.",
        a_base_intercept=8.95,
        a_base_slope=0.16,
        a_cat_base=0.09,
        a_cat_extra=0.11,
        a_cat_shift=0.08,
        a_cat_slope=5.2,
        a_cat_y_intercept=0.55,
        a_cat_y_slope=0.35,
        a_cat_noise=0.50,
        b_base_intercept=8.55,
        b_base_slope=0.25,
        b_noise=0.48,
        a_judge_noise=0.95,
        a_judge_shift_strength=0.85,
        b_judge_noise=0.70,
        x_nonlin_strength=0.20,
        hard_case_bonus=0.15,
    ),
    # Weird but still monotone enough for isotonic to be interpretable.
    "weird_nonlinear": ScenarioConfig(
        name="weird_nonlinear",
        description="The judge is nonlinear and the issue mix is more heterogeneous. The method still remains direct-only and isotonic.",
        a_base_intercept=9.10,
        a_base_slope=0.12,
        a_cat_base=0.07,
        a_cat_extra=0.16,
        a_cat_shift=-0.02,
        a_cat_slope=6.0,
        a_cat_y_intercept=0.45,
        a_cat_y_slope=0.40,
        a_cat_noise=0.45,
        b_base_intercept=8.30,
        b_base_slope=0.28,
        b_noise=0.45,
        a_judge_noise=0.88,
        a_judge_shift_strength=0.95,
        b_judge_noise=0.66,
        x_nonlin_strength=0.35,
        hard_case_bonus=0.25,
    ),
}

# Target audit variants are deliberately defined relative to a source-trained policy A.
AUDIT_VARIANTS: Tuple[str, ...] = (
    "stable",
    "mild_shift",
    "hard_shift",
    "fooled_judge",
    "weird_monotone",
)


# ======================================================================================
# SECTION 2. UTILITIES
# ======================================================================================


def set_pandas_display() -> None:
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def child_seed(seed: int, *keys: object) -> int:
    """Deterministic child seed.

    This avoids accidental dependence on loop order. The same section, scenario, and
    replication always get the same seed.
    """

    h = int(seed)
    for key in keys:
        h = (h * 1_000_003 + hash(str(key))) % (2**32 - 1)
    return int(h)


def make_rng(seed: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(child_seed(seed, *keys))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    sa = np.std(a)
    sb = np.std(b)
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_decile_summary(score: np.ndarray, value: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(score)
    score_sorted = score[order]
    value_sorted = value[order]
    bins = np.array_split(np.arange(len(score_sorted)), n_bins)
    return pd.DataFrame(
        {
            "score_decile_center": [float(score_sorted[idx].mean()) for idx in bins],
            "mean_target": [float(value_sorted[idx].mean()) for idx in bins],
        }
    )


def issue_type_from_x(x: float) -> str:
    if x < -0.40:
        return "routine question"
    if x < 0.20:
        return "billing dispute"
    if x < 0.65:
        return "refund / exception case"
    return "fraud / identity case"


def prettify_table(df: pd.DataFrame, title: str, n: Optional[int] = None) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if n is None:
        print(df.to_string(index=False))
    else:
        print(df.head(n).to_string(index=False))


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, float_format: str = "%.6f") -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False, float_format=float_format)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 150, close: bool = True) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


def chi2_df2_cdf(x: float) -> float:
    # Closed form for Chi-square(df=2).
    return 1.0 - math.exp(-x / 2.0)


# ======================================================================================
# SECTION 3. DGP
# ======================================================================================


def _latent_features(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate a richer latent structure than the original notebook.

    The main latent driver is x in [-1, 1], but we also create an issue severity score,
    a hard-case indicator, and a mild heteroskedasticity term. This keeps the synthetic
    problem legible while giving the plots more texture and the tail more structure.
    """

    x = rng.uniform(-1.0, 1.0, size=n)
    u = rng.normal(0.0, 1.0, size=n)
    severity = 0.5 * (x + 1.0) + 0.2 * np.tanh(u)
    hard_case = (severity > 0.70).astype(int)
    hetero = 0.85 + 0.25 * (severity - severity.mean())
    return {
        "x": x,
        "u": u,
        "severity": severity,
        "hard_case": hard_case,
        "hetero": hetero,
    }


def _judge_transform(y: np.ndarray, x: np.ndarray, scenario: ScenarioConfig, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    """Monotone but nonlinear judge distortion.

    This is still compatible with the spirit of isotonic calibration. The judge is not
    perfect and the mapping is not linear, but it remains broadly monotone except in the
    explicit failure variants.
    """

    shifted = y + scenario.a_judge_shift_strength * np.tanh((y - 7.4) / 1.5)
    nonlinear = scenario.x_nonlin_strength * np.sin(np.pi * x)
    noise = rng.normal(0.0, noise_sd, size=len(y))
    return shifted + nonlinear + noise


def sample_policy(
    policy: str,
    n: int,
    rng: np.random.Generator,
    scenario: ScenarioConfig,
    variant: str = "base",
    with_meta: bool = False,
) -> pd.DataFrame | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample synthetic data for a policy under a named scenario and variant.

    Parameters
    ----------
    policy : str
        Either "A" or "B".
    variant : str
        For policy comparison we usually use "base". For source-to-target transport
        auditing we use variants like "stable", "mild_shift", "hard_shift",
        "fooled_judge", and "weird_monotone".
    """

    feats = _latent_features(n, rng)
    x = feats["x"]
    severity = feats["severity"]
    hard_case = feats["hard_case"]
    hetero = feats["hetero"]

    if policy == "A":
        base = (
            scenario.a_base_intercept
            + scenario.a_base_slope * x
            + scenario.x_nonlin_strength * (x**2 - 0.35)
            + scenario.hard_case_bonus * hard_case
        )

        # Catastrophe probability for policy A. The key design knob is that the clear
        # tradeoff scenario keeps the catastrophe mass well above alpha, to avoid the
        # knife-edge instability from the earlier notebook.
        logits = (
            np.log(scenario.a_cat_base / max(1e-8, 1 - scenario.a_cat_base))
            + scenario.a_cat_extra
            + scenario.a_cat_slope * (x - scenario.a_cat_shift)
            + 0.8 * hard_case
        )
        base_cat = 1.0 / (1.0 + np.exp(-logits))
        base_cat = np.clip(base_cat, 0.001, 0.90)

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
            raise ValueError(f"Unknown A variant: {variant}")

        catastrophic = rng.binomial(1, p_cat, size=n)
        y = np.where(
            catastrophic == 1,
            scenario.a_cat_y_intercept
            + scenario.a_cat_y_slope * x
            + 0.18 * np.sin(2 * np.pi * severity)
            + rng.normal(0.0, scenario.a_cat_noise, size=n),
            base + rng.normal(0.0, 0.42 * hetero, size=n),
        )

        s = _judge_transform(y=y, x=x, scenario=scenario, noise_sd=scenario.a_judge_noise, rng=rng)

        if variant == "fooled_judge":
            # Explicit transport failure: catastrophic cases are systematically over-scored.
            s = s + 2.4 * catastrophic + 0.8 * hard_case
        elif variant == "weird_monotone":
            # Still mostly monotone, but compressed and warped in the tail.
            s = s + 0.65 * np.tanh((x - 0.3) * 3.0) - 0.25 * (y < 2.0)
        elif variant == "hard_shift":
            s = s + rng.normal(0.0, 0.35, size=n)

    elif policy == "B":
        # B is the safer policy. It has no catastrophic mass in this stylized setup.
        y = (
            scenario.b_base_intercept
            + scenario.b_base_slope * x
            + 0.10 * np.sin(1.8 * np.pi * x)
            + rng.normal(0.0, scenario.b_noise * hetero, size=n)
        )
        s = y + 0.15 * np.tanh((y - 8.0) / 1.2) + rng.normal(0.0, scenario.b_judge_noise, size=n)
        catastrophic = np.zeros(n, dtype=int)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    issue_type = np.array([issue_type_from_x(float(xx)) for xx in x])
    scenario_text = np.array(
        [
            f"{it}; confident but wrong explanation" if (cat == 1 and variant == 'fooled_judge')
            else f"{it}; brittle failure on a hard case" if cat == 1
            else f"{it}; handled normally"
            for it, cat in zip(issue_type, catastrophic)
        ]
    )

    if with_meta:
        return pd.DataFrame(
            {
                "policy": policy,
                "variant": variant,
                "X": x,
                "severity": severity,
                "hard_case": hard_case,
                "S": s,
                "Y": y,
                "catastrophic": catastrophic.astype(int),
                "issue_type": issue_type,
                "scenario": scenario_text,
            }
        )

    return x, s, y


# ======================================================================================
# SECTION 4. CALIBRATORS AND ESTIMATORS
# ======================================================================================


def fit_isotonic_mean(s_train: np.ndarray, y_train: np.ndarray, s_pred: np.ndarray) -> np.ndarray:
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train[order], y_train[order])
    return iso.predict(s_pred)


def fit_isotonic_tail_loss(s_train: np.ndarray, z_train: np.ndarray, s_pred: np.ndarray) -> np.ndarray:
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(s_train[order], z_train[order])
    return iso.predict(s_pred)


def estimate_direct_mean_isotonic(s_train: np.ndarray, y_train: np.ndarray, s_eval: np.ndarray) -> float:
    pred = fit_isotonic_mean(s_train, y_train, s_eval)
    return float(pred.mean())


def make_t_grid(y_train: np.ndarray, alpha: float, grid_size: int) -> np.ndarray:
    """Build a reasonably wide threshold grid.

    Using a grid anchored to the training distribution keeps the demo deterministic and
    transparent. This also keeps the optimizer simple and in line with the notebook.
    """

    t_lo = float(np.quantile(y_train, max(0.001, alpha / 5.0)) - 0.60)
    t_hi = float(np.quantile(y_train, min(0.60, alpha + 0.45)) + 0.35)
    return np.linspace(t_lo, t_hi, grid_size)


def estimate_direct_cvar_isotonic(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    grid_size: int,
    return_curve: bool = False,
) -> Tuple[float, float] | Tuple[float, float, np.ndarray, np.ndarray]:
    t_grid = make_t_grid(y_train=y_train, alpha=alpha, grid_size=grid_size)
    objective = []
    for t in t_grid:
        z_train = np.maximum(t - y_train, 0.0)
        pred_eval = fit_isotonic_tail_loss(s_train, z_train, s_eval)
        objective.append(float(t - pred_eval.mean() / alpha))

    objective = np.asarray(objective)
    best_idx = int(np.argmax(objective))
    est = float(objective[best_idx])
    t_hat = float(t_grid[best_idx])

    if return_curve:
        return est, t_hat, t_grid, objective
    return est, t_hat


def mean_estimator_with_diagnostics(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    s_eval: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
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


def cvar_estimator_with_diagnostics(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    grid_size: int,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    est, t_hat, t_grid, objective = estimate_direct_cvar_isotonic(
        s_train=s_train,
        y_train=y_train,
        s_eval=s_eval,
        alpha=alpha,
        grid_size=grid_size,
        return_curve=True,
    )

    val_mae_grid = []
    val_rmse_grid = []
    val_bias_grid = []
    val_corr_grid = []

    for t in t_grid:
        z_train = np.maximum(t - y_train, 0.0)
        z_val = np.maximum(t - y_val, 0.0)
        pred_val = fit_isotonic_tail_loss(s_train, z_train, s_val)
        val_mae_grid.append(float(np.mean(np.abs(pred_val - z_val))))
        val_rmse_grid.append(float(np.sqrt(np.mean((pred_val - z_val) ** 2))))
        val_bias_grid.append(float(np.mean(pred_val - z_val)))
        val_corr_grid.append(safe_corr(pred_val, z_val))

    best_idx = int(np.argmax(objective))
    sorted_obj = np.sort(objective)
    diag = {
        "t_hat": float(t_hat),
        "objective_margin_best_minus_second": float(sorted_obj[-1] - sorted_obj[-2]) if len(sorted_obj) >= 2 else float("nan"),
        "val_mae_grid_avg": float(np.mean(val_mae_grid)),
        "val_rmse_grid_avg": float(np.mean(val_rmse_grid)),
        "val_bias_grid_avg": float(np.mean(val_bias_grid)),
        "val_corr_grid_avg": float(np.nanmean(val_corr_grid)),
        "val_mae_at_t_hat": float(val_mae_grid[best_idx]),
        "val_rmse_at_t_hat": float(val_rmse_grid[best_idx]),
        "val_bias_at_t_hat": float(val_bias_grid[best_idx]),
        "val_corr_at_t_hat": float(val_corr_grid[best_idx]),
    }
    return est, diag, t_grid, objective


# ======================================================================================
# SECTION 5. TRUTH, DIAGNOSTICS, AND TRANSPORT AUDIT HELPERS
# ======================================================================================


def compute_truth(policy_df: pd.DataFrame, alpha: float) -> Dict[str, float]:
    y = policy_df["Y"].to_numpy()
    q_alpha = float(np.quantile(y, alpha))
    cvar_alpha = float(y[y <= q_alpha].mean())
    return {
        "true_mean": float(y.mean()),
        "true_q_alpha": q_alpha,
        "true_cvar_alpha": cvar_alpha,
        "catastrophe_rate_y_lt_2": float(np.mean(y < 2.0)),
    }


def compute_pre_diagnostics(policy_df: pd.DataFrame, alpha: float, n_deciles: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    x = policy_df["X"].to_numpy()
    s = policy_df["S"].to_numpy()
    y = policy_df["Y"].to_numpy()

    q_alpha = float(np.quantile(y, alpha))
    z_q = np.maximum(q_alpha - y, 0.0)

    xmat = np.column_stack([np.ones_like(x), x, x**2, x**3])
    beta_y = np.linalg.lstsq(xmat, y, rcond=None)[0]
    beta_z = np.linalg.lstsq(xmat, z_q, rcond=None)[0]
    y_hat_x = xmat @ beta_y
    z_hat_x = xmat @ beta_z

    dec_y = score_decile_summary(s, y, n_bins=n_deciles)
    dec_y["target_type"] = "Y"
    dec_z = score_decile_summary(s, z_q, n_bins=n_deciles)
    dec_z["target_type"] = "tail_loss_at_true_q"
    dec = pd.concat([dec_y, dec_z], ignore_index=True)

    pre = {
        "corr_S_Y": safe_corr(s, y),
        "corr_S_tail_loss_at_true_q": safe_corr(s, z_q),
        "corr_X_tail_loss_at_true_q": safe_corr(x, z_q),
        "r2_X_only_for_Y": float(1 - np.sum((y - y_hat_x) ** 2) / np.sum((y - y.mean()) ** 2)),
        "r2_X_only_for_tail_loss_at_true_q": float(1 - np.sum((z_q - z_hat_x) ** 2) / np.sum((z_q - z_q.mean()) ** 2)),
        "judge_decile_monotonicity_violations_for_Y": int(np.sum(np.diff(dec_y["mean_target"].to_numpy()) < 0)),
        "judge_decile_monotonicity_violations_for_tail_loss": int(np.sum(np.diff(dec_z["mean_target"].to_numpy()) > 0)),
    }
    return pre, dec


def split_train_val(s: np.ndarray, y: np.ndarray, rng: np.random.Generator, validation_frac: float) -> Tuple[np.ndarray, ...]:
    idx = np.arange(len(s))
    rng.shuffle(idx)
    split = int((1.0 - validation_frac) * len(idx))
    train_idx = idx[:split]
    val_idx = idx[split:]
    return s[train_idx], y[train_idx], s[val_idx], y[val_idx]


def two_moment_wald_audit(
    s_train: np.ndarray,
    y_train: np.ndarray,
    audit_df: pd.DataFrame,
    t0: float,
    alpha: float,
    wald_alpha: float,
) -> Dict[str, float]:
    """Core point audit from the notebook.

    This is kept intentionally simple because the user asked not to deviate from the
    elegant core methodology. A band audit is added separately as a sensitivity layer.
    """

    y_audit = audit_df["Y"].to_numpy()
    s_audit = audit_df["S"].to_numpy()

    g1 = (y_audit <= t0).astype(float) - alpha
    z_train = np.maximum(t0 - y_train, 0.0)
    mhat = fit_isotonic_tail_loss(s_train, z_train, s_audit)
    g2 = np.maximum(t0 - y_audit, 0.0) - mhat

    g = np.column_stack([g1, g2])
    gbar = g.mean(axis=0)
    shat = np.cov(g, rowvar=False, ddof=1)

    try:
        sinv = np.linalg.inv(shat)
    except np.linalg.LinAlgError:
        sinv = np.linalg.pinv(shat)

    wald = float(len(audit_df) * gbar.T @ sinv @ gbar)
    p_value = float(1.0 - chi2_df2_cdf(wald))
    reject = bool(p_value < wald_alpha)

    se1 = np.std(g1, ddof=1) / math.sqrt(len(g1))
    se2 = np.std(g2, ddof=1) / math.sqrt(len(g2))
    z1 = float(gbar[0] / se1) if se1 > 1e-12 else float("nan")
    z2 = float(gbar[1] / se2) if se2 > 1e-12 else float("nan")

    return {
        "t0": float(t0),
        "n_audit": int(len(audit_df)),
        "mean_g1_quantile_moment": float(gbar[0]),
        "mean_g2_stoploss_moment": float(gbar[1]),
        "z_g1": z1,
        "z_g2": z2,
        "wald_stat": wald,
        "p_value": p_value,
        "reject": reject,
    }


def band_audit_around_t_hat(
    s_train: np.ndarray,
    y_train: np.ndarray,
    audit_df: pd.DataFrame,
    t_grid: np.ndarray,
    t_hat: float,
    alpha: float,
    wald_alpha: float,
    band_width: int,
) -> pd.DataFrame:
    """Sensitivity audit over a narrow band around the pilot-selected threshold.

    This is not the core method. It is just a useful robustness layer for the demo.
    """

    hat_idx = int(np.argmin(np.abs(t_grid - t_hat)))
    lo = max(0, hat_idx - band_width)
    hi = min(len(t_grid), hat_idx + band_width + 1)
    rows = []
    for j in range(lo, hi):
        row = two_moment_wald_audit(
            s_train=s_train,
            y_train=y_train,
            audit_df=audit_df,
            t0=float(t_grid[j]),
            alpha=alpha,
            wald_alpha=wald_alpha,
        )
        row["t_grid_index"] = j
        row["is_pilot_t_hat"] = bool(j == hat_idx)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["band_reject_any"] = bool(out["reject"].any())
    return out


def compute_tail_residuals_fixed_t(
    s_train: np.ndarray,
    y_train: np.ndarray,
    probe_df: pd.DataFrame,
    t_fixed: float,
) -> pd.DataFrame:
    z_train = np.maximum(t_fixed - y_train, 0.0)
    pred_probe = fit_isotonic_tail_loss(s_train, z_train, probe_df["S"].to_numpy())
    oracle_probe = np.maximum(t_fixed - probe_df["Y"].to_numpy(), 0.0)
    resid = oracle_probe - pred_probe

    out = probe_df.copy()
    out["t_fixed"] = float(t_fixed)
    out["oracle_tail_loss"] = oracle_probe
    out["calibrated_tail_loss"] = pred_probe
    out["residual"] = resid
    out["judge_overestimated_safety"] = (out["residual"] > 0).astype(int)
    return out.sort_values("residual", ascending=False).reset_index(drop=True)


def residuals_by_score_decile(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    order = np.argsort(df["S"].to_numpy())
    tmp = df.iloc[order].reset_index(drop=True)
    bins = np.array_split(np.arange(len(tmp)), n_bins)
    return pd.DataFrame(
        {
            "score_decile_center": [float(tmp.loc[idx, "S"].mean()) for idx in bins],
            "mean_residual": [float(tmp.loc[idx, "residual"].mean()) for idx in bins],
            "mean_oracle_tail_loss": [float(tmp.loc[idx, "oracle_tail_loss"].mean()) for idx in bins],
            "mean_calibrated_tail_loss": [float(tmp.loc[idx, "calibrated_tail_loss"].mean()) for idx in bins],
        }
    )


# ======================================================================================
# SECTION 6. POLICY COMPARISON BLOCK
# ======================================================================================


def run_compare_policies_for_scenario(
    scenario: ScenarioConfig,
    cfg: FullConfig,
) -> Dict[str, pd.DataFrame]:
    est_cfg = cfg.est_cfg
    plot_cfg = cfg.plot_cfg
    root_seed = cfg.global_cfg.seed

    truth_rows: List[Dict[str, float]] = []
    pre_rows: List[Dict[str, float]] = []
    decile_frames: List[pd.DataFrame] = []
    sample_frames: List[pd.DataFrame] = []

    # ----- Truth and pre-estimation diagnostics -----
    for policy in ["A", "B"]:
        rng_truth = make_rng(root_seed, scenario.name, "truth", policy)
        truth_df_policy = sample_policy(
            policy=policy,
            n=est_cfg.truth_n,
            rng=rng_truth,
            scenario=scenario,
            variant="base",
            with_meta=True,
        )
        truth = compute_truth(truth_df_policy, alpha=est_cfg.alpha)
        pre, dec = compute_pre_diagnostics(truth_df_policy, alpha=est_cfg.alpha, n_deciles=plot_cfg.score_deciles)

        truth_rows.append({"scenario_name": scenario.name, "policy": policy, **truth})
        pre_rows.append({"scenario_name": scenario.name, "policy": policy, **pre})
        dec["scenario_name"] = scenario.name
        dec["policy"] = policy
        decile_frames.append(dec)

        sample_n = min(plot_cfg.sample_plot_n_per_policy, len(truth_df_policy))
        sample_frames.append(truth_df_policy.sample(sample_n, random_state=0).reset_index(drop=True))

    truth_df = pd.DataFrame(truth_rows)
    pre_diag_df = pd.DataFrame(pre_rows)
    decile_df = pd.concat(decile_frames, ignore_index=True)
    sample_df = pd.concat(sample_frames, ignore_index=True)

    truth_lookup = {
        row["policy"]: {
            "mean": row["true_mean"],
            "cvar": row["true_cvar_alpha"],
        }
        for _, row in truth_df.iterrows()
    }

    # ----- Monte Carlo replications -----
    rep_rows: List[Dict[str, float]] = []
    objective_curve_rows: List[Dict[str, float]] = []
    objective_summary_rows: List[Dict[str, float]] = []

    rep_iter = tqdm(range(est_cfg.replications), desc=f"compare::{scenario.name}", leave=False)
    for rep in rep_iter:
        for policy in ["A", "B"]:
            rng = make_rng(root_seed, scenario.name, "compare", rep, policy)
            _, s_calib, y_calib = sample_policy(policy, est_cfg.n_calib, rng, scenario, variant="base", with_meta=False)
            _, s_eval, _ = sample_policy(policy, est_cfg.n_eval, rng, scenario, variant="base", with_meta=False)
            s_train, y_train, s_val, y_val = split_train_val(s_calib, y_calib, rng, est_cfg.validation_frac)

            # Mean estimator.
            est_mean, mean_diag = mean_estimator_with_diagnostics(
                s_train=s_train,
                y_train=y_train,
                s_val=s_val,
                y_val=y_val,
                s_eval=s_eval,
            )
            rep_rows.append(
                {
                    "scenario_name": scenario.name,
                    "rep": rep,
                    "policy": policy,
                    "target": "mean",
                    "method": "mean_iso",
                    "estimate": est_mean,
                    "true_value": truth_lookup[policy]["mean"],
                    **mean_diag,
                }
            )

            # CVaR estimator.
            est_cvar, cvar_diag, t_grid, objective = cvar_estimator_with_diagnostics(
                s_train=s_train,
                y_train=y_train,
                s_val=s_val,
                y_val=y_val,
                s_eval=s_eval,
                alpha=est_cfg.alpha,
                grid_size=est_cfg.grid_size,
            )
            rep_rows.append(
                {
                    "scenario_name": scenario.name,
                    "rep": rep,
                    "policy": policy,
                    "target": f"cvar_alpha_{est_cfg.alpha:.2f}",
                    "method": "cvar_iso",
                    "estimate": est_cvar,
                    "true_value": truth_lookup[policy]["cvar"],
                    **cvar_diag,
                }
            )

            # Keep one objective curve per policy from rep 0 for communication plots.
            if rep == 0:
                for t, obj in zip(t_grid, objective):
                    objective_curve_rows.append(
                        {
                            "scenario_name": scenario.name,
                            "policy": policy,
                            "t": float(t),
                            "objective": float(obj),
                        }
                    )
                objective_summary_rows.append(
                    {
                        "scenario_name": scenario.name,
                        "policy": policy,
                        "estimated_cvar": est_cvar,
                        "t_hat": cvar_diag["t_hat"],
                    }
                )

    per_rep_df = pd.DataFrame(rep_rows)
    objective_curve_df = pd.DataFrame(objective_curve_rows)
    objective_summary_df = pd.DataFrame(objective_summary_rows)

    # ----- Summary tables -----
    summary_rows = []
    for (target, policy, method), g in per_rep_df.groupby(["target", "policy", "method"]):
        summary_rows.append(
            {
                "scenario_name": scenario.name,
                "target": target,
                "policy": policy,
                "method": method,
                "estimate_mean": float(g["estimate"].mean()),
                "estimate_std": float(g["estimate"].std(ddof=1)),
                "true_value": float(g["true_value"].iloc[0]),
                "bias": float(g["estimate"].mean() - g["true_value"].iloc[0]),
                "rmse": float(np.sqrt(np.mean((g["estimate"] - g["true_value"]) ** 2))),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "policy", "method"]).reset_index(drop=True)

    selection_rows = []
    true_best_lookup = {"mean": "A", f"cvar_alpha_{est_cfg.alpha:.2f}": "B" if scenario.name != "no_tradeoff" else "A"}
    for target, true_best in true_best_lookup.items():
        sub = per_rep_df[per_rep_df["target"] == target]
        wide = sub.pivot_table(index="rep", columns=["policy", "method"], values="estimate")
        for method in sorted(sub["method"].unique()):
            if true_best == "A":
                correct_rate = float((wide[("A", method)] > wide[("B", method)]).mean())
            else:
                correct_rate = float((wide[("B", method)] > wide[("A", method)]).mean())
            selection_rows.append(
                {
                    "scenario_name": scenario.name,
                    "target": target,
                    "method": method,
                    "true_best_policy": true_best,
                    "correct_policy_selected_rate": correct_rate,
                }
            )
    selection_df = pd.DataFrame(selection_rows).sort_values(["target", "method"]).reset_index(drop=True)

    gap_rows = []
    for target in ["mean", f"cvar_alpha_{est_cfg.alpha:.2f}"]:
        sub = per_rep_df[per_rep_df["target"] == target]
        wide = sub.pivot_table(index="rep", columns=["policy", "method"], values="estimate")
        for method in sorted(sub["method"].unique()):
            est_gap = wide[("A", method)] - wide[("B", method)]
            true_gap = (
                float(truth_df.loc[truth_df["policy"] == "A", "true_mean"].iloc[0])
                - float(truth_df.loc[truth_df["policy"] == "B", "true_mean"].iloc[0])
                if target == "mean"
                else float(truth_df.loc[truth_df["policy"] == "A", "true_cvar_alpha"].iloc[0])
                - float(truth_df.loc[truth_df["policy"] == "B", "true_cvar_alpha"].iloc[0])
            )
            gap_rows.append(
                {
                    "scenario_name": scenario.name,
                    "target": target,
                    "method": method,
                    "true_gap_A_minus_B": true_gap,
                    "estimated_gap_mean_A_minus_B": float(est_gap.mean()),
                    "estimated_gap_std_A_minus_B": float(est_gap.std(ddof=1)),
                    "gap_sign_correct_rate": float((np.sign(est_gap) == np.sign(true_gap)).mean()),
                }
            )
    gap_df = pd.DataFrame(gap_rows).sort_values(["target", "method"]).reset_index(drop=True)

    # Direct truth vs estimate table, which is often the most persuasive view.
    tv_rows = []
    for policy in ["A", "B"]:
        mean_sub = summary_df[(summary_df["policy"] == policy) & (summary_df["target"] == "mean")]
        cvar_sub = summary_df[(summary_df["policy"] == policy) & (summary_df["target"] == f"cvar_alpha_{est_cfg.alpha:.2f}")]
        tv_rows.append(
            {
                "scenario_name": scenario.name,
                "policy": policy,
                "true_mean": float(mean_sub["true_value"].iloc[0]),
                "estimated_mean": float(mean_sub["estimate_mean"].iloc[0]),
                "mean_bias": float(mean_sub["bias"].iloc[0]),
                "true_cvar": float(cvar_sub["true_value"].iloc[0]),
                "estimated_cvar": float(cvar_sub["estimate_mean"].iloc[0]),
                "cvar_bias": float(cvar_sub["bias"].iloc[0]),
            }
        )
    truth_vs_est_df = pd.DataFrame(tv_rows)

    return {
        "truth_df": truth_df,
        "pre_diag_df": pre_diag_df,
        "decile_df": decile_df,
        "sample_df": sample_df,
        "per_rep_df": per_rep_df,
        "summary_df": summary_df,
        "selection_df": selection_df,
        "gap_df": gap_df,
        "truth_vs_est_df": truth_vs_est_df,
        "objective_curve_df": objective_curve_df,
        "objective_summary_df": objective_summary_df,
    }


# ======================================================================================
# SECTION 7. AUDIT BLOCK
# ======================================================================================


def run_tail_transport_audit_for_scenario(scenario: ScenarioConfig, cfg: FullConfig) -> Dict[str, pd.DataFrame]:
    est_cfg = cfg.est_cfg
    audit_cfg = cfg.audit_cfg
    root_seed = cfg.global_cfg.seed

    # Source-side calibrator is always trained on policy A base.
    rng_source = make_rng(root_seed, scenario.name, "audit", "source")
    source_df = sample_policy("A", est_cfg.n_calib, rng_source, scenario, variant="base", with_meta=True)
    s_train = source_df["S"].to_numpy()
    y_train = source_df["Y"].to_numpy()

    # Reference threshold from the source side. This is only a reference line in plots.
    _, s_eval_base, _ = sample_policy("A", est_cfg.n_eval, rng_source, scenario, variant="base", with_meta=False)
    _, source_t0, source_t_grid, source_obj = estimate_direct_cvar_isotonic(
        s_train=s_train,
        y_train=y_train,
        s_eval=s_eval_base,
        alpha=est_cfg.alpha,
        grid_size=est_cfg.grid_size,
        return_curve=True,
    )

    audit_rows = []
    band_rows = []
    pilot_curve_rows = []

    for rep in tqdm(range(audit_cfg.audit_replications), desc=f"audit::{scenario.name}", leave=False):
        for variant in AUDIT_VARIANTS:
            rng_probe = make_rng(root_seed, scenario.name, "audit", rep, variant)
            probe_df = sample_policy("A", audit_cfg.n_probe_total, rng_probe, scenario, variant=variant, with_meta=True)

            idx = np.arange(len(probe_df))
            rng_probe.shuffle(idx)
            split = int(audit_cfg.pilot_frac * len(idx))
            pilot_df = probe_df.iloc[idx[:split]].reset_index(drop=True)
            audit_df = probe_df.iloc[idx[split:]].reset_index(drop=True)

            # Pilot chooses the target-specific threshold using the source-trained calibrator.
            est_pilot, t_hat_pilot, t_grid_pilot, obj_pilot = estimate_direct_cvar_isotonic(
                s_train=s_train,
                y_train=y_train,
                s_eval=pilot_df["S"].to_numpy(),
                alpha=est_cfg.alpha,
                grid_size=est_cfg.grid_size,
                return_curve=True,
            )

            # Save one pilot curve per variant for the first replication.
            if rep == 0:
                for t, obj in zip(t_grid_pilot, obj_pilot):
                    pilot_curve_rows.append(
                        {
                            "scenario_name": scenario.name,
                            "variant": variant,
                            "t": float(t),
                            "objective": float(obj),
                        }
                    )

            point_audit = two_moment_wald_audit(
                s_train=s_train,
                y_train=y_train,
                audit_df=audit_df,
                t0=t_hat_pilot,
                alpha=est_cfg.alpha,
                wald_alpha=audit_cfg.wald_alpha,
            )
            point_audit.update(
                {
                    "scenario_name": scenario.name,
                    "rep": rep,
                    "variant": variant,
                    "source_t0_reference": float(source_t0),
                    "pilot_t_hat": float(t_hat_pilot),
                    "pilot_estimated_cvar": float(est_pilot),
                }
            )
            audit_rows.append(point_audit)

            band_df = band_audit_around_t_hat(
                s_train=s_train,
                y_train=y_train,
                audit_df=audit_df,
                t_grid=t_grid_pilot,
                t_hat=t_hat_pilot,
                alpha=est_cfg.alpha,
                wald_alpha=audit_cfg.wald_alpha,
                band_width=audit_cfg.band_width,
            )
            band_df["scenario_name"] = scenario.name
            band_df["rep"] = rep
            band_df["variant"] = variant
            band_rows.append(band_df)

    audit_df = pd.DataFrame(audit_rows)
    band_df = pd.concat(band_rows, ignore_index=True)
    pilot_curve_df = pd.DataFrame(pilot_curve_rows)

    summary_rows = []
    for variant, g in audit_df.groupby("variant"):
        summary_rows.append(
            {
                "scenario_name": scenario.name,
                "variant": variant,
                "reject_rate": float(g["reject"].mean()),
                "mean_p_value": float(g["p_value"].mean()),
                "mean_wald_stat": float(g["wald_stat"].mean()),
                "mean_g1": float(g["mean_g1_quantile_moment"].mean()),
                "mean_g2": float(g["mean_g2_stoploss_moment"].mean()),
                "mean_pilot_t_hat": float(g["pilot_t_hat"].mean()),
                "sd_pilot_t_hat": float(g["pilot_t_hat"].std(ddof=1)),
            }
        )
    audit_summary_df = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)

    band_summary_rows = []
    for variant, g in band_df.groupby("variant"):
        band_summary_rows.append(
            {
                "scenario_name": scenario.name,
                "variant": variant,
                "band_reject_any_rate": float(g.groupby("rep")["band_reject_any"].max().mean()),
                "mean_band_wald_stat": float(g["wald_stat"].mean()),
                "mean_band_p_value": float(g["p_value"].mean()),
            }
        )
    band_summary_df = pd.DataFrame(band_summary_rows).sort_values("variant").reset_index(drop=True)

    # Tail residual inspection on the most problematic variant from the summary table.
    worst_variant = str(audit_summary_df.sort_values("mean_wald_stat", ascending=False)["variant"].iloc[0])
    worst_rep = int(
        audit_df.loc[audit_df["variant"] == worst_variant].sort_values("wald_stat", ascending=False)["rep"].iloc[0]
    )
    worst_t_hat = float(
        audit_df[(audit_df["variant"] == worst_variant) & (audit_df["rep"] == worst_rep)]["pilot_t_hat"].iloc[0]
    )
    rng_probe = make_rng(root_seed, scenario.name, "audit", worst_rep, worst_variant)
    probe_df = sample_policy("A", audit_cfg.n_probe_total, rng_probe, scenario, variant=worst_variant, with_meta=True)
    tail_resid_df = compute_tail_residuals_fixed_t(s_train=s_train, y_train=y_train, probe_df=probe_df, t_fixed=worst_t_hat)
    tail_resid_dec_df = residuals_by_score_decile(tail_resid_df, n_bins=cfg.plot_cfg.score_deciles)

    return {
        "source_df": source_df,
        "audit_df": audit_df,
        "audit_summary_df": audit_summary_df,
        "band_df": band_df,
        "band_summary_df": band_summary_df,
        "pilot_curve_df": pilot_curve_df,
        "tail_resid_df": tail_resid_df,
        "tail_resid_dec_df": tail_resid_dec_df,
        "source_reference_df": pd.DataFrame(
            {
                "scenario_name": [scenario.name],
                "source_t0_reference": [float(source_t0)],
                "source_reference_cvar": [float(source_obj.max())],
            }
        ),
        "worst_variant_df": pd.DataFrame(
            {
                "scenario_name": [scenario.name],
                "worst_variant": [worst_variant],
                "worst_rep": [worst_rep],
                "worst_t_hat": [worst_t_hat],
            }
        ),
    }


# ======================================================================================
# SECTION 8. SENSITIVITY SWEEPS
# ======================================================================================


def run_alpha_sensitivity(scenario: ScenarioConfig, cfg: FullConfig) -> pd.DataFrame:
    rows = []
    for alpha in tqdm(cfg.sweep_cfg.alpha_grid, desc=f"alpha_sweep::{scenario.name}", leave=False):
        # Use a shallow copy of config objects via dataclass replacement semantics.
        est_cfg = EstimationConfig(
            alpha=float(alpha),
            grid_size=cfg.est_cfg.grid_size,
            truth_n=cfg.est_cfg.truth_n,
            n_calib=cfg.est_cfg.n_calib,
            n_eval=cfg.est_cfg.n_eval,
            replications=cfg.sweep_cfg.replications_per_sensitivity,
            validation_frac=cfg.est_cfg.validation_frac,
        )
        cfg_alpha = FullConfig(
            global_cfg=cfg.global_cfg,
            est_cfg=est_cfg,
            audit_cfg=cfg.audit_cfg,
            sweep_cfg=cfg.sweep_cfg,
            plot_cfg=cfg.plot_cfg,
        )
        out = run_compare_policies_for_scenario(scenario, cfg_alpha)
        selection_df = out["selection_df"].copy()
        selection_df["alpha"] = alpha
        rows.append(selection_df)
    return pd.concat(rows, ignore_index=True)


def run_oracle_budget_sensitivity(scenario: ScenarioConfig, cfg: FullConfig) -> pd.DataFrame:
    rows = []
    for n_probe_total in tqdm(cfg.sweep_cfg.oracle_budget_grid, desc=f"budget_sweep::{scenario.name}", leave=False):
        audit_cfg = AuditConfig(
            n_probe_total=int(n_probe_total),
            pilot_frac=cfg.audit_cfg.pilot_frac,
            wald_alpha=cfg.audit_cfg.wald_alpha,
            audit_replications=cfg.sweep_cfg.replications_per_sensitivity,
            band_width=cfg.audit_cfg.band_width,
        )
        cfg_budget = FullConfig(
            global_cfg=cfg.global_cfg,
            est_cfg=cfg.est_cfg,
            audit_cfg=audit_cfg,
            sweep_cfg=cfg.sweep_cfg,
            plot_cfg=cfg.plot_cfg,
        )
        out = run_tail_transport_audit_for_scenario(scenario, cfg_budget)
        summary_df = out["audit_summary_df"].copy()
        summary_df["n_probe_total"] = int(n_probe_total)
        rows.append(summary_df)
    return pd.concat(rows, ignore_index=True)


# ======================================================================================
# SECTION 9. PLOTTING
# ======================================================================================


def plot_policy_distributions(sample_df: pd.DataFrame, scenario_name: str, plot_cfg: PlotConfig) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=plot_cfg.figsize_wide)
    policy_order = ["A", "B"]

    for policy in policy_order:
        vals_y = sample_df.loc[sample_df["policy"] == policy, "Y"].to_numpy()
        vals_s = sample_df.loc[sample_df["policy"] == policy, "S"].to_numpy()
        axes[0, 0].hist(vals_y, bins=plot_cfg.bins, density=True, alpha=0.45, label=f"Policy {policy}")
        axes[0, 1].hist(vals_s, bins=plot_cfg.bins, density=True, alpha=0.45, label=f"Policy {policy}")

        y_sorted = np.sort(vals_y)
        cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        axes[1, 0].plot(y_sorted, cdf, label=f"Policy {policy}")

        tail_cut = np.quantile(vals_y, 0.20)
        tail_vals = vals_y[vals_y <= tail_cut]
        axes[1, 1].hist(tail_vals, bins=max(20, plot_cfg.bins // 2), density=True, alpha=0.45, label=f"Policy {policy}")

    axes[0, 0].set_title(f"{scenario_name}: oracle distribution")
    axes[0, 1].set_title(f"{scenario_name}: judge distribution")
    axes[1, 0].set_title(f"{scenario_name}: oracle CDF")
    axes[1, 1].set_title(f"{scenario_name}: bottom-20% oracle zoom")
    axes[0, 0].set_xlabel("Oracle score Y")
    axes[0, 1].set_xlabel("Judge score S")
    axes[1, 0].set_xlabel("Oracle score Y")
    axes[1, 1].set_xlabel("Oracle score Y")
    for ax in axes.ravel():
        ax.legend()
    fig.suptitle("Policy distributions", fontsize=12)
    return fig


def plot_decile_panels(decile_df: pd.DataFrame, scenario_name: str, plot_cfg: PlotConfig) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for policy in ["A", "B"]:
        sub_y = decile_df[(decile_df["policy"] == policy) & (decile_df["target_type"] == "Y")]
        sub_z = decile_df[(decile_df["policy"] == policy) & (decile_df["target_type"] == "tail_loss_at_true_q")]
        axes[0].plot(sub_y["score_decile_center"], sub_y["mean_target"], marker="o", label=f"Policy {policy}")
        axes[1].plot(sub_z["score_decile_center"], sub_z["mean_target"], marker="o", label=f"Policy {policy}")
    axes[0].set_title(f"{scenario_name}: mean oracle by judge-score decile")
    axes[1].set_title(f"{scenario_name}: mean tail loss by judge-score decile")
    axes[0].set_xlabel("Judge score decile center")
    axes[1].set_xlabel("Judge score decile center")
    axes[0].set_ylabel("Mean target")
    axes[1].set_ylabel("Mean target")
    axes[0].legend()
    axes[1].legend()
    return fig


def plot_objective_curves(objective_curve_df: pd.DataFrame, objective_summary_df: pd.DataFrame, scenario_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in ["A", "B"]:
        sub = objective_curve_df[objective_curve_df["policy"] == policy]
        ax.plot(sub["t"], sub["objective"], label=f"Policy {policy}")
        t_hat = float(objective_summary_df.loc[objective_summary_df["policy"] == policy, "t_hat"].iloc[0])
        ax.axvline(t_hat, linestyle="--")
    ax.set_title(f"{scenario_name}: CVaR objective and selected t*")
    ax.set_xlabel("t")
    ax.set_ylabel("CVaR objective")
    ax.legend()
    return fig


def plot_audit_pilot_curves(pilot_curve_df: pd.DataFrame, source_ref_df: pd.DataFrame, scenario_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(pilot_curve_df["variant"].unique()):
        sub = pilot_curve_df[pilot_curve_df["variant"] == variant]
        ax.plot(sub["t"], sub["objective"], label=variant)
    ax.axvline(float(source_ref_df["source_t0_reference"].iloc[0]), linestyle="--", label="source reference t0")
    ax.set_title(f"{scenario_name}: target pilot objective curves")
    ax.set_xlabel("t")
    ax.set_ylabel("Pilot CVaR objective")
    ax.legend()
    return fig


def plot_tail_residual_panels(tail_resid_dec_df: pd.DataFrame, tail_resid_df: pd.DataFrame, label: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(tail_resid_dec_df["score_decile_center"], tail_resid_dec_df["mean_residual"], marker="o")
    axes[0].set_title(f"{label}: tail residual by judge-score decile")
    axes[0].set_xlabel("Judge score decile center")
    axes[0].set_ylabel("Mean residual")

    axes[1].plot(tail_resid_dec_df["score_decile_center"], tail_resid_dec_df["mean_oracle_tail_loss"], marker="o", label="Oracle tail loss")
    axes[1].plot(tail_resid_dec_df["score_decile_center"], tail_resid_dec_df["mean_calibrated_tail_loss"], marker="o", label="Calibrated tail loss")
    axes[1].set_title(f"{label}: oracle vs calibrated tail loss")
    axes[1].set_xlabel("Judge score decile center")
    axes[1].set_ylabel("Mean tail loss")
    axes[1].legend()
    return fig


def plot_alpha_sensitivity(alpha_df: pd.DataFrame, scenario_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in sorted(alpha_df["target"].unique()):
        sub = alpha_df[alpha_df["target"] == target]
        ax.plot(sub["alpha"], sub["correct_policy_selected_rate"], marker="o", label=target)
    ax.set_title(f"{scenario_name}: selection sensitivity to alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Correct policy selected rate")
    ax.legend()
    return fig


def plot_budget_sensitivity(budget_df: pd.DataFrame, scenario_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(budget_df["variant"].unique()):
        sub = budget_df[budget_df["variant"] == variant]
        ax.plot(sub["n_probe_total"], sub["reject_rate"], marker="o", label=variant)
    ax.set_title(f"{scenario_name}: audit reject rate by oracle budget")
    ax.set_xlabel("Probe size")
    ax.set_ylabel("Reject rate")
    ax.legend()
    return fig


# ======================================================================================
# SECTION 10. SCRIPTED REPORTING
# ======================================================================================


def report_compare_block(out: Dict[str, pd.DataFrame], scenario: ScenarioConfig) -> None:
    print("\n" + "#" * 100)
    print(f"SCENARIO: {scenario.name}")
    print(textwrap.fill(scenario.description, width=100))
    print("#" * 100)

    prettify_table(out["truth_df"].round(4), "True policy metrics")
    prettify_table(out["pre_diag_df"].round(4), "Pre-estimation diagnostics")
    prettify_table(out["truth_vs_est_df"].round(4), "Truth versus estimate summary")
    prettify_table(out["summary_df"].round(4), "Estimator summary")
    prettify_table(out["selection_df"].round(4), "Policy selection summary")
    prettify_table(out["gap_df"].round(4), "Policy gap summary")

    print("\nInterpretation notes")
    print("- The truth-versus-estimate table is the first place to look. It shows whether the direct estimator recovers the target quantities on average.")
    print("- The selection table is the decision-making view. It answers how often the estimated criterion picks the policy that is actually best under that estimand.")
    print("- The gap table is useful when the means are close. It tells whether the sign of A minus B is recovered reliably.")


def report_audit_block(out: Dict[str, pd.DataFrame], scenario: ScenarioConfig) -> None:
    prettify_table(out["audit_summary_df"].round(4), "Tail transport audit summary")
    prettify_table(out["band_summary_df"].round(4), "Band-audit sensitivity summary")
    prettify_table(out["worst_variant_df"].round(4), "Worst-case residual inspection target")
    prettify_table(
        out["tail_resid_df"][[
            "issue_type",
            "catastrophic",
            "scenario",
            "S",
            "Y",
            "oracle_tail_loss",
            "calibrated_tail_loss",
            "residual",
        ]].round(4),
        "Top residual cases",
        n=12,
    )

    print("\nInterpretation notes")
    print("- Stable should reject rarely. Fooled_judge should reject often. Hard_shift and weird_monotone are intermediate stress cases.")
    print("- The point audit is the core demo method. The band audit is only a sensitivity layer around the pilot-chosen threshold.")
    print("- The residual table makes the failure mode concrete by showing where the judge was too optimistic about lower-tail safety.")


# ======================================================================================
# SECTION 11. ORCHESTRATION
# ======================================================================================


def save_block_outputs(
    scenario: ScenarioConfig,
    compare_out: Dict[str, pd.DataFrame],
    audit_out: Dict[str, pd.DataFrame],
    alpha_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    cfg: FullConfig,
    scenario_dir: Path,
) -> None:
    if cfg.global_cfg.save_tables:
        for name, df in compare_out.items():
            save_table(df, scenario_dir, stem=name, float_format=cfg.global_cfg.csv_float_format)
        for name, df in audit_out.items():
            save_table(df, scenario_dir, stem=name, float_format=cfg.global_cfg.csv_float_format)
        save_table(alpha_df, scenario_dir, stem="alpha_sensitivity", float_format=cfg.global_cfg.csv_float_format)
        save_table(budget_df, scenario_dir, stem="oracle_budget_sensitivity", float_format=cfg.global_cfg.csv_float_format)

    if cfg.global_cfg.save_figures:
        fig = plot_policy_distributions(compare_out["sample_df"], scenario.name, cfg.plot_cfg)
        save_figure(fig, scenario_dir, stem="policy_distributions", dpi=cfg.plot_cfg.dpi)

        fig = plot_decile_panels(compare_out["decile_df"], scenario.name, cfg.plot_cfg)
        save_figure(fig, scenario_dir, stem="decile_panels", dpi=cfg.plot_cfg.dpi)

        fig = plot_objective_curves(compare_out["objective_curve_df"], compare_out["objective_summary_df"], scenario.name)
        save_figure(fig, scenario_dir, stem="objective_curves", dpi=cfg.plot_cfg.dpi)

        fig = plot_audit_pilot_curves(audit_out["pilot_curve_df"], audit_out["source_reference_df"], scenario.name)
        save_figure(fig, scenario_dir, stem="audit_pilot_curves", dpi=cfg.plot_cfg.dpi)

        worst_label = str(audit_out["worst_variant_df"]["worst_variant"].iloc[0])
        fig = plot_tail_residual_panels(audit_out["tail_resid_dec_df"], audit_out["tail_resid_df"], label=worst_label)
        save_figure(fig, scenario_dir, stem="tail_residual_panels", dpi=cfg.plot_cfg.dpi)

        fig = plot_alpha_sensitivity(alpha_df, scenario.name)
        save_figure(fig, scenario_dir, stem="alpha_sensitivity", dpi=cfg.plot_cfg.dpi)

        fig = plot_budget_sensitivity(budget_df, scenario.name)
        save_figure(fig, scenario_dir, stem="oracle_budget_sensitivity", dpi=cfg.plot_cfg.dpi)


def run_scenario(scenario: ScenarioConfig, cfg: FullConfig, root_out_dir: Path) -> None:
    scenario_dir = root_out_dir / scenario.name
    ensure_dir(scenario_dir)

    print("\n" + "*" * 100)
    print(f"Running scenario: {scenario.name}")
    print(textwrap.fill(scenario.description, width=100))
    print("*" * 100)

    compare_out = run_compare_policies_for_scenario(scenario, cfg)
    report_compare_block(compare_out, scenario)

    audit_out = run_tail_transport_audit_for_scenario(scenario, cfg)
    report_audit_block(audit_out, scenario)

    alpha_df = run_alpha_sensitivity(scenario, cfg)
    prettify_table(alpha_df.round(4), "Alpha sensitivity")

    budget_df = run_oracle_budget_sensitivity(scenario, cfg)
    prettify_table(budget_df.round(4), "Oracle budget sensitivity")

    save_block_outputs(scenario, compare_out, audit_out, alpha_df, budget_df, cfg, scenario_dir)

    print("\nSection takeaways")
    print("- The script keeps the original direct-only isotonic methodology and adds breadth through scenarios, repetitions, and sensitivity sweeps.")
    print("- Each scenario folder contains tables and plots in a fixed order, so the demo is easy to show live or review offline.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Mean CJE and Direct CVaR-CJE demo script.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(DEFAULT_SCENARIOS.keys()),
        help="Scenario names to run. Defaults to all built-in scenarios.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    set_pandas_display()
    args = parse_args()

    cfg = FullConfig()
    global_cfg = cfg.global_cfg
    if args.out_dir is not None or args.show_figures:
        global_cfg = GlobalConfig(
            seed=cfg.global_cfg.seed,
            out_dir=args.out_dir or cfg.global_cfg.out_dir,
            save_tables=cfg.global_cfg.save_tables,
            save_figures=cfg.global_cfg.save_figures,
            show_figures=args.show_figures or cfg.global_cfg.show_figures,
            csv_float_format=cfg.global_cfg.csv_float_format,
        )
        cfg = FullConfig(
            global_cfg=global_cfg,
            est_cfg=cfg.est_cfg,
            audit_cfg=cfg.audit_cfg,
            sweep_cfg=cfg.sweep_cfg,
            plot_cfg=cfg.plot_cfg,
        )

    root_out_dir = Path(cfg.global_cfg.out_dir)
    ensure_dir(root_out_dir)

    print("\n" + "=" * 100)
    print("CONFIGURATION")
    print("=" * 100)
    print("Global config:")
    print(pd.Series(asdict(cfg.global_cfg)).to_string())
    print("\nEstimation config:")
    print(pd.Series(asdict(cfg.est_cfg)).to_string())
    print("\nAudit config:")
    print(pd.Series(asdict(cfg.audit_cfg)).to_string())
    print("\nSweep config:")
    print(pd.Series(asdict(cfg.sweep_cfg)).to_string())

    selected = []
    for name in args.scenarios:
        if name not in DEFAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Available: {sorted(DEFAULT_SCENARIOS)}")
        selected.append(DEFAULT_SCENARIOS[name])

    for scenario in selected:
        run_scenario(scenario, cfg, root_out_dir)

    print("\nDone.")
    print(f"Outputs written to: {root_out_dir.resolve()}")

    if cfg.global_cfg.show_figures:
        plt.show()


if __name__ == "__main__":
    main()
