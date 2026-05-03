"""
Audit-transport investigation: which moment catches which break?

Background
----------
The cvar_v5 audit has two moments:
    g_1 = 1{Y ≤ t̂_α} − α        (quantile correctness)
    g_2 = (t̂_α − Y)_+ − ĥ_{t̂_α}(s)  (marginal shortfall transport)

The paper's appendix_c_audit notes these test different things. We want
to measure on a known DGP exactly which moment moves under which break.

Setup
-----
Logger DGP:
    Y_L ~ Beta(a, b)
    S_L = scale · (Y_L − 0.5) + ε,   ε ~ N(0, σ²)

Target DGP (three independent perturbation knobs):
    Y_T ~ Beta(a, b + δ_y)
    S_T = (scale + δ_scale) · (Y_T − 0.5) + ε,
                                    ε ~ N(0, (σ + δ_sigma)²)

    δ_y      : changes Y's marginal (current cvar_v5 perturbation).
               Strong transport breaks via Bayes; marginal transport may
               or may not (PAVA's mean-preserving keeps it tight).
    δ_scale  : changes how steeply S responds to Y. Conditional shortfall
               directly differs.
    δ_sigma  : changes noise of S | Y. Conditional shortfall changes via
               heteroscedastic shift.

For the audit we draw:
    CALIB ← logger (oracle)
    AUDIT ← TARGET (oracle) — paper's intent (held out from target)
    EVAL  ← target (no oracle)

Population truths
-----------------
We approximate population quantities by drawing huge samples (n=200k) from
each DGP and computing empirical means of (t-Y)_+, P(Y≤t), and ĥ_t(s)
where ĥ_t is the calibrator's pooled fit (logger-trained).

For each perturbation:
    g_1*  =  P_target(Y ≤ t̂_α)  −  α
    g_2*  =  E_target[(t̂_α − Y)_+]  −  E_target[ĥ_{t̂_α}(s)]
    where t̂_α is computed from the population calibrator on the target's
    EVAL distribution.

A moment is INFORMATIVE for a break if |population_g| grows with the
break's strength.

Output
------
Writes a CSV at runs/<ts>_audit_transport/results.csv and prints the
table. No production code changes. Research-only — see ../README.md.

Run:
    python -m cvar_v5._archive.audit_transport_investigation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.isotonic import IsotonicRegression


# ---------------- DGP with logger + target perturbation knobs ----------------


@dataclass(frozen=True)
class PolicyParams:
    a: float = 1.0
    b: float = 1.0
    scale: float = 4.0
    sigma: float = 0.8


@dataclass(frozen=True)
class TargetPerturbation:
    delta_y: float = 0.0       # Y_T ~ Beta(a, b + δ_y)
    delta_scale: float = 0.0   # scale → scale + δ_scale
    delta_sigma: float = 0.0   # σ → σ + δ_sigma


def sample_logger(params: PolicyParams, n: int, seed: int, with_oracle: bool):
    rng = np.random.default_rng(seed)
    y = rng.beta(params.a, params.b, size=n)
    s = params.scale * (y - 0.5) + rng.normal(0.0, params.sigma, size=n)
    return s, (y if with_oracle else None)


def sample_target(
    params: PolicyParams, pert: TargetPerturbation, n: int, seed: int,
    with_oracle: bool,
):
    rng = np.random.default_rng(seed)
    y = rng.beta(params.a, params.b + pert.delta_y, size=n)
    sc = params.scale + pert.delta_scale
    sg = max(1e-6, params.sigma + pert.delta_sigma)
    s = sc * (y - 0.5) + rng.normal(0.0, sg, size=n)
    return s, (y if with_oracle else None)


# ---------------- Calibrator (pooled, no cross-fit needed for this study) ----


def fit_pooled_calibrator(s_calib, y_calib, t_grid: np.ndarray):
    """One IsotonicRegression(decreasing) per t in t_grid; returns predict(s) → (n, |T|)."""
    fits = []
    for t in t_grid:
        z = np.maximum(t - y_calib, 0.0)
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(s_calib, z)
        fits.append(ir)

    def predict(s_query: np.ndarray) -> np.ndarray:
        out = np.empty((len(s_query), len(t_grid)), dtype=np.float64)
        for j, ir in enumerate(fits):
            out[:, j] = ir.predict(s_query)
        return out

    return predict


# ---------------- Audit moments and Wald ------------------------------------


def saddle_argmax_t(H: np.ndarray, t_grid: np.ndarray, alpha: float):
    psi = t_grid - H.mean(axis=0) / alpha
    j_star = int(np.argmax(psi))
    return j_star, float(t_grid[j_star])


def g_pair(s, y, h_t, t, alpha):
    g1 = (y <= t).astype(np.float64) - alpha
    g2 = np.maximum(t - y, 0.0) - h_t
    return g1, g2


def population_audit_moments(
    params: PolicyParams,
    pert: TargetPerturbation,
    alpha: float,
    t_grid: np.ndarray,
    n_huge: int,
    seed: int,
):
    """
    Population-level g_1* and g_2*:
      1. Draw HUGE logger sample → fit pooled calibrator (logger-truth proxy).
      2. Draw HUGE target sample → t̂_α from the saddle-point on target.
      3. Draw INDEPENDENT HUGE target audit sample → compute g_1, g_2 means.
    """
    s_logger, y_logger = sample_logger(params, n_huge, seed=seed, with_oracle=True)
    cal_predict = fit_pooled_calibrator(s_logger, y_logger, t_grid)

    s_eval, _ = sample_target(params, pert, n_huge, seed=seed + 1, with_oracle=False)
    H_eval = cal_predict(s_eval)
    j_star, t_star = saddle_argmax_t(H_eval, t_grid, alpha)

    s_audit, y_audit = sample_target(
        params, pert, n_huge, seed=seed + 2, with_oracle=True,
    )
    h_t_audit = cal_predict(s_audit)[:, j_star]
    g1, g2 = g_pair(s_audit, y_audit, h_t_audit, t_star, alpha)
    return {
        "t_star": t_star,
        "g1_pop": float(g1.mean()),
        "g2_pop": float(g2.mean()),
    }


# ---------------- Finite-sample audit (single rep) --------------------------


def run_one_rep(
    params: PolicyParams,
    pert: TargetPerturbation,
    alpha: float,
    n_calib: int,
    n_audit: int,
    n_eval: int,
    t_grid: np.ndarray,
    seed: int,
):
    """
    One MC rep:
      CALIB ← logger
      EVAL  ← target (drives t̂_α via saddle-point)
      AUDIT ← target (held-out, has Y)
    Returns ḡ_1, ḡ_2 and W_n_g1g2, W_n_g2only computed from per-row sample cov.
    """
    s_c, y_c = sample_logger(params, n_calib, seed=seed, with_oracle=True)
    cal_predict = fit_pooled_calibrator(s_c, y_c, t_grid)

    s_e, _ = sample_target(params, pert, n_eval, seed=seed + 1, with_oracle=False)
    H_e = cal_predict(s_e)
    j_star, t_star = saddle_argmax_t(H_e, t_grid, alpha)

    s_a, y_a = sample_target(params, pert, n_audit, seed=seed + 2, with_oracle=True)
    h_t_a = cal_predict(s_a)[:, j_star]
    g1, g2 = g_pair(s_a, y_a, h_t_a, t_star, alpha)
    g_bar = np.array([g1.mean(), g2.mean()])

    cov_per_row = np.cov(np.vstack([g1, g2]), ddof=1)
    omega = cov_per_row / n_audit

    # 2-moment Wald (χ²_2 critical = 5.991 at η=0.05)
    omega_inv = np.linalg.pinv(omega)
    W_g1g2 = float(g_bar @ omega_inv @ g_bar)

    # 1-moment on g_2 (χ²_1 critical = 3.841 at η=0.05)
    W_g2 = float(g_bar[1] ** 2 / max(omega[1, 1], 1e-30))

    return {
        "t_star": t_star,
        "g1_bar": float(g_bar[0]),
        "g2_bar": float(g_bar[1]),
        "W_g1g2": W_g1g2,
        "W_g2_only": W_g2,
        "reject_g1g2": int(W_g1g2 > stats.chi2.ppf(0.95, df=2)),
        "reject_g2_only": int(W_g2 > stats.chi2.ppf(0.95, df=1)),
    }


# ---------------- The investigation ------------------------------------------


def main():
    out_dir = Path(__file__).parent.parent / "mc" / "runs" / (
        datetime.now().strftime("%Y-%m-%dT%H%M%S") + "_audit_transport"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    params = PolicyParams(a=1.0, b=1.0)  # uniform
    alpha = 0.10
    t_grid = np.linspace(0.0, 1.0, 121)
    n_huge = 200_000

    perturbations = {
        "null":             TargetPerturbation(),
        "delta_y_0.3":      TargetPerturbation(delta_y=0.3),
        "delta_y_1.0":      TargetPerturbation(delta_y=1.0),
        "delta_scale_+1":   TargetPerturbation(delta_scale=1.0),
        "delta_scale_-1":   TargetPerturbation(delta_scale=-1.0),
        "delta_scale_+2":   TargetPerturbation(delta_scale=2.0),
        "delta_sigma_+0.4": TargetPerturbation(delta_sigma=0.4),
        "delta_sigma_+0.8": TargetPerturbation(delta_sigma=0.8),
        "mixed_y_scale":    TargetPerturbation(delta_y=0.5, delta_scale=0.5),
    }

    print("=" * 90)
    print(f"POPULATION g* (n_huge={n_huge}, uniform policy, α={alpha})")
    print("=" * 90)
    print(f"{'pert':<22} {'t_star':>8} {'g1_pop':>11} {'g2_pop':>11}")
    pop_rows = []
    for name, pert in perturbations.items():
        m = population_audit_moments(params, pert, alpha, t_grid, n_huge, seed=99999)
        pop_rows.append({"pert": name, **m})
        print(f"{name:<22} {m['t_star']:>8.5f} {m['g1_pop']:>+10.5f}  {m['g2_pop']:>+10.5f}")
    pl.DataFrame(pop_rows).write_csv(out_dir / "population.csv")

    # Finite-sample size and rejection at MEDIUM scale
    n_calib, n_audit, n_eval, R = 600, 250, 1000, 100
    print()
    print("=" * 90)
    print(f"FINITE-SAMPLE rejection rates (R={R}, n_calib={n_calib}, "
          f"n_audit={n_audit}, n_eval={n_eval})")
    print("=" * 90)
    print(f"{'pert':<22} {'reject g1g2':>13} {'reject g2_only':>16}  "
          f"{'mean ḡ_1':>10} {'mean ḡ_2':>10}")

    fs_rows = []
    for name, pert in perturbations.items():
        rejs_g1g2 = []
        rejs_g2 = []
        g1_means = []
        g2_means = []
        for r in range(R):
            seed = 7919 * r + 31
            res = run_one_rep(params, pert, alpha, n_calib, n_audit, n_eval, t_grid, seed)
            rejs_g1g2.append(res["reject_g1g2"])
            rejs_g2.append(res["reject_g2_only"])
            g1_means.append(res["g1_bar"])
            g2_means.append(res["g2_bar"])
        row = {
            "pert": name,
            "reject_g1g2": float(np.mean(rejs_g1g2)),
            "reject_g2_only": float(np.mean(rejs_g2)),
            "mean_g1_bar": float(np.mean(g1_means)),
            "mean_g2_bar": float(np.mean(g2_means)),
        }
        fs_rows.append(row)
        print(f"{name:<22} {row['reject_g1g2']:>13.2f} {row['reject_g2_only']:>16.2f}  "
              f"{row['mean_g1_bar']:>+9.5f} {row['mean_g2_bar']:>+9.5f}")
    pl.DataFrame(fs_rows).write_csv(out_dir / "finite_sample.csv")

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
