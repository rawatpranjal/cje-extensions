"""Semi-synthetic data generating process fit to the Arena dataset.

Paper: §A (semi-synthetic DGP underlying §A.1–§A.4).

Per-policy fully-parametric DGP for use as a known-truth ground in the
power-analysis Monte Carlo (`cvar_v3/run_monte_carlo.py`).

Design:
  - Y marginal: empirical CDF over the policy's observed oracle labels.
    For `unhelpful`, a mixture of P(Y=0)=p_zero and the conditional CDF on Y>0.
  - S | Y: monotone isotonic m_p(Y) + heteroscedastic Gaussian noise binned
    by Y-quartile (4 bins). Output is clipped to [0, 1].
  - Cross-policy joint structure (r(Y_base, Y_target) ≈ 0.81 in the real
    data) is NOT preserved — each policy is sampled independently. The
    audit / CVaR analysis is per-policy (calibrate on base, eval on target),
    so per-policy marginals are what the estimator actually consumes.

Mis-specification knob:
  - `m_override` lets the caller force the conditional mean to come from
    another DGP's m (typically `dgp_base`). Together with `delta` and a
    perturbation shape, this gives a clean null (calibrator transports
    perfectly when m_override=dgp_base and delta=0) and a controllable
    alternative.

API:
  - fit_arena_dgp(data_root) -> dict[policy, PolicyDGP]
  - sample_synthetic(dgp, n, rng, delta=0.0, perturbation="none", ...)
  - cvar_truth(dgp, alpha, ...) — population CVaR for a given DGP+pertubation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression


POLICIES: tuple[str, ...] = (
    "base", "clone", "premium", "parallel_universe_prompt", "unhelpful",
)


@dataclass
class PolicyDGP:
    policy: str
    y_grid: np.ndarray             # sorted unique Y values observed for this policy
    y_probs: np.ndarray            # empirical pmf, sums to 1
    m_iso: IsotonicRegression      # E[S | Y] fitted on real (Y, S)
    sigma_by_quartile: np.ndarray  # 4 floats: σ of S - m(Y) per Y-quartile bin
    y_quartile_edges: np.ndarray   # 3 cut points (P25, P50, P75) for bin indexing
    # unhelpful-only mixture parameters (None for standard policies):
    p_zero: Optional[float] = None
    y_grid_pos: Optional[np.ndarray] = None  # Y>0 grid for the mixture's continuous part
    y_probs_pos: Optional[np.ndarray] = None
    mean_s_at_zero: Optional[float] = None
    sigma_s_at_zero: Optional[float] = None


def _load_policy_pairs(data_root: Path, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (S, Y) pairs from the policy's source file."""
    if policy == "base":
        df = pl.read_ndjson(data_root / "cje_dataset.jsonl")
        df = df.with_columns([
            pl.col("metadata").struct.field("judge_score").cast(pl.Float64).alias("s"),
            pl.col("metadata").struct.field("oracle_label").cast(pl.Float64).alias("y"),
        ]).select(["s", "y"]).drop_nulls()
    else:
        df = pl.read_ndjson(data_root / "responses" / f"{policy}_responses.jsonl")
        df = df.with_columns([
            pl.col("metadata").struct.field("judge_score").cast(pl.Float64).alias("s"),
            pl.col("metadata").struct.field("oracle_label").cast(pl.Float64).alias("y"),
        ]).select(["s", "y"]).drop_nulls()
    return df["s"].to_numpy(), df["y"].to_numpy()


def _fit_one_policy(policy: str, s: np.ndarray, y: np.ndarray) -> PolicyDGP:
    """Fit Y marginal + m(Y) + heteroscedastic σ by Y quartile."""
    # Empirical Y marginal (full)
    y_grid, counts = np.unique(y, return_counts=True)
    y_probs = counts / counts.sum()

    # Isotonic m(Y)
    m_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    m_iso.fit(y, s)

    # Quartile-binned σ of residuals
    edges = np.quantile(y, [0.25, 0.50, 0.75])
    bins = np.searchsorted(edges, y, side="right").clip(0, 3)  # 0..3
    resid = s - m_iso.predict(y)
    sigma_by_quartile = np.array([
        float(np.std(resid[bins == b], ddof=1)) if (bins == b).sum() > 1 else 0.05
        for b in range(4)
    ])

    dgp = PolicyDGP(
        policy=policy,
        y_grid=y_grid,
        y_probs=y_probs,
        m_iso=m_iso,
        sigma_by_quartile=sigma_by_quartile,
        y_quartile_edges=edges,
    )

    # Mixture component for `unhelpful`
    if policy == "unhelpful":
        zero_mask = y == 0.0
        if zero_mask.sum() > 0:
            dgp.p_zero = float(zero_mask.mean())
            dgp.mean_s_at_zero = float(s[zero_mask].mean())
            dgp.sigma_s_at_zero = float(np.std(s[zero_mask], ddof=1)) if zero_mask.sum() > 1 else 0.025
            # Conditional Y > 0 grid for the mixture's continuous part
            y_pos = y[~zero_mask]
            yg, cn = np.unique(y_pos, return_counts=True)
            dgp.y_grid_pos = yg
            dgp.y_probs_pos = cn / cn.sum()
    return dgp


def fit_arena_dgp(data_root: Path | str) -> dict[str, PolicyDGP]:
    """Fit a PolicyDGP for each of the 5 Arena policies."""
    data_root = Path(data_root)
    out: dict[str, PolicyDGP] = {}
    for p in POLICIES:
        s, y = _load_policy_pairs(data_root, p)
        out[p] = _fit_one_policy(p, s, y)
    return out


def _sample_y(dgp: PolicyDGP, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Y from the policy's empirical marginal (with mixture for unhelpful)."""
    if dgp.p_zero is not None and dgp.y_grid_pos is not None and dgp.y_probs_pos is not None:
        is_zero = rng.random(n) < dgp.p_zero
        y = np.empty(n)
        y[is_zero] = 0.0
        n_pos = int((~is_zero).sum())
        if n_pos > 0:
            y[~is_zero] = rng.choice(dgp.y_grid_pos, size=n_pos, p=dgp.y_probs_pos)
        return y
    return rng.choice(dgp.y_grid, size=n, p=dgp.y_probs)


def _apply_perturbation(
    m_y: np.ndarray, y: np.ndarray, delta: float,
    perturbation: str, q_low_threshold: Optional[float],
) -> np.ndarray:
    """Return a perturbed conditional mean for the target policy."""
    if delta == 0.0 or perturbation == "none":
        return m_y
    if perturbation == "uniform":
        return m_y + delta
    if perturbation == "tail":
        if q_low_threshold is None:
            raise ValueError("perturbation='tail' requires q_low_threshold (e.g. q_0.10 of Y_base)")
        return np.where(y <= q_low_threshold, m_y - delta, m_y)
    raise ValueError(f"unknown perturbation: {perturbation!r}")


def sample_synthetic(
    dgp: PolicyDGP,
    n: int,
    rng: np.random.Generator,
    delta: float = 0.0,
    perturbation: Literal["none", "uniform", "tail"] = "none",
    q_low_threshold: Optional[float] = None,
    m_override: Optional[PolicyDGP] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (S, Y) pairs from the policy DGP.

    The Y marginal and noise σ come from `dgp`. The conditional mean comes
    from `dgp.m_iso` by default, or from `m_override.m_iso` if provided —
    use this to enforce the well-specified null (target's S follows base's m
    so the calibrator transports perfectly at delta=0).

    For `unhelpful` Y=0 mass: S draws from N(mean_s_at_zero, sigma_s_at_zero)
    regardless of `m_override`, since base never observes Y=0 and m_base(0)
    is just the clipped extrapolation. Documented as a known regime quirk.
    """
    y = _sample_y(dgp, n, rng)
    m_source = m_override if m_override is not None else dgp
    m_y = m_source.m_iso.predict(y)
    m_y = _apply_perturbation(m_y, y, delta, perturbation, q_low_threshold)

    bins = np.searchsorted(dgp.y_quartile_edges, y, side="right").clip(0, 3)
    sigma_y = dgp.sigma_by_quartile[bins]
    if dgp.mean_s_at_zero is not None:
        zero_mask = y == 0.0
        m_y = np.where(zero_mask, dgp.mean_s_at_zero, m_y)
        sigma_y = np.where(zero_mask, dgp.sigma_s_at_zero, sigma_y)

    eps = rng.standard_normal(n)
    s = np.clip(m_y + sigma_y * eps, 0.0, 1.0)
    return s, y


def cvar_truth(
    dgp: PolicyDGP,
    alpha: float,
    delta: float = 0.0,
    perturbation: str = "none",
    n_truth: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Population CVaR for the target DGP.

    Y is δ-invariant, so cvar_truth depends on `dgp` and `alpha` only —
    delta and perturbation are accepted for API symmetry but ignored.
    """
    rng = rng or np.random.default_rng(0)
    y = _sample_y(dgp, n_truth, rng)
    q = float(np.quantile(y, alpha))
    tail = y[y <= q]
    return float(tail.mean()) if len(tail) else float("nan")


def q_lower_tail_threshold(dgp: PolicyDGP, alpha: float, n: int = 200_000,
                           rng: Optional[np.random.Generator] = None) -> float:
    """Compute q_α(Y) under the DGP's marginal — used as the cut-point for
    the `tail` perturbation in the MC runner."""
    rng = rng or np.random.default_rng(0)
    y = _sample_y(dgp, n, rng)
    return float(np.quantile(y, alpha))
