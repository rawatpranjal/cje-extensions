"""Semi-synthetic DGP fit to the HealthBench response data.

Per-policy fully-parametric DGP for use as a known-truth ground in the
MC validation. Adapted from cvar_v3/dgp.py with the following changes:

  - Loads from cvar_v4/healthbench_data/data/responses/{policy}_responses.jsonl
    (each row has metadata.judge_score and metadata.oracle_label).
  - Supports the 6 v4 policies (base, clone, premium, parallel_universe_prompt,
    unhelpful, risky) instead of v3's 5 Arena policies.
  - Adds mean_truth() for validating Direct Mean-CJE alongside cvar_truth().
  - cvar_truth uses the atom-split formula from
    cvar_v4/healthbench_data/analyze.py:cvar_alpha (HealthBench has ties at
    Y=0 so the boundary mass must split exactly).

Design (unchanged from cvar_v3 §11):
  - Y marginal: empirical CDF over the policy's observed oracle labels. For
    `unhelpful`, mixture P(Y=0)=p_zero × empirical Y>0.
  - S | Y: monotone isotonic m_p(Y) + heteroscedastic Gaussian noise binned
    by Y-quartile (4 bins). Output is clipped to [0, 1].
  - Mis-spec knob: m_override + (delta, perturbation ∈ {none, uniform, tail}).
    delta=0 with m_override=base.m_iso is the clean null.

Cross-policy joint structure NOT preserved (each policy independent). The
v4 estimator path is per-policy (calibrate on logger=base, eval on target),
so the per-policy marginals are exactly what the estimator consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression


POLICIES: tuple[str, ...] = (
    "base", "clone", "premium", "parallel_universe_prompt", "unhelpful", "risky",
)

DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[3] / "healthbench_data" / "data"
)


@dataclass
class PolicyDGP:
    policy: str
    y_grid: np.ndarray             # sorted unique Y values observed for this policy
    y_probs: np.ndarray            # empirical pmf, sums to 1
    m_iso: IsotonicRegression      # E[S | Y] fitted on real (Y, S)
    sigma_by_quartile: np.ndarray  # 4 floats: σ of S - m(Y) per Y-quartile bin
    y_quartile_edges: np.ndarray   # 3 cut points (P25, P50, P75) for bin indexing
    p_zero: Optional[float] = None
    y_grid_pos: Optional[np.ndarray] = None
    y_probs_pos: Optional[np.ndarray] = None
    mean_s_at_zero: Optional[float] = None
    sigma_s_at_zero: Optional[float] = None


# ---------------------------------------------------------------------------
# Truth helpers — atom-split CVaR (matches cvar_v4/healthbench_data/analyze.py)
# ---------------------------------------------------------------------------

def _atom_split_cvar(arr: np.ndarray, alpha: float) -> float:
    """Atom-split CVaR_α: averages exactly α·n units of mass.

    Copy of cvar_v4/healthbench_data/analyze.py:cvar_alpha so this module is
    importable without the healthbench_data package being on sys.path.
    """
    if arr.size == 0:
        return float("nan")
    n = arr.size
    sorted_y = np.sort(arr)
    target_mass = alpha * n
    k = int(np.floor(target_mass))
    remainder = target_mass - k
    if k == 0:
        return float(sorted_y[0])
    if k >= n:
        return float(sorted_y.mean())
    head_sum = float(sorted_y[:k].sum())
    if remainder > 0:
        head_sum += remainder * float(sorted_y[k])
    return head_sum / target_mass


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_policy_pairs(data_root: Path, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (S, Y) pairs from the policy's response file. Drops rows where
    either judge_score or oracle_label is null."""
    path = data_root / "responses" / f"{policy}_responses.jsonl"
    df = pl.read_ndjson(path)
    df = df.with_columns([
        pl.col("metadata").struct.field("judge_score").cast(pl.Float64).alias("s"),
        pl.col("metadata").struct.field("oracle_label").cast(pl.Float64).alias("y"),
    ]).select(["s", "y"]).drop_nulls()
    return df["s"].to_numpy(), df["y"].to_numpy()


def _fit_one_policy(policy: str, s: np.ndarray, y: np.ndarray) -> PolicyDGP:
    """Fit Y marginal + m(Y) + heteroscedastic σ by Y quartile."""
    y_grid, counts = np.unique(y, return_counts=True)
    y_probs = counts / counts.sum()

    m_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    m_iso.fit(y, s)

    edges = np.quantile(y, [0.25, 0.50, 0.75])
    bins = np.searchsorted(edges, y, side="right").clip(0, 3)
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

    # `unhelpful` is a Y=0 mixture in HealthBench (most rubric criteria fail).
    # Detect the mixture for any policy that has nontrivial Y=0 mass.
    zero_mask = y == 0.0
    if zero_mask.mean() >= 0.05:
        dgp.p_zero = float(zero_mask.mean())
        dgp.mean_s_at_zero = float(s[zero_mask].mean())
        dgp.sigma_s_at_zero = (
            float(np.std(s[zero_mask], ddof=1)) if zero_mask.sum() > 1 else 0.025
        )
        y_pos = y[~zero_mask]
        if y_pos.size > 0:
            yg, cn = np.unique(y_pos, return_counts=True)
            dgp.y_grid_pos = yg
            dgp.y_probs_pos = cn / cn.sum()
    return dgp


def fit_healthbench_dgp(data_root: Path | str | None = None) -> dict[str, PolicyDGP]:
    """Fit a PolicyDGP for each available HealthBench policy.

    data_root: directory containing responses/{policy}_responses.jsonl.
        Defaults to cvar_v4/healthbench_data/data/.

    Skips policies whose response file is missing. The MC runner asserts the
    requested policies are present.
    """
    data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    out: dict[str, PolicyDGP] = {}
    for p in POLICIES:
        path = data_root / "responses" / f"{p}_responses.jsonl"
        if not path.exists():
            continue
        s, y = _load_policy_pairs(data_root, p)
        if s.size < 20:
            continue
        out[p] = _fit_one_policy(p, s, y)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_y(dgp: PolicyDGP, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Y from the policy's empirical marginal (with mixture if any)."""
    if (dgp.p_zero is not None and dgp.y_grid_pos is not None
            and dgp.y_probs_pos is not None):
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
    if delta == 0.0 or perturbation == "none":
        return m_y
    if perturbation == "uniform":
        return m_y + delta
    if perturbation == "tail":
        if q_low_threshold is None:
            raise ValueError("perturbation='tail' requires q_low_threshold")
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

    Y marginal and noise σ from `dgp`. Conditional mean from `dgp.m_iso`
    by default, or `m_override.m_iso` (use base's m for the well-specified
    null on a target policy: target's S follows base's m so the calibrator
    transports perfectly at delta=0).

    For Y=0 mixture mass: S draws from N(mean_s_at_zero, sigma_s_at_zero)
    regardless of m_override (base never observes Y=0 outside `unhelpful`,
    so m_base(0) is just the clipped extrapolation).
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


# ---------------------------------------------------------------------------
# Population truths (delta-invariant — Y marginal does not move with delta)
# ---------------------------------------------------------------------------

def cvar_truth(
    dgp: PolicyDGP,
    alpha: float,
    n_truth: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Population CVaR_α(Y) under the DGP's Y marginal (atom-split)."""
    rng = rng or np.random.default_rng(0)
    y = _sample_y(dgp, n_truth, rng)
    return _atom_split_cvar(y, alpha)


def mean_truth(
    dgp: PolicyDGP,
    n_truth: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Population E[Y] under the DGP's Y marginal."""
    rng = rng or np.random.default_rng(0)
    y = _sample_y(dgp, n_truth, rng)
    return float(y.mean())


def q_lower_tail_threshold(
    dgp: PolicyDGP, alpha: float, n: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """q_α(Y) under the DGP's marginal — the cut-point for `tail` perturbation."""
    rng = rng or np.random.default_rng(0)
    y = _sample_y(dgp, n, rng)
    return float(np.quantile(y, alpha))
