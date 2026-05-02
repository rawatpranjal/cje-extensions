"""
End-to-end tests for cvar_cje.pipeline.

These exercise the public contract of `run_pipeline`:
    1. Given valid (oracle, eval, cfg) it returns (EstimateResult, AuditVerdict)
       and the α=1 collapse identity holds end-to-end.
    2. Invalid input raises a clear ValueError before any pipeline work begins.

The internal "set(CALIB.prompt_id) ∩ set(AUDIT.prompt_id) = ∅" assertion is
defensive — testable only by mocking `partition_oracle`. We prefer the public
contract (input validation + end-to-end correctness) and trust _crossfit's own
unit tests to cover the partition.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.isotonic import IsotonicRegression

from ..mc.dgp import DEFAULT_POLICIES, DGP
from .config import Config
from .pipeline import run_pipeline


# ---------- fixtures -----------------------------------------------------------


def _toy_data(
    n_oracle: int = 200,
    n_eval: int = 300,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """
    Synthetic monotone DGP with Y in [0, 1].

    Returns (oracle_df, eval_df, s_calib_for_reference, y_calib_for_reference).
    The reference s/y are returned alongside so the test can build an
    independent Mean-CJE reference for the α=1 collapse assertion.
    """
    rng = np.random.default_rng(seed)

    def draw(n: int):
        s = rng.normal(size=n)
        y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
        y = np.clip(y, 0.0, 1.0)
        return s, y

    s_o, y_o = draw(n_oracle)
    s_e, _ = draw(n_eval)

    oracle = pl.DataFrame({
        "prompt_id": [f"o{i}" for i in range(n_oracle)],
        "s": s_o,
        "y": y_o,
    })
    eval_df = pl.DataFrame({
        "prompt_id": [f"e{i}" for i in range(n_eval)],
        "s": s_e,
    })
    return oracle, eval_df, s_o, y_o


def _fast_cfg(alpha: float) -> Config:
    """Config with cheap audit_B so tests stay fast."""
    return Config(alpha=alpha, audit_B=20, K=5, seed=0)


# ---------- end-to-end correctness ---------------------------------------------


def test_pipeline_smoke_returns_estimate_and_verdict() -> None:
    """
    The pipeline returns (EstimateResult, AuditVerdict) with sensible types
    and the audit produces a real verdict (not an exception).
    """
    oracle, eval_df, _, _ = _toy_data()
    estimate, verdict = run_pipeline(oracle, eval_df, _fast_cfg(alpha=0.10))

    from .calibrator import make_t_grid
    cfg = _fast_cfg(alpha=0.10)
    expected_grid = make_t_grid(cfg.grid_lo, cfg.grid_hi, cfg.grid_size, cfg.grid_kind)

    assert estimate.alpha == 0.10
    assert 0.0 <= estimate.value <= 1.0
    assert estimate.threshold in tuple(expected_grid)

    assert verdict.decision in ("PASS", "REFUSE-LEVEL")
    assert verdict.W_n >= 0.0
    assert 0.0 <= verdict.p_value <= 1.0


def test_pipeline_alpha_one_collapse_end_to_end() -> None:
    """
    Run the full pipeline at α=1.0 and verify the saddle-point estimate equals
    an INDEPENDENT Mean-CJE reference (separate IsotonicRegression(increasing=
    True) fit on the same oracle rows that pipeline used as CALIB) to ≤ 1e-9.

    This is the project's load-bearing structural property — broken
    calibrator math, broken estimator math, or broken cross-fit pooling would
    all surface here. (RED-verified by mutation: replacing
    `psi = t_grid - H_mean / alpha` with `t_grid + H_mean / alpha` makes
    this test fail with a difference O(1).)
    """
    oracle, eval_df, _, _ = _toy_data(n_oracle=300, n_eval=400)
    estimate, _ = run_pipeline(oracle, eval_df, _fast_cfg(alpha=1.0))

    # Reconstruct CALIB by replicating pipeline's hash partition.
    from ._crossfit import partition_oracle
    calib_slice, _, _ = partition_oracle(oracle, K=5, seed=0)

    f_inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        calib_slice.s(), calib_slice.y()
    )
    mean_ref = float(np.mean(f_inc.predict(eval_df["s"].to_numpy())))

    assert abs(estimate.value - mean_ref) <= 1e-9, (
        f"α=1 collapse failed end-to-end: pipeline={estimate.value!r}, "
        f"reference={mean_ref!r}, |Δ|={abs(estimate.value - mean_ref):.3e}"
    )


# ---------- public-contract input validation -----------------------------------


def test_pipeline_rejects_y_out_of_range_high() -> None:
    """Y > 1 in oracle slice must raise ValueError before any work starts."""
    oracle, eval_df, _, _ = _toy_data(n_oracle=50, n_eval=50)
    oracle = oracle.with_columns(
        pl.when(pl.col("prompt_id") == "o0").then(1.5).otherwise(pl.col("y")).alias("y")
    )
    with pytest.raises(ValueError, match=r"y must be in \[0, 1\]"):
        run_pipeline(oracle, eval_df, _fast_cfg(alpha=0.10))


def test_pipeline_rejects_y_out_of_range_low() -> None:
    """Y < 0 in oracle slice must raise ValueError."""
    oracle, eval_df, _, _ = _toy_data(n_oracle=50, n_eval=50)
    oracle = oracle.with_columns(
        pl.when(pl.col("prompt_id") == "o0").then(-0.1).otherwise(pl.col("y")).alias("y")
    )
    with pytest.raises(ValueError, match=r"y must be in \[0, 1\]"):
        run_pipeline(oracle, eval_df, _fast_cfg(alpha=0.10))


def test_pipeline_rejects_duplicate_prompt_id() -> None:
    """Each oracle prompt_id must be unique (one row per prompt)."""
    oracle, eval_df, _, _ = _toy_data(n_oracle=50, n_eval=50)
    oracle = oracle.with_columns(
        pl.when(pl.col("prompt_id") == "o1").then(pl.lit("o0")).otherwise(pl.col("prompt_id")).alias("prompt_id")
    )
    with pytest.raises(ValueError, match="unique prompt_id"):
        run_pipeline(oracle, eval_df, _fast_cfg(alpha=0.10))


def test_pipeline_rejects_missing_oracle_column() -> None:
    """Oracle without `y` is rejected at validate time."""
    oracle, eval_df, _, _ = _toy_data(n_oracle=50, n_eval=50)
    oracle_no_y = oracle.drop("y")
    with pytest.raises(ValueError, match="missing columns.*'y'"):
        run_pipeline(oracle_no_y, eval_df, _fast_cfg(alpha=0.10))


def test_pipeline_rejects_missing_eval_column() -> None:
    """Eval without `s` is rejected at validate time."""
    oracle, eval_df, _, _ = _toy_data(n_oracle=50, n_eval=50)
    eval_no_s = eval_df.drop("s")
    with pytest.raises(ValueError, match="missing columns.*'s'"):
        run_pipeline(oracle, eval_no_s, _fast_cfg(alpha=0.10))
