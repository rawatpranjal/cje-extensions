"""
Code tests for `mc.dgp`.

These cover algebraic identities (α=1 collapse), determinism, and column
contracts. The explicitly named `test_statistical_*` tests below run small
fixed-seed Monte Carlo checks that the closed-form DGP truth matches samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from .dgp import DEFAULT_POLICIES, DGP


@pytest.fixture
def dgp() -> DGP:
    return DGP(DEFAULT_POLICIES)


@pytest.mark.parametrize("policy", list(DEFAULT_POLICIES.keys()))
def test_truth_cvar_alpha_one_equals_mean(dgp: DGP, policy: str) -> None:
    """At α=1, CVaR collapses to mean — must hold exactly in the truth."""
    np.testing.assert_allclose(
        dgp.truth_cvar(policy, alpha=1.0),
        dgp.truth_mean(policy),
        atol=0.0,
        rtol=0.0,
    )


def test_sample_determinism(dgp: DGP) -> None:
    """Same seed → same draws, byte for byte."""
    df1 = dgp.sample("uniform", n=200, with_oracle=True, seed=42)
    df2 = dgp.sample("uniform", n=200, with_oracle=True, seed=42)
    assert df1.equals(df2)


def test_sample_seed_independence(dgp: DGP) -> None:
    """Different seeds → different draws."""
    df1 = dgp.sample("uniform", n=200, with_oracle=True, seed=0)
    df2 = dgp.sample("uniform", n=200, with_oracle=True, seed=1)
    assert not np.allclose(df1["y"].to_numpy(), df2["y"].to_numpy())


def test_eval_slice_has_no_y(dgp: DGP) -> None:
    df = dgp.sample("uniform", n=10, with_oracle=False, seed=0)
    assert "y" not in df.columns
    assert "s" in df.columns
    assert "prompt_id" in df.columns


def test_delta_shifts_distribution(dgp: DGP) -> None:
    """delta>0 (b → b+delta) should shift the Y mean DOWN."""
    n = 50_000
    df0 = dgp.sample("uniform", n=n, with_oracle=True, seed=0, delta=0.0)
    df1 = dgp.sample("uniform", n=n, with_oracle=True, seed=0, delta=2.0)
    assert df1["y"].mean() < df0["y"].mean() - 0.05


_TRUTH_CHECK_SEEDS = tuple(range(10))
_TRUTH_CHECK_N_PER_SEED = 30_000


def _pool_y(dgp: DGP, policy: str) -> np.ndarray:
    """Draw and concatenate Y across 10 fixed seeds (300k pooled samples)."""
    return np.concatenate([
        dgp.sample(policy, n=_TRUTH_CHECK_N_PER_SEED,
                   with_oracle=True, seed=s)["y"].to_numpy()
        for s in _TRUTH_CHECK_SEEDS
    ])


@pytest.mark.parametrize("policy", list(DEFAULT_POLICIES.keys()))
def test_statistical_truth_mean_matches_samples(dgp: DGP, policy: str) -> None:
    """
    Pool 10 seeds × 30k samples = 300k. Compare to analytical truth_mean.

    Pooled SE is √10 ≈ 3.2× tighter than single-seed, so this test is more
    sensitive to a buggy formula while passing comfortably when correct.
    Multi-seed eliminates seed-cherry-pick risk.
    """
    y = _pool_y(dgp, policy)
    n = len(y)
    empirical = float(y.mean())
    truth = dgp.truth_mean(policy)
    se = float(y.std(ddof=1) / np.sqrt(n))
    z = abs(empirical - truth) / se

    assert z <= 4.0, (
        f"{policy}: emp_mean={empirical:.6f}, truth={truth:.6f}, "
        f"|Δ|={abs(empirical - truth):.2e}, SE={se:.2e}, z={z:.2f} > 4σ "
        f"(pooled {len(_TRUTH_CHECK_SEEDS)} seeds × {_TRUTH_CHECK_N_PER_SEED})"
    )


@pytest.mark.parametrize("policy", list(DEFAULT_POLICIES.keys()))
def test_statistical_truth_cvar_matches_samples(dgp: DGP, policy: str) -> None:
    """
    Pool 10 seeds × 30k = 300k. Compare empirical lower-α tail mean to
    analytical truth_cvar. SE comes from a 200-rep bootstrap of the empirical
    CVaR functional, which captures BOTH the within-tail variance AND the
    boundary-noise (which Y values fall in the tail varies).

    Bootstrap SE is the right SE for a non-linear sample functional like
    `mean of bottom-k order statistics`. The plug-in `tail.std / √k` under-
    counts SE for U-shaped distributions (Beta(0.5, 0.5)), which is why
    single-seed `tail_heavy` ran hot on some seeds with the old test.
    """
    alpha = 0.10
    y = _pool_y(dgp, policy)
    n_total = len(y)
    k = int(np.floor(alpha * n_total))

    def _emp_cvar(samples: np.ndarray) -> float:
        # Use partition (O(n)) — only need the k smallest values, not their order.
        return float(np.partition(samples, k)[:k].mean())

    truth = dgp.truth_cvar(policy, alpha)
    empirical = _emp_cvar(y)

    B = 200
    rng = np.random.default_rng(42)
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_total, size=n_total)
        boot[b] = _emp_cvar(y[idx])
    se = float(boot.std(ddof=1))
    z = abs(empirical - truth) / se

    assert z <= 4.0, (
        f"{policy}: emp_cvar={empirical:.6f}, truth={truth:.6f}, "
        f"|Δ|={abs(empirical - truth):.2e}, bootstrap_SE={se:.2e}, z={z:.2f} > 4σ "
        f"(pooled {len(_TRUTH_CHECK_SEEDS)} seeds × {_TRUTH_CHECK_N_PER_SEED}, "
        f"B={B})"
    )
