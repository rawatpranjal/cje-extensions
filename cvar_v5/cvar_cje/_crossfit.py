"""
Hash-based deterministic partition of the oracle slice.

Math contract:
    fold_id_i = hash(prompt_id_i, seed) mod (K + 1)

    bucket K       → AUDIT  slice  (held out from calibration)
    buckets 0..K-1 → CALIB  slice  (cross-fit folds for the calibrator)

    Determinism: same (prompt_id, seed) → same bucket across runs and machines
                 (uses hashlib.blake2b, not Python's randomized hash()).

Audit-holdout discipline (cvar_v4/sections/appendix_c.tex:158, 164-165):
    The audit slice MUST be disjoint from the rows used to train the calibrator.
    This module is the single point where that discipline is enforced.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

from .schema import CrossFitFolds, Slice


def _bucket(prompt_id: str, n_buckets: int, seed: int) -> int:
    """Stable, cross-process hash of (prompt_id, seed) into [0, n_buckets)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(prompt_id.encode("utf-8"))
    h.update(seed.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little") % n_buckets


def assign_buckets(prompt_ids: list[str], n_buckets: int, seed: int) -> np.ndarray:
    """Per-row bucket assignment in [0, n_buckets)."""
    return np.fromiter(
        (_bucket(pid, n_buckets, seed) for pid in prompt_ids),
        dtype=np.int64,
        count=len(prompt_ids),
    )


def partition_oracle(
    oracle: pl.DataFrame, K: int, seed: int
) -> tuple[Slice, Slice, CrossFitFolds]:
    """
    Hash-split oracle rows into CALIB (buckets 0..K-1) and AUDIT (bucket K).

    Returns:
        calib_slice: Slice(role="calib")    rows in buckets 0..K-1
        audit_slice: Slice(role="audit")    rows in bucket K
        folds:       CrossFitFolds          fold_id ∈ [0, K) aligned with calib_slice rows
    """
    if "prompt_id" not in oracle.columns:
        raise ValueError("oracle DataFrame must have a 'prompt_id' column")
    if "y" not in oracle.columns:
        raise ValueError("oracle DataFrame must have a 'y' column (oracle label)")

    n_buckets = K + 1
    buckets = assign_buckets(oracle["prompt_id"].to_list(), n_buckets, seed)

    calib_mask = buckets < K
    audit_mask = buckets == K

    calib_df = oracle.filter(pl.Series(calib_mask))
    audit_df = oracle.filter(pl.Series(audit_mask))

    if calib_df.height == 0:
        raise ValueError("partition_oracle produced an empty CALIB slice")
    if audit_df.height == 0:
        raise ValueError("partition_oracle produced an empty AUDIT slice")

    fold_id = buckets[calib_mask]  # already in [0, K)

    return (
        Slice(df=calib_df, role="calib"),
        Slice(df=audit_df, role="audit"),
        CrossFitFolds(n_folds=K, fold_id=fold_id),
    )
