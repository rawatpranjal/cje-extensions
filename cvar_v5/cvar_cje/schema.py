"""
Typed records exchanged between modules.

Slice / Row layout:
    Required columns of `Slice.df` (polars DataFrame):
        prompt_id : str
        s         : float    cheap-judge score
        y         : float    oracle label, present on role ∈ {calib, audit}
                             absent / null on role == eval
    Optional:
        any other columns are passed through (covariates reserved for future).

The pipeline asserts at runtime:
    set(calib.prompt_id) ∩ set(audit.prompt_id) == ∅
    Y ∈ [0, 1]   on labeled rows
    grid right-endpoint ≥ max(Y_calib)  (so α=1 collapse hits the boundary)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


SliceRole = Literal["calib", "audit", "eval"]


@dataclass(frozen=True)
class Slice:
    df: pl.DataFrame
    role: SliceRole

    def n(self) -> int:
        return self.df.height

    def prompt_ids(self) -> list[str]:
        return self.df["prompt_id"].to_list()

    def s(self) -> np.ndarray:
        return self.df["s"].to_numpy()

    def y(self) -> np.ndarray:
        # Caller is responsible for knowing y is present (calib/audit only).
        return self.df["y"].to_numpy()


@dataclass(frozen=True)
class CrossFitFolds:
    n_folds: int
    fold_id: np.ndarray  # shape (n_calib,), values in [0, n_folds)


@dataclass(frozen=True)
class EstimateResult:
    alpha: float
    value: float       # ĈVaR_α
    threshold: float   # t̂_α  (argmax over the grid)


AuditDecision = Literal["PASS", "REFUSE-LEVEL"]


@dataclass(frozen=True)
class AuditVerdict:
    W_n: float
    p_value: float
    g1: float
    g2: float
    omega_estimator: str
    decision: AuditDecision
