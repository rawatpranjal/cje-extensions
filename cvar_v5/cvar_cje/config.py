"""
Pipeline configuration.

All defaults match `cvar_v4/sections/method.tex` and `setup.tex`:
    α ∈ {0.10, 0.20}             (setup.tex:40-41)
    K = 5 folds                   (Mean-CJE convention; appendix_impl.tex)
    audit B = 2000                (setup.tex:41)
    grid T size = 61, tail_dense  (v4 convention 61 points; tail-dense
                                   spacing eliminates small-α discretization
                                   bias — see calibrator.py::make_t_grid)

Deferred-feature flags are False by default. Setting one to True without
implementing the feature triggers a NotImplementedError that points to the
matching anchor in TODO.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .calibrator import GridKind


OmegaEstimator = Literal[
    "analytical",
    "analytical_oua",
    "boot_remax_ridge",
    "boot_remax_no_ridge",
    "boot_fixed",
]


@dataclass(frozen=True)
class Config:
    # Estimand
    alpha: float = 0.10

    # Cross-fit / partition
    K: int = 5
    # Audit slice = bucket K of (K+1) hash buckets; CALIB = buckets 0..K-1.
    # Resulting AUDIT fraction is 1/(K+1) ≈ 0.167 at K=5.

    # Grid for t̂_α search
    grid_size: int = 61
    grid_lo: float = 0.0
    grid_hi: float = 1.0
    grid_kind: GridKind = "tail_dense"

    # Audit
    audit_B: int = 2000
    omega_estimator: OmegaEstimator = "boot_remax_ridge"
    audit_eta: float = 0.05  # χ² level for REFUSE gate

    # Reproducibility
    seed: int = 0

    # Deferred features (raise NotImplementedError if True)
    bootstrap_ci: bool = False
    jackknife_var_cal: bool = False
    plugin_variants: bool = False
    two_stage_calibrator: bool = False

    def __post_init__(self) -> None:
        # Validate immediately so misconfiguration fails before any work.
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha}")
        if self.K < 2:
            raise ValueError(f"K must be ≥ 2; got {self.K}")
        if self.grid_size < 2:
            raise ValueError(f"grid_size must be ≥ 2; got {self.grid_size}")
        if not (self.grid_lo < self.grid_hi):
            raise ValueError(
                f"grid_lo must be < grid_hi; got {self.grid_lo}, {self.grid_hi}"
            )
        if self.audit_B < 1:
            raise ValueError(f"audit_B must be ≥ 1; got {self.audit_B}")
        if not (0.0 < self.audit_eta < 1.0):
            raise ValueError(f"audit_eta must be in (0, 1); got {self.audit_eta}")

        if self.bootstrap_ci:
            raise NotImplementedError(
                "[ci-bootstrap] full-pipeline bootstrap CI is deferred; see TODO.md"
            )
        if self.jackknife_var_cal:
            raise NotImplementedError(
                "[var-cal-jackknife] Var_cal jackknife is deferred; see TODO.md"
            )
        if self.plugin_variants:
            raise NotImplementedError(
                "[plugin-quantile] / [plugin-ru-dual] plugin variants are deferred; see TODO.md"
            )
        if self.two_stage_calibrator:
            raise NotImplementedError(
                "[two-stage-calibrator] two-stage calibration is deferred; see TODO.md"
            )
