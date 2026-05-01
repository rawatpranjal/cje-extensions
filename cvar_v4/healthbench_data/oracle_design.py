"""Design-aware oracle slice selection.

Given full-oracle simulation data, decide which oracle labels the *estimator*
sees. The estimator weights selected rows by 1/π_i (Horvitz–Thompson) so
HT-weighted calibration and audit are unbiased under the design.

Three designs:
  - "uniform"          — every row gets π_i = coverage. Baseline.
  - "floor_tail"       — per-policy 3-bin stratification (bottom 20% / mid 60% /
                          top 20%) on cheap_score. Mid+top get π_min; bottom
                          gets a bonus tuned to preserve total coverage.
  - "floor_tail_band"  — floor_tail PLUS an extra bonus on the cheap-S band that
                          contains the pilot-estimated q̂_α. The pilot is a
                          uniform fraction of rows used to fit a quick
                          isotonic cheap→oracle calibrator; q̂_α is the
                          α-quantile of predicted Ŷ over all rows. The band
                          is the cheap-S quintile that contains q̂_α(S).

Stratification is per-policy (each policy's cheap-S quantiles are computed
on its own rows) so each policy's slice is balanced.

Selection is Bernoulli per row using a deterministic seed; π_i is the
inclusion probability used for HT weighting (independent of realization).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression


VALID_DESIGNS = ("uniform", "floor_tail", "floor_tail_band")


@dataclass
class DesignedSlice:
    prompt_id: str
    policy: str
    cheap_score: float
    stratum: str        # "uniform" | "bottom" | "mid" | "top" | "band"
    pi: float           # inclusion probability ∈ (0, 1]
    selected: bool      # realization of Bernoulli(pi)


def _per_policy_groups(rows: list[dict]) -> dict[str, list[int]]:
    """Return {policy_name: [row_idx, ...]} preserving input order."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["policy"], []).append(i)
    return groups


def _stratify_3bin(scores: np.ndarray, bottom_q: float = 0.20,
                   top_q: float = 0.80) -> np.ndarray:
    """Return labels: 'bottom' / 'mid' / 'top' for each row."""
    if scores.size == 0:
        return np.empty(0, dtype=object)
    lo = float(np.quantile(scores, bottom_q))
    hi = float(np.quantile(scores, top_q))
    out = np.full(scores.shape, "mid", dtype=object)
    out[scores <= lo] = "bottom"
    out[scores >= hi] = "top"
    return out


def _solve_bottom_pi(coverage: float, pi_min: float,
                     bottom_share: float = 0.20) -> tuple[float, float, bool]:
    """Tune (π_bottom, π_min) so bottom_share·π_bottom + (1−bottom_share)·π_min == coverage.

    Returns (pi_bottom, pi_min_actual, auto_bumped). When the implied π_bottom
    would exceed 1.0 (budget unsatisfiable at the requested floor), pin
    π_bottom = 1.0 and raise the floor instead. This preserves the budget
    while still oversampling the bottom — the alternative (silent clipping)
    leaves realized coverage well below target, which biases the
    floor_tail-vs-uniform comparison.
    """
    pi_bottom = (coverage - (1 - bottom_share) * pi_min) / bottom_share
    if pi_bottom < pi_min:
        # Very low coverage target — collapse to uniform to avoid weird cases
        return pi_min, pi_min, False
    if pi_bottom <= 1.0:
        return pi_bottom, pi_min, False
    # Budget unsatisfiable at the requested floor: pin π_bottom=1, raise pi_min.
    pi_min_new = (coverage - bottom_share * 1.0) / (1.0 - bottom_share)
    pi_min_new = max(0.0, min(1.0, pi_min_new))
    return 1.0, pi_min_new, True


def _bernoulli(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.random(pi.shape) < pi


def _identify_band_quintile(cheap_scores: np.ndarray,
                            predicted_y: np.ndarray,
                            alpha: float) -> tuple[float, float]:
    """Return (lo, hi) cheap-S range for the quintile that contains q̂_α(Ŷ).

    The band is defined on the cheap-S axis (since π allocation lives there);
    we find which cheap-S quintile contains the rows whose predicted oracle
    Ŷ lies near the α-quantile of Ŷ.
    """
    q_hat = float(np.quantile(predicted_y, alpha))
    # Rows whose predicted Ŷ is near q_hat (use a small neighborhood)
    eps = max(0.01, (predicted_y.max() - predicted_y.min()) / 20.0)
    near = np.abs(predicted_y - q_hat) <= eps
    if near.sum() == 0:
        # Fallback: take the row with predicted Ŷ closest to q_hat
        near = np.zeros_like(predicted_y, dtype=bool)
        near[np.argmin(np.abs(predicted_y - q_hat))] = True
    s_in_band = cheap_scores[near]
    if s_in_band.size == 0:
        return float("inf"), float("-inf")
    # Find which cheap-S quintile they fall in
    q20 = float(np.quantile(cheap_scores, 0.20))
    q40 = float(np.quantile(cheap_scores, 0.40))
    q60 = float(np.quantile(cheap_scores, 0.60))
    q80 = float(np.quantile(cheap_scores, 0.80))
    edges = [-math.inf, q20, q40, q60, q80, math.inf]
    band_med = float(np.median(s_in_band))
    for k in range(5):
        if edges[k] <= band_med < edges[k + 1]:
            return edges[k], edges[k + 1]
    return edges[-2], edges[-1]


def select_slice(
    rows: list[dict],
    *,
    design: str,
    coverage: float,
    alpha: float = 0.10,
    pilot_frac: float = 0.05,
    pi_min: float = 0.10,
    bottom_quantile: float = 0.20,
    band_pi_bonus: float = 0.30,
    seed: int = 42,
) -> list[DesignedSlice]:
    """Return one DesignedSlice per input row (same order).

    rows: each dict needs at least {"prompt_id", "policy", "cheap_score"}.
          For floor_tail_band, also needs "oracle_score" (the full-oracle
          label, used by the pilot calibrator). The pilot is a uniform
          subset of rows; only those rows' oracle labels are used to fit
          the pilot calibrator (so we can simulate not knowing the rest).
    coverage: target overall fraction of rows whose label is estimator-visible.
    alpha: tail level (used only for floor_tail_band's q̂_α).
    pilot_frac: fraction of rows in the uniform pilot for floor_tail_band.
    pi_min: minimum inclusion probability everywhere (the "floor").
    bottom_quantile: defines the bottom stratum (default 20%).
    band_pi_bonus: extra π added to the band stratum on top of pi_min.
    seed: deterministic Bernoulli realization.
    """
    if design not in VALID_DESIGNS:
        raise ValueError(f"design must be one of {VALID_DESIGNS}, got {design!r}")
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    n = len(rows)
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    pi = np.full(n, np.nan)
    stratum = np.full(n, "uniform", dtype=object)

    if design == "uniform":
        pi[:] = coverage
    else:
        # Per-policy stratification
        groups = _per_policy_groups(rows)
        for pol, idxs in groups.items():
            scores = np.array([rows[i]["cheap_score"] for i in idxs])
            labels = _stratify_3bin(scores, bottom_quantile, 1 - bottom_quantile)
            for k, i in enumerate(idxs):
                stratum[i] = labels[k]

            # Solve (bottom_pi, pi_min_actual) for this policy preserving coverage
            bottom_pi, pi_min_actual, auto_bumped = _solve_bottom_pi(
                coverage, pi_min, bottom_quantile,
            )
            if auto_bumped:
                print(f"[oracle_design] policy={pol}: budget unsatisfiable at "
                      f"pi_min={pi_min:.3f}, coverage={coverage:.3f}; "
                      f"auto-bumped pi_min → {pi_min_actual:.3f} so π_bottom=1.0")
            for k, i in enumerate(idxs):
                if labels[k] == "bottom":
                    pi[i] = bottom_pi
                else:
                    pi[i] = pi_min_actual

        if design == "floor_tail_band":
            # Pilot: uniform fraction of all rows; need oracle_score
            if not all("oracle_score" in r for r in rows):
                raise ValueError(
                    "floor_tail_band requires 'oracle_score' in every row "
                    "(simulation experiment with full oracle)."
                )
            pilot_n = max(5, int(round(pilot_frac * n)))
            pilot_idx = rng.choice(n, size=min(pilot_n, n), replace=False)
            pilot_s = np.array([rows[i]["cheap_score"] for i in pilot_idx])
            pilot_y = np.array([rows[i]["oracle_score"] for i in pilot_idx])
            # Quick isotonic S → Y on pilot
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            order = np.argsort(pilot_s)
            try:
                iso.fit(pilot_s[order], pilot_y[order])
                all_s = np.array([r["cheap_score"] for r in rows])
                pred_y = iso.predict(all_s)
                lo, hi = _identify_band_quintile(all_s, pred_y, alpha)
                # Bump π for rows in the band that aren't already in bottom
                in_band = (all_s >= lo) & (all_s < hi)
                for i in range(n):
                    if in_band[i] and stratum[i] != "bottom":
                        pi[i] = min(1.0, pi[i] + band_pi_bonus)
                        stratum[i] = "band"
            except ValueError:
                # Pilot too small / degenerate — silently fall back to floor_tail
                pass

    selected = _bernoulli(pi, rng)

    out = []
    for i, r in enumerate(rows):
        out.append(DesignedSlice(
            prompt_id=r["prompt_id"],
            policy=r["policy"],
            cheap_score=float(r["cheap_score"]),
            stratum=str(stratum[i]),
            pi=float(pi[i]),
            selected=bool(selected[i]),
        ))
    return out


def slice_summary(slices: list[DesignedSlice]) -> dict:
    """Quick counts: total, selected, by stratum, by policy."""
    n = len(slices)
    n_sel = sum(1 for s in slices if s.selected)
    by_stratum: dict[str, dict] = {}
    by_policy: dict[str, dict] = {}
    for s in slices:
        bs = by_stratum.setdefault(s.stratum, {"n": 0, "n_sel": 0, "pi_mean": 0.0})
        bs["n"] += 1
        bs["n_sel"] += int(s.selected)
        bs["pi_mean"] += s.pi
        bp = by_policy.setdefault(s.policy, {"n": 0, "n_sel": 0})
        bp["n"] += 1
        bp["n_sel"] += int(s.selected)
    for v in by_stratum.values():
        v["pi_mean"] = v["pi_mean"] / max(v["n"], 1)
    return {
        "n_total": n,
        "n_selected": n_sel,
        "coverage_realized": n_sel / max(n, 1),
        "by_stratum": by_stratum,
        "by_policy": by_policy,
    }
