"""Audit failure drill-down: binomial p-values for flagged cells.

Under transport, the count of audit rows with Y ≤ t̂ is Binomial(n, alpha).
We compare each cell's observed count against this reference and report the
two-sided binomial p-value.

Why a binomial p-value (and not just the |g₁| heuristic):
    The verdict heuristic |ḡ₁| ≤ 0.05 has per-row sensitivity of
    1 / n_audit. At n_audit = 26 (held-out logger slice), a single row
    crossing t̂ moves |ḡ₁| past 0.05 and triggers FLAG_TAIL even when
    the underlying tail-mass is exactly α. The binomial p-value tells
    us whether the observed count is statistically incompatible with
    Binom(n, α), filtering "real" transport failures from
    small-sample heuristic artifacts. Cells with FLAG and p < 0.05
    are likely structural transport failures; FLAG with p ≥ 0.05 is
    likely a sampling-noise flag. The full audit (`two_moment_wald_audit_xf`)
    does this more rigorously via paired bootstrap; this binomial check
    is a fast triage that uses only the audit-slice data.

Outputs:
    writeup/data/audit_drilldown.json
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.stats import binom

from ..analyze import step5_oracle_calibrated_uniform
from ._common import panel_size, write_json


def _two_sided_binom_p(n: int, k_obs: int, p: float) -> float:
    """Two-sided P(|K - n*p| >= |k_obs - n*p|) under Binom(n, p)."""
    expected = n * p
    abs_dev = abs(k_obs - expected)
    lower_bound = expected - abs_dev
    upper_bound = expected + abs_dev
    p_lo = binom.cdf(int(math.floor(lower_bound)), n, p) if lower_bound >= 0 else 0.0
    p_hi = (1.0 - binom.cdf(int(math.ceil(upper_bound)) - 1, n, p)
             if upper_bound <= n else 0.0)
    inner = min(p_lo, p_hi)
    return float(min(2.0 * inner if inner > 0 else max(p_lo, p_hi), 1.0))


def compute(alphas=(0.10, 0.20), coverage: float = 0.25, seed: int = 42) -> dict:
    out_rows = []
    for a in alphas:
        rows = step5_oracle_calibrated_uniform(
            coverage=coverage, alpha=a, seed=seed, verbose=False,
        )
        for r in rows:
            n = r["n_slice"]
            alpha = r["alpha"]
            g1 = r["mean_g1"]
            expected = alpha * n
            obs_count = int(round((g1 + alpha) * n))
            swing = 1.0 / n
            p_value = _two_sided_binom_p(n, obs_count, alpha)
            out_rows.append({
                "policy": r["policy"], "alpha": alpha, "n": n,
                "expected_count": expected,
                "observed_count": obs_count,
                "one_row_swing": swing,
                "binomial_p": p_value,
                "verdict": r["verdict"],
                "abs_error": r["abs_error"],
                "mean_g1": g1,
                "mean_g2": r["mean_g2"],
                "t_hat": r["t_hat"],
            })
    return {
        "alphas": list(alphas),
        "coverage": coverage,
        "seed": seed,
        "panel_size": panel_size(),
        "rows": out_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = compute(coverage=args.coverage, seed=args.seed)
    path = write_json("audit_drilldown.json", payload)
    print(f"[audit_drilldown] {'policy':28} {'α':>5} {'n':>4} {'exp':>6} {'obs':>4} "
          f"{'binom_p':>9}  {'verdict':>14}")
    for r in sorted(payload["rows"], key=lambda x: (x["alpha"], x["policy"])):
        marker = " *" if r["binomial_p"] < 0.05 and r["verdict"] != "PASS" else ""
        print(f"[audit_drilldown] {r['policy']:28} {r['alpha']:>5.2f} {r['n']:>4} "
              f"{r['expected_count']:>6.2f} {r['observed_count']:>4} "
              f"{r['binomial_p']:>9.3f}  {r['verdict']:>14}{marker}")
    print(f"[audit_drilldown] wrote {path}")
    print(f"[audit_drilldown] (* = flag with binomial p<0.05; statistically real)")


if __name__ == "__main__":
    main()
