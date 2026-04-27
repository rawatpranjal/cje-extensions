"""Targeted MC: validate the xf-audit's W statistic actually matches χ²_2.

Paper: §A.1 Wald-null distribution.
Output: cvar/audit_chi2_W_values.csv (consumed by cvar/regenerate_macros.py).

Focused null-only experiment. For each rep:
  1. Sample (s_calib, y_calib, s_eval, y_eval) from base→base, δ=0 DGP.
  2. Compute t̂ via the dual.
  3. Compute W with naive Σ̂ and with xf (paired-bootstrap, t̂-re-max) Σ̂.
  4. Record both W values.

Then compare empirical W distribution to χ²_2:
  - mean (target 2.0), median (target 1.386), variance (target 4.0)
  - quantiles at {50, 75, 90, 95, 99} %
  - Kolmogorov-Smirnov stat on the empirical CDF vs χ²_2 CDF
  - rejection rate at α ∈ {0.10, 0.05, 0.01}

Run at two sample sizes to check scaling: n_eval ∈ {1000, 2500}.

Wall-clock target: ≈5 min total at n_reps=300 per cell with multiprocessing.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, "cvar")
from dgp import fit_arena_dgp, sample_synthetic
from workhorse import (
    estimate_direct_cvar_isotonic,
    two_moment_wald_audit,
    two_moment_wald_audit_xf,
)

DATA = Path.home() / "Dropbox" / "cvar-cje-data" / "cje-arena-experiments" / "data"

# Two sample sizes; same alpha; lots of reps for tight percentile estimates.
CELLS = [
    {"n_oracle": 300, "n_eval": 1000, "alpha": 0.10, "n_reps": 300, "B_xf": 80},
    {"n_oracle": 625, "n_eval": 2500, "alpha": 0.10, "n_reps": 300, "B_xf": 80},
]

_DGPS: dict | None = None


def _one_rep(args):
    cell, rep = args
    rng = np.random.default_rng(10_000 + rep + 1_000_000 * cell["n_eval"])
    s_calib, y_calib = sample_synthetic(_DGPS["base"], cell["n_oracle"], rng)
    s_eval, y_eval = sample_synthetic(_DGPS["base"], cell["n_eval"], rng)
    _, t_hat, _, _ = estimate_direct_cvar_isotonic(
        s_calib, y_calib, s_eval, cell["alpha"], 61,
    )
    a_naive = two_moment_wald_audit(s_calib, y_calib, s_eval, y_eval, t_hat, cell["alpha"])
    a_xf = two_moment_wald_audit_xf(
        s_calib, y_calib, s_eval, y_eval, t_hat, cell["alpha"],
        B=cell["B_xf"], fold_seed=rep,
    )
    return cell["n_eval"], a_naive["wald_stat"], a_xf["wald_stat"]


def main():
    print("Fitting Arena DGP...")
    global _DGPS
    _DGPS = fit_arena_dgp(DATA)

    tasks = []
    for cell in CELLS:
        for rep in range(cell["n_reps"]):
            tasks.append((cell, rep))
    print(f"Total tasks: {len(tasks)}")

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(8, mp.cpu_count())) as pool:
        results = list(pool.imap_unordered(_one_rep, tasks, chunksize=8))
    print(f"Done in {time.time()-t0:.1f}s")

    # Aggregate per cell
    by_n = {}
    for n, w_n, w_x in results:
        by_n.setdefault(n, {"naive": [], "xf": []})
        by_n[n]["naive"].append(w_n)
        by_n[n]["xf"].append(w_x)

    print()
    print("=" * 80)
    print("χ²_2 reference values:")
    print(f"  mean = 2.000   median = 1.386   var = 4.000   95-pctl = 5.991   99-pctl = 9.210")
    print("=" * 80)

    chi2_2 = stats.chi2(df=2)
    for n_eval in sorted(by_n.keys()):
        for label in ["naive", "xf"]:
            W = np.array(by_n[n_eval][label])
            n = len(W)
            mean = float(np.mean(W))
            med = float(np.median(W))
            var = float(np.var(W, ddof=1))
            pcts = [50, 75, 90, 95, 99]
            qs = [float(np.percentile(W, p)) for p in pcts]
            ks_stat, ks_p = stats.kstest(W, chi2_2.cdf)
            rej_10 = float(np.mean(W > chi2_2.ppf(0.90)))
            rej_05 = float(np.mean(W > chi2_2.ppf(0.95)))
            rej_01 = float(np.mean(W > chi2_2.ppf(0.99)))
            label_str = f"n_eval={n_eval}, {label:<5}"
            print()
            print(f"--- {label_str} (n_reps={n}) ---")
            print(f"  mean   = {mean:7.3f}   target  2.000")
            print(f"  median = {med:7.3f}   target  1.386")
            print(f"  var    = {var:7.3f}   target  4.000")
            print(f"  pcts   {pcts}")
            print(f"  obs    {[round(q,2) for q in qs]}")
            print(f"  χ²_2   {[round(float(chi2_2.ppf(p/100)),2) for p in pcts]}")
            print(f"  KS     stat={ks_stat:.4f}, p={ks_p:.3g}  (small KS → close to χ²_2)")
            print(f"  reject 0.10/0.05/0.01 = {rej_10:.3f}/{rej_05:.3f}/{rej_01:.3f}")
            print(f"  target              = 0.100/0.050/0.010")

    # Save raw W values for QQ plot if needed
    out = Path("cvar/audit_chi2_W_values.csv")
    with out.open("w") as f:
        f.write("n_eval,audit,W\n")
        for n_eval, vals in by_n.items():
            for label, ws in vals.items():
                for w in ws:
                    f.write(f"{n_eval},{label},{w}\n")
    print(f"\nWrote raw W values to {out}")


if __name__ == "__main__":
    main()
