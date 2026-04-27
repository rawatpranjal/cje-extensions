"""Isolate the two audit failure modes.

Paper: §A.2 each moment catches its own failure.
Output: stdout only — §A.2 numbers are currently hardcoded in
cvar_v3/regenerate_macros.py until this script is updated to write a CSV.

The audit is a 2-moment test with
  g_1 = 1{Y <= t_hat} - alpha     (does t_hat hit the alpha-quantile?)
  g_2 = (t_hat - Y)_+ - g_hat(S)  (does the calibrator predict the stop-loss?)

Under the well-specified null, both are zero in expectation. The questions
this script answers, empirically:

  - When ONLY g_1 fails (t_hat is shifted but the calibrator at that
    threshold is well-specified), does the audit catch it?
  - When ONLY g_2 fails (t_hat is correct but the calibrator's
    prediction is biased), does the audit catch it?

Mode A (g_1 only): take the well-specified DGP (base->base, delta=0),
compute t_hat via the dual, then *evaluate the audit at t_hat + Delta*.
The calibrator at the shifted threshold is still well-specified
(transport holds), so g_2 has zero mean. But Pr_target(Y <= t_hat + Delta)
!= alpha, so g_1 has nonzero mean.

Mode B (g_2 only): take the same DGP, compute t_hat, and evaluate the
audit at the correct t_hat. But shift every calibrator prediction by a
constant c (a bias in the calibrator's predicted stop-loss). Because
the dual's argmax is invariant to a constant offset on g_t, t_hat is
unaffected: g_1 has zero mean. But ĝ_hat is biased by c, so g_2 has
nonzero mean equal to -c.

If the audit catches each mode independently, both empirical reject
rates rise above the null size while keeping the other moment inactive.
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
    fit_isotonic_tail_loss,
)

DATA = Path.home() / "Dropbox" / "cvar-cje-data" / "cje-arena-experiments" / "data"


def audit_isolated(
    s_train, y_train, s_audit, y_audit, t_hat, alpha,
    B=80, fold_seed=0, g_offset=0.0,
):
    """Same paired-bootstrap audit as `two_moment_wald_audit_xf` but with a
    constant shift `g_offset` added to every calibrator prediction (both in
    the original moments and inside the bootstrap). Used to inject a g_2-only
    failure (set g_offset > 0) while leaving g_1 alone."""
    n_train = len(s_train)
    n_audit = len(s_audit)
    grid_size = 61

    z_train_full = np.maximum(t_hat - y_train, 0.0)
    pred_audit_full = fit_isotonic_tail_loss(s_train, z_train_full, s_audit) + g_offset
    g1_full = (y_audit <= t_hat).astype(float) - alpha
    g2_full = np.maximum(t_hat - y_audit, 0.0) - pred_audit_full
    gbar = np.array([g1_full.mean(), g2_full.mean()])

    rng = np.random.default_rng(fold_seed)
    g_per_boot = np.empty((B, 2))
    for b in range(B):
        idx_t = rng.integers(0, n_train, size=n_train)
        idx_a = rng.integers(0, n_audit, size=n_audit)
        _, t_b, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx_t], y_train[idx_t], s_audit[idx_a], alpha, grid_size,
        )
        z_b = np.maximum(t_b - y_train[idx_t], 0.0)
        pred_b = fit_isotonic_tail_loss(s_train[idx_t], z_b, s_audit[idx_a]) + g_offset
        g1_b = ((y_audit[idx_a] <= t_b).astype(float) - alpha).mean()
        g2_b = (np.maximum(t_b - y_audit[idx_a], 0.0) - pred_b).mean()
        g_per_boot[b] = [g1_b, g2_b]

    Sigma = np.cov(g_per_boot, rowvar=False, ddof=1)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
    wald = float(gbar @ Sigma_inv @ gbar)
    p = float(1.0 - stats.chi2.cdf(wald, df=2))
    return wald, p, gbar


_DGPS = None


def _one_rep(args):
    label, t_shift, g_offset, rep = args
    rng = np.random.default_rng(8000 + rep)
    s_c, y_c = sample_synthetic(_DGPS["base"], 625, rng)
    s_e, y_e = sample_synthetic(_DGPS["base"], 2000, rng)
    _, t_hat, _, _ = estimate_direct_cvar_isotonic(s_c, y_c, s_e, 0.10, 61)
    t_use = t_hat + t_shift
    wald, p, gbar = audit_isolated(
        s_c, y_c, s_e, y_e, t_use, 0.10,
        B=80, fold_seed=rep, g_offset=g_offset,
    )
    return label, p, gbar[0], gbar[1]


def main():
    global _DGPS
    print("Fitting Arena DGP...")
    _DGPS = fit_arena_dgp(DATA)

    cells = [
        # (cell_id, label, t_shift, g_offset)
        ("baseline",  "baseline: both hold",                0.00,  0.000),
        ("modeA_t02", "g1 only: shift t-hat by +0.02",      0.02,  0.000),
        ("modeA_t05", "g1 only: shift t-hat by +0.05",      0.05,  0.000),
        ("modeA_t10", "g1 only: shift t-hat by +0.10",      0.10,  0.000),
        ("modeB_g005", "g2 only: shift g-hat by +0.005",    0.00,  0.005),
        ("modeB_g010", "g2 only: shift g-hat by +0.010",    0.00,  0.010),
        ("modeB_g020", "g2 only: shift g-hat by +0.020",    0.00,  0.020),
    ]
    cells_for_pool = [(label, ts, go) for _, label, ts, go in cells]
    label_to_id = {label: cid for cid, label, _, _ in cells}
    n_reps = 200

    tasks = []
    for label, ts, go in cells_for_pool:
        for r in range(n_reps):
            tasks.append((label, ts, go, r))

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(8, mp.cpu_count())) as pool:
        results = list(pool.imap_unordered(_one_rep, tasks, chunksize=8))
    print(f"Done in {time.time()-t0:.1f}s")

    by_label = {}
    for label, p, g1, g2 in results:
        by_label.setdefault(label, {"p": [], "g1": [], "g2": []})
        by_label[label]["p"].append(p)
        by_label[label]["g1"].append(g1)
        by_label[label]["g2"].append(g2)

    print()
    print(f"{'Cell':<40} {'reject':>8} {'mean(g1)':>10} {'mean(g2)':>10}")
    print("-" * 70)
    out_rows = []
    for cid, label, ts, go in cells:
        d = by_label[label]
        ps = np.array(d["p"])
        rej = float(np.mean(ps < 0.05))
        mg1 = float(np.mean(d["g1"]))
        mg2 = float(np.mean(d["g2"]))
        print(f"{label:<40} {rej:>8.3f} {mg1:>10.4f} {mg2:>10.4f}")
        out_rows.append({
            "cell_id": cid, "label": label, "t_shift": ts, "g_offset": go,
            "n_reps": len(ps), "reject_rate": rej, "mean_g1": mg1, "mean_g2": mg2,
        })

    out = Path("cvar_v3/results_failure_modes.csv")
    with out.open("w") as f:
        f.write("cell_id,label,t_shift,g_offset,n_reps,reject_rate,mean_g1,mean_g2\n")
        for r in out_rows:
            f.write(f"{r['cell_id']},{r['label']},{r['t_shift']},{r['g_offset']},"
                    f"{r['n_reps']},{r['reject_rate']:.6f},{r['mean_g1']:.6f},{r['mean_g2']:.6f}\n")
    print(f"\nWrote {len(out_rows)} cells to {out}")


if __name__ == "__main__":
    main()
