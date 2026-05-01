"""Step-by-step analysis of the HealthBench-as-CJE-dataset pilot.

Produces the four headline numbers the user wanted to see:
  - per-policy mean (naive average of cheap-S scores, no calibration)
  - per-policy CVaR_0.05 (mean of bottom 5% cheap-S scores)
  - audit verdict (only when oracle Y is available)
  - calibrated estimates via Direct CVaR-CJE (only when oracle Y is available)

When run before oracle scoring is done, falls back gracefully to "naive cheap-S only" mode.

Usage:
    python3 -m cvar_v4.healthbench_data.analyze --kind cheap-only      # use cheap-S as Y directly
    python3 -m cvar_v4.healthbench_data.analyze --kind oracle          # full Direct CVaR-CJE + audit
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Use the self-contained estimator from earlier (no cje-eval dependency)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from eda.deeper._estimator import (
        estimate_direct_cvar_isotonic,
        two_moment_wald_audit_xf,
        AUDIT_VARIANTS,
        bootstrap_cvar_ci,
        bootstrap_mean_ci,
        jackknife_var_cal,
        mean_transport_audit,
        simple_cvar_audit,
        fit_isotonic_mean,
    )
    HAVE_CJE = True
except ImportError:
    HAVE_CJE = False

from .policies import POLICIES, logger_policy
from .judge import _judge_path
from .oracle_design import select_slice, slice_summary, VALID_DESIGNS

DATA_DIR = Path(__file__).parent / "data"


def _load_judge_scores(policy_name: str, kind: str) -> dict[str, float]:
    """Load {prompt_id: score} for a (policy, kind=cheap|oracle) pair."""
    path = _judge_path(policy_name, kind)
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            s = d.get("score")
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                out[d["prompt_id"]] = float(s)
    return out


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    """Atom-split CVaR_alpha: mean of the lower α-mass, splitting ties at boundary.

    Formula: for sorted y_(1) ≤ y_(2) ≤ ... ≤ y_(n) and k = floor(α·n):
        CVaR_α = (1/(α·n)) · [Σ_{i=1}^{k} y_(i) + (α·n - k) · y_(k+1)]

    On ties at the boundary, this averages exactly α·n units of mass, NOT
    all rows tied at the quantile (which is what `mean(y[y <= quantile(α)])`
    would do — over-averaging on ties).
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


def bootstrap_stat(arr: np.ndarray, fn, B: int = 1000, seed: int = 42, ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (point, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, arr.size, size=arr.size)
        boots[b] = fn(arr[idx])
    point = float(fn(arr))
    return point, float(np.nanquantile(boots, (1 - ci) / 2)), float(np.nanquantile(boots, 1 - (1 - ci) / 2))


def step3_naive_per_policy_summary(kind: str = "cheap"):
    """Print per-policy mean and CVaR_0.05 from cheap-S scores alone (no calibration).

    This is the 'naive direct' view: treats S as if it were Y. Tells us
    nothing about transport, but shows whether the cross-policy spread
    is even worth the calibration step.
    """
    print("=" * 76)
    print(f"STEP 3 — Per-policy summary (naive: using {kind}-S scores as if they were Y)")
    print("=" * 76)
    print(f"{'policy':28} {'n':>4} {'mean':>8} {'std':>7} {'q_0.05':>8} {'CVaR_0.05':>11} {'CI 95%':>20}")
    print("-" * 90)
    rows = []
    for p in POLICIES:
        scores = _load_judge_scores(p.name, kind)
        if not scores:
            print(f"{p.name:28} (no {kind}-judge scores yet)")
            continue
        arr = np.array(list(scores.values()))
        n = arr.size
        mean = float(arr.mean())
        std = float(arr.std())
        q05 = float(np.quantile(arr, 0.05))
        cvar05, cvar_lo, cvar_hi = bootstrap_stat(arr, lambda a: cvar_alpha(a, 0.05))
        mean_pt, mean_lo, mean_hi = bootstrap_stat(arr, np.mean)
        rows.append({
            "policy": p.name, "n": n,
            "mean": mean_pt, "mean_lo": mean_lo, "mean_hi": mean_hi,
            "std": std, "q_05": q05,
            "cvar_05": cvar05, "cvar_05_lo": cvar_lo, "cvar_05_hi": cvar_hi,
        })
        print(f"{p.name:28} {n:>4} {mean_pt:>8.3f} {std:>7.3f} {q05:>8.3f} "
              f"{cvar05:>11.3f}  [{cvar_lo:.3f}, {cvar_hi:.3f}]")
    print()
    return rows


def step4_pairwise_same_mean_diff_tail(rows: list[dict]):
    """Find policy pairs where mean CIs overlap but CVaR_0.05 CIs don't.

    These are the "mean would call them tied; CVaR would not" cases — the headline
    pattern of CVaR-CJE.
    """
    print("=" * 76)
    print("STEP 4 — Same-mean / different-tail pairs (where CVaR matters most)")
    print("=" * 76)
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            mean_overlap = max(a["mean_lo"], b["mean_lo"]) <= min(a["mean_hi"], b["mean_hi"])
            cvar_overlap = max(a["cvar_05_lo"], b["cvar_05_lo"]) <= min(a["cvar_05_hi"], b["cvar_05_hi"])
            dmean = abs(a["mean"] - b["mean"])
            dcvar = abs(a["cvar_05"] - b["cvar_05"])
            ratio = dcvar / max(dmean, 0.001)
            pairs.append({
                "a": a, "b": b, "mean_overlap": mean_overlap, "cvar_overlap": cvar_overlap,
                "dmean": dmean, "dcvar": dcvar, "ratio": ratio,
            })
    # Sort by: most interesting first (mean overlap=True, cvar overlap=False, then ratio)
    pairs.sort(key=lambda p: (not p["mean_overlap"], p["cvar_overlap"], -p["ratio"]))
    print(f"{'policy A':28} {'policy B':28} {'Δmean':>7} {'ΔCVaR':>7} {'ratio':>7} verdict")
    print("-" * 95)
    for p in pairs:
        a_name = p["a"]["policy"]
        b_name = p["b"]["policy"]
        if p["mean_overlap"] and not p["cvar_overlap"]:
            verdict = "★ same-mean, different-tail (CVaR resolves it)"
        elif not p["mean_overlap"] and not p["cvar_overlap"]:
            verdict = "differ on both (mean is enough)"
        elif p["mean_overlap"] and p["cvar_overlap"]:
            verdict = "tied on both (statistically indistinguishable)"
        else:
            verdict = "mean-only difference"
        print(f"{a_name:28} {b_name:28} {p['dmean']:>7.3f} {p['dcvar']:>7.3f} "
              f"{p['ratio']:>7.1f}× {verdict}")
    print()
    return pairs


def step5_oracle_calibrated(verbose: bool = True):
    """Direct CVaR-CJE with cheap-S calibrated against oracle-Y per policy.

    Requires oracle scores. Falls back to message if not present.
    """
    print("=" * 76)
    print("STEP 5 — Direct CVaR-CJE (Cheap-S calibrated against Oracle-Y)")
    print("=" * 76)
    if not HAVE_CJE:
        print("(skipped — _estimator import failed)")
        return None

    # Per-policy: load cheap and oracle scores; compute calibrated CVaR with logger=base
    logger = logger_policy()
    s_log = _load_judge_scores(logger.name, "cheap")
    y_log = _load_judge_scores(logger.name, "oracle")
    common = set(s_log) & set(y_log)
    if len(common) < 5:
        print(f"(skipped — logger '{logger.name}' has only {len(common)} prompts with both cheap-S and oracle-Y; need ≥5)")
        return None

    s_log_arr = np.array([s_log[pid] for pid in common])
    y_log_arr = np.array([y_log[pid] for pid in common])

    print(f"{'target':28} {'n_audit':>8} {'CVaR (Direct)':>15} {'oracle truth':>14} "
          f"{'audit p':>9} verdict")
    print("-" * 90)

    results = []
    for p in POLICIES:
        s_t = _load_judge_scores(p.name, "cheap")
        y_t = _load_judge_scores(p.name, "oracle")
        common_t = list(set(s_t) & set(y_t))
        if len(common_t) < 5:
            continue
        s_target = np.array([s_t[pid] for pid in common_t])
        y_target = np.array([y_t[pid] for pid in common_t])
        # Direct CVaR-CJE
        try:
            cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(
                s_train=s_log_arr, y_train=y_log_arr,
                s_eval=s_target, alpha=0.05, grid_size=61,
            )
            audit = two_moment_wald_audit_xf(
                s_train=s_log_arr, y_train=y_log_arr,
                s_audit=s_target, y_audit=y_target,
                t0=t_hat, alpha=0.05, B=200, fold_seed=0,
            )
        except Exception as e:
            print(f"{p.name:28} ERROR: {e}")
            continue
        true_cvar = cvar_alpha(y_target, 0.05)
        flag = "🔥 audit FIRES" if audit["reject"] else "✅ audit accepts"
        print(f"{p.name:28} {len(common_t):>8} {cvar_est:>+15.3f} {true_cvar:>+14.3f} "
              f"{audit['p_value']:>9.4f} {flag}")
        results.append({
            "target": p.name, "n": len(common_t), "cvar_est": cvar_est,
            "true_cvar": true_cvar, "audit_p": audit["p_value"],
            "audit_reject": audit["reject"],
        })
    print()
    return results


def _logger_panel() -> tuple[np.ndarray, np.ndarray] | None:
    """(s_log, y_log) on prompts where logger has both cheap and oracle.
    Used as the calibration-training data."""
    logger = logger_policy()
    s_log = _load_judge_scores(logger.name, "cheap")
    y_log = _load_judge_scores(logger.name, "oracle")
    common = set(s_log) & set(y_log)
    if len(common) < 5:
        return None
    pids = sorted(common)
    return (np.array([s_log[pid] for pid in pids]),
            np.array([y_log[pid] for pid in pids]))


def _policy_panel(policy_name: str) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """(prompt_ids, s_target, y_target_full) for a policy. y_target_full is
    the full-oracle simulation truth (used as ground truth for verification)."""
    s_t = _load_judge_scores(policy_name, "cheap")
    y_t = _load_judge_scores(policy_name, "oracle")
    common = sorted(set(s_t) & set(y_t))
    if len(common) < 5:
        return None
    return common, np.array([s_t[pid] for pid in common]), np.array([y_t[pid] for pid in common])


def step5_oracle_calibrated_designed(
    *,
    design: str = "uniform",
    audit_variant: str = "two_moment",
    coverage: float = 0.25,
    alpha: float = 0.10,
    seed: int = 42,
    B: int = 100,
    B_ci: int = 500,        # bootstrap reps for cvar_est CI
    K_jackknife: int = 5,   # folds for Var_cal jackknife
    verbose: bool = True,
) -> list[dict]:
    """Designed-slice variant. Same flow as step5 but:
      - apply oracle_design.select_slice to mask which oracle labels are
        estimator-visible
      - HT-weight the calibrator by 1/π_i on selected rows
      - run the chosen audit variant
      - report full-oracle truth alongside the slice-based estimate
    """
    if not HAVE_CJE:
        print("(skipped — _estimator import failed)")
        return []
    if design not in VALID_DESIGNS:
        raise ValueError(f"design must be in {VALID_DESIGNS}, got {design!r}")
    if audit_variant not in AUDIT_VARIANTS:
        raise ValueError(f"audit_variant must be in {tuple(AUDIT_VARIANTS)}, got {audit_variant!r}")
    audit_fn = AUDIT_VARIANTS[audit_variant]

    log_panel = _logger_panel()
    if log_panel is None:
        if verbose:
            print(f"(skipped — logger has fewer than 5 prompts with cheap+oracle)")
        return []
    s_log_full, y_log_full = log_panel

    # Build the design slice on the LOGGER's rows (the calibration-training panel).
    # The slice tells us which logger oracle labels are visible to the estimator.
    logger = logger_policy()
    log_pids = sorted(set(_load_judge_scores(logger.name, "cheap")) & set(_load_judge_scores(logger.name, "oracle")))
    log_rows = [
        {"prompt_id": pid, "policy": logger.name, "cheap_score": float(s_log_full[i]),
         "oracle_score": float(y_log_full[i])}
        for i, pid in enumerate(log_pids)
    ]
    log_slice = select_slice(
        log_rows, design=design, coverage=coverage, alpha=alpha, seed=seed,
    )
    sel_mask = np.array([s.selected for s in log_slice])
    sel_pi = np.array([s.pi for s in log_slice])
    if sel_mask.sum() < 3:
        if verbose:
            print(f"(skipped — design={design} cov={coverage} produced only "
                  f"{sel_mask.sum()} selected logger rows; need ≥3)")
        return []

    s_train = s_log_full[sel_mask]
    y_train = y_log_full[sel_mask]
    w_train = 1.0 / sel_pi[sel_mask]
    if verbose:
        print(f"=== design={design}  audit={audit_variant}  cov={coverage}  α={alpha} ===")
        print(f"  logger slice: {sel_mask.sum()}/{len(sel_mask)} rows; "
              f"strata {[(s.stratum, s.selected) for s in log_slice[:5]]}...")

    results = []
    for p in POLICIES:
        panel = _policy_panel(p.name)
        if panel is None:
            continue
        pids, s_target, y_target_full = panel
        # The audit set is a designed slice of the *target policy* rows
        target_rows = [
            {"prompt_id": pid, "policy": p.name, "cheap_score": float(s_target[i]),
             "oracle_score": float(y_target_full[i])}
            for i, pid in enumerate(pids)
        ]
        target_slice = select_slice(
            target_rows, design=design, coverage=coverage, alpha=alpha, seed=seed,
        )
        a_mask = np.array([s.selected for s in target_slice])
        a_pi = np.array([s.pi for s in target_slice])
        s_audit = s_target[a_mask]
        y_audit = y_target_full[a_mask]
        w_audit = 1.0 / a_pi[a_mask]
        if a_mask.sum() < 3:
            if verbose:
                print(f"  {p.name:28} skipped (n_slice={a_mask.sum()} < 3)")
            continue
        try:
            cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(
                s_train=s_train, y_train=y_train, s_eval=s_target, alpha=alpha,
                sample_weight_train=w_train,
            )
            audit = audit_fn(
                s_train=s_train, y_train=y_train,
                s_audit=s_audit, y_audit=y_audit,
                t0=t_hat, alpha=alpha, B=B, fold_seed=seed,
                sample_weight_train=w_train, sample_weight_audit=w_audit,
            )
            # Bootstrap CI on cvar_est (eval-only variance from resampling logger slice)
            ci_res = bootstrap_cvar_ci(
                s_train=s_train, y_train=y_train, s_eval=s_target, alpha=alpha,
                sample_weight_train=w_train, B=B_ci, seed=seed,
            )
            # Var_cal from jackknife on logger oracle slice
            var_cal = jackknife_var_cal(
                s_oracle=s_train, y_oracle=y_train, s_eval=s_target, alpha=alpha,
                sample_weight_oracle=w_train, K=K_jackknife, seed=seed,
            )
            # Calibration-aware total variance and CI
            var_eval = ci_res["var_eval"]
            var_total = var_eval + var_cal
            half_width = 1.96 * math.sqrt(max(var_total, 0.0))
            ci_lo_total = float(cvar_est - half_width)
            ci_hi_total = float(cvar_est + half_width)
            var_cal_share = float(var_cal / var_total) if var_total > 0 else float("nan")
            # Mean transport audit (per-policy; independent of audit_variant or alpha)
            mean_audit = mean_transport_audit(
                s_train=s_train, y_train=y_train,
                s_audit=s_audit, y_audit=y_audit,
                sample_weight_train=w_train, sample_weight_audit=w_audit,
            )
        except Exception as e:
            if verbose:
                print(f"  {p.name:28} ERROR: {e}")
            continue
        full_truth = float(cvar_alpha(y_target_full, alpha))
        abs_err = float(abs(cvar_est - full_truth))
        in_ci_total = bool(ci_lo_total <= full_truth <= ci_hi_total)
        flag = "🔥 reject" if audit["reject"] else "✅ accept"
        m_flag = "🔥" if mean_audit["reject"] else "✅"
        if verbose:
            print(f"  {p.name:28} n_slice={a_mask.sum():>3} "
                  f"cvar={cvar_est:+.3f} [{ci_lo_total:+.3f},{ci_hi_total:+.3f}] "
                  f"truth={full_truth:+.3f} |err|={abs_err:.3f} "
                  f"Vc/Vt={var_cal_share:.2f} "
                  f"audit_p={audit['p_value']:.3f} {flag} "
                  f"mean_p={mean_audit['p_value']:.3f}{m_flag}")
        results.append({
            "policy": p.name, "design": design, "audit_variant": audit_variant,
            "coverage": coverage, "alpha": alpha, "seed": seed,
            "n_total": int(len(pids)), "n_slice": int(a_mask.sum()),
            "cvar_est": float(cvar_est), "t_hat": float(t_hat),
            "full_oracle_truth": full_truth, "abs_error": abs_err,
            "ci_lo": ci_res["ci_lo"], "ci_hi": ci_res["ci_hi"],  # eval-only percentile bootstrap
            "var_eval": float(var_eval), "var_cal": float(var_cal),
            "var_total": float(var_total), "var_cal_share": var_cal_share,
            "ci_lo_total": ci_lo_total, "ci_hi_total": ci_hi_total,  # calibration-aware
            "in_ci_total": in_ci_total,
            "audit_p": float(audit["p_value"]), "audit_reject": bool(audit["reject"]),
            "mean_g1": float(audit.get("mean_g1", float("nan"))),
            "mean_g2": float(audit.get("mean_g2", float("nan"))),
            "mean_audit_p": float(mean_audit["p_value"]),
            "mean_audit_reject": bool(mean_audit["reject"]),
            "mean_residual": float(mean_audit["residual_mean"]),
        })
    if verbose:
        print()
    return results


# Backward-compat wrapper for the legacy call site
def step5_oracle_calibrated(verbose: bool = True):
    return step5_oracle_calibrated_designed(
        design="uniform", audit_variant="two_moment",
        coverage=1.0, alpha=0.05, seed=42, verbose=verbose,
    )


# --- Minimal pilot path (uniform slice; no CIs; one audit) --------------------

def _verdict(mean_g1: float, mean_g2: float, threshold: float = 0.05) -> str:
    """Heuristic transport verdict from the two CVaR audit moments. Not a
    formal hypothesis test — just a flag for inspection.

    PASS if both |moments| ≤ threshold; otherwise tag the offending moment.
    """
    g1_bad = abs(mean_g1) > threshold
    g2_bad = abs(mean_g2) > threshold
    if not g1_bad and not g2_bad:
        return "PASS"
    if g1_bad and g2_bad:
        return "FLAG_BOTH"
    if g1_bad:
        return "FLAG_TAIL"
    return "FLAG_RESIDUAL"


def step5_oracle_calibrated_uniform(
    *,
    coverage: float = 0.25,
    alpha: float = 0.10,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """Uniform oracle slice. Atom-split CVaR truth. Single transport audit
    (mean_g1, mean_g2) at t̂ from the FULL target distribution.

    No design comparison, no CIs, no variance decomposition, no audit
    variants. The minimum reportable CVaR-CJE pilot result.

    Returns one row per policy with columns:
      policy, alpha, seed, coverage, n_total, n_slice,
      cvar_est, t_hat, full_oracle_truth, abs_error,
      mean_g1, mean_g2, verdict
    """
    if not HAVE_CJE:
        print("(skipped — _estimator import failed)")
        return []

    log_panel = _logger_panel()
    if log_panel is None:
        if verbose:
            print(f"(skipped — logger has fewer than 5 prompts with cheap+oracle)")
        return []
    s_log_full, y_log_full = log_panel

    logger = logger_policy()
    log_pids = sorted(set(_load_judge_scores(logger.name, "cheap")) & set(_load_judge_scores(logger.name, "oracle")))
    log_rows = [
        {"prompt_id": pid, "policy": logger.name, "cheap_score": float(s_log_full[i]),
         "oracle_score": float(y_log_full[i])}
        for i, pid in enumerate(log_pids)
    ]
    log_slice = select_slice(
        log_rows, design="uniform", coverage=coverage, alpha=alpha, seed=seed,
    )
    sel_mask = np.array([s.selected for s in log_slice])
    sel_pi = np.array([s.pi for s in log_slice])
    if sel_mask.sum() < 5:
        if verbose:
            print(f"(skipped — uniform cov={coverage} produced {sel_mask.sum()} logger rows; need ≥5)")
        return []

    # Split the selected logger rows 80/20 into calibration vs held-out audit.
    # The held-out portion is used as the audit slice when target = logger,
    # so the audit's residuals on base are not zero by construction.
    sel_idx = np.where(sel_mask)[0]
    n_sel = int(len(sel_idx))
    rng_split = np.random.default_rng(seed + 991)
    perm = rng_split.permutation(n_sel)
    n_cal = max(3, int(round(0.8 * n_sel)))
    cal_idx_in_sel = perm[:n_cal]
    heldout_idx_in_sel = perm[n_cal:]
    cal_global_idx = sel_idx[cal_idx_in_sel]
    heldout_global_idx = sel_idx[heldout_idx_in_sel]

    s_train = s_log_full[cal_global_idx]
    y_train = y_log_full[cal_global_idx]
    w_train = 1.0 / sel_pi[cal_global_idx]
    s_heldout_log = s_log_full[heldout_global_idx]
    y_heldout_log = y_log_full[heldout_global_idx]

    if verbose:
        print(f"=== uniform  cov={coverage}  α={alpha}  seed={seed} ===")
        print(f"  logger slice: {n_sel}/{len(sel_mask)} rows; "
              f"calibration={len(s_train)}, held-out for base audit={len(s_heldout_log)}")
        print(f"  {'policy':28} {'n_slice':>8} {'cvar_est':>10} {'truth':>10} "
              f"{'|err|':>8} {'mean_g1':>10} {'mean_g2':>10} {'verdict':>14}")
        print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*14}")

    results = []
    for p in POLICIES:
        panel = _policy_panel(p.name)
        if panel is None:
            continue
        pids, s_target, y_target_full = panel
        target_rows = [
            {"prompt_id": pid, "policy": p.name, "cheap_score": float(s_target[i]),
             "oracle_score": float(y_target_full[i])}
            for i, pid in enumerate(pids)
        ]
        target_slice = select_slice(
            target_rows, design="uniform", coverage=coverage, alpha=alpha, seed=seed,
        )
        a_mask = np.array([s.selected for s in target_slice])

        # Route the audit slice based on whether target = logger.
        # When target = logger (base), the target slice and the calibration
        # slice are the same rows; using them for the audit would be a
        # train-on-test leak. Use the held-out logger slice instead.
        if p.name == logger.name:
            s_audit = s_heldout_log
            y_audit = y_heldout_log
            n_audit_eff = int(len(s_audit))
        else:
            s_audit = s_target[a_mask]
            y_audit = y_target_full[a_mask]
            n_audit_eff = int(a_mask.sum())

        if n_audit_eff < 3:
            if verbose:
                print(f"  {p.name:28} skipped (n_audit={n_audit_eff} < 3)")
            continue
        try:
            audit = simple_cvar_audit(
                s_train=s_train, y_train=y_train,
                s_audit=s_audit, y_audit=y_audit,
                s_eval_full=s_target,                # FULL target — t̂ from here
                alpha=alpha, sample_weight_train=w_train,
            )
        except Exception as e:
            if verbose:
                print(f"  {p.name:28} ERROR: {e}")
            continue
        full_truth = float(cvar_alpha(y_target_full, alpha))         # atom-split CVaR
        cheap_only_cvar_v = float(cvar_alpha(s_target, alpha))       # baseline: no calibration
        mean_Y = float(y_target_full.mean())
        abs_err = float(abs(audit["cvar_est"] - full_truth))
        verdict = _verdict(audit["mean_g1"], audit["mean_g2"])

        # Mean-CJE companion: same calibrator (s_train, y_train) → fitted Y-on-S
        # average over the FULL target distribution (Direct Mean CJE estimate);
        # transport audit is the residual mean E[Y - f̂(S)] = 0 on s_audit/y_audit.
        cheap_only_mean = float(s_target.mean())
        mean_pred_full = fit_isotonic_mean(
            s_train, y_train, s_target, sample_weight=w_train,
        )
        mean_cje_est = float(np.asarray(mean_pred_full).mean())
        mean_abs_err = float(abs(mean_cje_est - mean_Y))
        m_audit = mean_transport_audit(
            s_train, y_train, s_audit, y_audit,
            sample_weight_train=w_train,
        )
        mean_verdict = "PASS" if not m_audit["reject"] else "FLAG_MEAN"

        # Bootstrap 95% CIs for the writeup figure error bars. B=200 keeps
        # this fast (≤ a few seconds per policy); the percentile CIs are
        # stable enough for visual presentation. Use seed = audit-seed + 7
        # so the bootstrap RNG path doesn't collide with the slice/audit
        # split RNG.
        cvar_ci = bootstrap_cvar_ci(
            s_train, y_train, s_target, alpha=alpha,
            sample_weight_train=w_train, B=200, seed=seed + 7,
        )
        mean_ci = bootstrap_mean_ci(
            s_train, y_train, s_target,
            sample_weight_train=w_train, B=200, seed=seed + 7,
        )

        if verbose:
            print(f"  {p.name:28} {n_audit_eff:>8} "
                  f"{audit['cvar_est']:>+10.3f} {full_truth:>+10.3f} "
                  f"{abs_err:>8.3f} {audit['mean_g1']:>+10.3f} {audit['mean_g2']:>+10.3f} "
                  f"{verdict:>14}")
        results.append({
            "policy": p.name, "alpha": float(alpha), "seed": int(seed),
            "coverage": float(coverage),
            "n_total": int(len(pids)), "n_slice": int(n_audit_eff),
            "cvar_est": float(audit["cvar_est"]), "t_hat": float(audit["t_hat"]),
            "full_oracle_truth": full_truth, "abs_error": abs_err,
            "cheap_only_cvar": cheap_only_cvar_v, "mean_Y": mean_Y,
            "mean_g1": float(audit["mean_g1"]), "mean_g2": float(audit["mean_g2"]),
            "verdict": verdict,
            # Mean-CJE companion fields
            "cheap_only_mean": cheap_only_mean,
            "mean_cje_est": mean_cje_est,
            "mean_abs_error": mean_abs_err,
            "mean_audit_residual": float(m_audit["residual_mean"]),
            "mean_audit_p": float(m_audit["p_value"]),
            "mean_audit_reject": bool(m_audit["reject"]),
            "mean_verdict": mean_verdict,
            # Bootstrap 95% CIs for figure error bars
            "cvar_ci_lo": float(cvar_ci["ci_lo"]),
            "cvar_ci_hi": float(cvar_ci["ci_hi"]),
            "cvar_se_boot": float(cvar_ci["var_eval"]) ** 0.5,
            "mean_ci_lo": float(mean_ci["ci_lo"]),
            "mean_ci_hi": float(mean_ci["ci_hi"]),
            "mean_se_boot": float(mean_ci["var_eval"]) ** 0.5,
        })
    if verbose:
        print()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["cheap-only", "oracle", "designed"], default="cheap-only",
                    help="cheap-only = step3+4 only; oracle = legacy step5 (full coverage, "
                         "two_moment audit); designed = designed slice + audit variant.")
    ap.add_argument("--design", choices=list(VALID_DESIGNS), default="uniform")
    ap.add_argument("--audit", choices=list(AUDIT_VARIANTS), default="two_moment")
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = step3_naive_per_policy_summary(kind="cheap")
    if not rows:
        print("No cheap-S scores found — run `python3 -m cvar_v4.healthbench_data.judge --kind cheap --all` first.")
        return
    pairs = step4_pairwise_same_mean_diff_tail(rows)

    if args.kind == "oracle":
        step5_oracle_calibrated()
    elif args.kind == "designed":
        step5_oracle_calibrated_designed(
            design=args.design, audit_variant=args.audit,
            coverage=args.coverage, alpha=args.alpha, seed=args.seed,
        )
    else:
        print("Skipping STEP 5. To run it: ")
        print("  python3 -m cvar_v4.healthbench_data.judge --kind oracle --all")
        print("  python3 -m cvar_v4.healthbench_data.analyze --kind oracle")
        print("Or with a designed slice:")
        print("  python3 -m cvar_v4.healthbench_data.analyze --kind designed --design floor_tail --audit g2_only --coverage 0.25 --alpha 0.10")


if __name__ == "__main__":
    main()
