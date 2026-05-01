"""Phase G: rigorous check of pilot against CVaR-CJE theory.

Q1 — pipeline conformance: did we run the canonical pipeline (cvar_v3/SCHEMA.md)?
Q2 — surrogate quality: does cheap S have signal for oracle Y?
Q3 — Mean-CJE: does the calibrated mean estimator beat naive cheap-S?
       does the mean residual audit detect transport failures?
Q4 — CVaR-CJE: does CVaR add value above Mean?
       does the CVaR audit have bite?

All computations from existing data; no API calls. Run:
    python3 -m cvar_v4.healthbench_data.methodology_audit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from .judge import _judge_path
from .policies import POLICIES, logger_policy

JUDGE = Path(__file__).parent / "judge_outputs"


def _load(pol: str, kind: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in _judge_path(pol, kind).open():
        d = json.loads(line)
        s = d.get("score")
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            out[d["prompt_id"]] = float(s)
    return out


# ---------------------------------------------------------------------------
# Q1: pipeline conformance
# ---------------------------------------------------------------------------

CANONICAL = [
    ("n_folds (cross-fit)",          "5",                "1 (calibrator trained on all of base)"),
    ("n_bootstrap",                  "2000",             "1000 (mean), 200 (audit)"),
    ("oracle_coverage",              "0.25 (headline)",  "1.00 (full oracle on all n=100)"),
    ("sample_sizes",                 "5000 (headline)",  "100"),
    ("seeds",                        "≥ 20",             "1 (seed=42)"),
    ("calibration_mode",             "two_stage + response_length",
                                                          "single-stage isotonic, no covariate"),
    ("inference_method",             "cluster_robust",   "i.i.d. (no clustering)"),
    ("TTC gate",                     "≥ 0.70 reported",  "not computed"),
    ("oua_jackknife",                "True",             "not used"),
    ("use_augmented_estimator",      "True",             "False (Direct only)"),
]


def Q1_pipeline_conformance():
    print("=" * 78)
    print("Q1 — pipeline conformance vs cvar_v3/SCHEMA.md")
    print("=" * 78)
    print(f"  {'parameter':32} {'canonical':24} ours")
    print("  " + "-" * 78)
    for name, canon, ours in CANONICAL:
        print(f"  {name:32} {canon:24} {ours}")
    print()
    print("  → We ran a SIMPLIFIED pilot. Numbers are illustrative, not paper-quality.")
    print("    Missing rigor: cross-fitting, multi-seed, cluster bootstrap, two-stage calibration,")
    print("    response_length covariate, TTC gate, OUA jackknife. Justification: this is")
    print("    a feasibility/EDA pass, not a paper reproduction. Each missing piece can be")
    print("    added incrementally before scaling.")


# ---------------------------------------------------------------------------
# Q2: surrogate quality
# ---------------------------------------------------------------------------

def Q2_surrogate_works():
    print()
    print("=" * 78)
    print("Q2 — does the cheap-S surrogate have signal for oracle-Y?")
    print("=" * 78)
    # Per policy: pearson, spearman, isotonic R² (within-policy: how well does
    # an isotonic fit on this policy's own (s, y) explain its y?)
    print(f"  {'policy':28} {'n':>4} {'Pearson':>9} {'Spearman':>10} "
          f"{'iso R²':>8} {'(S,Y) gap':>10}")
    print("  " + "-" * 76)
    summary: dict[str, dict] = {}
    for p in POLICIES:
        s = _load(p.name, "cheap")
        y = _load(p.name, "oracle")
        common = sorted(set(s) & set(y))
        if len(common) < 5:
            continue
        s_arr = np.array([s[i] for i in common])
        y_arr = np.array([y[i] for i in common])
        rho_p = float(stats.pearsonr(s_arr, y_arr).statistic)
        rho_s = float(stats.spearmanr(s_arr, y_arr).statistic)
        # Isotonic R² on its own data (no train/test split)
        order = np.argsort(s_arr)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(s_arr[order], y_arr[order])
        y_hat = iso.predict(s_arr)
        ss_res = float(((y_arr - y_hat) ** 2).sum())
        ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        gap = float(s_arr.mean() - y_arr.mean())
        summary[p.name] = {"pearson": rho_p, "spearman": rho_s,
                            "iso_r2": r2, "gap": gap, "n": len(common)}
        print(f"  {p.name:28} {len(common):>4} {rho_p:>+9.3f} {rho_s:>+10.3f} "
              f"{r2:>8.3f} {gap:>+10.3f}")
    print()
    print("  Interpretation:")
    print("    Pearson, Spearman: monotone signal strength of S for Y. > 0.7 = strong.")
    print("    iso R²: fraction of within-policy Y-variance explained by an isotonic")
    print("            S→Y fit (upper bound on what the surrogate can do for that policy).")
    print("    (S,Y) gap: cheap-S minus oracle-Y means. Positive = cheap is more lenient.")
    return summary


# ---------------------------------------------------------------------------
# Q3: Mean-CJE
# ---------------------------------------------------------------------------

def _fit_iso_on(s_train, y_train):
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train[order], y_train[order])
    return iso


def Q3_mean_cje():
    print()
    print("=" * 78)
    print("Q3 — Mean-CJE: calibrated mean vs naive cheap-S vs oracle truth")
    print("=" * 78)
    logger = logger_policy()
    s_log_d = _load(logger.name, "cheap")
    y_log_d = _load(logger.name, "oracle")
    common = sorted(set(s_log_d) & set(y_log_d))
    s_log = np.array([s_log_d[i] for i in common])
    y_log = np.array([y_log_d[i] for i in common])
    iso = _fit_iso_on(s_log, y_log)

    print(f"  {'target':28} {'n':>4} {'S_mean':>8} {'Y_mean':>8} "
          f"{'Direct':>8} {'|S-Y|':>7} {'|D-Y|':>7} {'Direct/S?':>10}")
    print("  " + "-" * 84)
    rows = []
    for p in POLICIES:
        s_t_d = _load(p.name, "cheap")
        y_t_d = _load(p.name, "oracle")
        common_t = sorted(set(s_t_d) & set(y_t_d))
        s_t = np.array([s_t_d[i] for i in common_t])
        y_t = np.array([y_t_d[i] for i in common_t])
        s_mean = float(s_t.mean())
        y_mean = float(y_t.mean())
        y_hat = iso.predict(s_t)
        direct = float(y_hat.mean())
        e_s = abs(s_mean - y_mean)
        e_d = abs(direct - y_mean)
        winner = "Direct ✓" if e_d < e_s else "naive S"
        rows.append({
            "target": p.name, "n": len(common_t),
            "s_mean": s_mean, "y_mean": y_mean, "direct": direct,
            "e_s": e_s, "e_d": e_d, "y_hat": y_hat, "y_t": y_t,
        })
        print(f"  {p.name:28} {len(common_t):>4} {s_mean:>+8.3f} {y_mean:>+8.3f} "
              f"{direct:>+8.3f} {e_s:>7.3f} {e_d:>7.3f} {winner:>10}")
    print()
    # Sumarize: count how many policies the Direct estimate beats naive S on
    n_direct_wins = sum(1 for r in rows if r["e_d"] < r["e_s"])
    print(f"  Direct beats naive S on {n_direct_wins}/{len(rows)} policies.")
    # Average bias reduction
    avg_e_s = float(np.mean([r["e_s"] for r in rows]))
    avg_e_d = float(np.mean([r["e_d"] for r in rows]))
    print(f"  Average |bias|: naive S = {avg_e_s:.3f}, Direct = {avg_e_d:.3f} "
          f"(reduction = {(1 - avg_e_d/max(avg_e_s,1e-9))*100:.0f}%)")

    # Mean residual audit: per policy, t-test on r_i = y_i - ŷ_i. H_0: E[r]=0.
    print()
    print("  --- Mean residual audit (paired t-test on Y - ĝ(S) per policy) ---")
    print(f"  {'target':28} {'mean_r':>8} {'sd_r':>7} {'t':>7} {'p':>9} verdict")
    print("  " + "-" * 76)
    for r in rows:
        resid = r["y_t"] - r["y_hat"]
        m = float(resid.mean())
        sd = float(resid.std(ddof=1))
        n = resid.size
        t_stat = m / (sd / np.sqrt(n))
        p_val = 2.0 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        flag = "🔥 FIRES" if p_val < 0.05 else "✅ accepts"
        print(f"  {r['target']:28} {m:>+8.4f} {sd:>7.3f} {t_stat:>+7.2f} {p_val:>9.4f} {flag}")
        r["mean_audit_p"] = float(p_val)
        r["mean_audit_t"] = float(t_stat)
        r["mean_resid"] = m
    print()
    print("  Interpretation: H_0 is 'the calibrator's prediction is unbiased on this")
    print("  target's population'. p < 0.05 → REJECT → transport fails for the mean.")
    return rows


# ---------------------------------------------------------------------------
# Q4: CVaR-CJE
# ---------------------------------------------------------------------------

def _cvar(arr, alpha=0.05):
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def _direct_cvar_grid(s_train, y_train, s_eval, alpha=0.05, grid_size=61):
    t_lo = float(np.quantile(y_train, max(0.001, alpha / 5.0)) - 0.60)
    t_hi = float(np.quantile(y_train, min(0.60, alpha + 0.45)) + 0.35)
    t_grid = np.linspace(t_lo, t_hi, grid_size)
    obj = np.empty(len(t_grid))
    iso_tail_per_t = []
    for i, t in enumerate(t_grid):
        z_train = np.maximum(t - y_train, 0.0)
        order = np.argsort(s_train)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(s_train[order], z_train[order])
        pred_eval = iso.predict(s_eval)
        obj[i] = float(t - pred_eval.mean() / alpha)
        iso_tail_per_t.append(iso)
    best = int(np.argmax(obj))
    return float(obj[best]), float(t_grid[best]), iso_tail_per_t[best]


def Q4_cvar_cje(mean_rows: list[dict]):
    print()
    print("=" * 78)
    print("Q4 — CVaR-CJE: tail value above mean? does the CVaR audit have bite?")
    print("=" * 78)
    logger = logger_policy()
    s_log_d = _load(logger.name, "cheap")
    y_log_d = _load(logger.name, "oracle")
    common = sorted(set(s_log_d) & set(y_log_d))
    s_log = np.array([s_log_d[i] for i in common])
    y_log = np.array([y_log_d[i] for i in common])

    # Per policy: Direct CVaR, S CVaR (naive), Y CVaR (truth)
    print(f"  {'target':28} {'n':>4} {'S_CVaR':>8} {'Y_CVaR':>8} "
          f"{'D_CVaR':>8} {'|S-Y|':>7} {'|D-Y|':>7} {'D vs S':>10}")
    print("  " + "-" * 84)
    rows = []
    for p in POLICIES:
        s_t_d = _load(p.name, "cheap")
        y_t_d = _load(p.name, "oracle")
        common_t = sorted(set(s_t_d) & set(y_t_d))
        s_t = np.array([s_t_d[i] for i in common_t])
        y_t = np.array([y_t_d[i] for i in common_t])
        s_cvar = _cvar(s_t)
        y_cvar = _cvar(y_t)
        d_cvar, t_hat, iso_tail = _direct_cvar_grid(s_log, y_log, s_t)
        e_s = abs(s_cvar - y_cvar)
        e_d = abs(d_cvar - y_cvar)
        winner = "Direct ✓" if e_d < e_s else "naive S"
        rows.append({
            "target": p.name, "n": len(common_t),
            "s_cvar": s_cvar, "y_cvar": y_cvar, "d_cvar": d_cvar,
            "e_s": e_s, "e_d": e_d, "t_hat": t_hat, "iso_tail": iso_tail,
            "s_t": s_t, "y_t": y_t,
        })
        print(f"  {p.name:28} {len(common_t):>4} {s_cvar:>+8.3f} {y_cvar:>+8.3f} "
              f"{d_cvar:>+8.3f} {e_s:>7.3f} {e_d:>7.3f} {winner:>10}")
    print()
    n_direct_wins = sum(1 for r in rows if r["e_d"] < r["e_s"])
    print(f"  Direct beats naive S on {n_direct_wins}/{len(rows)} policies for CVaR.")
    avg_e_s = float(np.mean([r["e_s"] for r in rows]))
    avg_e_d = float(np.mean([r["e_d"] for r in rows]))
    print(f"  Average |bias|: naive S = {avg_e_s:.3f}, Direct = {avg_e_d:.3f}")

    # Rank disagreement: does mean ordering match CVaR ordering?
    print()
    print("  --- Rank disagreement: mean vs CVaR ---")
    pol_order = [r["target"] for r in rows]
    mean_truths = {r["target"]: m["y_mean"] for r, m in zip(rows, mean_rows)}
    cvar_truths = {r["target"]: r["y_cvar"] for r in rows}
    # Sort by mean (desc) and by CVaR (desc)
    mean_rank = sorted(pol_order, key=lambda p: -mean_truths[p])
    cvar_rank = sorted(pol_order, key=lambda p: -cvar_truths[p])
    print("  Policies ranked by oracle mean (best → worst):")
    for i, p in enumerate(mean_rank, 1):
        print(f"    {i}. {p:28} mean={mean_truths[p]:+.3f}")
    print("  Policies ranked by oracle CVaR_0.05 (best → worst):")
    for i, p in enumerate(cvar_rank, 1):
        print(f"    {i}. {p:28} CVaR={cvar_truths[p]:+.3f}")
    swaps = []
    for i in range(len(pol_order)):
        for j in range(i+1, len(pol_order)):
            a, b = pol_order[i], pol_order[j]
            mean_winner = a if mean_truths[a] > mean_truths[b] else b
            cvar_winner = a if cvar_truths[a] > cvar_truths[b] else b
            if mean_winner != cvar_winner:
                swaps.append((a, b, mean_winner, cvar_winner))
    if swaps:
        print(f"  ⚠ Mean and CVaR disagree on {len(swaps)} pair(s):")
        for a, b, mw, cw in swaps:
            print(f"    {a} vs {b}: mean says {mw} better, CVaR says {cw} better")
    else:
        print("  Mean and CVaR rankings agree on every pair.")

    # Tail residual audit: per policy, paired t-test on
    # tail_resid_i = max(t̂ - y_i, 0) - ĝ_tail(s_i)
    print()
    print("  --- Tail residual audit (paired t-test on tail loss residuals) ---")
    print(f"  {'target':28} {'mean_r':>8} {'sd_r':>7} {'t':>7} {'p':>9} verdict")
    print("  " + "-" * 76)
    for r in rows:
        z_target = np.maximum(r["t_hat"] - r["y_t"], 0.0)
        z_pred = r["iso_tail"].predict(r["s_t"])
        resid = z_target - z_pred
        m = float(resid.mean())
        sd = float(resid.std(ddof=1))
        n = resid.size
        if sd < 1e-9:
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = m / (sd / np.sqrt(n))
            p_val = 2.0 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        flag = "🔥 FIRES" if p_val < 0.05 else "✅ accepts"
        print(f"  {r['target']:28} {m:>+8.4f} {sd:>7.3f} {t_stat:>+7.2f} {p_val:>9.4f} {flag}")
        r["tail_audit_p"] = float(p_val)
        r["tail_audit_t"] = float(t_stat)
    print()
    print("  Interpretation: H_0 is 'calibrator's predicted tail loss matches actual on")
    print("  the target population'. p < 0.05 → REJECT → CVaR transport fails.")
    return rows


def main():
    Q1_pipeline_conformance()
    Q2_surrogate_works()
    mean_rows = Q3_mean_cje()
    Q4_cvar_cje(mean_rows)


if __name__ == "__main__":
    main()
