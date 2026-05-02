"""Variance breakdown for Direct CVaR-CJE.

Reports, per (policy, alpha), both the plug-in CVaR estimator
V_plugin = sup_t [t - mean g_t(S)/alpha] and the augmented one-step
V_aug = V_plugin + g_bar_2, with five variance components computed
side-by-side so the math and the code logic can be diffed:

  full      — full pipeline bootstrap: resample logger + target-eval +
              target-audit; refit calibrator; re-maximize t_hat;
              recompute g_bar_2 per rep
  cal       — train-only bootstrap (eval + audit fixed)
  eval      — eval-only bootstrap (train + audit fixed)
  audit     — audit-only bootstrap (train + eval fixed); zero by
              construction for plug-in
  cal_jack  — analytical Var_cal via jackknife on the logger oracle slice
  audit_an  — analytical Var_audit via se_g2^2 at the production t_hat

Identity check column reports var_full vs (var_cal + var_eval + var_audit)
from the per-source bootstraps. They should agree up to bootstrap MC
noise; a wide gap is a code-vs-math signal.

Slice routing exactly mirrors analyze.step5_oracle_calibrated_uniform:
  - uniform design at coverage (default 0.25); HT weights = 1/pi.
  - 80/20 split of the selected logger rows into calibration vs held-out
    audit; the held-out portion is the audit slice when target == base.
  - Non-base policies use their own designed audit slice.

Outputs:
    writeup/data/variance_breakdown.json
    writeup/data/variance_breakdown.md

Usage:
    python -m cvar_v4.healthbench_data.analyses.variance_breakdown
    python -m cvar_v4.healthbench_data.analyses.variance_breakdown --B 50  # smoke
    python -m cvar_v4.healthbench_data.analyses.variance_breakdown --alphas 1.0  # identity
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..analyze import _logger_panel, _policy_panel, _load_judge_scores
from ..oracle_design import select_slice
from ..policies import POLICIES, logger_policy
from ...eda.deeper._estimator import (
    pipeline_bootstrap_cvar,
    jackknife_var_cal,
    cvar_audit_analytical_se,
    estimate_direct_cvar_isotonic,
    fit_isotonic_tail_loss,
)
from ._common import WRITEUP_DATA_DIR, LABELS


def _assemble_cell(
    coverage: float,
    alpha: float,
    seed: int,
) -> tuple[dict, dict]:
    """Build the same data layout step5_oracle_calibrated_uniform uses.

    Returns (logger_state, per_policy_state). logger_state has the
    calibration train arrays + held-out base audit arrays. per_policy_state
    maps policy name to (s_eval_full, s_audit, y_audit, w_audit) where
    s_eval_full is the FULL target cheap-score panel and (s_audit, y_audit,
    w_audit) is the policy's audit slice (or the held-out logger slice for
    base).
    """
    log_panel = _logger_panel()
    if log_panel is None:
        raise RuntimeError("logger panel is empty (need cheap+oracle on >=5 rows)")
    s_log_full, y_log_full = log_panel
    logger = logger_policy()
    log_pids = sorted(
        set(_load_judge_scores(logger.name, "cheap"))
        & set(_load_judge_scores(logger.name, "oracle"))
    )
    log_rows = [
        {"prompt_id": pid, "policy": logger.name,
         "cheap_score": float(s_log_full[i]), "oracle_score": float(y_log_full[i])}
        for i, pid in enumerate(log_pids)
    ]
    log_slice = select_slice(
        log_rows, design="uniform", coverage=coverage, alpha=alpha, seed=seed,
    )
    sel_mask = np.array([s.selected for s in log_slice])
    sel_pi = np.array([s.pi for s in log_slice])
    if sel_mask.sum() < 5:
        raise RuntimeError(
            f"uniform cov={coverage} produced {sel_mask.sum()} logger rows; need >=5"
        )

    sel_idx = np.where(sel_mask)[0]
    n_sel = int(len(sel_idx))
    rng_split = np.random.default_rng(seed + 991)
    perm = rng_split.permutation(n_sel)
    n_cal = max(3, int(round(0.8 * n_sel)))
    cal_global_idx = sel_idx[perm[:n_cal]]
    heldout_global_idx = sel_idx[perm[n_cal:]]

    s_train = s_log_full[cal_global_idx]
    y_train = y_log_full[cal_global_idx]
    w_train = 1.0 / sel_pi[cal_global_idx]
    s_heldout_log = s_log_full[heldout_global_idx]
    y_heldout_log = y_log_full[heldout_global_idx]
    w_heldout_log = 1.0 / sel_pi[heldout_global_idx]

    logger_state = {
        "s_train": s_train, "y_train": y_train, "w_train": w_train,
        "s_heldout": s_heldout_log, "y_heldout": y_heldout_log,
        "w_heldout": w_heldout_log, "name": logger.name,
    }

    per_policy: dict[str, dict] = {}
    for p in POLICIES:
        panel = _policy_panel(p.name)
        if panel is None:
            continue
        pids, s_target, y_target_full = panel
        target_rows = [
            {"prompt_id": pid, "policy": p.name,
             "cheap_score": float(s_target[i]), "oracle_score": float(y_target_full[i])}
            for i, pid in enumerate(pids)
        ]
        target_slice = select_slice(
            target_rows, design="uniform", coverage=coverage, alpha=alpha, seed=seed,
        )
        a_mask = np.array([s.selected for s in target_slice])
        a_pi = np.array([s.pi for s in target_slice])

        if p.name == logger.name:
            s_audit = s_heldout_log
            y_audit = y_heldout_log
            w_audit = w_heldout_log
        else:
            s_audit = s_target[a_mask]
            y_audit = y_target_full[a_mask]
            w_audit = 1.0 / a_pi[a_mask]

        per_policy[p.name] = {
            "s_eval_full": s_target,        # full target panel
            "y_eval_full": y_target_full,   # diagnostic only (for n_eval, not used in estimator)
            "s_audit": s_audit, "y_audit": y_audit, "w_audit": w_audit,
            "n_eval": int(len(s_target)),
            "n_audit": int(len(s_audit)),
        }

    return logger_state, per_policy


def _master_indices(n_train: int, n_eval: int, n_audit: int, B: int, seed: int):
    """Pre-generate B per-rep index arrays for train, eval, audit. Used to
    couple the RNG across the four bootstrap branches so MC noise cancels in
    the additivity check Var_full = Var_cal + Var_eval + Var_audit + cov."""
    rng = np.random.default_rng(seed)
    idx_t = rng.integers(0, n_train, size=(B, n_train))
    idx_e = rng.integers(0, n_eval, size=(B, n_eval))
    idx_a = rng.integers(0, max(n_audit, 1), size=(B, max(n_audit, 1)))
    if n_audit == 0:
        idx_a = np.zeros((B, 0), dtype=int)
    id_t = np.broadcast_to(np.arange(n_train), (B, n_train))
    id_e = np.broadcast_to(np.arange(n_eval), (B, n_eval))
    id_a = np.broadcast_to(np.arange(n_audit), (B, n_audit))
    return {
        "boot": {"train": idx_t, "eval": idx_e, "audit": idx_a},
        "id":   {"train": id_t, "eval": id_e, "audit": id_a},
    }


def _branch(
    log: dict, pol: dict, alpha: float, *, resample: tuple[str, ...],
    B: int, seed: int, master: dict | None = None,
) -> dict:
    """Run one bootstrap branch.

    If `master` is supplied, use coupled RNG: pass `master.boot[k]` for keys
    in `resample` and `master.id[k]` for the un-resampled sets. This makes
    Var_full and the marginal Var_x bootstraps share the same train/eval/
    audit index draws, so the only differences in V̂_b come from which sets
    are jiggled vs frozen — MC noise cancels in the additivity comparison.
    """
    kwargs = dict(
        s_train=log["s_train"], y_train=log["y_train"],
        s_eval_full=pol["s_eval_full"],
        s_audit=pol["s_audit"], y_audit=pol["y_audit"],
        alpha=alpha,
        sample_weight_train=log["w_train"],
        sample_weight_audit=pol["w_audit"],
        resample=resample, B=B, seed=seed,
    )
    if master is not None:
        kwargs["idx_train_per_b"] = master["boot"]["train"] if "train" in resample else master["id"]["train"]
        kwargs["idx_eval_per_b"]  = master["boot"]["eval"]  if "eval"  in resample else master["id"]["eval"]
        kwargs["idx_audit_per_b"] = master["boot"]["audit"] if "audit" in resample else master["id"]["audit"]
    return pipeline_bootstrap_cvar(**kwargs)


def compute(
    *,
    coverage: float = 0.25,
    alphas: tuple[float, ...] = (0.10, 0.20),
    B: int = 500,
    seed: int = 42,
    K_jack: int = 5,
    verbose: bool = False,
) -> dict:
    out_rows: list[dict] = []
    meta = {
        "coverage": coverage, "alphas": list(alphas), "B": B, "seed": seed,
        "K_jack": K_jack, "design": "uniform",
    }

    for alpha in alphas:
        log, per_policy = _assemble_cell(coverage, alpha, seed)
        n_train = int(len(log["s_train"]))
        for p in POLICIES:
            if p.name not in per_policy:
                continue
            pol = per_policy[p.name]
            if pol["n_audit"] < 3:
                if verbose:
                    print(f"  skip {p.name} alpha={alpha}: n_audit={pol['n_audit']} < 3")
                continue

            # Pre-generate one master set of bootstrap indices and pass it
            # to all four branches. This couples the RNG paths so MC noise
            # in Var_full and Var_(cal/eval/audit) cancels in the additivity
            # check — any remaining gap is the structural cross-source
            # covariance from the non-smooth sup_t operator.
            master = _master_indices(
                n_train=int(len(log["s_train"])), n_eval=pol["n_eval"],
                n_audit=pol["n_audit"], B=B, seed=seed,
            )
            full = _branch(log, pol, alpha, resample=("train", "eval", "audit"),
                           B=B, seed=seed, master=master)
            cal = _branch(log, pol, alpha, resample=("train",), B=B, seed=seed, master=master)
            evb = _branch(log, pol, alpha, resample=("eval",), B=B, seed=seed, master=master)
            aud = _branch(log, pol, alpha, resample=("audit",), B=B, seed=seed, master=master)

            # Analytical baselines at the production t_hat
            t_hat = full["t_hat_point"]
            var_cal_jack = jackknife_var_cal(
                s_oracle=log["s_train"], y_oracle=log["y_train"],
                s_eval=pol["s_eval_full"], alpha=alpha,
                sample_weight_oracle=log["w_train"], K=K_jack, seed=seed,
            )
            an = cvar_audit_analytical_se(
                log["s_train"], log["y_train"],
                pol["s_audit"], pol["y_audit"],
                t0=t_hat, alpha=alpha,
                sample_weight_train=log["w_train"],
            )
            var_audit_an = float(an["se_g2"] ** 2) if an["se_g2"] == an["se_g2"] else float("nan")
            var_sum_an_total = (var_cal_jack + var_audit_an
                                if var_cal_jack == var_cal_jack
                                and var_audit_an == var_audit_an else float("nan"))

            sum_boot_plug = cal["var_plug"] + evb["var_plug"] + aud["var_plug"]
            sum_boot_aug = cal["var_aug"] + evb["var_aug"] + aud["var_aug"]

            common = {
                "policy": p.name, "alpha": float(alpha), "seed": int(seed),
                "coverage": float(coverage), "B": int(B),
                "n_train": n_train, "n_eval": pol["n_eval"], "n_audit": pol["n_audit"],
                "t_hat_point": float(t_hat),
                "gbar2_point": float(full["gbar2_point"]),
                "var_cal_jack": float(var_cal_jack),
                "var_audit_an": float(var_audit_an),
                "var_sum_an_total": float(var_sum_an_total),
                # Grid-snap diagnostic: how many distinct t_hat values across
                # the B reps in the FULL bootstrap. If single-digit at B=500,
                # the threshold grid is too sparse and Var_eval is artificially
                # deflated by snapping to identical grid points.
                "n_unique_t_hat_full": int(full["n_unique_t_hat"]),
                "n_unique_t_hat_eval": int(evb["n_unique_t_hat"]),
            }

            # plug-in row
            out_rows.append({
                **common,
                "estimator": "plugin",
                "point": float(full["plug_point"]),
                "var_full_boot": float(full["var_plug"]),
                "ci_lo_full": float(full["ci_plug"][0]),
                "ci_hi_full": float(full["ci_plug"][1]),
                "var_cal_boot": float(cal["var_plug"]),
                "var_eval_boot": float(evb["var_plug"]),
                "var_audit_boot": float(aud["var_plug"]),
                "sum_boot_parts": float(sum_boot_plug),
            })
            # augmented row
            out_rows.append({
                **common,
                "estimator": "aug",
                "point": float(full["aug_point"]),
                "var_full_boot": float(full["var_aug"]),
                "ci_lo_full": float(full["ci_aug"][0]),
                "ci_hi_full": float(full["ci_aug"][1]),
                "var_cal_boot": float(cal["var_aug"]),
                "var_eval_boot": float(evb["var_aug"]),
                "var_audit_boot": float(aud["var_aug"]),
                "sum_boot_parts": float(sum_boot_aug),
            })

            if verbose:
                print(
                    f"  α={alpha} {p.name:30} "
                    f"plug={full['plug_point']:+.3f} aug={full['aug_point']:+.3f} "
                    f"var_full(plug)={full['var_plug']:.5f} "
                    f"sum_parts(plug)={sum_boot_plug:.5f}"
                )
    return {"meta": meta, "rows": out_rows}


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

def _fmt_var(v: float) -> str:
    if v != v:  # NaN
        return "  nan  "
    return f"{v:.5f}"


def _fmt_pt(v: float) -> str:
    if v != v:
        return " nan "
    return f"{v:+.3f}"


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    rows = payload["rows"]
    lines = []
    lines.append(f"# Variance breakdown — Direct CVaR-CJE")
    lines.append("")
    lines.append(
        f"design=uniform  coverage={meta['coverage']}  B={meta['B']}  "
        f"seed={meta['seed']}  K_jack={meta['K_jack']}"
    )
    lines.append("")
    lines.append("Columns:")
    lines.append("- **point** — V̂ (plug-in: sup_t [t − mean ĝ_t(S)/α]; aug: V̂ + ḡ_2)")
    lines.append("- **var_full** — full-pipeline bootstrap variance (resample train+eval+audit)")
    lines.append("- **CI 95%** — percentile interval from same full bootstrap")
    lines.append("- **var_cal_b / var_eval_b / var_audit_b** — per-source bootstrap variances (other sets fixed)")
    lines.append("- **var_cal_J** — jackknife Var_cal on logger oracle slice (K=5)")
    lines.append("- **var_audit_an** — analytical se_g2² at production t̂")
    lines.append("- **sum_an_total** — currently shipped formula `var_cal_J + var_audit_an`")
    lines.append("- **sum_b_parts** — `var_cal_b + var_eval_b + var_audit_b` (identity check vs var_full)")
    lines.append("")

    by_alpha: dict[float, list[dict]] = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], []).append(r)

    for alpha in sorted(by_alpha):
        lines.append(f"## α = {alpha}")
        lines.append("")
        header = (
            "| policy | est | point | var_full | CI 95% | "
            "var_cal_b | var_cal_J | var_eval_b | var_audit_b | var_audit_an | "
            "sum_an_total | sum_b_parts | n_tr | n_ev | n_au | t̂ | ḡ_2 | uniq t̂ |"
        )
        sep = "|" + "|".join(["---"] * 18) + "|"
        lines.append(header)
        lines.append(sep)
        # group by policy then estimator (plugin first)
        by_policy: dict[str, list[dict]] = {}
        for r in by_alpha[alpha]:
            by_policy.setdefault(r["policy"], []).append(r)
        for pol in [p.name for p in POLICIES]:
            if pol not in by_policy:
                continue
            for r in sorted(by_policy[pol], key=lambda x: 0 if x["estimator"] == "plugin" else 1):
                ci = f"[{r['ci_lo_full']:+.3f}, {r['ci_hi_full']:+.3f}]"
                row = (
                    f"| {LABELS.get(pol, pol)} | {r['estimator']} "
                    f"| {_fmt_pt(r['point'])} "
                    f"| {_fmt_var(r['var_full_boot'])} | {ci} "
                    f"| {_fmt_var(r['var_cal_boot'])} | {_fmt_var(r['var_cal_jack'])} "
                    f"| {_fmt_var(r['var_eval_boot'])} | {_fmt_var(r['var_audit_boot'])} "
                    f"| {_fmt_var(r['var_audit_an'])} | {_fmt_var(r['var_sum_an_total'])} "
                    f"| {_fmt_var(r['sum_boot_parts'])} "
                    f"| {r['n_train']} | {r['n_eval']} | {r['n_audit']} "
                    f"| {r['t_hat_point']:+.3f} | {r['gbar2_point']:+.4f} "
                    f"| {r['n_unique_t_hat_full']}/{r['B']} |"
                )
                lines.append(row)
        lines.append("")

        # quick consistency notes per α
        gaps_plug = []
        gaps_aug = []
        for r in by_alpha[alpha]:
            v = r["var_full_boot"]; s = r["sum_boot_parts"]
            if v <= 0 or v != v or s != s:
                continue
            gap = abs(v - s) / max(v, 1e-12)
            (gaps_plug if r["estimator"] == "plugin" else gaps_aug).append((r["policy"], gap))
        if gaps_plug or gaps_aug:
            lines.append("**Identity check (var_full vs sum_b_parts):**")
            for label, gaps in (("plugin", gaps_plug), ("aug", gaps_aug)):
                if not gaps:
                    continue
                worst = max(gaps, key=lambda x: x[1])
                lines.append(
                    f"- {label}: worst gap = {worst[1]*100:.1f}% on {worst[0]}; "
                    f"max across all policies = {max(g for _, g in gaps)*100:.1f}%"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--alphas", type=str, default="0.10,0.20",
                    help="Comma-separated list of alphas, e.g. 0.10,0.20")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K-jack", type=int, default=5, dest="K_jack")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())

    payload = compute(
        coverage=args.coverage, alphas=alphas, B=args.B, seed=args.seed,
        K_jack=args.K_jack, verbose=not args.quiet,
    )

    out_json = WRITEUP_DATA_DIR / "variance_breakdown.json"
    out_md = WRITEUP_DATA_DIR / "variance_breakdown.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    out_md.write_text(render_markdown(payload))
    print(f"[variance_breakdown] {len(payload['rows'])} rows -> {out_md}")


if __name__ == "__main__":
    main()
