"""Multi-seed coverage and MSE sweep for plug-in vs augmented Direct CVaR-CJE.

Tests the hypothesis from the inspection report (S2): does the augmented
estimator (V_plug + g_bar_2) actually have lower bias than plug-in, and
does it pay for that with worse MSE on small audit slices (e.g. base with
n_audit=26)?

For each seed in 0..N-1:
  - rebuild the data cell (uniform design, coverage=0.25)
  - run the FULL pipeline bootstrap with coupled master indices (B reps,
    resample train + eval + audit; refit calibrator, re-maximize t_hat)
  - record V_plug, V_aug, full-bootstrap CIs, full-oracle truth

Aggregate per (policy, alpha):
  coverage_plug, coverage_aug   — fraction of seeds covering full-oracle truth
  bias_plug,     bias_aug       — mean (V - truth) across seeds
  mse_plug,      mse_aug        — mean (V - truth)^2
  rmse_plug,     rmse_aug       — sqrt(mse)
  ci_width_plug, ci_width_aug   — mean CI width

Default config: 10 seeds × 2 alphas × 5 policies × B=200 (~30 s/seed local).
For the headline 50-seed sweep, pass --seeds 50 --B 500 (~25 min local).

Usage:
    python -m cvar_v4.healthbench_data.analyses.variance_sweep
    python -m cvar_v4.healthbench_data.analyses.variance_sweep --seeds 50 --B 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..policies import POLICIES
from ...eda.deeper._estimator import pipeline_bootstrap_cvar
from ._common import WRITEUP_DATA_DIR, LABELS, policy_pairs, cvar_alpha
from .variance_breakdown import _assemble_cell, _master_indices


def _full_truth(policy: str, alpha: float) -> float:
    _, _, y = policy_pairs(policy)
    return float(cvar_alpha(y, alpha))


def compute(
    *,
    n_seeds: int = 10,
    coverage: float = 0.25,
    alphas: tuple[float, ...] = (0.10, 0.20),
    B: int = 200,
    seed_start: int = 0,
    verbose: bool = False,
) -> dict:
    truths = {p.name: {a: _full_truth(p.name, a) for a in alphas} for p in POLICIES}
    raw: list[dict] = []
    t0 = time.time()
    for s in range(seed_start, seed_start + n_seeds):
        for alpha in alphas:
            try:
                log, per_policy = _assemble_cell(coverage, alpha, s)
            except Exception as e:
                if verbose:
                    print(f"  seed={s} alpha={alpha}: assemble failed: {e}")
                continue
            for p in POLICIES:
                if p.name not in per_policy:
                    continue
                pol = per_policy[p.name]
                if pol["n_audit"] < 3:
                    continue
                master = _master_indices(
                    n_train=int(len(log["s_train"])), n_eval=pol["n_eval"],
                    n_audit=pol["n_audit"], B=B, seed=s,
                )
                full = pipeline_bootstrap_cvar(
                    s_train=log["s_train"], y_train=log["y_train"],
                    s_eval_full=pol["s_eval_full"],
                    s_audit=pol["s_audit"], y_audit=pol["y_audit"],
                    alpha=alpha,
                    sample_weight_train=log["w_train"],
                    sample_weight_audit=pol["w_audit"],
                    resample=("train", "eval", "audit"),
                    B=B, seed=s,
                    idx_train_per_b=master["boot"]["train"],
                    idx_eval_per_b=master["boot"]["eval"],
                    idx_audit_per_b=master["boot"]["audit"],
                )
                raw.append({
                    "seed": int(s), "policy": p.name, "alpha": float(alpha),
                    "n_audit": pol["n_audit"],
                    "v_plug": float(full["plug_point"]),
                    "v_aug": float(full["aug_point"]),
                    "ci_lo_plug": float(full["ci_plug"][0]),
                    "ci_hi_plug": float(full["ci_plug"][1]),
                    "ci_lo_aug": float(full["ci_aug"][0]),
                    "ci_hi_aug": float(full["ci_aug"][1]),
                    "truth": float(truths[p.name][alpha]),
                    "t_hat": float(full["t_hat_point"]),
                    "n_unique_t_hat": int(full["n_unique_t_hat"]),
                })
        if verbose:
            print(f"  seed={s} done in {time.time()-t0:.1f}s elapsed total")

    # Aggregate per (policy, alpha)
    summary: list[dict] = []
    for p in POLICIES:
        for alpha in alphas:
            cells = [r for r in raw if r["policy"] == p.name and r["alpha"] == alpha]
            if not cells:
                continue
            v_plug = np.array([c["v_plug"] for c in cells])
            v_aug = np.array([c["v_aug"] for c in cells])
            truth = np.array([c["truth"] for c in cells])
            cov_plug = np.mean([
                c["ci_lo_plug"] <= c["truth"] <= c["ci_hi_plug"] for c in cells
            ])
            cov_aug = np.mean([
                c["ci_lo_aug"] <= c["truth"] <= c["ci_hi_aug"] for c in cells
            ])
            width_plug = float(np.mean([c["ci_hi_plug"] - c["ci_lo_plug"] for c in cells]))
            width_aug = float(np.mean([c["ci_hi_aug"] - c["ci_lo_aug"] for c in cells]))
            n_unique = float(np.mean([c["n_unique_t_hat"] for c in cells]))
            summary.append({
                "policy": p.name, "alpha": float(alpha),
                "n_seeds": int(len(cells)),
                "n_audit_avg": float(np.mean([c["n_audit"] for c in cells])),
                "truth": float(truth.mean()),  # truth is constant across seeds
                "bias_plug": float((v_plug - truth).mean()),
                "bias_aug":  float((v_aug - truth).mean()),
                "mse_plug": float(((v_plug - truth) ** 2).mean()),
                "mse_aug":  float(((v_aug - truth) ** 2).mean()),
                "rmse_plug": float(np.sqrt(((v_plug - truth) ** 2).mean())),
                "rmse_aug":  float(np.sqrt(((v_aug - truth) ** 2).mean())),
                "coverage_plug": float(cov_plug),
                "coverage_aug":  float(cov_aug),
                "ci_width_plug": width_plug,
                "ci_width_aug":  width_aug,
                "n_unique_t_hat_avg": n_unique,
            })
    return {
        "meta": {
            "n_seeds": int(n_seeds), "seed_start": int(seed_start),
            "coverage": float(coverage), "alphas": list(alphas), "B": int(B),
            "wall_seconds": float(time.time() - t0),
        },
        "summary": summary,
        "raw": raw,
    }


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    rows = payload["summary"]
    lines = []
    lines.append("# Multi-seed sweep — plug-in vs augmented Direct CVaR-CJE")
    lines.append("")
    lines.append(
        f"n_seeds={meta['n_seeds']} (start={meta['seed_start']})  coverage={meta['coverage']}  "
        f"B={meta['B']}  wall={meta['wall_seconds']:.1f}s"
    )
    lines.append("")
    lines.append("Coverage at 95% nominal (closer to 0.95 = better calibrated). "
                 "MSE = E[(V̂ - truth)^2]; lower is better. "
                 "Bias = E[V̂ - truth].")
    lines.append("")

    by_alpha: dict[float, list[dict]] = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], []).append(r)

    for alpha in sorted(by_alpha):
        lines.append(f"## α = {alpha}")
        lines.append("")
        lines.append(
            "| policy | n_au | truth | bias_plug | bias_aug | RMSE_plug | RMSE_aug "
            "| MSE_plug | MSE_aug | cov_plug | cov_aug | width_plug | width_aug | uniq t̂ |"
        )
        lines.append("|" + "|".join(["---"] * 14) + "|")
        for pol in [p.name for p in POLICIES]:
            r = next((x for x in by_alpha[alpha] if x["policy"] == pol), None)
            if r is None:
                continue
            lines.append(
                f"| {LABELS.get(pol, pol)} | {r['n_audit_avg']:.0f} "
                f"| {r['truth']:+.3f} "
                f"| {r['bias_plug']:+.4f} | {r['bias_aug']:+.4f} "
                f"| {r['rmse_plug']:.4f} | {r['rmse_aug']:.4f} "
                f"| {r['mse_plug']:.5f} | {r['mse_aug']:.5f} "
                f"| {r['coverage_plug']:.2f} | {r['coverage_aug']:.2f} "
                f"| {r['ci_width_plug']:.3f} | {r['ci_width_aug']:.3f} "
                f"| {r['n_unique_t_hat_avg']:.0f} |"
            )
        lines.append("")

        # Verdicts per cell
        lines.append("**Verdicts (this α):**")
        for r in by_alpha[alpha]:
            mse_winner = "plug" if r["mse_plug"] < r["mse_aug"] else "aug"
            mse_gap = abs(r["mse_plug"] - r["mse_aug"]) / max(r["mse_plug"], r["mse_aug"], 1e-12) * 100
            cov_plug_dev = abs(r["coverage_plug"] - 0.95)
            cov_aug_dev = abs(r["coverage_aug"] - 0.95)
            cov_winner = "plug" if cov_plug_dev < cov_aug_dev else "aug"
            lines.append(
                f"- {LABELS.get(r['policy'], r['policy'])} (n_au={r['n_audit_avg']:.0f}): "
                f"MSE winner = **{mse_winner}** (gap {mse_gap:.1f}%), "
                f"coverage closer to 0.95 = **{cov_winner}**"
            )
        lines.append("")

    # Hypothesis evaluation
    lines.append("## Hypothesis check")
    lines.append("")
    lines.append("Inspector hypothesis: plug-in wins MSE at small n_audit (base, n_au≈26); "
                 "augmented wins or ties at larger n_audit (others, n_au≈132).")
    lines.append("")
    base_results = [r for r in rows if r["policy"] == "base"]
    other_results = [r for r in rows if r["policy"] != "base"]
    if base_results:
        plug_wins_base = sum(1 for r in base_results if r["mse_plug"] < r["mse_aug"])
        lines.append(f"- base (n_au≈26): plug-in wins MSE in {plug_wins_base}/{len(base_results)} α-cells.")
    if other_results:
        plug_wins_other = sum(1 for r in other_results if r["mse_plug"] < r["mse_aug"])
        aug_wins_other = len(other_results) - plug_wins_other
        lines.append(f"- non-base (n_au≈132): plug-in wins MSE in {plug_wins_other}/{len(other_results)} cells; "
                     f"aug wins in {aug_wins_other}/{len(other_results)}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds (default 10)")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--alphas", type=str, default="0.10,0.20")
    ap.add_argument("--B", type=int, default=200, help="Bootstrap reps per seed (default 200)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(",") if x.strip())

    payload = compute(
        n_seeds=args.seeds, coverage=args.coverage, alphas=alphas,
        B=args.B, seed_start=args.seed_start, verbose=not args.quiet,
    )

    out_json = WRITEUP_DATA_DIR / "variance_sweep.json"
    out_md = WRITEUP_DATA_DIR / "variance_sweep.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    out_md.write_text(render_markdown(payload))
    print(f"[variance_sweep] {len(payload['raw'])} cells -> {out_md}")


if __name__ == "__main__":
    main()
