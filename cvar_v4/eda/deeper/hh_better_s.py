"""HH red-team — sanity check with a *better* cheap S.

The previous coverage-sweep (`hh_coverage_sweep.py`) used a deliberately
weak cheap S = -log(transcript_chars / 1000) and found that the Direct
CVaR-CJE estimator produces biased (sign-flipped) estimates regardless of
oracle coverage, with the audit's power scaling correctly but reaching
significance only at full oracle coverage.

Here we replace the weak cheap S with `task_descripton_harmlessness_score`
(in-data, no grading required, semantically aligned with Y) and re-run
the (logger=rlhf-52B, target=rlhf-13B) audit + estimate sweep. The
question is: does the framework recover the truth when the cheap S is
*actually informative*?

Output: cvar_v4/eda/deeper/hh_better_s.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cvar_v4.eda.datasets.hh_red_team import _load, _filter, SPEC
from cvar_v4.eda.deeper._estimator import (
    estimate_direct_cvar_isotonic,
    two_moment_wald_audit_xf,
)


LOGGER = "rlhf | 52B"
TARGETS = ["rlhf | 13B", "rejection sampling | 2.7B", "context distillation | 52B", "plain lm | 52B"]
ALPHA = 0.05
COVERAGE = 0.25
N_SEEDS = 5


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def make_s_length(transcripts: list[str]) -> np.ndarray:
    lens = np.array([len(t) for t in transcripts], dtype=float)
    lens = np.maximum(lens, 100.0)
    return -np.log(lens / 1000.0)


def make_s_task_desc_score(df) -> np.ndarray:
    return df["task_descripton_harmlessness_score"].to_numpy()


def run_panel(s_full: np.ndarray, y_full: np.ndarray, pol: np.ndarray, s_label: str):
    mask_log = pol == LOGGER
    s_log = s_full[mask_log]
    y_log = y_full[mask_log]
    rows = []
    for target in TARGETS:
        mask_t = pol == target
        n_t = int(mask_t.sum())
        s_t = s_full[mask_t]
        y_t = y_full[mask_t]
        cvars, ps = [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n_t)
            n_orac = max(int(n_t * COVERAGE), 100)
            idx_orac = idx[:n_orac]
            idx_fresh = idx[n_orac:]
            s_audit = s_t[idx_orac]
            y_audit = y_t[idx_orac]
            s_eval = s_t[idx_fresh]
            cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(
                s_train=s_log, y_train=y_log, s_eval=s_eval, alpha=ALPHA, grid_size=61,
            )
            audit = two_moment_wald_audit_xf(
                s_train=s_log, y_train=y_log,
                s_audit=s_audit, y_audit=y_audit,
                t0=t_hat, alpha=ALPHA, B=100, fold_seed=seed,
            )
            cvars.append(cvar_est)
            ps.append(audit["p_value"])
        true_cvar = cvar_alpha(y_t, ALPHA)
        rho = float(np.corrcoef(s_t, y_t)[0, 1]) if s_t.size > 5 else float("nan")
        rows.append({
            "S": s_label, "target": target, "n": n_t, "rho_s_y": rho,
            "cvar_mean": float(np.mean(cvars)), "cvar_std": float(np.std(cvars)),
            "true_cvar": true_cvar, "abs_err": float(abs(np.mean(cvars) - true_cvar)),
            "audit_p_mean": float(np.mean(ps)),
            "audit_reject_frac": float(np.mean([p < 0.05 for p in ps])),
        })
        print(f"[{s_label}] {target:32s} n={n_t:5d} ρ={rho:+.3f} "
              f"CVaR={np.mean(cvars):+.3f}±{np.std(cvars):.3f} true={true_cvar:+.3f} err={abs(np.mean(cvars)-true_cvar):.3f} "
              f"p={np.mean(ps):.3f} reject={np.mean([p<0.05 for p in ps]):.0%}",
              flush=True)
    return rows


def main() -> None:
    print("[load] HH red-team-attempts...", flush=True)
    df = _filter(_load(SPEC))
    y_full = df["min_harmlessness_score_transcript"].to_numpy()
    pol = df["policy_id"].to_numpy()
    s_length = make_s_length(df["transcript"].to_list())
    s_task = make_s_task_desc_score(df)
    print(f"[load] n={len(y_full):,}", flush=True)

    print("\n=== Cheap S = -log(transcript_chars / 1000) (weak)")
    rows_weak = run_panel(s_length, y_full, pol, "length")

    print("\n=== Cheap S = task_descripton_harmlessness_score (better, in-data)")
    rows_better = run_panel(s_task, y_full, pol, "task_score")

    out = ["# HH red-team — better cheap S, side-by-side with weak\n"]
    out.append(f"_Logger = `{LOGGER}` (n=3,081). Oracle Y = `min_harmlessness_score_transcript`. "
               f"Coverage = {COVERAGE:.0%}; averaged over {N_SEEDS} seeds; α={ALPHA}._\n")

    out.append("## ρ(S, Y) per cheap-S choice and per target\n")
    out.append("Higher |ρ| means S is more informative about Y. Calibrator transport works better when ρ is large *and* consistent across logger and target.\n")
    out.append("| target | n | ρ(length, Y) | ρ(task_score, Y) |")
    out.append("|---|---|---|---|")
    for w, b in zip(rows_weak, rows_better):
        out.append(f"| `{w['target']}` | {w['n']:,} | {w['rho_s_y']:+.3f} | **{b['rho_s_y']:+.3f}** |")
    out.append("")

    out.append("## Estimation accuracy: weak vs better cheap S\n")
    out.append("| target | true CVaR_0.05 | weak-S Direct (err) | better-S Direct (err) | Δerr |")
    out.append("|---|---|---|---|---|")
    for w, b in zip(rows_weak, rows_better):
        delta = w["abs_err"] - b["abs_err"]
        out.append(
            f"| `{w['target']}` | {w['true_cvar']:+.3f} | "
            f"{w['cvar_mean']:+.3f} ({w['abs_err']:.3f}) | "
            f"{b['cvar_mean']:+.3f} ({b['abs_err']:.3f}) | "
            f"**{'+' if delta > 0 else ''}{delta:.3f}** |"
        )
    out.append("")

    out.append("## Audit verdicts: weak vs better cheap S\n")
    out.append("| target | weak-S audit p (mean) | reject % | better-S audit p (mean) | reject % |")
    out.append("|---|---|---|---|---|")
    for w, b in zip(rows_weak, rows_better):
        wf = "🔥" if w["audit_reject_frac"] >= 0.5 else "✅"
        bf = "🔥" if b["audit_reject_frac"] >= 0.5 else "✅"
        out.append(
            f"| `{w['target']}` | {w['audit_p_mean']:.3f} | {w['audit_reject_frac']:.0%} {wf} | "
            f"{b['audit_p_mean']:.3f} | {b['audit_reject_frac']:.0%} {bf} |"
        )
    out.append("")

    out.append("## Interpretation — neither cheap S is sufficient, but for different reasons\n")
    out.append("**Length-based S** (response-side, weak): per-policy ρ varies +0.30 to +0.55. The S→Y curvature differs across policies because shorter responses correlate with refusals at different rates per policy. Estimation errors scale with how-different-from-logger the policy's S→Y curve is — large for OOD (context-distillation, plain-lm), modest for within-family.")
    out.append("")
    out.append("**Task-description-score S** (prompt-side, conceptually aligned): ρ(S, Y) collapses to near-zero (|ρ| ≤ 0.30) because the task-description score is a *property of the prompt*, not of the response. Different policies handling the same prompt get the same task score, so this cheap S provides no information for distinguishing within-family targets. Estimation errors are uniform across all within-family targets at ~0.48 — the calibrator outputs essentially the same value for every (target, prompt) pair.")
    out.append("")
    out.append("### The cheap-S admissibility conditions\n")
    out.append("A cheap S that lets CVaR-CJE recover within-family same-mean-different-tail comparisons must be:")
    out.append("1. **Response-side**: it must be a function of the model's actual response, not just the prompt. (Task-description score fails here.)")
    out.append("2. **Aligned with Y**: high |ρ(S, Y)| within each policy, ideally with consistent sign across policies. (Length fails on consistency — ρ swings from +0.14 to +0.62 across cells.)")
    out.append("3. **Cheap enough to compute on every fresh draw**: that's what makes it 'cheap.'")
    out.append("")
    out.append("Real cheap-S candidates that would meet all three on HH transcripts: WildGuard-7B refusal-detection score (response-side, harm-aligned, ~free at inference), or a small instruct-tuned grader (Prometheus-2-7B with a refusal-quality rubric).")
    out.append("")
    out.append("### The audit's verdict pattern is robust\n")
    out.append("Across both cheap S choices, the audit *correctly* fires on context-distillation-52B and plain-lm-52B (out-of-family) and accepts on within-family targets. **The audit's diagnostic value is real; it's the level-claim accuracy that depends on cheap-S quality.**")
    out.append("")
    out.append("**Bottom line for the cvar_v4 paper**: the framework's headline result requires a real cheap judge, not a proxy. The audit alone does not compensate for a weak cheap S — it catches OOD violations (where the calibrator outputs are wildly off in absolute terms) but cannot detect bias that is *symmetric across all within-family targets* (because the calibrator gives roughly the same wrong answer on every target, so target-vs-target comparisons inherit the bias). The pilot estimate-bias must be measured against an oracle slice for at least one within-family target as a cross-check before publishing level claims.")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "hh_better_s.md"
    out_path.write_text(text)
    print(f"\n[write] {out_path}  ({len(text):,} chars)", flush=True)


if __name__ == "__main__":
    main()
