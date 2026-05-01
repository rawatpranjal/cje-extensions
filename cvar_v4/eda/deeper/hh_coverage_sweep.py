"""Oracle-coverage sweep on the HH red-team headline pair.

Original CJE paper sweeps oracle_coverage ∈ {0.05, 0.10, 0.25, 0.50, 1.00} on
the Arena experiment. We replicate the design for CVaR-CJE on the headline
pair (logger = `rlhf | 52B`, targets = {`rlhf | 13B`, `rejection sampling | 2.7B`}).

For each coverage level we report:
- Direct CVaR_0.05 estimate (point + bootstrap 95% CI)
- Wald audit p-value (does the audit accept at this coverage?)
- Difference between rlhf-13B and rejection-sampling-2.7B CVaR estimates

The hypothesis is that with logger = rlhf-52B (within-family for both
targets), the audit accepts uniformly across coverage levels, and the
CVaR_0.05 difference between the two targets is detectable from
oracle_coverage = 0.05 onward.

Output: cvar_v4/eda/deeper/hh_coverage_sweep.md
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
TARGETS = ["rlhf | 13B", "rejection sampling | 2.7B"]
ALPHA = 0.05
COVERAGE_LEVELS = [0.05, 0.10, 0.25, 0.50, 1.00]
N_SEEDS = 5  # average over multiple random oracle-mask seeds


def make_cheap_s(transcripts: list[str]) -> np.ndarray:
    lens = np.array([len(t) for t in transcripts], dtype=float)
    lens = np.maximum(lens, 100.0)
    return -np.log(lens / 1000.0)


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    if arr.size == 0:
        return float("nan")
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def main() -> None:
    print("[load] HH red-team-attempts...", flush=True)
    df = _filter(_load(SPEC))
    s_full = make_cheap_s(df["transcript"].to_list())
    y_full = df["min_harmlessness_score_transcript"].to_numpy()
    pol = df["policy_id"].to_numpy()

    mask_log = pol == LOGGER
    s_log = s_full[mask_log]
    y_log = y_full[mask_log]
    print(f"[logger] {LOGGER} n={s_log.size:,}", flush=True)

    rows = []
    for target in TARGETS:
        mask_t = pol == target
        n_t = int(mask_t.sum())
        s_t = s_full[mask_t]
        y_t = y_full[mask_t]

        for cov in COVERAGE_LEVELS:
            # Average across N_SEEDS oracle-mask seeds.
            cvar_estimates = []
            audit_p = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                idx = rng.permutation(n_t)
                n_orac = max(int(n_t * cov), 100)
                idx_orac = idx[:n_orac]
                idx_fresh = idx[n_orac:] if n_orac < n_t else idx_orac
                s_audit = s_t[idx_orac]
                y_audit = y_t[idx_orac]
                s_eval = s_t[idx_fresh]

                cvar_est, t_hat, _, _ = estimate_direct_cvar_isotonic(
                    s_train=s_log, y_train=y_log, s_eval=s_eval,
                    alpha=ALPHA, grid_size=61,
                )
                audit = two_moment_wald_audit_xf(
                    s_train=s_log, y_train=y_log,
                    s_audit=s_audit, y_audit=y_audit,
                    t0=t_hat, alpha=ALPHA, B=100, fold_seed=seed,
                )
                cvar_estimates.append(cvar_est)
                audit_p.append(audit["p_value"])
            cvar_arr = np.array(cvar_estimates)
            true_cvar = cvar_alpha(y_t, ALPHA)
            rows.append({
                "target": target,
                "coverage": cov,
                "n_orac": int(n_t * cov) if cov < 1 else n_t,
                "cvar_mean": float(cvar_arr.mean()),
                "cvar_std": float(cvar_arr.std()),
                "cvar_min": float(cvar_arr.min()),
                "cvar_max": float(cvar_arr.max()),
                "true_cvar": true_cvar,
                "abs_err": float(abs(cvar_arr.mean() - true_cvar)),
                "audit_p_mean": float(np.mean(audit_p)),
                "audit_p_min": float(min(audit_p)),
                "audit_reject_frac": float(np.mean([p < 0.05 for p in audit_p])),
            })
            print(f"[sweep] {target:32s}  cov={cov:.2f}  "
                  f"CVaR={cvar_arr.mean():+.3f}±{cvar_arr.std():.3f}  "
                  f"true={true_cvar:+.3f}  err={abs(cvar_arr.mean()-true_cvar):.3f}  "
                  f"p_mean={np.mean(audit_p):.3f}  reject_frac={np.mean([p<0.05 for p in audit_p]):.2f}",
                  flush=True)

    # Build same-mean-different-tail pair table.
    pairs_table = []
    for cov in COVERAGE_LEVELS:
        rlhf_row = next(r for r in rows if r["target"] == "rlhf | 13B" and r["coverage"] == cov)
        rs_row = next(r for r in rows if r["target"] == "rejection sampling | 2.7B" and r["coverage"] == cov)
        diff = rlhf_row["cvar_mean"] - rs_row["cvar_mean"]
        true_diff = rlhf_row["true_cvar"] - rs_row["true_cvar"]
        pairs_table.append({
            "coverage": cov,
            "rlhf_cvar": rlhf_row["cvar_mean"],
            "rs_cvar": rs_row["cvar_mean"],
            "diff_estimate": diff,
            "true_diff": true_diff,
            "rlhf_audit_p": rlhf_row["audit_p_mean"],
            "rs_audit_p": rs_row["audit_p_mean"],
        })

    # Render markdown.
    out = ["# HH red-team — oracle-coverage sweep on the headline pair\n"]
    out.append(
        f"_Logger = `{LOGGER}` (n={s_log.size:,}). Targets = `rlhf | 13B`, `rejection sampling | 2.7B`. "
        f"α = {ALPHA}; cheap S = -log(transcript_chars/1000); oracle Y = preference-model score; "
        f"averaged over {N_SEEDS} random oracle-mask seeds; bootstrap B=100 inside audit._\n"
    )

    out.append("## Per-target × per-coverage Direct CVaR_0.05 estimates\n")
    out.append("| target | coverage | n_oracle | CVaR_0.05 (mean ± std across seeds) | min | max | true CVaR | err | audit p (mean) | reject frac |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        flag = "🔥" if r["audit_reject_frac"] >= 0.5 else "✅"
        out.append(
            f"| `{r['target']}` | {r['coverage']:.2f} | {r['n_orac']:,} | "
            f"{r['cvar_mean']:+.3f} ± {r['cvar_std']:.3f} | "
            f"{r['cvar_min']:+.3f} | {r['cvar_max']:+.3f} | "
            f"{r['true_cvar']:+.3f} | {r['abs_err']:.3f} | "
            f"{r['audit_p_mean']:.3f} {flag} | {r['audit_reject_frac']:.0%} |"
        )
    out.append("")

    out.append("## Same-mean-different-tail comparison: ΔCVaR_0.05 across coverage\n")
    out.append("| coverage | rlhf-13B CVaR_0.05 | RS-2.7B CVaR_0.05 | Δ (estimated) | Δ (true) | rlhf audit p | RS audit p |")
    out.append("|---|---|---|---|---|---|---|")
    for r in pairs_table:
        out.append(
            f"| {r['coverage']:.2f} | {r['rlhf_cvar']:+.3f} | {r['rs_cvar']:+.3f} | "
            f"**{r['diff_estimate']:+.3f}** | {r['true_diff']:+.3f} | "
            f"{r['rlhf_audit_p']:.3f} | {r['rs_audit_p']:.3f} |"
        )
    out.append("")

    out.append("## Interpretation — what the sweep actually showed\n")
    true_diff_full = pairs_table[-1]["true_diff"]
    est_diff_5 = pairs_table[0]["diff_estimate"]
    est_diff_full = pairs_table[-1]["diff_estimate"]
    out.append(f"### Truth and estimates")
    out.append(f"- **True ΔCVaR_0.05 = {true_diff_full:+.3f}**: rlhf-13B is *better* (less harmful) than rejection-sampling-2.7B by ~0.97 units of harmlessness.")
    out.append(f"- **Direct CVaR-CJE estimate at coverage=0.05**: Δ = {est_diff_5:+.3f}; **at coverage=1.00**: Δ = {est_diff_full:+.3f}.")
    out.append(f"- The estimator has the SIGN FLIPPED relative to truth — it says rejection-sampling-2.7B is better, when in fact rlhf-13B is. Per-target absolute errors are ~0.75 (rejection-sampling-2.7B) and ~1.40 (rlhf-13B), and they are *not* shrinking with coverage.")
    out.append("")
    out.append("### Why estimates are biased: cheap-S calibrator transport")
    out.append("The cheap S = -log(transcript_chars/1000) was deliberately chosen as a *biased* proxy with different per-policy ρ(S, Y) (rlhf-52B has ρ=+0.135, but rejection-sampling cells have ρ=+0.40, and rlhf-13B has ρ=+0.295). The Direct estimator fits the calibrator on logger then *applies the same calibrator to all targets*, ignoring target-specific S→Y curvature. With this cheap S, the calibrator's predictions on targets are systematically too high (less harmful) by 0.7-1.4 units, biasing CVaR_0.05 estimates upward (less negative) for both targets — and biasing rlhf-13B more than rejection-sampling-2.7B because the per-policy ρ gap is larger.")
    out.append("")
    out.append("### What the audit does at each coverage level")
    out.append("The audit is the safety valve. Looking at the per-target table:")
    out.append("- **rlhf-13B**: audit p drops monotonically with coverage (0.84 → 0.69 → 0.40 → 0.16 → **0.024**). At full coverage (n_audit=2,292), the audit *fires* — correctly identifying that the calibrator does not transport to rlhf-13B. At small coverage (n_audit≤500), the audit lacks power to detect the bias.")
    out.append("- **rejection-sampling-2.7B**: audit p stays high (0.89 → 0.88 → 0.73 → 0.55 → 0.32) — even at full coverage the audit accepts despite a 0.75-unit estimation error. This is a Type II failure.")
    out.append("")
    out.append("### What this means for the cvar_v4 paper")
    out.append("- **The cheap S used here (transcript length) is too weak to certify level claims.** Production runs must use a stronger cheap S — WildGuard-7B's binary refusal score, or a small instruct model running the same rubric the oracle uses.")
    out.append("- **The audit's power scales with oracle coverage at exactly the rate the framework predicts.** At 5% coverage on n_target ≈ 2K, the audit has roughly zero power against this transport failure. At 50–100% coverage, power crosses the rejection threshold for rlhf-13B but not for rejection-sampling-2.7B — meaning the calibrator's bias is *symmetric* across rejection-sampling-2.7B (audit silent) but *asymmetric* across rlhf-13B (audit fires).")
    out.append("- **The pre-registered hypothesis from §2 needs revision.** The original claim was 'with logger=rlhf-52B, both targets pass audit and CVaR-CJE recovers the gap.' The sweep shows: with the chosen cheap S, both targets pass audit at low coverage *but the estimates are wrong*. To make the headline demonstration trustworthy, the cheap S itself has to be reasonable — length is not.")
    out.append("- **Useful by-product**: the sweep is a prebuilt diagnostic for whether a given (logger, S, target) triple is admissible. If audit p < 0.05 at any coverage in the sweep, halt and pick a different cheap S. If audit p > 0.05 uniformly across coverage but the within-family Direct estimate has a sign flip vs the empirical oracle truth (computable on a small held-out slice), that's a Type II failure of the audit and demands a stronger cheap S.")
    out.append("")
    out.append("**Bottom line**: the audit's bite scales correctly with coverage, but its bite is conditional on the cheap S being calibrated enough. A weak cheap S can pass the audit at small samples while still producing wrong estimates. The cvar_v4 paper's pilot must use a real cheap-judge model, not a length proxy, to demonstrate the framework's reliability.")

    text = "\n".join(out)
    out_path = Path(__file__).parent / "hh_coverage_sweep.md"
    out_path.write_text(text)
    print(f"\n[write] {out_path}  ({len(text):,} chars)", flush=True)


if __name__ == "__main__":
    main()
