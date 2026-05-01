# HH red-team — oracle-coverage sweep on the headline pair

_Logger = `rlhf | 52B` (n=3,081). Targets = `rlhf | 13B`, `rejection sampling | 2.7B`. α = 0.05; cheap S = -log(transcript_chars/1000); oracle Y = preference-model score; averaged over 5 random oracle-mask seeds; bootstrap B=100 inside audit._

## Per-target × per-coverage Direct CVaR_0.05 estimates

| target | coverage | n_oracle | CVaR_0.05 (mean ± std across seeds) | min | max | true CVaR | err | audit p (mean) | reject frac |
|---|---|---|---|---|---|---|---|---|---|
| `rlhf | 13B` | 0.05 | 114 | -1.574 ± 0.005 | -1.579 | -1.565 | -0.175 | 1.399 | 0.838 ✅ | 0% |
| `rlhf | 13B` | 0.10 | 229 | -1.573 ± 0.008 | -1.586 | -1.564 | -0.175 | 1.398 | 0.689 ✅ | 0% |
| `rlhf | 13B` | 0.25 | 573 | -1.579 ± 0.008 | -1.593 | -1.567 | -0.175 | 1.404 | 0.397 ✅ | 0% |
| `rlhf | 13B` | 0.50 | 1,146 | -1.585 ± 0.017 | -1.614 | -1.563 | -0.175 | 1.410 | 0.155 ✅ | 0% |
| `rlhf | 13B` | 1.00 | 2,292 | -1.570 ± 0.000 | -1.570 | -1.570 | -0.175 | 1.395 | 0.024 🔥 | 100% |
| `rejection sampling | 2.7B` | 0.05 | 72 | -0.392 ± 0.004 | -0.395 | -0.385 | -1.142 | 0.749 | 0.892 ✅ | 0% |
| `rejection sampling | 2.7B` | 0.10 | 144 | -0.393 ± 0.004 | -0.396 | -0.384 | -1.142 | 0.749 | 0.875 ✅ | 0% |
| `rejection sampling | 2.7B` | 0.25 | 361 | -0.394 ± 0.003 | -0.398 | -0.389 | -1.142 | 0.748 | 0.728 ✅ | 0% |
| `rejection sampling | 2.7B` | 0.50 | 722 | -0.384 ± 0.010 | -0.397 | -0.369 | -1.142 | 0.757 | 0.552 ✅ | 0% |
| `rejection sampling | 2.7B` | 1.00 | 1,445 | -0.391 ± 0.000 | -0.391 | -0.391 | -1.142 | 0.750 | 0.318 ✅ | 0% |

## Same-mean-different-tail comparison: ΔCVaR_0.05 across coverage

| coverage | rlhf-13B CVaR_0.05 | RS-2.7B CVaR_0.05 | Δ (estimated) | Δ (true) | rlhf audit p | RS audit p |
|---|---|---|---|---|---|---|
| 0.05 | -1.574 | -0.392 | **-1.181** | +0.967 | 0.838 | 0.892 |
| 0.10 | -1.573 | -0.393 | **-1.181** | +0.967 | 0.689 | 0.875 |
| 0.25 | -1.579 | -0.394 | **-1.185** | +0.967 | 0.397 | 0.728 |
| 0.50 | -1.585 | -0.384 | **-1.201** | +0.967 | 0.155 | 0.552 |
| 1.00 | -1.570 | -0.391 | **-1.179** | +0.967 | 0.024 | 0.318 |

## Interpretation — what the sweep actually showed

### Truth and estimates
- **True ΔCVaR_0.05 = +0.967**: rlhf-13B is *better* (less harmful) than rejection-sampling-2.7B by ~0.97 units of harmlessness.
- **Direct CVaR-CJE estimate at coverage=0.05**: Δ = -1.181; **at coverage=1.00**: Δ = -1.179.
- The estimator has the SIGN FLIPPED relative to truth — it says rejection-sampling-2.7B is better, when in fact rlhf-13B is. Per-target absolute errors are ~0.75 (rejection-sampling-2.7B) and ~1.40 (rlhf-13B), and they are *not* shrinking with coverage.

### Why estimates are biased: cheap-S calibrator transport
The cheap S = -log(transcript_chars/1000) was deliberately chosen as a *biased* proxy with different per-policy ρ(S, Y) (rlhf-52B has ρ=+0.135, but rejection-sampling cells have ρ=+0.40, and rlhf-13B has ρ=+0.295). The Direct estimator fits the calibrator on logger then *applies the same calibrator to all targets*, ignoring target-specific S→Y curvature. With this cheap S, the calibrator's predictions on targets are systematically too high (less harmful) by 0.7-1.4 units, biasing CVaR_0.05 estimates upward (less negative) for both targets — and biasing rlhf-13B more than rejection-sampling-2.7B because the per-policy ρ gap is larger.

### What the audit does at each coverage level
The audit is the safety valve. Looking at the per-target table:
- **rlhf-13B**: audit p drops monotonically with coverage (0.84 → 0.69 → 0.40 → 0.16 → **0.024**). At full coverage (n_audit=2,292), the audit *fires* — correctly identifying that the calibrator does not transport to rlhf-13B. At small coverage (n_audit≤500), the audit lacks power to detect the bias.
- **rejection-sampling-2.7B**: audit p stays high (0.89 → 0.88 → 0.73 → 0.55 → 0.32) — even at full coverage the audit accepts despite a 0.75-unit estimation error. This is a Type II failure.

### What this means for the cvar_v4 paper
- **The cheap S used here (transcript length) is too weak to certify level claims.** Production runs must use a stronger cheap S — WildGuard-7B's binary refusal score, or a small instruct model running the same rubric the oracle uses.
- **The audit's power scales with oracle coverage at exactly the rate the framework predicts.** At 5% coverage on n_target ≈ 2K, the audit has roughly zero power against this transport failure. At 50–100% coverage, power crosses the rejection threshold for rlhf-13B but not for rejection-sampling-2.7B — meaning the calibrator's bias is *symmetric* across rejection-sampling-2.7B (audit silent) but *asymmetric* across rlhf-13B (audit fires).
- **The pre-registered hypothesis from §2 needs revision.** The original claim was 'with logger=rlhf-52B, both targets pass audit and CVaR-CJE recovers the gap.' The sweep shows: with the chosen cheap S, both targets pass audit at low coverage *but the estimates are wrong*. To make the headline demonstration trustworthy, the cheap S itself has to be reasonable — length is not.
- **Useful by-product**: the sweep is a prebuilt diagnostic for whether a given (logger, S, target) triple is admissible. If audit p < 0.05 at any coverage in the sweep, halt and pick a different cheap S. If audit p > 0.05 uniformly across coverage but the within-family Direct estimate has a sign flip vs the empirical oracle truth (computable on a small held-out slice), that's a Type II failure of the audit and demands a stronger cheap S.

**Bottom line**: the audit's bite scales correctly with coverage, but its bite is conditional on the cheap S being calibrated enough. A weak cheap S can pass the audit at small samples while still producing wrong estimates. The cvar_v4 paper's pilot must use a real cheap-judge model, not a length proxy, to demonstrate the framework's reliability.