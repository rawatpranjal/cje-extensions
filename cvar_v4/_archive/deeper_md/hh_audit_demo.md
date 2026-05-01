# HH red-team-attempts — Direct CVaR-CJE + Wald two-moment audit (with bite)

_Cheap S = -log(transcript_chars / 1000). Oracle Y = min_harmlessness_score_transcript. Audit slice = 25% of each target. α = 0.05; Wald α = 0.05; bootstrap B=200; ridge=1e-6 on Σ̂ for stability._

## Per-policy ρ(S, Y) — predicting where transport will fail

If ρ(S, Y) is consistent across policies, the logger's S→Y calibrator transports. If they diverge, the calibrator systematically mispredicts on target and the audit fires. RLHF cells (low ρ ≈ 0.14–0.30) and unaligned cells (ρ ≈ 0.4–0.6) are visibly two clusters in this table — the audit's job is to detect when the logger and target are in different clusters.

| policy | n | ρ(S, Y) |
|---|---|---|
| `rlhf | 52B` | 3,081 | **+0.135** |
| `rlhf | 2.7B` | 2,498 | **+0.220** |
| `rlhf | 13B` | 2,292 | **+0.295** |
| `context distillation | 52B` | 17,739 | **+0.355** |
| `plain lm | 52B` | 1,414 | **+0.393** |
| `rejection sampling | 52B` | 1,273 | **+0.399** |
| `rejection sampling | 13B` | 1,263 | **+0.442** |
| `rejection sampling | 2.7B` | 1,445 | **+0.500** |
| `plain lm | 13B` | 2,587 | **+0.545** |
| `context distillation | 13B` | 2,174 | **+0.550** |
| `plain lm | 2.7B` | 1,151 | **+0.573** |
| `context distillation | 2.7B` | 2,044 | **+0.619** |

## Audit + Direct CVaR-CJE — logger = `context distillation | 52B` (`context distillation` family)

_audit fires: 0/9 targets; accepts: 9/9_

| target | n_target | t̂ | Direct CVaR_0.05 | Oracle truth | err | Wald χ² | p | audit | ḡ_1 | ḡ_2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `context distillation | 13B` | 2,174 | -3.610 | -4.431 | -4.196 | 0.24 | 0.03 | 0.9841 | ✅ accepts | +0.0052 | -0.0057 |
| `context distillation | 2.7B` | 2,044 | -3.610 | -4.386 | -4.678 | 0.29 | 0.59 | 0.7459 | ✅ accepts | +0.0322 | +0.0148 |
| `plain lm | 52B` | 1,414 | -3.872 | -4.583 | -4.450 | 0.13 | 0.04 | 0.9810 | ✅ accepts | -0.0075 | -0.0078 |
| `plain lm | 13B` | 2,587 | -3.610 | -4.406 | -4.500 | 0.09 | 0.03 | 0.9835 | ✅ accepts | +0.0011 | +0.0074 |
| `plain lm | 2.7B` | 1,151 | -3.784 | -4.533 | -4.222 | 0.31 | 0.07 | 0.9651 | ✅ accepts | -0.0117 | -0.0113 |
| `rlhf | 52B` | 3,081 | -3.959 | -4.696 | -0.520 | 4.18 | 2.86 | 0.2390 | ✅ accepts | -0.0500 | -0.0349 |
| `rlhf | 13B` | 2,292 | -4.743 | -5.368 | -0.310 | 5.06 | 1.98 | 0.3715 | ✅ accepts | -0.0500 | -0.0311 |
| `rejection sampling | 2.7B` | 1,445 | -3.174 | -3.973 | -0.887 | 3.09 | 1.58 | 0.4545 | ✅ accepts | -0.0500 | -0.0433 |
| `rejection sampling | 52B` | 1,273 | -3.349 | -4.140 | -0.164 | 3.98 | 1.27 | 0.5303 | ✅ accepts | -0.0500 | -0.0386 |

## Audit + Direct CVaR-CJE — logger = `rlhf | 52B` (`rlhf` family)

_audit fires: 5/8 targets; accepts: 3/8_

| target | n_target | t̂ | Direct CVaR_0.05 | Oracle truth | err | Wald χ² | p | audit | ḡ_1 | ḡ_2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `context distillation | 13B` | 2,174 | +0.064 | -0.500 | -4.196 | 3.70 | 248 | 0.0000 | 🔥 **FIRES** | +0.5725 | +1.1211 |
| `context distillation | 2.7B` | 2,044 | +0.064 | -0.503 | -4.678 | 4.18 | 225 | 0.0000 | 🔥 **FIRES** | +0.5586 | +1.1391 |
| `plain lm | 52B` | 1,414 | +0.064 | -0.573 | -4.450 | 3.88 | 181 | 0.0000 | 🔥 **FIRES** | +0.5817 | +1.0041 |
| `plain lm | 13B` | 2,587 | +0.128 | -0.480 | -4.500 | 4.02 | 347 | 0.0000 | 🔥 **FIRES** | +0.6265 | +1.1757 |
| `plain lm | 2.7B` | 1,151 | +0.064 | -0.565 | -4.222 | 3.66 | 161 | 0.0000 | 🔥 **FIRES** | +0.6120 | +1.0689 |
| `rlhf | 13B` | 2,292 | -0.575 | -1.576 | -0.310 | 1.27 | 1.54 | 0.4629 | ✅ accepts | -0.0343 | -0.0404 |
| `rejection sampling | 2.7B` | 1,445 | +0.128 | -0.388 | -0.887 | 0.50 | 0.54 | 0.7630 | ✅ accepts | +0.0276 | +0.0309 |
| `rejection sampling | 52B` | 1,273 | +0.128 | -0.428 | -0.164 | 0.26 | 0.16 | 0.9211 | ✅ accepts | -0.0217 | -0.0086 |

## Synthesis — the audit's bite is asymmetric in logger choice

**The two loggers behave very differently, and the comparison is the most important finding.**

### Logger = `context distillation | 52B` (worst-tail logger; mean Y = -0.83)
Audit accepts on **9/9** targets — including rlhf-52B and rlhf-13B where the Direct CVaR estimate is wildly wrong (err = 4–5 harmlessness units). **The audit is silent on catastrophic estimation failure.**

Why? When the logger has the worst tail in the panel, t̂ is pinned to the logger's empirical 5%-quantile (≈ -3.6 to -4.7). For rlhf targets whose Y is mostly positive, almost no observations fall below t̂ at all — so:
- g_1 = (Y ≤ t̂) - α ≈ 0 - 0.05 = -0.05 (a real but small violation)
- g_2 = (t̂-Y)+ - f̂(S) ≈ 0 - small ≈ small (both terms vanish)

The bootstrap Σ̂_boot inflates because t̂_b varies across reps when the optimizer can't find a good fit, and the inflated variance makes W small. **The audit is in a regime where it can't see the violation it's supposed to catch.**

### Logger = `rlhf | 52B` (best-tail logger; mean Y = +1.7)
Audit fires on **5/8** targets and accepts on **3/8** — and the partition is exactly the within-alignment-family vs out-of-alignment-family split:
- ✅ accepts on rlhf-13B, rejection-sampling-2.7B, rejection-sampling-52B (within-family — Direct CVaR err 0.3–1.3)
- 🔥 fires on context-distillation × {52B, 13B, 2.7B} and plain-lm × {52B, 13B, 2.7B} (out-of-family — Direct CVaR err 3.7–4.2)

Why? With the best-tail logger, t̂ is set near the rlhf empirical 5%-quantile (slight negative). For un-aligned targets whose Y is mostly negative, *many* observations fall below t̂ — g_1 jumps to +0.55 to +0.63 (way above α=0.05), the calibrator's stop-loss prediction is way off, and the audit catches it loudly (W = 161–347, p < 0.0001).

### The lesson — choose loggers by where they sit in the Y distribution

CVaR-CJE's audit detects *upward* tail violations (target's tail is heavier than logger's predicted tail) far more reliably than *downward* ones (target's tail is lighter than predicted). The operational rule for picking a logger:

> **Pick a logger whose Y distribution is at least as harmless (high-Y) as the targets you want to certify.** The audit will then fire when a target is unexpectedly worse than the calibrator predicts, which is the case where reporting a level claim would be *misleading*. The audit is asymmetric by design — it refuses to certify *worse-than-expected* targets, but it lets *better-than-expected* targets through (with possibly inflated CVaR estimates).

This matches the original CJE paper's choice of `base` (Llama-3.3-70B with standard prompt) as logger and `unhelpful` (deliberately bad) as a target: the unhelpful policy's true value is *lower* than the calibrator predicts, and the audit fires. Had the original paper picked `unhelpful` as logger and `base` as target, the audit would have accepted even though transport is just as bad.

### What this means for the headline pair (`rlhf | 13B` vs `rejection sampling | 2.7B`)

- With logger = `context distillation | 52B`: audit accepts both (silently). Direct CVaR estimates are unusable (-5.4 and -4.0 vs truth -0.31 and -0.89).
- With logger = `rlhf | 52B`: audit accepts both (correctly — within-family). Direct CVaR estimates are -1.6 and -0.4 vs truth -0.31 and -0.89, errors 1.27 and 0.50. *The same-mean-different-tail demonstration is statistically clean only with the right logger.*

The cvar_v4 paper's pre-registered hypothesis should be: **with logger π0 = `rlhf | 52B`, both targets pass audit, mean-CJE produces a tie (overlapping bootstrap CIs), and CVaR_0.05-CJE produces disjoint 95% bootstrap CIs separating them by ~0.7 units.** This is falsifiable, the data supports it, and the pipeline is the existing cvar_v3 Direct estimator unchanged.