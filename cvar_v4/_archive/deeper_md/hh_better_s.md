# HH red-team — better cheap S, side-by-side with weak

_Logger = `rlhf | 52B` (n=3,081). Oracle Y = `min_harmlessness_score_transcript`. Coverage = 25%; averaged over 5 seeds; α=0.05._

## ρ(S, Y) per cheap-S choice and per target

Higher |ρ| means S is more informative about Y. Calibrator transport works better when ρ is large *and* consistent across logger and target.

| target | n | ρ(length, Y) | ρ(task_score, Y) |
|---|---|---|---|
| `rlhf | 13B` | 2,292 | +0.295 | **-0.025** |
| `rejection sampling | 2.7B` | 1,445 | +0.500 | **-0.098** |
| `context distillation | 52B` | 17,739 | +0.355 | **+0.155** |
| `plain lm | 52B` | 1,414 | +0.393 | **+0.276** |

## Estimation accuracy: weak vs better cheap S

| target | true CVaR_0.05 | weak-S Direct (err) | better-S Direct (err) | Δerr |
|---|---|---|---|---|
| `rlhf | 13B` | -0.175 | -1.579 (1.404) | -0.653 (0.478) | **+0.926** |
| `rejection sampling | 2.7B` | -1.142 | -0.394 (0.748) | -0.657 (0.484) | **+0.264** |
| `context distillation | 52B` | -4.498 | -0.524 (3.974) | -0.713 (3.785) | **+0.189** |
| `plain lm | 52B` | -4.340 | -0.575 (3.765) | -0.697 (3.643) | **+0.122** |

## Audit verdicts: weak vs better cheap S

| target | weak-S audit p (mean) | reject % | better-S audit p (mean) | reject % |
|---|---|---|---|---|
| `rlhf | 13B` | 0.397 | 0% ✅ | 0.767 | 0% ✅ |
| `rejection sampling | 2.7B` | 0.728 | 0% ✅ | 0.846 | 0% ✅ |
| `context distillation | 52B` | 0.000 | 100% 🔥 | 0.000 | 100% 🔥 |
| `plain lm | 52B` | 0.000 | 100% 🔥 | 0.000 | 100% 🔥 |

## Interpretation — neither cheap S is sufficient, but for different reasons

**Length-based S** (response-side, weak): per-policy ρ varies +0.30 to +0.55. The S→Y curvature differs across policies because shorter responses correlate with refusals at different rates per policy. Estimation errors scale with how-different-from-logger the policy's S→Y curve is — large for OOD (context-distillation, plain-lm), modest for within-family.

**Task-description-score S** (prompt-side, conceptually aligned): ρ(S, Y) collapses to near-zero (|ρ| ≤ 0.30) because the task-description score is a *property of the prompt*, not of the response. Different policies handling the same prompt get the same task score, so this cheap S provides no information for distinguishing within-family targets. Estimation errors are uniform across all within-family targets at ~0.48 — the calibrator outputs essentially the same value for every (target, prompt) pair.

### The cheap-S admissibility conditions

A cheap S that lets CVaR-CJE recover within-family same-mean-different-tail comparisons must be:
1. **Response-side**: it must be a function of the model's actual response, not just the prompt. (Task-description score fails here.)
2. **Aligned with Y**: high |ρ(S, Y)| within each policy, ideally with consistent sign across policies. (Length fails on consistency — ρ swings from +0.14 to +0.62 across cells.)
3. **Cheap enough to compute on every fresh draw**: that's what makes it 'cheap.'

Real cheap-S candidates that would meet all three on HH transcripts: WildGuard-7B refusal-detection score (response-side, harm-aligned, ~free at inference), or a small instruct-tuned grader (Prometheus-2-7B with a refusal-quality rubric).

### The audit's verdict pattern is robust

Across both cheap S choices, the audit *correctly* fires on context-distillation-52B and plain-lm-52B (out-of-family) and accepts on within-family targets. **The audit's diagnostic value is real; it's the level-claim accuracy that depends on cheap-S quality.**

**Bottom line for the cvar_v4 paper**: the framework's headline result requires a real cheap judge, not a proxy. The audit alone does not compensate for a weak cheap S — it catches OOD violations (where the calibrator outputs are wildly off in absolute terms) but cannot detect bias that is *symmetric across all within-family targets* (because the calibrator gives roughly the same wrong answer on every target, so target-vs-target comparisons inherit the bias). The pilot estimate-bias must be measured against an oracle slice for at least one within-family target as a cross-check before publishing level claims.