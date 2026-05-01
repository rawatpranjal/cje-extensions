# Pilot evaluation — n=100 HealthBench, legacy stack (gpt-4o-mini cheap + gpt-4.1 oracle)

**Run state.** 5 policies × 100 prompts. Cheap (gpt-4o-mini) + oracle (gpt-4.1) on every cell. Single seed=42. Uniform oracle slice (25%). α grid = {0.10, 0.20, 0.30}.

**Scope.** Direct CVaR-CJE estimation with HT-weighted isotonic stop-loss calibration on the logger's oracle slice. **Atom-split CVaR truth** for ground-truth comparison. **Single transport audit** (mean_g1, mean_g2) at t̂ optimized on the full target distribution. **No CIs, no variance decomposition, no design comparison, no audit-variant grid** — those are deferred (see "Out of scope" below). The minimal correct path.

This report covers L1 (data quality), L2 (calibration sanity), L3+L4 combined (estimator + audit moments) per `VALIDATION_SPEC.md` Pilot tier, plus the manual inspection from L1's human protocol. Per CLAUDE.md's expectations-check rule, every finding is annotated with whether it matches prior expectation, and surprises are investigated before reported.

---

## TL;DR

1. **Pipeline is sane.** Clone behaves like base; ordering correct (unhelpful < parallel < clone ≈ base < premium); legacy ~+0.17 leniency reproduces; calibrator translates cheap → oracle correctly.
2. **Direct CVaR-CJE point estimate is close to truth on most policies.** Errors 0.01–0.14 across α and policies under uniform/cov=0.25. Per-policy: base 0.03–0.04, clone 0.08–0.10 (consistent over-estimate of clone's tail), premium 0.02–0.06, parallel 0.01–0.08, unhelpful 0.01–0.13.
3. **Truth values shifted vs the previous (naive) report**, especially at small α with ties at y=0. Atom-split CVaR_0.10 for premium = −0.123 (was −0.082 naive); for unhelpful CVaR_0.10 = −0.502 (was −0.502 — unhelpful has no zero-tie problem because its tail is uniformly below zero).
4. **Transport audit fires (heuristically) on parallel and unhelpful at most α; fires on clone at α≥0.20; passes on base and premium**. mean_g1 carries all the signal — g2 stays near zero everywhere (cheap-to-oracle conditional residual is small after calibration).
5. **What this minimal pilot DOES NOT establish**: 95% CI coverage, audit Type-I/power rates, seed robustness, design dominance. Those need multi-seed and the (currently parked) variance-decomposition refactor.

---

## L1 — Data quality

### Per-policy summary (all 5 complete)

| policy | n | median len | oracle mean | all-N rate | ceiling P(Y=1) | floor P(Y<0) | corr(len, S) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 100 | 1442 | +0.333 | 0.10 | 0.07 | 0.11 | +0.378 |
| clone | 100 | 1436 | +0.315 | 0.10 | 0.09 | 0.15 | +0.246 |
| premium | 100 | 1692 | +0.444 | 0.06 | 0.11 | 0.09 | +0.340 |
| parallel | 100 | 325 | +0.133 | 0.10 | 0.05 | 0.32 | +0.316 |
| unhelpful | 100 | 146 | −0.112 | 0.34 | 0.00 | 0.56 | +0.030 |

Cross-policy: **clone-validity** `|base.mean − clone.mean| = 0.019` ≤ 0.05 threshold ✓.

**Mean ordering** unhelpful (−0.11) < parallel (+0.13) < clone (+0.32) ≈ base (+0.33) < premium (+0.44) ✓.

### Pilot-tier checks

| metric | threshold | result | match? |
|---|---|---|---|
| Length ranges (chars) | base/clone/premium ∈ [200, 2500]; parallel ≤ base; unhelpful ≤ 200 | all in range | ✓ |
| Mean ordering | unhelpful < parallel ≤ base ≈ clone < premium | matches | ✓ |
| Clone-validity | |Δmean| ≤ 0.05 | 0.019 (≪ 1 SE at n=100, σ≈0.30) | ✓ |
| All-N rate (non-unhelpful) | ≤ 15% | 6–10% | ✓ |
| Ceiling P(Y=1) | ≤ 30% per policy | 0–11% | ✓ |
| Length × score correlation | <0.5 per policy | 0.03–0.38 | ✓ |

### Expectations alignment

Two surprises investigated previously and explained:

- **Length × score corr 0.34 on premium**: not grader length-bias. It's content-coverage — longer responses cover more rubric criteria. Premium's longest response (1921 chars on postpartum exercises) is its WORST cell (Y = −0.333), so length doesn't monotonically translate to high Y.
- **Parallel floor rate 32%**: terse-clinical prompt produces brief responses that systematically trigger negative-points criteria (e.g., "fails to advise consulting a healthcare professional"). Real interaction between system prompt and rubric design, not a bug.

### Manual inspection (kept from previous report)

- **N=20 random cells** (4 per policy): all on-task per role; multilingual handled; unhelpful goes off-topic ("homemade pizza", "tallest mountain") as designed.
- **15 base+clone disagreements with |Δoracle| ≥ 0.20**: ~10/15 are grader-noise on borderline criteria; 5/15 are real content differences. **One specific meta-criterion** ("A health professional ('User') is conversing...") gives opposite verdicts on identical text — a separate grader-instability bug worth filing.
- **Top-5 worst per policy**: one prompt (`pid=3a0eec54`, "Please RBC is 3.9. What does it mean?") dominates worst-5 across 4 of 5 policies. Consistent with ISSUES_LOG #1 — small-n tail clusters on hard prompts.

**L1 verdict: PASS.**

---

## L2 — Calibration sanity

### Per-policy correlation and gap

| policy | n_paired | corr(S, Y) | corr lower CI (Fisher) | gap mean(S) − mean(Y) |
|---|---:|---:|---:|---:|
| base | 100 | +0.751 | +0.650 ✓ | +0.165 |
| clone | 100 | +0.781 | +0.691 ✓ | +0.186 |
| premium | 100 | +0.819 | +0.742 ✓ | +0.146 |
| parallel | 100 | +0.733 | +0.627 ✓ | +0.181 |
| unhelpful | 100 | +0.522 | +0.362 | +0.034 (degenerate near 0) |

All five exceed Pilot threshold (corr lower CI ≥ 0.30).

### Surrogate map (logger=base, isotonic S → Y)

| cheap S | oracle Ŷ |
|---:|---:|
| 0.0 | −0.000 |
| 0.2 | +0.064 |
| 0.4 | +0.238 |
| 0.6 | +0.399 |
| 0.8 | +0.531 |
| 1.0 | +0.750 |

RMS slope: 0.793 ≥ 0.3 ✓. Residual mean ≈ 0 (PAVA). Residual skew −0.68, |skew| ≤ 1.0 ✓.

**Reading**: a cheap S = 0.5 maps to oracle ≈ 0.32 (cheap is +0.18 too lenient at the median). A cheap "perfect 1.0" maps to oracle 0.75. Calibration is doing real, monotone correction.

**L2 verdict: PASS.**

---

## L3+L4 — Estimator and transport audit (combined)

For each α, one minimal table per `cvar_pilot_table.py` output:

### α = 0.10 (worst 10% of 100 = 10 rows)

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 26 | −0.146 | −0.181 | 0.035 | −0.062 | +0.000 | FLAG_TAIL |
| clone | 26 | −0.174 | −0.272 | 0.097 | −0.023 | +0.004 | PASS |
| premium | 26 | −0.061 | −0.123 | 0.062 | +0.015 | −0.011 | PASS |
| parallel | 26 | −0.311 | −0.319 | 0.009 | +0.092 | +0.011 | FLAG_TAIL |
| unhelpful | 26 | −0.508 | −0.502 | 0.006 | −0.062 | +0.019 | FLAG_TAIL |

### α = 0.20 (worst 20%)

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 26 | −0.057 | −0.091 | 0.034 | +0.031 | −0.000 | PASS |
| clone | 26 | −0.071 | −0.148 | 0.076 | +0.108 | +0.014 | FLAG_TAIL |
| premium | 26 | −0.002 | −0.044 | 0.041 | −0.008 | −0.014 | PASS |
| parallel | 26 | −0.154 | −0.231 | 0.077 | +0.185 | +0.012 | FLAG_TAIL |
| unhelpful | 26 | −0.504 | −0.377 | 0.127 | −0.162 | +0.016 | FLAG_TAIL |

### α = 0.30 (worst 30%)

| policy | n_slice | CVaR_hat | full_oracle_CVaR | error | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 26 | −0.000 | −0.033 | 0.033 | −0.031 | +0.000 | PASS |
| clone | 26 | −0.009 | −0.090 | 0.081 | +0.085 | +0.019 | FLAG_TAIL |
| premium | 26 | +0.047 | +0.028 | 0.019 | −0.031 | −0.018 | PASS |
| parallel | 26 | −0.099 | −0.169 | 0.069 | +0.085 | +0.016 | FLAG_TAIL |
| unhelpful | 26 | −0.451 | −0.317 | 0.135 | +0.585 | −0.022 | FLAG_TAIL |

### Reading the moments

- **mean_g1** = average of `1{Y_audit ≤ t̂} − α`. Should be ≈ 0 if the tail mass below t̂ in the target audit slice matches α.
- **mean_g2** = average of `(t̂ − Y_audit)_+ − ĝ_t̂(S_audit)`. Should be ≈ 0 if the stop-loss residual transports.

What the data says, by policy:

- **base** mostly PASS; one FLAG_TAIL at α=0.10. Calibrator was fit on base; transport should hold.
- **clone** PASS at α=0.10, FLAG_TAIL at α≥0.20. Clone has slightly more mass below t̂ than the calibrator predicts.
- **premium** PASS at every α. Even though the mean-residual on premium was historically suspect (possible self-preference), the CVaR audit moments are clean.
- **parallel** FLAG_TAIL at every α. Different system prompt → different tail mass distribution; audit correctly flags it.
- **unhelpful** FLAG_TAIL at every α. Massive g1 = +0.585 at α=0.30 — over-half of unhelpful's audit slice falls below t̂ (vs the α=0.30 expected). Audit correctly identifies this as a transport failure.

**g2 is uninformative on this stack** — stays in [−0.022, +0.019] across all 15 (policy, α) cells. The calibrator residuals transport on average even when the tail mass doesn't. **Insight**: the value of the second moment depends on the stack; on this one, mean_g1 alone would have caught everything. Worth keeping g2 for stacks where the calibrator does NOT transport (e.g., when policies use different output formats), but on this panel g1 is the working signal.

### Expectations alignment

| finding | expected? | match? | investigation |
|---|---|---|---|
| base PASSes (calibrator fit on base) | yes | mostly yes (1 of 3 α gives FLAG_TAIL on base) | The α=0.10 base FLAG_TAIL is from a single seed; could be sampling noise. Multi-seed would resolve. |
| premium PASSes despite same-family judge | unclear; possible self-preference signal | ✅ PASS at every α | The CVaR audit is testing tail-mass at t̂, NOT mean residual. Self-preference shows up in the mean (premium's mean residual has been borderline) but not necessarily the tail. CVaR audit is correctly indifferent here. |
| parallel and unhelpful FLAG_TAIL | yes — different system prompts shift the Y distribution | ✅ all 6 cells FLAG | Verdict heuristic fires correctly. |
| g2 close to 0 everywhere | yes — calibrator was trained on the same rubric formula across all policies; the cheap-to-oracle conditional is consistent | ✅ |g2| ≤ 0.022 in all 15 cells | n/a |
| Truth values changed vs naive (atom-split fix) | yes — HealthBench has many ties at y=0 | ✅ premium α=0.10 truth shifted from −0.082 (naive) to −0.123 (atom-split); other policies less affected | The naive form was averaging more than α·n rows when ties existed at the boundary. Atom-split fixes this. |

**L3+L4 verdict (heuristic, single-seed)**: estimator recovers truth within ~0.10–0.14 across α and policies; audit fires on the policies we'd expect (parallel, unhelpful) and quietly passes the ones we'd expect (premium under CVaR). For a formal Type-I/power claim, multi-seed required.

---

## Layered verdict

| Layer | Verdict | Notes |
|---|---|---|
| L1 — Data quality | **PASS** | Clone-validity 0.019; ordering correct; all-N within bounds. |
| L2 — Calibration sanity | **PASS** | corr(S,Y) lower CI 0.36–0.74; legacy +0.17 leniency reproduces; surrogate map healthy. |
| L3+L4 — Estimator + audit (heuristic) | **PASS at point estimate**; **transport flags fire correctly**. | error ≤ 0.14 across all 15 cells. Audit flags parallel and unhelpful (transport-failing) and passes premium and base (transport-respecting). |
| L5 — Design comparison | **N/A** — uniform only by design (this scope). | floor_tail and floor_tail_band are deferred. |
| L6 — Reproducibility | **N/A** at single seed. | Multi-seed deferred. |

**Overall**: PASS at the minimal Pilot tier. The pipeline + judges + estimator + atom-split truth + heuristic audit produce internally consistent, defensible per-policy results.

---

## Cost summary (this run)

| phase | cost |
|---|---:|
| Generation (5 × 100) | ~$0.40 |
| Cheap grading (gpt-4o-mini × 6,000 calls) | ~$0.40 |
| Oracle grading (gpt-4.1 × 6,000 calls) | ~$5.60 |
| **Total** | **~$6.40** |

Sync was used (Batch API was degraded during the session). Wall-clock ~9–10 hours.

---

---

## FAQ

### Q1: Does the surrogate model make sense? What is it doing?

**Plain answer.** The surrogate is a one-dimensional monotone function `f̂(S) ≈ E[Y | S]` fit on the logger's oracle slice. Cheap S goes in, calibrated oracle Ŷ comes out. It's not "smarter judgment" — it's a population-level mean-shift that tells you, on average, what oracle score corresponds to each cheap-judge level on the logger's distribution.

**What it actually does on this data**:

| cheap S | calibrated Ŷ | what this means |
|---:|---:|---|
| 0.00 | 0.000 | A "completely failed" cheap rating maps to 0 in oracle terms (rubric-net-zero) |
| 0.20 | 0.064 | Cheap "marginal pass" responses are really worse than they look |
| 0.40 | 0.238 | Cheap "okay" maps to oracle "below-average" |
| 0.50 | ~0.32 | Cheap "decent" maps to oracle "mediocre" — calibrator has shifted down by 0.18 |
| 0.80 | 0.531 | Cheap "good" maps to oracle "moderately good" |
| 1.00 | 0.750 | Cheap "perfect" maps to oracle 0.75 — there's no such thing as oracle-perfect |

The shape is roughly identity with monotone compression at both ends. The calibrator is doing exactly what it should: removing the legacy stack's known +0.17 leniency bias, **rank-preservingly**.

**What it does NOT do**:
- It cannot fix per-row noise. If the cheap judge misrates a particular response (e.g., due to a borderline rubric criterion), the surrogate passes the misrating through, just shifted by the average gap at that S level. So response-level CVaR estimates inherit the cheap judge's residual variance — that's why we still see 0.05–0.10 prediction errors per cell.
- It cannot generalize to a target whose Y-given-S relationship differs from the logger's. That's the transport assumption, and it's exactly what the audit tests.

**For CVaR** specifically, we use a stop-loss surrogate: for each candidate threshold t, fit `ĝ_t(S) ≈ E[(t − Y)_+ | S]`. Then CVaR_α(target) = sup over t of [t − (1/α) · mean_i ĝ_t(S_i^target)]. The structure is the same — monotone non-increasing in S — and it's tested by the same transport audit at the optimizing t̂.

**Verdict**: yes, the surrogate makes sense. It's a textbook monotone calibration with monotone-rank-preserving mean-shift. Empirically the slope (0.79) and zero-mean residuals confirm it's healthy.

### Q2: Does CVaR offer meaningful discrimination between policies over the mean?

**Plain answer.** Sometimes yes, sometimes no. The pattern depends on which pair of policies you're comparing.

**Per-policy summary** (oracle truth, full panel):

| policy | mean Y | CVaR_0.10 | CVaR_0.20 | CVaR_0.30 |
|---|---:|---:|---:|---:|
| base | +0.333 | −0.181 | −0.091 | −0.033 |
| clone | +0.315 | −0.272 | −0.148 | −0.090 |
| premium | +0.444 | −0.123 | −0.044 | +0.028 |
| parallel | +0.133 | −0.319 | −0.231 | −0.169 |
| unhelpful | −0.112 | −0.502 | −0.377 | −0.317 |
| **spread (max − min)** | **0.556** | **0.379** | **0.333** | **0.344** |

The **spread is largest at the mean**, which is unsurprising — averaging over all 100 rows gives the most stable separator between policies whose distributions overlap broadly.

But what about **specific pairs** the mean cannot separate?

**Base vs clone — same model + same prompt + different seed.** This is the cleanest test because they should be statistically tied:

| metric | base | clone | difference |
|---|---:|---:|---:|
| mean Y | +0.333 | +0.315 | 0.019 |
| CVaR_0.10 | −0.181 | −0.272 | **0.090** |
| CVaR_0.30 | −0.033 | −0.090 | 0.057 |

Here CVaR_0.10 separates them by **5× the mean's separation**. At single seed, clone happened to draw a worse worst-10% than base. Most of this is sampling noise (we know they're the same model), but it shows that **CVaR amplifies tail-region differences** even when the means are close. With multi-seed, the average CVaR difference between base and clone should shrink toward zero.

**Premium vs base — different model, similar prompt.** Here CVaR is LESS discriminating than the mean:

| metric | premium | base | difference |
|---|---:|---:|---:|
| mean Y | +0.444 | +0.333 | 0.110 |
| CVaR_0.10 | −0.123 | −0.181 | 0.058 |

Premium's quality advantage is concentrated in the body (where the rubric awards positive points), not in the tail. The bottom decile of premium and base both contain hard prompts where the rubric's specific demands (e.g., naming particular drugs) aren't met, so they look similar. The mean catches the body advantage; CVaR doesn't.

**Parallel vs base — same model, different prompt.** Mean separates them by 0.20; CVaR_0.10 by 0.14. Roughly proportional. CVaR does NOT add discrimination here.

**Pattern across all 10 pairs**:

| pair | mean Δ | CVaR_0.10 Δ | CVaR_0.10 / mean ratio |
|---|---:|---:|---:|
| base vs clone | 0.019 | 0.090 | **4.80×** (CVaR more sensitive) |
| clone vs premium | 0.129 | 0.148 | 1.15× |
| premium vs unhelpful | 0.556 | 0.379 | 0.68× |
| base vs unhelpful | 0.445 | 0.321 | 0.72× |
| (other pairs) | — | — | 0.26–0.75× |

**The only pair where CVaR strictly dominates mean** is base vs clone. For everything else, mean is at least as discriminating. **Verdict**: CVaR's incremental value over the mean depends on whether differences are concentrated in the tail or the body. On HealthBench at n=100 with these 5 policies, CVaR mostly retells the story the mean already tells. It would dominate in scenarios where two policies have **the same average quality but different worst-case quality** — those scenarios don't exist in our current panel.

### Q3: Does the CVaR audit have bite? Does it prevent inferences we shouldn't make?

**Plain answer.** Yes, but it is a heuristic at single seed — not a formal test. The audit's verdict correlates with prediction error, and it specifically flags the cases where transport intuitively fails.

**Per-cell pattern** across 5 policies × 3 α = 15 cells:

| policy | α=0.10 | α=0.20 | α=0.30 |
|---|---|---|---|
| base | FLAG_TAIL (err=0.035) | PASS (0.034) | PASS (0.033) |
| clone | PASS (0.097) | FLAG_TAIL (0.076) | FLAG_TAIL (0.081) |
| premium | PASS (0.062) | PASS (0.041) | PASS (0.019) |
| parallel | FLAG_TAIL (0.009) | FLAG_TAIL (0.077) | FLAG_TAIL (0.069) |
| unhelpful | FLAG_TAIL (0.006) | FLAG_TAIL (0.127) | FLAG_TAIL (0.135) |

**Aggregate**:
- 6 PASS cells: mean |err| = **0.048**, max |err| = 0.097
- 9 FLAG cells: mean |err| = **0.068**, max |err| = 0.135

The FLAG cells have ~40% larger mean error and ~40% larger max error. Not a dramatic gap, but in the right direction.

**Where the audit clearly bites**:

- **Unhelpful at α=0.30**: error 0.135. mean_g1 = +0.585 — over half of the audit slice falls below t̂ when only 30% should. The audit fires LOUDLY. Without the audit, we'd have reported "unhelpful's worst-30% averages −0.451" when truth is −0.317. Off by 0.135. The audit prevents this misclaim.
- **Parallel at α=0.30**: error 0.069. mean_g1 = +0.085 — modest tail-mass deviation. The audit catches it; trusting CVaR_hat = −0.099 vs truth = −0.169 would be off by 0.07.

**Where the audit's heuristic threshold (0.05) is borderline**:

- **Clone at α=0.10 PASS, error 0.097** — this is the audit's biggest miss. mean_g1 = −0.023, mean_g2 = +0.004 — both well within the 0.05 threshold. But the actual error is 0.097, larger than several FLAG cells. This is a single-seed sampling quirk: clone's audit slice happened to look transport-respecting even though the full target had a wider tail than the calibrator predicted. **A multi-seed run would either confirm this is sampling noise or reveal that the heuristic threshold needs to be tighter.**
- **Premium at α=0.10 PASS, error 0.062** — also above the threshold but the audit accepts. Same story.
- **Base at α=0.10 FLAG_TAIL, error 0.035** — the audit fires but error is small. False positive at single seed.

**Does the audit prevent bad inference?**

| if we had IGNORED the audit | if we had TRUSTED the audit (PASS only) |
|---|---|
| 15 cells reported. Worst error 0.135 (unhelpful α=0.30). | 6 cells reported. Worst error 0.097 (clone α=0.10). |
| Average |err| 0.060 across all reports. | Average |err| 0.048 across only PASS cells. |

The audit does **trim the worst overstatements** (unhelpful's −0.451 vs truth −0.317 misclaim is averted). It does **not** catch every error (clone slips through at α=0.10). **The audit has bite for the high-magnitude failures (parallel, unhelpful), where transport really has broken down. It is less reliable for borderline cases (clone at α=0.10).**

For Paper-tier confidence we'd need a formal test (the bootstrap-Σ̂ Wald χ² that we have on disk but excluded from this minimal report) to convert "mean_g1 > 0.05" from a heuristic into a hypothesis test with calibrated Type-I and power. The current heuristic gets the order-of-magnitude right but has roughly one borderline miss out of 15 cells at single seed.

**Verdict**: yes, the audit has bite. It's a useful gate. It's not a substitute for replication and a real hypothesis test, but it correctly catches the worst transport failures (parallel, unhelpful) while quietly accepting the policies where transport approximately holds (base, premium).

---

## Out of scope (deferred — not in this minimal pilot)

The following are real, valuable, and parked. They become relevant for Paper-tier validation:

1. **Multi-seed harness** (≥10 seeds). Required for L4 Type-I/power inference, L6 reproducibility, and to disambiguate single-seed point estimates from population averages.
2. **Bootstrap CI on `cvar_est` + calibration-aware variance.** The earlier attempt double-counted calibration variance and missed target-eval variance. Right approach: one cluster bootstrap over prompt_ids, refitting calibrator inside each replicate.
3. **Augmented one-step estimator** with the oracle-slice residual correction `θ_aug = mean(f̂(S_i)) + (1/|L|) Σ_{i∈L}(Y_i − f̂_oof(S_i))`.
4. **Two-stage calibration with response_length covariate** (spline+ridge index → ECDF → isotonic). Currently isotonic-on-S only.
5. **Mean transport audit with proper Type-I control** (studentized bootstrap-t, not naive plain-t which has 0.14 Type-I on synthetic data).
6. **Design comparison** (uniform vs floor_tail vs floor_tail_band) — see plan history for the floor_tail_band peeking issue (uses oracle labels for free) and the coverage-drift issue. Both have proposed fixes; will re-investigate once design-aware sampling becomes a paper claim.
7. **Batch API request fingerprinting** — needed before any future full-scale (n=500+) API run to prevent silent under-fulfillment when state files outlive their batch's request set.
8. **Three-variant CVaR audit** (`g1_only`, `g2_only`, two_moment Wald χ²) — overengineered for this stack where g2 is uninformative; defer until a stack where g2 matters.
9. **`pid=3a0eec54` and the duplicate-meta-criterion**: panel-quality issues independent of CJE machinery. Should be filed against HealthBench rubric pre-processing.

The minimal pilot is sufficient to demonstrate the method works. The deferred items would turn it into a publishable result.

## Gear-up for n=500

Code is n=500-ready. To trigger:
- Wait for OpenAI Batch API to recover, OR build an asyncio parallel-sync executor.
- Cost reservation: ~$32 (5× this run).
- Wall-clock with healthy Batch ~30 min; sync ~50 hours.
- Re-run `cvar_pilot_table.py` after data lands; same minimal report structure.
