# Validation Spec — `cvar_v4/healthbench_data`

The acceptance contract for any run of the CVaR-CJE pipeline. `SPEC_SHEET.md` says what to compute; this document says what evidence demonstrates the computation is trustworthy.

## Validation philosophy

We validate the **Direct CVaR-CJE estimator + designed oracle slice + transport audit** against **full-oracle simulation truth**. "Truth" here is `cvar_alpha(y_target_full, alpha)` over every row's oracle label — a deliberate construction, since this is a simulation experiment in which we collected oracle labels for every row and then masked some to simulate a partial-oracle slice.

**Two scope conditions to keep front of mind:**

1. **Conditional on oracle = truth.** All validation is conditional on the oracle LLM judge (gpt-4.1 in the legacy stack) being treated as ground truth for the rubric. If the oracle judge has its own biases relative to physician scores, our entire validation chain is biased the same way. Layer 1 partly defends against this with human protocols, but the conditional is unremovable without physician labels.
2. **Estimation vs inference.** At a single seed we get **point estimates** of bias, |err|, p-values. At multiple seeds we can do **inference** about Type-I error rate, power, and CI coverage. Pilot tier (single seed) does point estimation. Paper tier (≥10 seeds) does inference. Don't confuse the two — a single audit p-value of 0.04 is not "Type-I error 0.04"; it is one realization.

**Two tiers:**
- **Pilot** — what an n=100, single-seed run can show. Mostly sanity, monotonicity, right-direction effects, single-seed point estimates with their SE explicitly written down.
- **Paper** — what an n=500+, ≥10-seed, multi-α, multi-stack run must show. Inference about Type-I, power, CI coverage; multiple-comparisons control; paired-t tests across seeds.

**Layer ordering matters.** Failure of L1 invalidates the relevance of L2–L6; failure of L2 invalidates L3–L6; etc. Don't jump ahead. The estimator can't fix bad data; the audit can't compensate for a broken estimator.

**Each threshold below is annotated with `(SE ≈ ...)` showing the standard error we'd expect at the relevant n,** so the reader can judge whether a near-miss is sampling noise or a real failure.

---

## L1 — Data quality

**Targets.** Are responses on-task? Are scores plausibly distributed? Is rubric grading well-behaved?

### Quantitative metrics

| metric | source | Pilot threshold (n=100, single seed) | Paper threshold (n=500, multi-seed) |
|---|---|---|---|
| per-policy mean response length (chars) | responses.jsonl | base/clone/premium ∈ [200, 2500]; parallel ≤ base; unhelpful ≤ 200 | same |
| per-policy oracle mean ordering | judge_outputs/*_oracle.jsonl | `unhelpful < parallel ≤ base ≈ clone ≤ premium` | same |
| clone-validity: \|base.mean − clone.mean\| | judge_outputs | ≤ 0.05 (SE ≈ 2σ/√n ≈ 2·0.30/√100 ≈ 0.06 at the mean; threshold = ~1 SE of expected sampling noise — so this is a sanity check, not a tight test) | ≤ 0.03 (SE ≈ 0.027 at n=500) |
| all-N oracle cell rate per policy | row-level scores | ≤ 15% on non-unhelpful (Wilson 95% lower bound on a 0.10 true rate at n=100 is 0.06; threshold of 0.15 = upper bound of a 0.10 true rate) | ≤ 5% (a much tighter Paper bar — at n=500 SE on a 0.05 rate is √(0.05·0.95/500) ≈ 0.010) |
| ceiling rate P(Y = 1.0) per policy | row-level scores | ≤ 30% (rough; ceiling >50% indicates the rubric saturates and CIs near the top will be artificially narrow) | ≤ 25% |
| floor rate P(Y < 0) | row-level scores | only unhelpful + parallel | same |
| length × score Pearson corr per policy | join responses + scores | \|corr\| < 0.5 (SE on sample corr ≈ √(1-r²)/√n ≈ 0.075 at n=100; threshold of 0.5 well above the SE) | < 0.3 (SE ≈ 0.041 at n=500) |

### Human protocol

- **Pilot, N=20 random**: read `prompt + response + cheap verdicts + oracle verdicts`. Look for nonsense, off-topic non-unhelpful responses, grader hallucinations.
- **Pilot, N=5 base+clone disagreements** with |Δscore| ≥ 0.20: confirm divergence is grader-noise on borderline criteria, not policy-output divergence.
- **Pilot, top-5 worst oracle scores per policy**: are these hard prompts (specific clinical asks) or genuine response failures?
- **Paper, N=50 stratified by (policy, score bin)**, plus a 5% subsample re-graded by a third judge to bound grader-noise variance σ_g². Grader noise contributes to L4 audit power; need its scale.

### Failure response

- on-task rate < 80%: regenerate with adjusted system prompts (panel restart, documented).
- all-N rate too high: cheap/oracle prompt is over-strict OR rubric is mismatched to general medical-assistant outputs. Either widen α (use 0.20–0.30 instead of 0.10) so the bottom rows aren't all-N, pre-filter prompts whose rubrics demand named drugs/tests, or accept it as a panel feature with explicit caveat.
- clone-validity violated by more than 2 SE: investigate seed determinism, prompt non-determinism, grader-instability.

---

## L2 — Calibration sanity

**Targets.** Does cheap S carry signal for oracle Y? Is the surrogate map well-behaved?

### Quantitative metrics

Sample-correlation SE at n=100 is ≈ 0.075 if true ρ=0.7, so a point estimate of 0.65 is consistent with ρ=0.7. Use **lower 95% Fisher CI bound**, not point estimate, for thresholds.

| metric | source | Pilot threshold | Paper threshold |
|---|---|---|---|
| per-policy Pearson corr(S, Y), 95% lower CI bound (Fisher z) | join cheap + oracle | ≥ 0.30 (excluding unhelpful, where Y is clustered near 0 → corr is degenerate) | ≥ 0.50 |
| mean cheap-oracle gap mean(S) − mean(Y) per policy | per-policy means | reported, no threshold (free parameter — small gap = calibration is near-identity) | reported with bootstrap CI |
| isotonic surrogate map slope | fit isotonic on logger panel; mean of pairwise (Δŷ/Δs) over knot intervals weighted by Δs | ≥ 0.3 | ≥ 0.5 with bootstrap CI lower bound > 0.3 |
| residual `Y − f̂(S)` distribution under leave-one-out | OOF residuals on oracle slice | mean: \|μ\| < 0.05 (≈ SE at n=100 with σ_residual ≈ 0.2: 0.02; threshold = 2.5 SE) | \|μ\| < 0.02 |
| residual symmetry (skewness) | OOF residuals | \|skew\| < 1.0 (rough; heavy skew → t-tests in L4 are biased) | \|skew\| < 0.5 |

### Human protocol

- Scatter plot S vs Y per policy with isotonic line. Look for: heteroscedasticity (variance non-constant in S), bend at the top (saturation), gaps in support of S.
- Inspect 5 highest |Y − f̂(S)| residual rows: rubric disagreements? grader hallucinations? legitimately ambiguous responses?
- Histogram of residuals across all policies. Symmetric? Heavy-tailed? If heavy-tailed, the bootstrap-Σ̂ approximation in L4 may need studentization.

### Failure response

- corr lower bound below threshold: cheap judge has insufficient signal — switch to a better cheap model (gpt-4o instead of mini), richer judge prompt, or out-of-family judge (different vendor) to reduce self-preference.
- |gap| > 0.4: cheap and oracle scoring different rubrics or formulas — re-check `_aggregate_score` and rubric pipeline.
- isotonic map flat (slope < 0.3): cheap is uninformative — same fix as low correlation.

---

## L3 — Estimator accuracy (Direct CVaR-CJE)

**Targets.** Does V̂_Direct(π') recover the full-oracle truth? Is its CI honest?

### Critical implementation gap

Three pressing items in `CLAUDE.md` directly bias L3 metrics:
- **No augmented one-step estimator** → V̂_Direct has first-order bias from fitting f̂ on the oracle slice and evaluating on all data. Expect bias of order O(1/|L|).
- **No two-stage calibration with response_length** → bias persists when length confounds rubric satisfaction. Currently isotonic-on-S only.
- **No calibration-aware variance** → bootstrap captures evaluation variance only; misses the oracle delete-one-fold jackknife `Var_cal`. **Coverage will be undercovered.** Expect 95% CI to achieve maybe 80–90% empirical coverage on the current code.

These gaps **must be closed before Paper-tier validation is meaningful** for L3. Pilot-tier validation can proceed but interpret coverage shortfalls as expected.

### Quantitative metrics

Read from `results_design_comparison_alpha_*.jsonl`. Each row has `cvar_est`, `full_oracle_truth`, `abs_error`, `audit_p`, `audit_reject`, `n_slice`.

| metric | source | Pilot threshold | Paper threshold |
|---|---|---|---|
| per-policy \|cvar_est − full_oracle_truth\| under (uniform, cov=0.25) | results JSONL | ≤ 0.10 on ≥4/5 policies (SE on the bias is roughly σ_y/√n_slice ≈ 0.30/√25 ≈ 0.06; threshold = 1.7 SE) | ≤ 0.05 on ≥4/5 (SE ≈ 0.024 at n_slice=125; threshold = 2 SE) |
| 95% CI realized coverage rate | bootstrap intervals across seeds | NA at single seed | nominal-95% CI achieves empirical coverage ≥ 0.90 across ≥10 seed-policy cells (note: until V_cal is implemented, 0.85 is the achievable target) |
| MSE vs cheap-only baseline | bootstrap MSE | NA at single seed (need replicates) | E[\|cvar_est_CJE − truth\|²] ≤ 0.70 × E[\|cvar_est_cheap − truth\|²], 95% bootstrap CI on ratio < 1 |
| Var_cal / Var_total share per policy | jackknife decomposition | NOT IMPLEMENTED — see CLAUDE.md issue #3 | reported per policy with 90% jackknife CI |

### Human protocol

- Per-policy bar chart: cheap mean | Direct CVaR-CJE | full-oracle truth, with 95% CI bars on Direct. Visually confirm Direct sits between cheap and truth (closer to truth) for non-unhelpful policies.
- Inspect cells where Direct is FARTHER from truth than cheap (ideally rare, ≤ 1/5 policies). If common, the calibrator is harming.
- Compare cvar_est across (uniform, floor_tail, floor_tail_band) for the same policy: should cluster around truth, not scatter.

### Failure response

- bias persists across seeds: calibration mis-specified — add two-stage calibration with `response_length` covariate, then add the augmented estimator.
- CI coverage < 0.85: bootstrap captures eval variance only — add calibration-aware variance via oracle delete-one-fold jackknife.
- MSE ratio ≥ 1: cheap is so well-calibrated that calibrator is fitting noise — observed on Phase 1 nano/mini stack; choose a stack with a real cheap-oracle gap (e.g., legacy gpt-4o-mini → gpt-4.1).

---

## L4 — Audit calibration (size and power)

**Targets.** Does the audit reject when it should and accept when it should?

### Quantitative metrics with effect-size pinning

Power without an effect size is a meaningless number. Define the effect size **a priori** from the data:

- δ_g (transport-fail effect size) := empirical |E[g]|/SD[g] under unhelpful, where g is the audit's moment vector. Computed from the observed unhelpful slice. The audit's power is its ability to detect a deviation of magnitude δ_g.
- For our panel, δ_g_unhelpful is expected to be large (≥ 1.5 — unhelpful's tail mass differs hugely from base's), so power should be high.

Type-I needs many replications to estimate. With B=2000 bootstrap reps (per `SPEC_SHEET.md`) and ≥10 seeds, the Wilson 95% CI on a true 0.05 rate is roughly [0.025, 0.095].

| metric | source | Pilot threshold | Paper threshold |
|---|---|---|---|
| empirical Type-I on transport-respecting policies (base, clone) at nominal α=0.05 | multi-seed audit_reject rate | NA (single seed = single Bernoulli draw) | upper Wilson 95% CI bound on observed rate ≤ 0.10 across ≥10 seeds × 2 policies = 20 cells (nominal target 0.05; allow 0.05 over-rejection budget) |
| empirical power on unhelpful at nominal α=0.05 | audit_reject rate | ≥1/3 audit variants rejects at p<0.10 (single seed, weak evidence) | lower Wilson 95% CI bound on observed rate ≥ 0.70 across ≥10 seeds (target power 0.80; allow 10pp shortfall) |
| p-value distribution under null | audit p-values from base + clone across seeds | NA | KS test p > 0.10 vs Uniform(0,1); needs ≥30 null cells |
| moment attribution: g1_only vs g2_only vs two_moment | inspect mean_g1, mean_g2 columns; B=2000 bootstrap CI | report per cell | paired comparison of rejection rates across seeds, BH-corrected for the 3 audit × 5 policy × 2 α = 30 paired tests |

**B (bootstrap reps) requirements:** percentile CIs at 95% need B ≥ 1000 for stability; SPEC_SHEET says 2000 for cluster bootstrap. The current `compare_designs` default is B=200 (smoke); rerun at B=2000 before Paper-tier scoring.

### Human protocol

- For each rejected (policy, audit) cell: check `mean_g1` and `mean_g2`. Which moment dominates?
- Power ordering: at small n, expect g1_only most powerful (single-df beats two-df under noisy tail signal). At larger n, expect two_moment to overtake g1_only when both moments carry real signal. Confirm this trend.
- Save bootstrap-Σ̂ samples on a few cells; eyeball for heavy tails or asymmetry that would invalidate the χ² approximation. If found, switch to studentized bootstrap.

### Failure response

- Type-I above its budget: paired-bootstrap Σ̂ underestimates variance. Verify t̂ is re-maximized inside each rep (already implemented per `cvar_v3/SCHEMA.md §12`). If still high, increase B or studentize.
- Power below 0.70 lower bound: increase B, increase n, or use stratum-robust SE on designed slices.
- Both single-moment audits never fire but two_moment does: the joint test is doing real work — keep all three. Report all three to expose what drives any reject.

---

## L5 — Design comparison (uniform vs floor_tail vs floor_tail_band)

**Targets.** Does the design-aware slice beat uniform at the same total budget?

### Multiple-comparisons accounting

We're testing 3 designs × 3 audits × 5 policies × 2 α = **90 paired comparisons** if everything is independent. Naive p<0.05 inflates family-wise Type-I to 99%. **Use BH-corrected paired-t at FDR=0.10** for the design-vs-uniform comparisons; pre-register the comparisons.

### Quantitative metrics

| metric | source | Pilot threshold | Paper threshold |
|---|---|---|---|
| realized coverage per design | n_slice / n_total averaged across policies | ≥ 0.95 × target coverage (verifies budget-conservation auto-bump) | same; per-policy lower CI bound ≥ 0.90 × target |
| \|err\|_floor_tail / \|err\|_uniform median across policies | results JSONL paired by (policy, audit, α) | < 1.0 (right-direction; single-seed has high variance — accept up to 1.2) | ≤ 0.85 with paired-t **q < 0.10** (BH-corrected over the 30 paired comparisons in the (audit × policy × α) grid) across ≥10 seeds |
| CI width ratio floor_tail / uniform | bootstrap CI widths | ≤ 1.0 median | ≤ 0.85 median with paired-t q < 0.10 (BH-corrected) |
| audit rejection rate at unhelpful: floor_tail vs uniform paired by seed | paired comparison | floor_tail ≥ uniform on the single seed | McNemar's exact test on the 2x2 reject table across seeds, q < 0.10 (BH-corrected) |
| floor_tail_band vs floor_tail | one of {bias, CI width, power} | required to beat or tie | required to **strictly** beat on at least one with q < 0.10; otherwise drop |

### Human protocol

- For each policy, list the prompts the bottom stratum selected. Are these the lowest-Y prompts? (cheap-S correctly orders the tail = stratification is informative.)
- Inspect auto-bumped pi_min cases (where the requested floor was unsatisfiable). Verify the bump preserves the spirit (oversample bottom, give a real floor everywhere).
- For floor_tail_band: does the band actually cover q̂_α? Plot cheap-S quintile bins with the band marked. If the band is in the wrong place (e.g., upper quintile because q̂_α is mis-located by the pilot), the design is malfunctioning.

### Failure response

- floor_tail loses to uniform: cheap-S poorly orders Y in the tail. Try (i) pooled-cheap-S deciles instead of per-policy, (ii) a pilot calibrator that predicts Y from cheap-S and stratifies on predicted Y rather than raw S.
- floor_tail_band doesn't strictly beat floor_tail: drop band complexity from the contribution; keep floor_tail as the recommended design.
- Realized coverage off target: budget-conservation logic regression — re-test the auto-bump fix.

---

## L6 — Reproducibility

**Targets.** Are conclusions robust to seed, α, and coverage?

### Quantitative metrics

Strict monotonicity is too strong as a threshold (single-seed noise can flip orderings). Use **weak monotonicity within sampling noise**: a violation only counts if the reversal exceeds 2 SE.

| metric | source | Pilot threshold | Paper threshold |
|---|---|---|---|
| across-seed standard error of cvar_est per policy | multi-seed loop | NA | SE ≤ 0.05 (Wilson CI on 10-seed estimate is wide; n=500 + 10 seeds gives ~0.03 expected SE on stable cells) |
| audit-decision stability (frac. seeds with same reject/accept) per (policy, audit, design) | multi-seed loop | NA | ≥ 0.80 for transport-respecting; ≥ 0.85 for unhelpful (audit decisions on unhelpful should be near-deterministic at n=500) |
| α-coherence: as α grows from 0.05 → 0.30, CI width and \|bias\| weakly decrease, allowing 2-SE noise | α grid {0.05, 0.10, 0.20, 0.30} | NA | no policy-α reversal exceeds 2 SE |
| coverage-coherence: as cov grows from 0.10 → 1.00, CI width decreases | coverage grid | NA | strictly weakly decreasing within 2 SE per policy |

### Human protocol

- Multi-seed comparison tables side-by-side. Are there policies where the audit decision flips across seeds? Red flag for power.
- α-coherence plot: CI width vs α on a single chart. Smooth decreasing curve expected.
- Coverage-coherence plot: CI width vs coverage. Smooth decreasing curve expected.

### Failure response

- Across-seed instability (SE > 0.05): increase n; SE scales as 1/√n.
- α-incoherence (e.g., α=0.10 has tighter CI than α=0.30): small-α tail degenerate due to all-N pattern (ISSUES_LOG #1). Either widen α or pre-filter degenerate prompts.
- Coverage-incoherence: design or estimator has a budget bug.

---

## Tiered acceptance gates (one-page summary)

### Pilot tier (n=100 single seed) — pass requires:

- **L1**: data quality OK (length ranges, ordering, clone-validity within 1 SE); all-N rate ≤ 15% on non-unhelpful.
- **L2**: corr(S, Y) lower CI bound ≥ 0.30 on non-unhelpful; isotonic map slope ≥ 0.3.
- **L3**: |cvar_est − full_oracle_truth| ≤ 0.10 on ≥4/5 policies under (uniform, cov=0.25).
- **L4**: ≥1 audit variant rejects unhelpful at p<0.10. (single seed = weak evidence.)
- **L5**: floor_tail not catastrophically worse than uniform (|err| ratio < 1.2 single-seed).
- **L6**: NA at single seed; defer to Paper.

### Paper tier (n=500+, ≥10 seeds, multi-α, multi-stack, B=2000 bootstrap) — pass requires:

- **L1+L2**: as Pilot with stricter thresholds (corr lower CI ≥ 0.50; all-N ≤ 5%; clone-validity ≤ 0.03).
- **L3**: 95% CI realized coverage rate ≥ 0.90 across seed-policy cells (assumes Var_cal implemented; until then, expect 0.85); MSE ratio CI ⊂ [0, 1) vs cheap-only.
- **L4**: empirical Type-I upper Wilson bound ≤ 0.10 on transport-respecting; power lower Wilson bound ≥ 0.70 on unhelpful; null KS p > 0.10.
- **L5**: floor_tail beats uniform at q < 0.10 (BH-corrected) on at least 2/3 of |err|, CI width, audit power; floor_tail_band strictly beats floor_tail on at least one.
- **L6**: per-policy SE across seeds ≤ 0.05; audit-decision stability ≥ 0.80; α and coverage trends weakly monotone within 2 SE.

---

## n-dependent expectations

| n | what's possible | what's NOT |
|---|---|---|
| 10 (smoke) | Plumbing verification only. None of L1–L6 thresholds apply. | Anything statistical |
| 100 (pilot) | L1 sanity, L2 single-seed point estimates of corr/gap, L3 single-seed |err| comparison, L4 single-seed audit verdicts on unhelpful only | Type-I rate; multi-seed inference; stat tests |
| 500 (interim) | Pilot + L4 size/power with bootstrap CIs, L5 paired-t at single seed, multi-α grid | Multi-stack robustness; external validity beyond LLM oracles |
| 1000+ (paper) | All layers full | (publication ready) |

---

## Mapping to existing tooling

- **Pipeline outputs**: `data/responses/{policy}_responses.jsonl`, `judge_outputs/{policy}_{cheap,oracle}.jsonl`, `data/cje_dataset.jsonl`, `cost_ledger.jsonl`.
- **Design-comparison results**: `results_design_comparison_alpha_{X.X}.{jsonl,md}` produced by `compare_designs.py`. Columns: `policy, design, audit_variant, cvar_est, full_oracle_truth, abs_error, audit_p, audit_reject, mean_g1, mean_g2, n_slice, n_total`.
- **L1, L2** computations: ad-hoc Python on the raw response/score files.
- **L3, L4, L5** computations: aggregate over rows of `results_design_comparison_*.jsonl`.
- **L6** computations: requires a multi-seed harness — **NOT YET BUILT**. Wrap `compare_designs.run_grid` in a loop over seeds, write a `seed` column, aggregate.

---

## Pressing implementation gaps that block Paper tier

These are tracked in `CLAUDE.md`; closing them is a precondition for Paper-tier validation:

1. **Two-stage calibration not implemented** (blocks L3 paper-tier MSE claim and unbiasedness).
2. **Augmented one-step estimator not computed** (blocks L3 unbiasedness and CI honesty).
3. **Calibration-aware variance not computed** (blocks L3 CI coverage rate; current code expected to undercover by ~5–10%).
4. **Multi-seed harness not built** (blocks L4 size/power, L6 entirely).
5. **Mean transport audit not implemented** (blocks reporting at the Mean estimand alongside CVaR; SPEC_SHEET requires).
6. **`B = 2000` bootstrap reps not run** (current `compare_designs` default is B=200; ok for smoke, not for Paper-tier audit Type-I/power inference).

Pilot-tier validation is fully achievable on the current code today.
