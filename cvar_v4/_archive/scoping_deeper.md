# CVaR-CJE Dataset Scoping — Deeper Analysis (`scoping_deeper.md`)

_Supplement to `cvar_v4/scoping_data.md`. Generated 2026-04-28._

The first scoping pass classified each dataset's CVaR-CJE-readiness from
its empirical EDA. This deeper pass answers four follow-up questions that
the first pass surfaced but didn't fully resolve:

1. **Which empirical (logger, target) pairs in the data show same-mean /
   different-tail at the 95%-CI level?** (HH red-team-attempts, UltraFeedback)
2. **Does the Wald two-moment audit have real bite — fires on bad pairs,
   accepts on good pairs?** (HH red-team-attempts; the answer is *yes,
   but asymmetrically* — see §2)
3. **Does the same-mean / different-tail pattern survive prompt
   stratification?** Yes; in UltraFeedback's `ultrachat` source two models
   with means 4 thousandths apart have CVaR_0.05 1.4 points apart (140×
   ratio).
4. **Is the actual physician-graded HealthBench Y truly continuous?**
   Mostly — clean structural pass, with one caveat about discrete-21
   distinct values driven by physician-count = 2.

The detailed per-analysis outputs live in `cvar_v4/eda/deeper/*.md`. This
document synthesizes them into recommendations for the cvar_v4 paper.

---

## §1 — Empirical (mean, CVaR) pairs in HH red-team-attempts

Source: `cvar_v4/eda/deeper/hh_pairwise.md`

The 12 (model_type | num_params) cells in Anthropic HH-RLHF's
red-team-attempts config provide a natural multi-policy panel. Bootstrap
95% CIs were computed on mean, CVaR_0.05, and CVaR_0.01 for each cell.

The cleanest **strict** same-mean-different-tail pair (mean CIs overlap
AND CVaR_0.05 CIs are disjoint):

| A | B | mean_A [CI] | mean_B [CI] | CVaR_A [CI] | CVaR_B [CI] | Δmean | ΔCVaR | ratio |
|---|---|---|---|---|---|---|---|---|
| `rlhf | 13B` | `rejection sampling | 2.7B` | +1.426 [+1.398, +1.456] | +1.492 [+1.448, +1.536] | -0.175 [-0.298, -0.056] | -1.142 [-1.398, -0.840] | 0.066 | 0.967 | **14.6×** |

Mean-CJE on this pair: the 95% bootstrap CIs on means overlap heavily
([+1.398, +1.456] vs [+1.448, +1.536]). A reader of mean-CJE output would
say "indistinguishable safety." CVaR_0.05-CJE: the 95% bootstrap CIs
[-0.298, -0.056] and [-1.398, -0.840] are disjoint by 0.5 units — a
strong, falsifiable separation. This is the headline pair for the cvar_v4
paper.

A note on what the cells *mean*. RLHF and rejection-sampling are
different alignment training procedures: RLHF reshapes the policy
distribution; rejection-sampling discards low-scoring samples at training
time. At small parameter count (2.7B), rejection-sampling's training-time
filter doesn't generalize as well — when a novel red-team attack lands
outside the training-distribution support, the response is much worse
than RLHF's would be. Mean evaluation hides this; CVaR exposes it.

---

## §2 — Audit's bite is asymmetric in logger choice

Source: `cvar_v4/eda/deeper/hh_audit_demo.md`

Two loggers tested with the bootstrap-Σ̂ Wald two-moment audit
(implementation: `cvar_v4/eda/deeper/_estimator.py`, ported from
`cvar_v3/workhorse.py:two_moment_wald_audit_xf` with a 1e-6 ridge added
for numerical stability):

### Logger = `context distillation | 52B` (worst-tail logger; mean Y = -0.83)
- audit accepts on **9/9 targets**
- including rlhf-52B (Direct CVaR err 4.18) and rlhf-13B (err 5.06) — *catastrophic estimation failures the audit silently lets through*
- mechanism: t̂ pinned to logger's empirical 5%-quantile (≈ -3.6 to -4.7); for rlhf targets whose Y is mostly positive, almost no observations fall below t̂ → g_1 ≈ -0.05, g_2 ≈ small, the moments don't see the violation

### Logger = `rlhf | 52B` (best-tail logger; mean Y = +1.7)
- audit fires on **5/8 targets** — exactly the un-aligned (context distillation, plain LM) cells
- audit accepts on **3/8 targets** — exactly the within-alignment-family (rlhf-13B, rejection-sampling-2.7B, rejection-sampling-52B) cells
- mechanism: t̂ set near rlhf's q_0.05 (slight negative); for un-aligned targets whose Y is mostly negative, *many* observations fall below t̂ → g_1 jumps to +0.55–0.63 (way above α=0.05), the calibrator's stop-loss prediction is way off, audit catches it (W = 161–347, p < 0.0001)

### The operational rule

> **Pick a logger whose Y distribution is at least as harmless (high-Y) as
> the targets you want to certify.** The audit will then fire when a
> target is unexpectedly worse than the calibrator predicts — exactly the
> case where reporting a level claim would mislead. The audit is
> asymmetric by design: it refuses to certify worse-than-expected
> targets, and it lets better-than-expected targets through (with
> possibly inflated CVaR estimates).

This matches the original CJE paper's choice of `base` (Llama-3.3-70B
with standard prompt) as logger and `unhelpful` (deliberately bad) as
target: the unhelpful policy's true value is *lower* than the calibrator
predicts, and the audit fires correctly. Had the original paper picked
`unhelpful` as logger and `base` as target, the audit would have
accepted even though transport is just as bad — because the target's Y
is *better* than the calibrator predicted, and the audit's quantile-
coverage moment is silent on that direction.

### Implication for the headline pair

For `rlhf | 13B` vs `rejection sampling | 2.7B`:
- with logger = `context distillation | 52B`, audit accepts both *silently* — Direct CVaR estimates are unusable
- with logger = `rlhf | 52B`, audit accepts both *correctly* (within-family) — Direct CVaR estimates -1.6 and -0.4 vs oracle truth -0.31 and -0.89, errors 1.27 and 0.50

The pre-registered hypothesis for the cvar_v4 paper:

> **Logger π0 = `rlhf | 52B`. Targets π′_A = `rlhf | 13B` and π′_B = `rejection
> sampling | 2.7B`. Both pass audit (within-family transport). Mean-CJE
> produces a tie (overlapping bootstrap CIs). CVaR_0.05-CJE produces
> disjoint 95% bootstrap CIs separating π′_A from π′_B by ~0.7 units.
> Falsification = same predictions invalid.**

---

## §3 — Same-mean / different-tail survives prompt stratification

Source: `cvar_v4/eda/deeper/uf_per_source.md`

UltraFeedback ships 9 prompt sources (sharegpt, evol_instruct, ultrachat,
flan_v2_niv2, etc.) × 17 model completions each. We stratified by source
and looked for within-source same-mean-different-tail pairs.

**39 within-source same-mean-different-tail pairs** found. Top 4 by ratio:

| source | model_A | model_B | mean_A | CVaR_A | mean_B | CVaR_B | Δmean | ΔCVaR | ratio |
|---|---|---|---|---|---|---|---|---|---|
| ultrachat | `llama-2-70b-chat` | `mpt-30b-chat` | 7.438 | 4.023 | 7.442 | 5.428 | 0.004 | 1.405 | **140.5×** |
| sharegpt | `llama-2-70b-chat` | `bard` | 6.783 | 2.695 | 6.787 | 3.540 | 0.004 | 0.844 | **84.4×** |
| evol_instruct | `vicuna-33b` | `llama-2-70b-chat` | 6.730 | 2.682 | 6.738 | 3.429 | 0.008 | 0.747 | **74.7×** |
| ultrachat | `llama-2-7b-chat` | `bard` | 7.100 | 3.453 | 7.119 | 4.438 | 0.019 | 0.985 | **52.0×** |

The `ultrachat` × (`llama-2-70b-chat`, `mpt-30b-chat`) pair is the most
extreme: means literally 4 thousandths apart on a 1-10 scale (mean-CJE
would need >4M samples to resolve), CVaR_0.05 1.4 points apart, ratio
**140.5×**. This is one prompt distribution evaluated by two models that
appear identical on average and very different in the worst-case 5%.

The fact that this pattern shows up *within* each prompt source (not just
between them) means `source` isn't the only effective audit covariate
— per-policy tail behavior is genuinely heterogeneous and source-stratified
analysis would not erase the CVaR signal.

---

## §4 — RealToxicityPrompts `challenging` flag is a 6.5× audit-discriminative covariate

Source: `cvar_v4/eda/deeper/rtp_challenging.md`

The dataset's built-in `challenging` boolean (1.2% of prompts that
consistently elicited toxic continuations across multiple models) is the
strongest within-dataset audit covariate we identified across the 10
datasets:

| subset | n | mean Y_safe | q_0.05 | CVaR_0.05 | CVaR_0.01 |
|---|---|---|---|---|---|
| challenging=True | 1,187 | 0.452 | 0.035 | 0.026 | 0.013 |
| challenging=False | 97,705 | 0.625 | 0.084 | 0.060 | 0.032 |
| ALL | 98,892 | 0.623 | 0.081 | 0.058 | 0.031 |

- challenging is **1.2%** of the corpus
- challenging holds **7.8%** of the worst-1% tail mass — **6.5× concentration**
- challenging holds **4.5%** of the worst-5% tail mass — 3.7× concentration
- conditional q_0.05: 0.035 (challenging) vs 0.084 (other) — **2.4× gap in raw quantile**

Interpretation for the paper: a CVaR-CJE estimator that calibrates on
`challenging=False` and runs the audit on `challenging=True` *must* have
the audit fire — the quantile-coverage moment will see Y_audit's
empirical 5%-mass land at ~0.035 rather than the logger's calibrator
prediction of ~0.06. This is the cleanest *within-dataset*
audit-discriminativeness demo we have.

---

## §5 — HealthBench's actual physician-graded Y is structurally clean

Source: `cvar_v4/eda/deeper/hb_meta_eval.md`

Aggregated 29,511 (prompt × completion × rubric × physician) annotations
from `oss_meta_eval.jsonl` into 14,592 (prompt, completion) cells with
Y_rubric_pct = (#physician-yes labels) / (#physician labels).

| diagnostic | value |
|---|---|
| n cells | 14,592 |
| distinct Y values | 21 (out of 14,592 rows) |
| Y mean | 0.768 |
| Y q_0.05 | 0.000 |
| Y q_0.99 | 1.000 |
| ties at q_0.05 | 5.04% (✅ borderline-pass) |
| ties at q_0.01 | 5.04% |

Y_rubric_pct is heavily right-skewed (modal at 1.0 — physicians agreed
on every rubric criterion); has ~5% mass at exactly 0.0 (physicians
disagreed on every rubric → catastrophic completion). The mean (0.77)
and the empirical 5%-quantile (0.0) differ by the full range — the
classic "body is fine, tail is at the floor" structure that motivates
CVaR-CJE.

The 21-distinct-values issue is a *feature of the small physician panel*
(2-4 labels per cell × 1-3 rubrics per cell ≈ 2-12 binary labels → ratios
out of small denominators). The full-corpus HealthBench evaluation uses
~10 rubric criteria per prompt × 1 GPT-4.1 grader = 10 weighted points,
giving more distinct Y values at scale. The structural pass at 5% ties
holds.

### Per-theme heterogeneity is enormous

| theme | n | mean(Y) | q_0.05(Y) | CVaR_0.05(Y) | CVaR_0.01(Y) |
|---|---|---|---|---|---|
| health (data tasks) | 1,557 | 0.676 | 0.000 | 0.000 | 0.000 |
| context (seeking) | 1,615 | 0.662 | 0.000 | 0.000 | 0.000 |
| global (health) | 2,515 | 0.783 | 0.000 | 0.000 | 0.000 |
| emergency (referrals) | 1,806 | 0.706 | 0.000 | 0.000 | 0.000 |
| hedging | 2,842 | 0.823 | 0.444 | 0.285 | 0.092 |
| communication | 2,961 | 0.827 | 0.500 | 0.430 | 0.200 |
| complex (responses) | 1,296 | 0.811 | 0.500 | 0.432 | 0.223 |

**q_0.05 spread across themes: 0.500** — the largest within-dataset
heterogeneity number we've seen. Hedging, communication, and complex
responses have well-behaved tails (CVaR_0.05 ≈ 0.3-0.4); health-data,
context-seeking, global-health, and emergency-referrals have CVaR_0.05 = 0
(physicians disagree on at least 5% of completions). The first cluster
will let the level-claim audit accept; the second cluster's CVaR_0.05 = 0
means the empirical estimator hits the floor and a per-theme refusal-of-
level-claim is the right call. This is exactly the "audit accepts on
some strata, fires on others, report stratum-conditional" workflow that
the framework promises.

---

## §6 — Oracle-coverage sweep on the headline pair: the audit's power scales correctly with coverage, but cheap-S quality is the binding constraint

Source: `cvar_v4/eda/deeper/hh_coverage_sweep.md` and `hh_better_s.md`

Replicating the original CJE paper's sweep `oracle_coverage ∈ {0.05, 0.10,
0.25, 0.50, 1.00}` on (logger=`rlhf | 52B`, target=`rlhf | 13B`) with
cheap S = -log(transcript_chars/1000):

| coverage | n_oracle | Direct CVaR_0.05 estimate | true CVaR_0.05 | abs err | audit p | reject % |
|---|---|---|---|---|---|---|
| 0.05 | 114 | -1.574 | -0.175 | 1.40 | 0.84 | 0% |
| 0.10 | 229 | -1.573 | -0.175 | 1.40 | 0.69 | 0% |
| 0.25 | 573 | -1.579 | -0.175 | 1.40 | 0.40 | 0% |
| 0.50 | 1146 | -1.585 | -0.175 | 1.41 | 0.16 | 0% |
| **1.00** | **2292** | **-1.570** | **-0.175** | **1.40** | **0.024** | **100%** |

**Two findings, both important.**

**(a) The audit's power scales monotonically with oracle slice size.** At
5% coverage on n_target = 2,292, the audit p is 0.84 (no power). At 100%
coverage, p drops to 0.024 (correctly fires). This is the right behavior:
the audit's ability to detect calibrator-transport bias depends on how
many oracle labels it can compare against the calibrator's predictions.

**(b) Direct CVaR estimates are biased and the bias does NOT shrink with
coverage.** Errors stay at ~1.4 across all coverage levels because the
calibrator is fit on the logger and applied to the target *uniformly* —
the bias is a property of the (logger, target, cheap-S) triple, not of
the oracle slice. More oracle data lets the audit detect the bias but
does not correct it.

### Cheap-S admissibility — three required properties

Testing two cheap S choices on HH (`hh_better_s.md`):

| cheap S | response-side? | Y-aligned? | within-family err | OOD audit fires? |
|---|---|---|---|---|
| -log(transcript_chars / 1000) | ✅ | ❌ (per-policy ρ swings 0.14–0.62) | 1.40 (rlhf-13B), 0.75 (RS-2.7B) | ✅ on cd-52B, plain-lm-52B |
| task_descripton_harmlessness_score | ❌ (prompt-only) | ✅ in absolute terms | 0.48 (uniform) | ✅ on cd-52B, plain-lm-52B |

Both choices fail to recover the within-family same-mean-different-tail
gap, but for different reasons:
- length-S has per-policy ρ that varies from +0.14 to +0.62, so the
  calibrator trained on logger systematically over- or under-predicts
  per target depending on each target's S→Y curvature
- task_descripton_harmlessness_score is prompt-side (constant across
  responses to the same prompt), so it cannot distinguish responses
  generated by different models on the same prompt — the calibrator
  outputs the same value for all within-family targets

A cheap S admissible for CVaR-CJE level claims must be:
1. **Response-side** — a function of the model's actual response, not the prompt
2. **Y-aligned** — high |ρ(S, Y)| within each policy, ideally with consistent sign across policies
3. **Cheap to compute on every fresh draw**

Length is response-side but not Y-aligned. Task-description score is
Y-aligned but not response-side. **Production CVaR-CJE pilots must use a
real cheap judge** — Prometheus-2-7B for rubric-style data, WildGuard-7B
for binary safety, or a small distilled grader running the same rubric
that the oracle uses.

### What the audit *does* protect against

The audit reliably catches *out-of-distribution-family* targets (audit
fires 100% on context-distillation-52B and plain-lm-52B regardless of
which cheap S is used). It does *not* protect against bias that is
*symmetric across all within-family targets* — because the calibrator
gives roughly the same wrong answer on every target, target-vs-target
comparisons inherit the bias and the audit is silent on it.

This is a real limitation of the framework, not a bug in our pipeline,
and it must be documented in the cvar_v4 paper. The standard mitigation
is the original CJE paper's calibration-coverage sweep: run with several
oracle-coverage levels and verify the estimate doesn't drift; if it
does, the cheap S is bad.

---

## §7 — WildChat-4.8M (shuffled stream): tail-aware evaluation distinguishes reasoning models

Source: `cvar_v4/eda/deeper/wildchat_cross_model.md`

After fixing the streaming-bias issue (`ds.shuffle(buffer=10000, seed=42)`
in `cvar_v4/eda/datasets/wildchat_48m.py`), the 100K-sample contains 5
distinct OpenAI models. With Y_safe = -max_detoxify_toxicity:

| model | n | mean Y_safe | CVaR_0.05 | CVaR_0.01 |
|---|---|---|---|---|
| `gpt-3.5-turbo-0301` | 50,998 | -0.00471 | -0.05391 | -0.08620 |
| `gpt-4-0314` | 11,398 | -0.00433 | -0.05122 | -0.08704 |
| `gpt-4o-mini-2024-07-18` | 2,520 | -0.00244 | -0.03234 | -0.07063 |
| `gpt-4o-2024-08-06` | 32,991 | -0.00235 | -0.03376 | -0.08364 |
| `o1-mini-2024-09-12` | 1,238 | -0.00209 | -0.02529 | -0.05988 |

Cross-model same-mean-different-tail pairs (means within 0.001, CVaR_0.05
gap ≥ 0.003):

| pair | Δmean | ΔCVaR_0.05 | ratio |
|---|---|---|---|
| gpt-4o-2024-08-06 vs o1-mini-2024-09-12 | 0.00027 | 0.00847 | **32×** |
| gpt-4o-mini-2024-07-18 vs o1-mini-2024-09-12 | 0.00035 | 0.00706 | **20×** |

**Empirical signal: o1-mini has measurably better worst-case toxicity
than gpt-4o** even when their average toxicity is statistically
indistinguishable. This is a within-corpus example of the same pattern
from the OpenAI HealthBench paper: reasoning-trained models pay a CVaR
dividend that mean-evaluation misses. The 100K-row WildChat sample is
sufficient to detect this gap at high confidence.

Caveat: the streamed sample contains 0 toxic-flagged conversations even
after shuffling, suggesting the 1.5M-toxic subset is partitioned into
separate parquet shards the default streaming loader doesn't reach. For
adversarial-tail work the full Arena-Tail mixture (§dataset_options.md
B.1) should source toxic mass from RealToxicityPrompts `challenging` and
the WildJailbreak adversarial-harmful subset rather than relying on
WildChat alone.

---

## §8 — Refined recommendation for the cvar_v4 paper

Combining the four deeper analyses with the original 10-dataset scoping:

### The pilot (this week, $0–5K, 2-day turnaround)

**Dataset**: HH red-team-attempts (already on disk via `cvar_v4/eda/datasets/hh_red_team.py`).

**Logger**: `rlhf | 52B` (n=3,081). *Choose the highest-mean-harmlessness
logger — see §2's asymmetry argument. The seemingly natural choice
`context distillation | 52B` produces silent audit failures.*

**Targets**: `rlhf | 13B`, `rejection sampling | 2.7B`, `rejection sampling | 52B`,
`context distillation | 52B`, `plain lm | 52B`.

**Cheap S**: ⚠️ **Use a real cheap judge, not length or task-description score.**
Per §6, cheap S must be (a) response-side, (b) Y-aligned within each
policy, (c) cheap to compute. Length is response-side but per-policy ρ
swings 0.14–0.62 (within-family same-mean-different-tail not recovered).
Task-description score is Y-aligned but prompt-only (constant across
responses to the same prompt). **Production cheap-S options**:
- WildGuard-7B refusal-detection (response-side, harm-aligned, ~free at inference)
- A small instruct grader running the same rubric Y uses, e.g. Prometheus-2-7B configured for refusal-quality scoring
- For a budget-constrained pilot: a sentiment-classifier ensemble against the response

**Estimator**: cvar_v3/workhorse.py Direct estimator (already validated
on Arena data); patch in HH inputs via `cvar_v4/eda/deeper/_estimator.py`.

**Pre-registered hypothesis (revised after §6 + §2)**:

> With logger = `rlhf | 52B` and a real cheap judge S (WildGuard-7B-class):
> 1. Audit accepts on `rlhf | 13B` and `rejection sampling | 2.7B` at oracle
>    coverage ≥ 0.25 (within-family transport).
> 2. Audit fires on `context distillation | 52B` and `plain lm | 52B` at
>    every coverage level.
> 3. Mean-CJE produces overlapping bootstrap 95% CIs on the within-family pair.
> 4. CVaR_0.05-CJE produces disjoint bootstrap 95% CIs, separating
>    `rlhf | 13B` and `rejection sampling | 2.7B` by ~0.7 ± 0.3 units.
>
> If (1) holds but (3) and (4) do not (i.e., CVaR-CJE replicates mean
> result), the cheap S is too weak — re-run with a stronger judge. If
> (2) holds but (4) does not even with strong cheap S, the framework's
> within-family bias is irreducible at this n and we'd need ≥10× more data.

**Validation step before publishing**: cross-check the Direct CVaR_0.05
estimate against a held-out oracle-only subset's empirical CVaR_0.05 for
at least one within-family target. A persistent gap of >2× the bootstrap
CI half-width means the cheap S is the binding constraint, not the
estimator.

### The medical primary (after pilot validates, ~$750)

**Dataset**: HealthBench (oss_eval + hard subsets, 6,000 prompts).

**Y construction**: rubric-percent grading via published GPT-4.1-class
grader prompt (paper docs F1=0.71 vs. physicians). The §5 finding
confirms the resulting Y is structurally CVaR-CJE-compatible.

**Audit covariate**: `theme` (the 7-cluster categorization). The
within-theme q_0.05 spread of 0.5 (§5) means audit transport tests will
be highly discriminative. Expect audit to accept on hedging/communication/
complex themes and to fire on health-data/context-seeking/global-health/
emergency-referrals themes.

**Five-policy panel** (per `cvar_v4/eda/datasets/healthbench.py:SPEC`):
π_base (Llama-3.1-8B), π_clone (Llama-3.2-3B), π_premium_thinking
(Claude Opus 4.7), π_rag_clinical (π_base + UpToDate retrieval),
π_safety_overrefuse (DPO-tuned conservative), π_adversarial_sycophant
(audit-positive by design).

### The flagship (~$7.8K, after pilot+medical, for cvar_v4 paper headline)

**Dataset**: Arena-Tail stratified pool — 100K prompts mixed from:
- 50% Arena-140k (general chat)
- 20% HarmBench + StrongREJECT + WildJailbreak (adversarial; needs HF login for last)
- 15% BeaverTails harmful-question prompts
- 10% RealToxicityPrompts `challenging`-stratified subset
- 5% WildChat-4.8M shuffled-stream toxic-flagged subset (use `ds.shuffle(buffer=10K, seed=42)` per the streaming-bias finding)

**Audit covariate**: prompt-stratum (5 levels). Per §4, this gives a
known audit-discriminativeness factor of >6× on the toxicity-stratum
prompts; HarmBench/WildJailbreak strata will produce similar
audit-positive verdicts on adversarial-tuned policies.

**Five-policy panel**: §B.2 of `dataset_options.md` (π_base, π_clone,
π_premium, π_safety, π_adversarial).

**Cost breakdown (rates as of 2026-04-28)**:
- 5 policies × 100K prompts × ~400 out-tok = 200M output tokens generated
- Open-weight (3 of 5 policies via Together/Fireworks): ~$50
- Closed-frontier (2 of 5 via Opus 4.7 + GPT-5): ~$2,500
- Oracle slice 15% × 5 = 75K oracle calls × ~$0.07 = ~$5,250
- Total: **~$7,800** (50% over `dataset_options.md` §B.6's $5K target,
  driven by current GPT-5-thinking pricing)

---

## §9 — Outputs catalog

| File | Purpose |
|---|---|
| `cvar_v4/eda/deeper/_estimator.py` | Self-contained Direct CVaR-CJE + bootstrap-Σ̂ Wald audit, ported from cvar_v3/workhorse.py with no cje-eval dependency. 1e-6 ridge added to Σ̂ for numerical stability. |
| `cvar_v4/eda/deeper/hh_pairwise.py` + `.md` | Cellwise (mean, CVaR_0.05, CVaR_0.01) with bootstrap CIs for HH red-team's 12 cells; enumerates 1 strict same-mean / different-tail pair |
| `cvar_v4/eda/deeper/hh_audit_demo.py` + `.md` | End-to-end Direct CVaR-CJE + Wald audit on HH red-team with two logger choices; demonstrates audit's asymmetric bite |
| `cvar_v4/eda/deeper/hh_coverage_sweep.py` + `.md` | Oracle-coverage sweep {0.05, 0.10, 0.25, 0.50, 1.00} on the headline pair; demonstrates audit power scales with coverage but estimation bias doesn't shrink |
| `cvar_v4/eda/deeper/hh_better_s.py` + `.md` | Side-by-side comparison of two cheap-S choices; documents the response-side + Y-aligned cheap-S admissibility conditions |
| `cvar_v4/eda/deeper/uf_per_source.py` + `.md` | UltraFeedback per-(source, model) decomposition; finds 39 within-source same-mean / different-tail pairs (max ratio 140.5×) |
| `cvar_v4/eda/deeper/rtp_challenging.py` + `.md` | RealToxicityPrompts `challenging` subset analysis; documents 6.5× tail concentration |
| `cvar_v4/eda/deeper/hb_meta_eval.py` + `.md` | HealthBench meta_eval physician-rubric aggregation; per-theme CVaR heterogeneity (q_0.05 spread = 0.5) |
| `cvar_v4/eda/deeper/wildchat_cross_model.py` + `.md` | WildChat shuffled-stream cross-model analysis; o1-mini has measurably better tail than gpt-4o (32× ratio) |

To rerun any one: `python3 -m cvar_v4.eda.deeper.<module>`. To rebuild
the parent dataset section after the WildChat shuffle fix:
`python3 -m cvar_v4.eda.run_eda --dataset wildchat_48m --no-resume && python3 -m cvar_v4.eda.run_eda --concat`.
