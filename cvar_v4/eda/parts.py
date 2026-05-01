"""Hand-written Part 1 (executive table) and Part 3 (recommendation) for scoping_data.md.

Kept separate from run_eda.py so the auto-concat path is purely mechanical
(`run_eda.py --concat` glues sections; this module supplies the bookend prose).
"""
from __future__ import annotations

PART_1 = """## Part 1 — Executive comparison table

Each row is one of the 10 datasets we ran the 20-block EDA + scoping recipe
against. Columns are derived directly from the per-dataset blocks (Block 6 for
n_usable, Block 9 for the tie verdict at q_0.05, Block 10 for CVaR/mean ratios,
Block 11 for multi-policy nativity, Block 19/20 for the pricing and feasibility
estimates). Verdicts use the four-tier vocabulary:
**primary** (CVaR-CJE-ready as Y target),
**secondary** (Y borderline; usable with smoothing or as supporting evidence),
**prompt-source** (Y unsuitable; useful only as a stratified prompt pool),
**deferred** (gating, login, or streaming-bias caveat).

| # | Dataset | n_usable | Best Y candidate | Y type | tie@q_0.05 | CVaR_0.05 half-width | multi-policy | gen $ | oracle 15% $ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmBench + StrongREJECT | 713 | _(none in data)_ | n/a | n/a | n/a (no Y) | no | $43 | $37 | **prompt-source** (adversarial seed) |
| 2 | HealthBench | 6,000 | total_positive_points _(rubric weights, proxy for true rubric-% Y)_ | continuous | 5.80% ⚠️ | 1.4% ✅ | no (gen needed) | $369 | $378 | **primary medical** (borderline tie; Y is continuous after grading) |
| 3 | RealToxicityPrompts | 98,892 | continuation_toxicity | continuous | **2.70% ✅** | 1.7% ✅ | no (gen needed) | $257 | $5,192 | **primary toxicity** |
| 4 | UltraFeedback | 255,612 | overall_score | "continuous" but quantized | 5.28% ⚠️ | 0.3% ✅ | **yes (17 models)** | $0 (in-data) | $45,626 | **secondary** (Likert-derived ratings fail; overall_score borderline) |
| 5 | Anthropic HH red-team-attempts | 38,961 | min_harmlessness_score_transcript | continuous | **0.78% ✅** | **1.0% ✅** | **yes (12 cells: 4 model_types × 3 sizes)** | $0 (in-data) | $4,909 | **★ PRIMARY — ready-made CVaR-CJE testbed** |
| 6 | BeaverTails | 327,752 | safety_score (synthetic from is_safe + n_harm) | discrete-11 | 7.15% ⚠️ | 0.1% ✅ | no | $3,637 | $17,207 | **prompt-source** (Y too discrete for primary) |
| 7 | WildJailbreak | DEFERRED | _(gated; login needed)_ | — | — | — | — | — | — | **deferred** — `hf auth login` + terms acceptance |
| 8 | PKU-SafeRLHF | 164,196 | severity_level | Likert-4 | **48.41% ❌** | (degenerate) | yes (3 cells) | $2,628 | $5,172 | **prompt-source** (severity is too coarse) |
| 9 | Arena-140k | 271,149 | winner_score / response_tokens / if_score | binary / pairwise / Likert | **47.85% ❌** | (degenerate) | yes (53 models) | (would need fresh gen) | $150,893 | **prompt-source** (the user's anchor; pairwise label only — re-elicit Y) |
| 10 | WildChat-4.8M (100K stream) | 99,081 | max_detoxify_toxicity | continuous in principle | 49.96% ❌ in sample | (sample-biased) | yes (2 in 100K) | $5,536 | $2,081 | **deferred-streaming** — first-100K sample is non-toxic-only; rerun with `ds.shuffle(buffer=10K)` to get representative |

**Reading the table**: the only dataset that satisfies every CVaR-CJE
admissibility property simultaneously — continuous Y, structurally sound at
q_0.05, multi-policy native, n large enough to resolve CVaR_0.01, no fresh
generation cost — is **HH red-team-attempts**. This is the literal claim of
`dataset_options.md` C-row, now empirically verified row-by-row.

For a *medical-domain* primary the next-best fit is **HealthBench**, where Y
will be continuous once we grade fresh policy responses against the published
rubric (the per-row rubric-weight proxy is a borderline structural pass at
q_0.05). For a *toxicity-tail* primary the next-best fit is
**RealToxicityPrompts**, which uniquely combines continuous Perspective Y with
a built-in `challenging` audit covariate.

The remaining datasets are most useful as **stratified prompt sources** for an
Arena-Tail-style pool (see Part 3): each one passes some of the six A.1
criteria but fails on Y discreteness or pairwise-only structure, so they
require fresh S/Y elicitation before they could host CVaR-CJE primary.

"""


PART_3 = """## Part 3 — Recommended pipeline + pre-registered hypothesis

### 3.1 Comparison to the original CJE paper

The 2025 Causal Judge Evaluation paper (`arxiv_source/sections/experiments.tex`,
arXiv:2512.11150) ran its empirical demonstration on:

- **n = 4,961 Arena prompts** (after TF filtering of the original 5K Arena
  slice) with `metadata.judge_score` ∈ [0,1] (GPT-4.1-nano) and
  `metadata.oracle_label` ∈ [0,1] (GPT-5).
- **5 policies**: `base` (Llama-3.3-70B, π0), `clone` (same model + same
  prompt, different seed; TF-reliability test), `premium` (Llama-3.1-405B;
  size-scaling), `parallel_universe_prompt` (modified system prompt;
  prompt-engineering), `unhelpful` (deliberately low-quality; adversarial
  stress test). ~25K total responses (5 × 5K prompts).
- **Oracle coverage sweep** {0.05, 0.10, 0.25, 0.50, 1.00}; headline at 0.25.
- **Sample-size sweep** {250, 500, 1000, 2500, 5000}; headline at 5,000.
- **`unhelpful` excluded from accuracy/uncertainty metrics** (intentionally
  violates transportability), but *included in ranking metrics* — testing
  whether estimators correctly identify the adversarial policy as worst.
- **Headline result**: `direct+cov` achieves 99% pairwise ranking accuracy at
  full sample with 5% oracle coverage; CIs capture ground-truth value for the
  4 transportable policies and the audit correctly flags `unhelpful`.
- **Honest limitations the paper documents**: TTC ranges 0.19–0.49 for
  non-clone policies (all below the 0.7 threshold); raw ESS for the clone
  policy is only ~26% (teacher-forcing brittleness from tokenization
  mismatch / API non-determinism); coverage-limited efficiency inflates SEs
  by 24–61× even with stabilized weights.

The paper's design template is **exactly** the §B.2 5-policy panel that
`dataset_options.md` proposes. Our scoping inherits this template and
generalizes from mean V(π′) to CVaR_α with two structural changes:

1. **Y must be continuous, not just numeric**: CVaR_α has a non-smooth
   influence function at the empirical quantile; ties at q_α inflate variance
   and degenerate the calibrator's transport audit. The 11-quantile Block 9
   tie-fraction check is the empirical screen for this — Likert and pairwise
   datasets fail it loud and clear (Arena-140k 47.85%; PKU-SafeRLHF 48.41%;
   UltraFeedback rating columns 10-12%; even BeaverTails at 7.15% is
   borderline).
2. **Tail-region oracle support matters more than overall TTC**: for CVaR_α
   estimation we need the 5%-or-1% tail of *each target policy* to be
   covered by oracle labels, not just the body. The Block 14 stratification
   heterogeneity gives a forward-looking diagnostic: large q_0.05 spread
   across an audit covariate (HH model_type spread = 3.94, HealthBench theme
   spread = 11) signals the audit will be discriminative; small spread
   (Arena-140k language, ~0) signals it won't.

### 3.2 Recommended pilot — HH red-team-attempts as the immediate workhorse

**Pilot proposal**: run the existing `cvar_v3/workhorse.py` Direct estimator
on HH red-team-attempts (38,961 rows × 12 (model_type, num_params) cells).

- π0 = `plain LM | 52B` (1,414 rows; the un-aligned baseline; produces the
  worst tail; matches the paper's "weakest baseline" pattern).
- π′ panel:
  - π_clone = `plain LM | 13B` (2,587 rows; size-ablation of π0)
  - π_premium = `RLHF | 52B` (3,081 rows; full safety stack — best mean,
    best tail; corresponds to paper's `premium`)
  - π_parallel_universe = `context distillation | 52B` (17,739 rows;
    HHH-prompted; matches paper's `parallel_universe_prompt` slot)
  - π_safety = `rejection sampling | 52B` (1,273 rows; best q_0.05 = 0.19;
    might be over-aggressive — different audit-failure axis)
  - π_unhelpful_proxy = `plain LM | 2.7B` (1,151 rows; smallest plain LM —
    weakest tail; corresponds to paper's `unhelpful` slot)
- S (cheap judge) = re-elicit a smaller cheap-S model (WildGuard-7B or
  Detoxify) over each transcript so we get a budget-controlled cheap S.
- Y (oracle) = `min_harmlessness_score_transcript` (the dataset's own
  preference-model score; validated against human preferences in Bai et al.
  2022). This *is* a high-quality oracle; the paper's GPT-5 oracle is even
  stronger but the dataset's score is fit-for-purpose.
- Oracle coverage sweep {0.05, 0.15, 0.25, 1.00}; headline at 0.15 (matching
  the elevated coverage CVaR-CJE wants vs. the paper's 0.05).
- Cost: ~$5K oracle annotation if we re-elicit; **$0 if we accept the
  in-data preference-model score as Y**.
- Time-to-first-result: 1 day (re-using cvar_v3 pipeline) — most of the
  CJE-2026 paper's stack works as-is on this data.
- Falsifiable hypothesis: see §3.5 below.

### 3.3 Recommended primary medical demonstration — HealthBench

**Primary medical proposal** (matching dataset_options.md §B.6 — pilot scale):

- 6,000 prompts (5K oss_eval + 1K hard) × 5-policy panel × ~400 out-tok/row.
- π_base = Llama-3.1-8B-Instruct (clinical-tuned baseline).
- π_clone = Llama-3.2-3B-Instruct.
- π_premium_thinking = Claude Opus 4.7 thinking (frontier reasoning).
- π_rag_clinical = π_base + RAG over UpToDate / NICE guidelines (medical
  domain-specialist).
- π_safety_overrefuse = π_base + DPO toward conservative refusal
  (audit-positive on responding-under-uncertainty theme).
- π_adversarial_sycophant = π_base + 'agree with user' system prompt
  (audit-positive on rubric).
- S = Prometheus-2-7B with HealthBench rubric (open-weight, ~free).
- Y = GPT-5-thinking grader using the published HealthBench grader prompt;
  optional multi-judge consensus across {Opus 4.7, GPT-5} on a slice.
- Cost: **~$369 generation + $378 oracle @ 15%** = **~$750 total**.
- Worst-at-k validation: HealthBench's authors document GPT-5 worst-at-16 ≈
  46% Hard subset vs. GPT-4o ≈ 0% — our CVaR_α estimates should reproduce
  the same gap on the same subset, validating the method against an
  external worst-case-aware benchmark.

### 3.4 Recommended flagship demonstration — Arena-Tail stratified pool

**Flagship proposal** (matching dataset_options.md §B.6 — full scale):

A stratified prompt pool of ~100K total prompts assembled from:

| Stratum | Source | Share | Diagnostic from this scoping |
|---|---|---|---|
| General chat | Arena-140k user prompts | 50% | 271K available; multi-policy via 53 in-data models or fresh gen |
| Adversarial / jailbreak | HarmBench + StrongREJECT + WildJailbreak (post-login) | 20% | 713 from §1 + ~262K from WildJailbreak after auth |
| Safety-sensitive direct | BeaverTails harmful-question prompts | 15% | 327K available; clean 14-class harm strata |
| Toxicity continuum | RealToxicityPrompts `challenging` subset | 10% | ~1.2K with continuous Perspective Y; built-in audit-discriminative |
| Naturalistic toxic | WildChat-4.8M toxic subset (filtered) | 5% | 1.5M toxic conversations; needs shuffled streaming pull |

5 policies × 100K prompts × ~400 out-tok = **500K responses, ~83M output
tokens**. At 2026-04-28 pricing:

- Open-weight (3 policies via Together): ~$50 total.
- Closed-frontier (2 policies via OpenAI / Anthropic / Google): ~$2,500 total.
- Oracle slice @ 15% × 5 policies = 75K oracle calls × ~$0.07 = **~$5,250**.
- **Total flagship cost ≈ $7,800**, within the $5K target in
  `dataset_options.md` §B.6 by ~50% — primarily because the GPT-5 reasoning
  rate ($10/M-in, $40/M-out) is higher than the original $5K estimate
  assumed.

### 3.5 Pre-registered falsifiable hypothesis

For the **HH red-team-attempts pilot**, before running any estimator, we
predict (from Block 14's empirical q_0.05 spread):

1. **Mean ordering**: rejection_sampling_52B > rlhf_52B > context_distillation_52B
   > plain_lm_52B (in `mean(Y)`).
2. **CVaR_0.05 ordering**: same. *(We expect ranking to match — the
   intervention-quality differences are large enough to dominate
   tail-vs-mean differences here.)*
3. **CVaR_0.01 ordering**: same with one possible exception — the
   rejection-sampling tail might cross the RLHF tail at α=0.01 because
   rejection-sampling discards low-scoring samples whereas RLHF reshapes
   the policy. This is the interesting hypothesis worth testing.
4. **Audit transport on plain LM**: the paper's audit should *reject*
   transport for the plain-LM-52B policy when we calibrate on rlhf_52B as
   reference (because plain LM tail spreads to -7.5 while rlhf tail caps at
   ~-1.5 — distributional mismatch on the tail). The audit's discriminative
   value is that it should *not* reject for the rejection-sampling vs RLHF
   pair (similar tails).
5. **n_per_policy floor**: at the 1,151-row floor (`plain LM | 2.7B` cell),
   CVaR_0.01 has 11 effective tail samples. The Wald CI half-width for that
   cell will be the binding precision constraint and will dominate the
   pilot's reportable α.

**Falsification criteria**: if mean ordering does *not* hold (e.g., if RLHF
at 52B does worse mean than the unaligned plain LM), we have a calibrator
or weighting bug — the paper's pipeline already validates this case. If
CVaR ordering is wildly different from mean ordering on a dataset where
mean and tail should match (i.e., big intervention effects), the CVaR
estimator has a tail-influence-function bug worth diagnosing before
moving to other datasets.

### 3.6 What changed vs. dataset_options.md's recommendations

| dataset_options.md said | Empirical scoping found |
|---|---|
| HealthBench is "the strongest single match" | Confirmed for medical domain; structural-tie at q_0.05 is borderline (5.80%) — workable |
| HH red-team-attempts is "the cleanest small-scale validation" | **Strongly confirmed.** It is the only dataset that passes every CVaR-CJE structural check unambiguously |
| BeaverTails has "severity levels mappable to continuous" | Partially: severity is binary (`is_safe`); we synthesize `safety_score` ∈ [-12, 1] from harm-class counts, but it's still discrete-11 and ties at q_0.05 are borderline (7.15%) |
| PKU-SafeRLHF has "19-class harm severity levels" | The 19 classes are present but `severity_level` is integer 0–3 with 48% mass at 0 — STRUCTURAL FAIL for primary CVaR use |
| Arena-100k/140k is "natural extension of original CJE" | Confirmed as **prompt source**; the pairwise label structure is fundamentally incompatible with CVaR_α as a per-policy scalar value (CVaR of a difference ≠ difference of CVaRs) |
| WildChat-4.8M is "naturalistic and largest" | Streaming-first-100K hit only the non-toxic prefix; needs shuffled pull. The full 4.74M would resolve this but at ~30 GB+ disk |

### 3.7 Caveats noted during execution

- **WildChat-4.8M streaming bias**: the first 100K rows are dataset-ordered
  by some key (likely model + timestamp); the toxic-flagged subset is
  sparse early and concentrated later. Re-run with
  `ds = ds.shuffle(buffer_size=10_000, seed=42)` before iterating to get
  a representative sample, or pull from later parquet shards directly.
- **Tie-fraction heuristic eps**: Block 9 uses `eps = 0.5% × range(Y)` to
  define the tie window. For Y with a heavy right tail (Arena-140k
  response_tokens has max=137K), this eps balloons and over-flags ties.
  The verdict is still right (tokens are not a CVaR-CJE Y), but the
  diagnostic warrants per-dataset eps tuning if we ever use it as the
  primary admissibility filter.
- **Generation cost is order-of-magnitude**: actual cost depends heavily
  on response-length distribution per policy (refusal-heavy policies are
  ~5× cheaper than verbose ones); the block-20 numbers assume the dataset's
  own response-length distribution as an estimator.
- **Rate citations are 2026-04-28**: GPT-5-thinking ($10/M-in, $40/M-out),
  Claude Opus 4.7 thinking ($15/M-in, $75/M-out), Llama-3.1-8B via Together
  ($0.20/M). All numbers should be flagged stale after 2026-Q2.
- **WildJailbreak is deferred**: gating + terms-acceptance is the only
  blocker; once `hf auth login` is done, the harness module needs ~50 lines
  of polars to fold into the pipeline.

### 3.8 Bottom-line recommendation

Run **HH red-team-attempts as the pilot this week** (1-day turnaround,
~$0–5K oracle cost depending on whether we re-elicit Y, and the existing
`cvar_v3/workhorse.py` pipeline works as-is). Use the result to validate
the CVaR-CJE estimator end-to-end, then run **HealthBench** as the
medical-domain primary demonstration ($750), then build the **Arena-Tail
flagship** ($7.8K) for the cvar_v4 paper's headline result. WildJailbreak
joins the flagship pool once login is set up; WildChat-4.8M joins once
we re-pull with shuffled streaming.

The pre-registered hypothesis in §3.5 is the falsification anchor: if the
HH red-team pilot fails the predicted ordering, halt and debug the
estimator before scaling to HealthBench or the flagship pool.
"""
