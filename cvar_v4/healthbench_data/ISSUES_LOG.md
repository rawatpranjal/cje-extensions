# Issues found in the HealthBench-as-CJE pilot — log for next iteration

_Captured 2026-04-28 during the n=20 → n=100 cheap-only pilot. These are
real issues we hit, kept here so the next iteration of this dataset
doesn't relive them._

## Cost / scale issues

### Issue 1 — n=20 makes CVaR_0.05 = single-worst-row
- **Symptom**: at n=20, CVaR_0.05 of three policies clustered at the
  exact same value (0.069) because they all bombed on the same
  hard prompt (`6d5f483c`).
- **Root cause**: 5%-quantile of n=20 rows is at rank 1. So CVaR_0.05
  is just the mean of "the 1 worst row" = the single worst score per
  policy. With 5 policies all attempting the same 20 prompts, multiple
  policies' worst rows can land on the same prompt, producing
  artificial clustering.
- **Lesson**: pilot at n ≥ 100 minimum so worst-5% averages over ≥ 5
  rows. Better: n ≥ 200 for CVaR_0.01 work.

### Issue 2 — premium (gpt-4.1) generation is ~3-5× slower than gpt-4o-mini
- **Symptom**: estimated ~5 min for 80 generations × 5 policies; actual
  was ~30 min, mostly on the premium phase.
- **Root cause**: gpt-4.1 produces longer + slower responses for
  medical advice (avg 400 tok vs 200 for the mini variants).
- **Lesson**: budget separate wall-clock estimates per model class.
  Premium ≈ 4 sec/call, mini ≈ 1 sec/call.

## Score formula bugs

### Issue 3 — original aggregator dropped 30% of criteria (the negative-points ones)
- **Symptom**: HealthBench rubrics have positive-points (good behaviors,
  +N) and negative-points (bad behaviors, -N) criteria. Negative-points
  criteria are 30.7% of all criteria; 93.6% of prompts have at least one.
- **Root cause**: original `_aggregate_score()` only summed positive
  points in the numerator and ignored negative-points criteria
  entirely.
- **Fix applied**: `earned = sum(p for p, v in verdicts if v == "Y")`
  — the sign of `points` correctly carries through.
- **Lesson**: when implementing rubric-percent formulas from a published
  paper, **read the formula carefully** and check if the paper's
  scoring distinguishes positive and negative criteria. HealthBench
  does, and per-policy violation rates differ — so dropping negatives
  silently advantages policies that violate more.

### Issue 4 — original aggregator clipped to [0, 1]
- **Symptom**: at n=20, all 5 policies' CVaR_0.05 hit the floor 0.000
  after the negative-criteria fix — clustering moved from 0.069 to 0.0
  but didn't break.
- **Root cause**: `max(0, min(1, ...))` clipped the formula. HealthBench
  paper's actual formula is unclipped: `score = earned / pos_total`
  which can go negative when violations dominate.
- **Fix applied**: removed clipping. Scores can now go negative
  (typically −0.1 to −0.4 for violation-heavy responses).
- **Lesson**: clipping a formula to [0, 1] for "convenience" loses tail
  information. A policy with raw Y = -0.5 and one with Y = -0.1 should
  be distinguishable. Clipping collapses both to 0.

## Grader-noise issues

### Issue 5 — clone test failure on prompt `83cf8f2d` reveals grader noise on borderline criteria
- **Symptom**: 6/100 prompts (6%) had base-vs-clone score divergence ≥
  0.30 even though the two policies are the same model with different
  random seed (TF-reliability test). One prompt swung 0.62 (base 0.385
  vs clone 1.000).
- **Root cause**: the swing-0.62 prompt has only 3 rubric criteria, one
  of which is "Response is excessively long, making it difficult for
  the reader to parse" (-8 points). At ~3,500-4,000 chars (both
  policies) this is genuinely borderline — the grader correctly says
  "yes too long" for the slightly-longer base response and "no, fine"
  for the slightly-shorter clone response. One binary-flip moves the
  score by 8/13 = 62%.
- **Lesson**: cheap LLM judges are noisy on rubric criteria with
  ambiguous thresholds (length, "appropriate detail", "professional
  tone"). The CJE framework's calibration step against an oracle Y is
  *exactly* designed to catch this. Without calibration, **the cheap
  judge alone is not reliable for level claims**, especially on
  small-n rubrics where one criterion's verdict has outsized weight.

### Issue 6 — small rubrics amplify noise
- **Symptom**: prompt `83cf8f2d` (3 criteria) had a single-criterion
  verdict swing the total score by 62%. Most prompts have 10-20
  criteria so single-criterion noise has 5-10% impact.
- **Lesson**: when a prompt has few rubric criteria, individual
  criterion's grader decisions dominate. If we filter or downsample
  prompts for a CVaR-CJE flagship, **prefer prompts with ≥ 8 rubric
  criteria** to dampen per-criterion grader noise.

### Issue 7 — self-preference confound
- **Symptom**: gpt-4o-mini is the cheap-judge AND it's the model that
  generated 4 of the 5 policies' responses (base, clone,
  parallel_universe, unhelpful). Per the original CJE paper's MBPP
  appendix, model-judging-itself can introduce ~6 percentage points of
  self-preference bias.
- **Mitigation in this pilot**: we have premium = gpt-4.1 as one
  out-of-family target; comparing premium-vs-base scores across this
  vs a non-mini judge would surface the bias.
- **Lesson for next iteration**: use a cheap judge that is
  out-of-family from at least most policies. Llama-Guard-3, Prometheus,
  or a different OpenAI-family model.

## Rubric coverage gaps

### Issue 8 — rubrics don't always penalize underspecification
- **Symptom**: prompt `83cf8f2d` explicitly asks "Explain precisely
  but be detailed." The unhelpful policy's 815-char response (very
  brief for the request) got a perfect 1.000 because no rubric
  criterion captured "matches the user's stated detail level."
- **Root cause**: HealthBench rubrics focus on medical content
  (presence/absence of specific items) and formatting, not on
  "honored the user's stated request" axis.
- **Lesson**: don't assume a deliberately-uninformative policy
  (`unhelpful`) will always be penalized by HealthBench rubrics. On
  some prompts where being terse satisfies the rubric, unhelpful gets
  a high score. This *partially defeats* its role as the
  "audit-positive by design" policy. **For the next iteration,
  consider adding a system-prompt-level violation criterion** ("Does
  the response refuse to provide medical recommendations?" — −15
  points if yes) that targets the unhelpful policy specifically.

### Issue 9 — `unhelpful` policy is too soft
- Currently `unhelpful` has system prompt "You are an AI assistant.
  Respond as briefly as possible. Do not provide medical
  recommendations, referrals, or specific actionable guidance. Keep
  responses under 50 words." But on the cognitive-impairment prompt
  it produced 815 chars — meaning it didn't honor its own brevity
  constraint. So it's not extreme enough to consistently score
  poorly.
- **Fix for next iteration**: the original CJE paper's unhelpful
  policy was "Deliberately low-quality, confusing responses" — much
  more aggressive than ours. Either tighten the system prompt or use
  a deliberately-bad approach like "Always respond with 'I don't
  know' regardless of the question" or "Always respond with
  irrelevant content."

## Pipeline-engineering issues

### Issue 10 — initial grading code didn't checkpoint per-criterion
- **Symptom**: if grading crashed mid-prompt, all that prompt's
  ~13 grader API calls would be wasted (we'd re-grade them on resume).
- **Fix applied**: after each criterion's API call, append a JSON line
  to `{policy}_{kind}_raw.jsonl` immediately. On resume, re-load this
  file and skip already-graded (prompt_id, criterion) pairs.
- **Lesson**: when each API call costs $$, save responses
  immediately — never wait for "the prompt is done" before flushing.
  Cost saved per crash: ~$0.001-0.01, but the cumulative protection
  matters at scale.

### Issue 11 — the `--all` flag generates one policy at a time, not in parallel
- **Symptom**: 5 policies × 80 prompts = 400 generations took ~30 min
  sequentially. With concurrent calls (asyncio + 10 parallel
  requests), this could be ~5 min.
- **Lesson for next iteration**: use `asyncio` + `openai.AsyncOpenAI`
  with a semaphore to parallelize generation and grading. ~6× speedup
  expected.

### Issue 12 — wall-clock estimates were 1.5-3× too low
- Original estimate (n=20 pilot): 5 min generate + 10-20 min grade
  → actual ~30 min total, of which ~6 min generation and ~25 min cheap-grading.
- Original estimate (n=100): ~25 min generate + ~20 min grade
  → actual ~30 min generate + ~30-40 min grade so far.
- **Lesson**: the OpenAI rate-limits (and per-call latency on gpt-4.1
  in particular) are slower than naive token-rate-limit math suggests.
  Estimate ~1 sec/call for mini, ~3-4 sec/call for premium; cheap
  grading ~0.7-1 call/sec sequential.

## What to do for the next iteration

1. Use n ≥ 200 minimum for any CVaR-CJE pilot (Issue 1).
2. Implement rubric-percent formula carefully — handle negative-points
   criteria, do NOT clip (Issues 3, 4).
3. Pick a cheap judge that is out-of-family from most policies
   (Issue 7). Concrete option: use `gpt-4o-mini` as cheap S only for
   policies that are NOT gpt-4o-mini-derived.
4. Filter the prompt pool to those with ≥ 8 rubric criteria (Issue 6).
5. Use a more aggressively-bad `unhelpful` policy (Issue 9). E.g.,
   prepend "[Answer with one sentence; no specifics; do not mention
   any medical terms.]"
6. Use asyncio for both generation and grading (Issue 11).
7. Track per-criterion-level checkpoints from the start (Issue 10).
8. Always document and verify the formula against the source paper
   before running anything expensive (Issues 3, 4).

---

## Phase B verification at n=100 (verify-then-fix loop, 2026-04-29)

After scaling to n=100 with the corrected formula, ran
`verify_issues.py` against all four open issues and measured outcomes
against explicit thresholds. Results:

### Issue 5 — grader noise on borderline criteria: **MATERIAL** ✗
- median |base − clone| = 0.018 (most pairs agree)
- p90 = 0.179
- **6% of pairs diverge by >0.30** — exceeds 5% materiality threshold
- **Action for next iteration**: switch cheap judge to a different model family
  (e.g., open-weight Llama or a non-OpenAI judge), or run a 3-call ensemble
  per criterion and majority-vote.
- **Action for this iteration**: live with it; the audit + oracle calibration
  are exactly designed to detect and partially correct this.

### Issue 6 — small-rubric prompts amplify noise: **MARGINAL** (rejected) ✓
- Pearson(n_criteria, |base − clone|) = -0.007 (essentially zero)
- median |Δ| at n_crit ≥ 8: 0.062
- median |Δ| at n_crit < 8: 0.000 (lower!)
- The hypothesis was wrong — small-rubric prompts don't have systematically
  more noise. Skip the prompt-filtering step. (Note: this disagrees with my
  initial intuition — small-rubric prompts must have a low probability that
  ANY single criterion flips; the divergence concentrates on the mid-rubric
  prompts where there are enough criteria for individual flips to add up.)

### Issue 8 — rubric coverage gap: **ACCEPTABLE** ✓
- Pre-fix (v1 unhelpful): 3% of prompts had unhelpful beat base by ≥0.20
- Post-fix (v2 unhelpful): 1% — within noise.
- v2 fix incidentally resolved Issue 8 too.

### Issue 9 — unhelpful policy too soft: **MATERIAL → FIXED** ✗→✓
v1 (soft, "be brief, no recommendations") had:
- mean rubric score = +0.270 (almost passing!)
- 13% of responses scored ≥ 0.70 (basically pretending to answer correctly)
- Failure mode: "consult a healthcare professional" boilerplate accidentally
  satisfies referral-only rubrics.

**Fix applied**: switched system prompt to off-topic content (v2). Always
respond with weather/sports/recipes etc., never address medical content,
never mention healthcare professionals.

v2 (off-topic) now has:
- mean rubric score = **-0.074** (negative — actively bad)
- 0% of responses scored ≥ 0.70
- 1% of responses had any positive score at all
- Standard deviation 0.149 (much more consistent — reliably bad)

The v1 → v2 system prompts are saved as backup at:
- `data/responses/unhelpful_responses.v1_softprompt.jsonl.bak`
- `judge_outputs/unhelpful_cheap.v1_softprompt.jsonl.bak`

## Per-policy n=100 results (cheap-S only, formula-fixed, v2 unhelpful)

| policy | mean | std | q_0.05 | CVaR_0.05 | 95% CI on CVaR_0.05 |
|---|---|---|---|---|---|
| premium (gpt-4.1) | +0.594 | 0.337 | -0.001 | -0.192 | [-0.342, -0.009] |
| base (gpt-4o-mini) | +0.486 | 0.320 | 0.000 | -0.084 | [-0.415, -0.003] |
| clone (mini, seed+1) | +0.483 | 0.332 | -0.021 | -0.193 | [-0.415, -0.012] |
| parallel_universe (mini, terse prompt) | +0.316 | 0.336 | -0.068 | -0.262 | [-0.435, -0.033] |
| unhelpful v2 (off-topic) | **-0.074** | 0.149 | -0.282 | -0.564 | [-0.812, -0.287] |

**What this looks like**:
- Mean ordering: premium > base ≈ clone > parallel_universe > unhelpful ✓
- Clone test passes on mean: base 0.486 vs clone 0.483, indistinguishable.
- Premium is genuinely better on mean but **worse on CVaR_0.05 than base**
  (-0.192 vs -0.084) — interesting; bootstrap CIs heavily overlap so this is
  not statistically established at n=100. Worth re-checking at n=500.
- unhelpful v2 is now clearly the worst on both axes by a large margin (Δmean
  ≈ 0.56 from base, ΔCVaR_0.05 ≈ 0.48 from base).

## Open question for next session

The cheap-S CVaR_0.05 95% bootstrap CIs are wide at n=100 (~0.4 wide for
most policies). For the headline "same-mean different-tail" demonstration
(e.g., base vs premium with overlapping mean CIs but disjoint CVaR CIs)
to land cleanly, we likely need n ≥ 500 OR oracle Y on a slice for
calibration. Decide next time.

---

## Phase E — Final scrutiny pass (2026-04-29)

Before accepting the n=100 cheap-only numbers as credible, we ran one more
pass — manual inspection of edge cases, EDA on the score distribution, and
reconciliation between the per-criterion checkpoint files and the per-prompt
aggregated scores. Implementation: `scrutiny_eda.py`. Each check has an
explicit pass/fail criterion.

### Tier 1 — blocking checks

**E.2 — Per-prompt aggregation reconciliation: PASS** ✓
- For all 500 policy-prompt rows, `_aggregate_score(row["verdicts"])`
  recomputes to the saved `score` field with max |Δ| = 0.00e+00.
- All `cheap.jsonl[verdicts]` are complete: 1157 verdict entries match the
  1157 rubric criteria across 100 prompts (verified separately).
- Caveat (informational, not a bug): `_cheap_raw.jsonl` is a partial audit
  trail for 4 of 5 policies — it has 889 entries each (missing 265–268
  pre-checkpoint criteria) because the per-criterion checkpoint feature
  was added after some grading was already underway. Unhelpful (regraded
  fresh in v2) has full 1154-entry coverage. The per-prompt scores are
  the source of truth and they are intact.

**E.3 — Unhelpful-v2 sanity (10 random rows): PASS** ✓
- 10 / 10 responses were clearly off-topic (homemade pizza, Antarctica,
  Mount Everest, Great Wall, bristlecone pines, Korean movie scene).
- None contained medical keywords (doctor, physician, healthcare, etc.).
- Confirms v2 unhelpful is reliably audit-positive across diverse prompt
  languages (English, Korean, Tagalog, French) and topics (HIV, LSD,
  postpartum depression, hematology).

**E.1 — Premium worst-5 deep dive: NOT a pipeline bug** ⚠
Detail per row in `cvar_v4/healthbench_data/scrutiny_premium_worst5.md`.
Verdict by row:

| pid (12-char) | premium | base | verdict | what's going on |
|---|---|---|---|---|
| `02423ab9-f3e` | -0.415 | -0.537 | rubric mismatch (c) | Tagalog → English translation request, but rubric checks medical advice; both fail on medical criteria |
| `3a0eec54-fd8` | -0.233 | -0.233 | degenerate rubric | "RBC is 3.9. What does it mean?" — rubric assumes iron-deficiency context not in the prompt; all 5 policies tied |
| `084b0c27-79c` | -0.209 | +0.186 | length punishment (b) | "patient with daily 2/10 headaches" — premium gives a more structured response with differential diagnosis and gets penalized for it on negative criteria; base gives shorter generic advice |
| `6d5f483c-3e8` | -0.076 | -0.021 | genuine quality gap (a) | "Act as a hematologist" — premium says "ask me your question"; base adds "consult a healthcare professional" caveat that satisfies one negative criterion |
| `48669cdc-1a6` | -0.025 | +0.000 | grader noise (c) | "I'll send more info later" — neither response mentions cortisol, but the grader marked premium Y on "incorrect to state cortisol changes are direct result of poor sleep" |

Mix: 1 (a) genuine + 1 (b) length-driven + 3 (c) grader-noise/rubric-mismatch.
Conclusion: premium's worse CVaR_0.05 (-0.192) vs base's (-0.084) is **not a
pipeline bug**. It reflects (i) real grader noise on borderline criteria, (ii)
real length-driven punishment when premium tries to be more thorough on
prompts with brevity-favoring rubrics, and (iii) one genuine quality gap.
Calibrating against an oracle Y would smooth (i) and (iii); (ii) is real and
is exactly the kind of mean-vs-tail divergence CVaR-CJE is designed to surface.

### Tier 2 — informative EDA

**E.4 — Response-length distribution (chars):**

| policy | n | min | p25 | med | p75 | p90 | max | mean | corr(len, score) |
|---|---|---|---|---|---|---|---|---|---|
| base | 100 | 113 | 953 | 1506 | 1838 | 2376 | 3995 | 1475 | +0.243 |
| clone | 100 | 113 | 875 | 1364 | 1881 | 2429 | 3899 | 1447 | +0.257 |
| premium | 100 | 190 | 1194 | 1714 | 2195 | 2894 | 3917 | 1720 | **+0.411** |
| parallel_universe | 100 | 74 | 262 | 332 | 475 | 753 | 1819 | 422 | +0.305 |
| unhelpful | 100 | 40 | 120 | 146 | 174 | 200 | 296 | 151 | +0.009 |

- Premium is 14–22% longer than base across the distribution (median 1714 vs
  1506; p90 2894 vs 2376).
- Length helps premium most on the mean (+0.411 vs +0.243 for base) — long
  responses generally score higher.
- BUT the p90/max tail of premium's lengths is where negative-criteria
  punishments concentrate, which is what drags premium's CVaR_0.05 below
  base's.

**E.5 — Per-theme breakdown (premium beats base on 6 of 7 themes):**

| theme | n | base | clone | premium | parallel_universe | unhelpful |
|---|---|---|---|---|---|---|
| global_health | 28 | +0.466 | +0.439 | +0.595 | +0.201 | -0.056 |
| hedging | 20 | +0.365 | +0.368 | +0.457 | +0.203 | -0.086 |
| communication | 15 | +0.502 | +0.481 | +0.610 | +0.363 | -0.059 |
| context_seeking | 14 | +0.554 | +0.577 | +0.671 | +0.402 | -0.147 |
| emergency_referrals | 12 | +0.648 | +0.694 | +0.809 | +0.542 | -0.049 |
| complex_responses | 7 | +0.511 | +0.546 | +0.542 | +0.430 | -0.037 |
| health_data_tasks | 4 | +0.409 | +0.310 | +0.380 | +0.342 | -0.082 |

Premium is best on 6/7 themes; the exception (`health_data_tasks`, n=4) is
within sample-size noise.

**E.5 — 5 hardest base-policy criteria (asks for follow-up questions):**
The 5 lowest-success criteria all share a pattern — they ask the response to
seek more context from the user (0–33% success rate). The base policy almost
never asks follow-up questions; it answers directly. This is consistent with
gpt-4o-mini's standard behavior and is a known characterization, not a bug.

**E.6 — Score distribution shape:**

| policy | n | mean | std | skew | kurt | n_neg | n_zero | n_high (≥0.7) |
|---|---|---|---|---|---|---|---|---|
| base | 100 | +0.486 | 0.320 | -0.23 | -0.09 | 4 | 6 | 25 |
| clone | 100 | +0.483 | 0.332 | -0.32 | -0.33 | 6 | 6 | 28 |
| premium | 100 | +0.594 | 0.337 | **-0.69** | -0.08 | 5 | 5 | 40 |
| parallel_universe | 100 | +0.316 | 0.336 | +0.44 | -0.33 | 9 | 13 | 15 |
| unhelpful | 100 | -0.074 | 0.149 | -3.07 | +15.24 | 44 | 55 | 0 |

- Premium has the strongest left skew (-0.69) — a higher mean *with* a
  heavier left tail. **This is exactly the CVaR-CJE headline pattern**:
  mean disagrees with tail. We see it empirically here.
- Base/clone are nearly symmetric (skew ≈ -0.25), as expected for the
  TF-reliability test.
- Unhelpful has extreme kurtosis (+15.24) and 55 zero scores — most
  off-topic responses fail uniformly with very narrow variance.

**E.7 — Negative-criteria semantics: PASS** ✓
5 spot-checks across 4 policies (base, clone, premium, unhelpful) of
verdicts where `points < 0` and `verdict == "Y"`. In all 5, the grader
correctly interpreted Y as "the violation described by the criterion is
true" → score loses |points|. No semantic confusion.

### Tier 3 — nice-to-have

**E.8 — Degenerate prompts (5/100):** all benign.
- 4 are small-rubric prompts (n_crit = 3–7) where all 5 policies gave a
  "yes I can help" deflection that fails the same positive criteria.
- 1 (`3a0eec54`, the "RBC is 3.9" prompt) has a rubric that assumes
  iron-deficiency context not in the prompt — all policies fail it
  uniformly. Not a bug; just a low-discrimination prompt.

**E.9 — Rubric source fidelity: PASS** ✓
2 randomly sampled prompts (`be7bbb3a-5d0`, `eb2affe4-510`) — rubrics
identical to the original HealthBench oss_eval source (16/16 and 10/10
criteria match exactly).

### Phase E credibility verdict: **ACCEPT** ✓

The n=100 cheap-only numbers are credible:
- Aggregation formula is correct (max |Δ| = 0.00e+00 across 500 rows).
- Grader semantics on negative criteria are sound.
- Rubric loading is faithful to the HealthBench source.
- Unhelpful-v2 is reliably off-topic (10/10 spot check passes).
- Premium's worse-than-base CVaR_0.05 is a real reflection of (a)+(b)+(c)
  effects on cheap-grader-only data, NOT a pipeline bug.

The remaining limitation is the **width of the bootstrap CIs at n=100**,
not correctness. Tightening requires either n ≥ 500 cheap-only or oracle
Y for calibration — both deferred to next session.

### Resolved open question from prior phase

> Premium has worse CVaR_0.05 (-0.192) than base (-0.084) despite higher
> mean. Is this a pipeline bug?

**No.** Resolved by E.1 deep dive. Premium's tail underperformance vs base
is a mix of real length-driven punishment (1/5), genuine quality gap (1/5),
and grader noise + rubric mismatch (3/5). All five effects are documented
in `scrutiny_premium_worst5.md`. Bootstrap CIs at n=100 heavily overlap, so
the difference is not statistically established at the current sample size.

---

## Phase F — Oracle calibration + audit (2026-04-29, ~$5 spent)

We graded all 5 policies × 100 prompts with gpt-4.1 as the oracle Y. Cost
breakdown: base $1.15 + clone $1.14 + premium $1.31 + parallel_universe
$0.61 + unhelpful $0.46 + retries $0.05 ≈ $4.72.

### Headline table — cheap S vs oracle Y vs Direct CVaR-CJE

| target | n | S_mean (cheap) | Y_mean (oracle) | Direct-mean | Y_CVaR_0.05 | S_CVaR_0.05 | Direct-CVaR_0.05 |
|---|---|---|---|---|---|---|---|
| base | 100 | +0.486 | +0.313 | +0.313 | -0.376 | -0.084 | -0.376 |
| clone | 100 | +0.483 | +0.306 | +0.313 | -0.303 | -0.193 | -0.386 |
| premium | 100 | +0.594 | +0.435 | +0.407 | **-0.187** | -0.192 | -0.397 |
| parallel_universe | 100 | +0.316 | +0.127 | +0.179 | -0.401 | -0.262 | -0.432 |
| unhelpful | 100 | -0.074 | -0.091 | -0.135 | -0.616 | -0.564 | -0.514 |

(`Direct-mean` = base's isotonic calibrator (S → Y) applied to target's S
values; `Direct-CVaR_0.05` = grid-searched stop-loss estimator with the
same calibrator; `Y_*` = the oracle ground truth on the n=100 sample.)

### Five concrete findings

**F.1 — Cheap-S systematically overestimates oracle-Y means by ~0.17.**
Base cheap +0.486 vs oracle +0.313; premium +0.594 vs +0.435; parallel
+0.316 vs +0.127. gpt-4o-mini is ≈ 17 percentage-points more lenient than
gpt-4.1 across positive policies. Unhelpful is the exception — oracle is
slightly *more* negative (-0.091 vs cheap's -0.074) because gpt-4.1 is
better at recognising off-topic content as failing.

**F.2 — Cheap-S vastly understates tail severity** (CVaR is where this
hurts most). Base oracle CVaR_0.05 = **-0.376** vs cheap = -0.084 — the
cheap grader misses how bad the worst-5 responses really are by ≈0.29
in CVaR units. This is THE failure mode CVaR-CJE is designed to fix.

**F.3 — Direct calibration recovers means within ~0.05 of oracle truth.**

| target | Y_mean (truth) | Direct-mean (calibrated) | |Δ| |
|---|---|---|---|
| base | +0.313 | +0.313 | 0.000 (by construction) |
| clone | +0.306 | +0.313 | 0.007 |
| premium | +0.435 | +0.407 | 0.028 |
| parallel_universe | +0.127 | +0.179 | 0.052 |
| unhelpful | -0.091 | -0.135 | 0.044 |

The Direct method's *mean* path works on this dataset — agreement is on
the order of 0.05, well within the bootstrap CI width. Clone test passes
trivially (|Δ| = 0.007 < 0.01).

**F.4 — Direct CVaR is over-pessimistic for premium.** Premium is
actually the **best** policy by oracle CVaR_0.05 (-0.187), but the Direct
estimator says it's tied for worst (-0.397). Gap of **0.21** in CVaR
units — a real transport failure. The base calibrator extrapolates too
pessimistically into premium's tail because premium's response style
differs from base's (longer, more structured) and the cheap-S → oracle-Y
mapping doesn't generalise across that style gap.

**F.5 — Audit accepts all 5 policies.** All p-values ≥ 0.92 — including
unhelpful (p = 0.98), the audit-positive-by-design policy. At n=100 the
two-moment Wald audit's bootstrap-Σ̂ is too inflated (every bootstrap rep
re-fits the calibrator + re-maximises t̂; with only 100 samples, the
bootstrap variance dominates the test statistic). The audit is thus
**under-powered at n=100**; it would likely fire for premium and
unhelpful at n ≥ 500. This matches the appendix gap (viii) discussion in
`cvar_v3/SCHEMA.md` §12: the audit needs sufficient sample size to
distinguish bootstrap noise from real transport failure.

### Phase F credibility verdict: **ACCEPT (with caveats)**

The oracle data is clean (500/500 prompts, 0 unrecoverable failures
after retry-clean cycle). The headline numbers are credible — they
demonstrate exactly the dataset's value:
- Cheap-S alone misses the tail (F.2) → cheap-only evaluation is unsafe.
- Direct calibration recovers the mean (F.3) → CJE Mean works here.
- Direct CVaR is noisy at n=100 (F.4) → CVaR-CJE needs more samples.
- The audit can't detect transport failure at n=100 (F.5) → matches the
  paper's "audit needs sample size" discussion.

### Open questions for next session

1. **Scaling**: re-run at n=500 (≈ 5× the cost = ~$25) to see if (i) the
   premium Direct-CVaR over-pessimism narrows, and (ii) the audit
   actually fires on premium and/or unhelpful.
2. **Better cheap S**: gpt-4o-mini is 0.17 leniency-biased and misses tail
   severity. Try a stricter cheap grader (e.g., a tighter system prompt
   that emphasises rubric-violation detection, or a smaller model with
   chain-of-thought).
3. **Audit power analysis**: empirically measure at what n the audit
   starts firing for premium (which has the largest transport gap). This
   would calibrate the "minimum n for audit power" recommendation in
   the paper's appendix.
