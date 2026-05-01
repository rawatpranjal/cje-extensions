# Proper HealthBench Direct CJE and CVaR-CJE Spec Sheet

## Purpose

Use HealthBench to test the main Causal Judge Evaluation idea correctly.

- Cheap judge score `S` is useful but biased.
- Oracle score `Y` is expensive, so only a labeled slice is available.
- Calibration maps `S` and minimal covariates onto the oracle scale.
- A transport audit decides whether calibrated estimates are valid for each policy.
- CVaR-CJE extends the same audit-gated logic from average quality to worst-tail quality.

This is a Direct CJE study. Do not use IPS, DR, log probabilities, TTC, or weight stabilization unless a separate OPE experiment is explicitly started.

## Estimands

Run the mean path before the tail path as an execution sanity check, but treat mean and CVaR as separate estimands with separate audits.

- Mean-CJE target: policy-level mean oracle quality.
- Tail targets: lower-tail `CVaR_0.10` and `CVaR_0.05`, with companion `q_0.10` and `q_0.05`.
- Use `alpha = 0.10` as the higher-power tail check because it roughly doubles the tail sample relative to `alpha = 0.05`.
- Use `alpha = 0.05` as the stricter worst-tail check when sample size and oracle coverage make it stable enough.
- HealthBench scores must not be clipped to `[0, 1]`; negative rubric penalties are tail information.

## Required Data Layout

Each policy-response row should contain:

- `prompt_id`
- `policy`
- `response`
- cheap judge score `S`
- oracle score `Y`, observed only on the oracle-labeled slice during estimation
- `response_length`

Policies:

- `base`
- `clone`
- `premium`
- `parallel`
- `unhelpful`

Cheap judge:

- One fixed cheap grader for all policy-responses.
- Do not change the cheap judge, prompt, scoring formula, or rubric mid-run.

Oracle:

- One fixed oracle grader for the oracle slice.
- Call out judge-policy family overlap as a confound, especially `gpt-4o-mini` judging `gpt-4o-mini` policies or `gpt-4.1` judging a `gpt-4.1` premium policy.

## Human Expert Scores

The local HealthBench source does not contain true human expert scores for our generated policy responses.

What the data has:

- HealthBench prompts.
- Expert-written rubric criteria with positive and negative point values.
- Physician-agreement metadata such as `physician_agreed_category` tags.
- No per-response human score `Y_human_i` for the new `base`, `clone`, `premium`, `parallel`, or `unhelpful` responses we generate.

What we currently use:

- Cheap score `S`: a cheap LLM grader applying the HealthBench rubric.
- Oracle score `Y`: a more expensive LLM grader applying the same rubric.
- Full-oracle verification: the expensive LLM oracle labels all rows, but the estimator only sees a frozen oracle slice.

Interpretation:

- This validates cheap-judge-to-expensive-judge calibration.
- It does not prove agreement with human physicians on these generated responses.
- In writeups, call `Y` the expensive oracle grader or rubric oracle. Do not call it human ground truth.

Why we are not using human expert scores:

- They are not present for these generated policy responses.
- The released HealthBench data gives the rubric scaffold, not physician labels for every response we create.
- Getting true human labels would require a separate physician-grading pass.

If human-grounded claims are needed:

- Add physician labels on a held-out subset or the full policy-response panel.
- Treat those labels as a separate final validation target.
- Report whether cheap `S` and LLM-oracle `Y` both recover the physician score.
- Do not tune prompts, policies, alpha, or calibration choices on the human-held-out labels.

## Oracle Strategy

For a methods validation run, oracle-label all policy-response rows if budget allows.

Use the full oracle panel in two separate roles:

- Estimation role: pretend only a pre-specified oracle slice is observed, for example `25%` of rows. Calibration, residual correction, audits, and confidence intervals may use only this slice.
- Verification role: keep the remaining oracle labels hidden from the estimator. Use the full oracle panel only after estimation to check whether Direct Mean-CJE and Direct CVaR-CJE recover the full-oracle truth.

This lets the paper say: the method only uses a cheap judge plus a small oracle slice, but the experiment bought full oracle labels so the method can be checked against ground truth.

Recommended verification design:

- Full oracle for all rows in the frozen HealthBench policy-response panel.
- Multiple pre-specified `25%` oracle masks on the same full-oracle panel.
- Report stability across masks.
- Never use the hidden oracle labels to tune the calibrator, choose covariates, choose prompts, choose policies, choose alpha, or decide which audit to trust.

If full oracle labels are not affordable, the run can still be useful as a pilot, but it cannot cleanly verify whether CJE or CVaR-CJE recovered the true policy levels.

## GPT-5.4 Judge Stack and Cost

Recommended staged GPT-5.4-family judge stack:

- Phase 1 cheap judge `S`: `gpt-5.4-nano`.
- Phase 1 provisional oracle `Y`: `gpt-5.4-mini`.
- Phase 2 full oracle `Y`: `gpt-5.4`, after the pipeline and presentation are stable.
- Optional adjudication oracle: `gpt-5.4-pro` on a small subset only.

Phase 1 is a calibration rehearsal, not the final strongest oracle claim. It tests whether the pipeline, policy panel, audits, CIs, and tables behave correctly at low cost. Phase 2 upgrades the oracle to `gpt-5.4` for the final verification run.

Do not make `gpt-5.4-pro` the default oracle for the whole HealthBench panel unless budget and API plumbing justify it. The official model page says `gpt-5.4-pro` is Responses-API-only and may take minutes for hard requests, while the current HealthBench grading code uses Chat Completions. Using pro therefore requires an API-path change, not just a model-string edit.

Official token prices checked on 2026-04-29:

| model | role | input $/1M | output $/1M | note |
| --- | --- | ---: | ---: | --- |
| `gpt-4o-mini` | old cheap judge | 0.15 | 0.60 | current pilot cheap judge |
| `gpt-4.1` | old oracle | 2.00 | 8.00 | current pilot oracle |
| `gpt-5.4-nano` | new cheap judge | 0.20 | 1.25 | cheapest GPT-5.4-class judge |
| `gpt-5.4-mini` | provisional oracle / robustness cheap judge | 0.75 | 4.50 | Phase 1 oracle |
| `gpt-5.4` | final oracle | 2.50 | 15.00 | Phase 2 oracle |
| `gpt-5.4-pro` | optional adjudication oracle | 30.00 | 180.00 | too expensive for default full-panel oracle |

Cost estimates below use the actual `n = 100` HealthBench grading token logs and scale them linearly. They cover grading only, not response generation. They assume every policy-response gets cheap `S`, and oracle `Y` is either full-panel or a `25%` estimator-visible slice.

| prompt n | cheap judge | oracle | total with 25% oracle | total with full oracle |
| ---: | --- | --- | ---: | ---: |
| 500 | `gpt-4o-mini` | `gpt-4.1` | 7.37 | 25.21 |
| 500 | `gpt-5.4-nano` | `gpt-5.4-mini` | 4.16 | 10.88 |
| 500 | `gpt-5.4-nano` | `gpt-5.4` | 9.38 | 31.78 |
| 500 | `gpt-5.4-mini` | `gpt-5.4` | 14.64 | 37.05 |
| 500 | `gpt-5.4-nano` + `gpt-5.4-mini` | `gpt-5.4` | 16.56 | 38.96 |
| 500 | `gpt-5.4-nano` | `gpt-5.4-pro` | 91.52 | 360.34 |
| 1000 | `gpt-4o-mini` | `gpt-4.1` | 14.75 | 50.42 |
| 1000 | `gpt-5.4-nano` | `gpt-5.4-mini` | 8.31 | 21.75 |
| 1000 | `gpt-5.4-nano` | `gpt-5.4` | 18.76 | 63.57 |
| 1000 | `gpt-5.4-mini` | `gpt-5.4` | 29.29 | 74.09 |
| 1000 | `gpt-5.4-nano` + `gpt-5.4-mini` | `gpt-5.4` | 33.12 | 77.92 |
| 1000 | `gpt-5.4-nano` | `gpt-5.4-pro` | 183.04 | 720.67 |
| 5000 | `gpt-4o-mini` | `gpt-4.1` | 73.73 | 252.07 |
| 5000 | `gpt-5.4-nano` | `gpt-5.4-mini` | 41.55 | 108.76 |
| 5000 | `gpt-5.4-nano` | `gpt-5.4` | 93.82 | 317.83 |
| 5000 | `gpt-5.4-mini` | `gpt-5.4` | 146.44 | 370.45 |
| 5000 | `gpt-5.4-nano` + `gpt-5.4-mini` | `gpt-5.4` | 165.59 | 389.60 |
| 5000 | `gpt-5.4-nano` | `gpt-5.4-pro` | 915.20 | 3603.35 |

Recommendation:

- First run: use `gpt-5.4-nano` as cheap `S` and `gpt-5.4-mini` as provisional oracle `Y`.
- If Phase 1 passes the sanity checks, rerun oracle labels with `gpt-5.4` for the final verification table.
- Optionally keep `gpt-5.4-mini` as a second cheap judge in Phase 2 to show that the results are not tied to a single cheap model.
- Reserve `gpt-5.4-pro` for a small adjudication subset, for example hard tail rows or cheap/oracle disagreement rows.

Self-preference warning:

- Moving judges to GPT-5.4 does not remove same-family bias if the policies are also GPT-5.4-family models.
- If the policy panel is also switched to GPT-5.4 variants, report this as a same-family validation and avoid human-ground-truth language.
- For a stronger paper claim, include at least one policy outside the GPT-5.4 family or add a human/physician validation subset.

## Policy Design and Variation

The policy panel should create structured variation, not arbitrary model churn.

Required roles:

- `base`: the reference logger policy.
- `clone`: same model and prompt as `base`, different seed. It should behave like `base`.
- `premium`: stronger model, same task prompt. It should usually improve mean quality.
- `parallel`: same model family as `base`, different system prompt. It should induce style or support shifts that may affect calibration.
- `unhelpful`: deliberately bad or off-topic policy. It should create a clear audit-stress row.

The panel is useful only if it creates these patterns:

- Cheap `S` has signal for oracle `Y`.
- Cheap `S` is visibly biased, so calibration has something to fix.
- `base` and `clone` are close enough that the clone check is meaningful.
- `premium` differs from `base`, preferably with better mean quality.
- `parallel` differs from `base` through response style, length, or actionability.
- `unhelpful` is clearly bad on both mean and tail, with very few high-scoring outliers.
- At least one comparison shows mean and tail are not the same story.

Pilot check from the current `n = 100` panel, using the expensive LLM oracle as `Y`:

| policy | oracle mean | oracle CVaR_0.10 | oracle CVaR_0.05 | average response chars |
| --- | ---: | ---: | ---: | ---: |
| `base` | +0.313 | -0.279 | -0.376 | 1475 |
| `clone` | +0.306 | -0.255 | -0.303 | 1447 |
| `premium` | +0.435 | -0.113 | -0.187 | 1720 |
| `parallel_universe_prompt` | +0.127 | -0.346 | -0.401 | 422 |
| `unhelpful` | -0.091 | -0.495 | -0.616 | 151 |

This is the right rough shape for a pilot:

- `base` and `clone` are close on mean.
- `premium` is better on mean and tail in this sample.
- `parallel` is lower quality and much shorter, so it tests style/length transport.
- `unhelpful` is clearly worst and creates a stress-test row.
- The cheap judge is too lenient for the main non-unhelpful policies, with cheap-minus-oracle mean gaps around `+0.16` to `+0.19`.

Caveats:

- `n = 100` is not enough for final CVaR claims.
- `unhelpful` may be too extreme to be the only failure mode; it is useful as a stress test, not as the whole paper.
- The cheap judge and several policies are from the same model family, which creates self-preference risk.
- The premium policy and current oracle are also both `gpt-4.1`, which is a confound for premium validation.
- Before the main run, freeze the policy prompts and judge models. If any policy is changed, restart the comparison.

## Minimum Proper Run

- Prompt sample: `n = 500` HealthBench prompts minimum.
- Total policy-responses: `500 prompts x 5 policies = 2500 rows`.
- Oracle coverage: at least `25%` of rows, with enough labeled rows per policy for audits.
- Debug-only runs may use `n = 100`, but they cannot support CVaR validity claims.
- Prefer `n = 1000` prompts if budget allows.

Tail sample sizes per policy:

- With `n = 500`, `alpha = 0.10` uses about 50 tail rows per policy and `alpha = 0.05` uses about 25.
- With `n = 1000`, `alpha = 0.10` uses about 100 tail rows per policy and `alpha = 0.05` uses about 50.

If full oracle labels are purchased:

- `n = 500` means oracle grading `2500` policy-responses.
- `n = 1000` means oracle grading `5000` policy-responses.
- The estimator should still be reported as using the pre-specified oracle slice, not the full oracle panel.

## Calibration Recipe

Use the original Direct CJE structure.

1. Score every policy-response with the cheap judge.
2. Label an oracle slice with `Y`.
3. Fit a cross-fitted calibration model from `S + response_length` to `Y`.
4. Use 5 folds for cross-fitting.
5. Estimate policy means from calibrated scores on all rows.
6. Add residual correction from oracle-labeled rows.
7. Use bootstrap confidence intervals with calibrator refit.

Required covariate:

- `response_length`

Optional later covariate:

- HealthBench theme or rubric group, only after the simple path is validated.

Do not add many covariates before the simple mean path has been inspected and validated.

## Mean Transport Audit

For each policy, test whether the calibrated residual mean is zero on the oracle-labeled rows:

`mean(Y_i - f(S_i, X_i)) = 0`

Gate rule:

- Audit passes: the policy may receive a mean level estimate.
- Audit fails: mark the policy as `REFUSE MEAN`.

Do not describe a failed audit as a minor caveat. A failed mean audit blocks the mean level claim. It does not automatically block the CVaR level claim, because CVaR has a different transport condition.

## CVaR Extension

Run CVaR-CJE after the mean path has been inspected. Mean failure is evidence that the pipeline needs scrutiny, but it does not logically prove that the tail bridge fails.

For lower-tail `CVaR_alpha`, where `alpha` is planned in advance:

1. Fit stop-loss calibrators for `(t - Y)_+`.
2. Estimate `CVaR_alpha` by optimizing over threshold `t`.
3. Re-optimize `t` inside the bootstrap.
4. Refit calibrators inside the bootstrap.
5. Report companion `q_alpha`.

Planned alpha values:

- `alpha = 0.10`: primary HealthBench tail diagnostic if `n = 500`.
- `alpha = 0.05`: stricter tail diagnostic, primary only if the full-oracle check shows enough stability.

Do not pick alpha after seeing which result looks best. Pre-specify whether the main tail estimand is `CVaR_0.10`, `CVaR_0.05`, or both. If `n = 500`, use `CVaR_0.10` as the higher-power main tail result and `CVaR_0.05` as a stricter secondary result. If `n = 1000` with full oracle labels, report both if both are stable.

CVaR is stricter than the mean path in the sense that it has fewer effective observations. It still needs its own audit, and a mean audit failure should not automatically be converted into a CVaR audit failure.

## CVaR Transport Audit

For each policy, audit both tail location and stop-loss residual transport.

Minimum audit objects:

- Tail mass near the estimated `q_alpha`.
- Stop-loss residuals at the selected threshold.

Gate rule:

- Mean audit passes and CVaR audit passes: `VALID MEAN + VALID CVAR`.
- Mean audit passes and CVaR audit fails: `VALID MEAN ONLY`.
- Mean audit fails and CVaR audit passes: `VALID CVAR ONLY`.
- Mean audit fails and CVaR audit fails: `REFUSE ALL LEVEL CLAIMS`.

Do not report rejected CVaR rows as meaningful calibrated tail levels.

## Required Mean Output Table

Include one row per policy:

| policy | cheap mean | oracle mean | Direct CJE mean | mean 95% CI | mean audit p-value | gate |
| --- | ---: | ---: | ---: | --- | ---: | --- |

Gate values:

- `PASS`
- `FAIL`
- `REFUSE MEAN`

## Required CVaR Output Table

Include one row per policy:

| policy | alpha | cheap CVaR | oracle CVaR | Direct CVaR | CVaR 95% CI | CVaR audit p-value | gate |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |

Gate values:

- `PASS`
- `FAIL`
- `REFUSE CVAR`
- `VALID CVAR ONLY`
- `REFUSE ALL LEVEL CLAIMS`

## Final Reporting Table

The final table must be audit-gated:

| policy | mean | mean CI | mean audit | alpha | q_alpha | CVaR_alpha | CVaR CI | CVaR audit | status |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |

Allowed `status` values:

- `VALID MEAN + VALID CVAR`
- `VALID MEAN ONLY`
- `VALID CVAR ONLY`
- `REFUSE MEAN`
- `REFUSE CVAR`
- `REFUSE ALL LEVEL CLAIMS`

No estimand with a failed relevant audit gets a clean level estimate.

## Ideal Results Presentation

Present the results as a validation experiment, not as a cheap-score leaderboard.

Main text order:

1. State the validation design: full oracle labels were collected for verification, but the estimator only used the pre-specified oracle slice.
2. Show that the cheap judge has signal but visible bias.
3. Show Direct Mean-CJE against full-oracle mean truth.
4. Show Direct CVaR-CJE against full-oracle tail truth for the pre-specified alpha values.
5. Apply mean and CVaR audit gates separately.
6. Interpret only rows whose relevant audit passes.

Primary table:

| policy | estimand | alpha | cheap | Direct CJE | full-oracle truth | Direct error | cheap error | 95% CI | audit p-value | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |

Use `estimand = mean` with `alpha` blank or `NA`. Use `estimand = CVaR` with `alpha = 0.10` or `0.05`.

Audit-gated summary table:

| policy | mean | mean CI | mean audit | q_0.10 | CVaR_0.10 | CVaR_0.10 CI | CVaR_0.10 audit | q_0.05 | CVaR_0.05 | CVaR_0.05 CI | CVaR_0.05 audit | status |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- |

If the table gets too wide, make `CVaR_0.10` the main table and move `CVaR_0.05` to a robustness table.

Recommended figures:

- Cheap-vs-oracle calibration plot on the oracle slice, colored by policy.
- Absolute error bars: cheap estimate error versus Direct CJE error, computed against full-oracle truth.
- Mean versus CVaR scatter, using Direct estimates for valid rows and visually marking refused rows.
- Tail curve over alpha, only if enough full-oracle data make the curve stable.

Presentation rules:

- Put `PASS`, `FAIL`, `VALID MEAN ONLY`, `VALID CVAR ONLY`, `REFUSE CVAR`, or `REFUSE ALL LEVEL CLAIMS` directly in tables.
- For a refused estimand, show the numeric estimate only as diagnostic evidence, not as a valid level claim.
- Gray out or otherwise mark refused rows in figures.
- Do not average statuses across policies.
- Do not say `CJE works` unless Direct CJE improves over cheap-only against full-oracle truth for the relevant estimand.
- Do not say `CVaR-CJE works` unless Direct CVaR improves over cheap-only against full-oracle CVaR truth for audit-passing policies.
- Report failures as results. A failed audit is evidence about non-transport, not a nuisance to explain away.

## Explicit Anti-Goals

Do not:

- Treat cheap-only HealthBench scores as the answer.
- Claim CJE works from cheap-only results.
- Claim base validation from training on base and evaluating on base.
- Skip the mean audit.
- Run CVaR-CJE before the mean path has been inspected and debugged.
- Overinterpret `n = 100`; at `CVaR_0.10`, that is only 10 tail prompts, and at `CVaR_0.05`, that is only 5.
- Use CVaR audit p-values from `n = 100` as evidence of validity.
- Add IPS, DR, log-probability, or OPE machinery to this HealthBench run.
- Add many estimator variants to the main result.
- Change policies, judge prompts, oracle prompts, or score formulas mid-run.
- Clip HealthBench scores to `[0, 1]`.
- Hide audit failures in prose.
- Report rejected policies as if their levels are meaningful.
- Say `CVaR-CJE works` if the Direct CVaR estimate is farther from oracle than cheap-only.

## Success Criteria

The run is successful only if:

- The cheap judge has signal but visible bias.
- Direct Mean-CJE improves over cheap-only estimates.
- `clone` behaves like `base`.
- The mean audit rejects clearly non-transporting policies or gives a defensible non-rejection.
- CVaR reveals tail information not captured by the mean.
- CVaR-CJE improves tail estimates for audit-passing policies.
- Audit-failing policies are refused, not interpreted.

## Implementation Checklist

Before estimation:

- Freeze policies.
- Freeze cheap judge.
- Freeze oracle judge.
- Freeze score formula.
- Freeze prompt sample.
- Record sample size, oracle fraction, folds, bootstrap count, and tail alpha.
- If using full oracle verification, freeze which labels are estimator-visible before any estimation.

Mean path:

- Build full policy-response panel.
- Add cheap scores for every row.
- Add oracle labels on the labeled slice.
- Fit 5-fold cross-fitted `S + response_length -> Y` calibration.
- Estimate Direct Mean-CJE with residual correction.
- Bootstrap with calibrator refit.
- Run per-policy mean transport audits.
- If mean gates fail, document and debug before interpreting mean levels. Do not automatically convert mean failure into CVaR failure.

CVaR path:

- Reuse the same frozen data, judges, and score formula.
- Fit stop-loss calibrators for threshold grid values.
- Optimize `t` for each pre-specified alpha.
- Bootstrap with threshold re-optimization and calibrator refit.
- Run per-policy CVaR transport audits.
- Apply final status labels.

Reporting:

- Show mean and CVaR tables.
- Show full-oracle truth only as verification, not as an input to estimation.
- Put `PASS`, `FAIL`, `VALID MEAN ONLY`, `VALID CVAR ONLY`, `REFUSE CVAR`, or `REFUSE ALL LEVEL CLAIMS` directly in tables.
- Separate diagnostic rejected rows from valid level estimates.
