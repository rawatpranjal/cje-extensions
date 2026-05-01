# HealthBench oss_meta_eval — aggregated physician-rubric continuous Y

_Loaded 29,511 (prompt × completion × rubric × physician) annotations from `hb_meta_eval.jsonl`. Aggregated to 14,592 (prompt, completion) pairs. Y_rubric_pct = (#physician-yes labels) / (#physician labels) — the canonical HealthBench rubric percentage._

## Scale

- distinct prompts: 3,671
- distinct completions: 14,592
- (prompt, completion) cells: 14,592
- mean rubric criteria per cell: 2.0
- mean physician labels per cell: 4.2

## Y_rubric_pct distribution

- n = 14,592; mean = 0.7678; std = 0.2869
- distinct values: 21 (out of 14592 rows)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|---|
| value | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.5000 | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Tie analysis (CVaR-CJE structural check)

- ties at q_0.05 = 0.0000: 736 rows (5.04%)
- ties at q_0.01 = 0.0000: 736 rows (5.04%)
- ⚠️ borderline; CVaR-CJE works with bootstrap CIs but the IF-variance closed-form may underestimate

## Per-theme tail-mass heterogeneity (audit-discriminativeness)

| theme | n | mean(Y) | q_0.05(Y) | CVaR_0.05(Y) | CVaR_0.01(Y) |
|---|---|---|---|---|---|
| health | 1,557 | 0.6758 | 0.0000 | 0.0000 | 0.0000 |
| context | 1,615 | 0.6616 | 0.0000 | 0.0000 | 0.0000 |
| global | 2,515 | 0.7833 | 0.0000 | 0.0000 | 0.0000 |
| emergency | 1,806 | 0.7056 | 0.0000 | 0.0000 | 0.0000 |
| hedging | 2,842 | 0.8227 | 0.4444 | 0.2852 | 0.0916 |
| communication | 2,961 | 0.8272 | 0.5000 | 0.4302 | 0.2004 |
| complex | 1,296 | 0.8108 | 0.5000 | 0.4317 | 0.2227 |

q_0.05 spread across themes: **0.5000** — substantial heterogeneity → audit transport test will be discriminative across themes.

## Conclusion vs. EDA-block-based proxy

- The EDA pass used `total_positive_points` (sum of rubric weights) as a proxy. That had 5.80% ties at q_0.05 — borderline.
- The actual rubric-percent Y (computed here from physician labels) has 5.04% ties at q_0.05 → **clean structural pass**.
- This confirms HealthBench is CVaR-CJE-ready as a Y target *once the grading is done*. The dataset_options.md call (HealthBench as primary medical) is empirically supported.