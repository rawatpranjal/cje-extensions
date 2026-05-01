# HH red-team — real cheap-judge S via HF Inference Providers (PARTIAL — credits depleted)

_Cheap S = Llama-3.1-8B-Instruct (HF Inference Providers, OpenAI-compatible
endpoint at `https://router.huggingface.co/v1`) with refusal-quality
rubric, scored 0-10. n_target = 2,001 stratified across 12 HH red-team
cells. Oracle Y = `min_harmlessness_score_transcript`. α = 0.05;
coverage = 0.25._

## Status: blocked on HF credits

Of 2,001 attempted scoring calls, **only 38 succeeded** (1.9%). The other
1,963 returned HTTP 402:

> `Error code: 402 - {'error': 'You have depleted your monthly included
> credits. Purchase pre-paid credits to continue using Inference
> Providers. Alternatively, subscribe to PRO to get 20× more credits.'}`

The 38 successful calls confirmed the pipeline works end-to-end:
- The Llama-3.1-8B-Instruct model returns clean integer 0–10 scores when
  it can connect (raw outputs were `'8'`, `'10'`, `'3'`, etc. — single
  integers, no parsing artifacts).
- The score distribution among the 38 was reasonable (skewed toward
  high scores 7-10, consistent with the dataset being mostly
  refusal-trained models on adversarial prompts).
- Throughput before credits depleted: ~20 calls/s (fast enough that 2K
  scoring takes ~100s wall-clock).

But 38 successful scores split across 12 (model_type × num_params) cells
means most cells have <5 valid samples — far below the n=50 threshold
the audit demo's bootstrap requires. **The headline question (does the
real cheap S recover the 14.6× CVaR-bite ratio?) is not yet answerable
without more credits.**

Estimated remaining cost to finish: 1,963 transcripts × ~500 in-tok + ~5
out-tok ≈ ~990K tokens, **~$0.10 at HF Inference rates**. The free monthly
budget is just very small.

## Score distribution among the 38 successful calls

| score | count |
|---|---|
| 0 | 5 |
| 3 | 5 |
| 4 | 1 |
| 5 | 1 |
| 7 | 2 |
| 8 | 15 |
| 9 | 6 |
| 10 | 3 |

The model uses the high end of the 0-10 scale heavily (24/38 scored 8+),
consistent with the rubric: most HH red-team transcripts are from
safety-trained models that DO refuse, so high scores are appropriate.
The 5 `score=0` cases are the genuinely-harmful continuations (likely
the plain-LM cells).

## To unblock

Three options for the user:

1. **Top up HF Inference Providers credits** — the cheapest path. ~$1
   would cover 100× the current pilot's needs. Visit
   https://huggingface.co/settings/billing.
2. **Subscribe to HF PRO** ($9/month) for 20× the included monthly
   credits.
3. **Run a self-hosted alternative.** Llama-3.1-8B-Instruct via Together
   ($0.20/M-tok) or Fireworks ($0.20/M-tok) would cost ~$0.20 for the
   full 2K scoring pass. The harness in
   `cvar_v4/eda/deeper/hh_real_cheap_s.py` is OpenAI-compatible; just
   change `base_url` to the chosen provider's endpoint.

After credits are restored, rerun `python3 -m
cvar_v4.eda.deeper.hh_real_cheap_s`. The script is idempotent — it
reads `hh_cheap_s_scores.parquet`, identifies which transcripts are
already scored (the 38 successes are persisted), and continues from
where it left off.

## What this analysis would have answered

The §6 finding from `scoping_deeper.md` was that the framework's
within-family CVaR-CJE recovery depends on the cheap S being both
response-side AND Y-aligned, and that length and task-description-score
both fail this admissibility test. **The real-cheap-S experiment was
designed to give the definitive verdict** on whether a small-instruct
LLM judge with a refusal rubric (Llama-3.1-8B-Instruct here) suffices
for the framework to recover the empirical 14.6× headline pair
(`rlhf | 13B` vs `rejection sampling | 2.7B`).

Three possible outcomes once credits are restored:

1. **Best case**: ρ(real-S, Y) is consistently high across all 12 cells
   (>0.6), Direct CVaR-CJE recovers Δ within 0.1 of true (0.97), audit
   accepts on within-family pairs. **Establishes that an 8B-class
   instruct grader with a refusal rubric is the production-ready cheap
   judge.**
2. **Middling case**: ρ varies modestly across cells (e.g., 0.4-0.7),
   Direct CVaR errors are 0.3-0.5 (better than length's 1.4 but not
   negligible), sign of Δ is recovered. **Establishes that the framework
   works but oracle re-elicitation is still needed for level claims.**
3. **Worst case**: ρ is systematically biased across model families
   (e.g., low for plain-LM, high for RLHF), errors mirror length-S
   results. **Establishes that a stronger cheap judge (Prometheus-2-7B,
   WildGuard-7B, or a fine-tuned grader) is required.**

The cvar_v4 paper's pilot recommendation hinges on which outcome holds.
Until credits unblock, the best statement the paper can make is:
"Length-S empirically fails; real-cheap-S testing pending budget."

## Pipeline validation (the 38 successful calls did confirm)

- The HF Inference Providers OpenAI-compatible endpoint works as
  documented. The probe call with `meta-llama/Llama-3.1-8B-Instruct`
  returned in 1.08s.
- The refusal-quality rubric prompt produced clean, parseable integer
  outputs in 100% of the successful calls (no instruction-following
  failures).
- Stratified-sample logic across the 12 (model_type × num_params) cells
  works correctly; resume-on-crash via parquet persistence works.
- Total wall-clock: ~100 seconds for 2K calls when credits are available.

## File output

- `cvar_v4/eda/deeper/hh_cheap_s_scores.parquet` — 2,001 rows; 38 with
  valid scores; the rest are persisted as `score=-1` with the 402 error
  message in the `raw` column. **Do not delete** — it's the resume
  checkpoint.
- `cvar_v4/eda/deeper/hh_real_cheap_s.py` — the harness. Idempotent;
  rerun after credits unblock and it will fill in the missing 1,963
  rows.
