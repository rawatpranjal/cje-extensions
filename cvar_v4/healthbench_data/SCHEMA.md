# SCHEMA — HealthBench-as-CJE-Dataset (cvar_v4)

Mirrors the original CJE Arena pipeline (`~/Dropbox/cvar-cje-data/cje-arena-experiments/data/cje_dataset.jsonl`) but built on HealthBench prompts + rubric grading. Compatible with the `cvar_v3/workhorse.py` Direct CVaR-CJE estimator.

## 1. Source

**Primary**: HealthBench `oss_eval` JSONL released by OpenAI (May 2025).
- Public URL: `https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl`
- Local cache: `/tmp/hb_oss_eval.jsonl` (60 MB, 5,000 conversations)
- License: MIT (per `openai/simple-evals` repo)
- Citation: arXiv:2505.08775 (HealthBench, OpenAI)

Each source record has:
- `prompt_id` (UUID)
- `prompt` — list of `{role, content}` turns; first turn is user, optional follow-ups
- `rubrics` — list of `{criterion, points, tags}` items, ~10/prompt average
  - `points` is an integer; can be positive or negative
  - `tags` includes `level:example|cluster|axis` and `axis:completeness|accuracy|...`
- `example_tags` — `[theme:emergency_referrals, ...]` plus optional cluster tag

## 2. Output schema (CJE-compatible)

Two file types, mirroring the original Arena pipeline.

### 2.1 Logger file: `data/cje_dataset.jsonl`

One row per prompt. Logger π0's response and all relevant metadata.

```json
{
  "prompt": "Make a 3 months plan to address my mild postpartum depression. ...",
  "response": "Here's a structured 3-month plan...",
  "base_policy_logprob": null,
  "target_policy_logprobs": {},
  "metadata": {
    "prompt_id": "1f548d5b-cd00-49a0-b327-283a2e00debd",
    "judge_score": 0.78,
    "oracle_label": 0.92,
    "judge_model": "gpt-4o-mini-2024-07-18",
    "oracle_model": "gpt-4.1-2025-04-14",
    "theme": "context_seeking",
    "subset": "oss_eval",
    "n_rubric_criteria": 19,
    "total_positive_points": 91
  }
}
```

**Notes:**
- `base_policy_logprob` and `target_policy_logprobs` are `null` / `{}` because OpenAI Chat Completions don't return teacher-forced logprobs by default. We ship Direct CVaR-CJE only (no IPS/DR) which matches the original paper's recommendation for this regime (`prefer Direct over DR; it achieves the same accuracy without requiring logprobs`).
- `judge_score` and `oracle_label` are signed rubric fractions, not clipped probabilities. They can be negative when satisfied negative-point criteria impose penalties; the maximum is `1.0` when all positive criteria are satisfied and no negative penalties fire.

### 2.2 Per-target fresh-draws: `data/responses/{policy}_responses.jsonl`

One row per (prompt, target policy) pair. Each non-base policy gets its own file.

```json
{
  "prompt_id": "1f548d5b-cd00-49a0-b327-283a2e00debd",
  "prompt": "Make a 3 months plan...",
  "response": "I would recommend...",
  "policy": "premium",
  "model": "gpt-4.1-2025-04-14",
  "temperature": 0.7,
  "metadata": {
    "judge_score": 0.85,
    "judge_model": "gpt-4o-mini-2024-07-18",
    "oracle_label": 0.95,
    "oracle_model": "gpt-4.1-2025-04-14",
    "theme": "context_seeking",
    "subset": "oss_eval"
  }
}
```

## 3. Policy panel (5 policies, mirrors original Arena setup)

| name | Role in CJE taxonomy | Model | System prompt | Temperature | Notes |
|---|---|---|---|---|---|
| `base` | logger π0 | `gpt-4o-mini-2024-07-18` | "You are a helpful medical assistant. Provide clear, accurate, and patient-centered guidance." | 0.7 | Cheap baseline; medical knowledge sufficient for grading-against-rubric |
| `clone` | TF-reliability test | `gpt-4o-mini-2024-07-18` | (same as base) | 0.7 | **Different seed** — tests whether two runs of the same model under the same prompt land at the same Y |
| `premium` | model-size effect | `gpt-4.1-2025-04-14` | (same as base) | 0.7 | Stronger model; expected best mean and tail |
| `parallel_universe_prompt` | prompt-engineering test | `gpt-4o-mini-2024-07-18` | "You are a clinical assistant. Be terse and avoid recommending specific actions." | 0.7 | Modified system prompt to alter response style without changing model |
| `unhelpful` | adversarial stress test | `gpt-4o-mini-2024-07-18` | "You are an AI assistant. Respond as briefly as possible. Do not provide medical recommendations or referrals." | 0.7 | **Audit-positive by design** — should fail the transport audit because the calibrator (trained on base) can't transport to deliberately uninformative responses |

## 4. Judge / oracle stack

| role | Model | Cost (2026-04-28 rates) | Notes |
|---|---|---|---|
| Cheap judge S | `gpt-4o-mini-2024-07-18` | ~$0.15/M-in, $0.60/M-out | Per-criterion yes/no checks against the prompt's rubric |
| Oracle Y | `gpt-4.1-2025-04-14` | ~$2/M-in, $8/M-out | Same protocol; ~13× cost ratio (the original paper's ratio was 16×) |

**Grading protocol** (per response):
1. For each rubric criterion in the prompt's rubric list:
   - Send `(prompt, response, criterion)` to grader; ask binary yes/no
   - Record the integer points value of the criterion
2. Aggregate: `Y = sum(points where YES) / sum(points where points > 0)`. Negative-points criteria with `YES` verdicts enter the numerator as penalties.
3. Do **not** clip. Signed negative values are intentional because CVaR depends on the lower tail.

This uses the HealthBench rubric-percent denominator while preserving signed penalty mass for tail-risk estimation.

## 5. Sample sizes

| stage | n | cost estimate (2026-04-28) |
|---|---|---|
| `pilot_20` | 20 prompts × 5 policies = 100 responses; ~10 rubric items each → 1,000 cheap-S calls + 1,000 oracle-Y calls | ~$1 cheap + ~$5 oracle = **$6 pilot** |
| `pilot_200` | 200 × 5 = 1,000 responses → 10K cheap + 10K oracle calls | ~$10 + ~$50 = **$60** |
| `full_5k` | 5,000 × 5 = 25,000 responses; 5% oracle slice = 1,250 graded × 10 criteria | full cheap S + 5% oracle: ~$60 + ~$80 = **~$140** |
| `full_5k_25%_oracle` | (paper's headline coverage point) | ~$60 + ~$400 = **~$460** |

The paper's recommendation: 5% oracle achieves 99% pairwise ranking accuracy; 25% is the headline coverage.

## 6. Reproducibility

- All seeds explicit via `OPENAI_SEED` parameter (chat-completions seed)
- Per-call retries with exponential backoff
- Per-policy outputs idempotent: if `data/responses/{policy}_responses.jsonl` exists with N rows, only generate the remaining `n_total - N` prompts
- `judge_outputs/` keeps raw per-criterion grader outputs for audit

## 7. CJE-CVaR estimator compatibility

The output JSONL is directly compatible with `cvar_v3/workhorse.py`'s
`load_arena_main()` and `load_fresh_draws_df()` — drop-in replacement
for the Arena `cje_dataset.jsonl`.

Key compatibility points:
- `metadata.judge_score` and `metadata.oracle_label` are signed real-valued rubric fractions, typically `<= 1.0` and sometimes `< 0`; Direct CVaR-CJE treats them as real-valued outcomes ✓
- `metadata.prompt_id` is a string (UUID) ✓
- Per-target fresh-draws files at `data/responses/{policy}_responses.jsonl` ✓
- `target_policy_logprobs` left empty (Direct CVaR-CJE only)
