# HealthBench-as-CJE-Dataset

Mirror of the original CJE Arena pipeline (`~/Dropbox/cvar-cje-data/cje-arena-experiments/`) built on HealthBench prompts + OpenAI graders. Output JSONL is drop-in compatible with `cvar_v3/workhorse.py`'s Direct CVaR-CJE estimator.

## Files in this directory

| file | what it is |
|---|---|
| `SCHEMA.md` | full output-format spec (CJE-compatible) |
| `policies.py` | 5-policy panel: base / clone / premium / parallel_universe_prompt / unhelpful |
| `prompts.py` | extract HealthBench oss_eval prompts + rubrics → `data/prompts.jsonl` |
| `generate.py` | OpenAI Chat Completions per policy → `data/responses/{policy}_responses.jsonl` |
| `judge.py` | per-criterion rubric grading (cheap S = gpt-4o-mini, oracle Y = gpt-4.1) → `judge_outputs/` |
| `pipeline.py` | end-to-end orchestrator: `--pilot N` or `--full` |
| `analyze.py` | step-by-step analysis: per-policy mean / CVaR / audit |

## How to run

Set `OPENAI_API_KEY` in your environment, then:

```bash
# 1. Extract prompts (free, ~5 sec)
python3 -m cvar_v4.healthbench_data.prompts

# 2. Pilot (20 prompts × 5 policies, ~$2, ~30 min)
python3 -m cvar_v4.healthbench_data.pipeline --pilot 20

# 3. Or run incrementally:
python3 -m cvar_v4.healthbench_data.generate --all --limit 20    # ~$0.13
python3 -m cvar_v4.healthbench_data.judge --all --kind cheap     # ~$0.12
python3 -m cvar_v4.healthbench_data.judge --all --kind oracle    # ~$1.64
python3 -m cvar_v4.healthbench_data.pipeline --assemble-only

# 4. Analyze
python3 -m cvar_v4.healthbench_data.analyze --kind cheap-only   # naive S-only
python3 -m cvar_v4.healthbench_data.analyze --kind oracle       # full Direct CVaR-CJE + audit

# 5. Full run (5K prompts, ~$140 at 5% oracle coverage; ~$460 at 25%)
python3 -m cvar_v4.healthbench_data.pipeline --full
```

## Idempotency / resume

- `generate.py` skips prompts already in `responses/{policy}_responses.jsonl`.
- `judge.py` skips prompts already in `judge_outputs/{policy}_{cheap,oracle}.jsonl`.
- Crash and re-run; you'll only pay for what's missing.

## Output

After a successful pilot run:

```
data/
├── prompts.jsonl                                  # the 5,000 HealthBench prompts (or N if --limit)
├── cje_dataset.jsonl                              # logger π0's responses + scores (one row/prompt)
└── responses/
    ├── base_responses.jsonl                       # logger fresh-draws (also in cje_dataset.jsonl)
    ├── clone_responses.jsonl                      # target fresh-draws
    ├── premium_responses.jsonl
    ├── parallel_universe_prompt_responses.jsonl
    └── unhelpful_responses.jsonl
judge_outputs/                                     # raw per-criterion grader outputs (audit trail)
├── base_cheap.jsonl
├── base_oracle.jsonl
└── ... (10 files for the 5 policies × {cheap, oracle})
```

## Cost reference (2026-04-28 OpenAI rates)

| run | n_prompts × 5 | cheap S calls | oracle Y calls | total cost |
|---|---|---|---|---|
| pilot 20 | 100 | 1,140 | 1,140 | ~$2 |
| pilot 200 | 1,000 | 11,400 | 11,400 | ~$20 |
| full 5,000 @ 5% oracle | 25,000 | 285,000 | 14,250 | ~$140 |
| full 5,000 @ 25% oracle (paper headline) | 25,000 | 285,000 | 71,250 | ~$460 |

## Compatibility with cvar_v3 / cvar_v4 estimators

The output `cje_dataset.jsonl` and `responses/*_responses.jsonl` files are drop-in compatible with:
- `cvar_v3/workhorse.py:load_arena_main()` and `load_fresh_draws_df()`
- `cvar_v4/eda/deeper/_estimator.py` (self-contained Direct CVaR-CJE + audit)

Set the data root to `cvar_v4/healthbench_data/data/` instead of `~/Dropbox/cvar-cje-data/cje-arena-experiments/data/` and the existing harness pipelines run unchanged.
