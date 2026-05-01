# WildChat-4.8M (shuffled stream) — cross-model (mean, CVaR_0.05)

_n = 99,145 (conversation, model) cells from a shuffled 100K-row stream. Y_safe = -max_detoxify_toxicity (higher = safer; lower-tail = worst-toxicity continuations). Models: 5 distinct after shuffle._

## Per-model statistics

| model | n | mean | std | q_0.05 | CVaR_0.05 | CVaR_0.01 |
|---|---|---|---|---|---|---|
| `gpt-3.5-turbo-0301` | 50,998 | -0.00471 | 0.01277 | -0.02729 | -0.05391 | -0.08620 |
| `gpt-4-0314` | 11,398 | -0.00433 | 0.01233 | -0.02459 | -0.05122 | -0.08704 |
| `gpt-4o-mini-2024-07-18` | 2,520 | -0.00244 | 0.00862 | -0.00943 | -0.03234 | -0.07063 |
| `gpt-4o-2024-08-06` | 32,991 | -0.00235 | 0.00976 | -0.00520 | -0.03376 | -0.08364 |
| `o1-mini-2024-09-12` | 1,238 | -0.00209 | 0.00713 | -0.00735 | -0.02529 | -0.05988 |

## Cross-model same-mean-different-tail pairs

Criterion (Y range ≈ 0.1): |Δmean| ≤ 0.001 AND |ΔCVaR_0.05| ≥ 0.003 (3× the mean threshold).

| model_A | model_B | Δmean | ΔCVaR_0.05 | ratio |
|---|---|---|---|---|
| `gpt-4o-2024-08-06` | `o1-mini-2024-09-12` | 0.00027 | 0.00847 | **32×** |
| `gpt-4o-mini-2024-07-18` | `o1-mini-2024-09-12` | 0.00035 | 0.00706 | **20×** |

## Caveat — toxic subset is absent from this stream

Even with `ds.shuffle(buffer=10000, seed=42)` the streamed sample contains 0 conversations with `toxic=True`. The WildChat-4.8M release likely partitions toxic conversations into a separate file the streaming loader doesn't reach by default — the 100% non-toxic finding here is consistent with the dataset card's structure (3.2M non-toxic + 1.5M toxic split into different parquet shards). To pull representative toxic conversations we'd need to switch shards explicitly via `data_files='toxic/*.parquet'` or similar. **For audit-discriminative covariate work we should use this stream's `language` and `country` cardinality (70 / 191) as effective stratifiers, not `toxic`.**