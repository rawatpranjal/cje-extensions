# cvar_v4/eda — dataset scoping harness

Polars-only EDA + CVaR-CJE scoping recipe. Produces
`cvar_v4/scoping_data.md` (10 datasets × 20 blocks each + Parts 1/3).

## Re-running

```bash
# one dataset:
python3 -m cvar_v4.eda.run_eda --dataset hh_red_team

# rebuild scoping_data.md from existing per-dataset sections:
python3 -m cvar_v4.eda.run_eda --concat

# list registered datasets:
python3 -m cvar_v4.eda.run_eda --list
```

Each `run_eda --dataset <slug>` does: pre-flight `df -h` → load (HF
streaming or in-memory) → run 20 blocks → write
`sections/<NN>_<slug>.md` → `rm -rf` the HF cache for that dataset.
The script is idempotent — `--no-resume` to force regeneration.

## Block recipe (`recipe.py`)

20 blocks per dataset:

1. Identification & access
2. On-disk footprint after download
3. Splits
4. First row (truncated)
5. Field inventory (dtype, %null, %empty, n_unique)
6. Usable n after filtering
7. Y candidate enumeration (continuous / Likert / binary classification)
8. Score distribution per Y candidate (11 quantiles + ASCII histogram)
9. Tie analysis at q_α (CVaR-CJE structural check)
10. Tail-mass diagnostic (CVaR_α with bootstrap CI)
11. Multi-policy nativity
12. Existing cheap-judge / oracle pair
13. Audit-discriminative metadata (stratification covariates)
14. Per-stratum tail-mass heterogeneity (audit discriminativeness)
15. Prompt-length distribution (tiktoken cl100k_base)
16. Response-length distribution
17. Tail samples (bottom 5 by best Y)
18. Median samples (3 from [q_0.45, q_0.55])
19. π0 / π′ / S / Y proposal
20. Cost & CVaR-α resolution feasibility

## Polars-only invariant

No `pandas` import anywhere in this tree. HF `datasets` is used only as
the loader; the first action after `load_dataset(...)` is
`pl.from_arrow(ds.data.table)` (or `pl.from_dicts(rows)` for streamed
batches). All quantile, histogram, distinct-count, correlation, and
length-distribution work runs through polars.

## Dataset registry (`datasets/`)

Each module declares a `SPEC = DatasetSpec(...)`. Add a new dataset by:

1. Create `datasets/my_dataset.py` with `SPEC = DatasetSpec(order=N, ...)`.
2. Add the import + module to the list in `datasets/__init__.py:_registry()`.
3. Run `python3 -m cvar_v4.eda.run_eda --dataset my_dataset`.

Loading paths supported:
- HF default: `load_dataset(spec.hf_id, **spec.load_kwargs)` →
  `pl.from_arrow(ds.data.table)`.
- Streaming: `streaming=True`, harness uses `for row in ds: ...` up to
  `max_rows` (or 100K default).
- Custom: `spec.custom_loader(spec) -> pl.DataFrame` for non-HF sources
  (HealthBench JSONL, HarmBench/StrongREJECT GitHub CSVs).

## Disk management

`run_eda.py` calls `cleanup_hf_cache(slug, extra_patterns=[hf_id_tail])`
after each dataset is materialized. This handles the `_` ↔ `-`
normalization mismatch between Python slugs and HF on-disk folder names.
Streaming datasets skip the cleanup step (cache is minimal). The harness
also requires ≥2 GiB free disk before each load (`require_disk_gib(2.0)`).

## Status

All 10 datasets ran except WildJailbreak (gated; 1-line section in
`sections/07_wildjailbreak.md` notes the unblock command). To regenerate
that section:

```bash
hf auth login                # paste a read-scope HF token
# Visit https://huggingface.co/datasets/allenai/wildjailbreak and accept terms
# Add datasets/wildjailbreak.py spec module (see notes in 07_wildjailbreak.md)
python3 -m cvar_v4.eda.run_eda --dataset wildjailbreak --no-resume
python3 -m cvar_v4.eda.run_eda --concat
```

WildChat-4.8M was streamed first-100K-rows-only, which hit a non-toxic
prefix in the parquet ordering. To get a representative sample, the
spec's `_load` should change to:

```python
ds = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
ds = ds.shuffle(buffer_size=10_000, seed=42)  # add this line
```

and rerun.
