# Runbook: reproducing the CVaR-CJE pilot writeup

This document is the recipe for reproducing every number, table, and figure in `pilot.pdf` from scratch. It is also the path to scaling the pilot from `n_0 = 100` (current) to any larger `n_0` once the upstream data pipeline runs.

## Three-command reproduction

From the repository root:

```bash
# 1. Generate (or extend) the response and oracle data.
#    --pilot N selects the first N HealthBench prompts. Pre-existing rows are
#    skipped, so this is incremental.
python -m cvar_v4.healthbench_data.pipeline --pilot 100

# 2. Run the headline pilot tables and all seven diagnostics.
#    Writes JSON to writeup/data/ and figure PDFs to writeup/.
python -m cvar_v4.healthbench_data.analyses.run_all

# 3. Compile the LaTeX writeup. Figure includes pick up the new PDFs
#    automatically; numeric tables are currently hand-typed.
cd cvar_v4/healthbench_data/writeup
pdflatex pilot.tex
pdflatex pilot.tex   # second pass to resolve cross-references
```

To reproduce at a different sample size, change the first command. For example, `--pilot 500` extends the panel to 500 prompts. The analysis layer auto-detects the panel size from the data files; nothing else needs to change.

## What each piece produces

### Step 1: data generation (`pipeline.py`)

Generates `data/responses/{policy}_responses.jsonl` and `judge_outputs/{policy}_{cheap,oracle}.jsonl` for each of the five policies (`base`, `clone`, `premium`, `parallel_universe_prompt`, `unhelpful`). Calls the OpenAI API. Idempotent: re-running on the same `--pilot N` is a no-op when the data already exists.

Approximate cost at `n_0 = 100`: about $6.40. Linear in `n_0`. Wall clock with sync API: about 10 hours; with Batch API (when available): about 30 minutes.

### Step 2: analyses (`analyses/run_all.py`)

Read-only on `data/` and `judge_outputs/`. Writes to `writeup/data/` (JSON) and `writeup/` (figure PDFs). Auto-detects `n_0` from the logger response file.

Internally runs:

- **Pilot tables** (via `cvar_pilot_table.run`): the headline α=0.10 and robustness α=0.20 tables; outputs `pilot_table_alpha_*_cov_*_seed_*.{jsonl,md}` to `writeup/data/`.

- **Four general diagnostics**:
  - `budget_curve`: re-runs the estimator at coverages {0.10, 0.25, 0.50, 1.00} and plots `|err|` against coverage. Writes `budget_curve.pdf` (figure) and `budget_curve.json` (data).
  - `mean_vs_cvar`: scatter of (mean Y, CVaR_α) per policy with Direct estimates overlaid. Writes `mean_vs_cvar.pdf` and `mean_vs_cvar.json`.
  - `audit_truth`: cross-tabulates audit verdict (PASS / FLAG_*) against absolute error of Direct, at threshold 0.10. Writes `audit_truth.json`.
  - `tail_composition`: theme breakdown of each policy's worst-α oracle rows. Writes `tail_composition.json`.

- **Three targeted deep dives**:
  - `audit_drilldown`: binomial p-values for each cell's tail count. Tells us which audit flags are statistically real versus small-sample artifact. Writes `audit_drilldown.json`.
  - `base_clone_forensics`: the bottom-α split between base and clone, including which prompts are unique to each. Writes `base_clone.json`.
  - `reliability`: cheap-score decile bins against pooled oracle mean. Tells us whether the cheap judge has signal in the tail. Writes `reliability.json`.

Each script is also runnable on its own:

```bash
python -m cvar_v4.healthbench_data.analyses.budget_curve --alpha 0.10 --seed 42
python -m cvar_v4.healthbench_data.analyses.audit_drilldown
# ... etc
```

Wall clock: under one minute end-to-end on `n_0 = 100`. Linear in `n_0`.

### Step 3: LaTeX (`pilot.tex`)

Picks up `budget_curve.pdf` and `mean_vs_cvar.pdf` automatically via `\includegraphics`. Numeric tables in the writeup are currently hand-typed and synchronized to the analysis JSON outputs.

When scaling to a new `n_0`, the table cell values will need to be updated by hand. The analysis JSONs are the source of truth; cross-check each cell against them.

## Layout

```
cvar_v4/healthbench_data/
├── data/                    # raw response data (input to analyses)
│   ├── prompts.jsonl
│   ├── responses/
│   └── ...
├── judge_outputs/           # cheap and oracle scores (input to analyses)
├── pipeline.py              # data generation
├── analyze.py               # estimator (step5_oracle_calibrated_uniform)
├── cvar_pilot_table.py      # headline pilot table
├── analyses/                # diagnostic modules
│   ├── _common.py           # shared loaders + matplotlib styling
│   ├── budget_curve.py
│   ├── mean_vs_cvar.py
│   ├── audit_truth.py
│   ├── tail_composition.py
│   ├── audit_drilldown.py
│   ├── base_clone_forensics.py
│   ├── reliability.py
│   └── run_all.py           # end-to-end orchestrator
├── writeup/
│   ├── pilot.tex            # LaTeX source
│   ├── pilot.pdf            # compiled writeup
│   ├── budget_curve.pdf     # figure 1
│   ├── mean_vs_cvar.pdf     # figure 2
│   ├── data/                # all JSON artifacts (this is the source of truth for numeric content)
│   │   ├── pilot_table_alpha_0.1_cov_0.25_seed_42.jsonl
│   │   ├── pilot_table_alpha_0.2_cov_0.25_seed_42.jsonl
│   │   ├── budget_curve.json
│   │   ├── mean_vs_cvar.json
│   │   ├── audit_truth.json
│   │   ├── tail_composition.json
│   │   ├── audit_drilldown.json
│   │   ├── base_clone.json
│   │   └── reliability.json
│   └── RUNBOOK.md           # this file
```

## Future work

The minimum viable path stops here. Extensions:

1. **Auto-render LaTeX tables from JSON**. Today the writeup's numeric tables are hand-typed and may drift from the JSON. A future `analyses/render_tables.py` could read each JSON and emit `.tex` fragments that `pilot.tex` `\input{}`s. This makes the writeup fully reproducible from the data files alone.

2. **Multi-seed orchestration**. `run_all.py` currently runs at one seed. A future version could loop over a list of seeds, aggregate, and report mean and standard error of every diagnostic. This is the path to honest CIs and Type-I / power claims.

3. **Calibration-aware bootstrap CIs**. The writeup deliberately omits CIs because the existing variance machinery double-counted calibration uncertainty. A clean fix is a single calibration-aware cluster bootstrap that resamples prompt clusters and refits the calibrator inside each replicate.

4. **Scaling to `n_0 = 500` or larger**. The data pipeline is parametric and the analyses auto-detect `n_0`. The blocker is upstream API throughput (sync was used in the current pilot because Batch was degraded). Once Batch is healthy or an async parallel-sync executor is added, the same three-command recipe extends to any size.
