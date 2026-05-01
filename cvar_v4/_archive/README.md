# `_archive/` — superseded markdown outputs

_Created 2026-04-28 during the consolidation pass that produced
`cvar_v4/scoping.md` as the master entry point._

## What's in here

11 markdown files whose content has been folded into `cvar_v4/scoping.md`
and `cvar_v4/sections/appendix_dataset.tex`. Nothing was deleted —
files are just moved out of the active workspace so the live tree
contains a single navigational entry point.

```
_archive/
├── scoping_deeper.md                        (24 KB — superseded by scoping.md Part 2)
└── deeper_md/                               (10 files, ~75 KB total)
    ├── hb_meta_eval.md                      (HealthBench physician-rubric Y)
    ├── hh_audit_demo.md                     (HH audit + asymmetric-bite finding)
    ├── hh_better_s.md                       (cheap-S admissibility comparison)
    ├── hh_coverage_sweep.md                 (oracle-coverage × audit-power)
    ├── hh_pairwise.md                       (HH 12-cell mean/CVaR bootstrap CIs)
    ├── hh_real_cheap_s.md                   (HF-Inference real cheap-S, paused on credits)
    ├── replication_per_dataset.md           (10-dataset rank-ordered survey, 6-dim grading)
    ├── rtp_challenging.md                   (RTP `challenging` 6.5× concentration)
    ├── uf_per_source.md                     (UltraFeedback within-source 140.5× ratio)
    └── wildchat_cross_model.md              (WildChat o1-mini vs gpt-4o 32× ratio)
```

## What was kept in the active tree

Source code, build inputs, and the long-form reference all stay live:

- `cvar_v4/scoping.md` — master entry point (rank-ordered exec table,
  headline findings, recommendations, methodology, file inventory)
- `cvar_v4/scoping_data.md` — long-form 20-block per-dataset EDA reference
  (built by `run_eda.py --concat` from `eda/sections/*.md`)
- `cvar_v4/eda/deeper/*.py` — 9 rerunnable analysis scripts (NOT archived;
  re-running any will regenerate its `.md` output next to the script)
- `cvar_v4/eda/deeper/_estimator.py` — self-contained Direct CVaR-CJE +
  Wald audit (no `cje-eval` dependency)
- `cvar_v4/eda/deeper/hh_cheap_s_scores.parquet` — HF-Inference cheap-S
  resume checkpoint (38/2,001 rows scored before HF credits depleted)
- `cvar_v4/eda/sections/01..10_*.md` — raw 20-block EDA per dataset;
  these are the **build inputs** for `scoping_data.md`
- `cvar_v4/eda/datasets/*.py` — 10 dataset specs
- `cvar_v4/eda/{run_eda.py, recipe.py, _utils.py, parts.py}` — harness

## How to revive an archived file

```bash
# To bring a single file back to its original location:
mv cvar_v4/_archive/deeper_md/hh_pairwise.md cvar_v4/eda/deeper/hh_pairwise.md

# To regenerate a file from its source script (preferred; produces fresh
# numbers if data has changed):
python3 -m cvar_v4.eda.deeper.hh_pairwise

# To restore everything in one shot:
mv cvar_v4/_archive/deeper_md/*.md cvar_v4/eda/deeper/
mv cvar_v4/_archive/scoping_deeper.md cvar_v4/
rmdir cvar_v4/_archive/deeper_md cvar_v4/_archive
```

## Why these specifically were archived

All 11 files are derivative outputs of the deeper-analysis `.py`
modules. Their content has been folded into `scoping.md` (Parts 2 + 4)
with the headline numbers preserved. The `.py` modules remain, so any
analysis is rerunnable on demand and will write a fresh `.md` next to
the script.

`scoping_data.md` (157 KB, the long-form 20-block reference) was
**not** archived because it carries detail that `scoping.md` only
summarizes (full ASCII histograms, per-quantile distribution tables,
field inventories, tail samples). Its build inputs in
`eda/sections/*.md` were also kept so `run_eda.py --concat` continues
to work.

`scoping.md` is now the single navigational entry point for the
dataset scoping work; everything else is either a long-form reference,
a rerunnable analysis script, or archived here.
