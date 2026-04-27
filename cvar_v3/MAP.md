# Code ↔ paper section map

`cvar_v3/extension_appendix.tex` is the paper. Every numerical value in it comes
from `cvar_v3/numbers.tex`, which is regenerated from CSVs by
`cvar_v3/regenerate_macros.py`. To refresh the paper after re-running any
experiment: re-run that script and recompile the .tex.

## Pipeline

```
data/cje_dataset.jsonl ────┐
data/responses/*.jsonl ────┤
data/logprobs/*.jsonl ─────┤
                           ▼
              run_arena.py ────────► results_arena.csv     ──► §6 main table, alpha-sweep
                           │                                    + headline narrative numbers
                           │
              dgp.py (semi-synth, fit from arena data)
                           │
              run_monte_carlo.py ──► results_mc.csv        ──► §A.3 reject + coverage table
                           │
              audit_chi2_calibration_mc.py
                           │       ──► audit_chi2_W_values.csv ──► §A.1 Wald-null table
                           │
              audit_failure_modes_mc.py (no CSV yet — §A.2 hardcoded in regenerator)
                           │
              n_sweep_synthetic.py ──► results_n_sweep.csv ──► §A.4 n-sweep table
                           │
                           ▼
              regenerate_macros.py ──► numbers.tex
                           │
                           ▼
              extension_appendix.tex (\input{numbers}) ──► extension_appendix.pdf
```

## File ↔ section table

| File | Paper section it produces |
|---|---|
| `workhorse.py` | §3 Estimator, §4 Audit, §5 Inference, Algorithm 1 |
| `dgp.py` | §A (semi-synthetic DGP, used by all of §A.1–§A.4) |
| `run_arena.py` | §6 Arena results — main table + alpha-sweep |
| `audit_chi2_calibration_mc.py` | §A.1 — Wald audit's null distribution |
| `audit_failure_modes_mc.py` | §A.2 — each moment catches its own failure |
| `run_monte_carlo.py` | §A.3 — audit reject rate and CI coverage |
| `n_sweep_synthetic.py` | §A.4 — what scaling n does and does not buy |
| `regenerate_macros.py` | All numbers in the paper (writes `numbers.tex`) |
| `validate_mean.py` | Mean-CJE benchmark gate (must pass before CVaR is trusted) |
| `tests_workhorse.py` | Unit tests for `workhorse.py` |
| `tests_dgp.py` | Unit tests for `dgp.py` |
| `make_comparison.py` | `comparison.md` (paper-vs-our-numbers cross-check) |
| `make_power_report.py` | `power_analysis.md` (internal report from `results_mc.csv`) |
| `show_alpha_sweep.py` | Quick stdout summary of alpha-sweep (no file output) |
| `notation.md`, `SCHEMA.md`, `power_targets.md` | Internal docs (not in paper) |

## Refresh recipe (after a re-run)

1. Re-run whichever experiment changed:
   - `python3 cvar_v3/run_arena.py` (≈30 min) for §6 numbers
   - `python3 cvar_v3/run_monte_carlo.py --medium` (≈13 min) for §A.3
   - `bash cvar_v3/run_n_sweep_cloud.sh` (cloud) for §A.4
   - `python3 cvar_v3/audit_chi2_calibration_mc.py` (≈5 min) for §A.1
2. `python3 cvar_v3/regenerate_macros.py` (instant) → updates `numbers.tex`.
3. `cd cvar && pdflatex extension_appendix.tex` (twice for refs) → updated PDF.
