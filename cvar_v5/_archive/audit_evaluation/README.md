# audit_evaluation

Research harness for evaluating audit-Ω̂ methods on a known-truth DGP.

**Math doc**: `evaluation.md` — derivation, the four diagnostic axes, the
six-probe sanity protocol.

**Code layout**:

| File | Purpose |
|---|---|
| `evaluation.md`           | Math derivation and rubric. Read first. |
| `_truths.py`              | Population truths from the DGP: `t*`, `h*_t(s)`, `Σ_oracle`. Numerical quadrature. |
| `_harness.py`             | Per-rep pipeline + per-method Ω̂. RepResult exposes per-fold t̂^(-k) and raw arrays for bootstrap-based bias correctors. |
| `_bias_corrections.py`    | Three bias-correction methods: `bc_jk_cal` (varies ĥ), `bc_jk_full` (varies ĥ AND t̂^(-k)), `bc_boot` (full-pipeline bootstrap). |
| `_bias_diagnostic.py`     | `diagnose_bias` — center-of-test only. Returns BiasDiag with z-scores. |
| `_variance_diagnostic.py` | `diagnose_variance` — Ω̂ vs Σ_full / n_audit. Returns VarDiag with frob_rel_err and spectral_ratio. |
| `_size_diagnostic.py`     | `diagnose_size` — empirical and per-rep-Ω̂ Jensen-correct predicted size. |
| `_diagnostics.py`         | Legacy 4-axis diagnostic; thin shim. New code should use the three diagnostic modules above. |
| `_sanity.py`              | "Test the test": nine probes (S1–S9) verifying framework correctness. |
| `run_evaluation.py`       | CLI runner. Sanity → main eval → write three CSVs + truth + log. |

**Output**: writes to `cvar_v5/mc/runs/<ts>_audit_evaluation/` with files:
- `sanity.csv` — probes S1–S9 pass/fail
- `bias_report.csv` — per (variance × bc) center + z
- `variance_report.csv` — per variance method, Ω̂-vs-target
- `size_report.csv` — per (variance × bc) empirical and predicted size
- `truth.csv` — Σ_oracle, Σ_full, ε, run params
- `log.txt`

## Run

```bash
python -m cvar_v5._archive.audit_evaluation.run_evaluation
# arguments: --R 300 --n-calib 600 --n-audit 250 --n-eval 1000 --B 200
```

A complete run (with sanity) takes ~5–10 min. The sanity probes themselves
are the slow part; pass `--skip-sanity` once you've verified the framework
once.

## Pytest contract

Excluded from the top-level pytest run by `_archive/conftest.py`. To run
inline tests deliberately (none currently — research code):

```bash
cd cvar_v5/_archive && python -m pytest audit_evaluation/
```
