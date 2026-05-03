# audit_evaluation

Research harness for evaluating audit-Ω̂ methods on a known-truth DGP.

**Math doc**: `evaluation.md` — derivation, the four diagnostic axes, the
six-probe sanity protocol.

**Code layout**:

| File | Purpose |
|---|---|
| `evaluation.md`        | Math derivation and rubric. Read first. |
| `_truths.py`           | Population truths from the DGP: `t*`, `h*_t(s)`, `Σ_oracle`. Numerical quadrature. |
| `_harness.py`          | Per-rep pipeline + per-method Ω̂. |
| `_diagnostics.py`      | 4-axis diagnostic per method, predicted-size from non-central χ². |
| `_sanity.py`           | "Test the test": six probes (S1–S6) verifying framework correctness. |
| `run_evaluation.py`    | CLI runner. Sanity → main eval → write CSV + log. |

**Output**: writes to `cvar_v5/mc/runs/<ts>_audit_evaluation/`.

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
