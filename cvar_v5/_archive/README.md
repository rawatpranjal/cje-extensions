# cvar_v5/_archive

Frozen research code. **Not part of the production estimator path.**
**Not collected by pytest** (see `conftest.py`).

## What's here

### Joint-calibrator study (closed 2026-05-02)

Investigation of whether a joint `ĝ(s, t)` could cut `var_calib` below the per-t isotonic baseline. Falsified — see `cvar_v5/TODO.md::[joint-calibrator]`.

| File | Purpose |
|---|---|
| `calibrator_alt.py` | 4 alt methods: `fit_smoothed_per_t`, `fit_distribution_regression`, `fit_bivariate_isotonic_pooled`, `fit_gbm_monotone`. Each returns a `CalibratorGrid`-compatible object. |
| `calibrator_alt_test.py` | Structural + statistical tests for the alt methods. Skipped by pytest. |
| `joint_calibrator_sweep.py` | Sweep harness comparing baseline vs alts. Output is/was at `mc/runs/<ts>_joint_calibrator/`. |

### Plug-in CVaR comparison study (closed 2026-05-02)

Investigation of whether classical plug-in CVaR variants on the calibrated mean reward `m̂(s) = 1 − ĥ_1(s)` could match or beat the Direct saddle-point. **Falsified.** Direct dominates by ~14× on RMSE across the full panel; plug-ins carry a Jensen-induced upward bias of ~+0.05 to +0.12. See `cvar_v5/TODO.md::[plugin-quantile]` / `[plugin-ru-dual]`.

| File | Purpose |
|---|---|
| `plugin_estimators.py` | `plugin_quantile_cvar` (empirical α-quantile + tail mean) and `plugin_ru_dual_cvar` (RU dual on m̂). Both return `EstimateResult` with the same signature as `estimate_direct_cvar`. |
| `plugin_estimators_test.py` | Code tests: α=1 collapse to Mean-CJE, monotone-in-α, range, constant-Y, parameter validation. Skipped by pytest. |
| `estimator_comparison_sweep.py` | Sweep harness comparing all 3 estimators on the parametric Beta panel. Output at `mc/runs/<ts>_estimator_comparison/`. |

## Why archived, not deleted

- The negative finding is in `TODO.md` prose. Re-implementing if questions arise is not free.
- One sub-finding (`distribution_regression` strongly wins on U-shaped Y, Beta(0.5, 0.5)) is a real lead for future real-data work where `Y` saturates near boundaries.
- Future engineers should be able to read the implementation rather than reconstruct from prose.

## Why not in production tree

- None of these methods beat the per-t isotonic baseline at the tested scale.
- Default `Config.omega_estimator` and the calibrator factory in production use only `cvar_cje/calibrator.py`.
- Including them in the live test suite adds wall time without informing decisions.

## Pytest contract

`conftest.py` in this directory contains:
```python
collect_ignore_glob = ["*"]
```
so pytest never discovers anything here. To run the archived tests deliberately:
```bash
cd cvar_v5/_archive && python -m pytest calibrator_alt_test.py
```

## Reviving any of these

If `Y` distribution at real data has heavy mass at boundaries, copy `fit_distribution_regression` back into `cvar_cje/calibrator_alt.py` (sibling to `calibrator.py`), wire as a `Config.calibrator_kind` option, retest. Other methods are not worth reviving without specific evidence.
