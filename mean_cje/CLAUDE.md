# mean_cje

Empirical scrutiny of the original CJE paper's Mean-CJE machinery on a
known-truth DGP. Goal: figure out which pieces of the paper's stack are
needed for unbiased estimates and 90–95% coverage.

**Spec:** `docs/superpowers/specs/2026-05-02-mean-cje-empirical-scrutiny-design.md`

## Hard scope rule

Nothing in `mean_cje/` lives outside the spec. If a file isn't named in
the spec, it doesn't belong here.

## What this folder is NOT

- Not a production estimator. Production CVaR-CJE lives in `cvar_v5/`.
- Not an audit-size study. That's a separate question.
- Not a real-data adapter.

## Components

- `lib/calibrator.py`  isotonic on S, with cross-fit
- `lib/estimators.py`  plug_in_mean, aipw_one_step
- `lib/variance.py`    var_eval_if, var_cal_jackknife, wald_ci, bootstrap variants
- `lib_test.py`        code tests
- `exp_coverage.py`    CLI ablation runner

## DGP

Imported from `cvar_v5.mc.dgp.DGP` and `DEFAULT_POLICIES`. Don't duplicate.

## Tests

`pytest mean_cje/lib_test.py`. Code-only tests; no statistical tests at
the unit level — statistical claims are validated via `exp_coverage.py`
end-to-end.
