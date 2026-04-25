# cje-extensions

Extensions to [Causal Judge Evaluation](https://arxiv.org/abs/2512.11150).

## Contents

- [`cvar_v2/`](cvar_v2/) — current version of the CVaR extension. Direct estimator on real Arena data, transport audit with cross-fit Wald variance, semi-synthetic Monte Carlo validation, and the LaTeX appendix.
  - `extension_appendix.tex`, `extension_appendix.pdf` — the appendix.
  - `workhorse.py`, `dgp.py` — core estimator, audit, bootstrap, and per-policy DGP fit.
  - `run_arena.py`, `validate_mean.py` — real-data sweeps and the Mean-validation gate.
  - `run_monte_carlo.py`, `audit_chi2_calibration_mc.py`, `audit_failure_modes_mc.py` — Monte Carlo experiments backing the appendix.
  - `make_power_report.py`, `make_comparison.py`, `show_alpha_sweep.py`, `fill_appendix_table.py` — aggregators.
  - `tests_workhorse.py`, `tests_dgp.py` — sanity tests.
  - `SCHEMA.md` — pipeline schema and citations.
  - `power_analysis.md`, `power_targets.md`, `comparison.md`, `validate_mean_report.md` — generated reports.
  - `run_full_cloud.sh` — opt-in cloud runner.
- [`cvar/`](cvar/) — initial demo of the CVaR extension on synthetic data (notebooks).
- [`arxiv_source/`](arxiv_source/) — LaTeX source of the CJE paper.
