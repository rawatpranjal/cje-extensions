"""Monte Carlo validation of the cvar_v4 estimator stack on a known-truth
semi-synthetic HealthBench DGP.

Layout:
  dgp.py           — fit per-policy DGP from HealthBench data
  pipeline_step.py — inner MC rep: full v4 pipeline, returns one CSV row
  runner.py        — outer MC loop, fork-pool parallel
  report.py        — aggregate results_mc.csv to mc_validation.md + figure
  tests.py         — invariants gate (DGP round-trip, audit size, power monotone)

The estimator/audit/CI machinery is reused unmodified from
cvar_v4/eda/deeper/_estimator.py and cvar_v4/healthbench_data/oracle_design.py.
"""
