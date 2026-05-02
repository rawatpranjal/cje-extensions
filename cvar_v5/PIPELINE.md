# cvar_v5 pipeline

```
              oracle_rows (logger, with Y)        eval_rows (target, fresh)
                          │                                      │
                          ▼                                      │
             ┌──────────────────────────┐                        │
             │ partition_oracle         │                        │
             │ hash(prompt_id) mod K+1  │                        │
             └────────┬──────────┬──────┘                        │
                      │          │                               │
                      ▼          ▼                               │
                  CALIB ⫫ AUDIT  (assertion: prompt_id sets disjoint)
                      │          │                               │
                      ▼          │                               │
             ┌────────────────┐  │                               │
             │ fit calibrator │  │                               │
             │  grid (K-fold) │  │                               │
             └────────┬───────┘  │                               │
                      │          │                               │
            ĥ_t(s) ───┴──────────┴───────────────────────────────┤
                      │          │                               │
                      │          │                               ▼
                      │          │                     ┌────────────────┐
                      │          │                     │ estimate_direct│
                      │          │                     │ Ψ̂_α(t), find   │
                      │          │                     │ t̂_α            │
                      │          │                     └────────┬───────┘
                      │          │                              │
                      │          │                       (t̂_α, ĈVaR_α)
                      │          │                              │
                      │          ▼                              │
                      │   ┌──────────────┐                      │
                      │   │ wald audit   │ ◄── t̂_α ─────────────┘
                      │   │ Ω̂ via chosen │
                      │   │ estimator    │
                      │   └──────┬───────┘
                      │          │
                      │      AuditVerdict
                      │          │
                      └──────────┴────────►  pipeline returns
                                            (EstimateResult, AuditVerdict)
```

## Per-node math contract

| Node | Math |
|------|------|
| `partition_oracle` | `fold_id_i = hash(prompt_id_i, seed) mod (K+1)`. Bucket `K` is AUDIT; `0..K-1` are CALIB folds. |
| `fit_calibrator_grid` | For each `t ∈ T`: `ĥ_t(s) ≈ E_p0[(t−Y)_+ \| s]` via `IsotonicRegression(increasing=False)`. Cross-fit OOF predictions on CALIB; pooled fit applied to EVAL/AUDIT. |
| `estimate_direct_cvar` | `Ψ̂_α(t) = t − (1/(αn)) Σ_i ĥ_t(s_i)`; `ĈVaR_α = max_t Ψ̂_α(t)`; `t̂_α = argmax_t Ψ̂_α(t)` over `T = linspace(0,1,61)`. |
| `two_moment_wald_audit` | `g_1 = 1{Y ≤ t̂_α} − α`, `g_2 = (t̂_α − Y)_+ − ĥ_{t̂_α}(s)`. `W_n = n_audit · ḡᵀ Ω̂⁻¹ ḡ ~ χ²_2`. Reject (REFUSE-Level) if `W_n > χ²_{2,1−η}`. |
| α=1 collapse | `t̂_1 = 1` (grid right-endpoint); `g_1 ≡ 0`; `g_2 = f̂(s) − Y` (negative mean residual); `ĈVaR_1 = (1/n) Σ_i f̂(s_i) = Mean-CJE`. |

## Runtime invariants (assertions in `pipeline.py`)

- `set(CALIB.prompt_id) ∩ set(AUDIT.prompt_id) = ∅`
- `Y ∈ [0,1]` element-wise where present (oracle rows only)
- `eval_rows` carries no `Y` requirement (target-policy fresh draws)
- `t̂_α ∈ T`

## MC outputs

Each `python -m cvar_v5.mc.runner --<mode>` invocation creates a fresh
timestamped directory under `cvar_v5/mc/runs/<YYYY-MM-DDTHHMMSS>_<mode>/`
containing `results_mc.csv`, `run_config.json` (full Config + ModeParams +
git_sha), `log.txt`, and (after `python -m cvar_v5.mc.report`) `mc_validation.md`.
Old runs are preserved; nothing is overwritten. `report.py` defaults to the
latest run dir; pass `--run-dir <path>` to target a specific one.
