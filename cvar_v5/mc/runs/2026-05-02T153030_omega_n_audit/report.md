# omega_n_audit sweep — report

Run dir: `/Users/pranjal/Code/cvar-cje/cvar_v5/mc/runs/2026-05-02T153030_omega_n_audit`

Setting: n_oracle ∈ [1500, 3000, 6000], R=50, α=0.1, audit_B=500, K=5, all 4 policies. Target size η=0.05.

## Mean size by (n_audit, Ω̂), averaged over policies × reps

| n_oracle | n_audit (median) | Ω̂ | mean_size | size_dev |
|---|---|---|---|---|
| 1500 | 249 | analytical | 0.165 | 0.115 |
| 1500 | 249 | boot_fixed | 0.165 | 0.115 |
| 1500 | 249 | boot_remax_no_ridge | 0.120 | 0.070 |
| 1500 | 249 | boot_remax_ridge | 0.000 | 0.050 |
| 3000 | 504 | analytical | 0.230 | 0.180 |
| 3000 | 504 | boot_fixed | 0.240 | 0.190 |
| 3000 | 504 | boot_remax_no_ridge | 0.150 | 0.100 |
| 3000 | 504 | boot_remax_ridge | 0.000 | 0.050 |
| 6000 | 1002 | analytical | 0.410 | 0.360 |
| 6000 | 1002 | boot_fixed | 0.425 | 0.375 |
| 6000 | 1002 | boot_remax_no_ridge | 0.190 | 0.140 |
| 6000 | 1002 | boot_remax_ridge | 0.000 | 0.050 |

## Decision (at largest n_oracle = 6000)

| Ω̂ | mean_size | size_dev | calibrates (size_dev < 0.05)? |
|---|---|---|---|
| boot_remax_ridge | 0.000 | 0.050 | no |
| boot_remax_no_ridge | 0.190 | 0.140 | no |
| analytical | 0.410 | 0.360 | no |
| boot_fixed | 0.425 | 0.375 | no |

**Result**: NO Ω̂ calibrates even at the largest n_audit. `[omega-research-n-audit]` closes as 'increasing n doesn't help'. Escalate to sub-anchor `[omega-research-derivation]` (re-derive Ω̂ structurally).