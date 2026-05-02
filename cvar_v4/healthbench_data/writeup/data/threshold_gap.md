# Threshold gap t̂ vs t*

- **α** = `0.1`
- **coverage** = `0.25`
- **seed** = `42`
- `t̂` = saddle-point threshold from FULL cheap target distribution
- `t*` = atom-split α-quantile of full oracle Y panel (truth)
- `gap = t̂ − t*` — small = direct cutoff agreement; non-trivial gap with low |error| = stop-loss correction via g₂

| policy | n_audit | t̂ | t* | gap | |error| | mean_g1 | mean_g2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | 26 | -0.108 | -0.103 | -0.004 | 0.091 | -0.023 | +0.006 | PASS |
| clone | 132 | -0.108 | -0.083 | -0.025 | 0.075 | -0.002 | -0.005 | PASS |
| premium | 132 | -0.057 | +0.000 | -0.057 | 0.031 | +0.006 | -0.003 | PASS |
| parallel | 132 | -0.185 | -0.219 | +0.034 | 0.132 | +0.044 | +0.013 | PASS |
| unhelpful | 132 | -0.236 | -0.286 | +0.049 | 0.081 | +0.044 | -0.003 | PASS |
| risky | 132 | -0.159 | -0.189 | +0.029 | 0.150 | +0.006 | -0.002 | PASS |
