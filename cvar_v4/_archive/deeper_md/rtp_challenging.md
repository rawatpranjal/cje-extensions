# RealToxicityPrompts — `challenging` subset stratified analysis

_n = 98,892 (prompt, continuation) pairs. Y_safe = 1 - continuation_toxicity (so lower tail = most toxic continuations). `challenging` boolean flag identifies prompts that consistently elicited toxic continuations._

## Subset sizes

- `challenging = True`: **1,187** rows (1.2%)
- `challenging = False`: **97,705** rows (98.8%)

## Y_safe = 1 - toxicity per subset

| subset | n | mean | std | q_0.05 | q_0.01 | CVaR_0.05 | CVaR_0.01 |
|---|---|---|---|---|---|---|---|
| challenging=True | 1,187 | 0.4520 | 0.3450 | 0.0347 | 0.0190 | 0.0255 | 0.0129 |
| challenging=False | 97,705 | 0.6250 | 0.3057 | 0.0840 | 0.0429 | 0.0595 | 0.0319 |
| ALL | 98,892 | 0.6229 | 0.3067 | 0.0813 | 0.0419 | 0.0583 | 0.0309 |

## Tail concentration in `challenging` subset

At Y_safe ≤ q_0.01 = 0.0419 (the global 1% threshold):
- 989 rows total in this tail
- 77 of them (7.8%) are in `challenging=True`
- vs `challenging=True` is 1.2% of the full corpus
- **Concentration ratio**: 6.49× — the challenging subset is 6.5× over-represented in the worst-1% tail.

At Y_safe ≤ q_0.05 = 0.0813 (the 5% threshold):
- 4,945 rows; 221 (4.5%) are challenging — concentration ratio 3.72×.

## CVaR-CJE motivation: π_base policy on full corpus vs π_audit-restricted

Suppose we evaluate two hypothetical policies whose responses we'd generate fresh:

- π_uniform: applies same generation strategy to every prompt → its tail toxicity matches the dataset's overall pattern
- π_challenging-aware: routes `challenging` prompts to a stricter refusal policy, others to π_uniform

On the full 98,892-row corpus the two policies' *mean* Y_safe will be near-identical (challenging is only 1.2% of mass). But their *CVaR_0.05* will diverge by approximately the per-subset gap in `Y_safe @ q_0.05` between challenging and other subsets:

- Y_safe q_0.05 conditional on `challenging=True`: **0.0347**
- Y_safe q_0.05 conditional on `challenging=False`: **0.0840**
- gap: **0.0493**

This is exactly the 'same prompts, different response strategies → similar mean, different tail' story that motivates the method — and the `challenging` flag IS the audit covariate that surfaces it.

## Audit prediction

Per the framework, calibrate on the `challenging=False` subset (the body of the data, ~98.8%) and run the audit on `challenging=True`. The audit should fire because:
- The Y conditional distribution under challenging-True is shifted left of challenging-False
- The S→Y relationship (Perspective toxicity → 1 - Perspective toxicity) is mechanically similar but the realized values differ
- An honest CVaR-CJE pipeline must refuse to certify the level claim on the challenging stratum and report a stratum-conditional CVaR instead.

This is the cleanest *within-dataset* audit-discriminativeness demo we have — **a single boolean covariate that the audit must flag**.