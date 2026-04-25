# Validate Mean CJE — blocking benchmark report

- Oracle coverage: 0.25
- Seeds: [0, 1, 2, 3, 4] (each seed varies fold assignment AND oracle slice, per `base.py:100`)
- Estimator: `CalibratedDirectEstimator` (`cje-eval==0.2.10`)
- Tolerance: |median(Mean) − oracle truth| ≤ 0.01 AND across-seed [min, max] ⊇ oracle truth
- Adversarial policies exempt: ['unhelpful']

| Policy | Median Mean | Across-seed [min, max] | Oracle truth | \|Δ\| | range ⊇ truth | Pass |
|---|---|---|---|---|---|---|
| `clone` | 0.7580 | [0.7556, 0.7648] | 0.7620 | 0.0040 | ✓ | ✓ |
| `parallel_universe_prompt` | 0.7703 | [0.7656, 0.7755] | 0.7708 | 0.0005 | ✓ | ✓ |
| `premium` | 0.7641 | [0.7588, 0.7683] | 0.7623 | 0.0019 | ✓ | ✓ |
| `unhelpful` | 0.4228 | [0.3186, 0.4494] | 0.1426 | 0.2802 | ✗ | ✓ _(exempt (catastrophic shift))_ |

**Overall: PASS**
