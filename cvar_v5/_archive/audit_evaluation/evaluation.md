# Audit Ω̂ evaluation framework

Goal: on a known-truth DGP, decide whether each candidate Ω̂ method gives
a Wald audit with **nominal size at η = 0.05** AND **power that scales
with population signal strength**. Diagnose *why* each method passes or
fails, separating variance-estimation from center-of-test issues.

## 1. Setup

The Direct CVaR-CJE pipeline produces three slices per replicate:
- CALIB ⊂ logger oracle, used to fit ĥ_t (cross-fit, K folds).
- EVAL drawn from target, no oracle, drives t̂ via the saddle objective.
- AUDIT ⊂ target oracle (held out), where we evaluate the test.

For each AUDIT row i = 1..n_audit:
```
g_1i(t)  =  1{Y_i ≤ t} − α
g_2i(t)  =  (t − Y_i)_+ − ĥ_t(s_i)
g_i(t)   =  ( g_1i(t),  g_2i(t) )ᵀ                         in R²
```

```
ḡ        =  (1 / n_audit) · Σ_i  g_i(t̂)                   in R²
```

The Wald statistic at level η:
```
W_n      =  ḡᵀ · Ω̂⁻¹ · ḡ                                  reject if W_n > χ²_{2, 1−η}
```

Three nuisances that contribute to the *true* sampling variance of ḡ:
```
N1  audit-side               ḡ varies with which n_audit rows we drew.
N2  calibrator-fit           ĥ_t varies with which n_calib oracle rows we drew.
N3  argmax-on-grid           t̂ varies with which n_eval rows we drew.
```

The truth of the DGP we control:
- Y_target ~ Beta(a, b + δ_y).
- S | Y normal with linear mean (scale + δ_scale)·(Y − 0.5) and σ + δ_sigma.
- t* := q_α(Y_target) — closed form via Beta inverse-CDF.
- h*_t(s) := E_target[(t − Y)_+ | S = s] — a 1-D integral, computable to
  numerical precision via fine quadrature.

## 2. The Wald test under H_0: when does size equal η?

Define ε(calib, eval) := E_audit[ ḡ | calib, eval ]. This is the conditional
mean of ḡ given fixed calibrator and threshold. Three regimes:

- **Population truth**: ĥ = h*, t̂ = t*. Then ε(h*, t*) = 0 by transport
  assumption. ḡ ~ N(0, Σ_oracle / n_audit). Wald → χ²_2 if Ω̂ → Σ_oracle / n_audit.
- **Finite-sample**: ĥ has finite-sample bias relative to h*; t̂ has finite-
  sample variance relative to t*. So ε(ĥ, t̂) ≠ 0. Even with the right Ω̂,
  Wald is non-central χ²_2.
- **Population non-transport**: data violates transport. ε > 0 by construction.
  This is the alternative we want to detect.

Decompose under H_0 (population transport):
```
ḡ           =  ε(ĥ, t̂)            +  audit-side noise
              ↑ "bias-into-null"     ↑ N1 contribution; Var = Σ_audit / n_audit
              random across (calib, eval) realizations

E[ḡ]        =  E_{calib, eval}[ ε(ĥ, t̂) ]                    = bias_systematic
Var[ḡ]      =  E[Var(ḡ | ĥ, t̂)]  +  Var[ε(ĥ, t̂)]            = Σ_audit / n_audit  +  V_pipeline
                ↑ N1                    ↑ N2 + N3
```

Where V_pipeline includes V_cal (N2) and V_argmax (N3) and their cross-term.

The Wald statistic, marginally over (calib, eval) draws:
```
W_n                 =  ḡᵀ Ω̂⁻¹ ḡ
                    ~  non-central χ²_2(λ)        approximately

with non-centrality  λ  =  E[ḡ]ᵀ Var(ḡ)⁻¹ E[ḡ]
                         =  bias_systematic ᵀ (Σ_audit / n_audit + V_pipeline)⁻¹ bias_systematic
```

Wald has nominal size iff TWO conditions hold simultaneously:
1. **Variance condition**: `Ω̂ ≈ Var(ḡ) = Σ_audit / n_audit + V_pipeline`.
2. **Center condition**: `bias_systematic = E[ḡ] ≈ 0` (i.e., `λ ≈ 0`).

If (1) fails: size deviates from η depending on `Ω̂ / Var(ḡ)` ratio
(under-estimate ⇒ over-reject; over-estimate ⇒ under-reject).

If (2) fails: even with the right Ω̂, Wald has non-centrality λ > 0,
producing over-rejection. The size approximately equals
`Pr(non-central χ²_2(λ) > 5.991)`.

If both fail: combined effect.

**Practical implication**: variance estimation alone CANNOT calibrate
the audit when ε ≠ 0. The center must also be addressed.

## 3. Three layers of population truth

What we can compute from the known DGP, in order of difficulty.

### 3.1 Σ_oracle — closed-form / 1-D integrals

Population variance of g_i evaluated at the truth (h*, t*):
```
Σ_oracle, 11  =  Var_target(1{Y ≤ t*})                    =  α (1 − α)
Σ_oracle, 22  =  Var_target((t* − Y)_+ − h*_{t*}(s))       =  E[(g_2*)²]                (1-D integral)
Σ_oracle, 12  =  Cov_target(1{Y ≤ t*}, (t* − Y)_+ − h*_{t*}(s))                          (1-D integral)
```

We compute these by numerical quadrature on `p_target(s, Y) = Beta(Y) · normal(s | Y)`.
This is the lower bound on what any honest Ω̂ must produce — the variance
under perfect information.

### 3.2 Σ_full — empirical Monte Carlo

The actual Var(ḡ) under the realized estimator. No closed form for
isotonic + grid argmax. Approximated by direct simulation:

```
For r = 1..R   (R large, e.g. 5000):
    draw fresh (calib^(r), eval^(r), audit^(r)) under H_0
    fit ĥ^(r), compute t̂^(r), compute ḡ^(r)

Σ_full       :=  n_audit  ·  sample-cov_r(ḡ^(r))
```

Since (calib, eval, audit) are independent across r, sample-cov is unbiased
for Var(ḡ). The factor n_audit puts it on the per-row scale comparable
to Σ_oracle.

By the variance decomposition (independence of calib, eval, audit):
```
Var(ḡ)        =  Σ_audit / n_audit             (N1)
              +  V_cal                          (N2)
              +  V_argmax                       (N3)
              +  cross-terms

Σ_full        ≈  Σ_oracle  +  n_audit · V_pipeline   (with V_pipeline = V_cal + V_argmax + cross)
```

So at any specific (n_audit, n_calib, n_eval), Σ_full minus Σ_oracle
equals the pipeline-side variance contribution scaled to per-row units.

### 3.3 ε — the bias-into-null

Under H_0 (population transport), the audit's null is "E[ḡ] = 0 at the
population truth (h*, t*)". At finite-sample (ĥ, t̂), the realized ḡ has
non-zero mean:

```
ε            :=  E[ḡ_realized]   =  mean_r ( ḡ^(r) )            (same R reps as Σ_full)
```

This is the bias-into-null: the deterministic shift of ḡ due to using
estimators (ĥ, t̂) instead of the truth (h*, t*).

ε vanishes as (n_calib, n_eval) → ∞ but is non-zero at finite samples.
The size of |ε| relative to √(Var(ḡ)) determines the non-centrality
of the Wald statistic.

## 4. The unified diagnostic per method

For each candidate Ω̂ method M, we compute four axes against the DGP truth:

```
A.  Variance bias          :=  mean_r ( Ω̂_M^(r) )  −  Σ_full
                                ↑ does the method's mean match the true variance?

B.  Variance dispersion    :=  trace_var_r ( Ω̂_M^(r) )
                                ↑ how unstable is the method per-rep?

C.  Center bias            :=  mean_r ( ḡ_used_M^(r) )
                                ↑ does the test stat have non-zero center?
                                  (depends on whether method bias-corrects)

D.  Empirical size         :=  fraction_r ( W_M^(r) > χ²_{2, 0.95} )
                                ↑ the operational metric

Predicted size from (A, C) :=  Pr( non-central χ²_2( λ_M ) > 5.991 )
                                   λ_M = ε_used_M ᵀ Ω̂_M_avg⁻¹ ε_used_M  · n_audit · scaling
```

Pass-fail rubric:

```
Method M passes IFF:
    |variance bias| / Σ_full          <  10%        (A)
    Variance dispersion small          relative to mean_M (B)
    |Center bias|                      <  some threshold (C; only meaningful if method bias-corrects)
    |empirical size  −  η|             <  0.02      (D, the headline)
    predicted size                     ≈  empirical size                (consistency)
```

Variance-only methods (1-6) are exempt from C. They're judged on A, B, D.
Bias-correction methods (7+) are judged on all four.

## 5. Diagnosis of failure modes from the four axes

| pattern | A | C | D | diagnosis |
|---|---|---|---|---|
| pass | 0 | 0 | η | calibrated |
| (ignored — see methods 1-6) | <0 | n/a | >η | Ω̂ underestimates (variance failure) |
| bias-into-null | 0 | ≠0 | >η | center failure |
| compounded | <0 | ≠0 | >η | both |
| ridge-like | >0 | n/a | <η | Ω̂ over-estimates (conservative) |
| anti-conservative | n/a | <0 | <η | center over-correction |

This separates the *cause* of failure cleanly. We can decide whether to:
- fix Ω̂ (improve A) — methods 1-6 territory
- fix center (introduce/improve bias correction) — method 7 territory
- fix both — composition of methods

## 6. "Test the test" — sanity protocol

Before trusting the diagnostic on real methods, verify it correctly
identifies passing/failing on synthetic methods with known answers.

### 6.1 Probe S1: oracle method passes

Define synthetic method `M_oracle` with:
- Ω̂ = Σ_oracle / n_audit (closed-form truth, capturing N1 only).
- ḡ_used = ḡ − ε (using oracle ε from Layer 3.3).

Both A and C are zero by construction. Predicted size = η = 0.05.
**Empirical size on R reps must equal η ± MC noise (0.014 at R=1000).**
If our framework reports empirical size away from η, the framework is broken.

### 6.2 Probe S2: variance-only oracle over-rejects predictably

Define `M_var_oracle`:
- Ω̂ = Σ_oracle / n_audit (perfect, but only N1).
- ḡ_used = ḡ (no bias correction; ε ≠ 0).

A: ` Σ_oracle − Σ_full < 0` (under-estimates because misses pipeline variance).
But primarily the FAILURE is C ≠ 0.

Predicted size:
```
λ        =  n_audit · ε ᵀ Σ_oracle⁻¹ ε
size_pred =  Pr( non-central χ²_2(λ) > 5.991 )
```

**Compute predicted size analytically; compare to empirical. Match within MC noise.**
This validates the non-centrality formula.

### 6.3 Probe S3: artificially-inflated Ω̂ under-rejects predictably

Define `M_inflated`:
- Ω̂ = 4 · Σ_full / n_audit (4× the true variance).
- ḡ_used = ḡ − ε (bias-corrected).

A: `+3 · Σ_full` (positive bias).
C: 0.

Wald statistic = ḡ_corrected ᵀ Ω̂⁻¹ ḡ_corrected ~ χ²_2 / 4 (deflated by factor 4).
Predicted size:
```
size_pred  =  Pr( χ²_2 / 4  >  5.991 )  =  Pr( χ²_2 > 23.96 )  ≈  6 · 10⁻⁶
```

**Empirical size must be near zero (< 0.005 at R=1000).**
If empirical is meaningfully above this, framework is broken.

### 6.4 Probe S4: Σ_full decomposition by ablation

Compute three variance components on the DGP via separated MC:

```
V_audit_only       :  pin (calib, eval) at one realization; vary AUDIT;
                      take Var(ḡ).  ≈ Σ_oracle / n_audit at large enough n_audit.

V_calib_only       :  pin (audit, eval); vary CALIB; take Var(ḡ).  ≈ V_cal.

V_eval_only        :  pin (calib, audit); vary EVAL; take Var(ḡ).  ≈ V_argmax.
```

Total predicted from independence:
```
V_predicted        =  V_audit_only  +  V_calib_only  +  V_eval_only
```

**V_predicted should approximately equal Σ_full / n_audit. Match within ~10%.**
If they don't, our Σ_full simulation is missing cross-terms or the
independence assumption is wrong.

### 6.5 Probe S5: ε vanishes as n_calib grows

Compute ε at multiple n_calib values: {300, 600, 1500, 3000}.
**|ε| should monotonically shrink** roughly at the calibrator's rate
(n_calib^{-1/3} for isotonic).

If ε does NOT shrink, either:
- our DGP is poorly defined (transport doesn't hold even at population);
- the calibrator's bias does not vanish (which would falsify standard
  isotonic theory at our DGP).

### 6.6 Probe S6: ε of population (h*, t*) is exactly zero

Use the closed-form h* and t* in place of (ĥ, t̂). Compute ḡ on R reps,
take mean. **Result must be ≈ 0** (within MC noise σ_MC = std(ḡ) / √R).
If non-zero, our population-truth computation is wrong (e.g., quadrature
error in h* or wrong t*).

## 7. End-to-end procedure

```
1. Compute Σ_oracle (Layer 3.1) — once, via numerical integration.
2. Run R = 1000 reps under H_0:
   - Compute realized (ĥ^(r), t̂^(r), ḡ^(r), Ω̂_M^(r)) for each method M.
3. Compute Σ_full (Layer 3.2) and ε (Layer 3.3) from the R reps.
4. Run sanity probes S1–S6. If any fail beyond MC tolerance, halt and
   diagnose.
5. For each method M, compute the four axes (A, B, C, D).
6. Compute predicted size from (A, C); compare to D.
7. Apply pass-fail rubric.
8. Repeat at multiple (n_audit, n_calib) to check robustness.
9. Test power: repeat steps 2-7 under H_1 (transport break). Power
   curve as a function of break strength.
```

A method's verdict:
```
PASS   :  size near η at every (n_audit, n_calib) tested,
          predicted size matches empirical (consistency),
          power scales with population signal under H_1.

FAIL   :  size away from η at some (n_audit, n_calib), AND/OR
          predicted size disagrees with empirical (framework or method bug).
```

## 8. Notes on rates

For isotonic at our DGP, ε ~ O(n_calib^{-1/3}). For the audit to remain
calibrated as samples grow (asymptotic justification), we'd need:
```
n_audit · ε² → 0   ⇔   n_audit · n_calib^{-2/3} → 0   ⇔   n_audit = o(n_calib^{2/3})
```

Empirically we test only finite (n_audit, n_calib); this rate analysis
just tells us which scaling regimes to NOT test (those where size
inflation is asymptotic, regardless of method).
