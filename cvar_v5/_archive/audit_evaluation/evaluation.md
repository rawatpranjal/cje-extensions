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

### 3.4 Three bias-correction methods

Each method produces a corrected `ḡ_used := ḡ − ε̂` from a different
estimate ε̂ of the bias-into-null. Bias correction is orthogonal to the
choice of Ω̂ method.

#### BC-jk-cal — jackknife on calibrator only

For each oracle fold `k = 1..K`, using cached `ĥ^(-k)` (held-out fit):

```
ḡ^(-k)_A   :=  (1 / n_audit) · Σ_i  g_i( t̂_pooled ;  ĥ^(-k) )
              ↑ keeps t̂ fixed at the original t̂ from the pooled calibrator,
              varies only the calibrator across folds.

ε̂_A        :=  (K − 1) · ( mean_k(ḡ^(-k)_A)  −  ḡ )
ḡ_bc_A     :=  ḡ  −  ε̂_A
            =  K · ḡ  −  (K − 1) · mean_k(ḡ^(-k)_A)
```

Crucially, `g_1 = 1{Y ≤ t̂_pooled} − α` does NOT depend on ĥ. So
`ḡ^(-k)_A[g_1] = ḡ[g_1]` for every k, and the correction on g_1
component is identically zero. **BC-jk-cal corrects g_2 only.**

This is the canonical OUA jackknife adaptation from Mean-CJE — it works
when the bias is concentrated in g_2 (calibrator-fit bias). For our DGP,
the bias is concentrated in g_1 (argmax-induced), so BC-jk-cal is inert.

#### BC-jk-full — jackknife on calibrator AND threshold (NEW)

For each fold `k`, also re-compute the saddle threshold using ĥ^(-k):

```
t̂^(-k)     :=  argmax_t   [ t  −  mean_eval( ĥ_t^(-k)(s_e) )  /  α ]
              ↑ uses the SAME EVAL slice but the leave-one-out calibrator.

ḡ^(-k)_B   :=  (1 / n_audit) · Σ_i  g_i( t̂^(-k) ;  ĥ^(-k) )
              ↑ NOW g_1^(-k) DOES vary across folds, since 1{Y ≤ t̂^(-k)} − α
              shifts when t̂^(-k) shifts.

ε̂_B        :=  (K − 1) · ( mean_k(ḡ^(-k)_B)  −  ḡ )
ḡ_bc_B     :=  ḡ  −  ε̂_B
            =  K · ḡ  −  (K − 1) · mean_k(ḡ^(-k)_B)
```

Captures both the calibrator-fit and argmax contributions to bias.
**This is the piece needed to bias-correct g_1.**

Cost: K evaluations of the saddle objective on EVAL — cheap, since
the per-fold ĥ^(-k) is already cached.

Caveat: the formula `K · ḡ − (K − 1) · ḡ_jk` AMPLIFIES per-rep variance.
With K = 5, the bias-corrected estimator has roughly K² · Var(ḡ) on
components where ḡ_jk is uncorrelated with ḡ, and intermediate values
when correlation is partial. The size diagnostic must use the empirical
Var(ḡ_used) of the corrected statistic — not Var(ḡ) — to predict size
correctly.

#### BC-boot — full-pipeline bootstrap (NEW)

For `b = 1..B` (B ~ 100 typically):

```
For b = 1..B:
    Resample CALIB rows  →  refit  ĥ^(b)
    Resample EVAL rows   →  re-max t̂^(b) on bootstrap eval with ĥ^(b)
    Resample AUDIT rows  →  ḡ^(b) := mean over bootstrap audit at (ĥ^(b), t̂^(b))

bias_boot   :=  mean_b(ḡ^(b))  −  ḡ
ε̂_C         :=  bias_boot
ḡ_bc_C      :=  ḡ  −  bias_boot
              =  2 · ḡ  −  mean_b(ḡ^(b))
```

Captures all three nuisances jointly INCLUDING cross-terms (the bias
estimate sees the full pipeline's joint randomness). Cost: B
calibrator refits per audit verdict (vs K for the jackknife methods).

Empirical question: does the bootstrap mean-shift correction work
better than jackknife on non-smooth (isotonic + argmax) estimators?
The standard theory assumes smooth bias expansions; isotonic doesn't
have one, so this is a measurement, not a proof.

### 3.5 Per-rep Ω̂ in the predicted-size formula (Jensen-correct)

Earlier draft used `mean_r(Ω̂_M^(r))` as the inverse in the predicted
Wald statistic:

```
size_pred_naive  =  Pr(  ḡ̃ᵀ (mean Ω̂_M)⁻¹ ḡ̃  >  c  )
                    ḡ̃ ~ N(center, true_var)
```

This systematically under-predicts empirical size because by Jensen on
the convex map `Ω → Ω⁻¹`, `(mean Ω̂)⁻¹` is *smaller* than
`mean(Ω̂⁻¹)`, so the smoothed prediction misses the per-rep tail
behavior where small Ω̂ realizations cause easy rejections.

The Jensen-correct version uses per-rep Ω̂ samples:

```
For r_sim = 1..n_sim:
    idx   ~  Uniform({1, ..., R})           sample a rep uniformly
    Ω̂    :=  Ω̂_M^(idx)                     (NOT the average)
    ḡ̃   ~  N(center, true_var)
    W    :=  ḡ̃ᵀ Ω̂⁻¹ ḡ̃

size_pred  :=  (1 / n_sim) · Σ_{r_sim}  1{ W  >  c }
```

The Wald inverse is drawn from the empirical distribution of `Ω̂_M^(r)`,
not its mean. By Fubini, this equals the per-rep nested-mean prediction.

`true_var` here is computed from `ḡ_used_per_rep` itself (not from raw
ḡ), so any variance amplification introduced by bias correction is
captured automatically.

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

### 6.7 Probe S7: BC-jk-cal corrects g_2 only

For every rep:
```
Δg_1^(r)  :=  ḡ_bc_A^(r)[g_1] − ḡ^(r)[g_1]
Δg_2^(r)  :=  ḡ_bc_A^(r)[g_2] − ḡ^(r)[g_2]
```
Pass:
- `max_r |Δg_1^(r)|  <  1e-12`   (g_1 ⊥ ĥ; per-rep zero)
- `max_r |Δg_2^(r)|  >  1e-9`    (correction does something on g_2)

Falsifies if the implementation accidentally includes some t̂ variation
in BC-jk-cal.

### 6.8 Probe S8: BC-jk-full reduces |center| on both components

```
center_raw[c]   :=  mean_r ( ḡ^(r)[c] )
center_bc_B[c]  :=  mean_r ( ḡ_bc_B^(r)[c] )
tol             :=  1.5 · max(SE_bc_B)
Pass: |center_bc_B[c]| < |center_raw[c]| + tol  for c ∈ {0, 1}
```

The tolerance allows for MC noise. If BC-jk-full doesn't reduce the
center on at least one component (especially g_1, which is the point),
something's wrong with the formula or the per-fold t̂^(-k) computation.

### 6.9 Probe S9: BC-boot ≈ BC-jk-full directionally

```
Δ^(r)        :=  ḡ_bc_C^(r) − ḡ_bc_B^(r)         per rep
center_diff  :=  mean_r ( Δ^(r) )                ∈ R²
se_diff[c]   :=  std_r ( Δ^(r)[c] ) / sqrt(R)
z[c]         :=  | center_diff[c] | / se_diff[c]
Pass:  max_c z[c]  ≤  2.5
```

The two correctors estimate the same population quantity ε via different
finite-sample formulas. They should agree directionally; persistent
disagreement (z > 2.5) suggests one of them targets the wrong quantity
on this DGP.

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

## 7 — Bias-correction findings — parked 2026-05-03

### What we tried

Three bias-correction methods (BC-jk-cal, BC-jk-full, BC-boot) — see
§3.4 — implemented in `_bias_corrections.py` and run at R=300 on the
uniform DGP at n_calib=600, n_audit=250, n_eval=1000.

### What we found

**Bias-into-null is below the noise floor on this DGP.**

```
bc_label        center_g1     center_g2     z_g1    z_g2    passed
none            +0.000373     +0.000098     +0.29   +1.41   yes
bc_jk_cal       +0.000373     +0.000100     +0.29   +1.42   yes
bc_jk_full      −0.000587     −0.000025     −0.33   −0.33   yes
bc_boot         −0.000214     −0.000008     −0.15   −0.12   yes
```

All four are within 1.5σ_MC of zero. The raw center is ~0.0004 on g_1
and ~0.0001 on g_2 — too small to matter operationally on this DGP.

**Bias-correction either doesn't help or actively hurts size.** The
jackknife formula `K·ḡ − (K−1)·ḡ_jk` amplifies per-rep fluctuations:

```
analytical_oua + none           size = 0.070  (best)
analytical_oua + bc_jk_cal      size = 0.067
analytical_oua + bc_jk_full     size = 0.193  ← variance amplified
analytical_oua + bc_boot        size = 0.117
```

bc_jk_cal does essentially nothing because g_2 was already centered.
bc_jk_full and bc_boot move the center slightly but inflate per-rep
variance by ~2×, which dominates and inflates size.

**The headline failure is variance estimation, not bias.**

```
spectral_ratio (mean(Ω̂) / Σ_full):
analytical            0.731
boot_remax            0.621
analytical_oua        0.731   ← OUA jackknife adds nothing (see §3.6)
boot_remax_oua        0.622   ← same
boot_remax_ridge     16.78    ← way over-shoots
```

Every non-ridge method captures only 60–75% of the true variance.
That is the size-inflation mechanism on this DGP.

### Where the code went

Preserved at:
- `_bias_corrections.py` — all three methods.
- `_bias_diagnostic.py` — center-of-test diagnostic.
- `_size_diagnostic.py` — Jensen-correct predicted size.
- Sanity probes S7-S9 — all pass.

Provenance: `cvar_v5/mc/runs/2026-05-03T070835_audit_evaluation/`.

### Do not re-investigate

Unless a future DGP shows materially larger bias (heteroscedastic S|Y,
very small n_calib, skewed Y), the bias side is solved on uniform Y.
The next iteration tackles the variance gap.
