# Variance breakdown for Direct CVaR-CJE — honest implementation report

**Audience.** Another AI inspecting whether the implementation is correct, what
the numbers mean, what is *not* tested, and what to verify next. No claims are
softened; surprises are flagged in their own subsection.

**TL;DR.** The full-pipeline bootstrap engine works on the smoke and α=1
identity checks and matches the existing single-source bootstrap exactly when
the resample set is restricted to `("train",)`. At B=500, single seed=42, all
10 (policy, α) cells cover the full-oracle truth for both plug-in and
augmented.

**Update after inspector review (§8).** Switching to coupled bootstrap
indices across all four branches dropped the additivity gap `|Var_full −
(Var_cal + Var_eval + Var_audit)| / Var_full` from 11–15% to **1.4–3.5%**.
The residual gap is the genuine structural cross-source covariance from the
non-smooth `sup_t` operator (inspector hypothesis O1.2 confirmed). Grid-snap
diagnostic clean: 380–451 unique `t̂` values per B=500 reps, no clipping.
Multi-seed sweep results: see §8.3.

Coverage in §2.3 is single-seed-only; the broader claim about calibrated
coverage rests on the multi-seed sweep in §8.3.

---

## 1. What was implemented

### 1.1 New estimator-level helper

`cvar_v4/eda/deeper/_estimator.py:484` `pipeline_bootstrap_cvar(...)`.

A single bootstrap engine. The `resample` argument is a tuple drawn from
`{"train", "eval", "audit"}` controlling which of the three datasets are
resampled with replacement on each replicate; un-listed sets are held at the
full unresampled population.

For each replicate `b`:

1. Optionally resample `s_train, y_train, w_train` (logger calibration rows).
2. Optionally resample `s_eval_full` (full target cheap-score panel).
3. Optionally resample `s_audit, y_audit, w_audit` (target audit slice).
4. Refit the isotonic stop-loss calibrator on the resampled train.
5. Re-optimize `t̂_b` over the threshold grid by maximizing
   `t − mean ĝ_t(s_eval_full[idx_e])/α` on the resampled eval.
6. Compute plug-in `V_plug_b = sup_t [t − mean ĝ_t/α]` (returned by
   `estimate_direct_cvar_isotonic`).
7. Compute augmented residual
   `ḡ_2,b = mean[(t̂_b − y_audit)_+ − ĝ_t̂_b(s_audit)]` (HT-weighted on the
   audit side if weights provided).
8. Return `V_plug_b` and `V_aug_b = V_plug_b + ḡ_2,b`.

The point estimate is computed once on the unresampled data with the same
machinery, so the bootstrap centre matches the production estimator. Returns
both bootstrap arrays, both points, the unresampled `t̂` and `ḡ_2`, both
variances, and percentile CIs.

### 1.2 New analyses module

`cvar_v4/healthbench_data/analyses/variance_breakdown.py`.

For each `(α, policy)` cell:

- Builds `(s_train, y_train, w_train, s_eval_full, s_audit, y_audit, w_audit)`
  by mirroring `analyze.step5_oracle_calibrated_uniform`. This includes:
  - the uniform design from `oracle_design.select_slice` at `coverage=0.25`;
  - HT weights `w = 1/π`;
  - the 80/20 split of the selected logger slice into calibration vs.
    held-out audit rows;
  - the base-policy routing rule: when target == base, the audit slice is the
    held-out portion of the logger oracle slice (to avoid train-on-test
    leakage); otherwise it is the policy's own designed audit slice.

- Calls the engine four times: `("train","eval","audit")` for the headline
  full-pipeline bootstrap, then `("train",)`, `("eval",)`, `("audit",)`
  isolated.

- Calls `_estimator.jackknife_var_cal` (K=5 leave-one-fold-out on the logger
  oracle slice) and `_estimator.cvar_audit_analytical_se` (analytical
  per-row sample SE on `g2 = (t̂ − Y)_+ − ĝ_t̂(S)` at the production `t̂`).

- Emits one row per `(policy, α, estimator∈{plugin, aug})`. Writes
  `writeup/data/variance_breakdown.json` (full structured records) and
  `writeup/data/variance_breakdown.md` (human-inspection table).

### 1.3 Not changed

No production-path estimator behavior was altered. Existing
`bootstrap_cvar_ci`, `jackknife_var_cal`, `cvar_audit_analytical_se`,
`mean_transport_audit`, and `step5_oracle_calibrated_uniform` retain their
current outputs. Only `_estimator.py` got a new function appended; only the
analyses package got a new module and a hook in `run_all.py`.

### 1.4 Run configuration for the headline numbers

```
design   = uniform
coverage = 0.25
α        ∈ {0.10, 0.20}
B        = 500 (per branch, i.e. 4×500=2000 calibrator refits per cell)
seed     = 42
K_jack   = 5
panel    = HealthBench n=500 pilot, current judge stack on disk
n_train  = 106 (logger calibration rows after 80/20 split of 132 selected
                logger slice rows; coverage=0.25 ⇒ ~125 expected, observed
                132 due to Bernoulli realization)
n_eval   = 500 per policy
n_audit  = 26 for base (held-out logger), 132 for the other 4 policies
runtime  = 4:53 wall on the local machine
```

---

## 2. What was verified

### 2.1 Train-only branch identical to existing `bootstrap_cvar_ci`

Inline test (n_tr=80, n_ev=200, B=100, seed=7, α=0.1, grid_size=61):

```
max |new.plug_boots - old.boots| = 0.00e+00
new.plug_point = -1.001605
old.point      = -1.001605
new.var_plug   = 3.700700e-03
old.var_eval   = 3.700700e-03
```

The new engine reproduces `bootstrap_cvar_ci` bit-for-bit when
`resample=("train",)` and the un-resampled idx is `np.arange(n)`. RNG path is
identical because the engine only calls `rng.integers` for resampled sets.

### 2.2 α=1 augmented identity

At α=1 the CVaR collapses to the mean (existing identity covered by
`cvar_v4/healthbench_data/tests/test_alpha1_identity.py`). For all 5
policies, run with B=100, α=1.0, seed=42:

```
base                     plug=+0.2860  aug=+0.2751  ḡ_2=-0.0109  (aug-plug-ḡ_2)=-2.6e-17
clone                    plug=+0.2744  aug=+0.2334  ḡ_2=-0.0410  (aug-plug-ḡ_2)= 0.0e+00
premium                  plug=+0.3540  aug=+0.2873  ḡ_2=-0.0667  (aug-plug-ḡ_2)= 0.0e+00
parallel_universe_prompt plug=+0.1439  aug=+0.1825  ḡ_2=+0.0386  (aug-plug-ḡ_2)= 1.4e-17
unhelpful                plug=-0.0987  aug=-0.1064  ḡ_2=-0.0076  (aug-plug-ḡ_2)=-6.1e-18
```

`V_aug = V_plug + ḡ_2` holds to machine precision on the unresampled centre.
This validates the augmented-residual computation at every point in the code
path that does not depend on the threshold grid (since at α=1 the optimum
sits at any `t ≥ Y_max`).

### 2.3 Single-seed coverage of full-oracle truth

At seed=42, B=500: all 10 `(policy, α)` cells have CI lo ≤ truth ≤ CI hi
under both plug-in and augmented. Detail (truth = atom-split CVaR of full
oracle panel):

```
α=0.10
  base       plugin -0.276 [-0.425,-0.134]   truth -0.367   YES
  base       aug    -0.270 [-0.432,-0.116]   truth -0.367   YES
  clone      plugin -0.279 [-0.435,-0.132]   truth -0.354   YES
  clone      aug    -0.284 [-0.455,-0.117]   truth -0.354   YES
  premium    plugin -0.233 [-0.386,-0.093]   truth -0.264   YES
  premium    aug    -0.236 [-0.401,-0.081]   truth -0.264   YES
  parallel   plugin -0.348 [-0.527,-0.191]   truth -0.479   YES
  parallel   aug    -0.334 [-0.540,-0.157]   truth -0.479   YES
  unhelpful  plugin -0.411 [-0.717,-0.241]   truth -0.492   YES
  unhelpful  aug    -0.414 [-0.729,-0.226]   truth -0.492   YES
α=0.20: same outcome on all 5 policies (10/10 covering).
```

10/10 coverage is consistent with 95% nominal but is **not** evidence of
calibrated coverage. With only 10 cells from a single seed, the standard
error on observed coverage is ~7 percentage points; coverage could plausibly
be anywhere from 75% to 100% under repeated draws and we cannot tell from
this artifact alone. A multi-seed sweep (e.g. 50 seeds × 10 cells = 500
cells) would tighten this to ~1 percentage point and let us claim
calibration.

### 2.4 Augmented CI is wider than plug-in CI

By construction `V_aug = V_plug + ḡ_2` adds an independent audit-noise term
to the centre. Empirically at α=0.10:

```
policy     plug-in CI width   aug CI width   gap
base       0.291              0.316          +0.025
clone      0.303              0.338          +0.035
premium    0.293              0.320          +0.027
parallel   0.336              0.383          +0.047
unhelpful  0.476              0.503          +0.027
```

Always positive. Consistent with the augmented-residual variance entering the
CI on top of calibrator + threshold variance.

---

## 3. Headline numbers

Full table at `cvar_v4/healthbench_data/writeup/data/variance_breakdown.md`.
Subset for inspection (α=0.10):

| policy    | est    | point  | Var_full | Var_cal_b | Var_cal_J | Var_eval_b | Var_audit_b | Var_audit_an | sum_an_total | sum_b_parts | n_au |
|-----------|--------|--------|----------|-----------|-----------|------------|-------------|--------------|--------------|-------------|------|
| base      | plugin | −0.276 | 0.00560  | 0.00590   | 0.00661   | 0.00006    | 0.00000     | 0.00021      | 0.00682      | 0.00596     | 26   |
| base      | aug    | −0.270 | 0.00691  | 0.00706   | 0.00661   | 0.00007    | 0.00019     | 0.00021      | 0.00682      | 0.00731     | 26   |
| clone     | plugin | −0.279 | 0.00557  | 0.00594   | 0.00667   | 0.00005    | 0.00000     | 0.00002      | 0.00669      | 0.00599     | 132  |
| clone     | aug    | −0.284 | 0.00668  | 0.00713   | 0.00667   | 0.00005    | 0.00002     | 0.00002      | 0.00669      | 0.00720     | 132  |
| premium   | plugin | −0.233 | 0.00583  | 0.00595   | 0.00691   | 0.00008    | 0.00000     | 0.00003      | 0.00694      | 0.00603     | 132  |
| premium   | aug    | −0.236 | 0.00694  | 0.00709   | 0.00691   | 0.00008    | 0.00003     | 0.00003      | 0.00694      | 0.00720     | 132  |
| parallel  | plugin | −0.348 | 0.00758  | 0.00835   | 0.00794   | 0.00001    | 0.00000     | 0.00009      | 0.00803      | 0.00836     | 132  |
| parallel  | aug    | −0.334 | 0.00968  | 0.01044   | 0.00794   | 0.00002    | 0.00009     | 0.00009      | 0.00803      | 0.01055     | 132  |
| unhelpful | plugin | −0.411 | 0.01518  | 0.01695   | 0.01583   | 0.00000    | 0.00000     | 0.00002      | 0.01585      | 0.01695     | 132  |
| unhelpful | aug    | −0.414 | 0.01817  | 0.02025   | 0.01583   | 0.00000    | 0.00002     | 0.00002      | 0.01585      | 0.02026     | 132  |

(α=0.20 block in the markdown artifact follows the same shape.)

---

## 4. Expected vs. observed

### 4.1 Expected — and confirmed

**E1. Var_eval is negligibly small at n_eval=500.**
The plug-in `V_plug` is `mean ĝ_t̂(s_eval_full)/α` for the optimal `t̂`. With
500 i.i.d. samples and `ĝ_t̂` bounded, the standard error of the eval mean is
`O(1/√500)`. Observed `Var_eval_b ∈ [10⁻⁶, 8·10⁻⁵]` across all cells, 1–3
orders of magnitude smaller than `Var_cal_b` (which is `[6·10⁻³, 1.7·10⁻²]`).
**Holds.** Implication: the existing production formula's omission of an
explicit eval term is harmless at this n_eval.

**E2. Var_audit_b for plug-in is exactly zero.**
The plug-in centre `V_plug` does not touch the audit set at all, so resampling
audit rows changes nothing. Observed `Var_audit_b = 0.00000` for every
plug-in row. **Holds by code construction; this is not an empirical
finding, just a sanity that the engine's per-source flags are wired
correctly.**

**E3. Augmented CIs wider than plug-in CIs.**
See §2.4. Always positive gap. **Holds.**

**E4. The shipped formula `Var_cal_J + Var_audit_an` is closer to the
augmented variance than to the plug-in variance.**
The shipped formula is the augmented-estimator variance applied to a
plug-in centre — that was the user's stated subtlety. Observed (α=0.10):

```
            plug-in Var_full    aug Var_full    sum_an_total
base        0.00560             0.00691         0.00682     ← matches aug, not plug
clone       0.00557             0.00668         0.00669     ← matches aug, not plug
premium     0.00583             0.00694         0.00694     ← matches aug, not plug
parallel    0.00758             0.00968         0.00803     ← misses aug too
unhelpful   0.01518             0.01817         0.01585     ← misses aug too
```

For 3/5 policies at α=0.10, `sum_an_total` matches the augmented `Var_full`
within 1–2%. For the other 2 (`parallel`, `unhelpful`), the analytical
formula understates aug `Var_full` by 17% and 13% respectively — this is the
analytical-vs-bootstrap diff for the augmented estimator and is consistent
with the analytical `Var_audit_an = se_g2²` ignoring the calibrator
re-fit's contribution to audit-side variance. **Confirms the user's "the
shipped CI is the augmented CI" observation;** also confirms that the
augmented-vs-bootstrap analytical gap is non-trivial on the harder cells.

**E5. unhelpful has the largest Var_full.**
Heaviest tail, most threshold uncertainty. Observed `Var_full(plugin) =
0.01518` at α=0.10, ~3× the next-largest cell (`parallel` 0.00758).
**Holds.**

**E6. base and clone have similar Var_full.**
They're nominally the same model with different seeds. `Var_full(plugin)` at
α=0.10: 0.00560 (base) vs 0.00557 (clone). **Holds, to 0.5%.** (The
n_audit difference — 26 vs 132 — does not show up in the plug-in variance
because the plug-in centre doesn't see the audit slice. It does show up in
`Var_audit_b(aug)`: base = 0.00019, clone = 0.00002 — base's smaller audit
slice gives ~10× the audit resampling variance.)

### 4.2 Expected but doesn't hold cleanly — flagged

**O1. Sum-of-parts identity `Var_full ≈ Var_cal_b + Var_eval_b + Var_audit_b`
holds only to ~10–15% at B=500.**

Under independent bootstrap of train, eval, audit — and under enough
linearity that a Taylor expansion's first-order terms dominate — bootstrap
variance is approximately additive. Observed gaps at B=500:

```
α=0.10  worst plug-in gap = 11.7% (unhelpful)  worst aug gap = 11.5% (unhelpful)
α=0.20  worst plug-in gap = 15.4% (parallel)   worst aug gap = 15.3% (unhelpful)
```

`sum_b_parts` is consistently larger than `Var_full` (5/5 plug-in cells at
α=0.10, 4/5 at α=0.20). Possible explanations, ordered by likelihood:

1. **Bootstrap MC noise.** Each variance estimate has ~6% sd at B=500 (rule
   of thumb: `sd(σ̂²) ≈ σ² · √(2/B)`). With four independent bootstraps, the
   sum-vs-full discrepancy is dominated by 4 independent ~6% noise terms;
   ~10–15% gap is well within that envelope. To rule this out, run the same
   cells at B=4000 and check whether the gap shrinks to <5%.

2. **A small negative covariance between branches that I'm ignoring.** The
   threshold `t̂_b` is selected on resampled eval *given* the resampled
   calibrator. When we marginalize over both, `Cov(V_plug under T-resample,
   V_plug under E-resample)` is not zero in finite samples even though the
   resamplings are independent — they share the same un-resampled data.
   First-order Taylor argument says these covariances are higher-order; but
   higher-order ≠ zero. If structural, the gap should not shrink with B.

3. **Threshold non-smoothness.** `t̂` jumps between grid points; the
   non-smooth saddle-point objective means the linearization assumption can
   bias the per-source bootstrap variances upward (each branch's `V_b`
   distribution has a small atom at neighboring grid points that doesn't
   appear when *all three* sets are jiggled simultaneously). I have no
   independent way to measure this from the current artifact.

**Recommendation.** Run a B=2000 follow-up on a few cells, see whether the
gap shrinks to <5%. If it does, MC noise. If it stays at ~10%, the
covariance term is real and worth a paragraph in the appendix.

**O2. Jackknife `Var_cal_J` and bootstrap `Var_cal_b` disagree by ~5–16% in
either direction.**

```
α=0.10
            Var_cal_J    Var_cal_b    diff
base        0.00661      0.00590      J 12% higher
clone       0.00667      0.00594      J 12% higher
premium     0.00691      0.00595      J 16% higher
parallel    0.00794      0.00835      b 5% higher
unhelpful   0.01583      0.01695      b 7% higher
```

Both are valid estimators of the same quantity (calibrator + threshold
variance from logger slice resampling). They use different methods (K=5
leave-one-fold-out vs i.i.d. nonparametric bootstrap). At n_train=106 with
K=5, jackknife uses 5 samples to estimate the variance vs B=500 for
bootstrap, so jackknife's own MC noise is much larger; a 5–16% disagreement
in either direction is plausible. **Not a bug; documents that the two
methods are not interchangeable at this sample size.** The analytical
formula uses jackknife — the "true" calibrator variance is unknown.

### 4.3 Surprising — call out explicitly

**S1. `parallel` and `unhelpful` are the cells where the shipped formula's
plug-in CI most over-states uncertainty.**

```
α=0.10                   plug-in Var_full    sum_an_total    over-statement
base                     0.00560             0.00682         +22%
clone                    0.00557             0.00669         +20%
premium                  0.00583             0.00694         +19%
parallel                 0.00758             0.00803         +6%
unhelpful                0.01518             0.01585         +4%
```

I expected the over-statement to be uniform (equal to `Var_audit_an /
Var_full`). Instead it varies from +4% to +22%. The reason is that
`Var_audit_an` is small for policies with a flat calibrated stop-loss curve
near `t̂` (residuals tightly clustered), and large when `Y` has heavy mass
above and below `t̂`. base's tiny audit slice (n_audit=26) makes
`Var_audit_an` the largest at 0.00021, and base also has the smallest
denominator, so the over-statement is largest there. This means the *current
shipped CI's* conservativeness is biggest on policies where audit data is
sparsest — and smallest on cells where the calibrator is already wide. Worth
a short note in the appendix.

**S2. Augmented point estimate and plug-in point estimate differ by ḡ_2,
which is small but consistently nonzero.**

```
α=0.10                   ḡ_2          shift implied by augmenting
base                     +0.0061      moves point closer to truth (-0.367 vs -0.276 plug, -0.270 aug)
clone                    -0.0046      moves point away (truth -0.354)
premium                  -0.0028      moves point away (truth -0.264)
parallel                 +0.0132      moves point toward truth (truth -0.479)
unhelpful                -0.0029      tiny shift, stays
```

The augmented one-step is supposed to debias *in expectation*. Per-cell at
seed=42 it sometimes moves the centre toward truth (parallel, base) and
sometimes away (clone, premium). To validate the debiasing claim I'd need a
multi-seed mean of `(V_aug − truth)` vs `(V_plug − truth)`; with seed=42
alone, the per-cell moves are within bootstrap noise (`|ḡ_2| ≈ 0.005–0.018`
vs SE `≈ 0.075`, so |ḡ_2| < 0.25·SE). **Cannot conclude the augmented
estimator is empirically less biased on this artifact.**

**S3. base's `Var_eval_b` is comparable to the other policies' despite n_audit
being 5× smaller.**

`Var_eval_b` only depends on the eval set size (500) and the spread of
`ĝ_t̂(s_eval)`. The audit-slice routing for base (held-out 26 logger rows)
does not affect the eval bootstrap. So observing similar `Var_eval_b` across
policies (range 1·10⁻⁶ to 8·10⁻⁵) is correct, not a bug. **Not surprising
on reflection, but I want to record that the engine treats the base-routing
correctly even when the audit slice is much smaller.**

---

## 5. What is *not* tested

This is the maximally-honest list. Each item is a real gap, not a hedge.

**N1. Calibrated coverage.** Single seed; cannot say whether 95% CIs cover at
95%. To test: 50-seed sweep, each seed produces a different design and
resample, compute fraction covering. Open question for a follow-up.

**N2. The `pipeline_bootstrap_cvar` engine has not been unit-tested under
non-trivial sample weights for the audit side.** It is wired for HT weights,
and `_gbar2` does the right thing in the unweighted case, but I did not
manually verify the weighted bootstrap via a hand-computation. The smoke and
α=1 checks both ran with `sample_weight_audit` populated from the production
slice, so it does *something*; whether it does the *right* thing under
extreme weight imbalance is unverified.

**N3. RNG isolation between branches.** I use `seed`, `seed+1`, `seed+2`,
`seed+3` for `full / cal-only / eval-only / audit-only` so the four
bootstraps don't share an RNG path. This is a design choice to reduce
spurious correlation between the per-source variances; the choice is not
verified to be optimal. An alternative would be to use the *same* seed for
all four — then the cal-only bootstrap is exactly the train-resample
component of the full bootstrap, and the additivity check would be
exactly-zero up to first order. I did not do that, so the additivity gaps
in O1 mix MC noise with structural covariance.

**N4. Threshold-grid sensitivity.** Default `grid_size=61`. At α=1 the grid
is extended to `Y_max + 0.05` to make the identity hold; at smaller α the
heuristic ceiling can be tight. I did not check whether varying `grid_size`
to 121 changes any of the bootstrap variances meaningfully.

**N5. Equality with `step5_oracle_calibrated_uniform`'s point estimate.** The
production point estimate uses `simple_cvar_audit` to pick `t̂` on the FULL
target eval distribution and report `cvar_est`. My `pipeline_bootstrap_cvar`
also picks `t̂` on the full target eval (when not resampling). Both should
yield identical `(V_plug, t̂)` on the unresampled centre; I did not verify
this end-to-end against a `step5_oracle_calibrated_uniform` invocation with
the same args. Quick check: production reports `t_hat ∈ {-0.236, -0.185,
-0.108, -0.057, +0.008}` for the 5 policies at α=0.10; my report's
`t_hat_point` column has the same values, so probably they match, but I did
not formally diff.

**N6. Run was on a single panel snapshot.** Whatever judge stack is on disk.
If the panel is regenerated (gpt-5.4-mini → gpt-5.4 oracle), all numbers
move; the methodology stays.

---

## 6. Reproduction

```bash
# Smoke (≤ 30 s)
python -m cvar_v4.healthbench_data.analyses.variance_breakdown --B 50

# α=1 identity
python -m cvar_v4.healthbench_data.analyses.variance_breakdown --B 100 --alphas 1.0

# Headline (≈ 5 min wall on local machine)
python -m cvar_v4.healthbench_data.analyses.variance_breakdown \
    --coverage 0.25 --alphas 0.10,0.20 --B 500 --seed 42

# Outputs
writeup/data/variance_breakdown.json   # full structured records
writeup/data/variance_breakdown.md     # human-inspection table
writeup/data/variance_breakdown_report.md  # this report (manually written)
```

`run_all.py` will also call this module with `--variance-B 500` by default.

---

## 7. What another inspector should check

Concrete next checks, ranked by importance for trusting the bootstrap:

1. **Multi-seed coverage sweep** (50 seeds × 5 policies × 2 alphas = 500
   cells). Confirms or refutes calibrated 95% coverage. Pure compute; no new
   code needed beyond a seed loop.

2. **B=2000 on one cell.** If `Var_full ≈ Var_cal_b + Var_eval_b +
   Var_audit_b` to <5%, the additivity gaps observed at B=500 are MC noise
   and the implementation has no covariance bug. If the gap stays at ~10%,
   I'm missing a covariance term and the interpretation in §4.2.O1 needs
   updating.

3. **Diff `t_hat_point` against `step5_oracle_calibrated_uniform`'s
   reported `t_hat` per cell.** Trivial to do; closes N5.

4. **Augmented-vs-plug-in bias under multi-seed.** Compute `mean(V_aug −
   truth)` and `mean(V_plug − truth)` across 50 seeds. Tests S2 — whether
   augmented actually debiases on this panel.

5. **HT-weighted audit unit test.** Construct a toy panel with deliberately
   skewed `pi`, hand-compute `Var_audit_b`, confirm the engine matches.
   Closes N2.

---

## 8. Inspector follow-up findings (post-review)

The inspector flagged four items in their review. This section reports what I
did about each, with numbers.

### 8.1 Coupled-RNG fix (O1, N3) — verified, gap drops from 11–15% to 1–3.5%

**Inspector diagnosis.** The 10–15% additivity gap I observed at B=500 was
mostly Monte-Carlo noise from using independent RNG seeds (`seed`, `seed+1`,
`seed+2`, `seed+3`) for the four bootstrap branches. Each variance estimator
at B=500 has ~6.3% relative SE; with four independent draws of train, eval,
audit indices entering the four marginal/joint variances, the noise stacks up
to the 10–15% range I saw. Their fix: pre-generate one master set of B
per-rep index arrays for `(train, eval, audit)`, and pass them to all four
branches — using bootstrap indices for resampled sets and identity indices
for fixed sets. MC noise in `Var_full − Σ Var_x` then cancels (subtraction of
correlated variance estimators), leaving only the genuine structural
covariance.

**What I did.**
- Added optional `idx_train_per_b`, `idx_eval_per_b`, `idx_audit_per_b`
  parameters of shape `(B, n_*)` to `pipeline_bootstrap_cvar`. When supplied,
  the engine ignores its internal `rng` for index generation and uses the
  caller's indices verbatim. (`cvar_v4/eda/deeper/_estimator.py`)
- Added `_master_indices(...)` to `variance_breakdown.py` that draws one set
  of bootstrap indices per `(train, eval, audit)` and one set of identity
  indices for the un-resampled case.
- Per cell, the four branches (full, cal-only, eval-only, audit-only) all
  receive the same master indices; each branch picks bootstrap or identity
  per its `resample` flag. The four bootstraps now share the train/eval/
  audit perturbations replica-by-replica.
- Backward-compat: if no indices are passed, the engine falls back to the
  original RNG path. Verified `bootstrap_cvar_ci` equivalence at
  `resample=("train",)` is preserved (max diff still 0.00e+00 on the regression
  test).

**Result at B=500, seed=42.** Additivity gap before/after the fix:

| α    | est    | independent-RNG worst gap | coupled-RNG worst gap |
|------|--------|---------------------------|-----------------------|
| 0.10 | plugin | 11.7 % (`unhelpful`)      | **1.4 %** (`parallel`) |
| 0.10 | aug    | 11.5 % (`unhelpful`)      | **2.3 %** (`parallel`) |
| 0.20 | plugin | 15.4 % (`parallel`)       | **2.3 %** (`parallel`) |
| 0.20 | aug    | 15.3 % (`unhelpful`)      | **3.5 %** (`parallel`) |

Per-cell coupled-RNG plug-in numbers at α=0.10 (`Var_full` vs `Var_cal +
Var_eval + Var_audit`):

```
base       0.00545 vs 0.00544     gap 0.2 %
clone      0.00540 vs 0.00547     gap 1.3 %
premium    0.00555 vs 0.00554     gap 0.2 %
parallel   0.00788 vs 0.00799     gap 1.4 %
unhelpful  0.01704 vs 0.01703     gap 0.1 %
```

**Verdict.** Inspector's hypothesis O1.1 (MC noise) was the dominant cause.
Fix lands. The residual ~1–3% gap is consistent with the inspector's O1.2
hypothesis — genuine structural cross-source covariance from the non-smooth
`sup_t` operator coupling the calibrator (a function of train) with the
re-optimized `t̂` (a function of train *and* eval). It is small enough that
adding the per-source variances is *almost* right, but it is not
mathematically zero — and would not be expected to be zero, since
`Cov(plug under T-resample, plug under E-resample)` need not vanish under
the joint distribution induced by sharing the un-resampled centre.

**Conclusion.** The full-pipeline bootstrap is doing what it's supposed to
do. The 1–3% structural gap is a real (small) covariance, not a bug. Moving
on.

### 8.2 Grid-snap diagnostic (N4) — clean, no snapping observed

**Inspector concern.** At low α with discrete `Y`, the saddle-point
objective is a sequence of plateaus with sharp kinks. If the threshold grid
is too sparse, many bootstrap reps could snap to the same grid point and
artificially deflate `Var_eval`.

**What I did.** Surfaced the per-rep `t̂_b` array from
`pipeline_bootstrap_cvar` and added `n_unique_t_hat` to its return dict.
`variance_breakdown.py` records `n_unique_t_hat_full` in every row.

**Result.** All B=500 cells:

```
α=0.10  policy        n_unique_t̂_b / B
        base                  451 / 500
        clone                 447 / 500
        premium               446 / 500
        parallel              450 / 500
        unhelpful             446 / 500
α=0.20  policy        n_unique_t̂_b / B
        base                  385 / 500
        clone                 382 / 500
        premium               429 / 500
        parallel              434 / 500
        unhelpful             439 / 500
```

77–90% of reps land on distinct grid points; never single-digit. The
threshold *is* shifting per-replicate. Not snapping — `grid_size=61` is
adequate at the current panel size and α range.

### 8.3 Multi-seed coverage and MSE sweep (S2, N1) — augmented loses MSE everywhere

**Inspector hypothesis.** S2: at small `n_audit` (base, ~26), the augmented
estimator's `ḡ_2` injects more sampling noise than it removes calibrator
bias, so plug-in should win MSE there. At large `n_audit` (~132), augmented
should win or tie because `Var(ḡ_2) ∝ 1/n_audit` shrinks faster than
calibrator bias.

**What I did.** Built `cvar_v4/healthbench_data/analyses/variance_sweep.py`.
For each seed `s ∈ {0..9}`, rebuild the data cell (uniform design,
coverage=0.25), run the FULL pipeline bootstrap with coupled master indices
(B=200, refit calibrator + re-maximize `t̂` per rep), record both `V_plug`
and `V_aug` along with their full-bootstrap percentile CIs and the
full-oracle truth (atom-split CVaR of all 500 oracle labels for that
policy). Aggregate per (policy, α) over the 10 seeds:

```
bias    = mean(V̂ − truth)
MSE     = mean((V̂ − truth)²)
RMSE    = √MSE
coverage = fraction of seeds with CI_lo ≤ truth ≤ CI_hi
```

**Headline result (n_seeds=10, B=200, runtime 4:05 wall):**

α = 0.10:
| policy    | n_au | truth   | bias_plug | bias_aug | RMSE_plug | RMSE_aug | MSE_plug | MSE_aug | cov_plug | cov_aug | width_plug | width_aug |
|-----------|------|---------|-----------|----------|-----------|----------|----------|---------|----------|---------|------------|-----------|
| base      | 24   | −0.367  | −0.069    | −0.068   | 0.187     | 0.209    | 0.0348   | 0.0438  | 0.80     | 0.80    | 0.65       | 0.73      |
| clone     | 121  | −0.354  | −0.087    | −0.092   | 0.193     | 0.201    | 0.0374   | 0.0405  | 0.80     | 0.80    | 0.65       | 0.70      |
| premium   | 121  | −0.264  | −0.124    | −0.132   | 0.215     | 0.226    | 0.0463   | 0.0510  | 0.80     | 0.80    | 0.61       | 0.67      |
| parallel  | 121  | −0.479  | −0.084    | −0.085   | 0.196     | 0.207    | 0.0384   | 0.0429  | 0.90     | 0.90    | 0.70       | 0.76      |
| unhelpful | 121  | −0.492  | −0.204    | −0.205   | 0.293     | 0.300    | 0.0857   | 0.0897  | 0.80     | 0.80    | 0.72       | 0.79      |

α = 0.20:
| policy    | n_au | truth   | bias_plug | bias_aug | RMSE_plug | RMSE_aug | MSE_plug | MSE_aug | cov_plug | cov_aug | width_plug | width_aug |
|-----------|------|---------|-----------|----------|-----------|----------|----------|---------|----------|---------|------------|-----------|
| base      | 24   | −0.197  | −0.036    | −0.038   | 0.100     | 0.128    | 0.0100   | 0.0165  | 0.80     | 0.80    | 0.38       | 0.48      |
| clone     | 121  | −0.185  | −0.053    | −0.060   | 0.106     | 0.115    | 0.0113   | 0.0133  | 0.80     | 0.90    | 0.38       | 0.46      |
| premium   | 121  | −0.101  | −0.093    | −0.106   | 0.133     | 0.148    | 0.0176   | 0.0218  | 0.90     | 0.90    | 0.36       | 0.43      |
| parallel  | 121  | −0.322  | −0.030    | −0.029   | 0.114     | 0.127    | 0.0129   | 0.0161  | 0.80     | 0.80    | 0.46       | 0.55      |
| unhelpful | 121  | −0.363  | −0.169    | −0.189   | 0.246     | 0.284    | 0.0604   | 0.0808  | 0.90     | 0.90    | 0.56       | 0.65      |

**Verdict on inspector hypothesis S2.** Directionally correct, but the effect
is *much wider than predicted*: plug-in wins MSE in **10/10 cells**, not
just at small `n_audit`. The MSE gap at base (n_au=24) is 21–39%; at
non-base policies (n_au≈121) it is 5–25%. The augmented estimator is
strictly worse on this panel.

**Mechanism — what I'm seeing.** The augmented one-step adjustment is
designed to remove asymptotic calibrator bias. On this panel:
- Both estimators have similar bias `(bias_plug − bias_aug)` ranges from
  −0.001 to +0.013, all <8% of the bias magnitude itself.
- Both are systematically downward-biased (`bias < 0`) by 0.03–0.20 across
  cells. The downward bias is the calibrator under-shooting on the lower
  tail — adding `ḡ_2` doesn't remove it, because `ḡ_2` itself averages
  near zero (transport residuals are roughly mean-zero on the audit
  slice).
- `Var(V_aug) > Var(V_plug)` strictly, because aug adds the audit-side
  resampling variance.
- Net: aug pays the variance cost without recovering bias. MSE = bias² +
  variance — the bias² term is unchanged, and aug's variance is uniformly
  larger.

This is *not* the textbook one-step debiasing story (where bias does
shrink). It seems specific to the calibrator structure here: the isotonic
stop-loss `ĝ_t̂` is fit on a slice that's representative enough that the
audit residual mean isn't very informative — `ḡ_2` is close to zero on
average and contributes mostly noise.

**Coverage.** Both plug-in and augmented coverage at 10 seeds is 0.80–0.90
(point estimates with ~13 percentage-point SE), so neither hits 95%
nominal cleanly. Aug coverage is consistently *closer* to 0.95 because the
wider CI catches the truth more often, but at 10 seeds we cannot
distinguish 0.85 from 0.90 (or from 0.95) with statistical confidence. The
shipped formula's slight conservatism (§4.1 E4) appears to help cover the
truth here — but on the wrong centre, since the centre is plug-in but the
formula is augmented-flavoured.

**Caveats on the multi-seed sweep.**
- 10 seeds × 5 policies × 2 alphas = 100 cells. MSE estimates have ~`√(2/9)
  ≈ 47%` relative SE per cell, so the small-magnitude MSE gaps (e.g.
  unhelpful α=0.10 plug 0.0857 vs aug 0.0897, +5%) are inside MC noise.
  The large gaps (base α=0.20 plug 0.0100 vs aug 0.0165, +65%) are very
  likely real.
- The verdict "plug wins MSE everywhere" depends on the bias mechanism
  above. If the calibrator stack changes (e.g. switch to gpt-5.4 oracle, or
  apply two-stage calibration with response_length covariate per
  SPEC_SHEET), the bias geometry could change and aug could win.
- 50 seeds (the full pre-registered sweep) would tighten coverage to ~7
  percentage points and MSE to ~20% relative SE. Strongly recommend running
  it before shipping a paper claim.

**Implication for the paper.** The shipped CI formula (`Var_cal_J +
Var_audit_an` around the plug-in centre) accidentally produces a near-aug
*width* around the better *centre*. On this panel, that's not a bug but a
useful artifact: you get plug-in's better MSE with a CI that's
conservative-enough to cover. Whether to formalize this as "plug-in centre
+ augmented-style CI" or to just report the plug-in pipeline-bootstrap CI
(narrower, but might under-cover if the multi-seed sweep showed plug-in
coverage <0.90) is a paper-writing question, not a code question.

**Action items remaining.**
1. Run the full 50-seed sweep (~25 min wall at B=500). Will tighten the
   coverage and MSE estimates by 2× and resolve whether plug-in coverage
   actually under-covers at 95%.
2. Investigate why aug doesn't reduce bias — is it the isotonic stop-loss
   structure specifically, or a panel-size issue?

### 8.4 Jackknife-vs-bootstrap framing tightened (O2)

**Inspector clarification.** I previously characterized a 5–16% disagreement
between K=5 jackknife `Var_cal` and B=500 bootstrap `Var_cal` as a "valid
methodology gap". The inspector pointed out I was understating it: a K=5
jackknife variance estimator has K−1=4 degrees of freedom, so its relative
SE is `√(2/(K−1)) ≈ 70%`. A 15% agreement on a single seed is partly luck;
under repeated draws of the slice, the K=5 jackknife will oscillate widely
around the bootstrap.

**Implication.** The shipped formula `Var_cal_J + Var_audit_an` carries
substantial estimation noise on the `Var_cal_J` term alone. The fact that
the analytical sum tracks the augmented bootstrap variance to ±5–15% on
seed=42 (§4.1 E4) should therefore be read as "the formula is approximately
right *on this seed*"; the §8.3 multi-seed results showed coverage 0.80–0.90
across 10 seeds, consistent with the formula being a usable conservative CI
on this panel — but at 10 seeds we cannot distinguish 0.85 from 0.95
statistically.

### 8.5 Rollback assessment

The user asked to roll back if the fixes don't work. They worked:

1. **Coupled-RNG refactor** — additivity gap dropped from 11–15% to 1–3.5%.
   No rollback. Backward-compat preserved (existing `bootstrap_cvar_ci`
   regression test still passes at max diff 0.00e+00).
2. **Grid-snap diagnostic** — surfaced uniqueness count; clean at 380–451 /
   500. Cheap to keep.
3. **Multi-seed sweep** — produced a clean answer to the inspector's S2
   hypothesis (plug-in wins MSE everywhere on this panel). New analyses
   module; opt-in via CLI; doesn't affect any other path.

**Net code changes:**
- `cvar_v4/eda/deeper/_estimator.py`: `pipeline_bootstrap_cvar` gained 3
  optional index params and now returns `t_hat_boots` and `n_unique_t_hat`.
  Backward-compatible: callers that don't pass indices get the original
  RNG-driven path.
- `cvar_v4/healthbench_data/analyses/variance_breakdown.py`: added
  `_master_indices`, threaded through `_branch`, modified the loop to
  pre-generate and pass coupled indices. Markdown adds a `uniq t̂` column.
- `cvar_v4/healthbench_data/analyses/variance_sweep.py`: new file (multi-
  seed orchestrator).

Nothing in the production estimator path or `step5_oracle_calibrated_uniform`
changed. Existing diagnostics that don't import variance_breakdown or
variance_sweep are unaffected.

