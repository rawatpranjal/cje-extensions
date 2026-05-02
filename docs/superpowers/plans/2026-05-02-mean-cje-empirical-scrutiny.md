# mean_cje Empirical Scrutiny — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a swappable Mean-CJE implementation in `mean_cje/`, run an ablation on the parametric Beta DGP, and produce a CSV + README answering "which pieces of the paper's machinery are needed for unbiased estimates and 90–95% coverage."

**Architecture:** Single-policy DGP reused from `cvar_v5/mc/dgp.py`. Six configurations (plug-in vs AIPW × IF-only vs IF+jackknife × Wald vs bootstrap) run on a 4-policy × 5-oracle-size × R-rep grid. Each library function is independently swappable.

**Tech Stack:** Python 3.11, numpy, scikit-learn IsotonicRegression, polars (DGP rows), scipy.stats (no scipy needed for the ablation itself), multiprocessing for the experiment.

---

## File Structure

```
mean_cje/
├── CLAUDE.md                  scope rules
├── __init__.py                empty
├── lib/
│   ├── __init__.py            empty
│   ├── calibrator.py          Calibrator class: fit, predict, predict_oof
│   ├── estimators.py          plug_in_mean, aipw_one_step
│   └── variance.py            var_eval_if, var_cal_jackknife, wald_ci,
│                              bootstrap_ci_aipw, bootstrap_ci_plugin
├── lib_test.py                code tests for the library
├── exp_coverage.py            CLI ablation runner
├── runs/                      output (untracked)
└── README.md                  findings (filled in after run)
```

**`Calibrator` interface:**
```
Calibrator.fit(s_oracle, y_oracle, K=5)  → trains pooled fit + K fold-out fits
Calibrator.predict(s)                    → array of f̂(s_i) (uses pooled fit)
Calibrator.predict_oof(s_oracle)         → array where row i uses f̂^(-fold(i))
                                           (only valid for the rows it was fit on)
```

**Bootstrap drivers** are estimator-specific, so they live in `variance.py` rather than as one generic loop. The plug-in bootstrap refits the calibrator and recomputes `mean f̂(s_eval)`. The AIPW bootstrap also recomputes the residual term using cross-fit predictions on the bootstrap oracle.

---

## Task 1: Folder skeleton + CLAUDE.md

**Files:**
- Create: `mean_cje/__init__.py`
- Create: `mean_cje/lib/__init__.py`
- Create: `mean_cje/CLAUDE.md`

- [ ] **Step 1: Create empty `__init__.py` files**

```bash
mkdir -p mean_cje/lib mean_cje/runs
touch mean_cje/__init__.py mean_cje/lib/__init__.py
```

- [ ] **Step 2: Create `mean_cje/CLAUDE.md` with scope rules**

```markdown
# mean_cje

Empirical scrutiny of the original CJE paper's Mean-CJE machinery on a
known-truth DGP. Goal: figure out which pieces of the paper's stack are
needed for unbiased estimates and 90–95% coverage.

**Spec:** `docs/superpowers/specs/2026-05-02-mean-cje-empirical-scrutiny-design.md`

## Hard scope rule

Nothing in `mean_cje/` lives outside the spec. If a file isn't named in
the spec, it doesn't belong here.

## What this folder is NOT

- Not a production estimator. Production CVaR-CJE lives in `cvar_v5/`.
- Not an audit-size study. That's a separate question.
- Not a real-data adapter.

## Components

- `lib/calibrator.py`  isotonic on S, with cross-fit
- `lib/estimators.py`  plug_in_mean, aipw_one_step
- `lib/variance.py`    var_eval_if, var_cal_jackknife, wald_ci, bootstrap variants
- `lib_test.py`        code tests
- `exp_coverage.py`    CLI ablation runner

## DGP

Imported from `cvar_v5.mc.dgp.DGP` and `DEFAULT_POLICIES`. Don't duplicate.

## Tests

`pytest mean_cje/lib_test.py`. Code-only tests; no statistical tests at
the unit level — statistical claims are validated via `exp_coverage.py`
end-to-end.
```

- [ ] **Step 3: Commit**

```bash
git add mean_cje/__init__.py mean_cje/lib/__init__.py mean_cje/CLAUDE.md
git commit -m "Add mean_cje skeleton + scope CLAUDE.md"
```

---

## Task 2: Calibrator with cross-fit

**Files:**
- Create: `mean_cje/lib/calibrator.py`
- Test: `mean_cje/lib_test.py` (created)

- [ ] **Step 1: Write the failing test (file creation)**

```python
# mean_cje/lib_test.py
"""Code tests for mean_cje.lib. No statistical tests at this level."""

from __future__ import annotations

import numpy as np
import pytest

from mean_cje.lib.calibrator import Calibrator


def _toy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = 1.0 / (1.0 + np.exp(-s)) + 0.05 * rng.normal(size=n)
    y = np.clip(y, 0.0, 1.0)
    return s, y


def test_calibrator_predict_is_monotone_increasing() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    s_query = np.linspace(s.min(), s.max(), 50)
    pred = cal.predict(s_query)
    assert (np.diff(pred) >= -1e-12).all()


def test_calibrator_pooled_mean_equals_oracle_mean() -> None:
    """PAVA preserves the slice mean exactly."""
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    pooled_pred = cal.predict(s)
    assert abs(pooled_pred.mean() - y.mean()) <= 1e-12


def test_predict_oof_shape_and_finite() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    oof = cal.predict_oof(s)
    assert oof.shape == s.shape
    assert np.isfinite(oof).all()


def test_predict_oof_differs_from_pooled() -> None:
    """OOF predictions should differ from pooled at most rows."""
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    pooled = cal.predict(s)
    oof = cal.predict_oof(s)
    assert (np.abs(pooled - oof) > 0).mean() > 0.5
```

- [ ] **Step 2: Run tests, expect import failure**

```bash
cd /Users/pranjal/Code/cvar-cje && python3 -m pytest mean_cje/lib_test.py -v
```

Expected: ImportError or ModuleNotFoundError on `Calibrator`.

- [ ] **Step 3: Implement `Calibrator`**

```python
# mean_cje/lib/calibrator.py
"""
Isotonic calibrator with K-fold cross-fit.

Math contract:
    f̂      = IsotonicRegression(increasing=True).fit(s_oracle, y_oracle)
    f̂^(−k) = same, fit on oracle rows where fold_id ≠ k

The pooled fit is used for predict() (used by EVAL averages).
The K fold-out fits are used for predict_oof() (used in the AIPW residual
term and in the jackknife).

PAVA preserves the training-slice mean exactly:
    mean_i  f̂(s_oracle_i)  ==  mean_i  y_oracle_i.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self) -> None:
        self._pooled: IsotonicRegression | None = None
        self._folded: list[IsotonicRegression] = []
        self._fold_id: np.ndarray | None = None
        self._s_train: np.ndarray | None = None
        self._K: int | None = None

    def fit(
        self,
        s_oracle: np.ndarray,
        y_oracle: np.ndarray,
        K: int = 5,
        seed: int = 0,
    ) -> "Calibrator":
        s = np.asarray(s_oracle, dtype=np.float64).ravel()
        y = np.asarray(y_oracle, dtype=np.float64).ravel()
        if len(s) != len(y):
            raise ValueError("s and y must have the same length")
        if K < 2:
            raise ValueError(f"K must be >= 2; got {K}")

        rng = np.random.default_rng(seed)
        fold_id = rng.integers(0, K, size=len(s))

        pooled = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s, y)
        folded: list[IsotonicRegression] = []
        for k in range(K):
            train = fold_id != k
            ir_k = IsotonicRegression(increasing=True, out_of_bounds="clip")
            ir_k.fit(s[train], y[train])
            folded.append(ir_k)

        self._pooled = pooled
        self._folded = folded
        self._fold_id = fold_id
        self._s_train = s
        self._K = K
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        if self._pooled is None:
            raise RuntimeError("Calibrator not fit")
        return self._pooled.predict(np.asarray(s, dtype=np.float64).ravel())

    def predict_oof(self, s_oracle: np.ndarray) -> np.ndarray:
        """For each row i with fold_id[i] = k, predict using f̂^(−k)."""
        if not self._folded:
            raise RuntimeError("Calibrator not fit")
        s = np.asarray(s_oracle, dtype=np.float64).ravel()
        if self._s_train is None or len(s) != len(self._s_train):
            raise ValueError("predict_oof requires the same s used in fit()")
        out = np.empty_like(s)
        for k in range(self._K):
            mask = self._fold_id == k
            if mask.any():
                out[mask] = self._folded[k].predict(s[mask])
        return out

    def refit_excluding_fold(
        self, s_oracle: np.ndarray, y_oracle: np.ndarray, k: int,
    ) -> IsotonicRegression:
        """For the V̂_cal jackknife: refit on oracle \\ fold_k. Returns a fresh fit."""
        if self._fold_id is None:
            raise RuntimeError("Calibrator not fit")
        s = np.asarray(s_oracle, dtype=np.float64).ravel()
        y = np.asarray(y_oracle, dtype=np.float64).ravel()
        train = self._fold_id != k
        return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s[train], y[train])

    @property
    def K(self) -> int:
        if self._K is None:
            raise RuntimeError("Calibrator not fit")
        return self._K

    @property
    def fold_id(self) -> np.ndarray:
        if self._fold_id is None:
            raise RuntimeError("Calibrator not fit")
        return self._fold_id
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
python3 -m pytest mean_cje/lib_test.py -v
```

All four tests should pass.

- [ ] **Step 5: Commit**

```bash
git add mean_cje/lib/calibrator.py mean_cje/lib_test.py
git commit -m "Add Calibrator: isotonic + K-fold cross-fit"
```

---

## Task 3: Estimators (plug-in + AIPW one-step)

**Files:**
- Create: `mean_cje/lib/estimators.py`
- Modify: `mean_cje/lib_test.py` (add tests)

- [ ] **Step 1: Add failing tests**

```python
# Append to mean_cje/lib_test.py

from mean_cje.lib.estimators import plug_in_mean, aipw_one_step


def test_plug_in_mean_equals_eval_average_of_calibrator() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    s_eval = np.random.default_rng(1).normal(size=300)
    plug = plug_in_mean(s_eval, cal)
    assert abs(plug - cal.predict(s_eval).mean()) <= 1e-12


def test_aipw_residual_term_zero_when_oof_equals_y() -> None:
    """If OOF predictions exactly equal y, AIPW reduces to plug-in."""
    s, y = _toy(n=100)
    cal = Calibrator().fit(s, y, K=5)
    s_eval = np.random.default_rng(1).normal(size=200)

    # Build a fake calibrator whose predict_oof returns y exactly.
    class _ExactOof(Calibrator):
        def predict_oof(self, s_oracle):
            return y.copy()

    fake = _ExactOof().fit(s, y, K=5)
    plug = plug_in_mean(s_eval, cal)
    aipw = aipw_one_step(s_eval, s, y, fake, calibrator_pooled=cal)
    # residual term mean(y - y) = 0 ⇒ aipw == plug
    assert abs(aipw - plug) <= 1e-12


def test_aipw_returns_finite() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    s_eval = np.random.default_rng(2).normal(size=300)
    val = aipw_one_step(s_eval, s, y, cal)
    assert np.isfinite(val)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
python3 -m pytest mean_cje/lib_test.py -v
```

- [ ] **Step 3: Implement estimators**

```python
# mean_cje/lib/estimators.py
"""
Two estimators for E[Y]:

    plug_in_mean(s_eval, cal):
        V̂  =  (1/n_eval) · Σ  f̂(s_eval_i)

    aipw_one_step(s_eval, s_oracle, y_oracle, cal):
        θ̂_aug  =  (1/n_eval) · Σ  f̂(s_eval_i)
                + (1/|L|) · Σ_{j ∈ L}  ( y_oracle_j − f̂^(−j)(s_oracle_j) )
                                         ↑ cross-fit prediction at row j
"""

from __future__ import annotations

import numpy as np

from .calibrator import Calibrator


def plug_in_mean(s_eval: np.ndarray, calibrator: Calibrator) -> float:
    return float(calibrator.predict(s_eval).mean())


def aipw_one_step(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    calibrator: Calibrator,
    *,
    calibrator_pooled: Calibrator | None = None,
) -> float:
    """
    AIPW one-step. `calibrator` provides predict_oof on the oracle slice;
    `calibrator_pooled` (defaults to `calibrator`) provides the pooled fit
    used for the EVAL term.
    """
    pooled = calibrator_pooled or calibrator
    plug = pooled.predict(s_eval).mean()
    oof_residual = (y_oracle - calibrator.predict_oof(s_oracle)).mean()
    return float(plug + oof_residual)
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
python3 -m pytest mean_cje/lib_test.py -v
```

- [ ] **Step 5: Commit**

```bash
git add mean_cje/lib/estimators.py mean_cje/lib_test.py
git commit -m "Add plug_in_mean and aipw_one_step estimators"
```

---

## Task 4: Variance components (IF + jackknife) + Wald CI

**Files:**
- Create: `mean_cje/lib/variance.py`
- Modify: `mean_cje/lib_test.py`

- [ ] **Step 1: Add failing tests**

```python
# Append to mean_cje/lib_test.py

from mean_cje.lib.variance import (
    var_eval_if, var_cal_jackknife, wald_ci,
)


def test_var_eval_if_positive_and_scales_with_n_inv() -> None:
    s_eval = np.random.default_rng(0).normal(size=500)
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    v = var_eval_if(s_eval, cal)
    assert v > 0
    # Doubling n should roughly halve V̂_main.
    s_big = np.random.default_rng(0).normal(size=1000)
    v_big = var_eval_if(s_big, cal)
    assert v_big < v


def test_var_cal_jackknife_is_nonnegative() -> None:
    s, y = _toy()
    cal = Calibrator().fit(s, y, K=5)
    s_eval = np.random.default_rng(0).normal(size=500)
    v_cal = var_cal_jackknife(s_eval, s, y, cal, estimator="plug_in")
    assert v_cal >= 0


def test_wald_ci_widens_with_variance() -> None:
    lo1, hi1 = wald_ci(0.5, 0.001)
    lo2, hi2 = wald_ci(0.5, 0.01)
    assert (hi2 - lo2) > (hi1 - lo1)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
python3 -m pytest mean_cje/lib_test.py -v
```

- [ ] **Step 3: Implement variance and CI**

```python
# mean_cje/lib/variance.py
"""
Variance components and CIs.

Math contracts:

V̂_eval (IF):
    φ_i  =  f̂(s_eval_i) − V̂
    V̂_eval  =  (1/n_eval²) · Σ_i  φ_i²

V̂_cal (jackknife, paper appendix_notation:140):
    For k = 1..K:
        f̂^(−k) := refit calibrator omitting fold k of the oracle
        V̂^(−k) := estimator(eval, f̂^(−k))   (matches the chosen estimator)
    V̄  := mean_k V̂^(−k)
    V̂_cal := ((K−1)/K) · Σ_k (V̂^(−k) − V̄)²

Wald CI:
    V̂  ±  1.96 · √V̂_total      with V̂_total = V̂_eval + V̂_cal

Bootstrap CIs are estimator-specific; live in this file as
bootstrap_ci_plugin and bootstrap_ci_aipw.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .calibrator import Calibrator
from .estimators import plug_in_mean, aipw_one_step


EstimatorName = Literal["plug_in", "aipw"]


def var_eval_if(s_eval: np.ndarray, calibrator: Calibrator) -> float:
    """V̂_eval via per-row centered IF: (1/n²) Σ φ_i² with φ_i = f̂(s_i) − V̂."""
    pred = calibrator.predict(s_eval)
    v_hat = pred.mean()
    phi = pred - v_hat
    n = len(s_eval)
    return float((phi ** 2).sum() / (n * n))


def _eval_estimator(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    cal_full: Calibrator,
    estimator: EstimatorName,
) -> float:
    if estimator == "plug_in":
        return plug_in_mean(s_eval, cal_full)
    elif estimator == "aipw":
        return aipw_one_step(s_eval, s_oracle, y_oracle, cal_full)
    else:
        raise ValueError(f"unknown estimator {estimator!r}")


def var_cal_jackknife(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    calibrator: Calibrator,
    *,
    estimator: EstimatorName,
) -> float:
    """
    Delete-one-oracle-fold jackknife on the chosen estimator.

    For each k, we use the same refit_excluding_fold mechanism the calibrator
    already supports. Each refit is then wrapped in a tiny shim Calibrator
    so the estimator sees a well-formed object.
    """
    K = calibrator.K
    fold_id = calibrator.fold_id
    estimates: list[float] = []
    for k in range(K):
        ir_k = calibrator.refit_excluding_fold(s_oracle, y_oracle, k)

        class _ShimCal(Calibrator):
            def __init__(self, base: Calibrator, ir_pooled) -> None:
                super().__init__()
                self._pooled = ir_pooled
                self._folded = base._folded
                self._fold_id = base._fold_id
                self._s_train = base._s_train
                self._K = base._K

        shim = _ShimCal(calibrator, ir_k)
        estimates.append(_eval_estimator(
            s_eval, s_oracle, y_oracle, shim, estimator,
        ))
    arr = np.asarray(estimates)
    v_bar = arr.mean()
    return float((K - 1) / K * ((arr - v_bar) ** 2).sum())


def wald_ci(v_hat: float, v_total: float, level: float = 0.95) -> tuple[float, float]:
    """V̂ ± z · √V̂_total."""
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level}")
    if v_total < 0:
        raise ValueError(f"v_total must be ≥ 0; got {v_total}")
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(level, 2))
    if z is None:
        from scipy import stats
        z = float(stats.norm.ppf(1 - (1 - level) / 2))
    half = z * np.sqrt(v_total)
    return float(v_hat - half), float(v_hat + half)


def bootstrap_ci_plugin(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    *,
    B: int,
    K: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """
    Cluster bootstrap on the plug-in V̂. Each rep:
        - resample oracle rows with replacement (size n_oracle)
        - refit calibrator on bootstrap oracle
        - V̂^(b) = (1/n_eval) Σ f̂^(b)(s_eval_i)        (eval slice fixed)

    Returns percentile interval of the {V̂^(b)}.
    """
    rng = np.random.default_rng(seed)
    n = len(s_oracle)
    samples = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        s_b = s_oracle[idx]
        y_b = y_oracle[idx]
        cal_b = Calibrator().fit(s_b, y_b, K=K, seed=seed + b)
        samples[b] = plug_in_mean(s_eval, cal_b)
    lo = float(np.percentile(samples, 100 * (1 - level) / 2))
    hi = float(np.percentile(samples, 100 * (1 - (1 - level) / 2)))
    return lo, hi


def bootstrap_ci_aipw(
    s_eval: np.ndarray,
    s_oracle: np.ndarray,
    y_oracle: np.ndarray,
    *,
    B: int,
    K: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """As above, but compute θ̂_aug per replicate."""
    rng = np.random.default_rng(seed)
    n = len(s_oracle)
    samples = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        s_b = s_oracle[idx]
        y_b = y_oracle[idx]
        cal_b = Calibrator().fit(s_b, y_b, K=K, seed=seed + b)
        samples[b] = aipw_one_step(s_eval, s_b, y_b, cal_b)
    lo = float(np.percentile(samples, 100 * (1 - level) / 2))
    hi = float(np.percentile(samples, 100 * (1 - (1 - level) / 2)))
    return lo, hi
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
python3 -m pytest mean_cje/lib_test.py -v
```

- [ ] **Step 5: Commit**

```bash
git add mean_cje/lib/variance.py mean_cje/lib_test.py
git commit -m "Add IF variance, jackknife V_cal, Wald + bootstrap CIs"
```

---

## Task 5: Experiment runner + smoke test

**Files:**
- Create: `mean_cje/exp_coverage.py`
- Modify: `mean_cje/lib_test.py` (smoke import test)

- [ ] **Step 1: Add a smoke import test**

```python
# Append to mean_cje/lib_test.py

def test_exp_coverage_module_imports() -> None:
    import mean_cje.exp_coverage  # noqa: F401
```

- [ ] **Step 2: Run, expect import failure**

```bash
python3 -m pytest mean_cje/lib_test.py::test_exp_coverage_module_imports -v
```

- [ ] **Step 3: Implement `exp_coverage.py`**

```python
# mean_cje/exp_coverage.py
"""
Ablation runner. Six configurations × 4 policies × 5 oracle sizes.
Outputs to mean_cje/runs/<ts>/results.csv with one row per cell.

Configurations (see spec):
    1 plug-in   + IF only         + Wald
    2 plug-in   + IF + jackknife  + Wald
    3 AIPW      + IF only         + Wald
    4 AIPW      + IF + jackknife  + Wald
    5 plug-in   + bootstrap
    6 AIPW      + bootstrap

For each cell: bias, RMSE, coverage at 95%, mean CI half-width.

CLI:
    python -m mean_cje.exp_coverage -w 4
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from cvar_v5.mc.dgp import DEFAULT_POLICIES, DGP
from .lib.calibrator import Calibrator
from .lib.estimators import plug_in_mean, aipw_one_step
from .lib.variance import (
    var_eval_if, var_cal_jackknife, wald_ci,
    bootstrap_ci_plugin, bootstrap_ci_aipw,
)


LOG = logging.getLogger("mean_cje")


# Sweep parameters
_POLICIES = list(DEFAULT_POLICIES.keys())
_N_ORACLES = (50, 100, 250, 500, 1000)
_N_EVAL = 2000
_R_WALD = 200
_R_BOOT = 50
_B = 500
_K = 5
_LEVEL = 0.95


CONFIGS = [
    {"id": 1, "estimator": "plug_in", "variance": "if",      "ci": "wald"},
    {"id": 2, "estimator": "plug_in", "variance": "if+jack", "ci": "wald"},
    {"id": 3, "estimator": "aipw",    "variance": "if",      "ci": "wald"},
    {"id": 4, "estimator": "aipw",    "variance": "if+jack", "ci": "wald"},
    {"id": 5, "estimator": "plug_in", "variance": "boot",    "ci": "bootstrap"},
    {"id": 6, "estimator": "aipw",    "variance": "boot",    "ci": "bootstrap"},
]


def _make_run_dir() -> Path:
    base = Path(__file__).parent / "runs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out = base / ts
    n = 2
    while out.exists():
        out = base / f"{ts}_{n}"
        n += 1
    out.mkdir()
    return out


def _setup_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "log.txt", mode="w"),
        ],
    )


def _one_rep(config: dict, policy: str, n_oracle: int, rep: int) -> dict:
    """One (config, policy, n_oracle, rep) cell. Returns one row."""
    dgp = DGP(DEFAULT_POLICIES)
    seed = 7919 * rep + 31
    oracle = dgp.sample(policy, n=n_oracle, with_oracle=True, seed=seed)
    eval_df = dgp.sample(policy, n=_N_EVAL, with_oracle=False, seed=seed + 1)
    s_oracle = oracle["s"].to_numpy()
    y_oracle = oracle["y"].to_numpy()
    s_eval = eval_df["s"].to_numpy()

    cal = Calibrator().fit(s_oracle, y_oracle, K=_K, seed=seed)

    # Point estimate
    if config["estimator"] == "plug_in":
        v_hat = plug_in_mean(s_eval, cal)
    else:
        v_hat = aipw_one_step(s_eval, s_oracle, y_oracle, cal)

    # CI
    if config["ci"] == "wald":
        v_eval = var_eval_if(s_eval, cal)
        v_cal = (
            var_cal_jackknife(s_eval, s_oracle, y_oracle, cal,
                              estimator=config["estimator"])
            if config["variance"] == "if+jack" else 0.0
        )
        lo, hi = wald_ci(v_hat, v_eval + v_cal, level=_LEVEL)
    else:
        if config["estimator"] == "plug_in":
            lo, hi = bootstrap_ci_plugin(s_eval, s_oracle, y_oracle,
                                         B=_B, K=_K, seed=seed, level=_LEVEL)
        else:
            lo, hi = bootstrap_ci_aipw(s_eval, s_oracle, y_oracle,
                                       B=_B, K=_K, seed=seed, level=_LEVEL)

    truth = dgp.truth_mean(policy)
    return {
        "config_id": config["id"],
        "estimator": config["estimator"],
        "variance": config["variance"],
        "ci": config["ci"],
        "policy": policy,
        "n_oracle": n_oracle,
        "rep": rep,
        "v_hat": float(v_hat),
        "truth": float(truth),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def _cells_to_run() -> list[tuple]:
    cells = []
    for config in CONFIGS:
        R = _R_BOOT if config["ci"] == "bootstrap" else _R_WALD
        for policy, n_oracle in product(_POLICIES, _N_ORACLES):
            for rep in range(R):
                cells.append((config, policy, n_oracle, rep))
    return cells


def _run_cell(args: tuple) -> dict:
    return _one_rep(*args)


def _aggregate(rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.group_by(["config_id", "estimator", "variance", "ci", "policy", "n_oracle"])
        .agg([
            (pl.col("v_hat") - pl.col("truth")).mean().alias("bias"),
            ((pl.col("v_hat") - pl.col("truth")) ** 2).mean().sqrt().alias("rmse"),
            ((pl.col("ci_lo") <= pl.col("truth")) & (pl.col("truth") <= pl.col("ci_hi")))
                .mean().alias("coverage"),
            ((pl.col("ci_hi") - pl.col("ci_lo")) / 2).mean().alias("mean_half_width"),
            pl.len().alias("R"),
        ])
        .sort(["config_id", "policy", "n_oracle"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="mean_cje coverage ablation")
    parser.add_argument("-w", "--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or _make_run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)
    LOG.info("mean_cje coverage ablation; out_dir=%s, workers=%d",
             out_dir, args.n_workers)

    cells = _cells_to_run()
    LOG.info("total cells: %d (across 6 configs × 4 policies × 5 oracle sizes)",
             len(cells))

    t0 = time.time()
    rows: list[dict] = []
    if args.n_workers <= 1:
        for i, c in enumerate(cells):
            rows.append(_run_cell(c))
            if (i + 1) % max(1, len(cells) // 20) == 0:
                LOG.info("  %d/%d (%.1f%%, %.1fs)", i+1, len(cells),
                         100*(i+1)/len(cells), time.time()-t0)
    else:
        ctx = get_context("fork")
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_cell, cells, chunksize=4)):
                rows.append(row)
                if (i + 1) % max(1, len(cells) // 20) == 0:
                    LOG.info("  %d/%d (%.1f%%, %.1fs)", i+1, len(cells),
                             100*(i+1)/len(cells), time.time()-t0)

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "results.csv")
    summary = _aggregate(df)
    summary.write_csv(out_dir / "summary.csv")
    LOG.info("wrote results.csv and summary.csv in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke import test, expect PASS**

```bash
python3 -m pytest mean_cje/lib_test.py::test_exp_coverage_module_imports -v
```

- [ ] **Step 5: Run a tiny smoke version end-to-end**

Set tiny budgets via inline overrides:

```bash
python3 -c "
import mean_cje.exp_coverage as ec
ec._N_ORACLES = (100,)
ec._N_EVAL = 200
ec._R_WALD = 3
ec._R_BOOT = 2
ec._B = 20
import sys; sys.argv = ['exp_coverage', '-w', '2']
ec.main()
"
```

Expected: completes in < 30 s, writes a `runs/<ts>/results.csv` and `summary.csv`.

- [ ] **Step 6: Inspect summary**

```bash
ls -la mean_cje/runs/ | tail
cat mean_cje/runs/$(ls -1t mean_cje/runs/ | head -1)/summary.csv | head
```

- [ ] **Step 7: Commit**

```bash
git add mean_cje/exp_coverage.py mean_cje/lib_test.py
git commit -m "Add coverage ablation runner with 6 configurations"
```

---

## Task 6: Run the full experiment

**Files:**
- Generated: `mean_cje/runs/<ts>/results.csv` and `summary.csv`

- [ ] **Step 1: Run with full budget**

```bash
cd /Users/pranjal/Code/cvar-cje && python3 -m mean_cje.exp_coverage -w 4
```

Expected wall: 60–90 minutes (per spec). Tail of `log.txt` should show `wrote results.csv and summary.csv`.

- [ ] **Step 2: Inspect summary table**

```bash
RUN_DIR=mean_cje/runs/$(ls -1t mean_cje/runs/ | head -1)
cat $RUN_DIR/summary.csv | column -ts,
```

Look for: coverage column moves through 6 configurations, hopefully tracking the 0–50% / 70–89% / 95% pattern from the paper.

- [ ] **Step 3: Commit run config marker** (NOT the CSVs — they're untracked)

```bash
echo "Latest run: $(basename $RUN_DIR)" > mean_cje/LATEST_RUN.txt
# Just record the run name as a marker, don't track gigabytes of CSV.
```

(Or skip — runs/ is untracked anyway. No commit needed if there's nothing to add.)

---

## Task 7: Findings — write `mean_cje/README.md`

**Files:**
- Create: `mean_cje/README.md`

- [ ] **Step 1: Write README with empirical findings**

After Task 6 produces the summary, fill the README with the actual numbers. Template:

```markdown
# mean_cje — empirical scrutiny of the original CJE paper

## What this folder is

Read the spec: `docs/superpowers/specs/2026-05-02-mean-cje-empirical-scrutiny-design.md`.

## TL;DR

Six configurations of the Mean-CJE inference machine, run on the
parametric Beta DGP at oracle sizes ∈ {50, 100, 250, 500, 1000}, R=200
(Wald) / R=50 (bootstrap), 95% CI target.

Results (averaged over 4 policies):

| # | configuration | n_oracle=50 | n_oracle=250 | n_oracle=1000 |
|---|---|---|---|---|
| 1 | plug-in + IF only + Wald                       | <%coverage%> | <%coverage%> | <%coverage%> |
| 2 | plug-in + IF + jackknife + Wald                | <%coverage%> | <%coverage%> | <%coverage%> |
| 3 | AIPW + IF only + Wald                          | <%coverage%> | <%coverage%> | <%coverage%> |
| 4 | AIPW + IF + jackknife + Wald                   | <%coverage%> | <%coverage%> | <%coverage%> |
| 5 | plug-in + bootstrap (paper's "naive bootstrap") | <%coverage%> | <%coverage%> | <%coverage%> |
| 6 | AIPW + bootstrap (paper's gold standard)       | <%coverage%> | <%coverage%> | <%coverage%> |

(Fill in from `runs/<ts>/summary.csv`.)

Bias (averaged over 4 policies):

| # | configuration | n_oracle=50 | n_oracle=1000 |
|---|---|---|---|
| 1 | plug-in     | <%bias%> | <%bias%> |
| 3 | AIPW        | <%bias%> | <%bias%> |

## Findings

(Fill in after looking at the table.)

1. **Bias removal**: <which estimator removes the bias at small n_oracle>
2. **Coverage 90–95%**: <which configurations hit the target>
3. **What's NOT essential**: <pieces that didn't move the needle>

## Implications for cvar_v5

(Translate findings to the CVaR audit problem.)

## Running

```bash
python -m mean_cje.exp_coverage -w 4
# results in mean_cje/runs/<ts>/{results.csv, summary.csv, log.txt}
```
```

- [ ] **Step 2: Commit**

```bash
git add mean_cje/README.md
git commit -m "Add mean_cje README with empirical findings"
```

---

## Verification (run after all tasks)

1. `pytest mean_cje/lib_test.py -v` → all tests pass.
2. `python -m mean_cje.exp_coverage -w 4` → completes in < 90 min.
3. `mean_cje/runs/<ts>/summary.csv` exists with 6 × 4 × 5 = 120 rows.
4. `mean_cje/README.md` documents the findings.

## Self-Review

**Spec coverage**: All 8 components of the spec's "files to create" table have a task. The 6 configurations are enumerated explicitly in Task 5. The DGP reuse from `cvar_v5/mc/dgp.py` is concrete (no ambiguity).

**Placeholder scan**: `<%coverage%>` and `<%bias%>` markers in Task 7 are intentional fillable templates after the run; they're labelled as such in Step 1 of Task 7.

**Type consistency**: `Calibrator.fit/predict/predict_oof/refit_excluding_fold` signatures match across Tasks 2, 3, 4, 5. `EstimatorName = Literal["plug_in", "aipw"]` used consistently.

**Scope**: Single subsystem. No need to decompose.
