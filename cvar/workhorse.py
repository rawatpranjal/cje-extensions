"""Minimal CVaR-CJE workhorse for the authors' Chatbot Arena data.

Paper: §3 Estimator, §4 Audit, §5 Inference, Algorithm 1.

Mean path: delegates to `cje-eval==0.2.10`'s `CalibratedDirectEstimator`
(identical to `ablations/core/base.py:204-213` of the authors' repo).

CVaR path: grid-search direct estimator with isotonic stop-loss calibrator,
plus cluster bootstrap for CIs and a two-moment Wald transport audit.

Polars for IO, numpy/sklearn for math, cje library at the boundary.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from cje.calibration import calibrate_dataset
from cje.data import load_dataset_from_jsonl
from cje.data.fresh_draws import compute_response_covariates, load_fresh_draws_auto
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.direct_method import CalibratedDirectEstimator

TARGET_POLICIES: tuple[str, ...] = (
    "clone", "premium", "parallel_universe_prompt", "unhelpful",
)
ORACLE_MASK_SEED = 42  # ablations/core/base.py:101


def fit_isotonic_mean(s_train, y_train, s_pred):
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train[order], y_train[order])
    return iso.predict(s_pred)


def fit_isotonic_tail_loss(s_train, z_train, s_pred):
    order = np.argsort(s_train)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(s_train[order], z_train[order])
    return iso.predict(s_pred)


def fit_isotonic_tail_loss_xf(
    s_train: np.ndarray,
    z_train: np.ndarray,
    s_pred: np.ndarray,
    K: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """K-fold cross-fitted isotonic stop-loss predictions.

    Returns (preds_pred, preds_train_xf) where:
      - preds_pred: predictions on `s_pred` averaged across the K fold-models.
      - preds_train_xf: predictions on `s_train` such that each row's value
        was produced by a calibrator that did NOT see that row. These
        residuals are exchangeable with future audit residuals — the right
        ingredient for an audit Σ̂ that captures calibrator-fit variance.

    Used by `two_moment_wald_audit_xf`.
    """
    n = len(s_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // K
    preds_train_xf = np.empty(n)
    preds_pred_per_fold = np.empty((K, len(s_pred)))
    for k in range(K):
        start = k * fold_size
        end = (k + 1) * fold_size if k < K - 1 else n
        heldout = perm[start:end]
        kept = np.concatenate([perm[:start], perm[end:]])
        order = np.argsort(s_train[kept])
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(s_train[kept][order], z_train[kept][order])
        preds_train_xf[heldout] = iso.predict(s_train[heldout])
        preds_pred_per_fold[k] = iso.predict(s_pred)
    return preds_pred_per_fold.mean(axis=0), preds_train_xf


def make_t_grid(y_train, alpha: float, grid_size: int = 61):
    t_lo = float(np.quantile(y_train, max(0.001, alpha / 5.0)) - 0.60)
    t_hi = float(np.quantile(y_train, min(0.60, alpha + 0.45)) + 0.35)
    return np.linspace(t_lo, t_hi, grid_size)


def estimate_direct_cvar_isotonic(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    alpha: float,
    grid_size: int = 61,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Direct CVaR estimator — grid search over stop-loss thresholds.

    Returns (cvar_estimate, t_hat, t_grid, objective).
    """
    t_grid = make_t_grid(y_train, alpha, grid_size)
    objective = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        z_train = np.maximum(t - y_train, 0.0)
        pred_eval = fit_isotonic_tail_loss(s_train, z_train, s_eval)
        objective[i] = float(t - pred_eval.mean() / alpha)
    best = int(np.argmax(objective))
    return float(objective[best]), float(t_grid[best]), t_grid, objective


def estimate_direct_mean_isotonic(s_train, y_train, s_eval) -> float:
    return float(fit_isotonic_mean(s_train, y_train, s_eval).mean())


def two_moment_wald_audit(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    t0: float,
    alpha: float,
    wald_alpha: float = 0.05,
) -> dict:
    """Per-policy transport audit — see `cvar/demo_utils.py:434-459`.

    Tests two moments on the audit set: (i) `1{Y<=t0} - alpha` and
    (ii) `(t0-Y)_+ - m_hat(S)` where `m_hat` is the stop-loss calibrator fit
    on the training slice. Large Wald statistic => calibrator does not
    transport.
    """
    s_train = np.asarray(s_train)
    y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit)
    y_audit = np.asarray(y_audit)

    g1 = (y_audit <= t0).astype(float) - alpha
    z_train = np.maximum(t0 - y_train, 0.0)
    mhat = fit_isotonic_tail_loss(s_train, z_train, s_audit)
    g2 = np.maximum(t0 - y_audit, 0.0) - mhat
    g = np.column_stack([g1, g2])
    gbar = g.mean(axis=0)
    shat = np.cov(g, rowvar=False, ddof=1)
    try:
        sinv = np.linalg.inv(shat)
    except np.linalg.LinAlgError:
        sinv = np.linalg.pinv(shat)
    wald = float(len(s_audit) * gbar.T @ sinv @ gbar)
    p_value = float(1.0 - stats.chi2.cdf(wald, df=2))
    return {
        "t0": float(t0),
        "wald_stat": wald,
        "p_value": p_value,
        "reject": bool(p_value < wald_alpha),
        "mean_g1": float(gbar[0]),
        "mean_g2": float(gbar[1]),
    }


def two_moment_wald_audit_xf(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_audit: np.ndarray,
    y_audit: np.ndarray,
    t0: float,
    alpha: float,
    B: int = 100,
    fold_seed: int = 0,
    wald_alpha: float = 0.05,
) -> dict:
    """Bootstrap-calibrated χ²₂ Wald audit.

    Naive `two_moment_wald_audit`'s Σ̂ omits the variance contributions from
    plug-in (t̂, ĝ) — the asymptotic variance of √n_audit · ḡ involves
    Var_train(integrated bias of ĝ) and Var_audit(t̂-perturbation), neither
    of which a sample covariance on `s_audit` alone can see.

    This version estimates the FULL asymptotic Σ via paired bootstrap of
    the (s_train, y_train) and (s_audit, y_audit) sets, refitting the
    calibrator inside each rep and re-evaluating the moments at the
    fixed t0 (the original t̂). Σ̂_boot then captures all three sources:
    eval-side residual variance, calibrator-fit variance, and audit-set
    sampling variance. The test statistic is

        W = ḡ' Σ̂_boot⁻¹ ḡ ~ χ²₂  under H_0,

    where Σ̂_boot is the bootstrap covariance of (ḡ_1, ḡ_2) per rep — note
    no extra n scaling because the bootstrap distribution of ḡ already
    has variance ≈ Σ_total / n_audit.

    Cost: B isotonic fits + B bootstrap-mean evaluations per audit call.
    Default B=100 keeps the inner audit cost ~1 sec on n_train≈600.
    """
    s_train = np.asarray(s_train)
    y_train = np.asarray(y_train)
    s_audit = np.asarray(s_audit)
    y_audit = np.asarray(y_audit)
    n_train = len(s_train)
    n_audit = len(s_audit)

    # ḡ on full data (for the test statistic numerator).
    z_train_full = np.maximum(t0 - y_train, 0.0)
    pred_audit_full = fit_isotonic_tail_loss(s_train, z_train_full, s_audit)
    g1_full = (y_audit <= t0).astype(float) - alpha
    g2_full = np.maximum(t0 - y_audit, 0.0) - pred_audit_full
    gbar = np.array([g1_full.mean(), g2_full.mean()])

    # Paired bootstrap over (train, audit). Re-maximize t̂_b on the bootstrap
    # data so the bootstrap variance captures t̂'s sampling variance too;
    # without this, Σ̂_boot under-estimates and W still over-rejects.
    rng = np.random.default_rng(fold_seed)
    g_per_boot = np.empty((B, 2))
    grid_size = 61  # match Algorithm 1's estimator grid; was 31 historically
    for b in range(B):
        idx_t = rng.integers(0, n_train, size=n_train)
        idx_a = rng.integers(0, n_audit, size=n_audit)
        # Refit t̂_b on bootstrap data via the dual.
        _, t_b, _, _ = estimate_direct_cvar_isotonic(
            s_train[idx_t], y_train[idx_t], s_audit[idx_a], alpha, grid_size,
        )
        z_b = np.maximum(t_b - y_train[idx_t], 0.0)
        pred_b = fit_isotonic_tail_loss(s_train[idx_t], z_b, s_audit[idx_a])
        g1_b = ((y_audit[idx_a] <= t_b).astype(float) - alpha).mean()
        g2_b = (np.maximum(t_b - y_audit[idx_a], 0.0) - pred_b).mean()
        g_per_boot[b] = [g1_b, g2_b]

    Sigma_boot = np.cov(g_per_boot, rowvar=False, ddof=1)

    try:
        Sigma_inv = np.linalg.inv(Sigma_boot)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_boot)
    wald = float(gbar @ Sigma_inv @ gbar)
    p_value = float(1.0 - stats.chi2.cdf(wald, df=2))
    return {
        "t0": float(t0),
        "wald_stat": wald,
        "p_value": p_value,
        "reject": bool(p_value < wald_alpha),
        "mean_g1": float(gbar[0]),
        "mean_g2": float(gbar[1]),
    }


def cluster_bootstrap_cvar(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_eval: np.ndarray,
    eval_cluster: np.ndarray,
    train_cluster: np.ndarray,
    alpha: float,
    grid_size: int,
    B: int,
    seed: int,
) -> tuple[float, float, int]:
    """Cluster bootstrap for CVaR.

    Resamples unique clusters with replacement on the train (calibration) slice
    and the eval slice **independently**, then refits the grid-search calibrator
    each rep. With one prompt per row in each set (the Arena layout), passing
    `train_cluster = np.arange(...)` reduces this to a row-level i.i.d.
    bootstrap; passing real `prompt_id` clusters generalises naturally if a
    prompt has multiple rows per set. Independent (rather than joint
    prompt-linked) resampling is documented as an honest divergence in the
    appendix's `Real gaps` (iii).

    Returns (ci_lo, ci_hi, n_failures).
    """
    rng = np.random.default_rng(seed)
    uniq_train = np.unique(train_cluster)
    uniq_eval = np.unique(eval_cluster)
    train_idx_by_cluster = {int(c): np.flatnonzero(train_cluster == c) for c in uniq_train}
    eval_idx_by_cluster = {int(c): np.flatnonzero(eval_cluster == c) for c in uniq_eval}

    estimates = np.empty(B)
    n_fail = 0
    for b in range(B):
        sampled_train = rng.choice(uniq_train, size=len(uniq_train), replace=True)
        sampled_eval = rng.choice(uniq_eval, size=len(uniq_eval), replace=True)
        train_rows = np.concatenate([train_idx_by_cluster[int(c)] for c in sampled_train])
        eval_rows = np.concatenate([eval_idx_by_cluster[int(c)] for c in sampled_eval])
        try:
            est, _, _, _ = estimate_direct_cvar_isotonic(
                s_train[train_rows],
                y_train[train_rows],
                s_eval[eval_rows],
                alpha,
                grid_size,
            )
        except Exception:
            est = np.nan
        if np.isfinite(est):
            estimates[b] = est
        else:
            estimates[b] = np.nan
            n_fail += 1

    valid = estimates[np.isfinite(estimates)]
    if len(valid) < max(10, B // 10):
        return float("nan"), float("nan"), n_fail
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(lo), float(hi), n_fail


def load_arena_main(data_root: Path | str) -> pl.DataFrame:
    """Load `data/cje_dataset.jsonl` with flattened metadata. Polars-first."""
    data_root = Path(data_root)
    df = pl.read_ndjson(data_root / "cje_dataset.jsonl")
    return df.with_columns(
        [
            pl.col("metadata").struct.field("prompt_id").alias("prompt_id"),
            pl.col("metadata").struct.field("judge_score").cast(pl.Float64).alias("judge_score"),
            pl.col("metadata").struct.field("oracle_label").cast(pl.Float64).alias("oracle_label"),
            pl.col("response").str.len_chars().cast(pl.Int64).alias("response_length"),
        ]
    ).select(
        [
            "prompt_id",
            "judge_score",
            "oracle_label",
            "response_length",
            "base_policy_logprob",
            "target_policy_logprobs",
        ]
    )


def load_fresh_draws_df(data_root: Path | str, policy: str) -> pl.DataFrame:
    """Fresh-draw schema: {prompt_id, prompt, response, policy, model, temperature,
    metadata: {judge_score, judge_model, oracle_label}}."""
    data_root = Path(data_root)
    df = pl.read_ndjson(data_root / "responses" / f"{policy}_responses.jsonl")
    return df.with_columns(
        [
            pl.col("metadata").struct.field("judge_score").cast(pl.Float64).alias("judge_score"),
            pl.col("metadata").struct.field("oracle_label").cast(pl.Float64).alias("oracle_label"),
            pl.col("response").str.len_chars().cast(pl.Int64).alias("response_length"),
        ]
    ).select(["prompt_id", "judge_score", "oracle_label", "response_length"])


@dataclass
class PolicyEstimate:
    policy: str
    n_oracle: int
    n_eval: int
    seed: int
    oracle_coverage: float
    # Direct Mean via cje library
    mean: float
    mean_se: float
    mean_ci_lo: float
    mean_ci_hi: float
    # Oracle ground truth (full-oracle average on fresh draws)
    oracle_truth: float
    # CVaR at multiple alphas
    cvar: dict = field(default_factory=dict)
    cvar_ci_lo: dict = field(default_factory=dict)
    cvar_ci_hi: dict = field(default_factory=dict)
    cvar_t_hat: dict = field(default_factory=dict)
    cvar_empirical_truth: dict = field(default_factory=dict)
    audit_p_value: dict = field(default_factory=dict)
    audit_reject: dict = field(default_factory=dict)
    n_bootstrap_failures: dict = field(default_factory=dict)

    def to_rows(self) -> list[dict]:
        """Flatten into one row per alpha for DataFrame export."""
        rows = []
        for alpha in sorted(self.cvar.keys()):
            rows.append(
                {
                    "policy": self.policy,
                    "seed": self.seed,
                    "oracle_coverage": self.oracle_coverage,
                    "n_oracle": self.n_oracle,
                    "n_eval": self.n_eval,
                    "alpha": alpha,
                    "mean": self.mean,
                    "mean_se": self.mean_se,
                    "mean_ci_lo": self.mean_ci_lo,
                    "mean_ci_hi": self.mean_ci_hi,
                    "oracle_truth": self.oracle_truth,
                    "cvar": self.cvar[alpha],
                    "cvar_ci_lo": self.cvar_ci_lo[alpha],
                    "cvar_ci_hi": self.cvar_ci_hi[alpha],
                    "cvar_t_hat": self.cvar_t_hat[alpha],
                    "cvar_empirical_truth": self.cvar_empirical_truth[alpha],
                    "audit_p_value": self.audit_p_value[alpha],
                    "audit_reject": self.audit_reject[alpha],
                    "n_bootstrap_failures": self.n_bootstrap_failures[alpha],
                }
            )
        return rows


def _mask_oracle_in_dataset(ds, oracle_coverage: float, mask_seed: int) -> set[str]:
    """Replicate `ablations/core/base.py:101-143` in place.

    `mask_seed` selects which 25% gets to be the oracle slice. Pass
    `ORACLE_MASK_SEED` (=42) to match the paper's fixed-mask convention,
    or pass the per-experiment seed to expose across-split variability.

    Returns the set of prompt_ids that retain their oracle_label.
    """
    rng = random.Random(mask_seed)
    oracle_indices = [i for i, s in enumerate(ds.samples) if s.oracle_label is not None]
    n_keep = max(2, int(len(oracle_indices) * oracle_coverage))
    keep = set(rng.sample(oracle_indices, min(n_keep, len(oracle_indices))))
    kept_prompts = {ds.samples[i].prompt_id for i in keep}
    for i, sample in enumerate(ds.samples):
        if i not in keep and sample.oracle_label is not None:
            sample.oracle_label = None
    return kept_prompts


def estimate_all_policies(
    data_root: Path | str,
    oracle_coverage: float = 0.25,
    alphas: Iterable[float] = (0.05, 0.10, 0.20),
    B: int = 1000,
    seed: int = 0,
    mask_seed: int | None = None,
    grid_size: int = 61,
    verbose: bool = False,
) -> list[PolicyEstimate]:
    """Run Direct Mean CJE (via cje library) and Direct CVaR-CJE (our code) for
    all 4 target policies on the Arena data.

    `seed` controls cje's fold assignment AND our cluster bootstrap RNG.
    `mask_seed` controls which 25% of the data becomes the oracle slice.
    If `mask_seed is None`, defaults to `seed` (per-experiment slice variability).
    Pass `mask_seed=ORACLE_MASK_SEED` (=42) to fix the slice across seeds (paper convention).
    """
    data_root = Path(data_root)
    alphas = tuple(alphas)
    if mask_seed is None:
        mask_seed = seed

    # 1. Load dataset, mask oracle, calibrate via cje
    ds = load_dataset_from_jsonl(
        str(data_root / "cje_dataset.jsonl"),
        target_policies=list(TARGET_POLICIES),
    )
    kept_prompts = _mask_oracle_in_dataset(ds, oracle_coverage, mask_seed)

    cds, cal = calibrate_dataset(
        ds,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
        # Paper-faithful: ablations/config.py:36 sets reward_calibration_mode="auto",
        # which is what spec.extra forwards into base.py:535. The library's
        # FlexibleCalibrator then selects between monotone and two_stage; on this
        # 25%-oracle slice with response_length covariates it selects "two_stage".
        calibration_mode="auto",
        random_seed=seed,
        covariate_names=["response_length"],  # paper's `direct+cov` (config.py:31)
    )
    sampler = PrecomputedSampler(cds, target_policies=list(TARGET_POLICIES))
    est = CalibratedDirectEstimator(
        target_policies=list(TARGET_POLICIES),
        reward_calibrator=cal.calibrator,
        inference_method="cluster_robust",
        oua_jackknife=True,
        use_augmented_estimator=True,
        run_diagnostics=False,
    )
    for p in TARGET_POLICIES:
        fd = load_fresh_draws_auto(data_root / "responses", policy=p, verbose=False)
        # Paper match: mask fresh-draw oracle labels to the calibration slice so
        # OUA / bootstrap can't "see" oracle labels the calibrator didn't see.
        # Replicates ablations/core/base.py:667-684.
        for fd_sample in fd.samples:
            if fd_sample.oracle_label is not None and fd_sample.prompt_id not in kept_prompts:
                fd_sample.oracle_label = None
        # Compute response_length covariate on fresh draws for direct+cov.
        fd = compute_response_covariates(fd, covariate_names=["response_length"])
        est.add_fresh_draws(p, fd)
    est.fit()
    res = est.estimate()
    ci_list = res.ci(0.05)

    # 2. Build our own polars view of the oracle slice for CVaR
    main_df = load_arena_main(data_root)
    calib_df = main_df.filter(pl.col("prompt_id").is_in(list(kept_prompts)))
    s_train = calib_df["judge_score"].to_numpy()
    y_train = calib_df["oracle_label"].to_numpy()
    # Cluster by prompt_id — one row per prompt in the base dataset, so one cluster = one row.
    train_cluster = np.arange(len(calib_df))

    out: list[PolicyEstimate] = []
    for idx, policy in enumerate(TARGET_POLICIES):
        fd_df = load_fresh_draws_df(data_root, policy)
        s_eval = fd_df["judge_score"].to_numpy()
        y_eval = fd_df["oracle_label"].to_numpy()  # full oracle on fresh draws
        eval_cluster = np.arange(len(fd_df))  # one fresh-draw row per prompt

        pe = PolicyEstimate(
            policy=policy,
            n_oracle=len(s_train),
            n_eval=len(s_eval),
            seed=seed,
            oracle_coverage=oracle_coverage,
            mean=float(res.estimates[idx]),
            mean_se=float(res.standard_errors[idx]),
            mean_ci_lo=float(ci_list[idx][0]),
            mean_ci_hi=float(ci_list[idx][1]),
            oracle_truth=float(np.mean(y_eval)),
        )
        for alpha in alphas:
            cvar, t_hat, _, _ = estimate_direct_cvar_isotonic(
                s_train, y_train, s_eval, alpha, grid_size,
            )
            lo, hi, n_fail = cluster_bootstrap_cvar(
                s_train, y_train, s_eval,
                eval_cluster=eval_cluster,
                train_cluster=train_cluster,
                alpha=alpha,
                grid_size=grid_size,
                B=B,
                seed=seed * 10_000 + idx * 100 + int(alpha * 1000),
            )
            # Empirical CVaR truth on the full-oracle fresh draws
            q_alpha = float(np.quantile(y_eval, alpha))
            tail = y_eval[y_eval <= q_alpha]
            emp_cvar = float(tail.mean()) if len(tail) else float("nan")

            # Transport audit: calibrator fit on base-policy oracle slice tested
            # against target-policy fresh-draw (S, Y) pairs.
            # Use the cross-fit (paired-bootstrap with t-hat re-maximised) variance.
            audit = two_moment_wald_audit_xf(
                s_train, y_train, s_eval, y_eval, t_hat, alpha,
                B=80, fold_seed=seed * 1000 + idx * 10 + int(alpha * 100),
            )
            pe.cvar[alpha] = cvar
            pe.cvar_ci_lo[alpha] = lo
            pe.cvar_ci_hi[alpha] = hi
            pe.cvar_t_hat[alpha] = t_hat
            pe.cvar_empirical_truth[alpha] = emp_cvar
            pe.audit_p_value[alpha] = audit["p_value"]
            pe.audit_reject[alpha] = audit["reject"]
            pe.n_bootstrap_failures[alpha] = n_fail
            if verbose:
                print(
                    f"  {policy} α={alpha}: CVaR={cvar:.4f} [{lo:.4f}, {hi:.4f}] "
                    f"truth={emp_cvar:.4f} audit_p={audit['p_value']:.3g}"
                )
        out.append(pe)
    return out
