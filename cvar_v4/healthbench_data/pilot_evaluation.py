"""Pilot-tier evaluation of L1 (data quality) and L2 (calibration sanity)
from VALIDATION_SPEC.md. Reads raw judge_outputs + responses; no API calls.

Handles partial-oracle policies gracefully (e.g., a policy still being graded
will be flagged as INCOMPLETE rather than treated as missing).

Usage:
    python -m cvar_v4.healthbench_data.pilot_evaluation
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

D = Path(__file__).parent
RESP_DIR = D / "data" / "responses"
JUDGE_DIR = D / "judge_outputs"

POLICIES = ["base", "clone", "premium", "parallel_universe_prompt", "unhelpful", "risky"]


@dataclass
class PolicyData:
    name: str
    n_responses: int
    response_lengths: np.ndarray  # chars
    cheap_scores: dict[str, float]   # {pid: S}
    oracle_scores: dict[str, float]  # {pid: Y}; may be empty if oracle pending
    has_full_oracle: bool


def _load_responses_lengths(policy: str) -> tuple[int, np.ndarray]:
    path = RESP_DIR / f"{policy}_responses.jsonl"
    if not path.exists():
        return 0, np.empty(0)
    lengths = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lengths.append(len(d.get("response", "")))
    return len(lengths), np.array(lengths)


def _load_judge_scores(policy: str, kind: str) -> dict[str, float]:
    path = JUDGE_DIR / f"{policy}_{kind}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s = d.get("score")
            if s is not None and not (isinstance(s, float) and math.isnan(s)):
                out[d["prompt_id"]] = float(s)
    return out


def load_all() -> list[PolicyData]:
    out = []
    for p in POLICIES:
        n, lengths = _load_responses_lengths(p)
        cheap = _load_judge_scores(p, "cheap")
        oracle = _load_judge_scores(p, "oracle")
        out.append(PolicyData(
            name=p, n_responses=n, response_lengths=lengths,
            cheap_scores=cheap, oracle_scores=oracle,
            has_full_oracle=(len(oracle) >= n and n > 0),
        ))
    return out


# --- L1: Data quality ---

def l1_metrics(policies: list[PolicyData]) -> dict:
    """Per-policy L1 metrics."""
    out = {}
    base_oracle = next(p for p in policies if p.name == "base").oracle_scores
    clone_oracle = next(p for p in policies if p.name == "clone").oracle_scores

    for p in policies:
        d: dict = {"name": p.name}
        d["n_responses"] = p.n_responses
        d["has_full_oracle"] = p.has_full_oracle
        d["mean_length"] = float(np.mean(p.response_lengths)) if p.n_responses else float("nan")
        d["median_length"] = float(np.median(p.response_lengths)) if p.n_responses else float("nan")

        if p.oracle_scores:
            y = np.array(list(p.oracle_scores.values()))
            d["n_oracle"] = len(y)
            d["oracle_mean"] = float(y.mean())
            d["oracle_min"] = float(y.min())
            d["oracle_max"] = float(y.max())
            d["all_n_count"] = int(np.sum(y == 0.0))
            d["all_n_rate"] = float(np.mean(y == 0.0))
            d["ceiling_rate"] = float(np.mean(y >= 1.0))
            d["floor_rate"] = float(np.mean(y < 0.0))
        else:
            d["n_oracle"] = 0
            d["oracle_mean"] = float("nan")
            d["all_n_rate"] = float("nan")
            d["ceiling_rate"] = float("nan")
            d["floor_rate"] = float("nan")

        # Length × score correlation per policy on the rows where both exist
        if p.cheap_scores and p.n_responses:
            # Need to align lengths to prompt_ids — re-read responses with pid
            path = RESP_DIR / f"{p.name}_responses.jsonl"
            with path.open() as f:
                rows = [json.loads(l) for l in f if l.strip()]
            len_by_pid = {r["prompt_id"]: len(r.get("response", "")) for r in rows}
            paired = [(len_by_pid[pid], p.cheap_scores[pid])
                      for pid in p.cheap_scores if pid in len_by_pid]
            if len(paired) >= 3:
                ls, ss = zip(*paired)
                d["len_score_corr"] = float(np.corrcoef(ls, ss)[0, 1])
            else:
                d["len_score_corr"] = float("nan")
        else:
            d["len_score_corr"] = float("nan")

        out[p.name] = d

    # Cross-policy: clone-validity
    if base_oracle and clone_oracle:
        common = set(base_oracle) & set(clone_oracle)
        if common:
            base_arr = np.array([base_oracle[pid] for pid in common])
            clone_arr = np.array([clone_oracle[pid] for pid in common])
            out["_clone_validity_diff"] = float(abs(base_arr.mean() - clone_arr.mean()))
        else:
            out["_clone_validity_diff"] = float("nan")
    else:
        out["_clone_validity_diff"] = float("nan")

    return out


def _fmt_pf(value: float, threshold: float, op: str = "<=") -> str:
    if math.isnan(value):
        return "?"
    if op == "<=":
        return "✓" if value <= threshold else "✗"
    if op == ">=":
        return "✓" if value >= threshold else "✗"
    return "?"


def print_l1(metrics: dict) -> None:
    print("=" * 90)
    print("L1 — Data quality (Pilot tier thresholds)")
    print("=" * 90)
    print(f"{'policy':28} {'n':>4} {'med_len':>8} {'oracle_mean':>11} {'all_N':>7} "
          f"{'ceil':>6} {'floor':>6} {'corr_len_s':>11} {'oracle?':>9}")
    print("-" * 90)
    for p in POLICIES:
        d = metrics[p]
        oracle_status = "FULL" if d["has_full_oracle"] else f"PARTIAL({d['n_oracle']})" if d["n_oracle"] else "PENDING"
        print(f"{p:28} {d['n_responses']:>4} "
              f"{d['median_length']:>8.0f} "
              f"{d['oracle_mean']:>+11.3f} "
              f"{d['all_n_rate']:>7.2f} "
              f"{d['ceiling_rate']:>6.2f} "
              f"{d['floor_rate']:>6.2f} "
              f"{d['len_score_corr']:>+11.3f} "
              f"{oracle_status:>9}")
    print()
    diff = metrics["_clone_validity_diff"]
    print(f"clone-validity |base.mean − clone.mean|: {diff:.3f}  threshold ≤ 0.05: "
          f"{_fmt_pf(diff, 0.05)}")

    # Pilot-tier check on all-N
    print("\nL1 Pilot pass criteria:")
    for p in POLICIES:
        if p == "unhelpful":
            continue
        d = metrics[p]
        if not d["has_full_oracle"]:
            print(f"  {p}: SKIP (oracle incomplete)")
            continue
        anr = d["all_n_rate"]
        flag = _fmt_pf(anr, 0.15)
        print(f"  {p} all-N rate {anr:.2f}  threshold ≤ 0.15: {flag}")

    # Ordering check
    print()
    have = {p: metrics[p]["oracle_mean"] for p in POLICIES if metrics[p]["has_full_oracle"]}
    if "unhelpful" in have:
        print(f"  ordering: unhelpful={have['unhelpful']:+.3f}  "
              + "  ".join(f"{k}={v:+.3f}" for k, v in have.items() if k != "unhelpful"))
    else:
        print("  ordering: SKIP (unhelpful oracle pending)")


# --- L2: Calibration sanity ---

def _fisher_z_lower(r: float, n: int, conf: float = 0.95) -> float:
    """Lower bound of Fisher z 95% CI on Pearson correlation."""
    if n < 4 or abs(r) >= 1.0:
        return float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_lo = z - 1.96 * se
    return (math.exp(2 * z_lo) - 1) / (math.exp(2 * z_lo) + 1)


def l2_metrics(policies: list[PolicyData]) -> dict:
    """Per-policy L2 metrics on the oracle slice (full or partial)."""
    out = {}
    for p in policies:
        d: dict = {"name": p.name}
        common = set(p.cheap_scores) & set(p.oracle_scores)
        d["n_paired"] = len(common)
        if len(common) >= 3:
            s = np.array([p.cheap_scores[pid] for pid in sorted(common)])
            y = np.array([p.oracle_scores[pid] for pid in sorted(common)])
            d["corr_S_Y"] = float(np.corrcoef(s, y)[0, 1]) if len(common) >= 4 else float("nan")
            d["corr_lower_CI"] = _fisher_z_lower(d["corr_S_Y"], len(common))
            d["mean_gap_S_minus_Y"] = float(s.mean() - y.mean())
        else:
            d["corr_S_Y"] = float("nan")
            d["corr_lower_CI"] = float("nan")
            d["mean_gap_S_minus_Y"] = float("nan")
        out[p.name] = d

    # Logger surrogate map (isotonic on base)
    base = next(p for p in policies if p.name == "base")
    common = sorted(set(base.cheap_scores) & set(base.oracle_scores))
    if len(common) >= 5:
        s = np.array([base.cheap_scores[pid] for pid in common])
        y = np.array([base.oracle_scores[pid] for pid in common])
        order = np.argsort(s)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(s[order], y[order])
        # Quintile evaluation points
        quint = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        pred = iso.predict(quint)
        out["_iso_map"] = {float(q): float(p) for q, p in zip(quint, pred)}
        # RMS slope
        slopes = np.diff(pred) / np.diff(quint)
        out["_iso_rms_slope"] = float(np.sqrt(np.mean(slopes ** 2)))
        # Residuals
        residuals = y - iso.predict(s)
        out["_residual_mean"] = float(residuals.mean())
        out["_residual_std"] = float(residuals.std())
        # skewness (Fisher-Pearson)
        if residuals.std() > 1e-9:
            out["_residual_skew"] = float(np.mean(((residuals - residuals.mean()) / residuals.std()) ** 3))
        else:
            out["_residual_skew"] = float("nan")
    else:
        out["_iso_map"] = {}
        out["_iso_rms_slope"] = float("nan")
        out["_residual_mean"] = float("nan")
        out["_residual_std"] = float("nan")
        out["_residual_skew"] = float("nan")

    return out


def print_l2(metrics: dict) -> None:
    print("=" * 90)
    print("L2 — Calibration sanity (Pilot tier thresholds)")
    print("=" * 90)
    print(f"{'policy':28} {'n_paired':>9} {'corr(S,Y)':>10} {'corr_LB':>10} {'gap':>9} {'verdict':<20}")
    print("-" * 90)
    for p in POLICIES:
        d = metrics[p]
        verdict = ""
        if not math.isnan(d["corr_lower_CI"]):
            if p == "unhelpful":
                verdict = "(degenerate near 0)"
            else:
                verdict = "✓ ≥0.30" if d["corr_lower_CI"] >= 0.30 else "✗ <0.30"
        print(f"{p:28} {d['n_paired']:>9} "
              f"{d['corr_S_Y']:>+10.3f} "
              f"{d['corr_lower_CI']:>+10.3f} "
              f"{d['mean_gap_S_minus_Y']:>+9.3f} "
              f"{verdict:<20}")
    print()
    print(f"Logger isotonic surrogate map (base):")
    for s_in in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        if s_in in metrics["_iso_map"]:
            print(f"  S = {s_in:.2f}  →  Ŷ = {metrics['_iso_map'][s_in]:+.3f}")
    rms = metrics["_iso_rms_slope"]
    print(f"  RMS slope: {rms:.3f}  threshold ≥ 0.3: {_fmt_pf(rms, 0.3, '>=')}")
    rmean = metrics["_residual_mean"]
    print(f"  residual mean: {rmean:+.3f}  threshold |.| ≤ 0.05: {_fmt_pf(abs(rmean), 0.05)}")
    rskew = metrics["_residual_skew"]
    print(f"  residual skew: {rskew:+.3f}  threshold |.| ≤ 1.0: {_fmt_pf(abs(rskew), 1.0)}")


def main() -> None:
    policies = load_all()
    print()
    print(f"Run state: {sum(p.has_full_oracle for p in policies)}/5 policies have full oracle\n")

    l1 = l1_metrics(policies)
    print_l1(l1)
    print()

    l2 = l2_metrics(policies)
    print_l2(l2)
    print()


if __name__ == "__main__":
    main()
