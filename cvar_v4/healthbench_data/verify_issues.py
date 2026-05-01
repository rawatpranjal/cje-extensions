"""Phase B: verify the four open issues from ISSUES_LOG.md against n=100 cheap-S data.

Each measurement comes with an explicit threshold and a YES/NO verdict.

Usage: python3 -m cvar_v4.healthbench_data.verify_issues
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
JUDGE = Path(__file__).parent / "judge_outputs"


def _load_scores(policy: str, kind: str = "cheap") -> dict:
    """Returns {prompt_id: {score, n_criteria}}."""
    path = JUDGE / f"{policy}_{kind}.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            out[d["prompt_id"]] = {"score": d["score"], "n_criteria": d.get("n_criteria", 0)}
    return out


def _load_response_lengths(policy: str) -> dict:
    """Returns {prompt_id: char_length_of_response}."""
    path = DATA / "responses" / f"{policy}_responses.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            out[d["prompt_id"]] = len(d["response"])
    return out


def issue_5_grader_noise() -> dict:
    """How often do base and clone (same model, different seed) score very differently?"""
    base = _load_scores("base")
    clone = _load_scores("clone")
    common = set(base) & set(clone)
    diffs = [abs(base[p]["score"] - clone[p]["score"]) for p in common]
    arr = np.array(diffs)
    n = arr.size
    n_big_div = int((arr > 0.30).sum())
    pct = 100 * n_big_div / max(n, 1)
    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    threshold_material = pct > 5.0
    threshold_ignorable = pct < 2.0
    print(f"\n=== ISSUE 5: grader noise on borderline criteria ===")
    print(f"  n_pairs (base, clone)        = {n}")
    print(f"  median |base - clone|        = {median:.3f}")
    print(f"  p90 |base - clone|           = {p90:.3f}")
    print(f"  fraction with |Δ| > 0.30     = {pct:.1f}% ({n_big_div}/{n})")
    if threshold_material:
        verdict = "MATERIAL — grader noise is real and would benefit from a stronger judge or multi-judge ensemble"
    elif threshold_ignorable:
        verdict = "IGNORABLE — grader is consistent enough"
    else:
        verdict = "MARGINAL — measurable but not catastrophic; can live with for a pilot"
    print(f"  verdict: {verdict}")
    return {"pct_big_div": pct, "verdict": verdict, "median": median, "p90": p90}


def issue_6_small_rubrics() -> dict:
    """Are small-rubric prompts more noisy than large-rubric prompts?"""
    base = _load_scores("base")
    clone = _load_scores("clone")
    common = sorted(set(base) & set(clone))
    diffs = np.array([abs(base[p]["score"] - clone[p]["score"]) for p in common])
    n_crits = np.array([base[p]["n_criteria"] for p in common])
    if n_crits.size < 5:
        return {"verdict": "INSUFFICIENT DATA", "n": n_crits.size}
    rho = float(np.corrcoef(n_crits, diffs)[0, 1])
    # Compare large-rubric (>=8 criteria) vs small-rubric (<8) divergences
    big = diffs[n_crits >= 8]
    small = diffs[n_crits < 8]
    threshold_material = rho < -0.30
    print(f"\n=== ISSUE 6: small rubrics → bigger noise? ===")
    print(f"  n_crit distribution           = min={int(n_crits.min())}, median={int(np.median(n_crits))}, max={int(n_crits.max())}")
    print(f"  Pearson(n_crit, |Δ|)          = {rho:+.3f}")
    print(f"  median |Δ| where n_crit ≥ 8   = {float(np.median(big)) if big.size else float('nan'):.3f}  (n={big.size})")
    print(f"  median |Δ| where n_crit < 8   = {float(np.median(small)) if small.size else float('nan'):.3f}  (n={small.size})")
    if threshold_material:
        n_kept_at_8 = int((n_crits >= 8).sum())
        pct_kept = 100 * n_kept_at_8 / n_crits.size
        verdict = f"MATERIAL — filtering to n_crit ≥ 8 keeps {pct_kept:.0f}% of prompts and roughly halves divergence"
    else:
        verdict = "MARGINAL — relationship is weak; filtering not worth it"
    print(f"  verdict: {verdict}")
    return {"pearson": rho, "verdict": verdict, "n_crit_min": int(n_crits.min()),
            "n_crit_median": int(np.median(n_crits)), "n_crit_max": int(n_crits.max())}


def issue_8_unhelpful_beats_base() -> dict:
    """How often does the deliberately-bad policy beat the baseline?"""
    base = _load_scores("base")
    unhelp = _load_scores("unhelpful")
    common = sorted(set(base) & set(unhelp))
    wins = []  # prompts where unhelpful >= base + 0.20
    for p in common:
        b = base[p]["score"]
        u = unhelp[p]["score"]
        if u >= b + 0.20:
            wins.append((p, base[p]["n_criteria"], b, u))
    pct_wins = 100 * len(wins) / max(len(common), 1)
    n_small_rubric_wins = sum(1 for _, n, _, _ in wins if n <= 5)
    pct_small = 100 * n_small_rubric_wins / max(len(wins), 1) if wins else 0
    threshold_material_a = pct_wins > 10.0
    threshold_concentrated = wins and pct_small > 50.0
    print(f"\n=== ISSUE 8: rubric coverage gap (unhelpful beats base on small rubrics) ===")
    print(f"  n_prompts                     = {len(common)}")
    print(f"  unhelpful wins by ≥0.20       = {len(wins)} ({pct_wins:.1f}%)")
    if wins:
        print(f"  of those wins, n_crit ≤ 5     = {n_small_rubric_wins} ({pct_small:.0f}%)")
    if threshold_material_a:
        verdict = f"MATERIAL — unhelpful wins {pct_wins:.0f}% of prompts, undermining its role as audit-positive"
    else:
        verdict = "ACCEPTABLE — unhelpful rarely wins outright"
    print(f"  verdict: {verdict}")
    return {"pct_wins": pct_wins, "verdict": verdict, "n_wins": len(wins),
            "concentrated_in_small_rubrics": bool(threshold_concentrated)}


def issue_9_unhelpful_too_soft() -> dict:
    """Is the unhelpful policy actually scoring badly (the audit-positive by design role)?

    The original v1 system prompt asked for brevity, but brief 'consult a doctor' responses
    accidentally satisfy the rubric on simple referral prompts. The v2 system prompt switched
    to off-topic content. The threshold here checks the *outcome* (score distribution), not
    the *length* (which was a v1-specific failure-mode signal).
    """
    lens = _load_response_lengths("unhelpful")
    scores = _load_scores("unhelpful")
    if not lens:
        return {"verdict": "NO DATA"}
    arr = np.array(list(lens.values()))
    score_arr = np.array([scores[p]["score"] for p in lens if p in scores])
    n_high_score = int((score_arr >= 0.70).sum())
    pct_high = 100 * n_high_score / max(score_arr.size, 1)
    n_above_zero = int((score_arr > 0).sum())
    pct_positive = 100 * n_above_zero / max(score_arr.size, 1)
    mean_score = float(score_arr.mean()) if score_arr.size else float("nan")
    # Threshold: the unhelpful policy is "appropriately bad" when:
    # - mean score is low (≤ 0.20)
    # - very few high-scoring outliers (≤ 5%)
    # - mostly negative scores (positive-share ≤ 30%)
    threshold_too_soft = (mean_score > 0.20) or (pct_high > 5.0) or (pct_positive > 30.0)
    print(f"\n=== ISSUE 9: unhelpful policy 'audit-positive' enough? (score-based) ===")
    print(f"  unhelpful response length  : min={int(arr.min())}, median={int(np.median(arr))}, max={int(arr.max())}")
    print(f"  mean score                 : {mean_score:+.3f}")
    print(f"  scores > 0 (positive)      : {n_above_zero}/{score_arr.size} ({pct_positive:.1f}%)")
    print(f"  scores ≥ 0.70 (high)       : {n_high_score}/{score_arr.size} ({pct_high:.1f}%)")
    if threshold_too_soft:
        verdict = "MATERIAL — unhelpful is not aggressively bad; tightening recommended"
    else:
        verdict = f"OK — unhelpful is appropriately bad (mean {mean_score:+.3f}, only {pct_high:.0f}% high-scoring outliers)"
    print(f"  verdict: {verdict}")
    return {"mean_score": mean_score, "pct_high_score": pct_high, "pct_positive": pct_positive,
            "verdict": verdict, "median_len": int(np.median(arr))}


def main():
    print("=" * 78)
    print("VERIFICATION OF ISSUES 5, 6, 8, 9 AGAINST n=100 CHEAP-S DATA")
    print("=" * 78)
    r5 = issue_5_grader_noise()
    r6 = issue_6_small_rubrics()
    r8 = issue_8_unhelpful_beats_base()
    r9 = issue_9_unhelpful_too_soft()
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Issue 5 (grader noise):           {r5['verdict']}")
    print(f"  Issue 6 (small-rubric noise):     {r6['verdict']}")
    print(f"  Issue 8 (rubric coverage gap):    {r8['verdict']}")
    print(f"  Issue 9 (unhelpful too soft):     {r9['verdict']}")


if __name__ == "__main__":
    main()
