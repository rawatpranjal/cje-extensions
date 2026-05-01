"""Phase E: final scrutiny pass before accepting n=100 cheap-S results.

Tier 1 — blocking: reconcile raw → aggregated; deep-dive premium worst 5; unhelpful sanity.
Tier 2 — informative: length, per-criterion, score shape, negative-criteria semantics.
Tier 3 — nice-to-have: degenerate prompts, rubric source fidelity.

Usage:
    python3 -m cvar_v4.healthbench_data.scrutiny_eda --check all
    python3 -m cvar_v4.healthbench_data.scrutiny_eda --check E.1
    python3 -m cvar_v4.healthbench_data.scrutiny_eda --check E.2
    ...
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .judge import _aggregate_score, _judge_path
from .policies import POLICIES

DIR = Path(__file__).parent
DATA = DIR / "data"
JUDGE = DIR / "judge_outputs"
RESPONSES = DATA / "responses"
PROMPTS = DATA / "prompts.jsonl"
OSS_EVAL_LOCAL = Path("/tmp/hb_oss_eval.jsonl")


def _load_cheap_rows(policy: str) -> list[dict]:
    rows = []
    with _judge_path(policy, "cheap").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_raw_rows(policy: str) -> list[dict]:
    path = JUDGE / f"{policy}_cheap_raw.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_responses(policy: str) -> dict[str, dict]:
    path = RESPONSES / f"{policy}_responses.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["prompt_id"]] = d
    return out


def _load_prompts() -> dict[str, dict]:
    out = {}
    with PROMPTS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["prompt_id"]] = d
    return out


# ---------------------------------------------------------------------------
# E.1 — Premium worst-5 deep dive (writes a markdown file for human reading)
# ---------------------------------------------------------------------------

def E1_premium_worst5() -> dict:
    print("=" * 78)
    print("E.1 — Premium's worst-5 deep dive")
    print("=" * 78)
    pr = _load_cheap_rows("premium")
    base = _load_cheap_rows("base")
    base_by_id = {r["prompt_id"]: r for r in base}
    pr_responses = _load_responses("premium")
    base_responses = _load_responses("base")
    prompts = _load_prompts()

    pr_sorted = sorted(pr, key=lambda r: r["score"])
    worst5 = pr_sorted[:5]
    print(f"  Premium has {len(pr)} prompts; bottom-5 scores: "
          + ", ".join(f"{r['score']:+.3f}" for r in worst5))
    out_path = DIR / "scrutiny_premium_worst5.md"
    lines = ["# E.1 — Premium worst-5 deep dive\n",
             "Per-row: prompt, premium response, base response, full rubric, "
             "verdict comparison, and our verdict (a/b/c).\n",
             "Verdict legend: (a) genuinely worse premium response; "
             "(b) length-driven negative-criterion punishment; "
             "(c) grader noise on borderline criteria.\n"]
    for i, r in enumerate(worst5, 1):
        pid = r["prompt_id"]
        score_premium = r["score"]
        base_row = base_by_id.get(pid)
        score_base = base_row["score"] if base_row else float("nan")
        prompt_text = prompts[pid]["prompt_text"]
        premium_resp = pr_responses[pid]["response"]
        base_resp = base_responses[pid]["response"]
        prem_verdicts = r["verdicts"]
        base_verdicts = {v["criterion"]: v for v in base_row["verdicts"]} if base_row else {}
        lines.append(f"## {i}. prompt_id `{pid[:12]}` — premium={score_premium:+.3f}, base={score_base:+.3f}\n")
        lines.append(f"**Prompt** ({len(prompt_text)} chars):\n\n```\n{prompt_text}\n```\n")
        lines.append(f"**Premium response** ({len(premium_resp)} chars):\n\n```\n{premium_resp}\n```\n")
        lines.append(f"**Base response** ({len(base_resp)} chars):\n\n```\n{base_resp}\n```\n")
        lines.append(f"**Per-criterion verdicts** (Y = criterion satisfied; for negative-points criteria, Y = violation TRUE):\n")
        lines.append("\n| points | criterion | premium | base |\n|---:|---|:---:|:---:|\n")
        for v in prem_verdicts:
            crit = v["criterion"]
            pts = v["points"]
            pv = v["verdict"]
            bv = base_verdicts.get(crit, {}).get("verdict", "?")
            crit_short = crit if len(crit) < 80 else crit[:77] + "..."
            lines.append(f"| {pts:+d} | {crit_short} | {pv} | {bv} |\n")
        # Quick automated summary: how much of the gap is due to negative criteria?
        prem_neg_violations = sum(abs(v["points"]) for v in prem_verdicts
                                   if v["points"] < 0 and v["verdict"] == "Y")
        base_neg_violations = sum(abs(v["points"]) for v in base_verdicts.values()
                                   if v["points"] < 0 and v["verdict"] == "Y")
        prem_pos_earned = sum(v["points"] for v in prem_verdicts
                               if v["points"] > 0 and v["verdict"] == "Y")
        base_pos_earned = sum(v["points"] for v in base_verdicts.values()
                               if v["points"] > 0 and v["verdict"] == "Y")
        lines.append(f"\n**Decomposition**: premium loses {prem_neg_violations} pts on neg-criterion violations "
                     f"vs base's {base_neg_violations}; premium earns {prem_pos_earned} pos pts vs base's {base_pos_earned}.\n")
        lines.append(f"**Hint for verdict**: ")
        if prem_neg_violations > base_neg_violations and prem_pos_earned >= base_pos_earned:
            lines.append("(b) likely length-driven punishment: premium earns ≥ base on positives but loses more on negatives.\n")
        elif prem_pos_earned < base_pos_earned and prem_neg_violations <= base_neg_violations:
            lines.append("(a) likely genuine: premium earns FEWER positive points than base.\n")
        else:
            lines.append("ambiguous — manual read needed.\n")
        lines.append("\n---\n")
    out_path.write_text("".join(lines))
    print(f"  → wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  Open in editor to read; auto-hint provided per row.")
    return {"out_path": str(out_path), "n_rows": len(worst5)}


# ---------------------------------------------------------------------------
# E.2 — Reconciliation: per-criterion raw → per-prompt aggregated score
# ---------------------------------------------------------------------------

def E2_reconcile_all() -> dict:
    print("=" * 78)
    print("E.2 — Per-criterion → per-prompt reconciliation")
    print("=" * 78)
    summary = {}
    all_pass = True
    for pol in [p.name for p in POLICIES]:
        rows = _load_cheap_rows(pol)
        # 1) Re-apply _aggregate_score to the per-prompt verdicts list and diff vs saved score.
        max_diff = 0.0
        n_total = len(rows)
        n_mismatch = 0
        for r in rows:
            pairs = [(int(v["points"]), v["verdict"]) for v in r["verdicts"]]
            recomputed = _aggregate_score(pairs)
            saved = r["score"]
            if isinstance(saved, float) and np.isnan(saved):
                if not (isinstance(recomputed, float) and np.isnan(recomputed)):
                    n_mismatch += 1
                continue
            d = abs(recomputed - saved)
            if d > max_diff:
                max_diff = d
            if d > 1e-9:
                n_mismatch += 1
        # 2) Check raw.jsonl is consistent with cheap.jsonl per-prompt verdicts.
        raw = _load_raw_rows(pol)
        # Build dict (pid, criterion) → list of verdicts seen in raw (preserve order)
        raw_pairs = defaultdict(list)
        for rr in raw:
            raw_pairs[(rr["prompt_id"], rr["criterion"])].append(rr["verdict"])
        n_dup_pairs = sum(1 for v in raw_pairs.values() if len(v) > 1)
        # For each per-prompt verdict, check the raw file has a matching entry
        n_raw_missing = 0
        n_raw_disagree = 0
        for r in rows:
            pid = r["prompt_id"]
            for v in r["verdicts"]:
                key = (pid, v["criterion"])
                vs = raw_pairs.get(key)
                if not vs:
                    n_raw_missing += 1
                elif v["verdict"] not in vs:
                    n_raw_disagree += 1
        # PASS criterion: per-prompt aggregation idempotent AND no raw entries that disagree
        # with cheap.jsonl. raw_missing > 0 is benign — those entries were graded before the
        # per-criterion checkpoint feature existed, so they only live in cheap.jsonl[verdicts]
        # (which is the source of truth and verified complete separately).
        ok = (max_diff < 1e-9 and n_mismatch == 0 and n_raw_disagree == 0)
        all_pass = all_pass and ok
        flag = "PASS" if ok else "FAIL"
        note = " (raw=partial audit trail)" if n_raw_missing > 0 else ""
        print(f"  [{pol:28}] n={n_total} max|Δ|={max_diff:.2e} "
              f"mismatch={n_mismatch} raw_missing={n_raw_missing} "
              f"raw_disagree={n_raw_disagree} dup_pairs={n_dup_pairs} {flag}{note}")
        summary[pol] = {
            "n_total": n_total, "max_diff": max_diff,
            "n_mismatch": n_mismatch, "n_raw_missing": n_raw_missing,
            "n_raw_disagree": n_raw_disagree, "n_dup_pairs": n_dup_pairs,
            "pass": ok,
        }
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return {"all_pass": all_pass, "per_policy": summary}


# ---------------------------------------------------------------------------
# E.3 — Unhelpful-v2 sanity: 10 random rows
# ---------------------------------------------------------------------------

def E3_unhelpful_sanity(n_sample: int = 10, seed: int = 1234) -> dict:
    print("=" * 78)
    print(f"E.3 — Unhelpful-v2 sanity ({n_sample} random rows)")
    print("=" * 78)
    resps = _load_responses("unhelpful")
    prompts = _load_prompts()
    pids = sorted(resps.keys())
    rng = random.Random(seed)
    sample_ids = rng.sample(pids, min(n_sample, len(pids)))
    medical_words = re.compile(
        r"\b(doctor|physician|nurse|healthcare|hospital|clinic|emergency|medication|"
        r"prescribe|diagnos|symptom|disease|condition|patient|medical|treatment|therapy)\b",
        re.IGNORECASE,
    )
    flagged = []
    for i, pid in enumerate(sample_ids, 1):
        prompt = prompts[pid]["prompt_text"]
        resp = resps[pid]["response"]
        match = medical_words.search(resp)
        flag = f"⚠ has '{match.group(0)}'" if match else "✓ off-topic"
        if match:
            flagged.append((pid, match.group(0)))
        prompt_short = prompt.replace("\n", " ")[:100]
        resp_short = resp[:200]
        print(f"\n  [{i:2d}] pid={pid[:12]} ({flag})")
        print(f"      prompt: {prompt_short}{'...' if len(prompt) > 100 else ''}")
        print(f"      response ({len(resp)} chars): {resp_short}{'...' if len(resp) > 200 else ''}")
    pass_rate = (n_sample - len(flagged)) / n_sample
    pass_ = pass_rate >= 0.9
    print(f"\n  {n_sample - len(flagged)}/{n_sample} clearly off-topic — {'PASS' if pass_ else 'FAIL'}")
    if flagged:
        print(f"  Flagged: {flagged}")
    return {"n_sample": n_sample, "n_flagged": len(flagged), "pass": pass_,
            "flagged": [(pid, w) for pid, w in flagged]}


# ---------------------------------------------------------------------------
# E.4 — Response-length distribution per policy
# ---------------------------------------------------------------------------

def E4_response_lengths() -> dict:
    print("=" * 78)
    print("E.4 — Response-length distribution per policy")
    print("=" * 78)
    print(f"  {'policy':28} {'n':>4} {'min':>5} {'p25':>5} {'med':>5} {'p75':>5} {'p90':>5} {'max':>5} "
          f"{'mean':>6}  corr(len,score)")
    summary = {}
    for pol in [p.name for p in POLICIES]:
        resps = _load_responses(pol)
        scores = {r["prompt_id"]: r["score"] for r in _load_cheap_rows(pol)}
        common = sorted(set(resps) & set(scores))
        lens = np.array([len(resps[pid]["response"]) for pid in common])
        scs = np.array([scores[pid] for pid in common])
        if lens.size < 2:
            continue
        rho = float(np.corrcoef(lens, scs)[0, 1])
        stats = {
            "n": int(lens.size), "min": int(lens.min()),
            "p25": int(np.percentile(lens, 25)),
            "med": int(np.median(lens)),
            "p75": int(np.percentile(lens, 75)),
            "p90": int(np.percentile(lens, 90)),
            "max": int(lens.max()), "mean": float(lens.mean()),
            "corr_len_score": rho,
        }
        summary[pol] = stats
        print(f"  {pol:28} {stats['n']:>4} {stats['min']:>5} {stats['p25']:>5} {stats['med']:>5} "
              f"{stats['p75']:>5} {stats['p90']:>5} {stats['max']:>5} {stats['mean']:>6.0f}  "
              f"{rho:+.3f}")
    return summary


# ---------------------------------------------------------------------------
# E.5 — Per-criterion success rates + per-theme breakdown
# ---------------------------------------------------------------------------

def E5_criterion_and_theme() -> dict:
    print("=" * 78)
    print("E.5 — Per-criterion success rates + per-theme breakdown")
    print("=" * 78)
    prompts = _load_prompts()

    # Per-criterion success rate for the BASE policy (baseline view)
    base_rows = _load_cheap_rows("base")
    crit_outcomes = defaultdict(lambda: {"y": 0, "n": 0, "points": 0})
    for r in base_rows:
        for v in r["verdicts"]:
            c = v["criterion"]
            d = crit_outcomes[c]
            d["points"] = v["points"]
            if v["verdict"] == "Y":
                d["y"] += 1
            elif v["verdict"] == "N":
                d["n"] += 1
    crit_rates = []
    for c, d in crit_outcomes.items():
        total = d["y"] + d["n"]
        if total < 5:
            continue
        crit_rates.append((c, d["points"], d["y"] / total, total))
    # Sort by success rate, asc
    crit_rates.sort(key=lambda x: x[2])
    print("  5 hardest criteria (lowest base-policy success rate, n≥5 prompts):")
    for c, p, rate, n in crit_rates[:5]:
        cs = c if len(c) < 70 else c[:67] + "..."
        print(f"    [{p:+3d}, n={n:3d}] {rate*100:5.1f}%  {cs}")
    print("  5 easiest criteria (highest base-policy success rate):")
    for c, p, rate, n in crit_rates[-5:]:
        cs = c if len(c) < 70 else c[:67] + "..."
        print(f"    [{p:+3d}, n={n:3d}] {rate*100:5.1f}%  {cs}")

    # Per-theme mean per policy
    print("\n  Per-theme mean score (rows: theme; cols: policies):")
    by_theme: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pol in [p.name for p in POLICIES]:
        for r in _load_cheap_rows(pol):
            theme = prompts[r["prompt_id"]].get("theme", "(unknown)")
            by_theme[theme][pol].append(r["score"])
    pol_names = [p.name[:14] for p in POLICIES]
    print(f"  {'theme':24}" + " ".join(f"{n:>10}" for n in pol_names) + "  n")
    rows = []
    for theme in sorted(by_theme.keys(), key=lambda t: -sum(len(v) for v in by_theme[t].values())):
        d = by_theme[theme]
        n_total = max(len(v) for v in d.values()) if d else 0
        means = [float(np.mean(d[p.name])) if d.get(p.name) else float("nan") for p in POLICIES]
        print(f"  {str(theme)[:24]:24}" + " ".join(f"{m:>+10.3f}" for m in means) + f"  {n_total}")
        rows.append({"theme": theme, "means": means, "n": n_total})
    return {"hardest": [(c[0], c[1], c[2], c[3]) for c in crit_rates[:5]],
            "easiest": [(c[0], c[1], c[2], c[3]) for c in crit_rates[-5:]],
            "themes": rows}


# ---------------------------------------------------------------------------
# E.6 — Score distribution shape per policy
# ---------------------------------------------------------------------------

def E6_score_shape() -> dict:
    print("=" * 78)
    print("E.6 — Score distribution shape")
    print("=" * 78)

    def _moment(arr, k):
        m = arr.mean()
        return np.mean((arr - m) ** k)

    print(f"  {'policy':28} {'n':>4} {'mean':>7} {'std':>7} {'skew':>6} {'kurt':>6} "
          f"{'n_neg':>6} {'n_zero':>7} {'n_high':>7}")
    summary = {}
    for pol in [p.name for p in POLICIES]:
        rows = _load_cheap_rows(pol)
        arr = np.array([r["score"] for r in rows if not (isinstance(r["score"], float) and np.isnan(r["score"]))])
        if arr.size < 2:
            continue
        m, s = arr.mean(), arr.std()
        skew = _moment(arr, 3) / (s ** 3 + 1e-12)
        kurt = _moment(arr, 4) / (s ** 4 + 1e-12) - 3
        n_neg = int((arr < 0).sum())
        n_zero = int((np.abs(arr) < 1e-9).sum())
        n_high = int((arr >= 0.7).sum())
        summary[pol] = {"n": int(arr.size), "mean": float(m), "std": float(s),
                        "skew": float(skew), "kurt": float(kurt),
                        "n_neg": n_neg, "n_zero": n_zero, "n_high": n_high}
        print(f"  {pol:28} {arr.size:>4} {m:>+7.3f} {s:>7.3f} {skew:>+6.2f} {kurt:>+6.2f} "
              f"{n_neg:>6} {n_zero:>7} {n_high:>7}")
    return summary


# ---------------------------------------------------------------------------
# E.7 — Negative-criteria semantics: 5 Y verdicts on negative-points criteria
# ---------------------------------------------------------------------------

def E7_negative_criteria_spotcheck(n: int = 5, seed: int = 7) -> dict:
    print("=" * 78)
    print(f"E.7 — Negative-criteria semantics ({n} spot-checks)")
    print("=" * 78)
    rng = random.Random(seed)
    candidates = []  # (policy, pid, criterion, points, response_text)
    for pol in [p.name for p in POLICIES]:
        resps = _load_responses(pol)
        rows = _load_cheap_rows(pol)
        for r in rows:
            for v in r["verdicts"]:
                if v["points"] < 0 and v["verdict"] == "Y":
                    pid = r["prompt_id"]
                    if pid in resps:
                        candidates.append((pol, pid, v["criterion"], v["points"],
                                            resps[pid]["response"]))
    print(f"  Pool: {len(candidates)} (policy, prompt, neg-criterion) tuples with verdict=Y")
    rng.shuffle(candidates)
    sample = candidates[:n]
    for i, (pol, pid, crit, pts, resp) in enumerate(sample, 1):
        resp_short = resp[:300] + ("..." if len(resp) > 300 else "")
        crit_short = crit if len(crit) < 100 else crit[:97] + "..."
        print(f"\n  [{i}] policy={pol} pid={pid[:12]} points={pts}")
        print(f"      CRITERION: {crit_short}")
        print(f"      RESPONSE ({len(resp)} chars): {resp_short}")
        print(f"      VERDICT=Y means: '{crit_short}' is TRUE → subtract {abs(pts)} pts")
    return {"n_candidates": len(candidates), "n_sampled": len(sample)}


# ---------------------------------------------------------------------------
# E.8 — Degenerate prompts (all 5 policies tie)
# ---------------------------------------------------------------------------

def E8_degenerate_prompts() -> dict:
    print("=" * 78)
    print("E.8 — Degenerate prompts (all 5 policies score identically)")
    print("=" * 78)
    by_id: dict[str, dict[str, float]] = defaultdict(dict)
    for pol in [p.name for p in POLICIES]:
        for r in _load_cheap_rows(pol):
            s = r["score"]
            if not (isinstance(s, float) and np.isnan(s)):
                by_id[r["prompt_id"]][pol] = s
    prompts = _load_prompts()
    degenerate = []
    for pid, pol_scores in by_id.items():
        if len(pol_scores) == 5 and len(set(pol_scores.values())) == 1:
            score = next(iter(pol_scores.values()))
            n_crit = prompts[pid].get("n_criteria", 0)
            n_neg = sum(1 for r in (prompts[pid].get("rubrics") or []) if r.get("points", 0) < 0)
            degenerate.append((pid, score, n_crit, n_neg))
    print(f"  {len(degenerate)}/{len(by_id)} prompts have all 5 policies scoring identically")
    for pid, score, nc, nn in degenerate:
        prompt_text = prompts[pid]["prompt_text"][:80]
        print(f"  pid={pid[:12]} score={score:+.3f} n_crit={nc} n_neg={nn}")
        print(f"      '{prompt_text}{'...' if len(prompts[pid]['prompt_text']) > 80 else ''}'")
    return {"n_degenerate": len(degenerate), "n_total": len(by_id),
            "rows": [(pid, s, nc, nn) for pid, s, nc, nn in degenerate]}


# ---------------------------------------------------------------------------
# E.9 — Rubric source fidelity
# ---------------------------------------------------------------------------

def E9_rubric_fidelity(n: int = 2, seed: int = 11) -> dict:
    print("=" * 78)
    print(f"E.9 — Rubric source fidelity ({n} prompts cross-checked vs HealthBench source)")
    print("=" * 78)
    if not OSS_EVAL_LOCAL.exists():
        print(f"  SKIP: source file {OSS_EVAL_LOCAL} not present")
        return {"skipped": True}
    src_by_id: dict[str, dict] = {}
    with OSS_EVAL_LOCAL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("prompt_id"):
                src_by_id[d["prompt_id"]] = d
    print(f"  Source has {len(src_by_id)} prompts")
    prompts = _load_prompts()
    pids = sorted(prompts.keys())
    rng = random.Random(seed)
    sample = rng.sample(pids, min(n, len(pids)))
    n_match = 0
    for pid in sample:
        ours = prompts[pid]
        src = src_by_id.get(pid)
        if src is None:
            print(f"  pid={pid[:12]} NOT IN SOURCE")
            continue
        our_rubrics = sorted([(r["points"], r["criterion"]) for r in (ours.get("rubrics") or [])])
        src_rubrics = sorted([(r["points"], r["criterion"]) for r in (src.get("rubrics") or [])])
        match = our_rubrics == src_rubrics
        if match:
            n_match += 1
        print(f"  pid={pid[:12]} our_n={len(our_rubrics)} src_n={len(src_rubrics)} "
              f"identical_rubrics={match}")
        if not match:
            print(f"    OUR FIRST: {our_rubrics[:2]}")
            print(f"    SRC FIRST: {src_rubrics[:2]}")
    return {"n_sample": len(sample), "n_match": n_match}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

CHECKS = {
    "E.1": E1_premium_worst5,
    "E.2": E2_reconcile_all,
    "E.3": E3_unhelpful_sanity,
    "E.4": E4_response_lengths,
    "E.5": E5_criterion_and_theme,
    "E.6": E6_score_shape,
    "E.7": E7_negative_criteria_spotcheck,
    "E.8": E8_degenerate_prompts,
    "E.9": E9_rubric_fidelity,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="all", help="One of: " + ", ".join(CHECKS) + ", or 'all'")
    args = ap.parse_args()
    if args.check == "all":
        results = {}
        for name, fn in CHECKS.items():
            print()
            results[name] = fn()
        print()
        print("=" * 78)
        print("ALL CHECKS COMPLETE")
        print("=" * 78)
    elif args.check in CHECKS:
        CHECKS[args.check]()
    else:
        raise SystemExit(f"Unknown --check {args.check!r}; available: {list(CHECKS)}")


if __name__ == "__main__":
    main()
