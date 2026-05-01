"""End-to-end data integrity check for the n=N HealthBench panel.

Run after `pipeline.py` produces `data/` and `judge_outputs/`. Reports any
silent failures the pipeline cannot detect on its own (length skews, NaNs,
stuck oracle verdicts, prompt-set drift across policies, etc.).

Output:
  - prints a tabular report to stdout
  - writes `writeup/data/integrity_check.json` with the same content
  - exits non-zero if any HARD check fails

Hard checks (non-zero exit):
  - row count mismatch between responses, judge outputs, and aggregated files
  - any NaN / non-finite Y or S
  - per-policy prompt-id sets disagree (panel must be aligned)

Soft checks (printed warnings, exit 0):
  - all-N rows (oracle said N to every criterion → Y at floor)
  - cheap-oracle Pearson correlation outside [0.2, 0.95]
  - per-policy mean Y outside priors from CLAUDE.md
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESPONSES_DIR = ROOT / "data" / "responses"
JUDGE_DIR = ROOT / "judge_outputs"
OUT_PATH = ROOT / "writeup" / "data" / "integrity_check.json"

POLICIES = ["base", "clone", "premium", "parallel_universe_prompt", "unhelpful"]
KINDS = ["cheap", "oracle"]

# Priors from CLAUDE.md (Mean CJE benchmark; reasonable bounds for our Y).
PRIOR_MEAN_Y = {
    "base": (0.0, 0.6),
    "clone": (0.0, 0.6),
    "premium": (0.1, 0.7),
    "parallel_universe_prompt": (-0.2, 0.4),
    "unhelpful": (-0.5, 0.1),
}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = st.fmean(xs)
    my = st.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
    return s[idx]


def main() -> int:
    hard_fails: list[str] = []
    soft_warns: list[str] = []
    report: dict = {"policies": {}, "panel": {}}

    print("=" * 72)
    print(f"[eda] integrity check for {ROOT.name}")
    print("=" * 72)

    # ---- Per-policy load and per-policy stats ----
    n_per_policy: dict[str, int] = {}
    pid_sets: dict[str, set[str]] = {}

    for pol in POLICIES:
        prow: dict = {"counts": {}, "y_stats": {}, "s_stats": {}}

        # responses
        rpath = RESPONSES_DIR / f"{pol}_responses.jsonl"
        if not rpath.exists():
            hard_fails.append(f"missing {rpath}")
            continue
        rows = load_jsonl(rpath)
        prow["counts"]["responses"] = len(rows)

        # cheap & oracle aggregated per-prompt
        score_rows: dict[str, dict[str, dict]] = {"cheap": {}, "oracle": {}}
        for kind in KINDS:
            jp = JUDGE_DIR / f"{pol}_{kind}.jsonl"
            jr = JUDGE_DIR / f"{pol}_{kind}_raw.jsonl"
            if not jp.exists():
                hard_fails.append(f"missing {jp}")
                continue
            if not jr.exists():
                hard_fails.append(f"missing {jr}")
                continue
            jrows = load_jsonl(jp)
            score_rows[kind] = {r["prompt_id"]: r for r in jrows}
            prow["counts"][kind] = len(jrows)
            prow["counts"][f"{kind}_raw"] = sum(1 for _ in jr.open())

        if not score_rows["cheap"] or not score_rows["oracle"]:
            continue

        if len(rows) != len(score_rows["cheap"]) or len(rows) != len(score_rows["oracle"]):
            hard_fails.append(
                f"{pol}: row mismatch responses={len(rows)} "
                f"cheap={len(score_rows['cheap'])} oracle={len(score_rows['oracle'])}"
            )

        pid_sets[pol] = {r["prompt_id"] for r in rows}
        n_per_policy[pol] = len(rows)

        # Build aligned (S, Y, n_criteria, response_length) arrays
        s_arr: list[float] = []
        y_arr: list[float] = []
        n_crit_arr: list[int] = []
        resp_len_arr: list[int] = []
        all_n_count = 0  # rows where oracle said N to every criterion
        nan_count = 0

        for resp in rows:
            pid = resp["prompt_id"]
            cheap = score_rows["cheap"].get(pid)
            oracle = score_rows["oracle"].get(pid)
            if cheap is None or oracle is None:
                hard_fails.append(f"{pol}: no scores for prompt {pid}")
                continue
            s = cheap["score"]
            y = oracle["score"]
            if not (math.isfinite(s) and math.isfinite(y)):
                nan_count += 1
                continue
            s_arr.append(s)
            y_arr.append(y)
            n_crit_arr.append(oracle["n_criteria"])
            resp_len_arr.append(len(resp["response"]))
            verds = oracle.get("verdicts") or []
            # All-N: every verdict is N (no criteria satisfied)
            if verds and all(v == "N" for v in verds):
                all_n_count += 1

        if nan_count:
            hard_fails.append(f"{pol}: {nan_count} non-finite Y/S")

        # Stats
        prow["y_stats"] = {
            "mean": st.fmean(y_arr) if y_arr else float("nan"),
            "median": st.median(y_arr) if y_arr else float("nan"),
            "min": min(y_arr) if y_arr else float("nan"),
            "max": max(y_arr) if y_arr else float("nan"),
            "p10": quantile(y_arr, 0.10),
            "p25": quantile(y_arr, 0.25),
            "p75": quantile(y_arr, 0.75),
            "p90": quantile(y_arr, 0.90),
        }
        prow["s_stats"] = {
            "mean": st.fmean(s_arr) if s_arr else float("nan"),
            "median": st.median(s_arr) if s_arr else float("nan"),
            "min": min(s_arr) if s_arr else float("nan"),
            "max": max(s_arr) if s_arr else float("nan"),
        }
        prow["pearson_s_y"] = pearson(s_arr, y_arr)
        prow["all_n_oracle_rows"] = all_n_count
        prow["mean_response_chars"] = st.fmean(resp_len_arr) if resp_len_arr else 0
        prow["mean_n_criteria"] = st.fmean(n_crit_arr) if n_crit_arr else 0
        prow["nan_count"] = nan_count

        # Soft checks
        lo, hi = PRIOR_MEAN_Y.get(pol, (-1.0, 1.0))
        if y_arr and not (lo <= prow["y_stats"]["mean"] <= hi):
            soft_warns.append(
                f"{pol}: mean Y={prow['y_stats']['mean']:+.3f} outside prior [{lo}, {hi}]"
            )
        corr = prow["pearson_s_y"]
        if math.isfinite(corr) and not (0.2 <= corr <= 0.95):
            soft_warns.append(
                f"{pol}: Pearson(S,Y)={corr:+.3f} outside [0.20, 0.95]"
            )
        if all_n_count > 0.05 * len(rows):
            soft_warns.append(
                f"{pol}: {all_n_count}/{len(rows)} rows have all-N oracle "
                f"({100 * all_n_count / len(rows):.1f}%)"
            )

        report["policies"][pol] = prow

    # ---- Panel-level checks ----
    if pid_sets:
        common = set.intersection(*pid_sets.values()) if pid_sets else set()
        per_pol_extra = {p: sorted(pid_sets[p] - common)[:3] for p in pid_sets}
        for pol, extra in per_pol_extra.items():
            if extra:
                hard_fails.append(
                    f"{pol}: {len(pid_sets[pol] - common)} prompts not in panel intersection "
                    f"(e.g. {extra[:3]})"
                )
        report["panel"]["n_common_prompts"] = len(common)
        report["panel"]["n_per_policy"] = n_per_policy

    # cje_dataset.jsonl size
    cje = ROOT / "data" / "cje_dataset.jsonl"
    if cje.exists():
        report["panel"]["cje_dataset_rows"] = sum(1 for _ in cje.open())

    # ---- Print report ----
    print()
    print(f"  panel intersection size: {report['panel'].get('n_common_prompts', '?')}")
    print(f"  cje_dataset.jsonl rows:  {report['panel'].get('cje_dataset_rows', '?')}")
    print()
    hdr = (f"  {'policy':<26} {'n_resp':>6} {'cheap':>6} {'oracle':>6} "
           f"{'mean_Y':>7} {'min_Y':>7} {'max_Y':>7} {'corr':>6} "
           f"{'all_N':>5} {'avg_len':>8} {'crits':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pol in POLICIES:
        if pol not in report["policies"]:
            print(f"  {pol:<26}  (no data)")
            continue
        p = report["policies"][pol]
        c = p["counts"]
        ys = p["y_stats"]
        print(f"  {pol:<26} "
              f"{c.get('responses', 0):>6} "
              f"{c.get('cheap', 0):>6} "
              f"{c.get('oracle', 0):>6} "
              f"{ys.get('mean', float('nan')):>+7.3f} "
              f"{ys.get('min', float('nan')):>+7.2f} "
              f"{ys.get('max', float('nan')):>+7.2f} "
              f"{p.get('pearson_s_y', float('nan')):>+6.2f} "
              f"{p.get('all_n_oracle_rows', 0):>5} "
              f"{p.get('mean_response_chars', 0):>8.0f} "
              f"{p.get('mean_n_criteria', 0):>6.1f}")

    # Tail check: bottom 10% Y per policy (the CVaR_0.10 region)
    print()
    print("  bottom-10% Y per policy (the α=0.10 tail):")
    for pol in POLICIES:
        if pol not in report["policies"]:
            continue
        p = report["policies"][pol]
        print(f"    {pol:<26} p10={p['y_stats']['p10']:+.3f}  "
              f"p25={p['y_stats']['p25']:+.3f}  median={p['y_stats']['median']:+.3f}")

    # ---- Verdict ----
    print()
    if hard_fails:
        print(f"  HARD FAILS ({len(hard_fails)}):")
        for f in hard_fails:
            print(f"    - {f}")
    if soft_warns:
        print(f"  SOFT WARNINGS ({len(soft_warns)}):")
        for w in soft_warns:
            print(f"    - {w}")
    if not hard_fails and not soft_warns:
        print("  ALL CHECKS PASSED.")

    report["hard_fails"] = hard_fails
    report["soft_warns"] = soft_warns

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {OUT_PATH}")

    return 1 if hard_fails else 0


if __name__ == "__main__":
    sys.exit(main())
