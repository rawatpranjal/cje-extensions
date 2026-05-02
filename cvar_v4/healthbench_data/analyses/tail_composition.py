"""Tail composition by HealthBench theme.

For each policy, identify the worst-α rows by oracle Y and tabulate which
themes they come from.

Why theme distribution in the tail is informative:
    HealthBench prompts are tagged with themes (emergency-referral,
    expertise-tailored, response-depth, etc.). If a policy's tail is
    dominated by one theme, the failure mode is THEMATIC and the
    policy might be fixable by training on more of that theme. If the
    tail is theme-uniform (the policy's worst rows look like a random
    sample of its overall theme distribution), the failure mode is
    DIFFUSE — a global quality problem, not a theme-specific one. For
    `unhelpful` we expect uniformity (it's bad everywhere); for
    `parallel_universe_prompt` we expect concentration on themes
    where the role-play prompt produces medically wrong content.
    This diagnostic does NOT gate any claim; it's a forensic step.

Outputs:
    writeup/data/tail_composition.json
"""
from __future__ import annotations

import argparse
from collections import Counter

from ._common import (LABELS, POLICIES, load_oracle_scores, load_prompts,
                       panel_size, write_json)


def compute(alpha: float = 0.10, top_k_default: int = 10) -> dict:
    prompts = load_prompts()
    n = panel_size()
    k = max(top_k_default, int(round(alpha * n)))

    out = {}
    for pol in POLICIES:
        oracle = load_oracle_scores(pol)
        scored = [(pid, r["score"]) for pid, r in oracle.items()
                  if r.get("score") is not None]
        scored.sort(key=lambda x: x[1])
        bottom = scored[:k]
        themes = Counter(prompts[pid].get("theme") for pid, _ in bottom
                          if pid in prompts)
        out[pol] = {
            "n_bottom": len(bottom),
            "themes": dict(themes),
            "rows": [
                {"prompt_id": pid, "score": y, "theme": prompts.get(pid, {}).get("theme")}
                for pid, y in bottom
            ],
        }
    return {
        "alpha": alpha,
        "k": k,
        "panel_size": n,
        "by_policy": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()
    payload = compute(alpha=args.alpha)
    path = write_json("tail_composition.json", payload)
    print(f"[tail_composition] α={args.alpha}, bottom-{payload['k']} per policy:")
    for p, info in payload["by_policy"].items():
        print(f"  {LABELS[p]:14}: {info['themes']}")
    print(f"[tail_composition] wrote {path}")


if __name__ == "__main__":
    main()
