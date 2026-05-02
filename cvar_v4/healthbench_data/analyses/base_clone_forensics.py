"""Base versus clone tail forensics.

Identifies the prompts in the bottom-k oracle rows of each policy and
flags those unique to one or the other.

Why a separate diagnostic for this pair:
    base and clone share the same generator (gpt-4o-mini with the same
    prompt template); they should be statistically indistinguishable
    on every estimand if the policy panel is well-constructed and the
    cheap/oracle judges are stable. A real base-vs-clone difference
    in the tail signals one of: (a) genuine sampling variation across
    temperature seeds, (b) judge noise on borderline criteria, or
    (c) a bug in the policy panel. Listing the prompts that appear
    ONLY in one bottom-k and not the other lets the analyst eyeball
    which is happening. The ideal output is heavy overlap (most
    prompts shared) — that's the shared-generator failure-mode
    signature, not real policy divergence.

Outputs:
    writeup/data/base_clone.json
"""
from __future__ import annotations

import argparse

from ._common import (load_oracle_scores, load_prompts, panel_size,
                       write_json)


def compute(alpha: float = 0.10, top_k_default: int = 10) -> dict:
    n = panel_size()
    k = max(top_k_default, int(round(alpha * n)))

    prompts = load_prompts()
    b_oracle = load_oracle_scores("base")
    c_oracle = load_oracle_scores("clone")

    def bottom(scores: dict[str, dict]) -> list[tuple[str, float]]:
        rows = [(pid, r["score"]) for pid, r in scores.items()
                if r.get("score") is not None]
        rows.sort(key=lambda x: x[1])
        return rows[:k]

    base_b = bottom(b_oracle)
    clone_b = bottom(c_oracle)
    base_set = {pid for pid, _ in base_b}
    clone_set = {pid for pid, _ in clone_b}
    common = base_set & clone_set
    base_only = base_set - clone_set
    clone_only = clone_set - base_set

    def row_for(pid: str) -> dict:
        prompt_text = prompts.get(pid, {}).get("prompt_text", "")[:100]
        return {
            "prompt_id": pid,
            "prompt_short": prompt_text,
            "base_Y": b_oracle[pid]["score"] if pid in b_oracle else None,
            "clone_Y": c_oracle[pid]["score"] if pid in c_oracle else None,
            "delta": (c_oracle[pid]["score"] - b_oracle[pid]["score"])
                      if pid in b_oracle and pid in c_oracle else None,
        }

    # Top |Δ| disagreements anywhere on the panel
    common_pids = set(b_oracle) & set(c_oracle)
    diffs = []
    for pid in common_pids:
        by = b_oracle[pid].get("score")
        cy = c_oracle[pid].get("score")
        if by is None or cy is None:
            continue
        diffs.append((pid, abs(cy - by)))
    diffs.sort(key=lambda x: -x[1])
    top_disagreements = [row_for(pid) for pid, _ in diffs[:8]]

    return {
        "alpha": alpha,
        "k": k,
        "panel_size": n,
        "common_in_both_bottoms": [row_for(pid) for pid in sorted(common)],
        "base_only": [row_for(pid) for pid in sorted(base_only)],
        "clone_only": [row_for(pid) for pid in sorted(clone_only)],
        "top_disagreements": top_disagreements,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()
    payload = compute(alpha=args.alpha)
    path = write_json("base_clone.json", payload)
    print(f"[base_clone] α={payload['alpha']}  bottom-{payload['k']} per policy")
    print(f"[base_clone] common in both bottoms: {len(payload['common_in_both_bottoms'])}")
    print(f"[base_clone] base only: {len(payload['base_only'])}, "
          f"clone only: {len(payload['clone_only'])}")
    if payload["clone_only"]:
        print(f"[base_clone] clone-only prompts:")
        for r in payload["clone_only"]:
            print(f"  {r['prompt_id'][:8]} base={r['base_Y']:+.3f} clone={r['clone_Y']:+.3f} Δ={r['delta']:+.3f}")
    print(f"[base_clone] wrote {path}")


if __name__ == "__main__":
    main()
