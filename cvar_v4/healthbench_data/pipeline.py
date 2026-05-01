"""End-to-end orchestrator: prompts → generate × 5 policies → judge cheap × 5 → judge oracle × 5
   → assemble CJE-format JSONL files.

Output layout (matches cvar_v3 / original Arena format):
    cvar_v4/healthbench_data/data/
        prompts.jsonl                      # extracted prompts + rubrics
        cje_dataset.jsonl                  # logger π0 file (one row per prompt)
        responses/
            base_responses.jsonl
            clone_responses.jsonl
            premium_responses.jsonl
            parallel_universe_prompt_responses.jsonl
            unhelpful_responses.jsonl

Usage:
    python3 -m cvar_v4.healthbench_data.pipeline --pilot 20
    python3 -m cvar_v4.healthbench_data.pipeline --full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .generate import generate_all, generate_all_megabatch
from .judge import grade_all, grade_all_megabatch, _judge_path
from .policies import POLICIES, logger_policy
from .prompts import build_prompts, load_prompts, PROMPTS_PATH

DATA_DIR = Path(__file__).parent / "data"
RESPONSES_DIR = DATA_DIR / "responses"
LOGGER_FILE = DATA_DIR / "cje_dataset.jsonl"


def _join_score(judge_jsonl: Path) -> dict[str, tuple[float, int]]:
    """Read a judge output file, return {prompt_id: (score, n_criteria)}."""
    out = {}
    if not judge_jsonl.exists():
        return out
    with judge_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            out[d["prompt_id"]] = (d.get("score"), d.get("n_criteria", 0))
    return out


def assemble_cje_jsonl():
    """Build the final CJE-format files, joining responses with cheap+oracle scores."""
    from .policies import JUDGE_CHEAP_MODEL, JUDGE_ORACLE_MODEL

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Per-policy: read responses, join with cheap+oracle scores, write final responses_with_scores
    for p in POLICIES:
        resp_in = RESPONSES_DIR / f"{p.name}_responses.jsonl"
        if not resp_in.exists():
            print(f"[assemble] skip {p.name}: no responses file")
            continue
        cheap_scores = _join_score(_judge_path(p.name, "cheap"))
        oracle_scores = _join_score(_judge_path(p.name, "oracle"))
        # Read responses, attach scores, rewrite to a temp file then swap
        tmp = resp_in.with_suffix(".tmp")
        n_total = n_with_cheap = n_with_oracle = 0
        with resp_in.open() as fin, tmp.open("w") as fout:
            for line in fin:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                pid = d["prompt_id"]
                cs = cheap_scores.get(pid)
                os_ = oracle_scores.get(pid)
                meta = d.setdefault("metadata", {})
                if cs is not None:
                    meta["judge_score"] = cs[0]
                    meta["judge_model"] = JUDGE_CHEAP_MODEL
                    n_with_cheap += 1
                if os_ is not None:
                    meta["oracle_label"] = os_[0]
                    meta["oracle_model"] = JUDGE_ORACLE_MODEL
                    n_with_oracle += 1
                fout.write(json.dumps(d) + "\n")
                n_total += 1
        tmp.replace(resp_in)
        print(f"[assemble] {p.name}: {n_total} responses, {n_with_cheap} with cheap-S, {n_with_oracle} with oracle-Y")

    # Build the logger file: one row per prompt with the logger's response + scores
    logger = logger_policy()
    logger_resp_path = RESPONSES_DIR / f"{logger.name}_responses.jsonl"
    if not logger_resp_path.exists():
        print(f"[assemble] WARN: {logger_resp_path} missing; cje_dataset.jsonl not built")
        return
    n_logger = 0
    with logger_resp_path.open() as fin, LOGGER_FILE.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            row = {
                "prompt": d["prompt"],
                "response": d["response"],
                "base_policy_logprob": None,  # OpenAI doesn't return TF logprobs by default
                "target_policy_logprobs": {},
                "metadata": {
                    "prompt_id": d["prompt_id"],
                    "judge_score": d.get("metadata", {}).get("judge_score"),
                    "oracle_label": d.get("metadata", {}).get("oracle_label"),
                    "judge_model": d.get("metadata", {}).get("judge_model"),
                    "oracle_model": d.get("metadata", {}).get("oracle_model"),
                    "theme": d.get("metadata", {}).get("theme"),
                    "subset": d.get("metadata", {}).get("subset"),
                },
            }
            fout.write(json.dumps(row) + "\n")
            n_logger += 1
    print(f"[assemble] wrote {LOGGER_FILE} ({n_logger} rows)")


def run(limit: int | None = None, kind_filter: str | None = None,
        batch: bool = True, megabatch: bool = True):
    """End-to-end run.

    megabatch=True (default): submit ONE batch for all policies' generation,
    then ONE batch for all (policy, prompt, criterion, kind=cheap+oracle)
    grading. Wall-clock ≈ 2x batch round-trip regardless of n_policies.

    megabatch=False: per-policy batches (n_policies × n_phases sequential).
    """
    t0 = time.time()
    print("=" * 60)
    flow = "megabatch" if megabatch else ("per-policy batch" if batch else "sync")
    print(f"[pipeline] start: limit={limit}, kind_filter={kind_filter}, flow={flow}")
    print("=" * 60)

    # Step 1: prompts (free)
    if not PROMPTS_PATH.exists():
        print("[step 1/3] building prompts.jsonl...")
        build_prompts(limit=None)
    else:
        print(f"[step 1/3] prompts.jsonl exists ({load_prompts().height:,} rows)")

    if megabatch:
        # Step 2: ONE batch for all generations
        print(f"\n[step 2/3] generate-all (mega-batch, limit={limit})...")
        generate_all_megabatch(limit=limit)

        # Step 3: ONE batch for cheap+oracle grading combined
        kinds = []
        if kind_filter in (None, "cheap"):
            kinds.append("cheap")
        if kind_filter in (None, "oracle"):
            kinds.append("oracle")
        print(f"\n[step 3/3] grade-all (mega-batch, kinds={kinds})...")
        grade_all_megabatch(kinds=kinds)
    else:
        # Per-policy fallback
        print(f"\n[step 2/4] generating responses (limit={limit})...")
        generate_all(limit=limit, batch=batch)
        if kind_filter in (None, "cheap"):
            print("\n[step 3/4] cheap-judge S grading on all 5 policies...")
            grade_all(kind="cheap", batch=batch)
        if kind_filter in (None, "oracle"):
            print("\n[step 4/4] oracle-judge Y grading on all 5 policies...")
            grade_all(kind="oracle", batch=batch)

    # Assemble final files
    print("\n[assemble] joining scores into CJE-format files...")
    assemble_cje_jsonl()

    print(f"\n[pipeline] done in {time.time()-t0:.0f}s")
    print(f"[output] {DATA_DIR}/cje_dataset.jsonl")
    print(f"[output] {RESPONSES_DIR}/<policy>_responses.jsonl")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, help="Pilot run with N prompts")
    g.add_argument("--full", action="store_true", help="Full 5K-prompt run")
    g.add_argument("--assemble-only", action="store_true", help="Just re-join existing files")
    ap.add_argument("--kind", choices=["cheap", "oracle"], default=None,
                    help="Skip the other grading kind (e.g., --kind cheap to do only cheap-S)")
    ap.add_argument("--sync", action="store_true",
                    help="Use sync per-call API instead of Batch API (default: batch)")
    ap.add_argument("--per-policy", action="store_true",
                    help="Use per-policy batches instead of mega-batch (default: mega-batch)")
    args = ap.parse_args()
    use_batch = not args.sync
    use_megabatch = use_batch and not args.per_policy
    if args.assemble_only:
        assemble_cje_jsonl()
    elif args.full:
        run(limit=None, kind_filter=args.kind, batch=use_batch, megabatch=use_megabatch)
    else:
        run(limit=args.pilot, kind_filter=args.kind, batch=use_batch, megabatch=use_megabatch)


if __name__ == "__main__":
    main()
