"""Extract HealthBench prompts + rubrics into a clean polars DataFrame.

Source: HealthBench oss_eval JSONL (5,000 conversations × ~10 rubric criteria each).

Each output record has:
- prompt_id (UUID)
- prompt_text (the first user-turn content; multi-turn prompts are flattened to first-turn for the original-paper parallel)
- rubrics (list of {criterion, points, tags})
- theme (parsed from example_tags)
- subset ('oss_eval')
- n_criteria, total_positive_points (precomputed)

Output: `cvar_v4/healthbench_data/data/prompts.jsonl`
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import polars as pl

DATA_DIR = Path(__file__).parent / "data"
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"

OSS_EVAL_LOCAL = Path("/tmp/hb_oss_eval.jsonl")
OSS_EVAL_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)


def _download_if_missing() -> Path:
    if OSS_EVAL_LOCAL.exists() and OSS_EVAL_LOCAL.stat().st_size > 1_000_000:
        return OSS_EVAL_LOCAL
    print(f"[download] fetching HealthBench oss_eval to {OSS_EVAL_LOCAL}")
    OSS_EVAL_LOCAL.parent.mkdir(exist_ok=True)
    subprocess.run(
        ["curl", "-sLf", "--max-time", "120", OSS_EVAL_URL, "-o", str(OSS_EVAL_LOCAL)],
        check=True,
    )
    return OSS_EVAL_LOCAL


def _parse_record(d: dict) -> dict[str, Any]:
    prompt_turns = d.get("prompt") or []
    user_first = next((t.get("content", "") for t in prompt_turns if t.get("role") == "user"), "")
    rubrics = d.get("rubrics") or []
    pos_points = sum(r.get("points", 0) for r in rubrics if r.get("points", 0) > 0)
    tags = d.get("example_tags") or []
    theme = next((t.split(":", 1)[1] for t in tags if t.startswith("theme:")), None)
    return {
        "prompt_id": d.get("prompt_id"),
        "prompt_text": user_first,
        "rubrics": rubrics,
        "theme": theme,
        "subset": "oss_eval",
        "n_criteria": len(rubrics),
        "total_positive_points": float(pos_points),
        "n_turns": len(prompt_turns),
    }


def build_prompts(out_path: Path = PROMPTS_PATH, limit: int | None = None) -> pl.DataFrame:
    """Build the clean prompts dataset. Returns the polars DataFrame and writes JSONL."""
    src = _download_if_missing()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with src.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = _parse_record(d)
            if rec["prompt_id"] and rec["prompt_text"]:
                records.append(rec)
            if limit is not None and len(records) >= limit:
                break

    # Write JSONL (rubrics is a list-of-dicts; polars handles this fine)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[write] {out_path}  ({len(records):,} records)")

    return pl.from_dicts(records)


def load_prompts(out_path: Path = PROMPTS_PATH) -> pl.DataFrame:
    """Load the cached prompts.jsonl (build_prompts must have run first)."""
    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} doesn't exist; run build_prompts() first")
    rows = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pl.from_dicts(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Build only the first N prompts (for pilots)")
    args = ap.parse_args()
    df = build_prompts(limit=args.limit)
    print()
    print("=== Summary ===")
    print(f"Total rows: {df.height:,}")
    if "theme" in df.columns:
        print("Theme distribution:")
        for r in df.group_by("theme").agg(pl.len().alias("n")).sort("n", descending=True).iter_rows(named=True):
            print(f"  {r['theme']!s:25} {r['n']:>6,}")
    if "n_criteria" in df.columns:
        print(f"Avg criteria per prompt: {float(df['n_criteria'].mean()):.1f}")
        print(f"Avg positive points per prompt: {float(df['total_positive_points'].mean()):.1f}")
