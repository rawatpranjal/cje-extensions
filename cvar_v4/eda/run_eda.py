"""CLI entrypoint for the cvar_v4 EDA harness.

Usage:
    python -m cvar_v4.eda.run_eda --dataset harmbench_strongreject
    python -m cvar_v4.eda.run_eda --dataset healthbench --max-rows 50000
    python -m cvar_v4.eda.run_eda --concat   # rebuild scoping_data.md from sections/

Polars-only; cleanup runs after each dataset.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa

from . import _utils as U
from . import recipe
from .datasets import _registry, get as get_spec
from .datasets._base import DatasetSpec


def _load_polars(spec: DatasetSpec, max_rows: Optional[int]) -> tuple[pl.DataFrame, dict[str, int]]:
    """Load → convert to polars. Returns (df, splits_info).

    spec.custom_loader (if set) handles non-HF sources and returns a polars frame
    directly; in that case splits_info is {}.
    """
    if spec.custom_loader is not None:
        df = spec.custom_loader(spec)
        if max_rows:
            df = df.head(max_rows)
        return df, {}

    from datasets import load_dataset  # noqa: F401 — defer import

    if spec.streaming:
        ds = load_dataset(spec.hf_id, **spec.load_kwargs, streaming=True)
        # If a single split returned, ds is IterableDataset; if dict-like, take the first split.
        if hasattr(ds, "items"):
            split = next(iter(ds.keys()))
            ds = ds[split]
        rows: list[dict] = []
        target = max_rows or 100_000
        for i, row in enumerate(ds):
            if i >= target:
                break
            rows.append(row)
        df = pl.from_dicts(rows) if rows else pl.DataFrame()
        return df, {}

    ds = load_dataset(spec.hf_id, **spec.load_kwargs)
    splits_info: dict[str, int] = {}
    if hasattr(ds, "items"):
        # multi-split DatasetDict
        for name in ds:
            splits_info[name] = ds[name].num_rows
        # pick the canonical split: prefer "train" else "test" else first
        primary = "train" if "train" in ds else ("test" if "test" in ds else next(iter(ds)))
        ds_main = ds[primary]
    else:
        splits_info = {"(single)": ds.num_rows}
        ds_main = ds

    if max_rows and ds_main.num_rows > max_rows:
        ds_main = ds_main.select(range(max_rows))

    # Arrow → polars (no pandas)
    table = ds_main.data.table if hasattr(ds_main.data, "table") else ds_main.data
    if not isinstance(table, pa.Table):
        table = pa.table(table)
    df = pl.from_arrow(table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    return df, splits_info


def _cache_size_for(slug: str) -> int:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.exists():
        return 0
    total = 0
    for entry in cache.iterdir():
        if entry.name.startswith("datasets--") and slug.lower() in entry.name.lower():
            for root, _, files in __import__("os").walk(entry):
                for f in files:
                    try:
                        total += __import__("os").path.getsize(__import__("os").path.join(root, f))
                    except OSError:
                        continue
    return total


def run_one(slug: str, max_rows: Optional[int] = None, resume: bool = True) -> Path:
    spec = get_spec(slug)
    out_path = U.SECTIONS_DIR / f"{spec.order:02d}_{spec.slug}.md"
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {out_path} already exists; pass --no-resume to regenerate", file=sys.stderr)
        return out_path

    U.SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    U.require_disk_gib(2.0)

    print(f"[load] {slug} (HF: {spec.hf_id}, streaming={spec.streaming}, max_rows={max_rows or spec.max_rows})", file=sys.stderr)
    t0 = time.time()
    df_raw, splits_info = _load_polars(spec, max_rows or spec.max_rows)
    t_load = time.time() - t0
    print(f"[load] done in {t_load:.1f}s; raw shape = {df_raw.shape}", file=sys.stderr)

    df = spec.usable_filter(df_raw) if spec.usable_filter else df_raw

    on_disk = _cache_size_for(slug)
    ctx = recipe.Ctx(
        spec=spec,
        df=df,
        df_raw=df_raw,
        splits_info=splits_info,
        on_disk_bytes=on_disk,
    )

    print(f"[blocks] running 20 blocks on n={df.height:,}", file=sys.stderr)
    t0 = time.time()
    section_md = recipe.run_all(ctx)
    print(f"[blocks] done in {time.time() - t0:.1f}s", file=sys.stderr)

    out_path.write_text(section_md)
    print(f"[write] {out_path}  ({len(section_md):,} chars)", file=sys.stderr)

    # cleanup unless streaming (cache was minimal)
    if not spec.streaming:
        # Derive cache patterns from the HF id (handles cases like
        # `lmarena-ai/arena-human-preference-140k` where the slug
        # `arena_140k` doesn't match the on-disk folder).
        extra: list[str] = []
        if "/" in spec.hf_id:
            extra.append(spec.hf_id.split("/", 1)[1].split()[0])
            extra.append(spec.hf_id.replace("/", "--"))
        freed = U.cleanup_hf_cache(slug, extra_patterns=extra)
        print(f"[cleanup] freed {U.fmt_mib(freed)} from HF cache", file=sys.stderr)
    free = U.disk_free_gib()
    print(f"[disk] free now: {free:.2f} GiB", file=sys.stderr)
    return out_path


def concat_to_scoping() -> Path:
    """Rebuild cvar_v4/scoping_data.md from sections/*.md plus parts.PART_1 + PART_3."""
    from . import parts
    sections = sorted(U.SECTIONS_DIR.glob("*.md"))
    if not sections:
        raise SystemExit(f"No sections in {U.SECTIONS_DIR}")
    out_parts = [
        "# CVaR-CJE Dataset Scoping (cvar_v4)\n",
        "_Generated 2026-04-28 by `cvar_v4/eda/run_eda.py --concat`. The 10 per-dataset sections in Part 2 "
        "are produced by `run_eda.py --dataset <slug>`; Parts 1 and 3 (executive table and recommended "
        "pipeline + pre-registered hypothesis) are hand-written in `cvar_v4/eda/parts.py` based on the "
        "diagnostics computed in Part 2._\n",
        parts.PART_1,
        "## Part 2 — Per-dataset deep dives\n",
    ]
    for s in sections:
        out_parts.append(s.read_text())
    out_parts.append(parts.PART_3)
    out = "\n".join(out_parts)
    U.SCOPING_PATH.write_text(out)
    print(f"[concat] wrote {U.SCOPING_PATH}  ({len(out):,} chars, {len(sections)} sections)", file=sys.stderr)
    return U.SCOPING_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="Dataset slug to run; see registry")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true", help="Re-run even if section file exists")
    ap.add_argument("--concat", action="store_true", help="Rebuild scoping_data.md from sections/")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for slug in sorted(_registry()):
            spec = get_spec(slug)
            print(f"  {spec.order:02d}  {slug}  ({spec.hf_id})  — {spec.role_short}")
        return
    if args.concat:
        concat_to_scoping()
        return
    if args.dataset:
        run_one(args.dataset, max_rows=args.max_rows, resume=not args.no_resume)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
