"""Shared utilities for the cvar_v4 EDA harness. Polars-only; no pandas."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import polars as pl
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = REPO_ROOT / "cvar_v4" / "eda"
SECTIONS_DIR = EDA_DIR / "sections"
SCOPING_PATH = REPO_ROOT / "cvar_v4" / "scoping_data.md"

_ENC = None


def encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding(name)
    return _ENC


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len(encoding().encode(text, disallowed_special=()))


def disk_free_gib(path: str = "/") -> float:
    s = shutil.disk_usage(path)
    return s.free / (1024**3)


def require_disk_gib(min_gib: float = 2.0) -> None:
    free = disk_free_gib()
    if free < min_gib:
        raise RuntimeError(f"Insufficient disk: {free:.2f} GiB free, need {min_gib} GiB")


def cleanup_hf_cache(slug_pattern: str, extra_patterns: list[str] | None = None) -> int:
    """rm -rf ~/.cache/huggingface/hub/datasets--*<slug_pattern>*. Returns bytes freed.

    Matches both dashed and underscored variants since HF normalizes dataset
    repo names to dashed form on disk while our slugs use underscores.
    Extra patterns let a spec name additional cache prefixes (e.g., the
    HarmBench module also pulls walledai/StrongREJECT).
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.exists():
        return 0
    candidates = {slug_pattern.lower(), slug_pattern.lower().replace("_", "-")}
    candidates.update(p.lower() for p in (extra_patterns or []))
    freed = 0
    for entry in cache.iterdir():
        if not entry.name.startswith("datasets--"):
            continue
        name_l = entry.name.lower()
        if any(p in name_l for p in candidates):
            try:
                size = _du_bytes(entry)
                shutil.rmtree(entry, ignore_errors=True)
                freed += size
            except OSError:
                pass
    return freed


def _du_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def fmt_gib(b: float) -> str:
    return f"{b / (1024**3):.2f} GiB"


def fmt_mib(b: float) -> str:
    return f"{b / (1024**2):.1f} MiB"


def truncate(s: str | None, n: int = 300) -> str:
    if s is None:
        return "<null>"
    s = str(s).replace("\n", " ").replace("\r", " ")
    if len(s) > n:
        return s[:n] + " …"
    return s


def ascii_histogram(values: Sequence[float], bins: int = 12, width: int = 40) -> str:
    """Inline ASCII histogram. Skips NaNs."""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return "(empty)"
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return f"all values = {lo:.4g} (n={arr.size})"
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    peak = counts.max() or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  [{edges[i]:>8.4g}, {edges[i+1]:>8.4g}) | {bar} {c}")
    return "\n".join(lines)


def quantiles(s: pl.Series, qs: Sequence[float]) -> dict[float, float]:
    return {q: float(s.quantile(q, interpolation="linear")) for q in qs}


def tie_fraction_at_quantile(s: pl.Series, q: float, eps_rel: float = 0.005) -> tuple[float, float, int]:
    """Return (q_value, tie_fraction, tie_count) — fraction of rows within ±eps_rel·range of q-th quantile."""
    qv = float(s.quantile(q, interpolation="linear"))
    rng = float(s.max()) - float(s.min())
    eps = max(rng * eps_rel, 1e-12)
    n_tie = int(s.filter((s >= qv - eps) & (s <= qv + eps)).len())
    return qv, n_tie / max(s.len(), 1), n_tie


def cvar_estimate(y: pl.Series, alpha: float) -> tuple[float, float]:
    """Empirical CVaR_α and quantile q_α (lower tail: E[Y | Y ≤ q_α])."""
    q = float(y.quantile(alpha, interpolation="linear"))
    tail_mean = float(y.filter(y <= q).mean()) if y.filter(y <= q).len() > 0 else float("nan")
    return tail_mean, q


def _empirical_cvar(y_arr: np.ndarray, alpha: float) -> float:
    if y_arr.size == 0:
        return float("nan")
    q = float(np.quantile(y_arr, alpha))
    tail = y_arr[y_arr <= q]
    return float(np.mean(tail)) if tail.size > 0 else float("nan")


def cvar_ci_halfwidth(y_arr: np.ndarray, alpha: float, n: int, ci: float = 0.95, B: int = 500, seed: int = 42) -> float:
    """Bootstrap-based 95% CI half-width for CVaR_α.

    Used in place of the closed-form influence-function formula because
    the IF variance becomes negative at the empirical quantile when Y
    has mass points (ties at q_α — see block 9's tie analysis).
    Bootstrap is robust to that.
    """
    if y_arr.size < 50:
        return float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, y_arr.size, size=y_arr.size)
        boots[b] = _empirical_cvar(y_arr[idx], alpha)
    if not np.isfinite(boots).any():
        return float("nan")
    lo = float(np.nanquantile(boots, (1 - ci) / 2))
    hi = float(np.nanquantile(boots, 1 - (1 - ci) / 2))
    return float((hi - lo) / 2)


def bootstrap_ci(y_arr: np.ndarray, fn: Callable[[np.ndarray], float], B: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = y_arr.size
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = fn(y_arr[idx])
    return float(np.nanquantile(boots, alpha / 2)), float(np.nanquantile(boots, 1 - alpha / 2))


def md_table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return f"| {' | '.join(headers)} |\n|{'|'.join(['---'] * len(headers))}|\n| (no rows) |"
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def df_describe(df: pl.DataFrame) -> list[dict]:
    """Per-column summary: name, dtype, %-null, %-empty (string only), distinct (capped)."""
    n = df.height
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        null_pct = (s.null_count() / max(n, 1)) * 100
        empty_pct = 0.0
        if s.dtype == pl.Utf8:
            try:
                empty_pct = (s.str.len_chars() == 0).sum() / max(n, 1) * 100
            except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
                empty_pct = 0.0
        try:
            n_unique = s.n_unique()
        except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
            n_unique = -1
        rows.append({"col": col, "dtype": dtype, "null_pct": null_pct, "empty_pct": empty_pct, "n_unique": n_unique})
    return rows
