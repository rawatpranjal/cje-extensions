"""Shared loaders and styling for the analyses package.

Every diagnostic module imports from here. Loaders read whatever rows are on
disk in `data/responses/` and `judge_outputs/`, so nothing here is hard-coded
to n=100.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Paths
PKG_ROOT = Path(__file__).resolve().parent.parent          # cvar_v4/healthbench_data/
DATA_DIR = PKG_ROOT / "data"
RESP_DIR = DATA_DIR / "responses"
JUDGE_DIR = PKG_ROOT / "judge_outputs"
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"

WRITEUP_DIR = PKG_ROOT / "writeup"
WRITEUP_DATA_DIR = WRITEUP_DIR / "data"
WRITEUP_FIG_DIR = WRITEUP_DIR  # figures land directly under writeup/ to match \includegraphics paths

POLICIES = ["base", "clone", "premium", "parallel_universe_prompt", "unhelpful", "risky"]
LOGGER = "base"

LABELS = {
    "base": "base",
    "clone": "clone",
    "premium": "premium",
    "parallel_universe_prompt": "parallel",
    "unhelpful": "unhelpful",
    "risky": "risky",
}

# Plot styling: colorblind-safe palette, distinct marker per policy
COLORS = {
    "base":                     "#1f77b4",
    "clone":                    "#7fa8c9",
    "premium":                  "#2ca02c",
    "parallel_universe_prompt": "#ff7f0e",
    "unhelpful":                "#d62728",
    "risky":                    "#9467bd",
}
MARKERS = {
    "base": "o",
    "clone": "s",
    "premium": "^",
    "parallel_universe_prompt": "D",
    "unhelpful": "v",
    "risky": "P",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def load_prompts() -> dict[str, dict]:
    return {p["prompt_id"]: p for p in _read_jsonl(PROMPTS_PATH)}


def load_oracle_scores(policy: str) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in _read_jsonl(JUDGE_DIR / f"{policy}_oracle.jsonl")}


def load_cheap_scores(policy: str) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in _read_jsonl(JUDGE_DIR / f"{policy}_cheap.jsonl")}


def load_responses(policy: str) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in _read_jsonl(RESP_DIR / f"{policy}_responses.jsonl")}


def panel_size() -> int:
    """Auto-detect n from the logger's responses file."""
    rows = _read_jsonl(RESP_DIR / f"{LOGGER}_responses.jsonl")
    return len(rows)


def write_json(name: str, payload) -> Path:
    """Write a JSON artifact to writeup/data/{name}.json."""
    WRITEUP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = WRITEUP_DATA_DIR / name
    if not name.endswith(".json"):
        path = path.with_suffix(".json")
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def fig_path(name: str) -> Path:
    WRITEUP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return WRITEUP_FIG_DIR / name


def policy_pairs(policy: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (prompt_ids, cheap_scores, oracle_scores) for one policy, only
    rows where both scores are non-null. Sorted by prompt_id."""
    cheap = load_cheap_scores(policy)
    oracle = load_oracle_scores(policy)
    common = sorted(
        pid for pid in set(cheap) & set(oracle)
        if cheap[pid].get("score") is not None and oracle[pid].get("score") is not None
    )
    s = np.array([cheap[pid]["score"] for pid in common])
    y = np.array([oracle[pid]["score"] for pid in common])
    return common, s, y


def all_pairs_pooled() -> list[tuple[str, str, float, float]]:
    """Pool (policy, prompt_id, cheap, oracle) rows across all five policies."""
    out = []
    for p in POLICIES:
        pids, s, y = policy_pairs(p)
        for pid, si, yi in zip(pids, s, y):
            out.append((p, pid, float(si), float(yi)))
    return out


def cvar_alpha(arr: np.ndarray, alpha: float) -> float:
    """Atom-split lower-tail CVaR_α. Same definition as analyze.cvar_alpha."""
    if arr.size == 0:
        return float("nan")
    n = arr.size
    sorted_y = np.sort(arr)
    target_mass = alpha * n
    k = int(np.floor(target_mass))
    remainder = target_mass - k
    if k == 0:
        return float(sorted_y[0])
    if k >= n:
        return float(sorted_y.mean())
    head_sum = float(sorted_y[:k].sum())
    if remainder > 0:
        head_sum += remainder * float(sorted_y[k])
    return head_sum / target_mass
