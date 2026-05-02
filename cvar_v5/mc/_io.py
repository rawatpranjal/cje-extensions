"""
I/O helpers for MC runs.

Three concerns, kept here to keep `runner.py` and `report.py` focused on math:

    1. Timestamped run directories under `cvar_v5/mc/runs/`. Each run gets its
       own dir; results, logs, and config land together.
    2. `run_config.json` artifact: every results CSV is paired with a JSON
       describing exactly how it was produced (Config, ModeParams, git_sha,
       n_workers, timestamp).
    3. Logging setup. `print()` is removed from runner/report; both write to
       a stream handler (stderr) and an optional file handler in the run dir.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_NAME = "cvar_v5.mc"
LOG = logging.getLogger(LOG_NAME)

DEFAULT_RUNS_BASE = Path(__file__).parent / "runs"

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def make_run_dir(mode: str, base: Path | None = None) -> Path:
    """
    Create `<base>/<YYYY-MM-DDTHHMMSS>_<mode>/` and return it.

    Sub-second collisions get a `_2`, `_3`, … suffix so back-to-back runs
    do not clobber each other.
    """
    base = base or DEFAULT_RUNS_BASE
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    candidate = base / f"{ts}_{mode}"
    n = 2
    while candidate.exists():
        candidate = base / f"{ts}_{mode}_{n}"
        n += 1
    candidate.mkdir()
    return candidate


def latest_run_dir(base: Path | None = None) -> Path:
    """
    Return the newest run dir under `base`. Names are ISO-ish so lexicographic
    sort == chronological sort.
    """
    base = base or DEFAULT_RUNS_BASE
    if not base.exists():
        raise FileNotFoundError(
            f"no run dir at {base}; have you run `python -m cvar_v5.mc.runner` yet?"
        )
    candidates = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not candidates:
        raise FileNotFoundError(f"no run dirs in {base}")
    return max(candidates, key=lambda d: d.name)


def git_sha() -> tuple[str, bool]:
    """
    Return (short_sha, dirty). Falls back to ("unknown", False) if git is
    unavailable or we're outside a repo. Never raises.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return sha, bool(dirty_out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", False


def serialize_run_config(
    run_dir: Path,
    mode: str,
    n_workers: int,
    seed_base: int,
    cfg: Any,
    mode_params: Any,
) -> Path:
    """
    Write `<run_dir>/run_config.json`. `cfg` and `mode_params` are dataclasses;
    we serialize via `asdict` so non-default fields are captured automatically.
    """
    sha, dirty = git_sha()
    payload = {
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "n_workers": n_workers,
        "seed_base": seed_base,
        "git_sha": sha,
        "git_dirty": dirty,
        "config": asdict(cfg),
        "mode_params": asdict(mode_params),
    }
    out = run_dir / "run_config.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def setup_logging(
    run_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the root logger with a stream handler (stderr) and, if `run_dir`
    is given, a file handler at `<run_dir>/log.txt`. Idempotent: existing
    handlers are cleared so re-entry doesn't duplicate output.
    """
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "log.txt", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.getLogger(LOG_NAME)
