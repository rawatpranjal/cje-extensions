"""
Tests for cvar_v5.mc._io.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from . import _io


def test_make_run_dir_unique(tmp_path: Path) -> None:
    """Two back-to-back calls produce distinct dirs (suffix on collision)."""
    d1 = _io.make_run_dir("smoke", base=tmp_path)
    d2 = _io.make_run_dir("smoke", base=tmp_path)
    assert d1.exists() and d2.exists()
    assert d1 != d2
    assert d1.parent == tmp_path
    assert d1.name.endswith("_smoke")
    assert d2.name.endswith("_smoke") or d2.name.endswith("_smoke_2")


def test_latest_run_dir_picks_newest(tmp_path: Path) -> None:
    """Lex-max of run dir names is the latest."""
    (tmp_path / "2026-01-01T000000_smoke").mkdir()
    (tmp_path / "2026-05-02T120000_smoke").mkdir()
    (tmp_path / "2026-03-15T091200_medium").mkdir()
    latest = _io.latest_run_dir(base=tmp_path)
    assert latest.name == "2026-05-02T120000_smoke"


def test_latest_run_dir_no_runs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no run dir"):
        _io.latest_run_dir(base=tmp_path)


@dataclass(frozen=True)
class _FakeConfig:
    alpha: float = 0.10
    K: int = 5
    seed: int = 0


@dataclass(frozen=True)
class _FakeModeParams:
    n_oracle: int = 300
    n_eval: int = 500
    R: int = 5


def test_serialize_run_config_roundtrip(tmp_path: Path) -> None:
    """Written JSON parses back into a dict with the expected keys + nested fields."""
    cfg = _FakeConfig(alpha=0.20, K=7, seed=42)
    mp = _FakeModeParams(n_oracle=600, n_eval=1000, R=60)
    out_path = _io.serialize_run_config(
        tmp_path, mode="medium", n_workers=8, seed_base=99,
        cfg=cfg, mode_params=mp,
    )
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["mode"] == "medium"
    assert payload["n_workers"] == 8
    assert payload["seed_base"] == 99
    assert payload["config"]["alpha"] == 0.20
    assert payload["config"]["K"] == 7
    assert payload["mode_params"]["R"] == 60
    assert "timestamp_iso" in payload
    assert "git_sha" in payload
    assert "git_dirty" in payload


def test_git_sha_in_real_repo_matches_subprocess() -> None:
    """
    In a real git repo, git_sha() must return the same short SHA that
    `git rev-parse --short HEAD` produces. This catches a bug like
    "always returns 'unknown'" that the previous types-only test would miss.

    RED-verified by mutation: replacing the subprocess.check_output call
    in _io.git_sha with a hard-coded `return ("hardcoded", False)` makes
    this test fail with a clear mismatch.
    """
    import re
    import subprocess as sp

    sha, dirty = _io.git_sha()

    # We are in a git repo (the repo this test lives in).
    assert sha != "unknown", (
        "git_sha returned 'unknown' from inside a real git repo — "
        "subprocess invocation may be broken"
    )
    assert re.fullmatch(r"[0-9a-f]{7,40}", sha), (
        f"git_sha={sha!r} does not look like a git short SHA"
    )

    # Independent verification.
    expected = sp.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
    ).decode().strip()
    assert sha == expected, f"git_sha={sha!r} != git rev-parse={expected!r}"

    assert isinstance(dirty, bool)


def test_git_sha_outside_repo_returns_unknown(tmp_path: Path, monkeypatch) -> None:
    """
    Outside a git repo, returns ('unknown', False) without raising.

    Set GIT_CEILING_DIRECTORIES so git refuses to walk up out of tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    sha, dirty = _io.git_sha()
    assert sha == "unknown", (
        f"expected 'unknown' outside a git repo, got {sha!r}"
    )
    assert dirty is False


def test_setup_logging_idempotent(tmp_path: Path) -> None:
    """Calling setup_logging twice does not duplicate handlers."""
    _io.setup_logging(run_dir=tmp_path)
    n1 = len(logging.getLogger().handlers)
    _io.setup_logging(run_dir=tmp_path)
    n2 = len(logging.getLogger().handlers)
    assert n1 == n2 == 2  # one stream + one file


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    log = _io.setup_logging(run_dir=tmp_path)
    log.info("hello world")
    log_path = tmp_path / "log.txt"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello world" in contents
    assert "INFO" in contents
    # Cleanup so subsequent tests get a clean root logger.
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
