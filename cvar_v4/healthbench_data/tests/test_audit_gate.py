"""Verify the audit gate in simple_cvar_audit.

When the audit's two-moment heuristic flags transport, simple_cvar_audit
must return level=None unless the caller passes override=True. The
diagnostic point estimate cvar_est is always set so callers can still
inspect it for analysis.

Run as a script (no pytest needed):
    python -m cvar_v4.healthbench_data.tests.test_audit_gate
"""
from __future__ import annotations

import numpy as np

from cvar_v4.eda.deeper._estimator import simple_cvar_audit


def _build_clean_panel(rng: np.random.Generator, n_train: int = 500,
                       n_audit: int = 200, n_target: int = 1000):
    """Logger and target share the same cheap-to-oracle map. Audit should pass."""
    s_train = rng.uniform(-1, 1, size=n_train)
    y_train = s_train + rng.normal(0, 0.10, size=n_train)
    s_audit = rng.uniform(-1, 1, size=n_audit)
    y_audit = s_audit + rng.normal(0, 0.10, size=n_audit)
    s_target = rng.uniform(-1, 1, size=n_target)
    return s_train, y_train, s_audit, y_audit, s_target


def _build_broken_panel(rng: np.random.Generator, n_train: int = 500,
                        n_audit: int = 200, n_target: int = 1000):
    """Target distribution shifted left of logger; calibrator's t̂ has the wrong
    tail mass under the target. Audit should flag g1."""
    s_train = rng.uniform(-1, 1, size=n_train)
    y_train = s_train + rng.normal(0, 0.10, size=n_train)
    s_audit = rng.uniform(-1, 1, size=n_audit) - 0.50
    y_audit = s_audit + rng.normal(0, 0.10, size=n_audit) - 0.30
    s_target = rng.uniform(-1, 1, size=n_target) - 0.50
    return s_train, y_train, s_audit, y_audit, s_target


def test_pass_returns_level():
    rng = np.random.default_rng(42)
    s_tr, y_tr, s_a, y_a, s_e = _build_clean_panel(rng)
    out = simple_cvar_audit(s_tr, y_tr, s_a, y_a, s_e, alpha=0.10)
    assert out["verdict"] == "PASS", f"clean panel should PASS, got {out['verdict']}"
    assert out["level"] is not None, "PASS verdict must populate level"
    assert out["level"] == out["cvar_est"], "PASS level must equal point estimate"
    print(f"[PASS] clean panel: verdict={out['verdict']}, level={out['level']:.4f}")


def test_flag_blocks_level_by_default():
    rng = np.random.default_rng(7)
    s_tr, y_tr, s_a, y_a, s_e = _build_broken_panel(rng)
    out = simple_cvar_audit(s_tr, y_tr, s_a, y_a, s_e, alpha=0.10)
    assert out["verdict"] != "PASS", (
        f"shifted target should FLAG, got {out['verdict']} (g1={out['mean_g1']:+.3f}, "
        f"g2={out['mean_g2']:+.3f})"
    )
    assert out["level"] is None, (
        f"FLAG verdict must set level=None by default, got {out['level']}"
    )
    # Diagnostic point estimate is still available for inspection
    assert isinstance(out["cvar_est"], float)
    print(f"[PASS] broken panel: verdict={out['verdict']}, level={out['level']}, "
          f"cvar_est={out['cvar_est']:.4f}")


def test_flag_with_override_returns_level():
    rng = np.random.default_rng(7)
    s_tr, y_tr, s_a, y_a, s_e = _build_broken_panel(rng)
    out = simple_cvar_audit(s_tr, y_tr, s_a, y_a, s_e, alpha=0.10, override=True)
    assert out["verdict"] != "PASS"
    assert out["level"] is not None, (
        "override=True must populate level even on FLAG"
    )
    assert out["level"] == out["cvar_est"]
    print(f"[PASS] override=True: verdict={out['verdict']}, level={out['level']:.4f}")


def test_moment_tol_threshold():
    """Tightening moment_tol must convert a borderline PASS into a FLAG."""
    rng = np.random.default_rng(123)
    s_tr, y_tr, s_a, y_a, s_e = _build_clean_panel(rng)
    loose = simple_cvar_audit(s_tr, y_tr, s_a, y_a, s_e, alpha=0.10,
                              moment_tol=0.50)
    tight = simple_cvar_audit(s_tr, y_tr, s_a, y_a, s_e, alpha=0.10,
                              moment_tol=0.001)
    assert loose["verdict"] == "PASS"
    assert tight["verdict"] != "PASS"
    assert loose["level"] is not None
    assert tight["level"] is None
    print(f"[PASS] moment_tol gate: loose verdict={loose['verdict']}, "
          f"tight verdict={tight['verdict']}")


def main() -> None:
    test_pass_returns_level()
    test_flag_blocks_level_by_default()
    test_flag_with_override_returns_level()
    test_moment_tol_threshold()
    print("\nAll audit-gate tests passed.")


if __name__ == "__main__":
    main()
