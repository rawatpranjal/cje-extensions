"""Audit verdict against actual error.

Cross-tabulates the audit verdict (PASS / FLAG_*) with the absolute error of
the Direct estimate against full-oracle truth, at a configurable error
threshold.

Outputs:
    writeup/data/audit_truth.json
"""
from __future__ import annotations

import argparse

from ..analyze import step5_oracle_calibrated_uniform
from ._common import panel_size, write_json


def compute(alphas=(0.10, 0.20), coverage: float = 0.25, seed: int = 42,
             err_threshold: float = 0.10) -> dict:
    all_rows = []
    for a in alphas:
        rows = step5_oracle_calibrated_uniform(
            coverage=coverage, alpha=a, seed=seed, verbose=False,
        )
        all_rows.extend(rows)

    counts = {"pass_low": [], "pass_high": [], "fail_low": [], "fail_high": []}
    for r in all_rows:
        is_pass = r["verdict"] == "PASS"
        is_low = r["abs_error"] <= err_threshold
        key = ("pass" if is_pass else "fail") + ("_low" if is_low else "_high")
        counts[key].append({
            "policy": r["policy"], "alpha": r["alpha"],
            "abs_error": r["abs_error"], "verdict": r["verdict"],
        })
    return {
        "alphas": list(alphas),
        "coverage": coverage,
        "seed": seed,
        "panel_size": panel_size(),
        "err_threshold": err_threshold,
        "counts": {k: len(v) for k, v in counts.items()},
        "cells": counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--err-threshold", type=float, default=0.10)
    args = ap.parse_args()
    payload = compute(coverage=args.coverage, seed=args.seed,
                      err_threshold=args.err_threshold)
    path = write_json("audit_truth.json", payload)
    c = payload["counts"]
    print(f"[audit_truth] pass_low={c['pass_low']}  pass_high={c['pass_high']}  "
          f"fail_low={c['fail_low']}  fail_high={c['fail_high']}")
    print(f"[audit_truth] wrote {path}")


if __name__ == "__main__":
    main()
