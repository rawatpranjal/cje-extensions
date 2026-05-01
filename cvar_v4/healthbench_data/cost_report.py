"""Print a running spend report from cost_ledger.jsonl.

Usage:
    python3 -m cvar_v4.healthbench_data.cost_report
"""
from __future__ import annotations

from .cost_ledger import CostLedger


def main() -> None:
    ledger = CostLedger()
    rows = ledger.rows()
    if not rows:
        print(f"(no entries in {ledger.path})")
        return

    summary = ledger.summary()
    total = ledger.total_usd()

    print(f"Cost ledger: {ledger.path}")
    print(f"  {len(rows)} rows | grand total: ${total:.4f}\n")

    print(f"{'phase':10} {'kind':8} {'model':32} {'n_req':>7} {'in_tok':>10} {'out_tok':>10} {'$':>9}")
    print("-" * 90)
    for (phase, kind, model), agg in sorted(summary.items()):
        kind_s = kind if kind is not None else "-"
        print(f"{phase:10} {kind_s:8} {model:32} {agg['n_requests']:>7,} "
              f"{agg['input_tokens']:>10,} {agg['output_tokens']:>10,} {agg['cost_usd']:>9.4f}")
    print("-" * 90)
    print(f"{'TOTAL':<70} ${total:.4f}")


if __name__ == "__main__":
    main()
