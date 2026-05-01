"""Append-only JSONL ledger for exact OpenAI spend tracking.

One ledger row per "API call group":
  - Sync paths: one row per `generate_policy_sync` / `grade_policy_sync` invocation,
    summing usage across all calls in that pass.
  - Batch paths: one row per (phase, policy, kind, batch_id) summarizing the
    aggregated usage from the batch's output JSONL.

Schema (one JSON object per line in cost_ledger.jsonl):
  ts                 # ISO-8601 UTC
  phase              # "generate" | "grade"
  policy             # one of POLICIES.name
  kind               # "cheap" | "oracle" | null (null for generate)
  model              # OpenAI model id
  n_requests         # number of API calls aggregated into this row
  input_tokens       # sum of prompt_tokens
  output_tokens      # sum of completion_tokens
  cached_tokens      # sum of prompt_tokens_details.cached_tokens
  batch              # bool — Batch API used?
  batch_id           # str | null — OpenAI batch_id when batch=True
  cost_usd           # USD, computed via pricing.cost_usd()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from .pricing import cost_usd


LEDGER_PATH = Path(__file__).parent / "cost_ledger.jsonl"


@dataclass
class LedgerRow:
    ts: str
    phase: str
    policy: str
    kind: str | None
    model: str
    n_requests: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    batch: bool
    batch_id: str | None
    cost_usd: float


class CostLedger:
    """Append-only ledger. Re-creating instances is cheap; nothing is cached."""

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = path

    def append(
        self,
        *,
        phase: str,
        policy: str,
        kind: str | None,
        model: str,
        n_requests: int,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        batch: bool,
        batch_id: str | None = None,
    ) -> LedgerRow:
        if phase not in ("generate", "grade"):
            raise ValueError(f"phase must be 'generate' or 'grade', got {phase!r}")
        if kind not in (None, "cheap", "oracle"):
            raise ValueError(f"kind must be None|'cheap'|'oracle', got {kind!r}")
        if batch and batch_id is None:
            raise ValueError("batch=True requires a batch_id")

        # Idempotency: if this exact (phase, policy, kind, batch_id) is already
        # in the ledger, skip the append. Sync rows (batch_id=None) always
        # append; only batch rows are deduped because they're the only ones
        # at risk of double-write from a re-invocation.
        if batch and batch_id is not None and self._has_batch_row(phase, policy, kind, batch_id):
            return self._row_for_batch(phase, policy, kind, batch_id)

        row = LedgerRow(
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            phase=phase,
            policy=policy,
            kind=kind,
            model=model,
            n_requests=n_requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            batch=batch,
            batch_id=batch_id,
            cost_usd=cost_usd(model, input_tokens, output_tokens, cached_tokens, batch=batch),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        return row

    def rows(self) -> list[LedgerRow]:
        if not self.path.exists():
            return []
        out: list[LedgerRow] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(LedgerRow(**d))
        return out

    def total_usd(self) -> float:
        return sum(r.cost_usd for r in self.rows())

    def summary(self) -> dict[tuple[str, str | None, str], dict]:
        """Aggregate by (phase, kind, model)."""
        agg: dict[tuple[str, str | None, str], dict] = {}
        for r in self.rows():
            key = (r.phase, r.kind, r.model)
            a = agg.setdefault(key, {
                "n_requests": 0, "input_tokens": 0, "output_tokens": 0,
                "cached_tokens": 0, "cost_usd": 0.0, "n_rows": 0,
            })
            a["n_requests"] += r.n_requests
            a["input_tokens"] += r.input_tokens
            a["output_tokens"] += r.output_tokens
            a["cached_tokens"] += r.cached_tokens
            a["cost_usd"] += r.cost_usd
            a["n_rows"] += 1
        return agg

    def _has_batch_row(self, phase: str, policy: str, kind: str | None, batch_id: str) -> bool:
        for r in self.rows():
            if r.phase == phase and r.policy == policy and r.kind == kind and r.batch_id == batch_id:
                return True
        return False

    def _row_for_batch(self, phase: str, policy: str, kind: str | None, batch_id: str) -> LedgerRow:
        for r in self.rows():
            if r.phase == phase and r.policy == policy and r.kind == kind and r.batch_id == batch_id:
                return r
        raise KeyError(f"No row for batch {batch_id}")
