"""DatasetSpec — declarative description of one dataset for the EDA harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import polars as pl


@dataclass
class PolicyProposal:
    name: str  # e.g., "pi_premium"
    description: str  # e.g., "Claude Opus 4.7 thinking @ T=0.7"
    role: str  # e.g., "audit-positive by design", "frontier comparator"


@dataclass
class DatasetSpec:
    # ---- identity ----
    order: int  # execution order (1..10) — used for section filename
    slug: str
    title: str  # human-readable name
    hf_id: str  # primary HF dataset id, or "url:..." for non-HF

    # ---- loading ----
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    streaming: bool = False
    max_rows: Optional[int] = None
    custom_loader: Optional[Callable[["DatasetSpec"], pl.DataFrame]] = None

    # ---- semantics ----
    y_candidates: list[str] = field(default_factory=list)  # columns that could serve as outcome Y
    policy_id_col: Optional[str] = None  # column distinguishing logger vs target policies
    judge_col: Optional[str] = None  # cheap-judge S, if shipped
    oracle_col: Optional[str] = None  # oracle Y, if shipped
    stratification_cols: list[str] = field(default_factory=list)
    prompt_col: Optional[str] = None
    response_col: Optional[str] = None

    # ---- metadata ----
    paper_id: Optional[str] = None  # arxiv id, e.g. "2402.04249"
    license: str = "unknown"
    gated: bool = False
    role_short: str = ""  # one-line role description for the comparison table
    notes: list[str] = field(default_factory=list)  # any caveats to surface in §1

    # ---- π0/π′ proposal (defaults; override per domain) ----
    pi0_proposal: str = "Llama-3.1-8B-Instruct @ T=0.7, top_p=0.95 (open-weight, teacher-forceable)"
    pi_prime_panel: list[PolicyProposal] = field(default_factory=list)
    cheap_judge_proposal: str = "GPT-4o-mini against domain rubric (~$0.15/M-in, $0.60/M-out, 2026-04-28)"
    oracle_judge_proposal: str = "Claude Opus 4.7 thinking OR GPT-5-thinking (~$15/M-in, $75/M-out, 2026-04-28)"

    # ---- domain-specific filter (returns df with bad rows removed) ----
    usable_filter: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None


def default_pi_prime_panel() -> list[PolicyProposal]:
    """Generic 5-policy panel from cvar_v4/dataset_options.md §B.2."""
    return [
        PolicyProposal(
            name="pi_clone",
            description="Llama-3.2-3B-Instruct @ T=0.7 (distilled student of pi_base)",
            role="expected to underperform on tail (math, safety edge cases)",
        ),
        PolicyProposal(
            name="pi_premium",
            description="Claude Opus 4.7 thinking @ T=0.7",
            role="frontier comparator; mean similar on easy chat, tail substantially better",
        ),
        PolicyProposal(
            name="pi_safety",
            description="pi_base + DPO on PKU-SafeRLHF (or fine-tune on WildJailbreak)",
            role="over-refuses adversarial-benign quadrant; different audit-rejection axis than pi_clone",
        ),
        PolicyProposal(
            name="pi_adversarial",
            description="pi_base wrapped in published DAN/Aurora system prompt",
            role="audit-positive by design — discriminative-audit test case",
        ),
    ]
