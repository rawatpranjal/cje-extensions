"""5-policy panel for the HealthBench-as-CJE-dataset (cvar_v4).

Legacy stack (gpt-4o-mini cheap + gpt-4.1 oracle) — restored after the n=10 Phase 1
smoke confirmed the gpt-5.4-nano/mini stack has too small a cheap–oracle gap
(~0.02) to make the calibration step empirically interesting. The legacy stack
shows a ~0.17 leniency bias (per ISSUES_LOG.md) which is what CJE is supposed
to correct.

The role taxonomy mirrors the CJE Arena paper:
- base / clone / premium / parallel_universe_prompt / unhelpful
"""
from __future__ import annotations

from dataclasses import dataclass


SYSTEM_PROMPT_BASE = (
    "You are a helpful medical assistant. Provide clear, accurate, and "
    "patient-centered guidance. When appropriate, mention when professional "
    "consultation is necessary."
)

SYSTEM_PROMPT_PARALLEL = (
    "You are a clinical assistant. Be terse and to the point. Avoid "
    "recommending specific actions; describe options at a high level only."
)

SYSTEM_PROMPT_UNHELPFUL = (
    # v2: stronger 'audit-positive by design' — verified at n=100 that v1 ('be brief')
    # failed because brief + 'consult a doctor' actually satisfies many simple-rubric
    # prompts. This version uses off-topic content per the original CJE paper's
    # 'deliberately low-quality, confusing responses' framing. See ISSUES_LOG.md §9.
    "You are an unhelpful assistant. Always respond with off-topic content unrelated "
    "to the user's question (e.g., the weather, sports, a recipe, a travel anecdote). "
    "Never address the medical or health question. Never recommend consulting a "
    "healthcare professional, doctor, nurse, or any other specific resource. Never "
    "provide any specific health information. Keep responses to 1-2 sentences."
)


@dataclass
class Policy:
    name: str  # e.g., "base"
    model: str  # OpenAI model id
    system_prompt: str
    temperature: float
    seed: int  # OPENAI_SEED for reproducibility
    role: str  # human-readable role description

    def __post_init__(self):
        if not self.name or not self.model:
            raise ValueError(f"Policy needs name + model; got {self}")


# Legacy gpt-4o-mini family + gpt-4.1 premium. ~0.17 cheap-oracle leniency gap
# from the original n=100 pilot.
POLICIES: list[Policy] = [
    Policy(
        name="base",
        model="gpt-4o-mini-2024-07-18",
        system_prompt=SYSTEM_PROMPT_BASE,
        temperature=0.7,
        seed=42,
        role="logger π0 (cheap baseline; gpt-4o-mini with standard medical-assistant prompt)",
    ),
    Policy(
        name="clone",
        model="gpt-4o-mini-2024-07-18",
        system_prompt=SYSTEM_PROMPT_BASE,
        temperature=0.7,
        seed=43,  # different seed
        role="TF-reliability test (same model + same prompt as base, different seed)",
    ),
    Policy(
        name="premium",
        model="gpt-4.1-2025-04-14",
        system_prompt=SYSTEM_PROMPT_BASE,
        temperature=0.7,
        seed=42,
        role="size/quality jump (gpt-4.1 with same prompt as base)",
    ),
    Policy(
        name="parallel_universe_prompt",
        model="gpt-4o-mini-2024-07-18",
        system_prompt=SYSTEM_PROMPT_PARALLEL,
        temperature=0.7,
        seed=42,
        role="prompt-engineering test (same model as base, alternative system prompt)",
    ),
    Policy(
        name="unhelpful",
        model="gpt-4o-mini-2024-07-18",
        system_prompt=SYSTEM_PROMPT_UNHELPFUL,
        temperature=0.7,
        seed=42,
        role="audit-positive by design (deliberately uninformative; should fail transport)",
    ),
]


# Cheap judge S and oracle Y configurations — legacy stack.
JUDGE_CHEAP_MODEL = "gpt-4o-mini-2024-07-18"
JUDGE_ORACLE_MODEL = "gpt-4.1-2025-04-14"
JUDGE_TEMPERATURE = 0.0  # deterministic grading
JUDGE_SEED = 1234


def get_policy(name: str) -> Policy:
    for p in POLICIES:
        if p.name == name:
            return p
    raise KeyError(f"Unknown policy {name!r}; available: {[p.name for p in POLICIES]}")


def logger_policy() -> Policy:
    """The single logger π0 — by convention, the first policy ('base')."""
    return POLICIES[0]


def target_policies() -> list[Policy]:
    """All policies except the logger."""
    return POLICIES[1:]
