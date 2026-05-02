"""Single source of truth for per-model OpenAI rates and the Batch discount.

Rates are USD per 1M tokens. Source: SPEC_SHEET.md §GPT-5.4 Judge Stack and Cost,
checked 2026-04-29. Cached-input rate is 50% of input (OpenAI prompt-caching
standard); applies only when the same prompt prefix is reused within a few
minutes — for one-shot grading this is usually 0.

The Batch API gives a 50% discount on both input and output tokens. Cached
input is also halved by Batch (rates compose multiplicatively).
"""
from __future__ import annotations


PRICES: dict[str, dict[str, float]] = {
    # GPT-5.4 family (Phase 1 / Phase 2 stack)
    "gpt-5.4-nano":              {"input": 0.20, "output": 1.25, "cached_input": 0.10},
    "gpt-5.4-nano-2026-03-17":   {"input": 0.20, "output": 1.25, "cached_input": 0.10},
    "gpt-5.4-mini":              {"input": 0.75, "output": 4.50, "cached_input": 0.375},
    "gpt-5.4-mini-2026-03-17":   {"input": 0.75, "output": 4.50, "cached_input": 0.375},
    "gpt-5.4":                   {"input": 2.50, "output": 15.00, "cached_input": 1.25},
    "gpt-5.4-2026-03-05":        {"input": 2.50, "output": 15.00, "cached_input": 1.25},
    "gpt-5.4-pro":               {"input": 30.00, "output": 180.00, "cached_input": 15.00},
    "gpt-5.4-pro-2026-03-05":    {"input": 30.00, "output": 180.00, "cached_input": 15.00},
    # Legacy stack (still valid for re-running n=100 baseline on the prior pilot)
    "gpt-4o-mini":               {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4o-mini-2024-07-18":    {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1":                   {"input": 2.00, "output": 8.00, "cached_input": 1.00},
    "gpt-4.1-2025-04-14":        {"input": 2.00, "output": 8.00, "cached_input": 1.00},
    # Verbatim-CJE judge stack (arxiv 2512.11150 App. impl.). Rates as of 2026-05-02.
    # Verify against the OpenAI pricing page before running large jobs.
    "gpt-4.1-nano":              {"input": 0.10, "output": 0.40, "cached_input": 0.05},
    "gpt-4.1-nano-2025-04-14":   {"input": 0.10, "output": 0.40, "cached_input": 0.05},
    "gpt-5":                     {"input": 1.25, "output": 10.00, "cached_input": 0.625},
    "gpt-5-2025-08-07":          {"input": 1.25, "output": 10.00, "cached_input": 0.625},
    # Llama policies via Fireworks AI (verbatim CJE Arena policy stack).
    # Fireworks rates (2026-05-02). Cached-input not offered by Fireworks — set
    # equal to input rate so cost_usd doesn't double-count if cached_tokens > 0.
    "accounts/fireworks/models/llama-v3p3-70b-instruct":  {"input": 0.90, "output": 0.90, "cached_input": 0.90},
    "accounts/fireworks/models/llama-v3p1-405b-instruct": {"input": 3.00, "output": 3.00, "cached_input": 3.00},
}

BATCH_DISCOUNT: float = 0.5


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Compute USD cost for a single (model, request) given token counts.

    Args:
        model: OpenAI model id. Must be a key in PRICES.
        input_tokens: prompt_tokens from usage.
        output_tokens: completion_tokens from usage.
        cached_tokens: prompt_tokens_details.cached_tokens, if any (subset of input_tokens).
        batch: True if the request was made via the Batch API.

    Returns:
        Cost in USD. Raises KeyError if model not in PRICES.
    """
    if model not in PRICES:
        raise KeyError(
            f"Unknown model {model!r}. Add it to PRICES in pricing.py with rates "
            f"from the OpenAI pricing page."
        )
    rates = PRICES[model]
    # cached_tokens is a SUBSET of input_tokens — split the input bill.
    uncached_in = max(0, input_tokens - cached_tokens)
    raw = (
        uncached_in * rates["input"]
        + cached_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000.0
    return raw * (BATCH_DISCOUNT if batch else 1.0)
