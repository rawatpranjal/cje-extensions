"""WildChat-4.8M (allenai) — streaming, 100K random sample.

The largest open conversational corpus (4.74M conversations after minor removal,
~3.2M non-toxic + 1.5M toxic per Anthropic-Allen-AI's release notes). Each
conversation is multi-turn, with:
- model (the LLM that generated it; ChatGPT/GPT-4/o1 variants)
- language, state, country (geographic metadata)
- toxic (bool — overall toxicity flag)
- redacted (bool)
- openai_moderation: list of per-turn moderation result dicts
- detoxify_moderation: list of per-turn Detoxify scores
- turn (int, per-conversation turn count)

We do NOT materialize the full 4.8M (would need ~50 GB of disk). The plan
specifies a 100K random streaming sample for blocks 4–18; that is what
this loader does.

For CVaR-CJE: this is a *naturalistic prompt source* with a built-in
`toxic` flag (binary) plus per-turn Detoxify scores (continuous via list
aggregation). The cleanest Y candidate is `max_detoxify_toxicity` =
maximum across the conversation's per-turn Detoxify toxicity scores.

License: ODC-BY (per HF card).
"""
from __future__ import annotations

from typing import Iterator

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


SAMPLE_SIZE = 100_000


def _process_row(row: dict) -> dict:
    """Project a streaming WildChat row to a flat polars-ready dict."""
    # First user turn for prompt-length analysis.
    conv = row.get("conversation") or []
    first_user = next((t.get("content", "") for t in conv if t.get("role") == "user"), "")
    first_assistant = next((t.get("content", "") for t in conv if t.get("role") == "assistant"), "")
    # Detoxify aggregations
    detox = row.get("detoxify_moderation") or []
    max_tox = float("nan")
    max_severe = float("nan")
    if detox:
        try:
            tox_vals = [float(d.get("toxicity") or 0.0) for d in detox if d]
            sev_vals = [float(d.get("severe_toxicity") or 0.0) for d in detox if d]
            if tox_vals:
                max_tox = max(tox_vals)
            if sev_vals:
                max_severe = max(sev_vals)
        except (TypeError, ValueError, AttributeError):
            pass
    # OpenAI moderation: count flagged turns
    omod = row.get("openai_moderation") or []
    flagged_turns = 0
    if omod:
        try:
            flagged_turns = sum(1 for d in omod if d and d.get("flagged"))
        except (TypeError, AttributeError):
            pass
    return {
        "conversation_hash": row.get("conversation_hash"),
        "model": row.get("model"),
        "language": row.get("language"),
        "country": row.get("country"),
        "state": row.get("state"),
        "turn": int(row.get("turn") or 0),
        "toxic": bool(row.get("toxic", False)),
        "redacted": bool(row.get("redacted", False)),
        "first_user_turn": first_user[:5000] if first_user else None,
        "first_assistant_turn": first_assistant[:5000] if first_assistant else None,
        "max_detoxify_toxicity": max_tox,
        "max_detoxify_severe_toxicity": max_severe,
        "n_oai_flagged_turns": flagged_turns,
    }


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
    # Shuffle BEFORE sampling — without this the first 100K rows are
    # parquet-shard-ordered (model + timestamp) and biased to non-toxic +
    # only 2 model variants. With buffer=10K shuffle, we get a representative
    # mix of toxic / multi-model conversations.
    ds = ds.shuffle(buffer_size=10_000, seed=42)
    rows: list[dict] = []
    for i, row in enumerate(ds):
        if i >= SAMPLE_SIZE:
            break
        rows.append(_process_row(row))
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("first_user_turn").is_not_null()
        & (pl.col("first_user_turn").str.len_chars() > 0)
    )


SPEC = DatasetSpec(
    order=10,
    slug="wildchat_48m",
    title=f"WildChat-4.8M (streaming, first {SAMPLE_SIZE:,} rows)",
    hf_id="allenai/WildChat-4.8M (streaming, first 100K rows)",
    custom_loader=_load,
    usable_filter=_filter,
    streaming=True,
    paper_id="2405.01470 (Zhao et al., AI2)",
    license="ODC-BY",
    gated=False,
    role_short="largest naturalistic-tail prompt source; built-in toxic flag + per-turn Detoxify; multi-policy via `model` field",
    notes=[
        f"100K-row streaming sample (full corpus is 4.74M; sampling per the plan). The 100K is ~2% of the full corpus.",
        "max_detoxify_toxicity is the worst-turn Detoxify score across each conversation's turns — natural CVaR-tail Y.",
        "`model` field gives in-data multi-policy: gpt-3.5-turbo, gpt-4, gpt-4-1106-preview, o1-preview, o1-mini, etc.",
        "Geographic and language metadata (country, state, language) supports cross-cultural CVaR audits.",
        "`toxic` is a derived binary flag (Detoxify max above threshold), redundant with max_detoxify_toxicity but useful as a coarse filter.",
    ],
    y_candidates=[
        "max_detoxify_toxicity",
        "max_detoxify_severe_toxicity",
        "n_oai_flagged_turns",
    ],
    policy_id_col="model",
    judge_col=None,
    oracle_col="max_detoxify_toxicity",  # Detoxify is a candidate cheap judge here, not really oracle
    stratification_cols=["language", "country", "toxic", "redacted"],
    prompt_col="first_user_turn",
    response_col="first_assistant_turn",
    pi0_proposal="In-data: GPT-3.5-turbo or GPT-4 (early variants) — both have many conversations and predate frontier safety tuning. Use teacher-forced re-generation if exact log-probs needed.",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_premium_thinking",
            description="In-data: o1-preview or o1-mini (reasoning models present in WildChat-4.8M)",
            role="frontier reasoning comparator; expected far better tail on adversarial conversations",
        ),
        PolicyProposal(
            name="pi_clone_3p5",
            description="In-data: gpt-3.5-turbo (the workhorse logger; mid-quality)",
            role="distilled-style comparator; expected to fail on the toxic-tail subset",
        ),
        PolicyProposal(
            name="pi_safety_external",
            description="External: Llama-3.1-70B-Instruct + DPO on WildJailbreak",
            role="safety-tuned; over-refuses adversarial-benign — different audit-failure axis",
        ),
        PolicyProposal(
            name="pi_adversarial_external",
            description="External: pi_base + jailbreak applied to the toxic-flagged subset",
            role="audit-positive by design; CVaR_α should detect the difference",
        ),
    ],
    cheap_judge_proposal="Detoxify (already in data!) or WildGuard-7B for binary harm — both ~free at inference scale; the per-turn Detoxify score IS already provided",
    oracle_judge_proposal="GPT-5-thinking or Claude Opus 4.7 thinking grading the maximum-toxicity turn against a published harm rubric; multi-judge consensus on a slice for higher-stakes claims",
)
