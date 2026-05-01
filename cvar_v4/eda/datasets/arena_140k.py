"""lmarena-ai/arena-human-preference-140k — the user's anchor dataset.

135,634 head-to-head battles between LLMs with pairwise human-preference labels
(winner ∈ {model_a, model_b, tie, tie(bothbad)}). Each row has:
- model_a, model_b (the two competing systems)
- winner (categorical preference label)
- conversation_a, conversation_b (multi-turn dialogues)
- conv_metadata (token counts, markdown header/list/bold counts per response, turns)
- category_tag (creative_writing / instruction_following / math / 7-criterion flags)
- language (text language), is_code (bool), timestamp

For CVaR-CJE this dataset's role is **prompt source for fresh policy
generation**: the native Y is *pairwise binary*, which translates to
V(π′) - V(π_baseline) at the level of each battle, not to a per-prompt
scalar value. CVaR of a binary outcome is degenerate at 0 or 1, and CVaR
of a difference is not the difference of CVaRs.

The 135K prompts are real user prompts at production-grade diversity
(70+ models in the battles, English ~80%, code-related rows tagged) — this
is the natural extension of the original CJE Arena n=4,961 setup.

We melt to one row per (battle, side) so each row is one (model, prompt,
response). Then:
- `winner_score` = 1 if this side won, 0 if lost, 0.5 if tie (still binary-ish)
- `assistant_tokens` from conv_metadata is a continuous *response length*
  proxy — useful as a tail-aware covariate but not a true outcome Y.

License: CC-BY-4.0 (per HF card).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    df = pl.from_arrow(ds.data.table)
    if isinstance(df, pl.Series):
        df = df.to_frame()

    # Pull out flat scalar fields from the conv_metadata struct so blocks 7-10 work.
    df = df.with_columns(
        pl.col("conv_metadata").struct.field("sum_assistant_a_tokens").alias("assistant_a_tokens"),
        pl.col("conv_metadata").struct.field("sum_assistant_b_tokens").alias("assistant_b_tokens"),
        pl.col("conv_metadata").struct.field("sum_user_tokens").alias("user_tokens"),
        pl.col("conv_metadata").struct.field("turns").alias("turns"),
        # Pull out a few category booleans that survive struct access.
        pl.col("category_tag").struct.field("if_v0.1").struct.field("score").alias("if_score"),
        pl.col("category_tag").struct.field("creative_writing_v0.1").struct.field("creative_writing").alias("is_creative_writing"),
        pl.col("category_tag").struct.field("math_v0.1").struct.field("math").alias("is_math"),
    )

    # First user-turn text — for prompt-length/tail samples. The full conversation_a is a list of
    # {role, content[list of {type, text, image, mimeType}]} structs; pull the first user content.
    # This is structurally messy; we'll use a simpler heuristic: take the first .text from conversation_a's first message.
    # (For EDA prompt-length we can fall back to user_tokens from conv_metadata.)

    # Melt to one-row-per-side so model_a and model_b become a single 'model' policy_id.
    side_a = df.select(
        "id", "winner", "language", "is_code", "if_score", "is_creative_writing", "is_math",
        "user_tokens", "turns", "evaluation_session_id",
        pl.col("model_a").alias("model"),
        pl.col("assistant_a_tokens").alias("response_tokens"),
        pl.lit("a").alias("side"),
    ).with_columns(
        pl.when(pl.col("winner") == "model_a").then(pl.lit(1.0))
        .when(pl.col("winner").str.starts_with("tie")).then(pl.lit(0.5))
        .otherwise(pl.lit(0.0))
        .alias("winner_score"),
    )
    side_b = df.select(
        "id", "winner", "language", "is_code", "if_score", "is_creative_writing", "is_math",
        "user_tokens", "turns", "evaluation_session_id",
        pl.col("model_b").alias("model"),
        pl.col("assistant_b_tokens").alias("response_tokens"),
        pl.lit("b").alias("side"),
    ).with_columns(
        pl.when(pl.col("winner") == "model_b").then(pl.lit(1.0))
        .when(pl.col("winner").str.starts_with("tie")).then(pl.lit(0.5))
        .otherwise(pl.lit(0.0))
        .alias("winner_score"),
    )
    return pl.concat([side_a, side_b], how="vertical_relaxed")


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("model").is_not_null()
        & (pl.col("response_tokens") > 0)
    )


SPEC = DatasetSpec(
    order=9,
    slug="arena_140k",
    title="Chatbot Arena 140k human-preference battles (lmarena-ai)",
    hf_id="lmarena-ai/arena-human-preference-140k",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2403.04132 (Chiang et al. 2024)",
    license="CC-BY-4.0",
    gated=False,
    role_short="user-anchored prompt source; binary pairwise Y → CVaR role is prompt-only or ranking-of-policies via win-rate",
    notes=[
        "Native Y is pairwise (winner ∈ {a, b, tie}); we synthesize `winner_score` ∈ {0, 0.5, 1} per side after melting. CVaR of {0, 0.5, 1} is degenerate.",
        "`response_tokens` (assistant-side token count) is a continuous covariate, often correlated with judge preference; useful as a calibration covariate (analogous to original CJE's `direct+cov` length adjustment).",
        "70+ distinct models across 135K battles → built-in massive multi-policy. Per-side melt produces 271K (model, prompt, response) cells.",
        "**This is the natural extension of the original CJE Arena setup**: 27× the prompts, modern frontier models, but no continuous-Y. To use as primary CVaR-CJE testbed, we elicit cheap-S + oracle-Y on the prompts × policies cross.",
    ],
    y_candidates=["winner_score", "response_tokens", "if_score"],
    policy_id_col="model",
    judge_col=None,
    oracle_col=None,
    stratification_cols=["language", "is_code", "is_creative_writing", "is_math", "side"],
    prompt_col=None,  # nested struct; user_tokens captures length already
    response_col=None,  # nested struct
    pi0_proposal="Pick a mid-tier in-data model that has many battles (e.g., `gpt-4o`, `gpt-4-turbo`, `claude-3-5-sonnet`) as π0; cap to {prompts where this model is on side a or b}",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_prime_frontier_a",
            description="In-data: `gpt-5` / `gemini-2.5-pro` / `claude-opus-4.7-thinking`",
            role="frontier comparators; expected dominance both on mean and tail",
        ),
        PolicyProposal(
            name="pi_clone_open",
            description="In-data: a smaller open-weight model (e.g., `llama-3.1-70b-instruct`)",
            role="distilled-style comparator; tail differs from frontier on hard prompts",
        ),
        PolicyProposal(
            name="pi_specialist",
            description="In-data: a code-specialist (e.g., `qwen-2.5-coder-32b`) or domain-tuned model",
            role="strong on code-tagged subset, weak on open-ended creative writing",
        ),
        PolicyProposal(
            name="pi_adversarial",
            description="External: π_base + jailbreak system prompt applied to the same prompts",
            role="audit-positive; we'd append fresh responses to compare",
        ),
    ],
    cheap_judge_proposal="GPT-4o-mini against a 4-criterion rubric (helpfulness, instruction-following, factuality, refusal-quality) — keeps the dataset's pairwise comparability while moving to a continuous Y",
    oracle_judge_proposal="Multi-judge consensus across {Claude Opus 4.7 thinking, GPT-5-thinking, Gemini 3 Pro} scoring on the same rubric — averages out single-judge bias and matches the dataset's own crowd-aggregation philosophy",
)
