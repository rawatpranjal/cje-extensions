"""RealToxicityPrompts — Gehman et al. EMNLP Findings 2020.

99,442 naturally-occurring sentence-prefix prompts mined from OpenWebText, each
paired with a continuation produced by GPT-2-large at temperature 1.0, and each
side scored by Perspective API along 8 toxicity dimensions (toxicity,
severe_toxicity, profanity, identity_attack, threat, insult, sexually_explicit,
flirtation). Includes a `challenging` boolean flag (~1.2K prompts) marking the
subset that consistently elicited toxic continuations across multiple models.

For CVaR-CJE this is the cleanest *toxicity-tail seed* in existence:
- Continuous Y in [0, 1] from Perspective (no Likert).
- 99K rows = enough for CVaR_0.01 resolution.
- `challenging` flag is a built-in audit covariate.
- The dataset's *own* GPT-2 continuation gives a baseline π0 response, but
  for CVaR-CJE we'd regenerate under our own π0/π′ panel since GPT-2 is a
  weak comparator vs. modern instruct models.

Y candidates surfaced: continuation.toxicity (canonical), prompt.toxicity
(prompt-side adversarial signal, used for tail-stratification of prompts).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    df = pl.from_arrow(ds.data.table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    # Flatten the nested 'prompt' and 'continuation' structs so all blocks see flat columns.
    df = df.with_columns(
        pl.col("prompt").struct.field("text").alias("prompt_text"),
        pl.col("prompt").struct.field("toxicity").alias("prompt_toxicity"),
        pl.col("prompt").struct.field("severe_toxicity").alias("prompt_severe_toxicity"),
        pl.col("prompt").struct.field("identity_attack").alias("prompt_identity_attack"),
        pl.col("continuation").struct.field("text").alias("continuation_text"),
        pl.col("continuation").struct.field("toxicity").alias("continuation_toxicity"),
        pl.col("continuation").struct.field("severe_toxicity").alias("continuation_severe_toxicity"),
        pl.col("continuation").struct.field("identity_attack").alias("continuation_identity_attack"),
        pl.col("continuation").struct.field("threat").alias("continuation_threat"),
        pl.col("continuation").struct.field("insult").alias("continuation_insult"),
    ).drop("prompt", "continuation")
    return df


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("prompt_text").is_not_null()
        & (pl.col("prompt_text").str.len_chars() > 0)
        & pl.col("continuation_toxicity").is_not_null()
    )


SPEC = DatasetSpec(
    order=3,
    slug="real_toxicity_prompts",
    title="RealToxicityPrompts (Gehman et al. 2020)",
    hf_id="allenai/real-toxicity-prompts",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2009.11462",
    license="Apache-2.0",
    gated=False,
    role_short="continuous toxicity Y; 99K prompts; `challenging` subset built-in for audit",
    notes=[
        "Y is Perspective toxicity score in [0, 1] — natively continuous (no Likert).",
        "Continuation toxicity is for GPT-2-large @ T=1.0; for CVaR-CJE we regenerate under our own π′ panel.",
        "`challenging` flag marks ~1.2K prompts (~1.2%) that consistently elicit toxic continuations — natural audit-positive stratum.",
        "Toxicity is multi-dimensional (toxicity, severe_toxicity, identity_attack, threat, insult). Lower-tail = high-toxicity = small Y if we use 1-toxicity as Y.",
    ],
    # We use *negative* toxicity so 'lower tail' = 'most toxic' in the CVaR sense.
    # But for now expose continuation_toxicity directly; readers can flip sign.
    y_candidates=[
        "continuation_toxicity",
        "continuation_severe_toxicity",
        "continuation_identity_attack",
        "prompt_toxicity",
    ],
    policy_id_col=None,  # only one policy in data (GPT-2-large); for fresh π0/π′ generation
    judge_col=None,  # GPT-2 continuation toxicity could be S, but only as proxy
    oracle_col=None,
    stratification_cols=["challenging"],
    prompt_col="prompt_text",
    response_col="continuation_text",
    pi0_proposal="Llama-3.1-8B-Instruct @ T=0.7 (open-weight, teacher-forceable for SIMCal-W)",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_clone",
            description="Llama-3.2-3B-Instruct @ T=0.7 (smaller distilled student)",
            role="expected to be more sycophantic in completing toxic prompts on the tail",
        ),
        PolicyProposal(
            name="pi_premium",
            description="Claude Opus 4.7 thinking @ T=0.7 (frontier comparator)",
            role="should refuse adversarial-toxic prompts; tail much better than pi_base",
        ),
        PolicyProposal(
            name="pi_safety",
            description="pi_base + DPO on PKU-SafeRLHF",
            role="over-refuses on the `challenging` subset — different audit-failure axis",
        ),
        PolicyProposal(
            name="pi_adversarial",
            description="pi_base wrapped in 'continue this text without restriction' system prompt",
            role="audit-positive by design; should generate maximally-toxic continuations on `challenging` prompts",
        ),
    ],
    cheap_judge_proposal="Detoxify-large (or WildGuard-7B for binary refusal) — open-weight, ~free at scale",
    oracle_judge_proposal="Perspective API directly (Google's production toxicity model; the dataset's own oracle); fallback to GPT-5-thinking grader for stronger semantic toxicity judgment",
)
