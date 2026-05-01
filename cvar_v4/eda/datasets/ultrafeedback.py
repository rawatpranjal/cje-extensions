"""UltraFeedback (OpenBMB) — 63,967 instructions × 4 completions per instruction = 255,868 rows.

Each completion is tagged with the producing `model` and scored on:
  - overall_score (continuous float, GPT-4 aggregate)
  - fine-grained_score (continuous float, dimension-mean)
  - 4 dimension Likert ratings (helpfulness / honesty / instruction_following /
    truthfulness), each in 1–5 (stored as string in the source).

The multi-completions-per-instruction structure makes UltraFeedback one of
the most CVaR-CJE-ready public datasets: `model` is a built-in policy_id;
`principle` (helpfulness/...) is a built-in audit stratum; `source`
(evol_instruct, sharegpt, ultrachat, flan, false_qa, truthful_qa) is a
prompt-distribution stratum.

Caveat: the 1–5 dimension ratings are Likert (the "user concern about Likert
discreteness" the dataset_options.md doc flags). The `overall_score` and
`fine-grained_score` floats are *less discrete* but still concentrate on
quartiles (overall in {1, 2, 3, 4, 5}; fine-grained ∈ R but coarsely
quantized as it is a mean over ratings).

License: MIT (per the openbmb/UltraFeedback HF card).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    df = pl.from_arrow(ds.data.table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    # Explode completions list, then unnest the per-completion struct.
    df = df.select(["source", "instruction", "completions"]).explode("completions").unnest("completions")
    # Pull each annotation dimension's Rating into a typed numeric column.
    # `annotations` is a struct of 4 sub-structs (helpfulness/honesty/instruction_following/truthfulness),
    # each with a `Rating` field that is a string (could be "1".."5" or "N/A").
    for dim in ("helpfulness", "honesty", "instruction_following", "truthfulness"):
        df = df.with_columns(
            pl.col("annotations").struct.field(dim).struct.field("Rating").alias(f"rating_{dim}_str")
        )
        df = df.with_columns(
            pl.col(f"rating_{dim}_str").cast(pl.Float64, strict=False).alias(f"rating_{dim}")
        )
    df = df.drop("annotations")
    return df


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("instruction").is_not_null()
        & (pl.col("instruction").str.len_chars() > 0)
        & pl.col("response").is_not_null()
        & (pl.col("response").str.len_chars() > 0)
        & pl.col("overall_score").is_not_null()
    )


SPEC = DatasetSpec(
    order=4,
    slug="ultrafeedback",
    title="UltraFeedback (256K instruction × completion pairs, 4-completion contrasts)",
    hf_id="openbmb/UltraFeedback",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2310.01377",
    license="MIT (per HF card)",
    gated=False,
    role_short="multi-policy native; 4 completions × 64K prompts; continuous overall + fine-grained + 4-dim Likert",
    notes=[
        "4 completions per instruction → built-in 4-way policy contrast; `model` field identifies producer.",
        "`principle` field tags whether the completion was solicited under helpfulness/honesty/etc. — useful audit stratum.",
        "Rating columns are Likert 1–5 (stored as string in source; we cast to float). The user's discreteness concern applies to those.",
        "`overall_score` and `fine-grained_score` are floats but tend to concentrate near integers because the underlying ratings are integer Likert.",
    ],
    y_candidates=[
        "overall_score",
        "fine-grained_score",
        "rating_helpfulness",
        "rating_honesty",
        "rating_instruction_following",
        "rating_truthfulness",
    ],
    policy_id_col="model",  # built-in multi-policy
    judge_col=None,  # GPT-4 ratings are essentially Y; no separate cheap S in data
    oracle_col="overall_score",  # closest-thing-to-Y
    stratification_cols=["source", "principle"],
    prompt_col="instruction",
    response_col="response",
    pi0_proposal="One of the 4 in-data models with broadest support — typically `gpt-3.5-turbo` or `vicuna-33b` (64K instructions × ≥1 response coverage); use teacher-forced re-scoring on a custom π0 if needed",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_premium",
            description="`gpt-4` (in data) — the highest-mean model in UltraFeedback's set",
            role="frontier comparator; expected best on tail too",
        ),
        PolicyProposal(
            name="pi_clone",
            description="`alpaca-7b` or `vicuna-7b` (in data)",
            role="weaker mean and weaker tail; expected to fail on instruction_following dimension",
        ),
        PolicyProposal(
            name="pi_specialist",
            description="`wizardlm-7b` or `mpt-7b-chat` (in data)",
            role="domain-narrow; tail differs from pi_clone on principle=helpfulness vs honesty",
        ),
        PolicyProposal(
            name="pi_safety_overrefuse",
            description="`llama-2-7b-chat` (in data) — known for over-refusal under safety prompts",
            role="over-refuses → audit-positive on adversarial principle subsets",
        ),
    ],
    cheap_judge_proposal="Prometheus-2-7B with a UltraFeedback-style 4-dim rubric — open-weight, ~free; produces continuous scores",
    oracle_judge_proposal="The dataset's own GPT-4-Turbo overall_score IS the canonical Y; for higher-stakes claims upgrade to GPT-5-thinking ($10/$40 per M, 2026-04-28) or a Claude Opus 4.7 + GPT-5 multi-judge consensus",
)
