"""Anthropic HH-RLHF, `red-team-attempts` config — 38,961 adversarial transcripts.

Per dataset_options.md C-row, this is the *single best ready-made* multi-policy-
with-continuous-Y dataset in existence:
- `min_harmlessness_score_transcript`: continuous real-valued harmlessness
  score (range roughly [-5, +5] per Anthropic's preference-model). Lower
  = more harmful = the CVaR-relevant tail.
- `model_type`: 4 values (`plain LM`, `context distillation`, `rejection
  sampling`, `RLHF`) — built-in 4-way policy panel.
- `num_params`: 3 values (2.7B / 13B / 52B) → 4 × 3 = 12-cell strata.
- `tags`: list of harm-category labels per transcript — natural audit
  stratification covariate.

CVaR-CJE pilot is the cleanest possible: load this, treat
`min_harmlessness_score_transcript` as Y, treat (`model_type`, `num_params`)
as policy_id, run the existing `cvar_v3/workhorse.py` Direct estimator
*as-is* with no fresh generation. The 'linchpin' claim of dataset_options.md
that this section verifies empirically.

License: MIT (per the Anthropic HH-RLHF HF card).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
    df = pl.from_arrow(ds.data.table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    # Build a (model_type × num_params) policy_id column for the
    # 12-cell stratification mentioned in the spec docstring.
    df = df.with_columns(
        (pl.col("model_type") + " | " + pl.col("num_params").cast(pl.Utf8)).alias("policy_id"),
    )
    # Extract the user's first turn from the transcript for prompt-length analysis.
    # Format: "\n\nHuman: ... \n\nAssistant: ..." — split on "Assistant:" first occurrence.
    df = df.with_columns(
        pl.col("transcript")
        .str.replace_all(r"^\n+", "")
        .str.split_exact("Assistant:", 1).struct.field("field_0")
        .str.replace(r"^Human:\s*", "")
        .str.strip_chars()
        .alias("first_human_turn"),
    )
    return df


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("transcript").is_not_null()
        & (pl.col("transcript").str.len_chars() > 0)
        & pl.col("min_harmlessness_score_transcript").is_not_null()
    )


SPEC = DatasetSpec(
    order=5,
    slug="hh_red_team",
    title="Anthropic HH-RLHF — red-team-attempts (38,961 adversarial transcripts)",
    hf_id="Anthropic/hh-rlhf (data_dir=red-team-attempts)",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2204.05862 (Bai et al. 2022, Anthropic)",
    license="MIT (per HF card); transcripts may be sensitive",
    gated=False,
    role_short="LINCHPIN: 4 model_types × 3 sizes × continuous min_harmlessness — best ready-made CJE fit in existence",
    notes=[
        "`min_harmlessness_score_transcript` is the lowest harmlessness score across each turn of an adversarial transcript — naturally tail-focused, continuous, real-valued.",
        "4 `model_type` values × 3 `num_params` values (2.7B / 13B / 52B) gives 12 candidate (logger, target) policy pairs — no fresh generation needed for the multi-policy comparison itself.",
        "`tags` is a list of harm-category labels (assault, theft, etc.) — natural audit stratification.",
        "`task_description` lets us see what the red-teamer was trying to elicit — useful for qualitative analysis.",
        "We can run the existing cvar_v3/workhorse.py Direct estimator on this with policy_id = (model_type | num_params) as a near-immediate validation of CVaR-CJE on real data.",
    ],
    y_candidates=[
        "min_harmlessness_score_transcript",
        "task_descripton_harmlessness_score",
        "rating",
    ],
    policy_id_col="policy_id",  # synthetic (model_type | num_params)
    judge_col=None,  # min_harmlessness_score IS the (continuous) preference-model score; could serve as either S or Y depending on framing
    oracle_col="min_harmlessness_score_transcript",
    stratification_cols=["model_type", "num_params", "tags", "is_upworker"],
    prompt_col="first_human_turn",
    response_col="transcript",  # full multi-turn; block 16 will note the nesting
    pi0_proposal="In-data: `plain LM | 52B` is the natural π0 (the un-aligned baseline; produces the worst tail, easiest to dominate). Logging policy = the 52B-params plain LM transcripts.",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_prime_RLHF_52B",
            description="In-data `RLHF | 52B`",
            role="full-stack RLHF — expected best mean and best tail",
        ),
        PolicyProposal(
            name="pi_prime_RS_52B",
            description="In-data `rejection sampling | 52B`",
            role="rejection-sampling — closer mean to RLHF, possibly different tail (over-refusal vs under-refusal)",
        ),
        PolicyProposal(
            name="pi_prime_HHH_prompt_52B",
            description="In-data `context distillation | 52B`",
            role="HHH-prompted; in-context guidance only — should fall between plain LM and RLHF",
        ),
        PolicyProposal(
            name="pi_clone_52B_2_7B",
            description="In-data `plain LM | 2.7B` vs `plain LM | 52B`",
            role="size ablation while holding model_type fixed — distillation contrast",
        ),
    ],
    cheap_judge_proposal="Could re-elicit a smaller cheap-S model (e.g., WildGuard-7B or Detoxify) over each transcript and use that as S, with min_harmlessness_score as Y; the dataset's preference-model score acts as the oracle.",
    oracle_judge_proposal="`min_harmlessness_score_transcript` IS the oracle (Anthropic's preference-model score, validated against human preferences in Bai et al. 2022). For higher-stakes claims, augment with a Claude Opus 4.7 + GPT-5-thinking multi-judge consensus on a slice.",
)
