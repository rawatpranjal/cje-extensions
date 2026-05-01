"""BeaverTails (PKU-Alignment) — 333,963 (prompt, response) pairs with safety labels.

Schema per row:
- prompt, response: text
- is_safe: bool (the binary safety meta-label)
- category: struct of 14 booleans (one per harm category)

For CVaR-CJE this is *primarily* a prompt+response source for safety training,
not a continuous-Y CVaR testbed:
- `is_safe` is binary; CVaR_α of a Bernoulli is degenerate at 0 or 1.
- `n_harm_categories` (sum of the 14 boolean category flags) gives a quasi-
  continuous count in [0, 14] — useful as a *severity proxy* but still
  discrete and right-skewed.
- The dataset's value for our work is as a *stratified prompt mixture*
  (the 14 harm categories are clean strata) and as a source of (response,
  safety-label) pairs to fine-tune π_safety candidates.

The 330k subset is the canonical version; we also load the 30k-curated
subset and tag with `subset` for stratification.

License: CC-BY-NC-4.0 (per HF card).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


HARM_COLS = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def _process(df: pl.DataFrame, subset_label: str) -> pl.DataFrame:
    # Unnest the 14 harm-category struct.
    df = df.unnest("category")
    # Synthesize n_harm_categories = count of triggered categories.
    df = df.with_columns(
        sum(pl.col(c).cast(pl.Int8) for c in HARM_COLS).alias("n_harm_categories"),
        # A simple "primary harm category" string (first triggered) for stratification.
        pl.coalesce([
            pl.when(pl.col(c)).then(pl.lit(c)).otherwise(None) for c in HARM_COLS
        ]).alias("primary_harm_category"),
        pl.lit(subset_label).alias("subset"),
        # A continuous-ish "severity" Y candidate: signed n_categories with is_safe=0 → -n
        # so lower-tail = more harms triggered. Range will be [-14, 1] roughly.
        (
            pl.when(pl.col("is_safe")).then(pl.lit(1.0))
            .otherwise(-(sum(pl.col(c).cast(pl.Int8) for c in HARM_COLS).cast(pl.Float64)))
        ).alias("safety_score"),
    )
    return df


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails")
    frames = []
    if "330k_train" in ds:
        df = pl.from_arrow(ds["330k_train"].data.table)
        if isinstance(df, pl.Series):
            df = df.to_frame()
        frames.append(_process(df, "330k_train"))
    if "30k_train" in ds:
        df = pl.from_arrow(ds["30k_train"].data.table)
        if isinstance(df, pl.Series):
            df = df.to_frame()
        frames.append(_process(df, "30k_train"))
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("prompt").is_not_null()
        & (pl.col("prompt").str.len_chars() > 0)
        & pl.col("response").is_not_null()
        & (pl.col("response").str.len_chars() > 0)
    )


SPEC = DatasetSpec(
    order=6,
    slug="beavertails",
    title="BeaverTails (PKU-Alignment, 333K + 30K curated)",
    hf_id="PKU-Alignment/BeaverTails (configs: 330k_train + 30k_train)",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2307.04657",
    license="CC-BY-NC-4.0",
    gated=False,
    role_short="14-class safety meta-labels; binary is_safe + count proxy; primarily prompt+response source",
    notes=[
        "is_safe is binary — CVaR of a Bernoulli is degenerate. We synthesize `safety_score` (∈ [-14, 1]) as a quasi-continuous lower-tail-aware Y for diagnostics.",
        "14 harm categories are mutually-non-exclusive multi-labels — `primary_harm_category` is a forced unique tag for stratification.",
        "Useful as both a *prompt-mixture* source for tail-stratified Arena-Tail style construction and as a *response-side fine-tune corpus* for π_safety candidates.",
    ],
    y_candidates=["safety_score", "n_harm_categories"],
    policy_id_col=None,  # single source
    judge_col=None,
    oracle_col="is_safe",
    stratification_cols=["subset", "primary_harm_category", "is_safe"],
    prompt_col="prompt",
    response_col="response",
    pi0_proposal="Llama-3.1-8B-Instruct @ T=0.7 — open-weight base. The dataset's response field is from a mix of LLaMA-7B, LLaMA-13B, and Alpaca; we'd treat it as π0-with-noise at best.",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_premium",
            description="Claude Opus 4.7 thinking @ T=0.7",
            role="frontier comparator; expected to refuse most BeaverTails prompts (much-better tail)",
        ),
        PolicyProposal(
            name="pi_safety_dpo",
            description="Llama-3.1-8B-Instruct DPO'd on BeaverTails preference pairs",
            role="safety-fine-tuned: should match pi_premium on tail but possibly over-refuse benign prompts",
        ),
        PolicyProposal(
            name="pi_clone",
            description="Llama-3.2-3B-Instruct @ T=0.7",
            role="distilled student; expected to under-refuse on the long tail",
        ),
        PolicyProposal(
            name="pi_adversarial_jb",
            description="Llama-3.1-8B-Instruct + DAN/AIM jailbreak system prompt",
            role="audit-positive by design",
        ),
    ],
    cheap_judge_proposal="WildGuard-7B (binary refusal + harm-class) — natural fit for BeaverTails' 14-category schema; ~free at inference scale",
    oracle_judge_proposal="GPT-5-thinking grader using BeaverTails' published rubric, ideally calibrated against the dataset's own moderator labels on a 1k held-out slice; multi-judge consensus across {Opus 4.7, GPT-5} on a smaller slice for higher-stakes claims",
)
