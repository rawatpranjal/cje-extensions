"""PKU-SafeRLHF (PKU-Alignment, NeurIPS 2024) — 82K (prompt, response_0, response_1) preference triples
with severity levels, 19-class harm categories, and per-response source tags.

The dataset_options.md doc lists this as "265K Q-A safety meta-labels (19 harm
categories with severity levels)" — the 265K number aggregates multiple
sub-configs (Alpaca-7B / Alpaca2-7B / Alpaca3-8B). The default config we load
here is 73,907 train + 8,211 test = 82,118 rows × 2 responses each = 164,236
(prompt, response) cells after melting.

Per row schema (before melt):
- prompt, response_0, response_1
- prompt_source, response_0_source, response_1_source (model identifiers)
- is_response_0_safe, is_response_1_safe (bool)
- response_X_harm_category: struct of 19 booleans
- response_X_severity_level: int (0..3 typical) — *the* CVaR-relevant continuous-ish Y
- better_response_id, safer_response_id (preference labels)

After melting we get one row per (prompt, response) with:
- severity_level (int, 0-3 typical), is_safe (bool), n_harm_categories (count of 19 flags)
- response_source (the policy_id_col!) — naturally multi-policy

License: CC-BY-NC-4.0 (per HF card).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal


HARM_COLS = [
    "Endangering National Security", "Insulting Behavior", "Discriminatory Behavior",
    "Endangering Public Health", "Copyright Issues", "Violence", "Drugs",
    "Privacy Violation", "Economic Crime", "Mental Manipulation", "Human Trafficking",
    "Physical Harm", "Sexual Content", "Cybercrime", "Disrupting Public Order",
    "Environmental Damage", "Psychological Harm", "White-Collar Crime", "Animal Abuse",
]


def _melt_to_per_response(df: pl.DataFrame) -> pl.DataFrame:
    # Build two parallel frames: one per response_id
    parts = []
    for rid in (0, 1):
        sub = df.select(
            "prompt",
            "prompt_source",
            pl.col(f"response_{rid}").alias("response"),
            pl.col(f"response_{rid}_source").alias("response_source"),
            pl.col(f"is_response_{rid}_safe").alias("is_safe"),
            pl.col(f"response_{rid}_harm_category").alias("harm_category_struct"),
            pl.col(f"response_{rid}_severity_level").alias("severity_level"),
            pl.col("better_response_id"),
            pl.col("safer_response_id"),
        ).with_columns(pl.lit(rid).alias("response_id"))
        # Flatten the 19-bool harm-category struct.
        sub = sub.unnest("harm_category_struct")
        # Sum boolean harm columns.
        sub = sub.with_columns(
            sum(pl.col(c).cast(pl.Int8) for c in HARM_COLS).alias("n_harm_categories"),
            # First triggered category as primary tag (or null).
            pl.coalesce([
                pl.when(pl.col(c)).then(pl.lit(c)).otherwise(None) for c in HARM_COLS
            ]).alias("primary_harm_category"),
            # was-this-the-preferred response?
            (pl.col("better_response_id") == pl.col("response_id")).alias("is_preferred"),
            (pl.col("safer_response_id") == pl.col("response_id")).alias("is_safer_label"),
        )
        parts.append(sub)
    melted = pl.concat(parts, how="vertical_relaxed")
    return melted


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")
    frames = []
    for split in ("train", "test"):
        if split in ds:
            df = pl.from_arrow(ds[split].data.table)
            if isinstance(df, pl.Series):
                df = df.to_frame()
            df = df.with_columns(pl.lit(split).alias("orig_split"))
            frames.append(df)
    raw = pl.concat(frames, how="vertical_relaxed")
    return _melt_to_per_response(raw)


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("prompt").is_not_null() & (pl.col("prompt").str.len_chars() > 0)
        & pl.col("response").is_not_null() & (pl.col("response").str.len_chars() > 0)
    )


SPEC = DatasetSpec(
    order=8,
    slug="pku_saferlhf",
    title="PKU-SafeRLHF (82K prompts × 2 responses melted to 164K rows)",
    hf_id="PKU-Alignment/PKU-SafeRLHF",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2406.15513",
    license="CC-BY-NC-4.0",
    gated=False,
    role_short="severity_level continuous-ish; 19 harm classes; multi-policy native (Alpaca variants)",
    notes=[
        "Original is pair-structured (response_0/response_1); we melt to one row per response so policy_id == response_source.",
        "`severity_level` is integer (0–3 typical) — better than binary but still discrete; tie analysis will flag this.",
        "is_preferred and is_safer_label flags identify the human-preferred response within each pair — useful as a Y-correlated covariate.",
        "Multi-policy native via `response_source` (Alpaca-7B, Alpaca2-7B, Alpaca3-8B).",
    ],
    y_candidates=["severity_level", "n_harm_categories"],
    policy_id_col="response_source",
    judge_col=None,
    oracle_col="severity_level",
    stratification_cols=["primary_harm_category", "is_safe", "orig_split"],
    prompt_col="prompt",
    response_col="response",
    pi0_proposal="In-data: `Alpaca-7B` is the natural π0 (the un-aligned baseline; produces the worst tail).",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_prime_alpaca3_8B",
            description="In-data `Alpaca3-8B` (or 70B variant)",
            role="distilled-DPO descendant; expected better mean and tail than π0",
        ),
        PolicyProposal(
            name="pi_prime_safety_dpo",
            description="External: Llama-3.1-8B + DPO on PKU-SafeRLHF preference pairs",
            role="aligned variant; expected to dominate tail at the cost of some helpfulness mean",
        ),
        PolicyProposal(
            name="pi_premium",
            description="External: Claude Opus 4.7 thinking @ T=0.7",
            role="frontier comparator; expected to dominate on both mean and tail",
        ),
        PolicyProposal(
            name="pi_adversarial_jb",
            description="External: π_base + DAN/AIM jailbreak",
            role="audit-positive by design",
        ),
    ],
    cheap_judge_proposal="WildGuard-7B (binary refusal + 13-class harm) — fits PKU-SafeRLHF's 19-class schema reasonably; ~free at inference",
    oracle_judge_proposal="GPT-5-thinking severity grader (1–4 scale matching PKU's published rubric); for higher-stakes claims, multi-judge consensus across {Opus 4.7, GPT-5}",
)
