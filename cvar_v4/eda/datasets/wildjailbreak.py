"""WildJailbreak (allenai, NeurIPS 2024) — 261,559 synthetic safety training pairs.

Schema per row (`train` config):
- vanilla: str (the direct/vanilla prompt; non-empty only for vanilla rows)
- adversarial: str (the adversarial-rephrased prompt; non-empty only for adversarial rows)
- completion: str (model response — refusal for harmful, compliance for benign)
- data_type: str ∈ {vanilla_harmful, vanilla_benign, adversarial_harmful, adversarial_benign}

The 4-quadrant `data_type` partition is the linchpin audit covariate per
`dataset_options.md` §B.2: a π_safety policy that over-refuses adversarial
benign prompts fails on the `adversarial_benign` quadrant; a π_adversarial
policy that under-refuses adversarial harmful prompts fails on the
`adversarial_harmful` quadrant — two distinct audit-failure axes that the
audit's per-stratum reporting cleanly separates.

For CVaR-CJE the dataset is **prompt-only** (no continuous Y in the data;
`completion` is GPT-3.5-class refusal/compliance text, useful as a refusal-
demonstration corpus rather than as a CVaR-CJE Y). Its role is *prompt source*
for fresh-policy generation, plus a clean 4-quadrant audit covariate.

We synthesize one helper field for unified access:
- `prompt`: coalesce(adversarial, vanilla) — the prompt actually shown to the
  model; non-empty for every row.

License: ODC-BY (per HF card). Gating: yes (terms-of-use acceptance required).
"""
from __future__ import annotations

import polars as pl

from ._base import DatasetSpec, PolicyProposal


def _load(spec) -> pl.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(
        "allenai/wildjailbreak", "train",
        delimiter="\t", keep_default_na=False,
    )
    # DatasetDict with a single 'train' split here (see HF card).
    split = ds["train"] if hasattr(ds, "items") and "train" in ds else next(iter(ds.values())) if hasattr(ds, "items") else ds
    df = pl.from_arrow(split.data.table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    # Synthesize a single `prompt` column = adversarial when present, else vanilla.
    df = df.with_columns(
        pl.when(pl.col("adversarial").str.len_chars() > 0)
        .then(pl.col("adversarial"))
        .otherwise(pl.col("vanilla"))
        .alias("prompt"),
        # Boolean view of the 2x2 split.
        pl.col("data_type").str.starts_with("adversarial").alias("is_adversarial"),
        pl.col("data_type").str.ends_with("harmful").alias("is_harmful"),
    )
    return df


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("prompt").is_not_null()
        & (pl.col("prompt").str.len_chars() > 0)
    )


SPEC = DatasetSpec(
    order=7,
    slug="wildjailbreak",
    title="WildJailbreak (allenai/wildjailbreak `train` config — 261,559 rows × 4 quadrants)",
    hf_id="allenai/wildjailbreak",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2406.18510 (Jiang et al. NeurIPS 2024)",
    license="ODC-BY",
    gated=True,
    role_short="cleanest 4-quadrant audit-discriminative seed; prompt source (no continuous Y in data)",
    notes=[
        "4-quadrant `data_type` partition is the linchpin audit covariate (vanilla×{harmful,benign} × adversarial×{harmful,benign}).",
        "Vanilla rows have `vanilla` non-empty and `adversarial` empty; adversarial rows are the reverse. We synthesize `prompt` = whichever is non-empty.",
        "`completion` field is GPT-3.5-class refusal/compliance demonstration; it is NOT a Y candidate (no per-row score, just text).",
        "For CVaR-CJE the role is *prompt source* for fresh-policy generation, with `data_type` as the audit stratum.",
        "Adversarial split uses tactics mined from in-the-wild jailbreaks — pre-built audit-positive prompt distribution.",
    ],
    y_candidates=[],  # no Y in data; downstream pipeline must elicit S/Y
    policy_id_col=None,
    judge_col=None,
    oracle_col=None,
    stratification_cols=["data_type", "is_adversarial", "is_harmful"],
    prompt_col="prompt",
    response_col="completion",
    pi0_proposal="Llama-3.1-8B-Instruct @ T=0.7 — open-weight base. Fresh generation needed; teacher-forced log-probs available for SIMCal-W weights.",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_premium",
            description="Claude Opus 4.7 thinking @ T=0.7",
            role="frontier comparator; expected to refuse most adversarial-harmful prompts and comply with adversarial-benign prompts (correct refusal calibration on tail)",
        ),
        PolicyProposal(
            name="pi_safety_overrefuse",
            description="Llama-3.1-8B-Instruct + DPO on WildJailbreak completion pairs",
            role="**audit-positive on adversarial_benign**: over-refuses adversarial-rephrased benign prompts → audit fires on this quadrant even though the policy is 'safer' on the adversarial_harmful quadrant",
        ),
        PolicyProposal(
            name="pi_clone",
            description="Llama-3.2-3B-Instruct @ T=0.7",
            role="distilled student; expected to be brittle on adversarial prompts (under-refusal under jailbreak)",
        ),
        PolicyProposal(
            name="pi_adversarial",
            description="Llama-3.1-8B-Instruct + DAN/AIM jailbreak system prompt",
            role="**audit-positive on adversarial_harmful**: under-refuses jailbreaks → audit fires on this quadrant; should accept on vanilla_harmful and adversarial_benign quadrants if the jailbreak prompt only activates under adversarial framing",
        ),
    ],
    cheap_judge_proposal="WildGuard-7B (binary refusal + harm-class) — ideal fit since both WildGuard and WildJailbreak are AllenAI's safety pipeline; ~free at inference scale",
    oracle_judge_proposal="GPT-5-thinking grader using a refusal-quality + capability-of-harm rubric (matching WildJailbreak paper's evaluation); for higher-stakes claims, multi-judge consensus across {Opus 4.7, GPT-5}",
)
