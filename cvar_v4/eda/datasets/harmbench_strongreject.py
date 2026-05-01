"""HarmBench (1528 behavior variants) + StrongREJECT (323 prompts) — folded.

Both are prompt-only adversarial-tail seeds. We concatenate them with a
`dataset_source` column so the EDA harness can see them as one and use
`dataset_source` as a stratification covariate.

HarmBench: Mazeika et al. 2024 (Center for AI Safety) — arXiv 2402.04249
StrongREJECT: Souly et al. NeurIPS 2024 — arXiv 2402.10260

The HF mirrors at `walledai/HarmBench` and `walledai/StrongREJECT` are gated;
we pull the canonical CSVs directly from each project's GitHub repo (both
ship under permissive open-source licenses — HarmBench MIT, StrongREJECT MIT).
"""
from __future__ import annotations

import io
import subprocess

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
STRONGREJECT_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)


def _http_get(url: str) -> bytes:
    """Use curl rather than urllib — system curl already trusts the right CA bundle on macOS,
    while the python.org Python ships without one configured."""
    out = subprocess.run(
        ["curl", "-sLf", "--max-time", "60", url],
        capture_output=True,
        check=True,
    )
    return out.stdout


def _load(spec) -> pl.DataFrame:
    """Pull both CSVs from GitHub, normalize, concatenate."""
    frames: list[pl.DataFrame] = []

    # ---- HarmBench ----
    hb_bytes = _http_get(HARMBENCH_URL)
    hb_df = pl.read_csv(io.BytesIO(hb_bytes))
    # canonical columns: Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID
    hb_df = hb_df.rename({"Behavior": "prompt"})
    # Keep both category granularities so block 13/14 has options.
    hb_df = hb_df.rename({"SemanticCategory": "category", "FunctionalCategory": "subcategory"})
    hb_df = hb_df.with_columns(pl.lit("HarmBench").alias("dataset_source"))
    keep = ["prompt", "category", "subcategory", "Tags", "BehaviorID", "dataset_source"]
    hb_df = hb_df.select([c for c in keep if c in hb_df.columns])
    frames.append(hb_df)

    # ---- StrongREJECT ----
    sr_bytes = _http_get(STRONGREJECT_URL)
    sr_df = pl.read_csv(io.BytesIO(sr_bytes))
    # canonical columns: category, source, forbidden_prompt
    sr_df = sr_df.rename({"forbidden_prompt": "prompt", "source": "subcategory"})
    sr_df = sr_df.with_columns(pl.lit("StrongREJECT").alias("dataset_source"))
    sr_df = sr_df.select([c for c in ("prompt", "category", "subcategory", "dataset_source") if c in sr_df.columns])
    frames.append(sr_df)

    # Align columns then concat
    all_cols = sorted(set().union(*(set(f.columns) for f in frames)))
    aligned = []
    for f in frames:
        f2 = f
        for c in all_cols:
            if c not in f2.columns:
                f2 = f2.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))
        aligned.append(f2.select(all_cols))
    return pl.concat(aligned, how="vertical_relaxed")


SPEC = DatasetSpec(
    order=1,
    slug="harmbench_strongreject",
    title="HarmBench + StrongREJECT (folded adversarial-tail seed)",
    hf_id="GitHub: centerforaisafety/HarmBench + alexandrasouly/strongreject (HF mirrors gated)",
    custom_loader=_load,
    paper_id="2402.04249 + 2402.10260",
    license="CC-BY-4.0 / MIT (permissive)",
    gated=False,
    role_short="prompt-only adversarial-tail seed; CVaR-CJE prompt source, no Y in data",
    notes=[
        "Both datasets are prompt-only (no model responses, no judge or oracle scores).",
        "For CVaR-CJE these are *prompt sources* — must generate fresh policy responses + S + Y externally.",
        "HarmBench has 400 behaviors × 7 functional categories × 33 target LLMs in its eval slice; we load the prompt set only.",
        "StrongREJECT has 313 forbidden prompts in 6 categories with a continuous-output auto-grader (the grader is downstream, not in the dataset itself).",
    ],
    y_candidates=[],  # no Y in either dataset
    policy_id_col=None,
    judge_col=None,
    oracle_col=None,
    stratification_cols=["dataset_source", "category", "subcategory"],
    prompt_col="prompt",
    response_col=None,
    pi0_proposal="Llama-3.1-8B-Instruct @ T=0.7, top_p=0.95 — base instruct model; teacher-forced log-probs available for SIMCal-W weights",
    pi_prime_panel=default_pi_prime_panel(),
    cheap_judge_proposal="WildGuard-7B (prompt-harm + response-harm + refusal triple-label) — appropriate for adversarial prompts; ~free at inference scale on local GPU",
    oracle_judge_proposal="StrongREJECT auto-grader (continuous 0–1 refusal+capability score) — the dataset's own grader is the natural oracle here, validated against human-labeled jailbreaks",
)
