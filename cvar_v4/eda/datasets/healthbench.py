"""HealthBench (OpenAI, May 2025) — top medical recommendation.

The OSS eval set ships 5,000 multi-turn clinical prompts, each with a
physician-written rubric (avg ~9.7 criteria/prompt; we observe ~19 in the
oss_eval dump because some prompts have many criteria). HealthBench's
canonical Y is *rubric-percentage* = (Σ points of met positive criteria) /
(Σ points of all positive criteria) ∈ [0, 1] — naturally continuous.

The dataset itself ships *only* prompts + rubric metadata; model responses
and Y are computed downstream by a GPT-4.1-class grader (or physician panel
on a slice; the paper documents F1=0.71 vs. physicians).

For the EDA we surface:
- Prompt metadata (n_turns, prompt-token length, example_tags / theme tags)
- Rubric structure (n_criteria_per_prompt, total positive/negative points,
  per-criterion point distribution, per-tag prevalence)

We also pull the `hard_*.jsonl` subset (HealthBench Hard, ~1K conversations)
and tag rows with `subset` ∈ {"oss_eval", "hard"} for stratification.

License: MIT (per OpenAI/simple-evals repo).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl

from ._base import DatasetSpec, default_pi_prime_panel, PolicyProposal

CACHE = Path("/tmp/cvar_v4_healthbench")
URLS = {
    "oss_eval": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
}


def _fetch(name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"{name}.jsonl"
    if out.exists() and out.stat().st_size > 1000:
        return out
    subprocess.run(
        ["curl", "-sLf", "--max-time", "120", URLS[name], "-o", str(out)],
        check=True,
    )
    return out


def _parse_one(line: str, subset: str) -> dict:
    d = json.loads(line)
    prompt_turns = d.get("prompt", [])
    user_first = next((t.get("content", "") for t in prompt_turns if t.get("role") == "user"), "")
    rubrics = d.get("rubrics", []) or []
    pts_pos = sum(r.get("points", 0) for r in rubrics if r.get("points", 0) > 0)
    pts_neg = sum(-r.get("points", 0) for r in rubrics if r.get("points", 0) < 0)  # store as positive magnitude
    tags = d.get("example_tags", []) or []
    # normalize theme tag
    theme = next((t.split(":", 1)[1] for t in tags if t.startswith("theme:")), None)
    cluster = next((t.split(":", 1)[1] for t in tags if t.startswith("cluster:")), None)
    return {
        "prompt_id": d.get("prompt_id"),
        "subset": subset,
        "prompt": user_first,
        "n_turns": len(prompt_turns),
        "n_criteria": len(rubrics),
        "total_positive_points": float(pts_pos),
        "total_negative_points": float(pts_neg),
        "theme": theme,
        "cluster": cluster,
        "n_tags": len(tags),
    }


def _load(spec) -> pl.DataFrame:
    rows: list[dict] = []
    for name in ("oss_eval", "hard"):
        path = _fetch(name)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_parse_one(line, subset=name))
                except json.JSONDecodeError:
                    continue
    return pl.from_dicts(rows)


def _filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("prompt").is_not_null() & (pl.col("prompt").str.len_chars() > 0))


SPEC = DatasetSpec(
    order=2,
    slug="healthbench",
    title="HealthBench (oss_eval + hard)",
    hf_id="HF: openai/healthbench (or direct openaipublic.blob.core.windows.net JSONL)",
    custom_loader=_load,
    usable_filter=_filter,
    paper_id="2505.08775",
    license="MIT (per simple-evals repo)",
    gated=False,
    role_short="primary medical recommendation; continuous rubric-% Y; worst-at-k native",
    notes=[
        "OpenAI explicitly designed HealthBench around worst-case reliability (worst-at-k curves).",
        "Y in [0, 1] = rubric-percent = (Σ points of met positive criteria) / (Σ all positive points). Continuous.",
        "Y is *not* in the data — it must be computed by grading model responses against each prompt's rubric.",
        "5,000 conversations × ~9.7 criteria/prompt × ~10 grading judges per response is the order of operations.",
        "HealthBench Hard subset is ~1K conversations where frontier models still fail badly (GPT-5 ≈ 46% vs. GPT-4o ≈ 0%).",
    ],
    # Y candidates: rubric-structure features, surfaced as numeric proxies for the EDA
    y_candidates=["total_positive_points", "n_criteria"],
    policy_id_col=None,  # no policies in data; would generate fresh
    judge_col=None,
    oracle_col=None,
    stratification_cols=["subset", "theme", "cluster"],
    prompt_col="prompt",
    response_col=None,
    pi0_proposal="Llama-3.1-8B-Instruct @ T=0.7 — open-weight clinical-tuned baseline; teacher-forced log-probs available",
    pi_prime_panel=[
        PolicyProposal(
            name="pi_clone",
            description="Llama-3.2-3B-Instruct @ T=0.7 (distilled student)",
            role="expected to underperform on tail (rare-disease, drug-interaction edge cases)",
        ),
        PolicyProposal(
            name="pi_premium_thinking",
            description="Claude Opus 4.7 thinking @ T=0.7 (extended reasoning)",
            role="frontier comparator; expected to dominate on emergency-referrals / uncertainty themes",
        ),
        PolicyProposal(
            name="pi_rag_clinical",
            description="pi_base + RAG over UpToDate / NICE guidelines",
            role="domain-knowledge lift; should improve depth/expertise themes more than emergency referrals",
        ),
        PolicyProposal(
            name="pi_safety_overrefuse",
            description="pi_base + DPO toward conservative refusal on medical advice",
            role="over-refuses ambiguous-safety prompts; tail differs from pi_premium on responding-under-uncertainty theme",
        ),
        PolicyProposal(
            name="pi_adversarial_sycophant",
            description="pi_base wrapped in 'agree with user' system prompt",
            role="audit-positive by design; should dramatically fail rubric on worst-at-k (mirrors Nature npj 2025 sycophancy results)",
        ),
    ],
    cheap_judge_proposal="Prometheus-2-7B with HealthBench rubric format — open-weight, ~free at inference; rubric-conditioned scoring",
    oracle_judge_proposal="GPT-4.1-class grader using the published HealthBench grader prompt (paper docs F1=0.71 vs. physicians); upgrade to GPT-5-thinking or multi-judge consensus for higher-stakes claims",
)
