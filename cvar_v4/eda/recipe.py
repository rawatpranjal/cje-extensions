"""20-block EDA + scoping recipe. Each block is a function `(ctx) -> str`
producing one markdown section. polars-only; no pandas calls anywhere."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import math
import numpy as np
import polars as pl

from . import _utils as U
from .datasets._base import DatasetSpec


# -------- context --------

@dataclass
class Ctx:
    spec: DatasetSpec
    df: pl.DataFrame  # post-load, post-filter
    df_raw: pl.DataFrame  # pre-filter (to compute usable_n delta)
    splits_info: dict[str, int]  # {split_name: n_rows}
    on_disk_bytes: int
    derived: dict[str, Any] = field(default_factory=dict)  # blocks can stash here for later blocks


# -------- helpers used by multiple blocks --------

def _best_y_col(spec: DatasetSpec, df: pl.DataFrame) -> Optional[str]:
    for c in spec.y_candidates:
        if c in df.columns and df[c].dtype.is_numeric():
            return c
    return None


def _is_continuous(s: pl.Series, threshold: int = 50) -> bool:
    """Continuous = numeric with > threshold distinct values."""
    if not s.dtype.is_numeric():
        return False
    return s.n_unique() > threshold


def _classify_y(s: pl.Series) -> str:
    if not s.dtype.is_numeric():
        if s.dtype == pl.Boolean:
            return "binary-bool"
        return f"non-numeric ({s.dtype})"
    nu = s.n_unique()
    lo, hi = float(s.min() or 0), float(s.max() or 0)
    if nu <= 2:
        return "binary"
    if nu <= 7 and all(float(v).is_integer() for v in s.unique() if v is not None):
        return f"Likert-{nu}"
    if nu > 50:
        return f"continuous (n_distinct={nu}, range=[{lo:.4g}, {hi:.4g}])"
    return f"discrete-{nu} (range=[{lo:.4g}, {hi:.4g}])"


# -------- the 20 blocks --------

def block_01_identification(ctx: Ctx) -> str:
    s = ctx.spec
    lines = [
        "### 1. Identification & access",
        f"- **HF id**: `{s.hf_id}`",
        f"- **Title**: {s.title}",
        f"- **Paper / arXiv**: {s.paper_id or 'n/a'}",
        f"- **License**: {s.license}",
        f"- **Gated**: {'**yes** — `huggingface-cli login` + terms acceptance required' if s.gated else 'no'}",
        f"- **Role**: {s.role_short or 'n/a'}",
    ]
    if s.notes:
        lines.append("- **Notes**:")
        for n in s.notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


def block_02_on_disk(ctx: Ctx) -> str:
    return (
        "### 2. On-disk footprint after download\n"
        f"- **Bytes**: {U.fmt_mib(ctx.on_disk_bytes)} ({ctx.on_disk_bytes:,} bytes)\n"
        f"- **Cleanup target**: `~/.cache/huggingface/hub/datasets--*{ctx.spec.slug}*` (post-section rm -rf)"
    )


def block_03_splits(ctx: Ctx) -> str:
    if not ctx.splits_info:
        return "### 3. Splits\n- (single-split or streaming; n_rows reported in block 6)"
    lines = ["### 3. Splits"]
    total = sum(ctx.splits_info.values())
    for split, n in ctx.splits_info.items():
        lines.append(f"- **{split}**: {n:,} rows")
    lines.append(f"- **Total across splits**: {total:,}")
    return "\n".join(lines)


def block_04_first_row(ctx: Ctx) -> str:
    if ctx.df.height == 0:
        return "### 4. First row\n- (empty dataframe)"
    row = ctx.df.head(1).row(0, named=True)
    lines = ["### 4. First row (truncated to 300 chars per field)"]
    lines.append("```")
    for k, v in row.items():
        lines.append(f"  {k}: {U.truncate(repr(v), 300)}")
    lines.append("```")
    return "\n".join(lines)


def block_05_field_inventory(ctx: Ctx) -> str:
    rows = U.df_describe(ctx.df)
    table_rows = [
        [r["col"], r["dtype"], f"{r['null_pct']:.1f}%", f"{r['empty_pct']:.1f}%", str(r["n_unique"]) if r["n_unique"] >= 0 else "-"]
        for r in rows
    ]
    return "### 5. Field inventory\n" + U.md_table(
        table_rows, ["column", "dtype", "%null", "%empty (str)", "n_unique"]
    )


def block_06_usable_n(ctx: Ctx) -> str:
    raw_n = ctx.df_raw.height
    use_n = ctx.df.height
    dropped = raw_n - use_n
    pct = (dropped / max(raw_n, 1)) * 100
    return (
        "### 6. Usable n after filtering\n"
        f"- **Raw rows**: {raw_n:,}\n"
        f"- **After filter (drop-null prompt/response, drop redacted)**: {use_n:,}\n"
        f"- **Dropped**: {dropped:,} ({pct:.2f}%)"
    )


def block_07_y_candidates(ctx: Ctx) -> str:
    s = ctx.spec
    if not s.y_candidates:
        return (
            "### 7. Y candidate enumeration\n"
            "- **No Y candidates declared in spec.** This dataset is treated as a *prompt source* "
            "for fresh policy generation; Y must be elicited externally (cheap S + oracle Y, see block 19)."
        )
    lines = ["### 7. Y candidate enumeration"]
    table = []
    for c in s.y_candidates:
        if c not in ctx.df.columns:
            table.append([c, "MISSING", "-", "-", "-"])
            continue
        col = ctx.df[c]
        cls = _classify_y(col)
        try:
            top5 = col.value_counts().sort("count", descending=True).head(5)
            top_str = ", ".join(f"{r[c]}={r['count']}" for r in top5.iter_rows(named=True))
        except Exception:
            top_str = "(value_counts unavailable)"
        flag = "✅" if "continuous" in cls else "⚠️ DISCRETE"
        table.append([c, str(col.dtype), cls, flag, U.truncate(top_str, 80)])
    lines.append(U.md_table(table, ["candidate", "dtype", "classification", "CVaR-OK?", "top-5 modal values"]))
    return "\n".join(lines)


def block_08_score_distribution(ctx: Ctx) -> str:
    s = ctx.spec
    cands = [c for c in s.y_candidates if c in ctx.df.columns and ctx.df[c].dtype.is_numeric()]
    if not cands:
        return "### 8. Score distribution\n- (no numeric Y candidates)"
    out = ["### 8. Score distribution per Y candidate"]
    qs_to_show = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    for c in cands:
        col = ctx.df[c].drop_nulls()
        if col.len() == 0:
            out.append(f"\n**{c}**: all null")
            continue
        try:
            mean = float(col.mean())
            std = float(col.std() or 0.0)
            n = col.len()
        except Exception:
            mean, std, n = float("nan"), float("nan"), col.len()
        qs = U.quantiles(col, qs_to_show)
        out.append(f"\n**{c}**  (n={n:,}, mean={mean:.4g}, std={std:.4g})")
        out.append("")
        out.append("| q | " + " | ".join(f"{q:g}" for q in qs_to_show) + " |")
        out.append("|---|" + "|".join(["---"] * len(qs_to_show)) + "|")
        out.append("| value | " + " | ".join(f"{qs[q]:.4g}" for q in qs_to_show) + " |")
        out.append("")
        out.append("Histogram (12 bins):")
        out.append("```")
        out.append(U.ascii_histogram(col.to_list(), bins=12, width=40))
        out.append("```")
    return "\n".join(out)


def block_09_ties(ctx: Ctx) -> str:
    s = ctx.spec
    cands = [c for c in s.y_candidates if c in ctx.df.columns and ctx.df[c].dtype.is_numeric()]
    if not cands:
        return "### 9. Tie analysis\n- (no numeric Y)"
    out = ["### 9. Tie analysis at quantile (CVaR-CJE structural check)"]
    table = []
    for c in cands:
        col = ctx.df[c].drop_nulls()
        if col.len() < 100:
            table.append([c, "n<100", "-", "-", "-", "-"])
            continue
        for q in (0.05, 0.01):
            qv, frac, n_tie = U.tie_fraction_at_quantile(col, q, eps_rel=0.005)
            disq = "⚠️ STRUCTURAL FAIL" if frac >= 0.10 else ("⚠️ borderline" if frac >= 0.03 else "✅")
            table.append([c, f"q_{{{q}}}", f"{qv:.4g}", f"{n_tie:,}", f"{frac*100:.2f}%", disq])
    out.append(U.md_table(table, ["candidate", "quantile", "value", "n_ties (±0.5%·range)", "tie-frac", "verdict"]))
    out.append("\n*Threshold: ≥10% tie fraction at q_α makes empirical CVaR_α discontinuous in the data → CVaR-CJE inadmissible without Y smoothing.*")
    return "\n".join(out)


def block_10_tail_mass(ctx: Ctx) -> str:
    s = ctx.spec
    cands = [c for c in s.y_candidates if c in ctx.df.columns and ctx.df[c].dtype.is_numeric()]
    if not cands:
        return "### 10. Tail-mass diagnostic\n- (no numeric Y)"
    out = ["### 10. Tail-mass diagnostic"]
    table = []
    for c in cands:
        col = ctx.df[c].drop_nulls()
        if col.len() < 200:
            table.append([c, "n<200", "-", "-", "-", "-"])
            continue
        arr = col.to_numpy()
        mu = float(np.mean(arr))
        cvar5, q5 = U.cvar_estimate(col, 0.05)
        cvar1, q1 = U.cvar_estimate(col, 0.01)
        # bootstrap CI for CVaR_0.05
        lo, hi = U.bootstrap_ci(arr, lambda a: float(np.mean(a[a <= np.quantile(a, 0.05)])) if (a <= np.quantile(a, 0.05)).any() else float("nan"), B=500)
        ratio5 = cvar5 / mu if mu != 0 else float("nan")
        ratio1 = cvar1 / mu if mu != 0 else float("nan")
        table.append([
            c,
            f"{mu:.4g}",
            f"{cvar5:.4g}  [{lo:.4g}, {hi:.4g}]",
            f"{ratio5:.3f}",
            f"{cvar1:.4g}",
            f"{ratio1:.3f}",
        ])
        ctx.derived.setdefault("tail", {})[c] = {
            "mean": mu, "cvar_05": cvar5, "cvar_01": cvar1, "q_05": q5, "q_01": q1, "n": col.len(),
        }
    out.append(U.md_table(table, ["candidate", "mean", "CVaR_0.05 (boot 95%)", "CVaR_0.05/mean", "CVaR_0.01", "CVaR_0.01/mean"]))
    out.append("\n*Interpretation: CVaR/mean << 1 means the lower tail is far from the body — the regime where CVaR-CJE adds value over mean estimation.*")
    return "\n".join(out)


def block_11_multi_policy(ctx: Ctx) -> str:
    s = ctx.spec
    if not s.policy_id_col or s.policy_id_col not in ctx.df.columns:
        # heuristic scan
        candidates = [c for c in ctx.df.columns if any(k in c.lower() for k in ("model", "policy", "source", "engine"))]
        if not candidates:
            return "### 11. Multi-policy nativity\n- **No policy-id column detected.** Single-source data; would need fresh policy generation for CJE."
        info = "; ".join(f"`{c}` ({ctx.df[c].n_unique()} distinct)" for c in candidates)
        return f"### 11. Multi-policy nativity\n- spec.policy_id_col not set; heuristic candidates: {info}"

    col = ctx.df[s.policy_id_col]
    distinct = col.n_unique()
    counts = col.value_counts().sort("count", descending=True).head(20)
    rows = [[r[s.policy_id_col], r["count"], f"{r['count']/ctx.df.height*100:.1f}%"] for r in counts.iter_rows(named=True)]
    out = [
        "### 11. Multi-policy nativity",
        f"- **policy_id_col**: `{s.policy_id_col}`",
        f"- **n distinct policies**: {distinct}",
        "",
        "**Top policies by row count:**",
        U.md_table(rows, [s.policy_id_col, "n_rows", "share"]),
    ]
    if distinct >= 2:
        out.append(f"\n*✅ Multi-policy native: can read off π0/π′ pairs without fresh generation.*")
        ctx.derived["multi_policy_n"] = distinct
    return "\n".join(out)


def block_12_S_Y_pair(ctx: Ctx) -> str:
    s = ctx.spec
    if not (s.judge_col and s.oracle_col and s.judge_col in ctx.df.columns and s.oracle_col in ctx.df.columns):
        return (
            "### 12. Existing cheap-judge / oracle pair\n"
            f"- **judge_col**: {s.judge_col or '(none declared)'}\n"
            f"- **oracle_col**: {s.oracle_col or '(none declared)'}\n"
            f"- *Both must be elicited externally (see block 19).*"
        )
    df = ctx.df.select([s.judge_col, s.oracle_col]).drop_nulls()
    n = df.height
    if n < 100:
        return f"### 12. Existing cheap-judge / oracle pair\n- (n={n} after dropna; too small for correlation)"
    sub = df.head(min(n, 5000))
    a = sub[s.judge_col].to_numpy().astype(float)
    b = sub[s.oracle_col].to_numpy().astype(float)
    pearson = float(np.corrcoef(a, b)[0, 1])
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
    # tail-region: bottom 10% of judge S
    q10 = np.quantile(a, 0.10)
    mask = a <= q10
    tail_corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() >= 10 else float("nan")
    return (
        "### 12. Existing cheap-judge / oracle pair\n"
        f"- **judge_col / oracle_col**: `{s.judge_col}` / `{s.oracle_col}`  (n={n:,}, sub={sub.height:,})\n"
        f"- **Pearson(S, Y)**: {pearson:.3f}\n"
        f"- **Spearman(S, Y)**: {spearman:.3f}\n"
        f"- **Tail-region corr (S ≤ q_0.10(S))**: {tail_corr:.3f}\n"
        f"- *Match to original CJE Arena (~0.5–0.8 correlation) is the design target.*"
    )


def block_13_stratification(ctx: Ctx) -> str:
    s = ctx.spec
    if not s.stratification_cols:
        return "### 13. Audit-discriminative metadata\n- (no stratification columns declared)"
    out = ["### 13. Audit-discriminative metadata (stratification covariates)"]
    for c in s.stratification_cols:
        if c not in ctx.df.columns:
            out.append(f"\n**{c}** — MISSING from data")
            continue
        col = ctx.df[c]
        nu = col.n_unique()
        cov = (col.is_not_null().sum() / max(ctx.df.height, 1)) * 100
        try:
            top = col.value_counts().sort("count", descending=True).head(8)
            top_str = "; ".join(f"`{r[c]}`={r['count']}" for r in top.iter_rows(named=True))
        except Exception:
            top_str = "(unavailable)"
        out.append(f"\n**{c}** (cardinality={nu}, coverage={cov:.1f}%): {top_str}")
    return "\n".join(out)


def block_14_per_stratum_tail(ctx: Ctx) -> str:
    s = ctx.spec
    y = _best_y_col(s, ctx.df)
    if not y or not s.stratification_cols:
        return "### 14. Per-stratum tail-mass heterogeneity\n- (skipped — needs both Y candidate and stratification cols)"
    out = ["### 14. Per-stratum tail-mass heterogeneity (audit-discriminativeness)"]
    out.append(f"Y = `{y}`; for each stratification column, the bottom-5% mean per level (top-8 levels):\n")
    for c in s.stratification_cols:
        if c not in ctx.df.columns:
            continue
        try:
            grouped = (
                ctx.df.lazy()
                .filter(pl.col(c).is_not_null() & pl.col(y).is_not_null())
                .group_by(c)
                .agg(
                    pl.col(y).count().alias("n"),
                    pl.col(y).mean().alias("mean"),
                    pl.col(y).quantile(0.05, "linear").alias("q_05"),
                )
                .filter(pl.col("n") >= 50)
                .sort("q_05")
                .head(8)
                .collect()
            )
        except Exception as e:
            out.append(f"\n**{c}**: error — {e}")
            continue
        rows = [[U.truncate(str(r[c]), 60), f"{r['n']:,}", f"{r['mean']:.4g}", f"{r['q_05']:.4g}"] for r in grouped.iter_rows(named=True)]
        out.append(f"\n**{c}** (sorted by q_0.05 ascending; lowest-tail strata first):")
        out.append(U.md_table(rows, [c, "n", "mean(Y)", "q_0.05(Y)"]))
        if grouped.height >= 2:
            spread = float(grouped["q_05"].max() - grouped["q_05"].min())
            out.append(f"\n*Heterogeneity: q_0.05 spread across levels = {spread:.4g}. Larger → audit transport test will be more discriminative.*")
    return "\n".join(out)


def block_15_prompt_length(ctx: Ctx) -> str:
    s = ctx.spec
    if not s.prompt_col or s.prompt_col not in ctx.df.columns:
        return "### 15. Prompt-length distribution\n- (no prompt column declared)"
    sub = ctx.df.head(min(ctx.df.height, 5000))[s.prompt_col].to_list()
    toks = [U.count_tokens(t) for t in sub]
    arr = np.array(toks)
    qs = {q: int(np.quantile(arr, q)) for q in (0.10, 0.50, 0.90, 0.99)}
    total_est = int(np.mean(arr) * ctx.df.height)
    ctx.derived.setdefault("len", {})["prompt_mean"] = float(np.mean(arr))
    ctx.derived["len"]["prompt_total_est"] = total_est
    return (
        "### 15. Prompt-length distribution (tiktoken cl100k_base)\n"
        f"- subsample n={len(toks):,}\n"
        f"- p10/p50/p90/p99 = {qs[0.10]}/{qs[0.50]}/{qs[0.90]}/{qs[0.99]}\n"
        f"- mean tokens/prompt = {float(np.mean(arr)):.1f}\n"
        f"- **total prompt tokens (est across full {ctx.df.height:,} rows)**: ~{total_est:,}"
    )


def block_16_response_length(ctx: Ctx) -> str:
    s = ctx.spec
    if not s.response_col or s.response_col not in ctx.df.columns:
        return "### 16. Response-length distribution\n- (no response column declared — generation cost depends on policy choice)"
    sub_series = ctx.df[s.response_col]
    if sub_series.dtype == pl.List(pl.Utf8) or sub_series.dtype == pl.List(pl.Struct):
        # nested response (e.g., multi-turn); fall back to count
        return f"### 16. Response-length distribution\n- (response column `{s.response_col}` is nested {sub_series.dtype}; per-turn handling deferred)"
    sub = sub_series.head(min(ctx.df.height, 5000)).to_list()
    toks = [U.count_tokens(t if isinstance(t, str) else None) for t in sub]
    arr = np.array(toks)
    qs = {q: int(np.quantile(arr, q)) for q in (0.10, 0.50, 0.90, 0.99)}
    total_est = int(np.mean(arr) * ctx.df.height)
    ctx.derived.setdefault("len", {})["response_mean"] = float(np.mean(arr))
    ctx.derived["len"]["response_total_est"] = total_est
    return (
        "### 16. Response-length distribution (tiktoken cl100k_base)\n"
        f"- subsample n={len(toks):,}\n"
        f"- p10/p50/p90/p99 = {qs[0.10]}/{qs[0.50]}/{qs[0.90]}/{qs[0.99]}\n"
        f"- mean tokens/response = {float(np.mean(arr)):.1f}\n"
        f"- **total response tokens (est across full {ctx.df.height:,} rows)**: ~{total_est:,}"
    )


def block_17_tail_samples(ctx: Ctx) -> str:
    s = ctx.spec
    y = _best_y_col(s, ctx.df)
    if not y:
        return "### 17. Tail samples\n- (no numeric Y candidate to sort by)"
    df = ctx.df.filter(pl.col(y).is_not_null())
    if df.height == 0:
        return "### 17. Tail samples\n- (Y all null)"
    bottom = df.sort(y).head(5)
    out = [f"### 17. Tail samples (bottom 5 by `{y}`)"]
    for i, row in enumerate(bottom.iter_rows(named=True), 1):
        out.append(f"\n**Sample {i}** — Y={row[y]}")
        for col in (s.prompt_col, s.response_col):
            if col and col in row:
                out.append(f"- `{col}`: {U.truncate(str(row[col]), 250)}")
        for col in s.stratification_cols:
            if col in row and row[col] is not None:
                out.append(f"- `{col}`: {U.truncate(str(row[col]), 80)}")
    return "\n".join(out)


def block_18_median_samples(ctx: Ctx) -> str:
    s = ctx.spec
    y = _best_y_col(s, ctx.df)
    if not y:
        return "### 18. Median samples\n- (no numeric Y to sort by)"
    df = ctx.df.filter(pl.col(y).is_not_null())
    if df.height == 0:
        return "### 18. Median samples\n- (Y all null)"
    q45 = float(df[y].quantile(0.45, "linear"))
    q55 = float(df[y].quantile(0.55, "linear"))
    band = df.filter((pl.col(y) >= q45) & (pl.col(y) <= q55)).head(3)
    out = [f"### 18. Median samples (3 rows in [q_0.45, q_0.55] = [{q45:.4g}, {q55:.4g}] of `{y}`)"]
    for i, row in enumerate(band.iter_rows(named=True), 1):
        out.append(f"\n**Sample {i}** — Y={row[y]}")
        for col in (s.prompt_col, s.response_col):
            if col and col in row:
                out.append(f"- `{col}`: {U.truncate(str(row[col]), 250)}")
    return "\n".join(out)


def block_19_pi_proposal(ctx: Ctx) -> str:
    s = ctx.spec
    out = ["### 19. π0 / π′ / S / Y proposal (CVaR-CJE scoping)"]
    out.append(f"- **π_base (logger π0)**: {s.pi0_proposal}")
    out.append("- **π′ panel**:")
    panel = s.pi_prime_panel or []
    for p in panel:
        out.append(f"  - **{p.name}** — {p.description}  *({p.role})*")
    if not panel:
        out.append("  - (use generic 5-policy panel from dataset_options.md §B.2)")
    out.append(f"- **Cheap judge S**: {s.cheap_judge_proposal}")
    out.append(f"- **Oracle Y**: {s.oracle_judge_proposal}")
    if ctx.derived.get("multi_policy_n", 0) >= 2:
        out.append(f"\n*Note: `{s.policy_id_col}` already partitions {ctx.derived['multi_policy_n']} policies in the data; no fresh generation is required for the multi-policy comparison itself — only the cheap-S / oracle-Y elicitation.*")
    return "\n".join(out)


def block_20_cost_resolution(ctx: Ctx) -> str:
    """Cost estimate (concrete @ 2026-04-28 rates) + CVaR_α resolution feasibility."""
    s = ctx.spec
    n_total = ctx.df.height
    out = ["### 20. Cost & CVaR-resolution feasibility"]

    # Generation cost (only if not multi-policy native)
    n_policies = ctx.derived.get("multi_policy_n") or len(s.pi_prime_panel) + 1  # +1 for π0
    p_mean = ctx.derived.get("len", {}).get("prompt_mean")
    r_mean = ctx.derived.get("len", {}).get("response_mean") or 400  # fallback assumption

    multi_policy = ctx.derived.get("multi_policy_n", 0) >= 2
    out.append("\n**Generation cost (rough @ 2026-04-28 API rates)**:")
    if multi_policy:
        out.append("- *Policies are already in-data (`policy_id` partitions {} cells).*".format(ctx.derived["multi_policy_n"]))
        out.append("- *Marginal generation cost for the multi-policy comparison itself = $0.* The numbers below are the hypothetical fresh-generation cost if you wanted to *augment* the in-data panel with extra π′ candidates (e.g., a frontier model not in the original 12-cell grid).")
    if not p_mean:
        out.append("- prompt-length not measured (no prompt_col); skipping per-token estimate")
    else:
        # closed-frontier rate (Opus 4.7 thinking): $15/M-in, $75/M-out
        # open-weight via Together (Llama-3.1-8B): $0.20/M-in, $0.20/M-out
        in_toks = n_total * p_mean
        out_toks = n_total * r_mean  # we'd generate r_mean tokens per row
        cost_open = (in_toks * 0.20 + out_toks * 0.20) / 1e6
        cost_frontier = (in_toks * 15 + out_toks * 75) / 1e6
        out.append(f"- per-policy: in={in_toks/1e6:.2f}M, out={out_toks/1e6:.2f}M tokens (assuming ~{r_mean:.0f} out-tok/row)")
        out.append(f"- per-policy cost: open-weight (Together Llama-3.1-8B @ $0.20/M) ≈ **${cost_open:.0f}**; frontier (Opus 4.7 thinking @ $15/$75 per M) ≈ **${cost_frontier:,.0f}**")
        # full panel:
        n_open = max(0, n_policies - 2)  # logger + clone open; rest closed
        n_closed = max(0, n_policies - n_open)
        full = n_open * cost_open + n_closed * cost_frontier
        out.append(f"- full {n_policies}-policy panel (assume {n_open} open + {n_closed} closed): ≈ **${full:,.0f}**")

    # Oracle cost (assuming GPT-5-thinking @ ~$10/M-in, $40/M-out; per-call ~3000 in + 1000 out tokens conservative)
    oracle_per_call = (3000 * 10 + 1000 * 40) / 1e6  # ≈ $0.07
    for cov in (0.05, 0.15, 0.25):
        slice_n = int(n_total * cov)
        cost = slice_n * oracle_per_call * n_policies
        out.append(f"- oracle slice @ {cov*100:.0f}% × {n_policies} policies = {slice_n*n_policies:,} calls × ~${oracle_per_call:.3f}/call ≈ **${cost:,.0f}** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)")

    # CVaR-resolution feasibility
    out.append("\n**CVaR-α resolution feasibility (Wald 95%-CI half-width on existing Y)**:")
    y = _best_y_col(s, ctx.df)
    if not y:
        out.append("- (no Y in data; resolution must be re-estimated after S/Y elicitation)")
    else:
        col = ctx.df[y].drop_nulls()
        if col.len() < 1000:
            out.append(f"- n={col.len()}; below 1000-row CVaR_0.05 floor; insufficient")
        else:
            arr = col.to_numpy().astype(float)
            for alpha in (0.05, 0.01):
                hw = U.cvar_ci_halfwidth(arr, alpha, n=arr.size)
                cvar_val = float(np.mean(arr[arr <= np.quantile(arr, alpha)])) if (arr <= np.quantile(arr, alpha)).any() else float("nan")
                rel = (hw / abs(cvar_val) * 100) if cvar_val and not math.isnan(cvar_val) else float("nan")
                ok = "✅ resolvable" if rel < 20 else ("⚠️ borderline" if rel < 50 else "❌ insufficient")
                out.append(f"- α={alpha}: CVaR≈{cvar_val:.4g}, half-width≈{hw:.4g} ({rel:.1f}% of CVaR) — {ok}  [n={arr.size:,}]")
            out.append("\n*Threshold: CI half-width ≤ 20% of CVaR is a comfortable resolution; 20–50% is publishable but tight; >50% means scale up n or reduce α aggressiveness.*")

    return "\n".join(out)


# -------- runner & renderer --------

ALL_BLOCKS = (
    block_01_identification,
    block_02_on_disk,
    block_03_splits,
    block_04_first_row,
    block_05_field_inventory,
    block_06_usable_n,
    block_07_y_candidates,
    block_08_score_distribution,
    block_09_ties,
    block_10_tail_mass,
    block_11_multi_policy,
    block_12_S_Y_pair,
    block_13_stratification,
    block_14_per_stratum_tail,
    block_15_prompt_length,
    block_16_response_length,
    block_17_tail_samples,
    block_18_median_samples,
    block_19_pi_proposal,
    block_20_cost_resolution,
)


def run_all(ctx: Ctx) -> str:
    """Run all 20 blocks in order; render full markdown section."""
    s = ctx.spec
    head = [
        f"# §{s.order:02d}. {s.title} (`{s.slug}`)\n",
        f"_HF id: `{s.hf_id}` · order #{s.order} · n_usable={ctx.df.height:,}_\n",
    ]
    rendered = []
    for fn in ALL_BLOCKS:
        try:
            rendered.append(fn(ctx))
        except Exception as e:
            rendered.append(f"### {fn.__name__}\n- ERROR: {type(e).__name__}: {e}")
    return "\n\n".join(head + rendered) + "\n"
