# Claude Instructions for cvar-cje

## What This Repo Is

Research repository for **Causal Judge Evaluation (CJE)** — a framework for reliably evaluating LLM systems using cheap surrogate judges calibrated against expensive oracle labels. Paper submitted to ICLR 2026, arXiv ID `2512.11150`.

Core idea: calibrate a cheap LLM judge against a small oracle slice, audit whether calibration transports to each target policy, and use bootstrap inference to get valid uncertainty. Addresses three failures in naive LLM-as-judge evaluation: ranking inversions, CI miscoverage, and IPS failure under low overlap.

## Key Files

- `arxiv_source/` — Full LaTeX source of the paper (sections, tables, figures)
- `first_cut.ipynb` — Demo notebook: Direct Mean CJE and Direct CVaR-CJE
- `direct_cvar_cje_demo.py` — Production script with Monte Carlo sweeps and 4 synthetic scenarios
- `2512.11150v3 (2).md` — Docling-converted markdown (formulas are placeholders — use LaTeX source instead)

## Standing Instructions

### Notation in chat
The user does not read LaTeX fluently. When explaining math in chat answers (not in `.tex` files), write expressions in indexed plain-text form:
- per-observation form: `y_i`, `s_i`, `r_i = y_i - f(s_i)`, `mean(r) = (1/n) Σ_i r_i`
- use `mean()`, `var()`, `t_hat` (or `t̂`), `r_bar` (or `mean(r)`), `^` for powers
- avoid `\E[]`, `\hat{}`, `\bar{}`, `\frac{}{}`, `\alpha`, etc. — Greek letters in unicode (α, β, π, σ) are fine
- this rule applies to chat explanations only; preserve exact LaTeX when editing `.tex` files

### arXiv Papers
Always download the LaTeX source for any arXiv paper before doing anything else with it:

```bash
curl -L https://arxiv.org/src/<paper-id> -o source.tar.gz && tar -xzf source.tar.gz -C <dir>
```

Never use docling on the PDF. Never read the PDF directly for content. Use the `.tex` source — it has exact math, tables, and figures. The `.md` file in this repo has `<!-- formula-not-decoded -->` placeholders throughout; prefer `arxiv_source/` for any paper content.

### Repo-local paper library

Most cited LLM-judge and CJE-adjacent works are already in `cvar_v4/papers/<arxiv-id>_<slug>/` as full LaTeX source (Zheng 2023 MT-Bench, Lee 2025, Chen 2026, the original CJE paper, G-Eval, HealthBench, Chatbot Arena, RewardBench, CalibraEval, etc.). Before downloading anything from arXiv or reading abstract HTML, `ls cvar_v4/papers/` and check.

When the paper IS in the repo, grep broadly to get a rich picture, not a brittle snippet:
- Across all `*.tex` files in the paper directory, not just `intro.tex` or `main.tex`. Tables, appendices, and experiment sections often hold the actual numbers.
- Across multiple synonyms — for "agreement": also `agree`, `kappa`, `inter-rater`; for "bias": also `verbosity`, `position`, `self-enhancement`, `length`, `leniency`; for "calibrate": also `oracle`, `surrogate`, `prediction-powered`.
- Read the surrounding paragraph plus any table/figure referenced from the matching line. A number means little without its setup (which model, which dataset, which split, which metric, which annotators). Note caveats the paper itself flags ("this only holds when…", "we did not test…") so we don't overclaim from the cite.

When the paper is NOT in `cvar_v4/papers/`, download per the arXiv rule above AND copy the extracted source into `cvar_v4/papers/<arxiv-id>_<slug>/` so the next session doesn't re-download.

### Citations and quotes — hard evidence required

The user does not trust me by default for citations or quoted passages from papers. Past sessions have surfaced paraphrases attributed to sources that did not contain them, polarity inversions in quoted statistics, and agent-relayed quotes that turned out to be hallucinated. The discipline below is non-negotiable.

Every time I propose a citation or quote a paper, I must show:
- The verbatim text from the primary source, in quotes.
- The exact file path and line number of that text in the local paper library (or, if the paper is not yet in the repo, the path of the freshly-downloaded source).
- A side-by-side mapping from the verbatim quote to the claim I am making in the prose, so the user can verify the claim is supported and not paraphrased into a stronger statement than the source allows.

Subagent reports (Explore, general-purpose, and any web-search tooling) are leads, not evidence. If a subagent gives me a quote, I must independently grep the file myself and confirm the quote exists verbatim before relaying it. If I cannot find the quote in the source, I drop the cite and tell the user the agent's quote was unverified.

If I have only read the abstract or a search-result snippet of a paper, I say so explicitly and either read more of the source before proposing the cite, or do not propose the cite at all. "Reading the abstract" is not the same as "reading the paper."

When I am wrong about a citation, the user catches it and trust degrades. The cost of one bad cite is much higher than the cost of multiple verification passes. Default to over-verifying.

### Investigation discipline
When the user asks to investigate a claim, a paper, a number, or a piece of evidence: read the FULL primary source, not just the abstract or a search-result snippet. Get exact paragraphs and exact quotes. Cite the verbatim text and the location (section, page, table) so the user can verify. Do not paraphrase the source; do not infer numbers that the source did not state; do not let a search-engine summary stand in for the paper. If the source is an arXiv paper, download the LaTeX source per the rule above. If the source is a webpage, fetch the full HTML, not just the abstract URL. If a fact is not supported by the primary source you actually read, say so before proposing it as a citation.

## Domain Concepts

- **CJE**: Causal Judge Evaluation — calibrated surrogate estimation for LLM policy ranking
- **TTC**: Target-Typicality Coverage — logger mass on target-typical regions; gate threshold ≥ 0.70
- **CLE**: Coverage-Limited Efficiency — precision floor for IPS under low overlap, even with high ESS
- **Transport audit**: estimand-specific per-policy residual test. Mean CJE uses the mean residual `E[Y - f(S,X)] = 0`; CVaR-CJE uses tail/shortfall transport at the selected threshold. A failed mean audit does **not** automatically invalidate a CVaR claim, and a passed mean audit does **not** certify a CVaR claim. Gate each reported estimand by its own transport test.
- **Two-stage calibration**: isotonic regression (stage 1) + covariate adjustment on response length (stage 2). Paper's best estimator is `direct+cov`.
- **CVaR-CJE**: extension to lower-tail risk (CVaR at 10%) rather than mean policy value

## Current Work: CVaR-CJE on Real Arena Data (Iterative)

**Status**: iterative — expect multiple sessions, each tightening rigor. Do not treat this as a one-shot scaffold.

**Target**: produce a defensible Direct CVaR-CJE@10% estimate for each of the 5 policies (base, clone, premium, parallel_universe_prompt, unhelpful) in the authors' Chatbot Arena experiment (~4989 prompts, `data/cje_dataset.jsonl`), matched against a Direct Mean CJE baseline on the same data.

**Mean benchmark** (engineering sanity check for the Arena reproduction):
Our Mean CJE must reproduce the paper's `direct+cov` numbers for all 5 policies within **|Δ| ≤ 0.01** AND our 95% CI must contain the paper's point estimate (or, equivalently, the oracle ground truth computed from full oracle labels). Oracle ground truth per policy (from `data/responses/{policy}_responses.jsonl`, full-oracle average):
- base ≈ 0.758, clone ≈ 0.762, premium ≈ 0.762, parallel_universe_prompt ≈ 0.771, unhelpful ≈ 0.143

If this benchmark fails, halt and debug the Arena reproduction path before claiming paper-match status. This is not a theoretical rule that CVaR requires mean transport. Mean-CJE and CVaR-CJE are separate estimands; the final CVaR claim is gated by the CVaR-specific tail/shortfall audit, not by the mean audit.

**Canonical authors' pipeline**: `~/Dropbox/cvar-cje-data/cje-arena-experiments/ablations/core/base.py` is the authoritative CJE Arena pipeline. The actual estimator `CalibratedDirectEstimator` lives in the `cje-eval==0.2.10` pip package (`cje.estimators.direct_method`). Prefer **using** `cje-eval==0.2.10` for the Mean baseline (guaranteed paper match) and extending it with CVaR logic on top, rather than reimplementing from scratch.

Key numeric choices (all with file:line citations in `cvar/SCHEMA.md`):
- `n_folds = 5` (cross-fit, hash of prompt_id)
- `oracle_coverage ∈ {0.05, 0.10, 0.25, 0.50, 1.00}`; headline is `0.25`
- `sample_sizes ∈ {250, 500, 1000, 2500, 5000}`; headline is `5000`
- `seeds = range(50)` (paper uses all 50; we use ≥ 20 for CVaR)
- `n_bootstrap = 2000` (calibration-aware; cluster by prompt_id)
- `calibration_mode = "auto"` (per `ablations/config.py:36`; FlexibleCalibrator selects `two_stage` on the 25%-oracle slice with covariates)
- `covariate_names = ["response_length"]` (for `direct+cov`)
- `oracle_masking_seed = experiment seed` (`random.seed(seed)` in `base.py:100`, not a fixed 42; the slice varies across seeds)
- `oua_jackknife = True`, `inference_method = "cluster_robust"`, `use_augmented_estimator = True`

**Working method each session**:
1. Read `cvar/SCHEMA.md` and the most recent `cvar/validate_mean_report.md` before changing any estimator code.
2. Run `cvar/tests_workhorse.py` first; if any test fails, fix before doing new work.
3. Changes to calibration, folding, or bootstrap require re-running `cvar/validate_mean.py` and confirming the benchmark still passes.
4. Only after the Mean benchmark passes do you rerun `cvar/run_arena.py` and regenerate `cvar/comparison.md`.
5. Record every numeric choice (fraction, folds, B, α) and its source in `cvar/SCHEMA.md`. No hardcoded constants without a citation.

**Power analysis on a semi-synthetic DGP**:
- `cvar/dgp.py` — fits per-policy parametric DGP (empirical Y + isotonic m + quartile σ; `unhelpful` is a Y=0 mixture). `cvar/SCHEMA.md` §11 documents the design.
- `cvar/tests_dgp.py` — DGP round-trip + audit power-monotone + xf-audit size at null (5/5).
- `cvar/run_monte_carlo.py [--medium|--full] [--n-workers N]` — outer MC loop with fork-context `multiprocessing.Pool`. Smoke ~3 min, medium ~13 min, full ~6 hr (Mac) / ~30 min (cloud CPU).
- `cvar/run_full_cloud.sh` — runs `--full` on a cloud CPU box (EC2 c7i / GCP n2; GPUs add no value here — sklearn isotonic, no matmul).
- `cvar/power_targets.md` — theory-vs-observed contract; gate for future audit/CI fixes.
- `cvar/make_power_report.py` — aggregates `cvar/results_mc.csv` into `cvar/power_analysis.md` (power curve, size, coverage, scaling, theory-vs-observed gate).
- `cvar/workhorse.py:two_moment_wald_audit_xf` — paired bootstrap audit with t̂ re-maximization. Fixes appendix gap (viii) — empirical size 0.05 at the truest null vs naive's 0.50. Schema §12 documents the derivation.

**Non-negotiables** (from the "no shortcuts" rule):
- Two-stage calibration (isotonic + response-length covariate), not single-stage.
- Cross-fitted oracle folds (5 folds, hash of `prompt_id`), not random-split.
- Calibration-aware cluster bootstrap (B=2000, cluster by prompt_id), not naive bootstrap.
- Per-policy TTC gate (≥ 0.70) and transport audit, both reported in the final table.
- Multi-seed (≥ 20 seeds) to expose across-split variability.
- Use `cje-eval==0.2.10` where possible for the Mean path; reimplementation is only justified for the CVaR extension on top.

**Data** (see `reference_arena_data` memory for details):
- Canonical location: `~/Dropbox/cvar-cje-data/cje-arena-experiments/`
- Primary file: `data/cje_dataset.jsonl` (9.5 MB, 4989 rows; paper reports 4961 after TF filtering)
- Target-policy responses: `data/responses/{policy}_responses.jsonl` (needed for Direct method fresh draws)
- Log-probs: `data/logprobs/` (5 policies × 5 passes, only needed for IPS-family estimators)
- Do NOT re-download per session; verify `sha256` matches `cvar/SCHEMA.md`.

**JSONL schema** (per row of `data/cje_dataset.jsonl`):
```
{
  "prompt": str,
  "response": str,             # base policy response
  "base_policy_logprob": float,
  "target_policy_logprobs": {"clone": float, "parallel_universe_prompt": float, "premium": float, "unhelpful": float},
  "metadata": {"prompt_id": str, "judge_score": float∈[0,1], "oracle_label": float∈[0,1]}
}
```
Oracle labels are present on ALL rows; `--oracle-coverage 0.25` simulates partial oracle by masking 75% via `random.sample` seeded with the experiment seed (so the slice varies across seeds, matching `base.py:100`).

**Tech stack**: polars + numpy + scikit-learn. DuckDB is an acceptable alternative for ingestion. No pandas in our workhorse (though `cje-eval` uses pandas internally — that's fine at the library boundary).
