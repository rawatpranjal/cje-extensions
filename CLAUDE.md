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

### arXiv Papers
Always download the LaTeX source for any arXiv paper before doing anything else with it:

```bash
curl -L https://arxiv.org/src/<paper-id> -o source.tar.gz && tar -xzf source.tar.gz -C <dir>
```

Never use docling on the PDF. Never read the PDF directly for content. Use the `.tex` source — it has exact math, tables, and figures. The `.md` file in this repo has `<!-- formula-not-decoded -->` placeholders throughout; prefer `arxiv_source/` for any paper content.

## Domain Concepts

- **CJE**: Causal Judge Evaluation — calibrated surrogate estimation for LLM policy ranking
- **TTC**: Target-Typicality Coverage — logger mass on target-typical regions; gate threshold ≥ 0.70
- **CLE**: Coverage-Limited Efficiency — precision floor for IPS under low overlap, even with high ESS
- **Transport audit**: per-policy mean residual test `E[Y - f(S,X)] = 0`; fail → recalibrate or refuse level claims
- **Two-stage calibration**: isotonic regression + covariate adjustment (e.g., response length)
- **CVaR-CJE**: extension to lower-tail risk (CVaR at 10%) rather than mean policy value
