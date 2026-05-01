# §09. Chatbot Arena 140k human-preference battles (lmarena-ai) (`arena_140k`)


_HF id: `lmarena-ai/arena-human-preference-140k` · order #9 · n_usable=271,149_


### 1. Identification & access
- **HF id**: `lmarena-ai/arena-human-preference-140k`
- **Title**: Chatbot Arena 140k human-preference battles (lmarena-ai)
- **Paper / arXiv**: 2403.04132 (Chiang et al. 2024)
- **License**: CC-BY-4.0
- **Gated**: no
- **Role**: user-anchored prompt source; binary pairwise Y → CVaR role is prompt-only or ranking-of-policies via win-rate
- **Notes**:
  - Native Y is pairwise (winner ∈ {a, b, tie}); we synthesize `winner_score` ∈ {0, 0.5, 1} per side after melting. CVaR of {0, 0.5, 1} is degenerate.
  - `response_tokens` (assistant-side token count) is a continuous covariate, often correlated with judge preference; useful as a calibration covariate (analogous to original CJE's `direct+cov` length adjustment).
  - 70+ distinct models across 135K battles → built-in massive multi-policy. Per-side melt produces 271K (model, prompt, response) cells.
  - **This is the natural extension of the original CJE Arena setup**: 27× the prompts, modern frontier models, but no continuous-Y. To use as primary CVaR-CJE testbed, we elicit cheap-S + oracle-Y on the prompts × policies cross.

### 2. On-disk footprint after download
- **Bytes**: 0.0 MiB (0 bytes)
- **Cleanup target**: `~/.cache/huggingface/hub/datasets--*arena_140k*` (post-section rm -rf)

### 3. Splits
- (single-split or streaming; n_rows reported in block 6)

### 4. First row (truncated to 300 chars per field)
```
  id: 'c4b9710c-8d64-4bee-a0b0-94637ae4cc65'
  winner: 'model_a'
  language: 'en'
  is_code: False
  if_score: 1
  is_creative_writing: False
  is_math: False
  user_tokens: 11
  turns: 1
  evaluation_session_id: 'a333a685-37f9-474d-b703-f079d8329552'
  model: 'gemini-2.5-pro'
  response_tokens: 1854
  side: 'a'
  winner_score: 1.0
```

### 5. Field inventory
| column | dtype | %null | %empty (str) | n_unique |
|---|---|---|---|---|
| id | String | 0.0% | 0.0% | 135632 |
| winner | String | 0.0% | 0.0% | 4 |
| language | String | 0.0% | 0.0% | 126 |
| is_code | Boolean | 0.0% | 0.0% | 2 |
| if_score | Int32 | 0.0% | 0.0% | 7 |
| is_creative_writing | Boolean | 0.0% | 0.0% | 2 |
| is_math | Boolean | 0.0% | 0.0% | 2 |
| user_tokens | Int32 | 0.0% | 0.0% | 3331 |
| turns | Int32 | 0.0% | 0.0% | 35 |
| evaluation_session_id | String | 0.0% | 0.0% | 115370 |
| model | String | 0.0% | 0.0% | 53 |
| response_tokens | Int32 | 0.0% | 0.0% | 8838 |
| side | String | 0.0% | 0.0% | 2 |
| winner_score | Float64 | 0.0% | 0.0% | 3 |

### 6. Usable n after filtering
- **Raw rows**: 271,268
- **After filter (drop-null prompt/response, drop redacted)**: 271,149
- **Dropped**: 119 (0.04%)

### 7. Y candidate enumeration
| candidate | dtype | classification | CVaR-OK? | top-5 modal values |
|---|---|---|---|---|
| winner_score | Float64 | discrete-3 (range=[0, 1]) | ⚠️ DISCRETE | 0.0=129750, 1.0=98337, 0.5=43062 |
| response_tokens | Int32 | continuous (n_distinct=8838, range=[1, 1.371e+05]) | ✅ | 12=446, 1=347, 13=330, 16=314, 17=304 |
| if_score | Int32 | Likert-7 | ⚠️ DISCRETE | 1=68974, 3=58249, 2=48704, 0=45912, 4=43415 |

### 8. Score distribution per Y candidate

**winner_score**  (n=271,149, mean=0.4421, std=0.4549)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 0 | 0 | 0 | 0 | 0 | 0.5 | 1 | 1 | 1 | 1 | 1 |

Histogram (12 bins):
```
  [       0,  0.08333) | ######################################## 129750
  [ 0.08333,   0.1667) |  0
  [  0.1667,     0.25) |  0
  [    0.25,   0.3333) |  0
  [  0.3333,   0.4167) |  0
  [  0.4167,      0.5) |  0
  [     0.5,   0.5833) | ############# 43062
  [  0.5833,   0.6667) |  0
  [  0.6667,     0.75) |  0
  [    0.75,   0.8333) |  0
  [  0.8333,   0.9167) |  0
  [  0.9167,        1) | ############################## 98337
```

**response_tokens**  (n=271,149, mean=1150, std=1725)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 13 | 59 | 125 | 321 | 747 | 1416 | 2403 | 3370 | 7135 | 1.965e+04 |

Histogram (12 bins):
```
  [       1, 1.143e+04) | ######################################## 270213
  [1.143e+04, 2.286e+04) |  745
  [2.286e+04, 3.428e+04) |  124
  [3.428e+04, 4.571e+04) |  40
  [4.571e+04, 5.714e+04) |  14
  [5.714e+04, 6.856e+04) |  6
  [6.856e+04, 7.999e+04) |  5
  [7.999e+04, 9.142e+04) |  1
  [9.142e+04, 1.028e+05) |  0
  [1.028e+05, 1.143e+05) |  0
  [1.143e+05, 1.257e+05) |  0
  [1.257e+05, 1.371e+05) |  1
```

**if_score**  (n=271,145, mean=2.007, std=1.408)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 4 | 4 | 5 | 5 |

Histogram (12 bins):
```
  [       0,   0.4167) | ########################### 45912
  [  0.4167,   0.8333) |  0
  [  0.8333,     1.25) | ######################################## 68974
  [    1.25,    1.667) |  0
  [   1.667,    2.083) | ############################ 48704
  [   2.083,      2.5) |  0
  [     2.5,    2.917) |  0
  [   2.917,    3.333) | ################################## 58249
  [   3.333,     3.75) |  0
  [    3.75,    4.167) | ######################### 43415
  [   4.167,    4.583) |  0
  [   4.583,        5) | ### 5891
```

### 9. Tie analysis at quantile (CVaR-CJE structural check)
| candidate | quantile | value | n_ties (±0.5%·range) | tie-frac | verdict |
|---|---|---|---|---|---|
| winner_score | q_{0.05} | 0 | 129,750 | 47.85% | ⚠️ STRUCTURAL FAIL |
| winner_score | q_{0.01} | 0 | 129,750 | 47.85% | ⚠️ STRUCTURAL FAIL |
| response_tokens | q_{0.05} | 59 | 135,135 | 49.84% | ⚠️ STRUCTURAL FAIL |
| response_tokens | q_{0.01} | 13 | 128,675 | 47.46% | ⚠️ STRUCTURAL FAIL |
| if_score | q_{0.05} | 0 | 45,912 | 16.93% | ⚠️ STRUCTURAL FAIL |
| if_score | q_{0.01} | 0 | 45,912 | 16.93% | ⚠️ STRUCTURAL FAIL |

*Threshold: ≥10% tie fraction at q_α makes empirical CVaR_α discontinuous in the data → CVaR-CJE inadmissible without Y smoothing.*

### 10. Tail-mass diagnostic
| candidate | mean | CVaR_0.05 (boot 95%) | CVaR_0.05/mean | CVaR_0.01 | CVaR_0.01/mean |
|---|---|---|---|---|---|
| winner_score | 0.4421 | 0  [0, 0] | 0.000 | 0 | 0.000 |
| response_tokens | 1150 | 29.67  [29.26, 30.58] | 0.026 | 7.933 | 0.007 |
| if_score | 2.007 | 0  [0, 0] | 0.000 | 0 | 0.000 |

*Interpretation: CVaR/mean << 1 means the lower tail is far from the body — the regime where CVaR-CJE adds value over mean estimation.*

### 11. Multi-policy nativity
- **policy_id_col**: `model`
- **n distinct policies**: 53

**Top policies by row count:**
| model | n_rows | share |
|---|---|---|
| claude-opus-4-20250514 | 10086 | 3.7% |
| gemini-2.5-flash | 9660 | 3.6% |
| gemini-2.5-pro | 9217 | 3.4% |
| mistral-medium-2505 | 9135 | 3.4% |
| qwen3-235b-a22b-no-thinking | 9076 | 3.3% |
| o3-2025-04-16 | 8526 | 3.1% |
| claude-sonnet-4-20250514 | 8295 | 3.1% |
| chatgpt-4o-latest-20250326 | 7650 | 2.8% |
| gemma-3-27b-it | 7326 | 2.7% |
| claude-3-7-sonnet-20250219-thinking-32k | 7149 | 2.6% |
| claude-3-7-sonnet-20250219 | 6852 | 2.5% |
| command-a-03-2025 | 6838 | 2.5% |
| claude-3-5-sonnet-20241022 | 6817 | 2.5% |
| o3-mini | 6624 | 2.4% |
| deepseek-r1-0528 | 6554 | 2.4% |
| gpt-4.1-2025-04-14 | 6540 | 2.4% |
| o4-mini-2025-04-16 | 6477 | 2.4% |
| claude-3-5-haiku-20241022 | 6466 | 2.4% |
| amazon.nova-pro-v1:0 | 6380 | 2.4% |
| claude-opus-4-20250514-thinking-16k | 6144 | 2.3% |

*✅ Multi-policy native: can read off π0/π′ pairs without fresh generation.*

### 12. Existing cheap-judge / oracle pair
- **judge_col**: (none declared)
- **oracle_col**: (none declared)
- *Both must be elicited externally (see block 19).*

### 13. Audit-discriminative metadata (stratification covariates)

**language** (cardinality=126, coverage=100.0%): `en`=142284; `pl`=27612; `und`=23657; `ru`=18517; `zh`=12998; `de`=8620; `ko`=4972; `ja`=4887

**is_code** (cardinality=2, coverage=100.0%): `False`=192452; `True`=78697

**is_creative_writing** (cardinality=2, coverage=100.0%): `False`=247990; `True`=23159

**is_math** (cardinality=2, coverage=100.0%): `False`=249373; `True`=21776

**side** (cardinality=2, coverage=100.0%): `b`=135576; `a`=135573

### 14. Per-stratum tail-mass heterogeneity (audit-discriminativeness)
Y = `winner_score`; for each stratification column, the bottom-5% mean per level (top-8 levels):


**language** (sorted by q_0.05 ascending; lowest-tail strata first):
| language | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| th | 186 | 0.457 | 0 |
| fi | 154 | 0.4481 | 0 |
| la | 104 | 0.4712 | 0 |
| bg | 84 | 0.4405 | 0 |
| lt | 100 | 0.42 | 0 |
| sr | 146 | 0.4247 | 0 |
| hu | 348 | 0.4483 | 0 |
| nn | 50 | 0.44 | 0 |

*Heterogeneity: q_0.05 spread across levels = 0. Larger → audit transport test will be more discriminative.*

**is_code** (sorted by q_0.05 ascending; lowest-tail strata first):
| is_code | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| False | 192,452 | 0.4427 | 0 |
| True | 78,697 | 0.4405 | 0 |

*Heterogeneity: q_0.05 spread across levels = 0. Larger → audit transport test will be more discriminative.*

**is_creative_writing** (sorted by q_0.05 ascending; lowest-tail strata first):
| is_creative_writing | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| False | 247,990 | 0.4432 | 0 |
| True | 23,159 | 0.4295 | 0 |

*Heterogeneity: q_0.05 spread across levels = 0. Larger → audit transport test will be more discriminative.*

**is_math** (sorted by q_0.05 ascending; lowest-tail strata first):
| is_math | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| False | 249,373 | 0.443 | 0 |
| True | 21,776 | 0.4317 | 0 |

*Heterogeneity: q_0.05 spread across levels = 0. Larger → audit transport test will be more discriminative.*

**side** (sorted by q_0.05 ascending; lowest-tail strata first):
| side | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| b | 135,576 | 0.4466 | 0 |
| a | 135,573 | 0.4375 | 0 |

*Heterogeneity: q_0.05 spread across levels = 0. Larger → audit transport test will be more discriminative.*

### 15. Prompt-length distribution
- (no prompt column declared)

### 16. Response-length distribution
- (no response column declared — generation cost depends on policy choice)

### 17. Tail samples (bottom 5 by `winner_score`)

**Sample 1** — Y=0.0
- `language`: en
- `is_code`: True
- `is_creative_writing`: False
- `is_math`: False
- `side`: a

**Sample 2** — Y=0.0
- `language`: de
- `is_code`: False
- `is_creative_writing`: False
- `is_math`: False
- `side`: a

**Sample 3** — Y=0.0
- `language`: en
- `is_code`: True
- `is_creative_writing`: False
- `is_math`: False
- `side`: a

**Sample 4** — Y=0.0
- `language`: pl
- `is_code`: False
- `is_creative_writing`: False
- `is_math`: False
- `side`: a

**Sample 5** — Y=0.0
- `language`: en
- `is_code`: False
- `is_creative_writing`: False
- `is_math`: False
- `side`: a

### 18. Median samples (3 rows in [q_0.45, q_0.55] = [0, 0.5] of `winner_score`)

**Sample 1** — Y=0.5

**Sample 2** — Y=0.0

**Sample 3** — Y=0.0

### 19. π0 / π′ / S / Y proposal (CVaR-CJE scoping)
- **π_base (logger π0)**: Pick a mid-tier in-data model that has many battles (e.g., `gpt-4o`, `gpt-4-turbo`, `claude-3-5-sonnet`) as π0; cap to {prompts where this model is on side a or b}
- **π′ panel**:
  - **pi_prime_frontier_a** — In-data: `gpt-5` / `gemini-2.5-pro` / `claude-opus-4.7-thinking`  *(frontier comparators; expected dominance both on mean and tail)*
  - **pi_clone_open** — In-data: a smaller open-weight model (e.g., `llama-3.1-70b-instruct`)  *(distilled-style comparator; tail differs from frontier on hard prompts)*
  - **pi_specialist** — In-data: a code-specialist (e.g., `qwen-2.5-coder-32b`) or domain-tuned model  *(strong on code-tagged subset, weak on open-ended creative writing)*
  - **pi_adversarial** — External: π_base + jailbreak system prompt applied to the same prompts  *(audit-positive; we'd append fresh responses to compare)*
- **Cheap judge S**: GPT-4o-mini against a 4-criterion rubric (helpfulness, instruction-following, factuality, refusal-quality) — keeps the dataset's pairwise comparability while moving to a continuous Y
- **Oracle Y**: Multi-judge consensus across {Claude Opus 4.7 thinking, GPT-5-thinking, Gemini 3 Pro} scoring on the same rubric — averages out single-judge bias and matches the dataset's own crowd-aggregation philosophy

*Note: `model` already partitions 53 policies in the data; no fresh generation is required for the multi-policy comparison itself — only the cheap-S / oracle-Y elicitation.*

### 20. Cost & CVaR-resolution feasibility

**Generation cost (rough @ 2026-04-28 API rates)**:
- *Policies are already in-data (`policy_id` partitions 53 cells).*
- *Marginal generation cost for the multi-policy comparison itself = $0.* The numbers below are the hypothetical fresh-generation cost if you wanted to *augment* the in-data panel with extra π′ candidates (e.g., a frontier model not in the original 12-cell grid).
- prompt-length not measured (no prompt_col); skipping per-token estimate
- oracle slice @ 5% × 53 policies = 718,521 calls × ~$0.070/call ≈ **$50,296** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)
- oracle slice @ 15% × 53 policies = 2,155,616 calls × ~$0.070/call ≈ **$150,893** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)
- oracle slice @ 25% × 53 policies = 3,592,711 calls × ~$0.070/call ≈ **$251,490** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)

**CVaR-α resolution feasibility (Wald 95%-CI half-width on existing Y)**:
- α=0.05: CVaR≈0, half-width≈0 (nan% of CVaR) — ❌ insufficient  [n=271,149]
- α=0.01: CVaR≈0, half-width≈0 (nan% of CVaR) — ❌ insufficient  [n=271,149]

*Threshold: CI half-width ≤ 20% of CVaR is a comfortable resolution; 20–50% is publishable but tight; >50% means scale up n or reduce α aggressiveness.*
