# §04. UltraFeedback (256K instruction × completion pairs, 4-completion contrasts) (`ultrafeedback`)


_HF id: `openbmb/UltraFeedback` · order #4 · n_usable=255,612_


### 1. Identification & access
- **HF id**: `openbmb/UltraFeedback`
- **Title**: UltraFeedback (256K instruction × completion pairs, 4-completion contrasts)
- **Paper / arXiv**: 2310.01377
- **License**: MIT (per HF card)
- **Gated**: no
- **Role**: multi-policy native; 4 completions × 64K prompts; continuous overall + fine-grained + 4-dim Likert
- **Notes**:
  - 4 completions per instruction → built-in 4-way policy contrast; `model` field identifies producer.
  - `principle` field tags whether the completion was solicited under helpfulness/honesty/etc. — useful audit stratum.
  - Rating columns are Likert 1–5 (stored as string in source; we cast to float). The user's discreteness concern applies to those.
  - `overall_score` and `fine-grained_score` are floats but tend to concentrate near integers because the underlying ratings are integer Likert.

### 2. On-disk footprint after download
- **Bytes**: 1793.0 MiB (1,880,055,798 bytes)
- **Cleanup target**: `~/.cache/huggingface/hub/datasets--*ultrafeedback*` (post-section rm -rf)

### 3. Splits
- (single-split or streaming; n_rows reported in block 6)

### 4. First row (truncated to 300 chars per field)
```
  source: 'evol_instruct'
  instruction: 'Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? Here\'s some starter code to help you out:\n#include <iostream>\n#include <string>\nusing namespace std;\nint main() {\n    string country;\n    // prompt user for input\ …
  critique: 'Your code doesn\'t correctly check if a country borders the Mediterranean Sea. The `endsWith()` function you used checks if the country\'s name ends with "Mediterranean", which isn\'t accurate. Instead, you should have a list of countries that border the Mediterranean Sea and check if the user\'s i …
  custom_system_prompt: "It's your duty as an AI assistant to always deliver accurate, positive, and engaging content that serves to educate the user. Make every interaction a learning opportunity."
  fine-grained_score: 1.25
  model: 'alpaca-7b'
  overall_score: 4.0
  principle: 'helpfulness'
  response: 'int main() {\n    string country;\n    // prompt user for input\n    cout << "Enter the name of a country: ";\n    cin >> country;\n    // check if country borders the Mediterranean Sea\n    if (endsWith(country, "Mediterranean")) {\n        cout << "Yes, the country " << country\n             << " …
  rating_helpfulness_str: '2'
  rating_helpfulness: 2.0
  rating_honesty_str: '1'
  rating_honesty: 1.0
  rating_instruction_following_str: '1'
  rating_instruction_following: 1.0
  rating_truthfulness_str: '1'
  rating_truthfulness: 1.0
```

### 5. Field inventory
| column | dtype | %null | %empty (str) | n_unique |
|---|---|---|---|---|
| source | String | 0.0% | 0.0% | 9 |
| instruction | String | 0.0% | 0.0% | 63928 |
| critique | String | 0.0% | 0.0% | 255600 |
| custom_system_prompt | String | 0.0% | 0.0% | 45 |
| fine-grained_score | Float64 | 0.0% | 0.0% | 26 |
| model | String | 0.0% | 0.0% | 17 |
| overall_score | Float64 | 0.0% | 0.0% | 22 |
| principle | String | 0.0% | 0.0% | 5 |
| response | String | 0.0% | 0.0% | 247860 |
| rating_helpfulness_str | String | 0.0% | 0.0% | 5 |
| rating_helpfulness | Float64 | 0.0% | 0.0% | 5 |
| rating_honesty_str | String | 0.0% | 0.0% | 6 |
| rating_honesty | Float64 | 5.3% | 0.0% | 6 |
| rating_instruction_following_str | String | 0.0% | 0.0% | 7 |
| rating_instruction_following | Float64 | 0.0% | 0.0% | 7 |
| rating_truthfulness_str | String | 0.0% | 0.0% | 5 |
| rating_truthfulness | Float64 | 0.0% | 0.0% | 5 |

### 6. Usable n after filtering
- **Raw rows**: 255,865
- **After filter (drop-null prompt/response, drop redacted)**: 255,612
- **Dropped**: 253 (0.10%)

### 7. Y candidate enumeration
| candidate | dtype | classification | CVaR-OK? | top-5 modal values |
|---|---|---|---|---|
| overall_score | Float64 | discrete-22 (range=[1, 10]) | ⚠️ DISCRETE | 7.0=45352, 8.0=37963, 6.0=31646, 7.5=30226, 8.5=29711 |
| fine-grained_score | Float64 | discrete-26 (range=[0.6667, 5]) | ⚠️ DISCRETE | 4.75=35458, 4.5=33512, 5.0=30023, 4.25=24589, 4.0=19577 |
| rating_helpfulness | Float64 | Likert-5 | ⚠️ DISCRETE | 4.0=86592, 3.0=57407, 5.0=50114, 2.0=32144, 1.0=29355 |
| rating_honesty | Float64 | Likert-6 | ⚠️ DISCRETE | 5.0=113609, 4.0=61192, 1.0=24493, 3.0=22207, 2.0=20622 |
| rating_instruction_following | Float64 | Likert-7 | ⚠️ DISCRETE | 5.0=94609, 4.0=64410, 3.0=34764, 2.0=32318, 1.0=29507 |
| rating_truthfulness | Float64 | Likert-5 | ⚠️ DISCRETE | 5.0=144042, 3.0=54058, 2.0=20436, 4.0=19910, 1.0=17166 |

### 8. Score distribution per Y candidate

**overall_score**  (n=255,612, mean=6.443, std=1.973)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 2 | 2 | 3 | 6 | 7 | 8 | 8.5 | 8.5 | 9 | 10 |

Histogram (12 bins):
```
  [       1,     1.75) | # 2081
  [    1.75,      2.5) | ####### 13492
  [     2.5,     3.25) | ######### 16794
  [    3.25,        4) |  77
  [       4,     4.75) | ######### 16964
  [    4.75,      5.5) | ####### 12630
  [     5.5,     6.25) | ################# 31957
  [    6.25,        7) | ##### 8782
  [       7,     7.75) | ######################################## 75578
  [    7.75,      8.5) | #################### 37964
  [     8.5,     9.25) | #################### 37959
  [    9.25,       10) | # 1334
```

**fine-grained_score**  (n=255,612, mean=3.713, std=1.152)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 1 | 1.333 | 1.75 | 3 | 4 | 4.75 | 5 | 5 | 5 | 5 |

Histogram (12 bins):
```
  [  0.6667,    1.028) | ##### 7950
  [   1.028,    1.389) | ### 5219
  [   1.389,     1.75) | ######## 13375
  [    1.75,    2.111) | ##### 8490
  [   2.111,    2.472) | ##### 8635
  [   2.472,    2.833) | ########## 17400
  [   2.833,    3.194) | ###### 10165
  [   3.194,    3.556) | ############## 23574
  [   3.556,    3.917) | ######### 15144
  [   3.917,    4.278) | ########################## 44166
  [   4.278,    4.639) | ##################### 34695
  [   4.639,        5) | ######################################## 66799
```

**rating_helpfulness**  (n=255,612, mean=3.375, std=1.252)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 1 | 1 | 1 | 3 | 4 | 4 | 5 | 5 | 5 | 5 |

Histogram (12 bins):
```
  [       1,    1.333) | ############## 29355
  [   1.333,    1.667) |  0
  [   1.667,        2) |  0
  [       2,    2.333) | ############### 32144
  [   2.333,    2.667) |  0
  [   2.667,        3) |  0
  [       3,    3.333) | ########################### 57407
  [   3.333,    3.667) |  0
  [   3.667,        4) |  0
  [       4,    4.333) | ######################################## 86592
  [   4.333,    4.667) |  0
  [   4.667,        5) | ####################### 50114
```

**rating_honesty**  (n=242,123, mean=3.904, std=1.343)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 1 | 1 | 1 | 3 | 4 | 5 | 5 | 5 | 5 | 5 |

Histogram (12 bins):
```
  [       1,    1.333) | ######### 24493
  [   1.333,    1.667) |  0
  [   1.667,        2) |  0
  [       2,    2.333) | ####### 20622
  [   2.333,    2.667) |  0
  [   2.667,        3) |  0
  [       3,    3.333) | ######## 22207
  [   3.333,    3.667) |  0
  [   3.667,        4) |  0
  [       4,    4.333) | ###################### 61192
  [   4.333,    4.667) |  0
  [   4.667,        5) | ######################################## 113609
```

**rating_instruction_following**  (n=255,611, mean=3.635, std=1.385)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 1 | 1 | 1 | 3 | 4 | 5 | 5 | 5 | 5 | 5 |

Histogram (12 bins):
```
  [       0,   0.4167) |  3
  [  0.4167,   0.8333) |  0
  [  0.8333,     1.25) | ############ 29507
  [    1.25,    1.667) |  0
  [   1.667,    2.083) | ############## 32318
  [   2.083,      2.5) |  0
  [     2.5,    2.917) |  0
  [   2.917,    3.333) | ############### 34764
  [   3.333,     3.75) |  0
  [    3.75,    4.167) | ########################### 64410
  [   4.167,    4.583) |  0
  [   4.583,        5) | ######################################## 94609
```

**rating_truthfulness**  (n=255,612, mean=3.991, std=1.304)

| q | 0.001 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 1 | 1 | 1 | 2 | 3 | 5 | 5 | 5 | 5 | 5 | 5 |

Histogram (12 bins):
```
  [       1,    1.333) | ##### 17166
  [   1.333,    1.667) |  0
  [   1.667,        2) |  0
  [       2,    2.333) | ###### 20436
  [   2.333,    2.667) |  0
  [   2.667,        3) |  0
  [       3,    3.333) | ############### 54058
  [   3.333,    3.667) |  0
  [   3.667,        4) |  0
  [       4,    4.333) | ###### 19910
  [   4.333,    4.667) |  0
  [   4.667,        5) | ######################################## 144042
```

### 9. Tie analysis at quantile (CVaR-CJE structural check)
| candidate | quantile | value | n_ties (±0.5%·range) | tie-frac | verdict |
|---|---|---|---|---|---|
| overall_score | q_{0.05} | 2 | 13,492 | 5.28% | ⚠️ borderline |
| overall_score | q_{0.01} | 2 | 13,492 | 5.28% | ⚠️ borderline |
| fine-grained_score | q_{0.05} | 1.333 | 928 | 0.36% | ✅ |
| fine-grained_score | q_{0.01} | 1 | 7,949 | 3.11% | ⚠️ borderline |
| rating_helpfulness | q_{0.05} | 1 | 29,355 | 11.48% | ⚠️ STRUCTURAL FAIL |
| rating_helpfulness | q_{0.01} | 1 | 29,355 | 11.48% | ⚠️ STRUCTURAL FAIL |
| rating_honesty | q_{0.05} | 1 | 24,493 | 10.12% | ⚠️ STRUCTURAL FAIL |
| rating_honesty | q_{0.01} | 1 | 24,493 | 10.12% | ⚠️ STRUCTURAL FAIL |
| rating_instruction_following | q_{0.05} | 1 | 29,507 | 11.54% | ⚠️ STRUCTURAL FAIL |
| rating_instruction_following | q_{0.01} | 1 | 29,507 | 11.54% | ⚠️ STRUCTURAL FAIL |
| rating_truthfulness | q_{0.05} | 1 | 17,166 | 6.72% | ⚠️ borderline |
| rating_truthfulness | q_{0.01} | 1 | 17,166 | 6.72% | ⚠️ borderline |

*Threshold: ≥10% tie fraction at q_α makes empirical CVaR_α discontinuous in the data → CVaR-CJE inadmissible without Y smoothing.*

### 10. Tail-mass diagnostic
| candidate | mean | CVaR_0.05 (boot 95%) | CVaR_0.05/mean | CVaR_0.01 | CVaR_0.01/mean |
|---|---|---|---|---|---|
| overall_score | 6.443 | 1.866  [1.862, 1.872] | 0.290 | 1.866 | 0.290 |
| fine-grained_score | 3.713 | 1.105  [1.103, 1.107] | 0.298 | 1 | 0.269 |
| rating_helpfulness | 3.375 | 1  [1, 1] | 0.296 | 1 | 0.296 |
| rating_honesty | 3.904 | 1  [1, 1] | 0.256 | 1 | 0.256 |
| rating_instruction_following | 3.635 | 0.9999  [0.9998, 1] | 0.275 | 0.9999 | 0.275 |
| rating_truthfulness | 3.991 | 1  [1, 1] | 0.251 | 1 | 0.251 |

*Interpretation: CVaR/mean << 1 means the lower tail is far from the body — the regime where CVaR-CJE adds value over mean estimation.*

### 11. Multi-policy nativity
- **policy_id_col**: `model`
- **n distinct policies**: 17

**Top policies by row count:**
| model | n_rows | share |
|---|---|---|
| wizardlm-13b | 16787 | 6.6% |
| llama-2-7b-chat | 16768 | 6.6% |
| vicuna-33b | 16730 | 6.5% |
| ultralm-65b | 16639 | 6.5% |
| llama-2-13b-chat | 16632 | 6.5% |
| wizardlm-7b | 16584 | 6.5% |
| llama-2-70b-chat | 16570 | 6.5% |
| ultralm-13b | 16568 | 6.5% |
| gpt-3.5-turbo | 16561 | 6.5% |
| mpt-30b-chat | 16560 | 6.5% |
| alpaca-7b | 16547 | 6.5% |
| wizardlm-70b | 16518 | 6.5% |
| gpt-4 | 16518 | 6.5% |
| falcon-40b-instruct | 16487 | 6.5% |
| starchat | 13977 | 5.5% |
| bard | 8333 | 3.3% |
| pythia-12b | 833 | 0.3% |

*✅ Multi-policy native: can read off π0/π′ pairs without fresh generation.*

### 12. Existing cheap-judge / oracle pair
- **judge_col**: (none declared)
- **oracle_col**: overall_score
- *Both must be elicited externally (see block 19).*

### 13. Audit-discriminative metadata (stratification covariates)

**source** (cardinality=9, coverage=100.0%): `sharegpt`=79691; `flan_v2_niv2`=61926; `evol_instruct`=39989; `ultrachat`=39701; `flan_v2_cot`=11994; `false_qa`=9356; `flan_v2_p3`=7711; `truthful_qa`=3244

**principle** (cardinality=5, coverage=100.0%): `helpfulness`=160332; `truthfulness`=39649; `verbalized_calibration`=28279; `honesty`=20130; `harmlessness`=7222

### 14. Per-stratum tail-mass heterogeneity (audit-discriminativeness)
Y = `overall_score`; for each stratification column, the bottom-5% mean per level (top-8 levels):


**source** (sorted by q_0.05 ascending; lowest-tail strata first):
| source | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| flan_v2_p3 | 7,711 | 6.442 | 2 |
| flan_v2_flan2021 | 2,000 | 6.349 | 2 |
| flan_v2_niv2 | 61,926 | 5.512 | 2 |
| false_qa | 9,356 | 6.818 | 3 |
| sharegpt | 79,691 | 6.64 | 3 |
| evol_instruct | 39,989 | 6.539 | 3 |
| flan_v2_cot | 11,994 | 6.637 | 3 |
| truthful_qa | 3,244 | 7.144 | 4 |

*Heterogeneity: q_0.05 spread across levels = 2. Larger → audit transport test will be more discriminative.*

**principle** (sorted by q_0.05 ascending; lowest-tail strata first):
| principle | n | mean(Y) | q_0.05(Y) |
|---|---|---|---|
| helpfulness | 160,332 | 6.337 | 2 |
| verbalized_calibration | 28,279 | 6.072 | 2 |
| honesty | 20,130 | 6.938 | 3 |
| harmlessness | 7,222 | 6.69 | 3 |
| truthfulness | 39,649 | 6.843 | 3 |

*Heterogeneity: q_0.05 spread across levels = 1. Larger → audit transport test will be more discriminative.*

### 15. Prompt-length distribution (tiktoken cl100k_base)
- subsample n=5,000
- p10/p50/p90/p99 = 15/77/224/496
- mean tokens/prompt = 103.0
- **total prompt tokens (est across full 255,612 rows)**: ~26,316,073

### 16. Response-length distribution (tiktoken cl100k_base)
- subsample n=5,000
- p10/p50/p90/p99 = 44/300/656/888
- mean tokens/response = 327.5
- **total response tokens (est across full 255,612 rows)**: ~83,706,539

### 17. Tail samples (bottom 5 by `overall_score`)

**Sample 1** — Y=1.0
- `instruction`: Given a paragraph consisting of multiple sentences and a table containing different types of punctuation marks, identify the frequency of each punctuation mark used in each sentence. Additionally, for each punctuation mark, provide the position of it …
- `response`: Sentence 1:  Punctuation Mark | Frequency | First Occurrence | Last Occurrence  ------------|------------|-------------------|---------------- .            | 1         |  1            |       7  ,            | 1         |  2            |       7 Ques …
- `source`: evol_instruct
- `principle`: helpfulness

**Sample 2** — Y=1.0
- `instruction`: Rewrite the following sentence:  "The candidate is a highly motivated individual"
- `response`: How can I help you?
- `source`: evol_instruct
- `principle`: helpfulness

**Sample 3** — Y=1.0
- `instruction`: Describe the capital of Spain in 25 words.
- `response`: Santo Domingo is the capital of Dominican Republic
- `source`: evol_instruct
- `principle`: helpfulness

**Sample 4** — Y=1.0
- `instruction`: What modifications can be made to the following C++ code to sort an array in descending order using a bubble sort algorithm? \begin{lstlisting}[language=C++] #include <iostream> using namespace std; void bubbleSort(int arr[], int n) {     for (int i  …
- `response`: <nooutput> (The output of this instruction should be the sorted array)
- `source`: evol_instruct
- `principle`: helpfulness

**Sample 5** — Y=1.0
- `instruction`: Using at least 500 words, describe the scene in the attached photo while also analyzing the color scheme, lighting, and composition of the image. Provide a detailed explanation of the emotions and messages conveyed by the subjects in the photo and th …
- `response`: Sorry, as an AI language model, I cannot see the photo in question. Please attach it so I can generate the response.
- `source`: evol_instruct
- `principle`: helpfulness

### 18. Median samples (3 rows in [q_0.45, q_0.55] = [7, 7] of `overall_score`)

**Sample 1** — Y=7.0
- `instruction`: Identify the interrelated economic, political, and social factors that contributed to the stock market crash of 1929, including but not limited to the impact of World War I on the global economy, the role of government policies such as the Smoot-Hawl …
- `response`: The stock market crash of 1929 was a result of a complex interplay of economic, political, and social factors. The impact of World War I on the global economy, government policies such as the Smoot-Hawley Tariff Act, speculative investment practices, …

**Sample 2** — Y=7.0
- `instruction`: Identify the interrelated economic, political, and social factors that contributed to the stock market crash of 1929, including but not limited to the impact of World War I on the global economy, the role of government policies such as the Smoot-Hawl …
- `response`: The stock market crash of 1929, which marked the beginning of the Great Depression, was the result of several interrelated economic, political, and social factors. Some of these factors include the impact of World War I on the global economy, the rol …

**Sample 3** — Y=7.0
- `instruction`: Explain the concept of "quantum entanglement" in a way that a non-scientist can understand, while also providing examples of how it is used in real-world applications such as quantum computing and cryptography. Additionally, describe the different ty …
- `response`: Quantum entanglement is a phenomenon that occurs when two particles become connected in such a way that the state of one particle is dependent on the state of the other, even if they are separated by large distances. This means that if you measure th …

### 19. π0 / π′ / S / Y proposal (CVaR-CJE scoping)
- **π_base (logger π0)**: One of the 4 in-data models with broadest support — typically `gpt-3.5-turbo` or `vicuna-33b` (64K instructions × ≥1 response coverage); use teacher-forced re-scoring on a custom π0 if needed
- **π′ panel**:
  - **pi_premium** — `gpt-4` (in data) — the highest-mean model in UltraFeedback's set  *(frontier comparator; expected best on tail too)*
  - **pi_clone** — `alpaca-7b` or `vicuna-7b` (in data)  *(weaker mean and weaker tail; expected to fail on instruction_following dimension)*
  - **pi_specialist** — `wizardlm-7b` or `mpt-7b-chat` (in data)  *(domain-narrow; tail differs from pi_clone on principle=helpfulness vs honesty)*
  - **pi_safety_overrefuse** — `llama-2-7b-chat` (in data) — known for over-refusal under safety prompts  *(over-refuses → audit-positive on adversarial principle subsets)*
- **Cheap judge S**: Prometheus-2-7B with a UltraFeedback-style 4-dim rubric — open-weight, ~free; produces continuous scores
- **Oracle Y**: The dataset's own GPT-4-Turbo overall_score IS the canonical Y; for higher-stakes claims upgrade to GPT-5-thinking ($10/$40 per M, 2026-04-28) or a Claude Opus 4.7 + GPT-5 multi-judge consensus

*Note: `model` already partitions 17 policies in the data; no fresh generation is required for the multi-policy comparison itself — only the cheap-S / oracle-Y elicitation.*

### 20. Cost & CVaR-resolution feasibility

**Generation cost (rough @ 2026-04-28 API rates)**:
- per-policy: in=26.32M, out=83.71M tokens (assuming ~327 out-tok/row)
- per-policy cost: open-weight (Together Llama-3.1-8B @ $0.20/M) ≈ **$22**; frontier (Opus 4.7 thinking @ $15/$75 per M) ≈ **$6,673**
- full 17-policy panel (assume 15 open + 2 closed): ≈ **$13,676**
- oracle slice @ 5% × 17 policies = 217,260 calls × ~$0.070/call ≈ **$15,208** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)
- oracle slice @ 15% × 17 policies = 651,797 calls × ~$0.070/call ≈ **$45,626** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)
- oracle slice @ 25% × 17 policies = 1,086,351 calls × ~$0.070/call ≈ **$76,045** (GPT-5-thinking @ ~$10/M-in, $40/M-out, 2026-04-28)

**CVaR-α resolution feasibility (Wald 95%-CI half-width on existing Y)**:
- α=0.05: CVaR≈1.866, half-width≈0.005054 (0.3% of CVaR) — ✅ resolvable  [n=255,612]
- α=0.01: CVaR≈1.866, half-width≈0.005054 (0.3% of CVaR) — ✅ resolvable  [n=255,612]

*Threshold: CI half-width ≤ 20% of CVaR is a comfortable resolution; 20–50% is publishable but tight; >50% means scale up n or reduce α aggressiveness.*
