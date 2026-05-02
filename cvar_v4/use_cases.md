# What LLM-as-a-judge is actually used for (literature scan, May 2026)

Quick scan to ground the first line of the intro. Goal: figure out what the
literature actually says LLM-as-judge ranks/compares, so we don't list
plausible-sounding nouns the cited work doesn't actually cover.

## Real use cases (with citations / sources)

### 1. Ranking candidate **models** (the dominant use case)
- **Chatbot Arena** ([LMSYS](https://www.lmsys.org/blog/2023-05-03-arena/),
  [arXiv:2403.04132](https://sky.cs.berkeley.edu/project/chatbot-arena/)):
  crowdsourced pairwise model battles → Elo leaderboard. 6M+ user votes by
  2026.
- **MT-Bench** ([Zheng et al., 2023, arXiv:2306.05685](https://arxiv.org/abs/2306.05685)):
  GPT-4 as judge scores 8 categories of multi-turn responses; produces a
  leaderboard of ~12 chat models. 80% agreement with humans.
- **RewardBench** ([Lambert et al., 2024](https://arxiv.org/html/2604.13717)):
  benchmark for ranking reward-model and judge-model accuracy at picking
  preferred responses.
- **JudgeBench** ([2024 ICLR](https://openreview.net/forum?id=G0dksFayVq)):
  evaluates the judges themselves on objective-correctness tasks.

### 2. Scoring/grading individual **responses** against rubrics
- **G-Eval** ([Liu et al., 2023](https://arxiv.org/abs/2303.16634)): prompt
  GPT-4 with a rubric to score summarization / dialogue quality.
- **Prometheus 2** ([Kim et al., 2024](https://arxiv.org/abs/2405.01535)):
  open-source judge that scores responses against fine-grained rubrics.
- **HealthBench** ([OpenAI 2025](https://openai.com/index/healthbench/)):
  rubric-graded medical-assistant outputs (this is our test bed).
- Production usage ([Confident AI](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method),
  [Langfuse](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge),
  [Patronus](https://www.patronus.ai/llm-testing/llm-as-a-judge)): customer
  support reply quality, RAG groundedness, safety/helpfulness scoring.

### 3. **Reward signal** in alignment training (RLAIF, DPO, on-policy RL)
- **RLAIF and self-reward** ([Survey, arXiv:2411.15594](https://arxiv.org/html/2411.15594v6),
  [Atla blog](https://atla-ai.com/post/llm-judges-as-reward-models)): the
  judge replaces or augments the human-annotated reward model. Used in
  Constitutional AI and follow-up self-improvement loops.
- **DPO data labeling** ([Arxiv:2509.02709](https://arxiv.org/html/2509.02709)):
  judges generate preference pairs that train direct-preference-optimization
  policies.

### 4. Comparing **system prompts / prompt variants** (real, secondary)
- Production prompt-engineering platforms (Langfuse, Patronus, Arize) all
  document "compare prompt A vs prompt B by averaging judge scores" as a
  primary workflow. Cited less often in academic papers because it's
  considered ops, not research.

### 5. **RAG pipeline / retrieval** evaluation (a special case of #2)
- Judge checks whether a response is grounded in retrieved context, used to
  compare retriever or chunking choices ([Confident AI](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method),
  [Langfuse](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)).
  Strictly a response-grading task with metadata, not its own thing.

## What is NOT really being ranked

- **"Tools"** as a standalone target: not a category in the cited
  literature. Tool-use *capability* is benchmarked
  ([ToolPRMBench, ACL 2026](https://aclanthology.org/)) but the model is the
  ranked unit, not the tool. Vague filler in our current draft.
- **"Safety interventions"**: guardrail filters and refusal classifiers are
  *measured* (BeaverTails, HarmBench, StrongREJECT) but not typically
  *ranked* by an LLM judge in a leaderboard sense. The judge is one piece of
  the safety eval pipeline, not the unit being compared.

## Implication for the intro first line

Drop "tools" and "safety interventions." The faithful version of what
people use LLM judges for is some subset of:
1. **rank candidate models** (Chatbot Arena, MT-Bench, RewardBench)
2. **rank candidate system prompts / response styles**
3. **score individual responses against rubrics** (G-Eval, Prometheus,
   HealthBench, RAG groundedness checks)
4. **provide a cheap reward signal in RLAIF / DPO**

A faithful one-liner: *"LLM evaluation pipelines increasingly use a cheap
LLM-as-judge to rank candidate models, compare system prompts, and grade
individual responses."* This stays close to the original cadence but only
includes things the cited literature actually does.

## Survey papers worth citing if we want to broaden the intro
- [A Survey on LLM-as-a-Judge, arXiv:2411.15594](https://arxiv.org/html/2411.15594v6)
- [LLMs-as-Judges: A Comprehensive Survey, arXiv:2412.05579](https://arxiv.org/html/2412.05579v2)
- [Eugene Yan, Evaluating LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- [Preference Leakage, ICLR 2026](https://openreview.net/) — contamination problem when judge and generator are related

## Cheap vs expensive judges: who uses what, where

The CJE framing splits a budget between a "cheap" judge (everywhere) and
an "oracle" judge (small slice). The literature does the same split,
though the words differ. Here is the actual map.

### Cheap judges in practice
**Models:** GPT-4o-mini, GPT-5 Nano, Claude Haiku 4.5, Gemini Flash,
Llama-3-8B-Instruct, DeepSeek V3.
**Pricing:** ~$0.05–$1 per million input tokens, roughly 5–30× cheaper
than the strong models.
**Where they get used:**
- **Online monitoring / dashboards** — score every production response,
  not a sample. ([Langfuse](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge),
  [Arize](https://arize.com/llm-as-a-judge/),
  [Patronus](https://www.patronus.ai/llm-testing/llm-as-a-judge))
- **CI / staging gates** — block a deploy if mean judge score on a fixed
  prompt set drops. Cost per check matters. ([Confident AI](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method),
  [Comet](https://www.comet.com/site/blog/llm-as-a-judge/))
- **Routing / triage** — first-pass cheap-judge filter, then escalate the
  borderline cases to a strong judge or a human.
- **Reward signal in RLAIF / DPO** — when human labels are too expensive
  and an oracle judge is too slow, the cheap judge labels millions of
  pairs. ([Atla](https://atla-ai.com/post/llm-judges-as-reward-models),
  [Survey 2411.15594](https://arxiv.org/html/2411.15594v6))
- **Active learning loops** — use cheap-judge uncertainty to pick which
  rows to send to the oracle judge or human ([Reward Modeling thesis,
  Berkeley 2025](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-82.pdf)).

### Strong / "oracle" judges in practice
**Models:** GPT-4, GPT-5, Claude Opus 4.x, Claude Sonnet 4.x, Gemini
Pro / Ultra.
**Pricing:** ~$3–$30 per million input tokens, often 10×+ more than
cheap.
**Where they get used:**
- **Public leaderboards** — MT-Bench uses GPT-4 as judge; Chatbot Arena
  uses crowdsourced humans (the oracle of last resort) plus a strong-LLM
  reference path. ([MT-Bench](https://arxiv.org/abs/2306.05685),
  [Chatbot Arena](https://www.lmsys.org/blog/2023-05-03-arena/))
- **Rubric calibration** — generate the ground-truth labels that a cheap
  judge will be calibrated against. This is exactly the role the oracle
  plays in CJE.
- **High-stakes / specialist domains** — clinical Q&A scoring
  ([HealthBench](https://openai.com/index/healthbench/),
  [PMC PMC12396308](https://pmc.ncbi.nlm.nih.gov/articles/PMC12396308/)),
  legal review, code review ([CodeJudgeBench, arXiv:2507.10535](https://arxiv.org/pdf/2507.10535)).
- **Adjudication** — when two cheap judges disagree, send to the strong
  judge to break the tie.
- **Reference-based scoring** — judge gets a "gold-standard" reference
  answer plus the candidate, scores relative to the reference.
- **Auditing the cheap judge** — periodic re-grading by the strong judge
  to detect cheap-judge drift.

### Empirical agreement numbers
- Strong-judge vs human: 80–90% agreement, comparable to human-human
  inter-annotator agreement
  ([MT-Bench](https://arxiv.org/abs/2306.05685),
  [Survey](https://arxiv.org/html/2411.15594v6)).
- Cheap-judge vs strong-judge: empirically lower, often 60–80%, with
  bias concentrated in the extremes (leniency at the top, lower
  resolution in the tail). This is exactly the bias HealthBench's
  gpt-4o-mini cheap judge displays in our setup.

### Implication for our paper
The cheap/oracle split we use (gpt-4o-mini cheap, gpt-4.1 oracle, 25%
slice) is the standard production split, not a special construction.
The intro should mention this casually: "a cheap LLM judge labels
everything; a stronger oracle judge labels a small slice." Two lines
worth, not a long detour.
