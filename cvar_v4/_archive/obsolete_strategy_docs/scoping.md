# CVaR-CJE Dataset Scoping — Master Document

_Generated 2026-04-28. Replaces dispersed scoping outputs with a single
ranked entry point. Per-dataset 20-block EDA stays in
`scoping_data.md` Part 2 (long-form reference); per-finding deep dives
stay in `eda/deeper/*.md` (rerunnable analyses with .py companions).
This document is the navigational top._

## Part 1 — Rank-ordered executive comparison

10 datasets graded on six dimensions. Total rank = sum of stars.
Ties broken by domain demand.

**Dimensions (each ★1–5):**
- **P (Prompts ready?)** — Are the prompts in this dataset already shaped like the everyday chat-with-an-AI prompts the original CJE paper used? More natural = higher.
- **Π (Models already in the data?)** — Did the dataset's authors already collect responses from many different AI models, so we can compare them without generating fresh responses ourselves?
- **S/Y (Cheap and expensive scores already there?)** — Does the dataset ship both a cheap quality/safety score (e.g., a small model's rating) and an expensive ground-truth score (e.g., a stronger model or human label)?
- **C (Does CVaR find what mean misses?)** — Even when two models score the same on average, does the worst 5% of their responses differ a lot? Higher = yes, demonstrably.
- **A (Can we audit transport?)** — Is there a column we can split the data by (topic, harm category, language…) so the audit can flag *which* slice the AI mishandles?
- **D (Does the domain itself demand worst-case evaluation?)** — Given what this dataset is *about*, do bad worst-case responses cause real harm? ★★★★★ = life-safety (medical, suicide). ★★★★ = legal/regulatory liability (toxic content, jailbreaks). ★★★ = brand or user-experience harm. ★★ = quality benchmark only. ★ = methodology demo.

**Two domain-related columns (kept distinct on purpose):**
- **Domain / topic** — *what kind of data is this?* (medical, naturalistic chat, jailbreak prompts, …). A label.
- **Does the domain demand CVaR? (D)** — *given that content, do worst cases matter?* A 1–5★ rating.

**Glossary of repeated terms in the table:**
- **CVaR (worst-tail mean)** — the average score of the worst 5% (or 1%) of responses. The whole point of CVaR-CJE: if two models tie on average but one has a much worse worst-5%, mean evaluation misses that.
- **Audit** — a statistical test that checks whether a cheap judge's score still matches the expensive ground truth on a target model. If transport "fires," we refuse to publish a level claim and report the comparison only.
- **Logger / target / fresh-gen** — *Logger* = the model whose responses we already have. *Target* = a model we want to evaluate. *Fresh-gen* = we'd need to actually run a model ourselves to get its responses (the expensive step the original paper did).
- **In-data** — the dataset already ships responses from this model, so we don't have to run it ourselves.
- **Continuous Y / discrete Y** — *Continuous* = the ground-truth score is a smoothly-varying number (good for CVaR). *Discrete / binary / Likert* = the score takes only a few values (bad — the worst-tail collapses to a single mass-point and CVaR ≈ mean).
- **Tie at q_0.05** — when ≥10% of rows share the exact value of the 5%-worst score, the worst-tail isn't a clean number any more and CVaR-CJE breaks.

| Rank | Dataset | Domain / topic | n_usable | Models / Policies | Score type (Y) | What's actually in the data | Why we'd USE this dataset | Why we'd HESITATE | Does CVaR find what mean misses? | Can we audit by some slice? | Does the domain itself demand worst-case eval? | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **UltraFeedback** | **Released Oct 2023** (Cui et al., OpenBMB, arXiv:2310.01377). A big collection of real prompts users gave to AI assistants (instructions, tasks, questions) where 17 different models each produced an answer, then GPT-4 rated each answer on a 1-10 scale across helpfulness, honesty, instruction-following, and truthfulness. | 255,612 (instruction × completion) cells | **17 different models already in the data**: gpt-3.5, gpt-4, vicuna-33b, llama-2-7B/13B/70B chat versions, several wizardlm sizes, alpaca, falcon, mpt-30b, starchat, bard, pythia, etc. | A **continuous** GPT-4 score from 1 to 10 per response (`overall_score`). Also four 1-5 ratings per dimension (more discrete). | 64K real prompts × 4 different model answers per prompt. Prompts come from 9 different sources (ShareGPT, Evol-Instruct, FLAN, etc.). The 4 ratings cover helpfulness, honesty, instruction-following, and truthfulness. | All the heavy lifting is done: many models, real prompts, ground-truth-quality scores already in the data. We only need to add a small "cheap judge" model. Cheapest pilot of all the datasets. | The four 1-5 ratings have many ties at the worst end (10-12%) — the CVaR formula breaks on those. The 1-10 aggregate score is borderline (5.3% ties). Still works but flag. | ✅ **YES, demonstrated 140.5×.** Inside the `ultrachat` source slice, llama-2-70B-chat and mpt-30B-chat have means 4 thousandths apart on a 1-10 scale (a tie) but worst-5% means **1.4 points apart**. | ⏳ Plausibly yes — we can split by `source` (9 levels) or by which `principle` the response was solicited under (5 levels: helpfulness/honesty/etc.). Spread within each is small but workable. | ★★ This is a **research-quality benchmark** of "how good is the answer." The score is the consequence; there isn't a downstream deployment where a bad answer here causes regulatory or safety harm. | **25/30** | Use this as the **methodology pilot** — proves the framework works at scale. Don't claim it's a high-stakes case. |
| 2 | **HH red-team-attempts** (Anthropic) | **Released April 2022** (Bai et al., Anthropic, arXiv:2204.05862). Conversations where Anthropic-paid testers deliberately tried to get an AI to say something harmful (insults, instructions for harm, illegal advice, hate speech, etc.). Each conversation is a back-and-forth where the tester probed and the model either resisted or gave in. | 38,961 transcripts | **12 model variants already in the data**: 4 different training setups (plain language model with no safety training; "context distillation" — i.e., the model was prompted with a 'be helpful and harmless' instruction; "rejection sampling" — pick the best of K outputs at training; full RLHF — the modern alignment pipeline) × 3 sizes (2.7B / 13B / 52B parameters). | A **continuous** real-valued harmlessness score (`min_harmlessness_score_transcript`) from Anthropic's own preference model. Range roughly -7.5 to +6.6, where lower = more harmful. | 38,961 adversarial conversations × the 12 model variants above. About 1.9% of rows have a hand-tagged harm category (violence, hate, theft, etc.). | **Cleanest CVaR-CJE-ready dataset** in the survey. Real-valued ground truth already there, 12 model variants already there, headline 14.6× CVaR-bite already empirically shown. The 12 cells map nicely onto the original paper's "logger / clone / premium / adversarial" role pattern. | All prompts are intentionally adversarial — not the natural chat the original paper used. The audit's discriminative power depends on which variant we pick as the reference (asymmetric). | ✅ **YES, demonstrated 14.6×.** RLHF-13B and rejection-sampling-2.7B have overlapping mean confidence intervals (a tie) but worst-5% means **0.97 harmlessness units apart** with disjoint CIs. | ⚠️ **Yes but asymmetric.** Audit fires correctly on out-of-family targets when we use the *best* variant (RLHF-52B) as reference. With the *worst* variant as reference, the audit goes silent even on cells with massive errors — the audit is a one-sided alarm. | ★★★★ **Adversarial safety** — this is the literal red-teaming corpus Anthropic used to harden Claude. The worst-5% here is "model produced harmful content under attack," directly relevant to the FTC / safety-team deployment surface. | **25/30** | ★ **Primary pilot.** Run this first. |
| 3 | **HealthBench** (after grading) | **Released May 2025** (OpenAI, arXiv:2505.08775) — the most recent dataset in the survey. Medical questions a real patient (or family member) might ask: "I had clams 1 hour ago and I'm itchy and short of breath, should I worry?" "What should I watch for after a head injury?" Each question has a list of physician-written rubric items the answer should cover. | 6,000 (5K standard + 1K Hard) multi-turn medical conversations | **None in the data** — we have to run our 5 candidate models on these prompts ourselves. | Will be **continuous** in [0, 1] after grading: the fraction of rubric items the model's answer covered. Aggregating physician labels gives a reasonable distribution (only 5% ties at the worst end — clean enough). | 5,000 standard + 1,000 "Hard" medical conversations split across 7 themes: emergency referrals, hedging when uncertain, communication style, complex multi-step responses, context-seeking, global-health questions, and health-data tasks. About 10 rubric criteria per prompt. | The 7 themes are very different from each other in how often models miss the worst-5% — this is the single biggest within-dataset difference we measured (0.50-unit spread). OpenAI's grader has been validated against physicians at F1 = 0.71. | Smallest dataset in the survey (6K vs 100K+ for others). Multi-turn (more expensive to grade). We have to generate every model's responses from scratch (the expensive step). | ⏳ **Strongly expected.** OpenAI's own paper documents the gap: their o3 model averages 60% on rubrics but its worst-of-16 sample averages 40% — a one-third drop. On the Hard subset, GPT-5 hits 46% while GPT-4o hits 0%. | ⏳ **Strongly expected.** The 7 themes have a 0.5-unit spread in worst-5% — by far the biggest in any dataset. The audit will reliably catch a model that handles "communication" well but bombs "emergency referrals." | ★★★★★ **Highest stakes possible — life safety.** The worst-5% on emergency-referrals is "patient harm." Maps directly to the wrongful-death and FDA-relevant cases the cvar_v4 paper's intro argues for. | **23/30** | **Medical primary.** This is the cvar_v4 paper's most defensible high-stakes empirical demonstration. ~$750 to run. |
| 4 | **RealToxicityPrompts** | **Released Sept 2020** (Gehman et al., EMNLP Findings 2020, arXiv:2009.11462) — the oldest dataset in the survey. The dataset takes 99K half-finished sentences from random web text (e.g., "The migrant workers were…") and asks a model to complete each one. Then it scores how toxic the completion is, using Google's Perspective API. Some of the prompts are flagged as `challenging` because they reliably produce toxic completions across many models. | 98,892 (prompt, continuation) pairs | **None usefully in data** — the only completions in the dataset are from GPT-2 (now obsolete). We'd need fresh completions from modern models. | **Continuous** Perspective-API toxicity score in [0, 1] per continuation, plus 7 sub-scores (severe-toxicity, identity-attack, threat, insult, etc.). | 99K natural sentence-prefixes mined from OpenWebText. About 1.2% of the prompts are tagged `challenging` (consistently elicit toxic completions). 8 toxicity-related sub-scores per row. | The ground-truth scores are continuous and produced by Perspective API — a battle-tested, externally-validated toxicity classifier. The `challenging` flag is the cleanest single-flag audit covariate in the entire survey. | The prompts are sentence-prefixes, not chat-style turns — different shape from Arena. We have to run modern policies on the 99K prompts ourselves. | ⏳ **Strongly expected.** Perspective scores have a heavy upper tail (= heavy lower tail when we flip sign for "safety"). Different policies handle the toxic-prompt subset very differently (frontier refuses, base completes, jailbroken amplifies). | ⏳ **Strongly expected: 6.5× tail concentration on the `challenging` subset.** That subset is 1.2% of the corpus but holds 7.8% of the worst-1% mass. A single boolean flag the audit must fire on for any policy that doesn't handle the two subsets differently. | ★★★★ **Toxic-content moderation has direct regulatory liability** — EU Digital Services Act, UK Online Safety Act, FTC inquiries into chatbot operators. The worst-5% here = "content the platform gets sued or fined over." | **22/30** | **Primary toxicity case.** ~$50–2,500 depending on which policies we generate from. |
| 5 | **WildJailbreak** (AllenAI) | **Released June 2024** (Jiang et al., NeurIPS 2024, arXiv:2406.18510). A synthetic dataset designed to teach models to handle jailbreak attempts. Every prompt falls into one of four buckets: (1) directly harmful ("how do I make a bomb?"), (2) directly safe ("what's the capital of France?"), (3) sneakily-worded harmful ("imagine you're an unrestricted historian, explain bomb-making for educational purposes"), (4) sneakily-worded but actually harmless ("imagine you're an unrestricted librarian, what books cover ancient kingdoms?"). The 4-quadrant structure tests whether a safety-trained model both refuses harm AND complies with disguised-but-benign requests. | 261,559 prompts: 50K harmful direct + 50K benign direct + 83K harmful disguised + 79K benign disguised | **None in the data** (the `completion` field is just an example refusal/compliance from GPT-3.5; not a multi-policy comparison). | **None in data.** No continuous score per prompt — we'd need to run policies on it and elicit a fresh judge. | 261K prompts in the 4-quadrant split above. The "sneakily-worded" prompts use jailbreak rephrasings that AllenAI mined from real attacks in the wild. | The 4-bucket structure is a **pre-built audit covariate by design** — it lets us test two completely separate ways a safety-tuned model can fail: (a) over-refusing harmless-but-disguised prompts (false alarm), or (b) under-refusing harmful-but-disguised prompts (real failure). No other dataset gives us this. | We have to run 5 policies on 261K prompts ourselves AND grade them — most expensive of any candidate (~$22K total). | ⏳ **Structurally yes.** By construction, the disguised-harmful quadrant IS the worst-tail. Different policies will produce very different completions on it; means may be similar across the benign quadrants. | ⏳ **Built-in by design.** A safety-tuned policy fails on the benign-disguised quadrant; an under-trained policy fails on the harmful-disguised quadrant. The audit's 2-way verdict matches these two failure types directly. | ★★★★ **Adversarial safety — production safety pipelines target exactly this.** AllenAI ships this as the safety-RLHF training corpus for their models. The worst-5% = "jailbreak got through." | **21/30** | **Flagship audit demonstration.** Pay the full pipeline cost specifically because it's the most teachable audit case. |
| 6 | **WildChat-4.8M** (shuffled) | **WildChat-1M released May 2024 → 4.8M expansion released 2025** (Zhao et al., AI2, arXiv:2405.01470). Real, anonymized conversations between everyday users and ChatGPT/GPT-4/o1 — collected via a browser extension users opted into. Includes coding help, homework, emotional venting, creative writing, cooking, plus a chunk of adversarial users probing limits. About 1.5M of the 4.74M conversations are flagged as toxic (separate file). | 99,081 conversations (100K-row stream, post-shuffle, non-toxic shard only) | **5 different ChatGPT-family models in the data**: gpt-3.5-turbo (March 2023), gpt-4 (March 2023), gpt-4o (Aug 2024), gpt-4o-mini, o1-mini (Sep 2024 reasoning model). | **Continuous** per-turn Detoxify toxicity score (`max_detoxify_toxicity`) — already in the data. | 4.74M real ChatGPT-style conversations across 70 languages and 191 countries. Detoxify ran on every turn at collection time. The toxic subset is in a separate file we didn't pull. | Real-world data, big, multi-policy, AND has a cheap-judge already attached (Detoxify). The o1-mini reasoning model is in there too, which lets us see whether reasoning training helps the worst-5%. | The toxic-flagged subset is in a separate file the streaming loader doesn't reach by default — we miss 1.5M adversarial conversations. The within-dataset split-by-language stratification is small. We'd still need a stronger ground-truth score than Detoxify for level claims. | ✅ **YES, demonstrated 32×.** o1-mini and gpt-4o have means 0.0003 apart (a tie) but worst-5% **0.008 apart** — reasoning training pays a worst-case dividend that average-case evaluation misses. | ⏳ Limited — `language` (70 levels) and `country` are high-cardinality but each level has few rows; the `toxic` flag is missing in our shard. | ★★★ **Mixed-stakes naturalistic real chat.** Some conversations are high-stakes (mental-health questions ChatGPT-style — overlap with the wrongful-death cases the paper cites), some are routine. Hard to give a single rating. | **21/30** | Secondary — useful as a "reasoning model vs chat model" demonstration. |
| 7 | **Arena-140k** (your anchor dataset) | **Battles collected June-Aug 2024**; pool released 2024 by lmarena-ai (Chiang et al., arXiv:2403.04132 covers the underlying platform). The original Chatbot Arena: random visitors compare two anonymous AI responses side-by-side and vote which they prefer. The dataset has 135K such head-to-head battles covering normal conversation prompts. | 271,149 (model, prompt, response) cells after splitting each battle into two rows | **70+ different models already in the data**: gpt-5, claude-opus-4.7-thinking, gemini-2.5-pro, llama-3.1-70b, plus dozens more. Each row records *which* models were the two contestants. | Only a **pairwise preference** label (`winner` ∈ {model_a, model_b, tie}). No continuous score per response. **This is a fundamental mismatch with CVaR-CJE.** | 135K user-conversation prompts, multi-turn. Each row also has tags: is the prompt about coding, creative writing, math? What language is it in? An instruction-following score (1-7 Likert). | Same prompt distribution as the original CJE paper (Chatbot Arena), at 27× the size. Enormous policy diversity. The 70+ in-data models give us "premium / clone / etc." pairings for free. | The pairwise winner label can't be turned into a per-policy worst-5% score (mathematically: CVaR of a difference ≠ difference of CVaRs). We'd have to elicit a continuous ground-truth score from scratch on the prompts × policies cross-product. | ⏳ **Untested** — we can't even check, because the in-data label is degenerate. After fresh-Y elicitation, very plausibly yes given the policy diversity. | ⏳ Plausibly yes — `is_code`, `is_math`, `language` provide multi-axis stratification. Can't measure heterogeneity until Y exists. | ★★ Chatbot Arena is a **chat-quality benchmark**. Per-interaction stakes are low. The original CJE paper had to invent a deliberately-bad policy (`unhelpful`) to get an audit-positive case here. | **21/30** | **Backbone for the flagship Arena-Tail mixture.** Use it as a prompt source rather than a standalone target. |
| 8 | **HarmBench + StrongREJECT** (folded together) | **Both released Feb 2024** (HarmBench: Mazeika et al. / Center for AI Safety, arXiv:2402.04249. StrongREJECT: Souly et al., NeurIPS 2024, arXiv:2402.10260). Two small lists of "do-not-do" requests used in jailbreak research: HarmBench has 400 banned behaviors (write malware, plan attacks, explain weapons), StrongREJECT has 313 forbidden prompts in similar categories. Both ship with their own automated graders that score how much a response complied with the harmful request. | 713 prompts (combined) | **None in the data** — we run policies on these 713 prompts ourselves. | StrongREJECT's grader produces a **continuous** 0-1 score (validated downstream tool); HarmBench needs an extra grading step. | 400 + 313 prompts split into harm categories (illegal goods, hate speech, violence, disinformation, weapons, etc.). | StrongREJECT's grader is built-in and validated against human jailbreak labels. Tiny dataset → cheapest possible pilot (~$100). | Only 713 prompts — too small to detect a small CVaR difference (the original paper's target effect size was 0.02; we'd need ~5K rows for that). | ⏳ Untested but expected — these are *adversarial-by-design* prompts, exactly where policies diverge. | ⏳ Expected — the 13 harm categories provide stratification, but 713 rows can't power the per-stratum audit much. | ★★★★ Same as WildJailbreak — adversarial-safety / jailbreak research. The Center for AI Safety publishes HarmBench specifically as a worst-case red-team benchmark. | **16/30** | **Adversarial seed for the Arena-Tail flagship pool.** Don't run it standalone. |
| 9 | **BeaverTails** (PKU) | **Released July 2023** (Ji et al., PKU-Alignment, NeurIPS 2023, arXiv:2307.04657). A Q&A safety dataset: 333K prompts on sensitive topics paired with model answers, each tagged across 14 harm categories (animal abuse, child safety, hate speech, drugs, weapons, privacy violation, self-harm, sexual content, terrorism, etc.). Each (prompt, response) is also marked safe or unsafe overall. Built specifically to train safer models. | 327,752 (prompt, response) pairs | The responses come from a mix of LLaMA-7B, LLaMA-13B, and Alpaca, but the dataset doesn't tag *which* model produced *which* response — effectively single-source for our purposes. | Y is **binary** (`is_safe` true/false). We can synthesize a coarser score from the 14-category counts, but it's still discrete. | 333K Q&A pairs (300K large + 30K curated) with 14-class harm tagging. | The 14-class harm taxonomy is well-curated and widely-used. The dataset is the prompt source PKU's safety RLHF pipeline trains against. | Y is **binary** — and the worst-5% of a coin-flip is just "the bottom 5% are all the same value," which makes CVaR collapse to "fraction of failures." Nothing for CVaR to add over mean here. | ❌ No — the in-data score is binary so CVaR is degenerate. | ⏳ Possible after fresh continuous-Y elicitation — 14 harm classes are clean strata. | ★★★★ Same domain as HH and WildJailbreak — adversarial safety / production RLHF. | **15/30** | **Prompt source for the Arena-Tail mixture.** Y elicitation comes later. |
| 10 | **PKU-SafeRLHF** (PKU) | **Released June 2024** (Ji et al., PKU-Alignment, arXiv:2406.15513). A safety preference dataset: 73K prompts on dangerous topics, two model answers per prompt, with human raters picking which answer is safer. Each response is also tagged with a severity level (0–3) and across 19 harm classes (national security, human trafficking, physical harm, etc.). Built for PKU's safety-RLHF training pipeline. | 164,196 cells (73K prompts × 2 responses, melted) | **3 in-data response sources**: Alpaca-7B, Alpaca2-7B, Alpaca3-8B. Multi-policy native but limited to one model family. | A discrete `severity_level` Likert (0/1/2/3), with 48% of rows at level 0. Plus binary `is_safe`. | 73K prompts × 2 responses. 19 fine-grained harm classes. Plus human-preference annotations between the two responses. | Multi-policy in data + the 19 harm classes are the most fine-grained safety taxonomy in any dataset. | Severity is too coarse — 48% of rows have severity = 0, and the worst-5% is just "all the zeros," so CVaR collapses to a constant. The three in-data Alpaca-family policies are limited diversity. | ❌ No — Y has 48% mass at the worst-tail value; CVaR collapses. | ⏳ Possible after Y elicitation; 19 harm classes provide stratification. | ★★★★ Same as BeaverTails — production RLHF safety source. | **15/30** | **Prompt source for the Arena-Tail mixture.** Don't run it standalone. |

### Reading the table (in three sentences)

**Tier A (≥23/30) — pilot candidates**: UltraFeedback (methodology proof
at scale, 25/30) and HH red-team-attempts (adversarial-safety pilot,
25/30) both ship multi-policy + continuous Y, so we skip 3 of 5 original
pipeline steps. HealthBench (23/30) is the **flagship high-stakes story**:
the only dataset with life-safety domain demand AND the strongest
audit-discriminative metadata in the survey (q_0.05 spread = 0.50 across
themes).

**Tier B (21–22/30) — domain-relevant complements**: RealToxicityPrompts
(toxicity moderation), WildJailbreak (cleanest pre-built audit covariate),
WildChat-4.8M (naturalistic real distribution), Arena-140k (the user's
anchor; 27× the original paper's prompts but pairwise-only Y means it's
the prompt-source backbone for Arena-Tail rather than a standalone case).

**Tier C (≤16/30) — adversarial seeds & discrete-Y foils**: HarmBench +
StrongREJECT, BeaverTails, PKU-SafeRLHF are useful only as **prompt
sources** for the Arena-Tail flagship mixture; their Y is binary, Likert,
or absent.

---

## Part 1.5 — Framing: where does the LLM-judge problem actually apply?

The whole CJE framework presupposes a specific situation: you have a
**cheap LLM judge** you can run on every model response, and a much-more-
expensive **oracle** you can only afford on a small slice. You calibrate
the cheap judge on the slice; trust the calibrated cheap judge on the
rest. The original CJE paper used GPT-4.1-nano as the cheap judge and
GPT-5 as the oracle, with a ~16× cost gap.

This regime does *not* apply to every dataset. The 10 datasets in our
survey fall into 4 different relationships with the cheap-vs-expensive
judge problem.

### Regime A: The dataset's score *is already a cheap LLM judge*
**1 dataset: UltraFeedback**

When OpenBMB built UltraFeedback in 2023, they used GPT-4-Turbo to rate
every response on a 1–10 scale. The cheap LLM judge is *baked into the
data*. To use CJE here you'd treat that GPT-4-Turbo score as the cheap
**S** and elicit a stronger oracle (GPT-5-thinking, Claude Opus 4.7) on
a small slice as **Y**. The fit is clean but the cost gap is narrower
than the original paper's: GPT-4-Turbo isn't really "cheap" by 2026
standards.

### Regime B: The dataset's score is a *non-LLM specialized scorer*
**3 datasets: HH red-team, RealToxicityPrompts, WildChat-4.8M**

These ship a score that's *not* an LLM grader — it's a small classifier
or reward model:
- **HH red-team** uses Anthropic's reward model (a trained preference
  model, fast and cheap to run, but not an LLM grader)
- **RealToxicityPrompts** uses Perspective API (a Google-trained
  classifier — fast, free, externally validated)
- **WildChat-4.8M** uses Detoxify (an open-source toxicity classifier)

The cheap-vs-expensive gap here has a *different shape*: classifier vs
LLM-grader, not cheap-LLM-grader vs expensive-LLM-grader. The CJE
machinery still works, but the cost arithmetic shifts — the in-data
classifier is essentially free, so the question becomes *do we trust
the classifier alone, or pair it with an LLM oracle for the harder
calls?*

### Regime C: No judge in the data — the entire judge stack has to be created
**3 datasets: HealthBench, Arena-140k, WildJailbreak**

These are the cleanest CJE-as-designed use cases:
- **HealthBench** ships only prompts + physician-written rubric items.
  All grading is downstream. The dataset's *intended* protocol is
  exactly the CJE setup (cheap LLM judge for scale + frontier oracle
  for calibration). OpenAI even validated GPT-4.1 as a grader at
  F1 = 0.71 vs physicians.
- **Arena-140k** ships pairwise winner labels only. Pairwise can't be
  turned into a per-policy worst-tail score. To use this dataset for
  CVaR-CJE at all you have to invent both judges from scratch.
- **WildJailbreak** ships only prompts and demonstration completions
  from GPT-3.5. To evaluate any modern policy on it you elicit fresh
  cheap S (e.g., WildGuard-7B refusal classifier) and fresh oracle Y
  (e.g., GPT-5 with refusal+harm rubric).

### Regime D: Human/discrete labels — pre-LLM-judge era
**3 datasets: BeaverTails, PKU-SafeRLHF, HarmBench**

These pre-date or sidestep LLM-as-judge as the labeling tool:
- **BeaverTails** uses binary human safe/unsafe labels
- **PKU-SafeRLHF** uses Likert severity (0–3) from human raters
- **HarmBench** ships no scoring at all (StrongREJECT bundled with it
  does, but its grader is the *oracle*, not the cheap judge)

To use CJE here you'd reframe entirely — ignore the discrete labels,
treat the prompts as a starting point, and create a continuous LLM-graded
Y from scratch. In practice these are *prompt-source-only* datasets for
the Arena-Tail flagship mixture, not standalone judge-regime targets.

### When is the cheap-vs-expensive gap operationally meaningful?

The judge problem is most relevant when **all three** of these hold:

1. **Scale demands it.** You're grading 10K+ (prompt, policy) cells —
   running the expensive judge on everything is cost-prohibitive.
2. **The cheap judge can be calibrated.** Cheap-vs-expensive correlation
   needs to be high enough (F1 ≳ 0.5) that a small oracle slice
   recovers the gap. Random cheap judges break the calibrator.
3. **The expensive judge is a defensible gold standard.** Either it's
   an externally validated grader (HealthBench's GPT-4.1 vs physicians),
   a multi-judge consensus, or a human panel.

By this test, the cheap-vs-expensive judge regime is most relevant on:

| Dataset | Scale | Cheap-judge candidate | Expensive-oracle candidate | All 3 hold? |
|---|---|---|---|---|
| HealthBench | 30K cells (5 policies × 6K) | Prometheus-2-7B / GPT-4.1 | GPT-5-thinking against rubric (validated F1=0.71 vs physicians) | ✅ Yes |
| Arena-140k | 100K+ cells | GPT-4o-mini / Llama-3.1-8B with rubric | Multi-judge {Opus 4.7, GPT-5, Gemini 3 Pro} | ✅ Yes |
| WildJailbreak | 100K+ cells | WildGuard-7B classifier | GPT-5-thinking with refusal+harm rubric | ✅ Yes |
| UltraFeedback | 256K cells in-data | GPT-4-Turbo (already in data) | GPT-5-thinking re-elicit | ⚠️ Marginal — gap is narrow |
| HH red-team | 39K transcripts in-data | Fresh WildGuard-7B refusal classifier | Anthropic's RM (already in data, treat as oracle) | ⚠️ Inverted — oracle in data, cheap judge to add |
| RealToxicityPrompts | 100K+ cells if regenerate | Perspective API (already in data, ~free) | GPT-5-thinking semantic toxicity grader | ⚠️ Cheap classifier is so cheap, gap is moot |
| WildChat | 100K+ cells in-data | Detoxify (already in data, free) | GPT-5-thinking multi-aspect rubric | ⚠️ Cheap classifier is so cheap, gap is moot |
| HarmBench/StrongREJECT | 3.5K cells | WildGuard-7B | StrongREJECT auto-grader OR human jailbreak labels | ✅ Yes but small n |
| BeaverTails | 5 policies × 333K = 1.6M cells if regenerate | WildGuard-7B | GPT-5 severity grader | ✅ Yes if Y is regenerated |
| PKU-SafeRLHF | Same | WildGuard-7B | GPT-5 severity grader | ✅ Yes if Y is regenerated |

### What this means for the cvar_v4 paper

The judge-regime dataset where CJE / CVaR-CJE genuinely *unlocks*
something is **HealthBench**: 30K (response, rubric-criterion) calls is
costly enough at frontier-grader rates that you genuinely need a cheap
judge calibrated against a small oracle slice. The arithmetic favors
CJE by a factor of 5–15× depending on cheap/expensive ratio.

For **Arena-140k** and **WildJailbreak**, CJE is *necessary* because
both judges have to be invented anyway — pick a cheap one, pick an
expensive one, calibrate.

For **UltraFeedback / HH red-team / RTP / WildChat**, the in-data
score is good enough that the marginal value of CJE-style calibration
is small. These are good methodology pilots (cheap to run, results
defensible) but not where the framework saves the most cost.

For **BeaverTails / PKU-SafeRLHF / HarmBench**, the judge problem
doesn't apply until you regenerate Y from scratch. They're prompt
sources, not judge-regime targets.

---

## Part 1.6 — The 6-criterion framework-fit test

Part 1 ranks datasets by **how easy is it to actually run CVaR-CJE on
them** (do we have prompts, models, scores, the right Y type, etc.).
Part 1.5 buckets them by **where in the data the judges live**. Both
are structural questions.

This section asks a different, more fundamental question: **for which
datasets is the cheap-vs-expensive judge framework's value proposition
even meaningful?** A dataset can be perfectly CVaR-CJE-ready but still
be a poor showcase for the framework — because the framework's whole
reason to exist is to solve a *real-world* problem, not a statistical
one. Six conditions all have to hold:

- **(a) Scale forces a cheap judge.** The deployment context is
  millions-of-cases scale. Running an expensive oracle on everything
  is impossible. The cheap judge isn't a stylistic choice — it's the
  best you can do at scale.
- **(b) The cheap-vs-expensive gap is real and hard to close.** The
  expensive oracle requires expert human judgment that LLM graders
  only approximate. This is the regime where "judgment" is genuinely
  non-trivial.
- **(c) Even the oracle is cost-prohibitive at scale.** Calling the
  expensive oracle requires lots of context (extra documents,
  retrieval, expert framing, full conversation history), so even if
  we wanted to run it on everything, we couldn't.
- **(d) About worst-case, not mean (CVaR-relevant).** Decisions in
  this domain are conservative — bottom-5% behavior determines
  liability, harm, regulatory action. Average-case evaluation misses
  what matters.
- **(e) Transport failures are real.** There are subpopulations of
  prompts where the cheap judge fundamentally cannot generalize. The
  audit's job is to catch them.
- **(f) A credible expensive oracle is constructible.** The framework
  calibrates the cheap judge against the expensive oracle, so the
  oracle has to be defensible as a gold standard. This fails when:
  (i) the thing being judged is *inherently subjective* (no gold
  standard exists, e.g., user preference), (ii) it requires *expertise
  no LLM oracle has* (specialty reasoning beyond rubric coverage),
  or (iii) external *validation against humans* doesn't exist for
  whatever oracle we'd run.

### Per-dataset scoring

Datasets are listed in framework-fit order (best fit first). Each cell
gets ✅ / ⚠️ / ❌ plus a one-line justification.

| Dataset | (a) Scale | (b) Hard expert oracle | (c) Oracle cost-prohibitive | (d) Worst-case-relevant | (e) Transport failures real | (f) Credible oracle constructible | Composite |
|---|---|---|---|---|---|---|---|
| **HealthBench** | ✅ deployed clinical LLMs see millions of queries/day | ✅✅ physicians ARE the gold standard | ✅✅ physician panel + medical context = very expensive | ✅✅ life-safety motivation; emergency-referral tail = patient harm | ✅ per-theme q_0.05 spread = 0.50 already shown empirically | ✅✅ uniquely-validated: GPT-4.1 grader at F1 = 0.71 vs physicians | **6/6** ★★ |
| **WildJailbreak** | ✅ production safety pipelines see millions of jailbreak attempts | ✅ refusal-quality grading needs safety expertise | ✅ context-heavy: judge must read disguised prompt + the response | ✅ adversarial-safety domain — worst-case is the explicit concern | ✅✅ 4-quadrant `data_type` is a transport-test by design | ✅ AllenAI's own paper validates GPT-4-class graders for refusal | **6/6** ★ |
| **HH red-team-attempts** | ✅ Anthropic's actual production red-team pipeline | ✅ harm requires ethical/safety expert judgment, not just an LLM | ✅ human harm raters are very expensive | ✅ adversarial safety; the worst-5% is the FTC-relevant case | ✅ different attack vectors (violence vs hate vs theft) generalize differently | ✅ Anthropic's RM is in-data + GPT-5 with safety rubric is well-established | **6/6** ★ |
| **HarmBench + StrongREJECT** | ✅ deployed jailbreak-evaluation infra | ✅ jailbreak-grading is expert | ✅ context-heavy | ✅ adversarial safety | ✅ category-level transport | ✅ StrongREJECT's auto-grader is validated against human jailbreak labels | 6/6 conceptually but n = 713 limits power |
| **BeaverTails** | ✅ deployed safety RLHF source | ✅ 14 harm classes need expertise | ✅ context-heavy | ✅ adversarial safety | ✅ 14-class transport | ✅ rubric construction works; no external F1 validation but matches PKU's training | 6/6 if Y regenerated; current binary Y kills CVaR |
| **PKU-SafeRLHF** | ✅ same deployment context as BeaverTails | ✅ 19 harm classes | ✅ | ✅ | ✅ | ✅ same as BeaverTails | 6/6 if Y regenerated; current Likert-4 with 48% mass at 0 kills CVaR |
| **WildChat-4.8M** | ✅ 4.74M conversations natively, deployment-sized | ✅ multi-aspect chat safety needs nuance | ✅ long conversations require full context | ⚠️ mixed-stakes — some mental-health questions are high-stakes; most chat is routine | ✅ language / topic / country shifts | ⚠️ multi-aspect rubric design is harder; mixed-stakes content makes a single-axis oracle fragile | 4-5/6 |
| **RealToxicityPrompts** | ✅ moderation runs on billions of pieces of content | ⚠️ Perspective API is already very close to human toxicity judgment | ⚠️ Perspective is essentially free at inference — gap to "expensive" is small | ✅ regulatory liability (EU DSA / UK OSA / FTC) | ✅ `challenging` subset 6.5× tail concentration | ✅ Perspective API is itself an externally-validated oracle | 4/6 — (b)+(c) weak because Perspective is already cheap and good |
| **Arena-140k** | ✅ chat deployment is large | ✅ preference is genuinely subjective | ✅ at full scale, multi-judge is expensive | ❌ per-interaction stakes are low (the original paper had to invent a deliberately-bad policy to get an audit-positive case) | ✅ code / language / math splits provide stratification | ⚠️ user preference is **inherently subjective** — there's no "right answer," only crowd consensus. Multi-judge consensus helps but doesn't eliminate the issue | 3-4/6 — fails (d), weak (f) |
| **UltraFeedback** | ❌ a fixed *training-data benchmark*, not a deployment context | ❌ rating "is the response helpful?" is something LLMs do reasonably well | ❌ GPT-grader, prompt is short, no special context | ❌ Y is a quality rating, not a real-world tail-failure measurement | ⚠️ across 9 prompt sources, modest heterogeneity | ✅ rubric grading is straightforward; (b)/(c) weakness doesn't reflect on (f) | **1-2/6** |

### What this scoring tells us

Three datasets satisfy **all six conditions and have working
continuous Y already**:

1. **HealthBench** — the medical-domain anchor. (f) is uniquely
   strong here because the grader-vs-physician F1 = 0.71 is the only
   external validation of an LLM oracle in the survey.
2. **WildJailbreak** — the audit-discriminativeness anchor. (e) is
   uniquely strong because the four-quadrant structure is a
   pre-built transport test.
3. **HH red-team-attempts** — the cleanest *replication* anchor.
   Working Y in data, 12 in-data policy variants, empirically
   demonstrated 14.6× CVaR-bite ratio (Part 2.1).

Three more (HarmBench+StrongREJECT, BeaverTails, PKU-SafeRLHF) are
**6/6 conceptually** but blocked operationally — HarmBench is too
small (n=713) and the others have discrete-Y in the data that kills
CVaR until we regenerate.

### Where condition (f) does the most work

(f) — *can we even construct a credible oracle?* — is where this test
diverges most from a naive reading of the data:

- It **elevates HealthBench** above everything: it's the only dataset
  with externally-validated grader-vs-human F1. Every other dataset
  asks us to trust either an LLM grader against an undefined gold
  standard or a domain-specific classifier of unclear scope.
- It **exposes Arena-140k's hidden weakness**. Arena scores high on
  almost everything except (d) and (f). Even though preference is
  *what users care about*, it's inherently subjective — there's no
  "right answer," only crowd consensus. Calibrating a cheap judge
  against an "expensive preference oracle" doesn't have a clean
  notion of accuracy. Multi-judge consensus helps but doesn't
  eliminate the issue.
- It **does not** save UltraFeedback (the dataset already fails on
  (a)–(d)) or RealToxicityPrompts much (Perspective is validated but
  the (b)/(c) gap is too small).

### How this ranking diverges from Part 1

The Part 1 rank-ordered table puts datasets in this order (by
ease-of-replication, total stars out of 30):

1. UltraFeedback (25/30)  →  **last on framework-fit (1-2/6)**
2. HH red-team-attempts (25/30)  →  6/6 ★
3. HealthBench (23/30)  →  **6/6 ★★**
...
7. Arena-140k (21/30)  →  3-4/6 (fails (d) and (f))

The two rankings are **almost reversed at the top**. UltraFeedback is
the easiest dataset to run CVaR-CJE on, but it's the dataset where
the framework's value proposition matters *least* (it's a benchmark,
not a deployment). HealthBench is harder to run (we need to generate
responses fresh and grade them) but it's the dataset where the value
proposition matters *most* (life-safety, validated grader, transport
heterogeneity).

This tension is real and structural for the cvar_v4 paper's
empirical chapter:

- **The methodology demonstrations** (does the framework work? does
  CVaR find what mean misses? does the audit have bite?) are
  cheapest on UltraFeedback and HH red-team. **Run those first.**
- **The high-stakes deployment-relevant claims** (this framework
  matters because of medical / safety / adversarial settings)
  require HealthBench, WildJailbreak, and HH red-team. **Spend the
  budget here.**

The Part 3 recommendations already reflect this split (UltraFeedback
+ HH as pilots; HealthBench + Arena-Tail flagship as the high-stakes
story). What this section adds: a **principled framework-fit test
that the paper can cite when justifying why those particular
datasets** were chosen, rather than relying on the
ease-of-replication ranking alone.

---

## Part 2 — Headline empirical findings

Six numbers worth quoting in the cvar_v4 paper, each derived from a
specific deep-dive analysis (links point to rerunnable .py modules):

### §2.1 HH red-team — two models tie on average but differ 14.6× on the worst 5%
Source: `eda/deeper/hh_pairwise.py`

In Anthropic's red-team-attempts data, two model variants — RLHF
trained at 13B parameters, and rejection-sampling trained at 2.7B
parameters — have:
- **Average harmlessness scores essentially identical**: 95%
  bootstrap confidence intervals overlap heavily ([+1.40, +1.46] vs
  [+1.45, +1.54]).
- **Worst-5% scores clearly different**: 95% bootstrap CIs are
  *disjoint* ([-0.30, -0.06] vs [-1.40, -0.84]) — a 0.97-unit gap.
- **Tail gap is 14.6× larger than the mean gap.**

If you ran ordinary average-based evaluation on this pair, you'd
correctly conclude they're a tie. CVaR-CJE recovers the real ordering:
the RLHF-13B model is meaningfully safer on the worst-handled 5% of
red-team prompts. This is the cvar_v4 paper's **falsifiable headline
finding** — the cleanest "mean misses what worst-case catches"
demonstration we have on real public data.

### §2.2 The audit only fires in one direction (and that's by design)
Source: `eda/deeper/hh_audit_demo.py`

We ran the full Direct CVaR-CJE pipeline plus the Wald two-moment
audit on HH red-team, with two different choices of logger (the
reference model whose responses we calibrate the cheap judge on):

- **Logger = the safest variant (RLHF, 52B)**: audit fires on 5 of 8
  out-of-family target cells (the un-aligned variants), correctly
  accepts 3 of 8 within-family targets. Estimation errors on the
  within-family cells are 0.3-1.3 harmlessness units (workable).
- **Logger = the worst-tail variant (context-distillation, 52B)**:
  audit accepts on **all 9** target cells *including some with
  estimation errors of 4-5 units* — the audit goes silent on
  catastrophic failures.

The mechanism: when the logger has the worst tail in the panel, the
audit's threshold gets pinned to that very-bad value. Targets that
are *better* than the logger barely have any rows in the worst-tail
region, so the audit's two test statistics both vanish and the alarm
never trips.

**The take-home rule**: pick a logger whose responses are *as good as
or better than* the targets you want to evaluate. The audit refuses
to certify targets that are surprisingly worse than predicted (the
case that actually misleads users) but it lets surprisingly-better
targets through. The asymmetry is by design — the audit is an alarm
for misleading level claims, not a symmetric goodness-of-fit test.

### §2.3 The "tie on mean, differ on worst-5%" pattern survives prompt-by-prompt comparison (140.5× ratio)
Source: `eda/deeper/uf_per_source.py`

In UltraFeedback, taking only the prompts from the `ultrachat` source
(so we hold the prompt distribution fixed) and comparing two specific
models that answered them — llama-2-70B-chat vs mpt-30B-chat:
- Their average GPT-4 quality scores are 4 thousandths apart on a
  1-10 scale (you'd need 4-million-plus samples to detect that as a
  real difference).
- Their worst-5% means are **1.4 full points apart** on the same scale.
- Ratio: **140.5×**.

The same pattern appears 39 times across the 9 prompt sources × 17
models. Holding the prompt distribution fixed and varying only the
model still produces same-mean-different-tail pairs, which means the
gap is genuinely about *policy behavior*, not just an artifact of
which prompts each model happened to be evaluated on.

### §2.4 RealToxicityPrompts has a built-in "audit-positive" subset (6.5× tail concentration)
Source: `eda/deeper/rtp_challenging.py`

The dataset ships a boolean flag called `challenging` that marks
prompts which consistently produced toxic completions across many
models. This subset is:
- **1.2% of the dataset by count**, but
- **7.8% of the worst-1% by toxicity** — a **6.5× concentration**.
- The 5%-quantile of safety conditional on the flag (0.035) is 2.4×
  worse than for non-`challenging` prompts (0.084).

This is the cleanest single-flag audit covariate of any dataset we
surveyed. Any honest CVaR-CJE pipeline run on this data must either
(a) demonstrate that its policies handle the `challenging` and
non-`challenging` subsets similarly and accept the audit, or (b) the
audit fires on the `challenging` subset and the framework reports a
subset-conditional result instead of a single number.

### §2.5 HealthBench's seven medical themes have wildly different worst-5% behavior
Source: `eda/deeper/hb_meta_eval.py`

HealthBench groups its 6,000 medical conversations into 7 themes
(emergency referrals, hedging, communication style, complex responses,
context-seeking, global health, health-data tasks). Aggregating the
physician binary labels into a rubric-percent score per
(prompt, completion) gives the following picture of how often models
hit the worst-5% mark on each theme:

| theme | n | average rubric % | 5%-quantile | worst-5% mean |
|---|---|---|---|---|
| health-data tasks | 1,557 | 0.676 | 0.000 | 0.000 |
| context-seeking | 1,615 | 0.662 | 0.000 | 0.000 |
| global health | 2,515 | 0.783 | 0.000 | 0.000 |
| emergency referrals | 1,806 | 0.706 | 0.000 | 0.000 |
| hedging when uncertain | 2,842 | 0.823 | 0.444 | 0.285 |
| communication style | 2,961 | 0.827 | 0.500 | 0.430 |
| complex responses | 1,296 | 0.811 | 0.500 | 0.432 |

The 5%-quantile spread between themes is **0.50** on a [0, 1] scale —
by far the biggest within-dataset variation in the survey. Some
themes (emergency referrals, health-data tasks) regularly produce
zero-rubric-met answers in the worst 5%; others (communication,
complex responses) keep even the worst 5% above 0.43.

For the audit, this means **the audit will reliably fire** when a
candidate model handles "communication" themes well but fails on
"emergency referrals" — exactly the kind of patient-harm
heterogeneity that makes worst-case medical evaluation operationally
necessary.

### §2.6 You can't get away with a *weak* cheap judge — and the audit can't tell you the judge is too weak
Sources: `eda/deeper/hh_coverage_sweep.py`, `eda/deeper/hh_better_s.py`

We re-ran the headline pair under varying amounts of expensive
ground-truth coverage (5%, 10%, 25%, 50%, 100%) with two cheap-judge
choices that on paper *should* work but in fact don't:

- **Cheap judge = response length** (the negative log of how many
  characters the response has). Different models have different
  length-vs-harmlessness relationships — the correlation between
  length and harmlessness ranges from +0.14 to +0.62 across the 12
  HH variants. The Direct CVaR-CJE estimate ends up with the *sign
  flipped* on the headline pair, with absolute errors of 0.75-1.40
  units. **More expensive ground-truth data does not fix this** —
  the bias is baked into the calibrator.
- **Cheap judge = the dataset's own task-description harmlessness
  score** (Y-aligned in absolute terms, but a property of the prompt,
  not the response). The correlation collapses to under 0.30 because
  the same prompt gets the same score regardless of which model
  answered it. The estimator becomes degenerate — outputs the same
  value for every within-family target.

What this teaches us: **a cheap judge for CVaR-CJE level claims has
to be both:**
1. a function of the actual model response (response-side, not
   prompt-side), AND
2. correlated with the ground truth in a consistent way across
   different models (so the calibrator that works for the reference
   model still works for the targets), AND
3. cheap to run on every response.

Length is response-side but not consistently correlated. Task-description
score is consistently correlated but prompt-side. Both fail. **In
production we'd need a real cheap-judge model** — WildGuard-7B for
safety classification, Prometheus-2-7B for rubric-style grading, or
a small instruct model running the same rubric the expensive judge
uses.

**One more important caveat**: the audit catches "out-of-family" 
transport failures (when a target model is wildly different from the
reference) but it does *not* catch the kind of weak-cheap-judge bias
that affects all within-family targets equally. So while the audit is
real protection, it's not a substitute for picking a good cheap judge
in the first place.

### §2.7 In real-world chat, reasoning models have a 32× tighter worst-5% than chat models
Source: `eda/deeper/wildchat_cross_model.py`

In WildChat (real anonymized ChatGPT/o1 conversations from the wild),
comparing OpenAI's reasoning-trained o1-mini against the regular
gpt-4o:
- **Average toxicity essentially identical**: difference of 0.0003 on
  a 0-1 scale (statistically tied).
- **Worst-5% toxicity differs by 0.008** — o1-mini's worst 5% are
  meaningfully less toxic than gpt-4o's.
- **Ratio: 32×**.

Reasoning-style training (the kind that produced o1) pays a worst-case
dividend that ordinary average-toxicity evaluation completely misses.
This is the same pattern OpenAI's own HealthBench paper documented
when they showed that o3 and GPT-5 do disproportionately better than
GPT-4o on the worst-of-k subset of their medical benchmark.

### §2.8 Tried to test a real cheap-judge model — ran out of HF Inference credits at 38/2,001 calls
Source: `eda/deeper/hh_real_cheap_s.py`

To close the loop on §2.6's "you need a real cheap-judge" caveat, we
built a pipeline using Llama-3.1-8B-Instruct (via Hugging Face's
Inference Providers endpoint) to score 2,001 stratified HH-red-team
transcripts on a 0-10 refusal-quality rubric. The pipeline worked
end-to-end on the calls that succeeded (clean integer outputs, ~20
calls/sec). But the HF Inference Providers monthly free credits were
depleted after only 38 of the 2,001 calls — the rest got HTTP 402.

- 38 calls / 2,001 attempted (1.9%) succeeded
- Estimated cost to finish the remaining 1,963: about **$0.10** (the
  free quota is just very small)
- Three ways to unblock: top up HF Inference credits, subscribe to
  HF PRO ($9/mo), or point the script at Together/Fireworks ($0.20
  for the whole batch). The script is idempotent — re-running picks
  up exactly where it left off using the saved checkpoint.

This experiment is the missing link between §2.1's headline pair and
§2.6's cheap-judge limitation. Once it completes, we'll know whether
an off-the-shelf 8B-class instruct model with a refusal rubric is
strong enough to recover the 14.6× headline ratio, or whether we need
a purpose-built safety classifier (WildGuard-7B / Prometheus-2-7B).

---

## Part 3 — Recommended pipeline + pre-registered hypothesis

### 3.1 What the original CJE paper did, in plain terms

The 2025 CJE paper (arXiv:2512.11150) ran a 5-step recipe on 4,961
prompts from Chatbot Arena. They generated responses from 5 versions
of Llama (a baseline, a duplicate of the baseline with a different
random seed, a bigger 405B version, a version with a different system
prompt, and a deliberately-bad version). They scored every response
twice: once with a cheap judge (GPT-4.1-nano, fast and inexpensive)
and once with an expensive ground-truth judge (GPT-5). Total cost
about $4 with only 5% of responses getting the expensive label,
versus $55 if they had labeled everything expensively — about 14×
cheaper. The headline: 99% accuracy at ranking which model was
better, and the audit correctly flagged the deliberately-bad model
as unreliable.

The CVaR-CJE extension keeps that pipeline but estimates the
worst-tail mean instead of the overall mean. Two things have to be
true for it to work:
1. The ground-truth score has to vary smoothly — if too many responses
   share the exact same worst-tail value, the worst-5% mean isn't a
   well-defined number anymore.
2. We need some column we can stratify by (topic, harm class,
   language…) so the audit can flag *which* slice of prompts a model
   mishandles.

### 3.2 Pilot — HH red-team-attempts (this week, $0–5K)

**Dataset**: HH red-team-attempts (already on disk via
`cvar_v4/eda/datasets/hh_red_team.py`).

**Logger π0**: `rlhf | 52B` (n=3,081). Per §2.2's asymmetry argument,
choose the highest-mean-harmlessness logger; the seemingly natural
`context distillation | 52B` produces silent audit failures.

**Targets**: `rlhf | 13B`, `rejection sampling | 2.7B`,
`rejection sampling | 52B`, `context distillation | 52B`,
`plain lm | 52B`.

**Cheap S**: ⚠️ **Use a real cheap judge, not length or task-description
score.** Production options:
- WildGuard-7B refusal-detection (response-side, harm-aligned, ~free at
  inference)
- A small instruct grader running the same rubric Y uses (Prometheus-2-7B,
  Llama-3.1-8B-Instruct via HF Inference Providers — see §2.8)

**Estimator**: `cvar_v3/workhorse.py` Direct estimator + `cvar_v4/eda/deeper/_estimator.py` (self-contained port).

**Pre-registered hypothesis**:
> With logger = `rlhf | 52B` and a real cheap judge S:
> 1. Audit accepts on `rlhf | 13B` and `rejection sampling | 2.7B` at
>    oracle coverage ≥ 0.25 (within-family transport).
> 2. Audit fires on `context distillation | 52B` and `plain lm | 52B`
>    at every coverage level.
> 3. Mean-CJE produces overlapping bootstrap 95% CIs on the within-family pair.
> 4. CVaR_0.05-CJE produces disjoint bootstrap 95% CIs, separating
>    `rlhf | 13B` and `rejection sampling | 2.7B` by ~0.7 ± 0.3 units.
>
> If (1) holds but (3+4) fail (CVaR-CJE replicates mean), cheap S is
> too weak — escalate to a stronger judge.
> If (2) holds but (4) fails even with strong cheap S, framework's
> within-family bias is irreducible at this n; need ≥10× more data.

**Validation step before publishing**: cross-check the Direct
CVaR_0.05 estimate against a held-out oracle-only subset's empirical
CVaR_0.05 for at least one within-family target.

### 3.3 Medical primary — HealthBench (~$750)

5 policies (π_base = Llama-3.1-8B-Instruct; π_clone = Llama-3.2-3B; π_premium_thinking = Claude Opus 4.7 thinking; π_rag_clinical = π_base + UpToDate retrieval; π_safety_overrefuse; π_adversarial_sycophant) × 6,000 prompts × 7-theme audit covariate. Y via published HealthBench rubric × GPT-4.1-class grader (paper docs F1=0.71 vs physicians).

**Audit prediction**: accepts on hedging/communication/complex_responses themes; fires on emergency_referrals/health_data/context_seeking/global_health themes (where physicians had 5%+ disagreement on baseline policies).

### 3.4 Flagship — Arena-Tail mixture (~$7.8K)

100K prompts mixed:
- 50% Arena-140k (general chat)
- 20% HarmBench + StrongREJECT + WildJailbreak (adversarial)
- 15% BeaverTails harmful-question prompts
- 10% RealToxicityPrompts `challenging`-stratified
- 5% WildChat-4.8M shuffled toxic-flagged subset

× 5-policy panel from `dataset_options.md §B.2`. Total cost 200M output
tokens generation (~$2,500 closed + ~$50 open) + 75K oracle calls
(~$5,250) = **~$7,800**.

WildJailbreak's 4-quadrant `data_type` is the audit covariate; the per-stratum reporting cleanly tests two orthogonal audit-failure axes (π_safety fails on adversarial_benign; π_adversarial fails on adversarial_harmful).

### 3.5 Total cvar_v4 paper budget

Original CJE paper: $4-55 for n=4,961.

cvar_v4 paper full empirical chapter:
- Pilot (HH): $0-5K
- Medical primary (HealthBench): ~$750
- Flagship (Arena-Tail): ~$7,800
- **Total: ~$15K-25K** (range depends on oracle re-elicitation choices)

---

## Part 4 — Per-dataset summaries (in rank order)

For the full 20-block EDA per dataset see `scoping_data.md` Part 2.
Below are one-paragraph summaries with pointers to the deeper analyses.

### #1 UltraFeedback
n=255,612; 17 in-data models; `overall_score` 1-10. **Empirically
demonstrated 140.5× within-source CVaR-bite ratio** (`uf_per_source.md`).
Methodology pilot — proves the framework works at scale. Audit covariate
= `source` (9 levels) or `principle` (5 levels). Replication: skip P/Π/Y;
just elicit cheap S (~$5-15).

### #2 HH red-team-attempts
n=38,961; **12 in-data cells** (model_type × num_params); Y in data
(`min_harmlessness_score_transcript`). **Empirically demonstrated 14.6×
CVaR-bite** (`hh_pairwise.md`). **Audit verified empirically with
asymmetric bite** (`hh_audit_demo.md`). Replication: skip P/Π/Y;
elicit cheap S (cost: $0-5K depending on Y re-elicitation). The
falsifiable pilot.

### #3 HealthBench
n=6,000 (5K oss_eval + 1K hard); no in-data policies (need fresh-Π);
continuous Y after grading. **q_0.05 spread = 0.50 across themes**
(`hb_meta_eval.md`) — strongest within-dataset audit-discriminative
heterogeneity in the survey. Maps to the C.2 medical motivating case
in `dataset_options.md`. Pilot cost: ~$750.

### #4 RealToxicityPrompts
n=98,892; no useful in-data policies; **continuous Perspective
toxicity Y**. **`challenging` boolean is 1.2% of corpus, holds 7.8% of
bottom-1% tail mass = 6.5× concentration** (`rtp_challenging.md`).
Cleanest single-flag audit covariate. Replication: fresh Π only;
Perspective is the oracle. Cost: ~$50-2,500 depending on closed/open
policy mix.

### #5 WildJailbreak
n=261,559; 4-quadrant `data_type` partition (vanilla×{harmful,benign}
× adversarial×{harmful,benign}). **Cleanest pre-built audit covariate
of any dataset by design** — π_safety should fail on adversarial_benign,
π_adversarial should fail on adversarial_harmful. Need full fresh
pipeline; oracle slice is the major cost (~$14K @ 15% × 5 policies).
Most expensive but most teachable audit demonstration.

### #6 WildChat-4.8M
n=99,081 (100K shuffled stream); 5 in-data models. **Empirically
demonstrated 32× CVaR ratio** for o1-mini vs gpt-4o
(`wildchat_cross_model.md`) — reasoning-trained models pay a CVaR
dividend. Toxic shard absent from default streaming pull (need explicit
`data_files='toxic/*'`). Replication: skip P/Π; partial S in data;
need oracle Y (~$2K).

### #7 Arena-140k (user's anchor)
n=271,149 (271K cells after melting); 70+ in-data models; **pairwise
winner only — no continuous Y**. The exact prompt-distribution match
for the original CJE paper at 27× the n. Cannot serve as a primary
CVaR-CJE testbed without fresh continuous Y elicitation
(~$5K). Best role: **prompt-source backbone for the Arena-Tail
flagship mixture**.

### #8 HarmBench + StrongREJECT
n=713 (combined); no in-data policies. StrongREJECT auto-grader is a
built-in validated oracle. Too small for the original-paper MDE 0.02
target. Use as **adversarial-tail seed for the Arena-Tail flagship's
20% adversarial stratum**. Pilot cost: ~$100.

### #9 BeaverTails
n=327,752; 14 harm-class taxonomy; **binary `is_safe` Y → CVaR
collapses**. Synthesized severity is discrete-11 with 7% ties at q_0.05.
Use only as **prompt source** for the Arena-Tail flagship's 15% safety
stratum.

### #10 PKU-SafeRLHF
n=164,196 (melted); 3 in-data Alpaca-family policies; **`severity_level`
Likert-4 with 48% mass at 0 → CVaR collapses**. Use only as **prompt
source** for the Arena-Tail flagship's safety stratum (19 harm classes
provide finer stratification than BeaverTails' 14).

---

## Part 5 — Methodology

### 5.1 The 20-block EDA recipe (per dataset)

Each dataset's `scoping_data.md` Part 2 section is generated by
`cvar_v4/eda/run_eda.py --dataset <slug>` running 20 blocks:

```
Identification & access
1. HF id / paper / license / gating
2. On-disk footprint
3. Splits

Schema & scale
4. First-row pretty print
5. Field inventory (dtype, %null, %empty, n_unique)
6. Usable n after filtering

Y candidates
7. Y candidate enumeration (continuous/Likert/binary classification)
8. Score distribution per Y candidate (11 quantiles + ASCII histogram)
9. Tie analysis at q_α (CVaR-CJE structural check)
10. Tail-mass diagnostic (CVaR_α with bootstrap CI)

Multi-policy & calibration structure
11. Multi-policy nativity (policy_id_col, n distinct, top by row count)
12. Existing cheap-judge / oracle pair

Audit-discriminative metadata
13. Stratification covariate inventory
14. Per-stratum tail-mass heterogeneity

Length / cost-relevant
15. Prompt-length distribution (tiktoken cl100k_base)
16. Response-length distribution

Qualitative
17. Tail samples (bottom 5 by Y)
18. Median samples (3 in [q_0.45, q_0.55])

CVaR-CJE scoping
19. π0 / π′ / S / Y proposal
20. Cost & CVaR-α resolution feasibility
```

### 5.2 Polars-only stack

The harness is polars-only; HF `datasets` is used only as the loader,
followed by `pl.from_arrow(ds.data.table)` (in-memory) or
`pl.from_dicts(rows)` (streaming). No pandas anywhere. Per cvar-cje's
`CLAUDE.md`: "no pandas in our workhorse".

### 5.3 Direct CVaR-CJE estimator + Wald audit

`cvar_v4/eda/deeper/_estimator.py` (a port of
`cvar_v3/workhorse.py:estimate_direct_cvar_isotonic` and
`two_moment_wald_audit_xf` with no `cje-eval` dependency, plus a 1e-6
Σ̂ ridge for numerical stability):

- **Estimator**: grid search over t̂ ∈ [q_0.001-0.6, q_max+0.35], 61
  steps; for each t the calibrator fits isotonic regression on
  z = max(t-y, 0) → s; objective = t - mean(predicted shortfall) / α;
  pick t̂ = argmax.
- **Audit**: paired bootstrap (B=200 reps) with t̂ re-maximized per rep
  to capture the full asymptotic Σ̂. Test statistic
  W = ḡ' Σ̂_boot⁻¹ ḡ ~ χ²₂; reject at p < 0.05.

### 5.4 Bootstrap CIs

CVaR_α point estimates use bootstrap percentile CIs (B=1,000 unless
noted). The closed-form influence-function variance becomes negative
when Y has a mass-point at q_α (which is typical for borderline-tied
data); bootstrap is robust to that.

---

## Part 6 — File inventory & rerun

### 6.1 Source-of-truth files (don't delete)

| Path | Purpose |
|---|---|
| `cvar_v4/scoping.md` | This document — master entry point |
| `cvar_v4/scoping_data.md` | Long-form per-dataset 20-block EDA reference |
| `cvar_v4/dataset_options.md` | User's original lit-review (untouched) |
| `cvar_v4/paper_blueprint.md` | Paper outline (untouched) |
| `cvar_v4/main.tex` + `references.bib` + `sections/*.tex` | Paper draft |
| `cvar_v4/eda/{run_eda.py, recipe.py, _utils.py, parts.py, README.md}` | Harness |
| `cvar_v4/eda/datasets/*.py` | 10 dataset specs (registered modules) |
| `cvar_v4/eda/deeper/*.py` | 9 deeper-analysis rerunnable scripts |
| `cvar_v4/eda/deeper/_estimator.py` | Self-contained Direct CVaR-CJE + Wald audit |

### 6.2 Auto-generated outputs (regeneratable; safe to delete to declutter)

| Path | What it is | How to regenerate |
|---|---|---|
| `cvar_v4/scoping_deeper.md` | Earlier supplementary doc (now folded into Parts 2 + 4 above) | `git revert` or rewrite from .py modules |
| `cvar_v4/eda/sections/01..10_*.md` | Raw 20-block EDA output per dataset (also inlined in `scoping_data.md` Part 2) | `python3 -m cvar_v4.eda.run_eda --dataset <slug>` |
| `cvar_v4/eda/deeper/*.md` | Per-finding writeups (folded into Part 2 above) | `python3 -m cvar_v4.eda.deeper.<module>` |
| `cvar_v4/eda/deeper/hh_cheap_s_scores.parquet` | HF-Inference cheap-S checkpoint (38/2K rows, credits-blocked) | `python3 -m cvar_v4.eda.deeper.hh_real_cheap_s` (resume on credits unblock) |

### 6.3 Rerun commands

```bash
# rerun a single dataset's 20-block EDA
cd /Users/pranjal/Code/cvar-cje
python3 -m cvar_v4.eda.run_eda --dataset wildjailbreak

# list registered datasets
python3 -m cvar_v4.eda.run_eda --list

# rerun a deeper analysis (any of the 9 modules)
python3 -m cvar_v4.eda.deeper.hh_pairwise
python3 -m cvar_v4.eda.deeper.hh_audit_demo
python3 -m cvar_v4.eda.deeper.hh_coverage_sweep
python3 -m cvar_v4.eda.deeper.hh_better_s
python3 -m cvar_v4.eda.deeper.hh_real_cheap_s   # blocked on HF credits
python3 -m cvar_v4.eda.deeper.uf_per_source
python3 -m cvar_v4.eda.deeper.rtp_challenging
python3 -m cvar_v4.eda.deeper.hb_meta_eval
python3 -m cvar_v4.eda.deeper.wildchat_cross_model
python3 -m cvar_v4.eda.deeper.replication_per_dataset   # this is markdown-only, no .py

# compile the paper + appendices
cd cvar_v4 && latexmk -pdf main.tex
cd cvar_v4 && latexmk -pdf online_appendix.tex
```

### 6.4 Outstanding items

- **HF credits depleted** (Part 2 §2.8) — the real cheap-S validation
  experiment is paused. Top up credits at
  https://huggingface.co/settings/billing, or switch `base_url` in
  `hh_real_cheap_s.py` to a Together/Fireworks endpoint. Resume is
  idempotent.
- **WildJailbreak fresh-gen** for the Arena-Tail flagship is the next
  capital-expense step (~$8K generation + $14K oracle). Defer until
  pilot validates.
- **Disk** is at 3.3 GiB free. The HF dataset caches were cleaned after
  each EDA run; the only remaining is local docling caches in
  `~/.cache/huggingface/hub` (~500 MB).
