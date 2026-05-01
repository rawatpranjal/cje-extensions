# Replicating the original CJE pipeline on each of our 10 datasets

_Source: `arxiv_source/sections/setup_dbp.tex` and `experiments.tex` (the
ICLR 2026 CJE paper, Landesberg et al., arXiv:2512.11150)._

The user's question: for each dataset we surveyed, how hard would it be to
*reconstruct* the original CJE paper's experimental setup, and how relevant
would the result be for the CVaR extension (does CVaR have bite over mean,
does the audit have bite, does the domain itself call for CVaR)?

## §0. The original CJE paper's pipeline — what we have to match

The paper's headline result (99% pairwise ranking accuracy at $4 cost,
14× cheaper than pure oracle) was built on a 5-step pipeline:

| step | resource | what they used | cost / scale |
|---|---|---|---|
| **(P)** Prompts | first-turn English Chatbot Arena prompts | 5,000 random; 4,961 after TF filtering | one-time scrape |
| **(Π)** Policy panel | 5 controlled LLM configs × all prompts | base (Llama-3.3-70B) / clone (same+seed) / premium (Llama-3.1-405B) / parallel_universe_prompt / unhelpful | ~25K generated responses |
| **(S)** Cheap judge | one LLM scores every response 0-100 | GPT-4.1-nano @ T=0 | ~$3.50 for 25K responses |
| **(Y)** Oracle | stronger LLM scores every response 0-100 | GPT-5 @ T=1.0 | ~$55 for 25K responses, ~16× S |
| **(A)** Audit + estimator | per-policy mean residual test + isotonic + 2-stage cov + bootstrap | response_length is the covariate; Bonferroni α=0.0125 across 4 testable policies (excluding unhelpful) | ~10-20% overhead |

Five design choices that matter for replication:
- **Continuous Y in [0, 1]** elicited fresh (not in-data labels)
- **Five policies engineered** to span (baseline, near-clone, scale-up,
  prompt-tweak, adversarial-bad)
- **Adversarial policy is by design** the audit-positive case (`unhelpful`
  produced mean residual -0.31, p<0.001 → REFUSE-LEVEL)
- **Length-as-covariate** (the +cov variant beats the base variant)
- **Sample-size grid** across {500, 1k, 2k, 3k, 5k} prompts × {5%, 10%,
  25%, 50%, 100%} oracle fractions × 50 seeds

For CVaR-CJE we additionally need: continuous Y *with no mass-point at
q_α* (else CVaR collapses), and ideally a covariate the per-stratum
audit can fire on.

## Per-dataset replication assessment

Each dataset is scored on six dimensions, 1–5 stars each. Total
replication score is summed; ranking at the bottom.

- **P (Prompts ready?)**: do prompts match Arena's "first-turn English chat" mold?
  More natural-conversation = higher score; small adversarial seeds = lower.
- **Π (Policies skippable?)**: is multi-policy in the data so we skip step 2?
  ★★★★★ = ≥5 distinct in-data policies; ★ = need full fresh-gen.
- **S/Y (Judge stack ready?)**: do natural cheap-S and oracle-Y candidates exist?
  ★★★★★ = both shipped; ★ = both need fresh elicitation.
- **C (CVaR bite)**: does CVaR_α detect tail differences mean would miss?
  ★★★★★ = empirically demonstrated with ratio ≥10×; ★ = Y is binary
  (CVaR collapses to mean).
- **A (Audit bite)**: is there a covariate the audit can fire on?
  ★★★★★ = built-in audit covariate with empirical heterogeneity;
  ★ = no useful covariate in data.
- **D (Domain demands CVaR?)**: distinct from C/A — this is about the
  dataset's *content*, not its statistical structure. Is the dataset
  mapped to a deployment context where worst-case behavior carries real
  consequence (fatal liability, regulatory action, brand damage)?
  ★★★★★ = life-safety stakes (medical, suicide, wrongful death — the
  C.1/C.2 cases in `dataset_options.md`); ★★★★ = harmful-content /
  regulatory liability (toxicity, jailbreak, children safety, content
  moderation); ★★★ = operational/brand/UX concerns; ★★ = pure quality
  benchmark with no deployment-context tail consequence; ★ = methodology
  demonstration only.

Sections below are in **descending order of total stars** (best
candidate first). Tie-broken by D (higher domain demand wins). The
summary table at the bottom is the same order.

---

### Rank #1 — UltraFeedback  (total: 25/30)

**P (Prompts) — ★★★★★** 64K instructions across 9 sources
(sharegpt, evol_instruct, ultrachat, flan_v2_*, false_qa, truthful_qa).
Source-stratified, multi-domain — strictly broader than Arena's
first-turn English chat.

**Π (Policies) — ★★★★★** 17 in-data models with ~16K rows each:
gpt-3.5-turbo, gpt-4, vicuna-33b, llama-2-{7,13,70}b-chat, wizardlm-13b/70b,
ultralm-13b/65b, alpaca-7b, falcon-40b, mpt-30b-chat, starchat, bard,
pythia-12b, plus more. Maps to multiple `premium` / `clone` / `parallel_universe`
roles directly; the worst (alpaca-7b, mean 5.43) makes a natural `unhelpful`.

**S/Y (Judge stack) — ★★★★★** Y in data: `overall_score` is GPT-4-Turbo's
0-10 aggregate; `fine-grained_score` is the dimension-mean. S would
still need fresh elicitation (Prometheus-2-7B with same 4-dim rubric is
the natural choice). The pair structure mirrors the original paper's
S/Y stack.

**C (CVaR bite) — ★★★★★ EMPIRICAL with the strongest within-source ratio
of any dataset.** `ultrachat × (llama-2-70b-chat, mpt-30b-chat)`: means
4 thousandths apart on 1-10 scale (statistically tied at any reasonable
n), CVaR_0.05 1.4 points apart, **140.5× ratio**. 39 such within-source
pairs. See `uf_per_source.md`.

**A (Audit bite) — ★★★** `source` (9 levels) and `principle`
(helpfulness/honesty/instruction_following/truthfulness/harmlessness/
verbalized_calibration) provide stratification. Per-stratum q_0.05 spread
across `principle` is small (~1 unit on 1-10 scale) — audit will be
discriminative on principle but not as cleanly as HH's model_type axis.

**D (Domain demand) — ★★** UltraFeedback is a *training-data benchmark*
for chat-quality alignment: Y is itself a quality rating, not a measure
of real-world consequence. The dataset doesn't map to a specific
deployment context where the tail produces fatal/regulatory/brand harm.
A "really bad rating" in UltraFeedback-style eval is a methodological
finding about the policy, not a deployment incident. **CVaR-bite here is
real but the dataset's value for the cvar_v4 paper is as a methodology
proof-of-concept, not a domain motivator.**

**Replication ease**: ★★★★★ (skips P, Π, Y; just elicit cheap S).
A pilot at original-paper scale (5K prompts × 5 policies) is 5K × 5 = 25K
cells, costing ~$5-15 for fresh S elicitation. **Cheaper than the original.**

---

### Rank #2 — Anthropic HH red-team-attempts  (total: 25/30)

**P (Prompts) — ★★★** Adversarial-only; 38,961 transcripts of red-team prompts
attempting to elicit harmful responses from Anthropic's pre-Claude models.
Not Arena-style natural chat. Conversational format; first-turn extractable.

**Π (Policies) — ★★★★★** 12 in-data cells (`model_type` × `num_params`):
plain LM, context distillation, rejection sampling, RLHF × {2.7B, 13B, 52B}.
The 4 model_type values map cleanly onto the original paper's role taxonomy:
plain LM ≈ `unhelpful` (audit-positive); context distillation ≈ `parallel_universe_prompt`;
RLHF ≈ `base`; rejection sampling ≈ `premium`-ish.

**S/Y (Judge stack) — ★★★★** Y in data: `min_harmlessness_score_transcript`
(Anthropic's preference-model score, validated against humans in Bai et al.
2022). Treats this directly as oracle. S would still need fresh elicitation
— **WildGuard-7B refusal-detection score** is the natural pick (response-side,
harm-aligned, free-at-inference). Cost ratio is roughly the inverse of
original: in-data Y is "free" but the cheap S costs a few cents per call.

**C (CVaR bite) — ★★★★★ EMPIRICAL.** Headline pair (`rlhf | 13B` vs
`rejection sampling | 2.7B`) has 95% mean CIs overlapping ([+1.40, +1.46]
vs [+1.45, +1.54]) and 95% CVaR_0.05 CIs disjoint ([-0.30, -0.06] vs
[-1.40, -0.84]) — **14.6× ratio** of CVaR-gap to mean-gap. See
`hh_pairwise.md`. CVaR-CJE recovers a clean ranking where mean-CJE
correctly reports a tie.

**A (Audit bite) — ★★★★ EMPIRICAL but ASYMMETRIC.** With logger=`rlhf | 52B`,
audit fires on 5/8 OOD targets (context-distillation, plain-lm cells) and
correctly accepts on within-family cells. With logger=`context distillation | 52B`
the audit is silent on every cell despite Direct CVaR errors of 4-5 units
— "**pick a logger whose Y dominates the targets**" rule. See `hh_audit_demo.md`.

**D (Domain demand) — ★★★★** Adversarial transcripts targeting harmlessness
— this is the literal red-teaming corpus Anthropic used to harden Claude
against jailbreaks. The tail is "model produced racist insults / harm
instructions under adversarial prompt", which is the exact deployment
concern moderation teams worry about. Not life-safety (★★★★★ tier
reserved for medical/suicide), but solidly in the FTC/Sec-6(b) regulatory
attention surface for chatbot operators.

**Replication ease**: ★★★★★ (skips P, Π, Y from original pipeline).
Estimated cost: $0–5K for the *full pilot* if we accept in-data Y; $5K
oracle re-elicitation if we want a stronger Y (matching original paper's
GPT-5 oracle).

---

### Rank #3 — HealthBench (after grading)  (total: 23/30)

**P (Prompts) — ★★★★** 6,000 multi-turn clinical conversations (5K oss_eval +
1K hard). Domain-specific (medical) but diverse across 7 themes (emergency
referrals, hedging, communication, complex responses, context seeking,
global health, health data). Multi-turn structure matches CJE Arena's
first-turn restriction by truncating to first user turn.

**Π (Policies) — ★** None in data — must generate fresh under our 5-policy
panel. Mirrors original pipeline cost exactly: 6K prompts × 5 policies = 30K
responses.

**S/Y (Judge stack) — ★★★★** Y elicited via GPT-4.1-class grader using
HealthBench's published rubric (paper documents F1=0.71 vs physicians).
S = Prometheus-2-7B with same rubric (open-weight, ~free).

**C (CVaR bite) — ★★★★ STRONG STRUCTURAL ARGUMENT, partially empirically
verified.** OpenAI's own paper documents o3 worst-at-16 = 40% vs mean 60%
(factor of 1.5) and HealthBench Hard: GPT-5=46% vs GPT-4o=0% (extreme tail
shift). On meta_eval physician aggregations the per-theme CVaR_0.05 ranges
0.0 (emergency_referrals, health-data, context_seeking, global_health) to
0.50 (communication, complex_responses) — see `hb_meta_eval.md`. Cross-
policy tail differences will manifest cleanly along this 7-theme axis.

**A (Audit bite) — ★★★★★ STRONGEST per-stratum heterogeneity in any dataset:
q_0.05 spread across themes = 0.50** (max 0.50, min 0.00). The audit
on a held-out theme will fire when a policy systematically mis-handles
emergency_referrals while looking fine on hedging — exactly the
medical-domain CVaR-CJE motivation.

**D (Domain demand) — ★★★★★ HIGHEST.** Medical advice with explicit
emergency_referrals theme = **life-safety stakes by design**. OpenAI built
the benchmark around worst-case reliability ("worst-at-k" is the headline
metric). Maps directly to the C.2 motivating case in `dataset_options.md`:
sycophancy at 100% compliance with illogical drug-equivalence requests
(Nature npj Digital Medicine 2025), missed sepsis recognition, FDA-relevant
clinical evaluation. Tail = patient harm. **This is the textbook CVaR-CJE
domain — the cvar_v4 paper's most defensible high-stakes empirical claim.**

**Replication ease**: ★★★★ (mirrors original pipeline directly:
fresh Π, fresh S+Y, then audit). Cost matches original paper at ~$750
for the pilot (HealthBench is half Arena's prompt count but more
expensive per-grade due to multi-turn rubric scoring).

---

### Rank #4 — RealToxicityPrompts  (total: 22/30)

**P (Prompts) — ★★★** 99,442 sentence-prefix prompts mined from
OpenWebText. Not conversational; designed to elicit continuations. Each
prompt is a partial sentence (mean ~30 tokens). Different in shape from
Arena chat but plenty of natural-distribution coverage.

**Π (Policies) — ★** GPT-2-large continuations are weak; modern frontier
models would dominate trivially. Need fresh gen at original-paper scale.

**S/Y (Judge stack) — ★★★★★** S = Detoxify-large or Perspective API
direct (continuous, free, well-validated). Y = Perspective API directly OR
GPT-5 toxicity grader. **The dataset's Y stack is BETTER than the original
paper's** because Perspective is a battle-tested oracle, not a single LLM
judge.

**C (CVaR bite) — ★★★★ STRUCTURAL.** Lower tail = most-toxic continuations
(Perspective scores in [0, 1] with a heavy upper tail = heavy lower tail
in 1-toxicity). Different policies will produce dramatically different
tail behaviors (frontier refuses, base completes, jailbroken amplifies).
Empirical verification awaits fresh-gen experiment.

**A (Audit bite) — ★★★★★ STRONGEST WITHIN-DATASET BUILT-IN COVARIATE.**
The `challenging` boolean is the cleanest single-flag audit covariate of
any dataset surveyed: 1.2% of corpus, 7.8% of bottom-1% tail mass — **6.5×
concentration**. q_0.05 conditional on challenging=True (0.035) vs
challenging=False (0.084) — 2.4× gap that any honest CVaR-CJE pipeline
must surface or refuse. See `rtp_challenging.md`.

**D (Domain demand) — ★★★★** Toxicity in deployed text generation maps
to the C.4 "content moderation" motivating case: Tay (Mar 2016: 96K
antisemitic tweets in 16 hours), Lee Lin Chin (rage), and the regulatory
exposure under EU DSA / UK Online Safety Act. Not life-safety, but
solidly in the FTC + state-AG regulatory attention surface for any
deployed text-generation system. The `challenging` subset in particular
is the curated audit-positive prompt-distribution that real platforms
worry about.

**Replication ease**: ★★★ (need fresh Π and fresh Y). 100K prompts × 5
policies × 200 out-tok ≈ 100M out-tok ≈ $50 (open-weight) to $2,500
(closed). Cheaper than original at scale.

---

### Rank #5 — WildJailbreak  (total: 21/30)

**P (Prompts) — ★★★★** 261,559 prompts across 4 quadrants
(50K vanilla_harmful + 50K vanilla_benign + 83K adversarial_harmful +
79K adversarial_benign). Synthetic but well-distributed; adversarial
quadrants mirror real jailbreak tactics from in-the-wild sources.

**Π (Policies) — ★** None in data — `completion` field is GPT-3.5-class
refusal/compliance demonstration. Need fresh gen at original-paper scale.

**S/Y (Judge stack) — ★★★★** S = WildGuard-7B (perfect fit — same
AllenAI safety pipeline). Y = GPT-5-thinking grader using paper's
refusal-quality + capability-of-harm rubric. Both ~1-policy-of-grading
work.

**C (CVaR bite) — ★★★★ STRUCTURAL.** By construction, adversarial_harmful
prompts ARE the lower tail. Different policies (π_safety vs π_adversarial)
should differ dramatically on this stratum; means may be similar across
benign quadrants. Awaits fresh-gen verification.

**A (Audit bite) — ★★★★★ THE CLEANEST AUDIT COVARIATE OF ANY DATASET BY
DESIGN.** The 4-quadrant `data_type` partition lets us test **two
orthogonal audit-failure axes**: a π_safety policy that over-refuses fails
on `adversarial_benign`; a π_adversarial policy that under-refuses fails
on `adversarial_harmful`. No other dataset gives us this kind of
pre-built audit-discriminativeness. See §07 of `scoping_data.md`.

**D (Domain demand) — ★★★★** Adversarial safety pairs by design — this is
the curated dataset for jailbreak hardening that AllenAI's safety pipeline
uses for production safety RLHF. Tail = jailbroken response under
adversarial prompt. Maps to the C.4 content-moderation motivating case
plus its own "deployed model under adversarial user" deployment surface.
Not life-safety, but a clean regulatory-and-brand liability case.

**Replication ease**: ★★★ (need fresh Π, S, Y). Estimated $8K generation
+ $14K oracle (15% slice × 5 policies). Most expensive replication
candidate but produces the strongest audit demonstration.

---

### Rank #6 — WildChat-4.8M (shuffled stream)  (total: 21/30)

**P (Prompts) — ★★★★★** 4.74M naturalistic user-LLM conversations,
ChatGPT/o1-style. Real-world distribution including 1.5M toxic.
Strictly larger and more diverse than Arena. The shuffled 100K sample
covers 70 languages × 191 countries with even multi-model representation.

**Π (Policies) — ★★★** 5 in-data models (gpt-3.5-turbo-0301, gpt-4-0314,
gpt-4o-2024-08-06, gpt-4o-mini, o1-mini-2024-09-12). Fewer than HH's 12 or
UltraFeedback's 17, but covers the reasoning vs chat axis cleanly.

**S/Y (Judge stack) — ★★★★** S in data: per-turn Detoxify scores
(`max_detoxify_toxicity`). Y would need fresh elicitation (GPT-5-thinking
with refusal+safety rubric). The dataset's existing Detoxify scores are
themselves a usable cheap S — fairly unique among datasets we surveyed.

**C (CVaR bite) — ★★★ EMPIRICAL.** o1-mini vs gpt-4o: Δmean = 0.0003,
ΔCVaR_0.05 = 0.008, **32× ratio**. Reasoning models pay a CVaR dividend
that mean evaluation misses. Smaller absolute gap than HH/UltraFeedback
because Detoxify scores are bounded near 0 for benign chat — but the
within-dataset cross-model contrast is real. See `wildchat_cross_model.md`.

**A (Audit bite) — ★★** Limited — `toxic` is unipopulated in our
non-toxic-shard sample, `redacted` has only 0.6% positive rate, and
`language`/`country` are high-cardinality. Full corpus has 1.5M toxic
conversations but partitioned into a separate shard the streaming loader
doesn't reach by default. Per-language audit possible but per-stratum
heterogeneity not yet measured.

**D (Domain demand) — ★★★** Mixed-stakes naturalistic data. Some
conversations carry real consequence (mental-health questions in
ChatGPT-style chat — overlapping with the C.1 companion-chatbot tail),
some are benign chit-chat. The 1.5M-toxic subset is a real audit-positive
sample if we can pull it. Aggregate domain demand is moderate — the
dataset is a *probe of real distribution* rather than a curated
high-stakes context.

**Replication ease**: ★★★★ (skips P, Π; partial S in data; just need
oracle Y). Cost: ~$2K oracle for 100K-sample × 5 policies. The $0
generation cost is a real advantage.

---

### Rank #7 — Arena-140k (user's anchor)  (total: 21/30)

**P (Prompts) — ★★★★★ THE EXACT MATCH for the original CJE paper's
prompt distribution.** 135,634 user-conversation prompts from Chatbot
Arena. 27× the n=4,961 the original paper used. Multi-turn but easily
truncated to first turn.

**Π (Policies) — ★★★★★** 70+ models in the in-data battles. Massive
multi-policy. But access via melting (one row per side) gives 271K
(model, prompt, response) cells.

**S/Y (Judge stack) — ★★** No continuous Y. Native label is pairwise
winner. CVaR of a difference ≠ difference of CVaRs — pairwise label is
fundamentally incompatible with per-policy CVaR_α as a scalar value.
**Need fresh elicitation of both S and Y.** S = GPT-4o-mini + 4-criterion
rubric; Y = multi-judge consensus of {Opus 4.7, GPT-5}.

**C (CVaR bite) — ⏳ UNTESTED.** Strong prior that the heterogeneity in
70+ models will produce same-mean-different-tail pairs (see UltraFeedback's
within-source 140× ratio as a structural analog), but cannot verify
without fresh continuous Y.

**A (Audit bite) — ★★★** `is_code`, `is_creative_writing`, `is_math`,
`if_score`, `language` provide multiple stratification axes. Per-stratum
tail-mass not yet measured (all current Y candidates are degenerate).

**D (Domain demand) — ★★** Chatbot Arena is a *quality-ranking
benchmark* — its tail is "a chatbot answer some users ranked low".
Per-interaction stakes are low. The dataset's role in the original CJE
paper was as a multi-policy benchmark, not as a deployment-context
high-stakes domain. For the cvar_v4 paper it is the natural backbone
for the Arena-Tail flagship pool, but on its own the Arena prompt
distribution does not motivate worst-case evaluation — that's why the
original paper's `unhelpful` policy had to be deliberately bad rather
than discovered in the wild.

**Replication ease**: ★★★★ (skips P, Π; needs both S and Y fresh).
Total cost: ~$5K (judge + oracle) for a 5-policy subset of the 70+ models.
The user's "anchor" claim — **this dataset is the natural cvar_v4-paper
analog of the original CJE Arena slice, just bigger.**

---

### Rank #8 — HarmBench + StrongREJECT  (total: 16/30)

**P (Prompts) — ★★** 713 prompts (400 HarmBench + 313 StrongREJECT).
Adversarial-only; tiny. Cannot replicate original's n=5K.

**Π (Policies) — ★** None in data — fresh gen needed.

**S/Y (Judge stack) — ★★★★** S = WildGuard-7B; Y = StrongREJECT
auto-grader (continuous 0-1, validated). Both are domain-purpose-built.

**C (CVaR bite) — ⏳ UNTESTED, expected.** 713 rows × 5 policies = 3,565
cells. Tiny n means CVaR_0.01 will not be resolvable; CVaR_0.05 maybe.

**A (Audit bite) — ★★** `dataset_source` (HarmBench vs StrongREJECT) and
`category` (13 levels) provide stratification, but small n limits
audit-test power.

**D (Domain demand) — ★★★★** Same as WildJailbreak — by-design adversarial
safety with the same regulatory-relevance footprint. The Center for AI
Safety ships HarmBench specifically as a worst-case red-team benchmark;
StrongREJECT's grader is built around refusal-vs-capability tradeoffs.
Tail = jailbreak success on a curated harmful-intent prompt.

**Replication ease**: ★★★ Cheap to run (smallest pilot ever, ~$100), but
under-powered for the original paper's MDE ≈ 0.02 target at n≈5K. Useful
as an **adversarial-tail seed for the flagship Arena-Tail mixture**, not
as a primary replication target.

---

### Rank #9 — BeaverTails  (total: 15/30)

**P (Prompts) — ★★★** 333,963 prompts in 14 harm-class taxonomy. Useful
as a *stratified* prompt source but heavily safety-oriented. Less natural
chat than Arena.

**Π (Policies) — ★** Response field is a mix of LLaMA-{7,13}B + Alpaca,
not cleanly tagged per-row. Effectively single-source for our purposes;
need fresh gen.

**S/Y (Judge stack) — ★★** Y is binary (`is_safe`); CVaR collapses.
Synthesized `safety_score` ∈ [-12, 1] is discrete-11. Need fresh continuous
Y elicitation.

**C (CVaR bite) — ★** Y too discrete for primary CVaR use (7.15% ties at
q_0.05 — borderline structural fail). Possible after fresh-Y elicitation
but not from in-data signals.

**A (Audit bite) — ★★★** 14 harm categories provide stratification
(useful but not yet shown to be tail-discriminative — synthetic Y was too
coarse to see).

**D (Domain demand) — ★★★★** Same domain as HH/WildJailbreak — production
safety RLHF source from PKU. Tail = harmful response on a sensitive
prompt across 14 explicit harm classes (animal abuse, child abuse,
self-harm, terrorism, etc.). Strong regulatory liability case.

**Replication ease**: ★★ (need fresh Π and Y; only S has a natural
candidate via WildGuard).

---

### Rank #10 — PKU-SafeRLHF  (total: 15/30)

**P (Prompts) — ★★★** 73,907 prompts × 2 responses (164K cells after
melting). Safety-oriented prompts; 19 harm-class taxonomy.

**Π (Policies) — ★★★** 3 in-data response sources (Alpaca-7B, Alpaca2-7B,
Alpaca3-8B). Multi-policy native but limited diversity (all Alpaca-family).

**S/Y (Judge stack) — ★★** Y in data is `severity_level` (Likert-4 with
48% mass at 0). Structurally fails CVaR-CJE check. Need fresh continuous
Y; cheap S could be WildGuard-7B.

**C (CVaR bite) — ★** Severity too coarse (4 levels, mass at 0).

**A (Audit bite) — ★★** 19 harm categories provide stratification but
limited by Y discreteness.

**D (Domain demand) — ★★★★** 19-class harm taxonomy with explicit
severity levels — the dataset is a production-ready safety-RLHF source
covering harm classes like Endangering National Security, Human
Trafficking, Physical Harm, etc. Strong regulatory liability per-class.

**Replication ease**: ★★ (skips P, Π; need fresh S and Y).

---

## Summary ranking — replication ease × CVaR-CJE relevance × domain demand

Ranking by total stars across the six dimensions (P + Π + S/Y + C + A + D).

| Rank | Dataset | P | Π | S/Y | C | A | D | Total | Effort to match original | CVaR > mean? | Audit bite? | Domain demands CVaR? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | UltraFeedback | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★ | ★★ | **25/30** | trivial — only need cheap S | ✅ EMPIRICAL 140.5× | ⏳ predicted (per principle) | ★★ benchmark only |
| 2 | HH red-team-attempts | ★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | **25/30** | trivial — Y in data | ✅ EMPIRICAL 14.6× | ⚠️ asymmetric in logger | ★★★★ adversarial safety |
| 3 | HealthBench | ★★★★ | ★ | ★★★★ | ★★★★ | ★★★★★ | ★★★★★ | **23/30** | mirror original pipeline exactly | ⏳ external (OpenAI worst-at-k) | ⏳ predicted (q_0.05 spread = 0.50) | ★★★★★ life-safety / medical |
| 4 | RealToxicityPrompts | ★★★ | ★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★★ | **22/30** | medium — fresh Π only | ⏳ predicted (Perspective heavy tail) | ⏳ predicted (`challenging` 6.5×) | ★★★★ content moderation |
| 5 | WildJailbreak | ★★★★ | ★ | ★★★★ | ★★★★ | ★★★★★ | ★★★★ | **21/30** | full pipeline; cleanest 4-quadrant audit | ⏳ structural | ⏳ predicted (4-quadrant by design) | ★★★★ adversarial safety |
| 6 | WildChat-4.8M | ★★★★★ | ★★★ | ★★★★ | ★★★ | ★★ | ★★★ | **21/30** | medium — partial S in data, need Y | ✅ EMPIRICAL 32× | ⏳ predicted (per language) | ★★★ naturalistic mixed-stakes |
| 7 | Arena-140k | ★★★★★ | ★★★★★ | ★★ | ⏳ | ★★★ | ★★ | **21/30** | medium — fresh S + Y | ⏳ untested | ⏳ predicted (is_code, language) | ★★ benchmark only |
| 8 | HarmBench + StrongREJECT | ★★ | ★ | ★★★★ | ⏳ | ★★ | ★★★★ | **16/30** | small full pipeline, n too small for MDE 0.02 | ⏳ untested | ⏳ predicted (category, source) | ★★★★ adversarial safety |
| 9 | BeaverTails | ★★★ | ★ | ★★ | ★ | ★★★ | ★★★★ | **15/30** | full pipeline + need to fix discrete Y | ❌ degenerate (binary Y) | ⏳ predicted (14 harm classes) | ★★★★ safety RLHF source |
| 10 | PKU-SafeRLHF | ★★★ | ★★★ | ★★ | ★ | ★★ | ★★★★ | **15/30** | partial — Y discrete | ❌ degenerate (Likert-4 mass at 0) | ⏳ predicted (19 harm classes) | ★★★★ safety RLHF source |

### Reading the ranking

The new D column reshuffles the middle of the table in two important ways:

**HealthBench rises from 4th to 3rd** because it's the only dataset
mapped to **life-safety stakes (★★★★★)**. Even though its replication
ease score (★) is the lowest of any candidate (no in-data multi-policy
panel), the domain weight is decisive: this is the cvar_v4 paper's most
defensible high-stakes empirical demonstration. The OpenAI HealthBench
paper itself frames the metric around worst-case reliability — we're not
inventing the motivation, we're inheriting it.

**UltraFeedback and Arena-140k drop on the domain weight axis.** Both
are dataset-of-models-rated-by-LLMs benchmarks with no specific
deployment context where the tail is operationally consequential. They
remain at the top of the *replication-ease* ranking (we have the data,
we have policies, we have Y), so they're still the right pilots —
**but the cvar_v4 paper's narrative arc has to do them as methodology
demonstrations and use HealthBench / RealToxicityPrompts /
WildJailbreak / safety-RLHF datasets for the high-stakes story**.

**The rank-5/6/7 three-way tie at 21/30 has clear narrative ordering
when D is the tiebreaker**: WildJailbreak (D=4, by-design adversarial
safety) beats WildChat (D=3, mixed naturalistic) beats Arena-140k (D=2,
benchmark only). For audit demonstrations specifically, WildJailbreak
moves up to rank-3-ish because it's the only candidate with both
strong audit-bite and strong domain demand.

### Tier interpretation with domain-demand integrated

**Tier A — Pilot candidates (≥23/30; replication trivial AND domain real)**:
- HH red-team-attempts (25): empirical CVaR-bite + adversarial-safety domain
- HealthBench (23): life-safety domain + audit-discriminative themes

UltraFeedback (25/30 by total) **belongs in Tier A as a methodology pilot**
but is a benchmark-only domain — it should be presented as the
"the framework works at scale" sanity check, not as the high-stakes
empirical claim.

**Tier B — Domain-relevant complements (21–22/30)**:
- RealToxicityPrompts (22): toxicity moderation regulatory case
- WildJailbreak (21): cleanest pre-built audit covariate by design
- WildChat-4.8M (21): naturalistic real-world distribution

Arena-140k (21/30 by total) **drops out of Tier B on domain weight** —
it's better used as the *prompt source* for the Arena-Tail flagship
mixture rather than as a standalone high-stakes case.

**Tier C — Adversarial seeds and discrete-Y foils**:
- HarmBench + StrongREJECT (16): adversarial-tail seed for flagship pool
- BeaverTails (15): safety RLHF source; Y too discrete for primary
- PKU-SafeRLHF (15): same as above

### One-paragraph recommendation (revised after domain weighting)

For the **cvar_v4 paper's empirical chapter**, run **UltraFeedback as
the first-day methodology pilot** (Y in data, 17 in-data policies,
demonstrated 140.5× within-source CVaR-bite — proves the framework works
at scale). Then **HH red-team-attempts** as the second-day adversarial-
safety check (Y in data, 12 in-data policies, demonstrated 14.6×
CVaR-bite, audit verified — proves the framework works on a real-domain
dataset). Use **HealthBench** as the **flagship high-stakes
demonstration** (mirrors original paper's pipeline exactly at ~$750;
life-safety motivation; q_0.05 spread of 0.50 across themes is the
strongest audit-discriminative signal in the survey). Pair HealthBench
with the **Arena-Tail mixture flagship** (~$7.8K, includes
WildJailbreak's 4-quadrant audit covariate) to get both the
domain-grounded headline and the audit-discriminativeness teach. The
original CJE paper spent ~$4-55 on n=4,961; the cvar_v4 paper's full
empirical chapter should run between ~$15K and ~$25K total depending on
oracle re-elicitation choices, with HealthBench providing the most
defensible operational narrative.
