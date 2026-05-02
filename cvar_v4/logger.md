# Session Log

## 2026-05-02 — Intro rewrite, method restructure, code↔paper mapping

### Goal

Rewrite the introduction so every claim is grounded in a verified
primary-source quote. Restructure the method into the four-pillar
order the user wanted (stop-loss → calibration → audits → inference).
Build a comprehensive code↔paper mapping document next to the
estimator. Add standing rules so the verification discipline persists.

### Achieved

**`cvar_v4/sections/introduction.tex` — full rewrite, four paragraphs**

- **Para 1 (problem motivation):** rebuilt around the cheap-judge
  problem with three concrete consequences (false A/B improvements,
  ranking inversion, broken CIs). Verbatim-verified cites:
  `Zheng2023Judging` (85% MT-Bench agreement; 9% vs 91%
  length-padding gap from Table 5),
  `Lee2025LLMJudgeReporting` + `Chen2026EfficientLLMJudge`
  (homogenization / over-under estimate by regime),
  `LandesbergNarayan2026CJE` (ranking inversion + 0/50 CI coverage).
  Each cite anchored once; downstream sentences flow from the anchor.
- **Para 2 (CVaR motivation):** medical / safety motivation via
  HealthBench worst-at-k (60% mean → ~40% worst-of-16); Jones 2025
  for deployment-scale rare behaviors; Rockafellar-Uryasev 2000 +
  Acerbi-Tasche 2002 for CVaR coherence in finance; Du 2023 for CVaR
  in risk-sensitive RL.
- **Para 3 (CJE recipe):** four-step plain-English recipe with
  alternatives (Rogan-Gladen-style and PPI-style) acknowledged as
  competing calibration approaches. No formulas. Variance treatment
  named (calibration noise can exceed half the total at small oracle
  slices).
- **Bridge + bullets (CVaR-CJE contribution):** three-piece structure
  in intuition-only form (no Greek, no formulas, no notation).
  Bullet 1 walks the naive plug-in failure (mean smooths variability
  → tail compressed → not consistent) before introducing the family
  of stop-loss curves. Bullet 2 names the held-out 80/20 self-audit,
  the four-cell readout, and the bootstrap-shared Wald covariance.
  Bullet 3 unpacks the bootstrap into "redo the whole pipeline inside
  each resample" with the two non-negotiable design choices
  (prompt-level resampling, threshold re-optimization).
- **Empirical results:** split into setup + results paragraphs.
  HealthBench described in detail (rubric scoring with signed point
  values; 93% of prompts have at least one criterion that *deducts*
  points). Five policies described with their roles
  (control / premium / different-prompt / adversarial). Results in
  plain prose: 60% MAE reduction, audit catches `unhelpful` at the
  worst 20%, ~4× variance reduction at small budgets, 30s laptop
  runtime.
- **Polarity bug fix:** "missed the true value in zero of 50 seeds"
  → "captured the true value in zero of 50 seeds". The CJE source
  states coverage is zero, not perfection.
- **Terminology standardization:** `policy` and `oracle` everywhere,
  matching the body. Defined `policy` parenthetically on first use
  in Para 1. Stripped stray `system`, `ground-truth`, `gold-standard`.
- **Title:** renamed to *Tail Risk Calibration for LLM-as-Judge*.
- **`\iclrfinalcopy`** added so the rendered PDF shows author names
  instead of "Anonymous authors / Paper under double-blind review".
  `\vspace{-2em}` after `\maketitle` to tighten the title-to-section
  gap.

**`cvar_v4/sections/method.tex` — restructured into four explicit
subsections**

- §2.1 Stop-loss representation of CVaR (`sec:method-stoploss`)
- §2.2 Calibration (`sec:method-calibration`)
- §2.3 Audits (`sec:method-audits`)
- §2.4 Inference: full-pipeline bootstrap (`sec:method-inference`)
- α=1 closing remark folded into §2.4
- Variance decomposition updated to match the new
  `appendix_d_bootstrap.tex`: was `Var_full ≈ Var_cal + Var_eval +
  Var_audit` (3 terms), now `Var_total ≈ Var_cal + Var_audit` (2
  terms, the eval slice folded in). Orphaned "≤ 3.5% additivity gap"
  sentence dropped — the new appendix has no 3-term decomposition or
  that gap claim. New `\cref{eq:var-total}` and
  `\cref{app:audit-holdout}` anchors added.
- Every `\cref{}` verified to resolve in the new appendix structure
  (the linter renamed/rewrote appendices during this session).

**`cvar_v4/references.bib` — three new entries**

- `Jones2025RareLLMBehaviors` (arXiv 2502.16797) — ICML 2025.
- `Allen2024TailCalibration` (arXiv 2407.03167) — Allen + Koh.
- `Du2023IteratedCVaR` (arXiv 2206.02678) — Du / Wang / Huang, ICLR
  2023.

Each entry has verbatim quote evidence at file:line in the plan file.

**`cvar_v4/papers/` — repo-local paper library expanded by six**

- `2410.02736_justice_or_prejudice/` (Ye 2024)
- `2502.16797_forecasting_rare_llm_behaviors/` (Jones 2025)
- `2407.03167_tail_calibration/` (Allen 2024)
- `2206.02678_iterated_cvar_rl/` (Du 2023)
- `cond-mat.0104295_acerbi_tasche_coherence_expected_shortfall/`
  (arXiv source for the 2002 paper)
- `JoR2000_rockafellar_uryasev_cvar_optimization/` (PDF + docling
  markdown — no LaTeX source available for this 2000 *Journal of
  Risk* paper)

**`/Users/pranjal/Code/cvar-cje/CLAUDE.md` — two new standing rules**

- `### Repo-local paper library` — check `cvar_v4/papers/` before
  downloading; grep broadly across multiple keywords; copy new arXiv
  papers in for the next session.
- `### Citations and quotes — hard evidence required` — every cite
  must come with a verbatim quote at a specific file:line in a
  primary source the agent has personally read. Subagent reports are
  leads, not evidence; re-grep before relaying. Trip-wire after a
  hallucinated Ye 2024 quote was caught and dropped this session.

**`cvar_v4/eda/deeper/MATH_NOTES.md` — comprehensive code↔paper
mapping (§5)**

New 540-line §5 ("The main derivation: paper math → code mapping")
walks every CVaR-CJE pipeline stage in execution order with three
blocks per subsection: paper math (with equation labels and
`file:line` citations), pseudocode mirroring `_estimator.py`, and the
verified code function and line range. The 14 stages: notation;
threshold-grid construction; mean calibrator; stop-loss family;
direct CVaR estimate; direct mean estimate; two-moment audit (point);
mean-transport audit (paired t-test); joint Wald; single-moment
audits; pipeline bootstrap (CVaR); pipeline bootstrap (mean);
variance decomposition; α=1 regression tests. Closing
cross-references table summarizes all 14 on one screen.

### Verification

- `latexmk -pdf -interaction=nonstopmode main.tex` from `cvar_v4/`
  produces `main.pdf` (20 pages, 441 KB).
- Zero `Citation undefined`. Zero `Reference undefined`. Cites for
  the three new bib entries (`Jones2025RareLLMBehaviors`,
  `Allen2024TailCalibration`, `Du2023IteratedCVaR`) all resolve.
- Zero emdashes in `introduction.tex` or `method.tex`.
- Title renders as *Tail Risk Calibration for LLM-as-Judge* with
  author names; "Anonymous authors / Paper under double-blind review"
  header is gone.
- For every cited paper, the verbatim quote backing the in-text claim
  was independently re-grepped from the local source before the cite
  was finalized — including the polarity-fix verification of the
  CJE 0/50 coverage quote.

### Known remaining items

- The intro's bridge paragraph cite to `Allen2024TailCalibration` is
  the only place the Allen paper is referenced in the body; the bib
  entry and the local paper copy are kept in case the cite migrates
  to a method-section discussion of formal mean-vs-tail calibration.
- The `≈4×` variance reduction claim and `60%` MAE reduction in the
  empirical-results paragraph are point estimates from a specific
  oracle-coverage / α / seed configuration; they should be rerun if
  results.tex is refreshed with new HealthBench data.
- Plan file with full decision-by-decision history, including
  verbatim quote evidence for every cite, lives at
  `/Users/pranjal/.claude/plans/help-me-fix-the-inherited-floyd.md`.

---

## 2026-05-02 - Lean appendix restructure

### Goal

Restructure the `cvar_v4` appendices so they carry proofs, theorems, and critical design-decision validation rather than repeating empirical results, pipeline setup, and stale tables. Fold the important mathematical content from `cvar_v4/eda/deeper/MATH_NOTES.md` into the active appendix set.

### Achieved

- Replaced the active appendix includes in `cvar_v4/main.tex` with a five-appendix structure:
  - `sections/appendix_a_estimands.tex`
  - `sections/appendix_b_stoploss_vs_plugin.tex`
  - `sections/appendix_c_audit.tex`
  - `sections/appendix_d_bootstrap.tex`
  - `sections/appendix_e_calibration_validation.tex`
- Removed the old active appendix route from `main.tex`:
  - `appendix_b_audit`
  - `appendix_c_bootstrap`
  - `appendix_d_alpha_one`
  - `appendix_e_forensics`
  - `appendix_f_calibrator_target`
- Created `appendix_a_estimands.tex` with the main formal estimands and identification logic:
  - Mean-CJE setup.
  - Lower-tail CVaR stop-loss form.
  - Direct CVaR-CJE identification theorem.
  - Atom-split discrete CVaR estimator.
  - `alpha=1` collapse theorem and implementation invariant.
  - Key clipping argument from `MATH_NOTES.md`: use `(t-Y)_+`, not `(Y-t)_+`, for lower-tail CVaR.
- Created `appendix_b_stoploss_vs_plugin.tex` to explain the stop-loss route versus plug-in calibration:
  - Stop-loss calibrates `E[(t-Y)_+ | S,X]`.
  - Plug-in calibration targets the CVaR of `E[Y | S,X]`, not the CVaR of `Y`.
  - Included the Jensen-gap/design-decision explanation from the deeper math notes.
  - Kept DGP validation conceptual rather than repeating long empirical results.
- Created `appendix_c_audit.tex` around the transport/refusal gate:
  - Two CVaR audit moments.
  - Two-moment transport theorem.
  - Wald audit limit.
  - Held-out audit design.
  - Clear statement that mean and CVaR audits are separate gates.
- Created `appendix_d_bootstrap.tex` for full-pipeline inference:
  - Bootstrap algorithm.
  - Re-optimizing `t_hat` inside each bootstrap replicate.
  - Estimator-variance versus audit-variance taxonomy from `MATH_NOTES.md`.
  - Mean versus CVaR inference roles.
  - Cost-equivalence definition.
- Created `appendix_e_calibration_validation.tex` for critical implementation diagnostics:
  - Cheap-score decile checks.
  - Tail-specific calibration gap.
  - Base-versus-clone sanity check.
  - Policy-wise residual transport.
  - Risky-policy mechanism check.
  - Threshold-gap diagnostic.
  - Core implementation invariants from `MATH_NOTES.md`.
- Lightly edited `sections/results.tex` to remove a broken reference to the deleted empirical appendix table.
- Adjusted `sections/macros.tex` so the `twocolproof` environment uses slightly narrower columns, reducing proof-table overfull boxes.

### Verification

- Compiled from `cvar_v4/` with:
  - `pdflatex -interaction=nonstopmode main.tex`
  - repeated after the appendix rewrite and after the proof-column width adjustment.
- Final compile status: success.
- Checked the log for hard failures and stale references:
  - no LaTeX errors,
  - no undefined references,
  - no unresolved label rerun warnings.
- Confirmed the active paper no longer references the removed appendix files or old appendix labels such as:
  - `tab:audittruth`
  - `tab:drilldown`
  - `fig:budget`
  - `fig:failuremode`
  - `tab:alpha-one`
  - `fig:app-calibrator-target`
  - `tab:app-calibrator-target-bias`

### Known Remaining Issues

- The appendix rewrite is intentionally theory/design focused and no longer carries the old empirical appendix tables.
- `sections/results.tex` still contains old HealthBench pilot numbers/model-stack text. Those should be refreshed after the new Llama-policy/OpenAI-judge data run.
- The final PDF still has some layout-only overfull/underfull box warnings in dense equations. The compile is clean with respect to correctness errors and references.
- The repo worktree already had many unrelated changes. This log records only the appendix-restructure work from this session.
