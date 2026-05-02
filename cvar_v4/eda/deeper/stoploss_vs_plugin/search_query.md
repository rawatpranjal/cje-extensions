# Search query for related-paper retrieval

Paste the block under "Query (paste this)" into a literature-search AI (Elicit, Consensus, Scite, Gemini Deep Research, etc.). The "Targets" section enumerates what we want returned. The "Anchor references" section lists known-relevant works to seed citation chains.

---

## Query (paste this)

> I am comparing two estimators for the **lower-tail Conditional Value at Risk** (`CVaR_α`, also called Expected Shortfall, Average Value at Risk, or Superquantile) of a target real-valued random variable `Y ∈ [0, 1]` under a target policy / target distribution. I have access to a cheap surrogate score `S` correlated with `Y` on a labeled calibration sample.
>
> **Estimator A: Stop-loss / shortfall regression (Rockafellar–Uryasev calibrator).** For each candidate threshold `t`, fit a regression `h_t(S) ≈ E[(t − Y)_+ | S]` of the lower-shortfall on the surrogate. Estimate
> ```
> CVaR_α  =  max_t  [ t − α^{-1} · E_target[ h_t(S) ] ]
> ```
> The functional that enters the RU objective is what is regressed on.
>
> **Estimator B: Plug-in via conditional mean.** Fit one regression `m̂(S) ≈ E[Y | S]`, then either (B1) take the empirical α-quantile / α-tail mean of `{m̂(S_eval,i)}_i` over target points, or (B2) compute the RU dual directly on `m̂`:
> ```
> CVaR_α  =  max_t  [ t − α^{-1} · mean_eval( (t − m̂(S))_+ ) ].
> ```
>
> By Jensen's inequality on the convex map `(t − ·)_+`, plug-in B is upward-biased on `CVaR_α` when `Var(Y | S) > 0`; the bias does not vanish with sample size. I have empirically verified this in simulation (bias of B grows monotonically with conditional residual SD; bias of A is within Monte-Carlo noise across an 11-point σ grid).
>
> **What I want from you. Find me papers that:**
>
> 1. **Bias of plug-in CVaR via conditional mean.** Prove or empirically demonstrate that estimating `CVaR_α` by computing the tail of `m̂(X) ≈ E[Y|X]` is biased when `Var(Y|X) > 0`, with the Jensen-gap argument or equivalent. Both theoretical (M-estimation, asymptotics) and simulation-based references welcome.
>
> 2. **Direct functional regression for risk measures.** Papers that establish the principle "regress on the loss whose minimizer is the target functional, not on `Y`." Keywords: *elicitability*, *consistent scoring functions*, *Osband's principle*, *strictly consistent loss*, *quantile regression* (Koenker–Bassett pinball loss), *expectile regression* (Newey–Powell), *CAViaR* (Engle–Manganelli), *ES regression* (Dimitriadis–Bayer, Patton–Ziegel–Chen, Fissler–Ziegel joint elicitability of `(VaR, ES)`).
>
> 3. **Influence function / semiparametric efficient estimation of CVaR / Expected Shortfall.** Papers giving the influence function of `CVaR_α` and using it for efficient (one-step / TMLE / DML) estimation under partial supervision or high-dimensional nuisance functions. Keywords: *Hong (2009) sensitivity*, *Trindade et al expected shortfall asymptotics*, *Belloni–Chernozhukov–Fernández-Val high-dim quantile*, *Chernozhukov et al double / debiased ML for CVaR*, *Kallus–Uehara doubly robust off-policy evaluation*.
>
> 4. **Off-policy evaluation (OPE) for risk-sensitive / CVaR objectives.** Papers that estimate target-policy `CVaR_α(Y)` (not just mean reward) under behavior–target distribution mismatch. Keywords: *risk-sensitive OPE*, *CVaR policy evaluation*, *Chow–Tamar–Mannor risk-constrained MDPs*, *Kallus–Uehara CVaR OPE*, *Huang et al off-policy CVaR estimation*, *Prashanth & Ghavamzadeh distributional RL CVaR*.
>
> 5. **Calibrated surrogate / prediction-powered evaluation of LLMs and AI systems.** Papers that use a cheap (LLM-judge or other) score, calibrated against a small expensive label slice, to evaluate a target policy. Keywords: *Causal Judge Evaluation (CJE)*, *prediction-powered inference (PPI / PPI++)*, *active inference with imputation*, *autoraters with calibration*, *G-Eval*, *CalibraEval*, *LLM-as-judge bias and calibration*. The paper I am working from is arXiv 2512.11150 (CJE); find papers that cite it on tail / risk evaluation, or that propose alternatives.
>
> 6. **Empirical comparisons (simulation or applied) of plug-in vs direct shortfall regression.** Especially in financial risk (operational loss, market VaR/ES), insurance (reserving), or evaluation-of-prediction-models contexts. I want the actual bias / RMSE comparison numbers.
>
> 7. **Joint elicitability theory for `(VaR, ES)`.** The result that ES is not by itself elicitable but the pair `(VaR_α, ES_α)` is jointly elicitable, with consistent scoring functions of a specific form. Authors: *Fissler & Ziegel (2016)*, *Acerbi & Székely (2014)*, *Nolde & Ziegel (2017)*, *Dimitriadis & Bayer (2019)*.
>
> **For each paper, please return:**
>
> - Title, all authors, year, venue (or arXiv ID)
> - 1–2 sentence summary
> - Whether it explicitly addresses (A) vs (B), only one, or only the underlying theory
> - Any quantitative result (a bias formula, an asymptotic rate, a simulation number) that is directly relevant
>
> Prioritize papers that quantify the bias of plug-in vs direct shortfall regression, or that justify the principle on theoretical grounds (elicitability / consistent loss / Bregman representation). Statistical-learning, financial risk, OPE/causal-inference, and LLM-evaluation literatures are all in scope.

---

## Targets (what to find)

1. Papers that *prove* or *quantify* the upward bias of plug-in CVaR estimation via `m̂(X) ≈ E[Y|X]`.
2. Papers stating the elicitability theorem for `(VaR, ES)` and consequences for direct shortfall regression.
3. Papers that build calibrated CVaR estimators for off-policy evaluation or LLM evaluation, especially with surrogate scores.
4. Empirical comparisons (simulation tables, applied financial / insurance studies) of plug-in vs direct shortfall regression.
5. Influence-function / DML / TMLE treatments of CVaR / ES estimation that give explicit efficient bounds.

## Anchor references (already known to be relevant; use as citation seeds)

- Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of conditional value-at-risk." *Journal of Risk* 2(3): 21–42.
- Acerbi, C. & Tasche, D. (2002). "On the coherence of expected shortfall." *Journal of Banking & Finance* 26(7).
- Koenker, R. & Bassett, G. (1978). "Regression quantiles." *Econometrica* 46(1).
- Newey, W. K. & Powell, J. L. (1987). "Asymmetric least squares estimation and testing." *Econometrica* 55(4).
- Engle, R. F. & Manganelli, S. (2004). "CAViaR: Conditional autoregressive value at risk by regression quantiles." *J. Business & Econ. Stat.* 22(4).
- Hong, L. J. (2009). "Estimating quantile sensitivities." *Operations Research* 57(1).
- Gneiting, T. (2011). "Making and evaluating point forecasts." *JASA* 106(494). (Elicitability framework.)
- Fissler, T. & Ziegel, J. F. (2016). "Higher order elicitability and Osband's principle." *Annals of Statistics* 44(4). (`(VaR, ES)` jointly elicitable.)
- Dimitriadis, T. & Bayer, S. (2019). "A joint quantile and expected shortfall regression framework." *Electronic J. of Statistics* 13.
- Patton, A. J., Ziegel, J. F. & Chen, R. (2019). "Dynamic semiparametric models for expected shortfall (and value-at-risk)." *J. Econometrics* 211(2).
- Acerbi, C. & Székely, B. (2014). "Backtesting expected shortfall." *Risk* 27.
- Nolde, N. & Ziegel, J. F. (2017). "Elicitability and backtesting: Perspectives for banking regulation." *Annals of Applied Statistics* 11(4).
- Belloni, A., Chernozhukov, V. & Fernández-Val, I. (2019). "Conditional quantile processes based on series or many regressors." *J. Econometrics*.
- Chernozhukov, V. et al. (2018). "Double / debiased machine learning for treatment and structural parameters." *Econometrics J.* 21(1).
- Kallus, N. & Uehara, M. (2020). "Doubly robust off-policy value and gradient estimation for deterministic policies." *NeurIPS*.
- Kallus, N. & Uehara, M. (2022). "Efficient evaluation of natural stochastic policies in offline reinforcement learning." *Biometrika*.
- Huang, A., Leqi, L., Lipton, Z. & Azizzadenesheli, K. (2021). "Off-policy risk assessment in contextual bandits." *NeurIPS*.
- Chow, Y., Ghavamzadeh, M., Janson, L. & Pavone, M. (2017). "Risk-constrained reinforcement learning with percentile risk criteria." *JMLR* 18(167).
- Tamar, A., Glassner, Y. & Mannor, S. (2015). "Optimizing the CVaR via sampling." *AAAI*.
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. & Zrnic, T. (2023). "Prediction-powered inference." *Science* 382.
- Zheng, L. et al. (2023). "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena." (LLM-judge calibration baseline.)
- Causal Judge Evaluation paper, arXiv 2512.11150 (the focal paper for this comparison).

## Suggested search syntaxes

For Google Scholar / Semantic Scholar:
- `"expected shortfall" "regression" "elicitability"`
- `"CVaR" "plug-in" bias`
- `"shortfall regression" Jensen`
- `"prediction-powered" OR "causal judge" "CVaR" OR "tail"`
- `"off-policy" "CVaR" "doubly robust"`
- `"joint elicitability" "value at risk" "expected shortfall"`

For arXiv:
- `cs.LG: CVaR off-policy evaluation surrogate`
- `stat.ME: expected shortfall regression elicitability`
- `q-fin.RM: shortfall regression`

## What this query is NOT asking for

- Generic introductions to CVaR or Expected Shortfall.
- Pure financial-portfolio optimization (`min CVaR(portfolio)`) without a regression / surrogate angle.
- Distributionally robust optimization (DRO) papers unless they explicitly build the estimator from a surrogate.
- LLM-evaluation papers that only compute *mean* judge agreement; we want tail / risk-sensitive evaluation.
