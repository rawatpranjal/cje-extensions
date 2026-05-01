# CVaR Appendix Literature Manifest

This directory records the local source files used for the CVaR appendix
rewrite. PDF papers were converted with Docling unless noted. The arXiv paper
was handled from LaTeX source, following the repo instruction not to use arXiv
PDFs for content.

## Used in the Appendix

- Rockafellar and Uryasev (2000), "Optimization of Conditional Value-at-Risk"
  - PDF: `../pdf/rockafellar_uryasev_2000_optimization_cvar.pdf`
  - Text: `../txt/rockafellar_uryasev_2000_optimization_cvar.txt`
  - Fallback text: `../txt/rockafellar_uryasev_2000_optimization_cvar_pdftotext.txt`
  - Note: Docling output was font-encoded in places, so `pdftotext` was used
    as a fallback for readable passages.
- Rockafellar and Uryasev (2002), "Conditional Value-at-Risk for General Loss
  Distributions"
  - PDF: `../pdf/rockafellar_uryasev_2002_general_loss.pdf`
  - Text: `../txt/rockafellar_uryasev_2002_general_loss.txt`
- Golodnikov, Kuzmenko, and Uryasev (2019), "CVaR Regression Based on the
  Mixed-Quantile Quadrangle"
  - PDF: `../pdf/golodnikov_kuzmenko_uryasev_2019_cvar_regression.pdf`
  - Text: `../txt/golodnikov_kuzmenko_uryasev_2019_cvar_regression.txt`
- Homburg, Weiss, Alwan, Frahm, and Gob (2021), "Evaluating Approximate Risk
  Forecasts for Count Processes"
  - PDF: `../pdf/homburg_weiss_alwan_frahm_gob_2021_count_processes.pdf`
  - Text: `../txt/homburg_weiss_alwan_frahm_gob_2021_count_processes.txt`
- Landsman and Valdez (2005), "Tail Conditional Expectations for Exponential
  Dispersion Models"
  - PDF: `../pdf/landsman_valdez_2005_tce_exponential_dispersion.pdf`
  - Text: `../txt/landsman_valdez_2005_tce_exponential_dispersion.txt`
- Luxenberg, Perez-Pineiro, Diamond, and Boyd (2025), "An Operator Splitting
  Method for Large-Scale CVaR-Constrained Quadratic Programs"
  - arXiv source archive: `../src/2504.10814_source.tar.gz`
  - Extracted source directory: `../src/2504.10814/`
  - Text from source: `../txt/jiang_boyd_2025_cvqp_arxiv2504_10814_source.txt`
  - Note: The text filename has an older shorthand name, but the BibTeX entry
    and appendix cite the authors from the source file.

## Used in the Classical Calibration Memo

- Ta, Shao, Li, and Wang (2020), "Generalized Regression Estimators with
  High-Dimensional Covariates"
  - Source URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC7313320/`
  - JATS XML from NCBI EFetch: `../src/pmc7313320_greg_high_dim.xml`
  - Text from XML: `../txt/ta_shao_li_wang_2020_greg_high_dim.txt`
  - Note: PMC direct PDF access was blocked by browser checks, so the
    text-mining XML source was converted with Pandoc.
- Liao, Spiegelman, and Carroll (2013), "Regression Calibration Is Valid When
  Properly Applied"
  - Source URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC4035111/`
  - JATS XML from NCBI EFetch: `../src/pmc4035111_regression_calibration_valid.xml`
  - Text from XML: `../txt/liao_spiegelman_carroll_2013_regression_calibration_valid.txt`
- Amorim, Tao, Lotspeich, Shaw, Lumley, and Shepherd (2021), "Two-Phase
  Sampling Designs for Data Validation in Settings with Covariate Measurement
  Error and Continuous Outcome"
  - Source URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC8715909/`
  - JATS XML from NCBI EFetch: `../src/pmc8715909_two_phase_validation_measurement_error.xml`
  - Text from XML: `../txt/amorim_tao_lotspeich_shaw_lumley_shepherd_2021_two_phase_validation.txt`
- Whittemore and Halpern (2013), "Two-stage Sampling Designs for External
  Validation of Personal Risk Models"
  - Source URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3971015/`
  - JATS XML from NCBI EFetch: `../src/pmc3971015_two_stage_risk_model_validation.xml`
  - Text from XML: `../txt/whittemore_halpern_2013_two_stage_risk_model_validation.txt`
- Yang and Ding (2025), "Two-phase rejective sampling and its asymptotic
  properties"
  - Source URL: `https://arxiv.org/src/2403.01477`
  - arXiv source archive: `../src/2403.01477_two_phase_rejective/source.tar.gz`
  - Extracted source directory: `../src/2403.01477_two_phase_rejective/`
  - Text from LaTeX source: `../txt/yang_ding_2025_two_phase_rejective_sampling_source.txt`
- Boe, Shaw, Midthune, Gustafson, Kipnis, Park, Sotres-Alvarez, and Freedman
  (2023), "Issues in Implementing Regression Calibration Analyses"
  - Source URL: `https://arxiv.org/abs/2209.12304`
  - arXiv abstract page: `../src/2209.12304_issues_regression_calibration_abs.html`
  - Text from abstract page: `../txt/boe_shaw_midthune_et_al_2023_issues_regression_calibration_abs.txt`
  - Note: `https://arxiv.org/src/2209.12304` returned a PDF rather than a
    LaTeX source archive, so the full arXiv paper was not read under the repo's
    arXiv-source rule.
- Deville and Sarndal (1992), "Calibration Estimators in Survey Sampling"
  - Source URL: `https://www.tandfonline.com/doi/abs/10.1080/01621459.1992.10475217`
  - Crossref metadata: `../src/crossref_deville_sarndal_1992_calibration.json`
  - Note: Taylor & Francis PDF access returned a 403 challenge; treated as
    metadata-only here.
- Sarndal, Swensson, and Wretman (1992), "Model Assisted Survey Sampling"
  - Source URL: `https://link.springer.com/book/9780387406206`
  - Springer metadata HTML: `../src/springer_sarndal_swensson_wretman_model_assisted.html`
  - Text from metadata page: `../txt/sarndal_swensson_wretman_1992_model_assisted_metadata.txt`
  - Note: book content was not downloaded.
- Carroll, Ruppert, Stefanski, and Crainiceanu (2006), "Measurement Error in
  Nonlinear Models"
  - Source URL: `https://doi.org/10.1201/9781420010138`
  - Crossref metadata: `../src/crossref_carroll_ruppert_stefanski_crainiceanu_measurement_error.json`
  - Note: book content was not downloaded.
- Prentice (1989), Buyse and Molenberghs (1998), Alonso et al. (2004), and
  Green, Yothers, and Sargent (2008), surrogate endpoint validation papers
  - Source URL: `https://pubmed.ncbi.nlm.nih.gov/`
  - PubMed metadata and abstracts: `../src/pubmed_surrogate_endpoint_metadata.xml`
  - Formatted XML text: `../txt/pubmed_surrogate_endpoint_metadata.txt`
  - Note: these are abstract/metadata-level reads, not full-text reads.

## Not Used

- `../pdf/liu_2024_nested_simulation_cvar_discrete_losses.pdf`
  - Converted text: `../txt/liu_2024_nested_simulation_cvar_discrete_losses.txt`
  - Reason: the downloaded file was a CV rather than the target nested
    simulation paper, so it was not cited or used in the appendix text.
