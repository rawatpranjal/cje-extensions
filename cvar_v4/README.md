# CVaR-CJE v4

This directory is for the arXiv-first CVaR-CJE paper.

## Files

- `paper_blueprint.md` -- writing plan, scope, what to avoid, what to explore.
- `main.tex` -- compact ML-style paper scaffold, intended to stay near 10 main-text pages.
- `sections/` -- main paper and online appendix sections.
- `references.bib` -- v4 bibliography.
- `dataset_options.md` -- longer dataset research note.
- `papers/` -- downloaded arXiv source bundles for papers used to shape the v4 scope.

## Compile

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The paper uses the local ICLR style files in `../arxiv_source/`.

