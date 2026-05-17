# MDPI LaTeX Manuscript Package

This folder is the submission-ready MDPI package for the EZNX-ATLAS-A
manuscript. It is designed to be self-contained: the manuscript sources, final
PDF figures, package-local statistical artifacts, and the code/environment
snapshot used to verify those artifacts are bundled here.

## Package identity

- Package version: `2.3.3`
- Package date: `2026-04-25`
- English submission master: `main_en.tex`
- French companion translation: `main.tex`
- Authoritative archival snapshot: this folder

The English manuscript is the submission master. The French manuscript is kept
aligned to the same evidence base for local review and communication. For the
journal upload itself, the English manuscript should be treated as the primary
submission source.

## Main contents

- `main_en.tex` and `main_en.pdf`: English submission source and compiled PDF.
- `main.tex` and `main.pdf`: French companion source and compiled PDF.
- `bibliography.bib`: BibTeX references shared by both manuscripts.
- `figures/`: final manuscript figures actually referenced by the `.tex` files.
- `artifacts/`: statistical outputs, integrity checks, protocol notes, and
  package-local verification tables.
- `source_snapshot/`: code and environment snapshot needed to inspect the
  training/evaluation pipeline without leaving this package.
- `CHECKSUMS.sha256`: package-local file checksums for archival verification.
- `VERSION`: package version identifier.
- `CHANGELOG.md`: high-level summary of package-level revisions.

## Artifacts

The `artifacts/` directory contains:

- `statistical_analysis_full.json`: paired statistics used for the main tables.
- `statistical_analysis_report.md`: human-readable summary of the paired tests.
- `seed_level_results.csv` and `seed_level_results.md`: seed-level summaries.
- `seed_json/`: the 30 raw seed-level result JSON files.
- `missingness_eval_demo_anthro_summary.json`: missingness robustness summary.
- `missingness_eval_demo_anthro_rows.csv`: per-mask-rate robustness rows.
- `dataset_integrity_report.json` and `.md`: patient-disjoint and prevalence verification.
- `statistical_analysis_protocol.md`: documentation of the 36-test analysis family and post-hoc power calculations.
- `table_bh_fdr_36_tests.csv` and `.md`: Supplementary Table S1 — all 36 BH-FDR corrected Wilcoxon tests.
- `tripod_ai_checklist.md`: TRIPOD-AI reporting checklist (29 items, Collins et al., BMJ 2024).
- `extended_runs/`: raw JSON results for the 5 single-seed extended experiments (H5H8, H7 GLU sweep, M3 ablation).
- `README.md`: schema notes for the artifact bundle.

## Source snapshot

The `source_snapshot/` directory includes the relevant training, analysis, and
figure-generation files, together with `requirements.txt`, `environment.yml`,
`LICENSE`, `CITATION.cff`, and the root package manifest from the project
release used for this manuscript.

## Compile

Compile the English submission version from this directory with:

```sh
pdflatex -interaction=nonstopmode main_en.tex
bibtex main_en
pdflatex -interaction=nonstopmode main_en.tex
pdflatex -interaction=nonstopmode main_en.tex
```

Compile the French companion version with:

```sh
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Metadata

- Author: Ezyn SEGNANE
- ORCID: `0009-0005-0538-4335`
- Correspondence: `ezyn.segnane@univ-nkc.mr`
- Institution: Department of Mathematics and Computer Science, University of
  Nouakchott, Mauritania

## Notes

- This package intentionally avoids depending on an external branch head or an
  unversioned web path for manuscript verification.
- The original archived workspace did not preserve a usable Git commit hash.
  The package therefore uses a local version identifier (`2.3.3`) together
  with `CHECKSUMS.sha256` for archival traceability.
- All 30 training runs were executed on a CPU-only machine (Intel Core i5,
  8 GB RAM, 500 GB storage) using PyTorch 2.3.1+cpu in a local Jupyter Notebook
  environment under Windows. No GPU or CUDA device was involved.
- The archived source snapshot sets `torch.backends.cudnn.deterministic=True`
  and `torch.backends.cudnn.benchmark=False`; these flags are present in the
  code but were inactive on the CPU-only runtime. CPU-only execution is
  inherently free of CUDA non-determinism.
- Patient-level prediction arrays for all 30 runs were not archived in the
  original release; this is documented explicitly in the manuscripts and in
  the artifact notes.
- The revised manuscripts now make two package-level diagnostics explicit in
  the main text and tables: the validation-selected blend collapses to
  `w_fused = 1.0` in all 30 retained runs, and the GLU gate accounts for
  about 69.6% of the trainable parameters.
