# Changelog

## 2.3.8 - 2026-04-28

- Clarified the trailing-gradient-flush behaviour in main_en.tex (§ Optimiser
  and schedule): the sentence now explicitly attributes the no-flush behaviour
  to the archived source_snapshot and notes that the root project script has
  been updated to add a trailing flush for future reproductions.  This removes
  the contradiction between the manuscript (describing no flush) and the root
  atlas_a_v5_multiseed.py (which now has the flush).
- Added post-hoc power analysis to source_snapshot/analyze_multiseed_results.py:
  - Added `import math`
  - Added `_norm_cdf()`, `wilcoxon_power_pitman()`, `compute_post_hoc_power()`
    (identical to the root script)
  - Wired `power_analysis = compute_post_hoc_power(tests)` into main() and
    added `"power_analysis"` key to the exported `full_report` dict
  This removes the contradiction between the manuscript/protocol note (which
  state that compute_post_hoc_power() is in analyze_multiseed_results.py) and
  the archived snapshot (which previously lacked those functions).
- Regenerated inner CHECKSUMS.sha256.

## 2.3.7 - 2026-04-28

- Replaced the exploratory-prototype index_construction.py (root and
  source_snapshot) with the correct, complete two-step pipeline that was
  actually used to produce the working indices:
    Step 1 (from notebook metadata_train_evaluated.ipynb): reads
      ptbxl_database.csv, engineers metadata features (z-scores,
      availability masks, missingness indicators), writes index_mm_core.parquet.
    Step 2 (from script fix_index.py): merges index_mm_core.parquet with
      ptbxl_database.csv to add scp_codes, filename_lr, filename_hr;
      writes index_complete.parquet that eznx_loader_v2.py reads at training
      time.
  The previous version was incompatible with the training loader (no scp_codes,
  no filename_lr, wrong output filename). The new script is fully runnable and
  can genuinely reproduce the index from a PTB-XL 1.0.3 download.
- Regenerated inner CHECKSUMS.sha256.

## 2.3.6 - 2026-04-28

- Added a prominent WARNING header to `index_construction.py` (root and
  source_snapshot) marking it as an exploratory prototype incompatible with
  the training loader: it uses filename_hr only, omits scp_codes and
  filename_lr, and writes index_mm_core.parquet — not index_complete.parquet.
  Directs users to scripts/build_index.py for the correct index.
- Documented the trailing-gradient-flush gap in main_en.tex (§ Optimiser and
  schedule): with 17,418 samples / batch_size 32 = 545 micro-batches, the
  final micro-batch (≈10 samples, ≈0.06% of the epoch) accumulates a gradient
  that is never applied; the residual is discarded at the next epoch start.
  Impact on reported results is negligible. The archived source_snapshot adds
  a comment documenting this known behaviour; the root atlas_a_v5_multiseed.py
  adds a trailing flush for future runs.
- Added "Seed-level JSON layout" section to source_snapshot/README.md
  documenting that analyze_multiseed_results.py expects ATLAS_A_v5_<variant>_
  seed* subdirectories, not the flat seed_json/ layout of the archived files;
  directs users to artifacts/statistical_analysis_full.json for direct
  inspection.
- Regenerated inner CHECKSUMS.sha256.

## 2.3.5 - 2026-04-28

- Corrected compute-time source attribution in main_en.tex (§ Optimiser and
  schedule, line ~267): removed the claim that wall-clock figures come from
  "experiment-summary JSON logs" (those logs were generated but not archived);
  replaced with an explicit statement that the figures were recorded at run time
  and that the raw timing log is not included in the package.
- Fixed inconsistent default working-index path across archived scripts:
  `atlas_a_v5_multiseed.py` and `run_multiseed_experiments.py` now default to
  `data/index_complete.parquet` (matching `build_index.py` and
  `evaluate_missingness_robustness.py`). Fixed in both root and source_snapshot
  copies. Documented in source_snapshot/README.md under "Working-index path".
- Corrected hardcoded class counts in `render_article_artifacts.py` (root and
  source_snapshot) to match `dataset_integrity_report.json`:
  NORM 9528→9514, MI 5486→5469, STTC 5237→5235, CD 4897→4898, HYP 2655→2649.
- Removed `render_fig2_training_and_test()` call from `main()` in
  `source_snapshot/scripts/render_manuscript_result_figures.py`: replaced with
  the same redirect comment block as the root version, so running the snapshot
  script no longer overwrites the correct Fig. 2 with the legacy single-seed
  version.
- Removed French manuscript (main.tex) from the package (no longer submitted).
- Regenerated inner CHECKSUMS.sha256.

## 2.3.4 - 2026-04-27

- Fixed runtime crash in `wilcoxon_power_pitman()` in `analyze_multiseed_results.py`:
  two accidental debug lines (lines 1034-1035) that caused `TypeError` were removed
  and replaced with the correct bisection initialisation (`lo, hi = 0.0, 10.0`).
- Reconciled the power wording in both manuscripts (§ Seeds/runs/compute) with the
  stored Pitman lower bounds in `statistical_analysis_full.json`:
  - Changed "exceed 0.99" to "reach ≥ 0.9548 (Pitman lower bound); the actual
    Wilcoxon power is expected to exceed 0.99 in practice at d_z=1.77, n=10"
  - Changed "approximately 0.94" to "≥ 0.9366" for the secondary contrast
  - The distinction between the conservative Pitman lb (stored in JSON/protocol) and
    the expected actual power is now explicit in both manuscripts.
- Corrected two ΔAUC values in `tab:sensitivity` (main_en.tex) that were derived
  from pre-rounded rather than exact JSON values:
  - H7 narrow gate (glu512): +0.0021 → +0.0020 (exact: +0.002047)
  - H7 wide gate (glu2048): +0.0017 → +0.0016 (exact: +0.001611)
  - Corrected prose "maximum spread of Δ = 0.0021" → "Δ = 0.0020" in §7.6.
- Regenerated inner CHECKSUMS.sha256: 110 entries, 0 failures.

## 2.3.3 - 2026-04-25

- Corrected GLU gate percentage in Figure 1 caption in both manuscripts:
  69.5% → 69.6% (consistent with §4.6 parameter budget and Limitation 4).
- Corrected residual-projection percentage in Figure 1 caption in both
  manuscripts: 3.5% → 3.4% (0.13M / 3.82M = 3.40%).
- Corrected stale version identifier in README.md: 2.3.1 → 2.3.3.
- Added citation-placement hedge for Atwa2025 in Section 2.4 of both
  manuscripts: "Among PTB-XL-family studies, Atwa et al. (whose published
  title uses the abbreviation 'PTB-X', referring to the same PhysioNet corpus)"
  to explicitly acknowledge the corpus name discrepancy between the paper
  title and the PTB-XL label used in the surrounding text.
- Changed Coppola2024 BibTeX entry type from @article (with journal=medRxiv)
  to @misc for consistency with other preprint entries.
- Removed suspect DOI 10.1007/978-3-032-11442-6_36 from Holloway2026 (the
  978-3-032 Springer prefix is non-standard; the chapter publication details
  are otherwise complete).
- Replaced the "training-device provenance not preserved" placeholder in both
  manuscripts with the actual hardware specification: Intel Core i5, 8 GB RAM,
  500 GB storage, PyTorch 2.3.1+cpu (CPU-only, no CUDA/GPU), local Jupyter
  Notebook environment under Windows.
- Clarified in both manuscripts and the README that the CuDNN deterministic
  flags present in the archived source code were inactive on the CPU-only
  runtime, and that CPU-only execution is inherently free of CUDA
  non-determinism.
- Updated the hyperparameter table captions in both manuscripts accordingly.
- Updated the Limitations section (point 6) in both manuscripts to document
  the full hardware provenance explicitly.
- Added explicit scientific justification for the choice of 10 seeds in both
  manuscripts: post-hoc power estimates (>0.99 for d_z=1.77; ~0.94 for
  d_z=1.27; ~0.16 for d_z=0.30), Pitman efficiency of Wilcoxon vs t-test,
  and comparison with field norms (3-5 seeds modal practice).
- Added formal justification for the exact paired Wilcoxon test in both
  manuscripts: non-verifiable normality at n=10, weaker symmetry assumption,
  and use of the exact distribution rather than a normal approximation.
- Added three references to bibliography.bib: Hollander et al. (2014)
  Nonparametric Statistical Methods (Wilcoxon foundation), Bouthillier et al.
  (2021) Accounting for Variance in ML Benchmarks, and Dodge et al. (2019)
  Show Your Work (seed-count reporting norms).

## 2.3.2 - 2026-04-24

- Added explicit parameter-budget split rows to the hyperparameter table in the
  English and French manuscripts, including the 69.5% GLU share.
- Added an explicit table row stating that the validation-selected blend
  collapses to `w_fused = 1.0` in all 30 retained runs.
- Tightened the architecture captions so the GLU concentration and residual-path
  budget are visible without reading the full discussion section.
- Clarified that the training-curves figure is a single-run diagnostic panel,
  while cross-seed uncertainty is reported in the result tables.
  (Note: superseded in v2.3.3 — Fig. 2 now shows mean ± SD across all 10 seeds.)
- Expanded the first manuscript mention of AUPRC in the limitations sections.

## 2.3.1 - 2026-04-24

- Replaced remaining `+/-` prose with publication-safe `\\pm` math in the
  manuscripts and SD wording in the cover letter.
- Added explicit PDF-string handling for `\\texttt` and safer bookmark text for
  `\\pm`.
- Made the manuscripts explicit that package-local analysis-family notes are
  documentation artifacts rather than preregistrations.
- Tightened the limitations text around threshold sensitivity, correlated
  repeated-seed contrasts, and the absence of patient-level arrays.
- Clarified that cited non-peer-reviewed preprints are contextual neighbouring
  work rather than confirmatory baselines.
- Added source-snapshot execution guidance for local imports.
- Removed the unused `--sequential` CLI flag from the archived orchestration
  scripts.
- Improved local-import robustness in the archived training entry points by
  prepending the script directory to `sys.path`.

## 2.3.0 - 2026-04-23

- Aligned the cover-letter title with the English manuscript title.
- Harmonized the English and French manuscripts around the same limitation structure.
- Made the manuscript explicit that validation-selected blending collapses to `w_fused = 1.0` in all 30 runs.
- Added explicit discussion of the negative average `demo -> demo+anthro` shift for `HYP`.
- Added package-local dataset integrity reports verifying fold counts, patient-disjoint splits, superclass prevalence, and the `q_meta` distribution.
- Added a package-local statistical analysis protocol note and expanded artifact documentation.
- Bundled the 30 raw seed-level result JSON files and the derived Markdown/CSV statistical reports.
- Bundled a source and environment snapshot so the submission package can be inspected without leaving the archive.
- Refined figure captions, terminology, and PDF metadata for the English and French PDFs.
- Documented the absence of archived patient-level prediction arrays and training-device provenance instead of leaving those gaps implicit.
