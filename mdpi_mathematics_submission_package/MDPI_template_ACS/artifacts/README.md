# Artifact Bundle

This directory contains the package-local numerical sources used to verify the
tables, figures, and statements reported in the manuscript.

## File map

- `statistical_analysis_full.json`
  - Master paired-statistics export.
  - Top-level keys are `statistics`, `pairwise_tests`, `seed_level_rows`, and
    `config`.
- `statistical_analysis_report.md`
  - Human-readable narrative summary of the paired tests.
- `table_results_latex.tex`
  - Derived LaTeX table fragment exported from the paired-statistics run.
- `seed_level_results.csv`
  - One row per run (`3 variants x 10 seeds = 30 rows`).
  - Includes aggregate metrics, per-class metrics, and selected thresholds.
- `seed_level_results.md`
  - Markdown rendering of the seed-level summary table.
- `seed_json/`
  - The 30 raw run-level result JSON files copied from the archived run folders.
- `missingness_eval_demo_anthro_summary.json`
  - Summary of the inference-time anthropometric masking study.
- `missingness_eval_demo_anthro_rows.csv`
  - Row-level export for the masking study across mask rates.
- `dataset_integrity_report.json` and `dataset_integrity_report.md`
  - Verification of fold counts, patient-disjoint splits, superclass prevalence,
    and `q_meta` distribution from the working PTB-XL index.
- `statistical_analysis_protocol.md`
  - Documentation of the executed analysis family, including the 36-test
    confirmatory BH-FDR family and post-hoc power calculations.
- `table_bh_fdr_36_tests.csv` and `table_bh_fdr_36_tests.md`
  - Explicit supplementary table (Table S1) of all 36 Wilcoxon tests with
    raw p-values, BH-adjusted p-values, mean differences, and Cohen d_z.
    Cited as Supplementary Table S1 in the manuscript.
- `tripod_ai_checklist.md`
  - TRIPOD-AI reporting checklist (Collins et al., BMJ 2024).
    29 items: 26 fully reported, 3 partially, 1 N/A.
- `extended_runs/`
  - Raw JSON result files from the 5 single-seed extended experiments
    run after the main 30-run release (reviewer requests H5, H7, H8, M3):
    - `results_ext_H5H8_seed2029_glu1152_lauc0.08.json` — extended metrics
      (AUPRC, Brier, DeLong CI, ECE) for the reference seed 2029 run
    - `results_ext_H7_glu512_seed2026_lauc0.08.json` — GLU width 512 sweep
    - `results_ext_H7_glu1152_seed2026_lauc0.08.json` — GLU width 1152 (ref)
    - `results_ext_H7_glu2048_seed2026_lauc0.08.json` — GLU width 2048 sweep
    - `results_ext_M3_seed2026_glu1152_lauc0.00.json` — LAUC ablation (λ=0)
  - These are single-seed exploratory results; they carry no confirmatory
    statistical weight. Source data for Table~\ref{tab:extended} and
    Table~\ref{tab:sensitivity} in the manuscript.

## Important interpretation notes

- Bootstrap intervals in this package are seed-level, not patient-level.
- The paired tests quantify stability across repeated random seeds on a fixed
  dataset split; they do not create independent patient cohorts.
- The archived paired statistics also show that the validation-selected blend
  collapses to `w_fused = 1.0` in all 30 retained runs; this is discussed
  explicitly in the revised manuscripts.
- Patient-level probability arrays for all 30 runs were not archived in the
  original release, so package-local verification is limited to the archived
  seed-level summaries and their derived reports.
- The `data_root` field in `missingness_eval_demo_anthro_summary.json` shows
  the local machine path used during evaluation. This path is not required for
  reproduction; the PTB-XL v1.0.3 dataset should be downloaded from PhysioNet
  (doi:10.13026/kfzx-aw45) and the local path set via the `EZNX_DATA_REAL`
  environment variable before running the evaluation scripts.
- All evaluation in this artifact bundle was performed on CPU only (device: cpu
  field in the missingness summary JSON), consistent with the CPU-only training
  environment documented in the manuscripts.
