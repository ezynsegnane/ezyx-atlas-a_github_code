# Scientific Reports — Pre-Submission Checklist

## TRIPOD+AI 2024 Items addressed by the 170-run campaign

| # | Item | Status | Where |
|---|------|--------|-------|
| 1 | Title — model type | ✓ | Manuscript title |
| 2 | Abstract — participants, predictors, outcomes, model, validation | ✓ | Abstract |
| 3 | Background — clinical problem motivation | ✓ | §1 Introduction |
| 4 | Objectives — stated before data collection | ✓ | §1 (pre-specified) |
| 5 | Study design | ✓ | §2 |
| 6 | Data source — PTB-XL v1.0.3, PhysioNet | ✓ | §2.1 |
| 7 | Participants — inclusion/exclusion | ✓ | §2.1 (all 21 799) |
| 8 | Outcome — 5-class multi-hot (NORM/MI/STTC/CD/HYP) | ✓ | §2.2 |
| 9 | Predictors — 12-lead ECG + structured demographics | ✓ | §2.2 |
| 10 | Sample size — 17 111 train / 2 193 val / 2 495 test | ✓ | §2.3 |
| 11 | Missing data handling — imputation + masking | ✓ | §3.1, missingness table |
| 12 | Statistical methods — Wilcoxon signed-rank, exact, n=20 | ✓ | §3.4, stats JSON |
| 13 | Development — architecture description | ✓ | §3 |
| 14 | Model performance — macro-AUC, F1, Brier, ECE | ✓ | Tables 2–5 |
| 15 | Calibration — Brier score + ECE (10-bin) | ✓ | calibration_report.json |
| 16 | Subgroup analysis — sex, age, anthropometric completeness | ✓ | subgroup_report.json |
| 17 | Uncertainty — SD over 20 seeds reported | ✓ | All tables |
| 18 | Comparison — 3-variant ablation | ✓ | Table 2 |
| 19 | Internal validation — hold-out fold 10 (PTB-XL standard) | ✓ | §2.3 |
| 20 | External validation | ✗ | Not available (single institution — L19) |
| 21 | Limitations — MCAR scope, single institution, single dataset | ✓ | §5 Limitations |
| 22 | Interpretation — clinical relevance | ✓ | §5 Discussion |
| 23 | Generalisability statement | ✓ | §5 |
| 24 | Open code | ✓ | GitHub (post-acceptance) |
| 25 | Open data | ✓ | PTB-XL on PhysioNet (DOI) |
| 26 | AI transparency — model card | ✓ | hardware_provenance.md |
| 27 | Reproducibility — seeds, deterministic flags, hardware spec | ✓ | hardware_provenance.md |

## PROBAST+AI Items

| Domain | Item | Status |
|--------|------|--------|
| Participants | Source population defined | ✓ |
| Predictors | Complete predictor list | ✓ |
| Outcome | Clear definition | ✓ |
| Analysis | Appropriate sample size | ✓ (17 111 train) |
| Analysis | No data leakage | ✓ (fold 10 never seen during training) |
| Analysis | No optimistic reporting | ✓ (20-seed sweep, mean ± SD) |
| Analysis | Calibration assessed | ✓ (Brier + ECE) |
| AI-specific | Uncertainty quantification | ✓ (cross-seed SD) |
| AI-specific | Fairness assessment | ✓ (sex/age subgroups) |

## Nature Portfolio Reporting Summary

| Section | Required | Done |
|---------|----------|------|
| Data availability | Yes | ✓ (PTB-XL PhysioNet DOI) |
| Code availability | Yes | ✓ (GitHub, CC-BY 4.0) |
| Statistics | Yes | ✓ (Wilcoxon, effect sizes) |
| Ethics | Yes | ✓ (PTB-XL approved dataset, retrospective) |

## Remaining gaps (cannot be addressed with single-institution data)

- **L19** External cohort validation (different hospital/scanner)
- **L20** Multi-institution generalisability study
- **L5**  K-fold cross-validation (requires complete replication ×10)
- **L10** Concat-linear baseline (excluded: introduces architectural contradictions)

These gaps are disclosed in §5 Limitations of the manuscript.
