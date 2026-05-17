# TRIPOD-AI Reporting Checklist

**Manuscript:** EZNX-ATLAS-A: Measuring the Incremental Contribution of Clinical Metadata
to 12-Lead ECG Superclass Classification on PTB-XL

**Reference:** Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated
guidance for reporting clinical prediction models that use regression or machine learning.
BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378.

**Study type:** Classification / diagnostic AI (retrospective, ECG + structured metadata)

---

## Section 1: Title and Abstract

| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 1 | Identify the study as developing, validating, or updating a prediction model, or evaluating model performance | Yes | Title; Abstract sentence 1 |
| 2 | Provide a structured summary including study design, data source, participants, sample size, predictors, outcome, performance measures, and conclusions | Yes | Abstract (200 words) |

---

## Section 2: Introduction

| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 3a | Explain the medical context (including whether diagnostic or prognostic) and rationale for developing or validating the model | Yes | §1 Introduction, §2.5–2.6 |
| 3b | Specify the objectives, including whether the study describes the development, validation, or updating of the model, or its performance evaluation | Yes | §1, Q1–Q4 research questions |

---

## Section 3: Methods

### Participants
| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 4a | Describe the study design and data source | Yes | §3.1 PTB-XL 1.0.3 |
| 4b | Specify the key dates and any differences between the source population and the study population | Partial | PTB-XL 1.0.3 used as released; collection dates described in Wagner et al. (2020) [cited]. Study uses all 21,799 records. |
| 5 | Describe the participants' eligibility criteria, and the settings and locations where data were collected | Yes | §3.1; PTB-XL official 10-fold partition; single-site (PTB Berlin) |
| 6 | Describe the outcome to be predicted, including when and how it was assessed | Yes | §3.1; SCP-ECG codes mapped to 5 superclasses via scp_statements.csv |
| 7 | Report the number of participants and outcomes | Yes | §3.1, Table 1 (17,418 / 2,183 / 2,198 per fold split) |

### Predictors
| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 8 | Describe all predictors used in the model | Yes | §3.2–3.3; 12-lead ECG (12×1000 samples) + age, sex, height, weight, BMI |
| 9 | For categorical predictors, report number and distribution of categories | Yes | sex: binary {0,1}; age, height, weight, BMI: continuous, z-normalised |

### Sample size
| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 10 | Explain how the study size was determined | Yes | §5.4: PTB-XL canonical split; 10 seeds chosen as principled CPU-budget compromise; post-hoc power analysis reported |

### Missing data
| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 11 | Describe how missing data were handled | Yes | §3.3: median imputation + explicit mask bits; inference-time masking stress test §6.4 |

### Model development
| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 12a | Describe how predictors were handled in the model | Yes | §4.2–4.4; dual MLP encoder; quality score q_meta |
| 12b | Describe any data pre-processing | Yes | §3.4 Preprocessing |
| 12c | Describe the type of model, all hyperparameters, and method used to build the model | Yes | §4, §5.1–5.3; Table 2 (hyperparameters) |
| 13a | Describe how the model was internally validated (e.g., cross-validation, bootstrapping) | Yes | §5.4–5.6; hold-out fold-10 test; 10 random seeds |
| 13b | For models using ML, describe all tuning, feature selection, or other model-building techniques | Yes | §5.3 blend-weight grid; §5.3 threshold tuning on fold-9 |
| 14 | Describe any adjustments made to predictions | Yes | §5.3 inference blending (collapsed to w*=1.0); §4.6 meta-correction term |

### Performance measures
| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 15 | Specify all measures used to assess model performance and, if relevant, to compare models | Yes | Macro-AUC, macro-F1*, per-class AUC; AUPRC, Brier, DeLong CI, ECE (seed 2029, §6.6) |
| 16 | Describe any model comparison, including selection of the final model | Yes | §5.4; three-variant paired ablation; §5.5 Wilcoxon + BH-FDR |

---

## Section 4: Results

| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 17a | Report the flow of participants through the study | Yes | §3.1 fold counts; Table 1 |
| 17b | Report the characteristics of the participants in each dataset | Partial | Table 1 reports superclass prevalences; patient-level demographics not individually tabulated (PTB-XL public cohort described in Wagner et al.) |
| 18 | Report performance of the model, including uncertainty | Yes | §6.1: macro-AUC mean ± SD, 95% bootstrap CIs; Table 2–3 |
| 19 | Report any model updating results | N/A | Single release; no model updating performed |

---

## Section 5: Discussion

| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 20 | Summarise the main findings of the study | Yes | §7.1–7.7; §8 Conclusions |
| 21 | Compare study results with other relevant studies and models | Yes | §7.4; Tables 5a–5b comparability audit |
| 22 | Discuss the clinical use and implications of the model | Yes | §7.2 MI/STTC; §7.3 HYP paradox; §7.4 comparability |
| 23 | Discuss the limitations of the study | Yes | §7.9 Limitations (6 items) |

---

## Section 6: Other information

| Item | TRIPOD-AI requirement | Reported? | Location |
|------|-----------------------|-----------|----------|
| 24 | Provide information on funding and the role of the funders | Yes | Funding statement: no external funding |
| 25 | State the availability of the study protocol, code, and data | Yes | Data availability statement; GitHub URL; supplementary package |

---

## Summary

| Category | Items | Fully reported | Partially reported | N/A |
|----------|-------|---------------|-------------------|-----|
| Title / Abstract | 2 | 2 | 0 | 0 |
| Introduction | 2 | 2 | 0 | 0 |
| Methods | 15 | 13 | 2 | 0 |
| Results | 4 | 3 | 1 | 1 |
| Discussion | 4 | 4 | 0 | 0 |
| Other | 2 | 2 | 0 | 0 |
| **Total** | **29** | **26** | **3** | **1** |

**Partially reported items:**

- **4b** (key dates): PTB-XL collection dates are described in the primary data paper (Wagner et al. 2020, cited) but not restated in this manuscript.
- **17b** (participant characteristics): Superclass prevalences are reported (Table 1); individual patient demographics are not tabulated because PTB-XL is a public de-identified dataset described in detail in Wagner et al. (2020).
- **Extended metrics (Item 15)**: AUPRC, Brier score, DeLong CIs, and ECE are reported for a single representative run (seed 2029, demo+anthro, fold-10); cross-seed estimates are unavailable because patient-level probability arrays were not preserved in the original 30-run release (documented in Limitation 3).
