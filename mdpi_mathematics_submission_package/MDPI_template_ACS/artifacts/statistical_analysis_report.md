# Analyse Statistique Multi-Graines - EZNX_ATLAS_A

*Généré le 2026-04-23 14:24:06*

## Analysis notes

- Bootstrap confidence intervals are computed on seed-level values, not by patient-level resampling.
- Pairwise inferential tests are two-sided paired Wilcoxon signed-rank tests on seed-matched variant differences.
- Because blend weights and decision thresholds are selected on fold 9 and then fixed on fold 10, p-values should be interpreted conditionally on this validation-based model-selection protocol.
- Cohen's d_z is reported as a descriptive paired effect size; Hedges-corrected g_z is exported alongside it for small-sample sensitivity.
- Consecutive integer seeds are used as transparent run identifiers only; each run re-seeds Python, NumPy, and PyTorch independently.

## Tableau 1. Performances sur le jeu de test (fold 10) - Validation multi-graines

| Méthode | Macro AUC | Macro F1 | n |
|---------|-----------|----------|---|
| ECG seul | 0.9274 ± 0.0008 | 0.7420 ± 0.0052 | 10 |
| ECG + demo | 0.9277 ± 0.0008 | 0.7449 ± 0.0033 | 10 |
| ECG + complet | 0.9289 ± 0.0009* | 0.7459 ± 0.0039 | 10 |

*p < 0.05; **p < 0.01 after BH-FDR correction (paired Wilcoxon vs ECG seul)

## Tableau 2. Performances par classe - Validation multi-graines

### AUC par classe

| Classe | ECG seul | ECG + demo | ECG + complet |
|--------|----------|------------|---------------|
| NORM | 0.9482 ± 0.0012 | 0.9489 ± 0.0008 | 0.9496 ± 0.0015 |
| MI | 0.9260 ± 0.0023 | 0.9268 ± 0.0018 | 0.9307 ± 0.0022* |
| STTC | 0.9353 ± 0.0012 | 0.9362 ± 0.0007 | 0.9366 ± 0.0011* |
| CD | 0.9187 ± 0.0025 | 0.9164 ± 0.0026 | 0.9176 ± 0.0019 |
| HYP | 0.9091 ± 0.0024 | 0.9101 ± 0.0016 | 0.9098 ± 0.0025 |

### F1 par classe

| Classe | ECG seul | ECG + demo | ECG + complet |
|--------|----------|------------|---------------|
| NORM | 0.8591 ± 0.0023 | 0.8627 ± 0.0034 | 0.8615 ± 0.0032 |
| MI | 0.7304 ± 0.0118 | 0.7342 ± 0.0085 | 0.7408 ± 0.0113 |
| STTC | 0.7669 ± 0.0084 | 0.7694 ± 0.0059 | 0.7719 ± 0.0081 |
| CD | 0.7470 ± 0.0056 | 0.7477 ± 0.0066 | 0.7475 ± 0.0063 |
| HYP | 0.6066 ± 0.0133 | 0.6105 ± 0.0064 | 0.6075 ± 0.0057 |

*p < 0.05; **p < 0.01 after BH-FDR correction (paired Wilcoxon vs ECG seul)

## Comparaisons appariees principales

| Contraste | Metrique | Diff. moyenne | IC bootstrap 95% (diff) | p raw | p BH-FDR | d_z | g_z (corrige) | n | +/-/0 |
|-----------|----------|---------------|--------------------------|-------|----------|-----|----------------|---|-------|
| none vs demo | Macro AUC | +0.0002 | [-0.0002, 0.0007] | 0.5566 | 0.6262 | 0.30 | 0.27 | 10 | 5/5/0 |
| none vs demo | Macro F1 | +0.0029 | [-0.0010, 0.0063] | 0.1934 | 0.3315 | 0.46 | 0.42 | 10 | 7/3/0 |
| none vs demo+anthro | Macro AUC | +0.0014 | [0.0009, 0.0019] | 0.0039 | 0.0281 | 1.77 | 1.62 | 10 | 9/1/0 |
| none vs demo+anthro | Macro F1 | +0.0038 | [0.0009, 0.0067] | 0.0645 | 0.1934 | 0.78 | 0.72 | 10 | 8/2/0 |
| demo vs demo+anthro | Macro AUC | +0.0012 | [0.0007, 0.0018] | 0.0039 | 0.0281 | 1.27 | 1.17 | 10 | 9/1/0 |
| demo vs demo+anthro | Macro F1 | +0.0010 | [-0.0013, 0.0032] | 0.4922 | 0.5716 | 0.25 | 0.23 | 10 | 5/5/0 |

Notes: the bootstrap CIs above are computed by resampling seed-level paired differences with replacement.
+/-/0 reports the number of positive, negative, and exactly zero paired seed differences.

## Resume des gains (ECG + complet vs ECG seul)

| Metrique | Baseline | Complet | Gain absolu | Gain relatif | p raw | p BH-FDR |
|----------|----------|---------|-------------|--------------|-------|----------|
| Macro AUC | 0.9274 | 0.9289 | +0.0014 | +0.15% | 0.0039 | 0.0281* |
| Macro F1 | 0.7420 | 0.7459 | +0.0038 | +0.52% | 0.0645 | 0.1934 |
| F1 NORM | 0.8591 | 0.8615 | +0.0024 | +0.28% | 0.1309 | 0.2771 |
| F1 MI | 0.7304 | 0.7408 | +0.0104 | +1.42% | 0.0195 | 0.1004 |
| F1 STTC | 0.7669 | 0.7719 | +0.0050 | +0.65% | 0.1055 | 0.2373 |
| F1 CD | 0.7470 | 0.7475 | +0.0005 | +0.06% | 0.3750 | 0.4821 |
| F1 HYP | 0.6066 | 0.6075 | +0.0010 | +0.16% | 0.8457 | 0.8457 |

*p < 0.05; **p < 0.01 after BH-FDR correction
