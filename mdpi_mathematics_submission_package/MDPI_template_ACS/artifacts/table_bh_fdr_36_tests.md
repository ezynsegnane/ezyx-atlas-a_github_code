# Supplementary Table S1: Full BH-FDR Confirmatory Family (36 Tests)

Exact two-sided paired Wilcoxon signed-rank tests on seed-matched variant differences.
Benjamini-Hochberg FDR control at q = 0.05 across all 36 tests simultaneously.
n = 10 paired runs. **Bold** rows survive BH-FDR correction.

| Contrast | Metric | Mean diff | p_raw | p_BH | Sig. BH | d_z |
|----------|--------|-----------|-------|------|---------|-----|
| demo - none | Macro-AUC | +0.0002 | 0.5566 | 0.6262 | no | 0.299 |
| demo - none | Macro-F1* | +0.0029 | 0.1934 | 0.3315 | no | 0.456 |
| demo - none | AUC NORM | +0.0007 | 0.0645 | 0.1934 | no | 0.622 |
| demo - none | AUC MI | +0.0008 | 0.7695 | 0.7915 | no | 0.321 |
| demo - none | AUC STTC | +0.0009 | 0.0645 | 0.1934 | no | 0.648 |
| demo - none | AUC CD | -0.0024 | 0.0195 | 0.1004 | no | -0.928 |
| demo - none | AUC HYP | +0.0010 | 0.1934 | 0.3315 | no | 0.473 |
| demo - none | F1* NORM | +0.0036 | 0.0488 | 0.1934 | no | 0.754 |
| demo - none | F1* MI | +0.0037 | 0.3750 | 0.4821 | no | 0.281 |
| demo - none | F1* STTC | +0.0025 | 0.2324 | 0.3803 | no | 0.370 |
| demo - none | F1* CD | +0.0006 | 0.6953 | 0.7585 | no | 0.072 |
| demo - none | F1* HYP | +0.0039 | 0.3750 | 0.4821 | no | 0.246 |
| **demo+anthro - demo** | **Macro-AUC** | **+0.0012** | **0.0039** | **0.0281** | **yes** | **1.275** |
| demo+anthro - demo | Macro-F1* | +0.0010 | 0.4922 | 0.5716 | no | 0.251 |
| demo+anthro - demo | AUC NORM | +0.0007 | 0.2754 | 0.4310 | no | 0.410 |
| **demo+anthro - demo** | **AUC MI** | **+0.0039** | **0.0020** | **0.0234** | **yes** | **2.173** |
| demo+anthro - demo | AUC STTC | +0.0004 | 0.1934 | 0.3315 | no | 0.351 |
| demo+anthro - demo | AUC CD | +0.0012 | 0.0840 | 0.2160 | no | 0.669 |
| demo+anthro - demo | AUC HYP | -0.0003 | 0.7695 | 0.7915 | no | -0.102 |
| demo+anthro - demo | F1* NORM | -0.0012 | 0.1602 | 0.3203 | no | -0.445 |
| demo+anthro - demo | F1* MI | +0.0066 | 0.0840 | 0.2160 | no | 0.647 |
| demo+anthro - demo | F1* STTC | +0.0025 | 0.3750 | 0.4821 | no | 0.308 |
| demo+anthro - demo | F1* CD | -0.0001 | 0.4922 | 0.5716 | no | -0.016 |
| demo+anthro - demo | F1* HYP | -0.0030 | 0.4316 | 0.5358 | no | -0.316 |
| **demo+anthro - none** | **Macro-AUC** | **+0.0014** | **0.0039** | **0.0281** | **yes** | **1.770** |
| demo+anthro - none | Macro-F1* | +0.0038 | 0.0645 | 0.1934 | no | 0.784 |
| demo+anthro - none | AUC NORM | +0.0014 | 0.0371 | 0.1670 | no | 0.846 |
| **demo+anthro - none** | **AUC MI** | **+0.0047** | **0.0020** | **0.0234** | **yes** | **1.637** |
| **demo+anthro - none** | **AUC STTC** | **+0.0013** | **0.0020** | **0.0234** | **yes** | **1.759** |
| demo+anthro - none | AUC CD | -0.0011 | 0.1055 | 0.2373 | no | -0.525 |
| demo+anthro - none | AUC HYP | +0.0008 | 0.3223 | 0.4821 | no | 0.254 |
| demo+anthro - none | F1* NORM | +0.0024 | 0.1309 | 0.2771 | no | 0.586 |
| demo+anthro - none | F1* MI | +0.0104 | 0.0195 | 0.1004 | no | 1.064 |
| demo+anthro - none | F1* STTC | +0.0050 | 0.1055 | 0.2373 | no | 0.721 |
| demo+anthro - none | F1* CD | +0.0005 | 0.3750 | 0.4821 | no | 0.062 |
| demo+anthro - none | F1* HYP | +0.0010 | 0.8457 | 0.8457 | no | 0.077 |

*Note: p_raw values bottom out at 0.0020 (minimum attainable two-sided p for n=10 exact Wilcoxon).*
*All p-values are conditional on validation-based model-selection protocol (blend weights and class thresholds selected on fold-9).*