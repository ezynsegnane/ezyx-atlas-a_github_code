# EZNX-ATLAS-A v5 — Pre-declared Analysis Plan

**Version:** 1.0  
**Status:** FINAL — committed before any training run  
**Proof of pre-declaration:** Git commit SHA of this file + SHA-256 recorded in every `results_*.json`

---

## 1. Study objective

Evaluate whether adding structured patient metadata (demographics and anthropometrics) to
a 12-lead ECG classifier improves diagnostic performance on the PTB-XL benchmark.

**Primary comparison:** macro-AUC of variant `demo+anthro` vs. variant `none` (ECG-only),
test fold 10, 20 independent random seeds.

---

## 2. Dataset and split

- Dataset: PTB-XL v1.0.3 (Wagner et al., 2020), 21,799 records, 100 Hz
- Label set DS5: NORM, MI, STTC, CD, HYP (multi-hot)
- Standard fold structure:
  - Training: folds 1–8 (ConcatDataset, ~17,440 records)
  - Validation: fold 9 (~2,180 records)
  - Test: fold 10 (~2,180 records) — **primary evaluation fold, never used for any design decision**

---

## 3. Primary endpoint (confirmatory)

- **Metric:** macro-AUC (mean of per-class AUC, skipping classes with fewer than 2 classes in test set)
- **Test fold:** 10 (standard PTB-XL split)
- **Comparison:** `demo+anthro` (n=20 seeds) vs. `none` (n=20 seeds)
- **Test:** Wilcoxon signed-rank, paired by seed, two-sided, α=0.05
  - `method='exact'` if n≤25 and no zero differences (scipy.stats.wilcoxon)
  - `method='approx'` with `continuity_correction=True` if ties or zero differences present
  - If all differences are zero (theoretically impossible): p=1.0, report as trivial
  - For Group F (n=4 folds): no statistical test; bootstrap CI only
- **Multiple comparison correction:** Benjamini-Hochberg FDR across the 3 pairwise
  comparisons of Group A (none vs demo, none vs demo+anthro, demo vs demo+anthro)

---

## 4. Fixed design decisions (immutable, pre-declared)

These decisions cannot change after this file is committed.

1. **w_fused = 1.0** — no blending search, no w optimization on fold 9.
   Final prediction: `sigmoid(logits_fused)` always.

2. **Checkpoint selection = val macro-AUC only** — no tie-breaking on delta_meta or any
   other metric. Best model = epoch with highest validation macro-AUC.

3. **Loss coupling (pre-declared constraint):**
   `fused_w = max(0.0, 0.60 − lauc_w)` — varying `lauc_w` simultaneously changes `fused_w`.
   Group C explores the loss landscape under this constraint and is labeled **exploratory**.

4. **pos_weights: dynamic, computed per run** — `neg_j / pos_j` per class, clipped to [0.5, 30.0],
   computed from training folds only (never from val or test).

5. **Determinism stack (all runs):**
   - `CUBLAS_WORKSPACE_CONFIG=:16:8` set before `import torch`
   - `use_deterministic_algorithms(True)`
   - `cudnn.deterministic=True`, `cudnn.benchmark=False`
   - `matmul.allow_tf32=False`, `cudnn.allow_tf32=False`
   - `num_workers=0` in all DataLoaders

---

## 5. Group taxonomy (270 descriptors, 250 unique GPU runs)

### Group A — Primary confirmatory (60 descriptors, 60 unique GPU runs)

- Variants: `none`, `demo`, `demo+anthro`
- Seeds: 2024–2043 (20 per variant)
- test_fold=10, val_fold=9
- Run naming: `ATLAS_A_v5_{variant}_seed{seed}`
- Role: **CONFIRMATORY** — primary claims are drawn from Group A only

### Group B — Fusion capacity sensitivity (40 descriptors, 30 unique GPU runs)

- Variant: `demo+anthro`; meta_hid ∈ {32, 64, 128, 256}; seeds 2024–2033 (10 per value)
- Note: meta_hid=128 = Group A demo+anthro seeds 2024–2033 (10 runs shared, not re-trained)
- **Architectural note (pre-declared):** `meta_hid` controls both (a) the fusion projection
  dimension (130 → meta_hid) and (b) the gate hidden dimension. These two roles are
  architecturally entangled in the current implementation. Group B therefore measures
  sensitivity to fusion+gate capacity jointly, not gate width alone.
- Run naming: `ATLAS_A_v5_demo+anthro_metaH{meta_hid}_seed{seed}`
- Role: **EXPLORATORY**

### Group C — LAUC-BCE balance sensitivity (50 descriptors, 40 unique GPU runs)

- Variant: `demo+anthro`; lauc_w ∈ {0.0, 0.04, 0.08, 0.12, 0.16}; seeds 2024–2033
- Note: lauc_w=0.08 = Group A demo+anthro seeds 2024–2033 (10 runs shared, not re-trained)
- **Coupling note (pre-declared):** `fused_w = max(0.0, 0.60 − lauc_w)`. Both weights change
  simultaneously. Group C explores the joint loss landscape and is not a clean single-axis sweep.
- Run naming: `ATLAS_A_v5_demo+anthro_lauc{lauc_w:g}_seed{seed}`
- Role: **EXPLORATORY**

### Group D — Augmentation sensitivity (20 descriptors, 20 unique GPU runs)

- Variant: `demo+anthro`, no_aug=True (collate_fn_val during training); seeds 2024–2043
- Run naming: `ATLAS_A_v5_demo+anthro_noaug_seed{seed}`
- Role: **EXPLORATORY**

### Group E — Architecture ablations (80 descriptors, 80 unique GPU runs)

All Group E runs use: variant=`demo+anthro`, meta_hid=128, lauc_w=0.08, seeds 2024–2033.

| # | arch_mode / mode | What is disabled or changed |
|---|------------------|-----------------------------|
| E1 | meta_only | ECG backbone output not used in fusion; only metadata path |
| E2 | concat | Gated fusion replaced by simple concatenation |
| E3 | additive | Gated fusion replaced by addition |
| E4 | no_glu | Gate (Sigmoid) removed |
| E5 | no_residual | ts_meta_residual injection (line 243) removed |
| E6 | no_q_meta | meta_quality = 1.0 always (availability signal disabled) |
| E7 | no_dropout | meta_dropout_p = 0.0 (no feature-level dropout) |
| E8 | trainmask | Standard arch; training with x_meta=0, mpm=0, meta_quality=0 |

`arch_mode=standard` is Group A — not re-trained in Group E.

**E8 trainmask — exact training protocol (pre-declared):**
- During training: `x_meta = zeros_like(x_meta)`, `mpm = zeros_like(mpm)`
  → `meta_quality = 0` → h_m ≈ 0 → model trains effectively ECG-only
- During evaluation: standard conditions (x_meta=real, mpm=real, meta_quality=real)
- Objective: "metadata gain requires training on real patient data"
- Expected result: AUC_trainmask ≈ AUC_none. If AUC_trainmask >> AUC_none, investigate.

Run naming: `ATLAS_A_v5_demo+anthro_{arch_mode}_seed{seed}`
Role: **EXPLORATORY**

### Group F — Multi-split sensitivity analysis (20 descriptors, 20 unique GPU runs)

**Target model:** `demo+anthro` — the pre-specified multimodal target model, independent of
Group A results.

**Fold selection rule (pre-declared, deterministic):**
Four alternate official splits were preselected to sample early and late portions of the
ordered PTB-XL split space. Within each test fold, the immediately following fold serves as
validation (val = test + 1).

| Split | test_fold | val_fold | train_folds |
|-------|-----------|----------|-------------|
| F-1 | 2 | 3 | 1,4,5,6,7,8,9,10 |
| F-2 | 3 | 4 | 1,2,5,6,7,8,9,10 |
| F-3 | 7 | 8 | 1,2,3,4,5,6,9,10 |
| F-4 | 8 | 9 | 1,2,3,4,5,6,7,10 |

Seeds: 2024–2028 (5 per split).

**Analysis unit: fold (4 units).** The 5 seeds within a split are not independent units of
generalizability. Report: mean ± SD (5 seeds) per fold, then inter-fold range and mean across
4 folds. No statistical test across the 20 runs as if independent.

**Pre-declared claim:** "Group F assesses whether fold 10 lies within the alternate-split
performance range. If fold 10 falls outside this range, this is reported as-is and discussed."

Run naming: `ATLAS_A_v5_demo+anthro_tf{test_fold}_vf{val_fold}_seed{seed}`
Role: **CONFIRMATORY (generalization check)**

---

## 6. Metadata controls — post-hoc decomposition (no retraining)

Applied to all 60 Group A checkpoints (demo+anthro, seeds 2024–2043).

### Architecture context (q_meta derivation)

`meta_quality` (= q_meta) is computed inside the model from `mpm` (meta_present_mask):
```
demo_quality  = mpm[:, :2].float().mean(dim=1, keepdim=True)
anthro_quality = mpm[:, 2:].float().mean(dim=1, keepdim=True)
meta_quality  = clamp(demo_quality + 0.5 * anthro_quality, max=1.0)
```
`meta_quality` modulates the metadata contribution in three places:
1. h_m = h_m * meta_quality (line 236)
2. h_ts += 0.10 * ts_meta_residual(h_m) * meta_quality (line 243)
3. logits_fused += 0.05 * meta_quality * logits_meta (line 246)

When mpm=0: meta_quality=0, all three terms vanish → model is effectively ECG-only.

### Four inference conditions

All conditions applied to the same 60 checkpoints. `x_meta` has 8 features:
`[age_z, sex01, height_z, weight_z, bmi_z, miss__height, miss__weight, miss__bmi]`.
`mpm` has 8 features:
`[mask__age, mask__sex, mask__height, mask__weight, mask__bmi, mask__miss_height, mask__miss_weight, mask__miss_bmi]`.

| Condition | x_meta | mpm | meta_quality | What is preserved |
|-----------|--------|-----|-------------|-------------------|
| `normal` | real | real | real (derived from real mpm) | Full metadata signal |
| `shuffle_val` | permuted across patients (rows shuffled jointly) | real | real | Presence structure preserved; value content destroyed |
| `mask_only` | zeros (all 8 features = 0) | real | real | Presence/availability signal only; values absent |
| `none` | zeros | zeros | 0 | No metadata signal |

**Decomposition:**
- Total metadata gain = AUC_normal − AUC_none (= delta_meta from Group A, consistency check)
- Value content signal = AUC_normal − AUC_shuffle_val
- Presence/availability signal = AUC_mask_only − AUC_none
- Shuffle residual = AUC_shuffle_val − AUC_mask_only (expected ≈ 0)

**Shuffle protocol:** one patient-level permutation is generated per `(variant, seed)` pair
before any batch loop, using a fixed deterministic RNG. The same permutation is then applied
jointly to all 8 features of `x_meta` for the whole test fold. `mpm` is NOT permuted, which
preserves the original availability structure.

**Note on miss__ fields:** x_meta features 5–7 (`miss__height`, `miss__weight`, `miss__bmi`)
are derived missingness indicators embedded in x_meta. They are included in the permutation
for `shuffle_val` and zeroed for `mask_only` and `none`, consistent with the respective
condition semantics.

---

## 7. Classical baselines (metadata-only)

### Primary classical baselines (always reported)

- **Models:** Logistic Regression (LR) and XGBoost, one-vs-rest per DS5 class
- **Feature set:** 8 metadata features as stored in `index_complete.parquet`
  (the exact columns used as `x_meta` by the neural model):
  `age_z`, `sex01`, `height_z`, `weight_z`, `bmi_z`,
  `miss__height`, `miss__weight`, `miss__bmi`.
  These are the z-scored / binary-encoded values produced by `index_construction.py`
  from the raw PTB-XL measurements; the parquet does not store un-normalised physical units.
- **Label derivation:** DS5 multi-hot labels derived from `scp_codes` via
  `scp_statements.csv` (identical mapping as `eznx_loader_v2.py`).
- **Missing value imputation:** median imputation on any residual NaN values;
  medians computed on training folds 1–8 only; applied to val/test without refit.
- **Standardization:** StandardScaler (zero-mean, unit-variance); fit on training folds 1–8 only;
  applied to validation and test without refit.
- **Preprocessing pipeline:** fit on train only (folds 1–8). No information from fold 9 or 10.
- **Evaluation:** macro-AUC on test fold 10
- **Purpose:** lower bound; metadata-only performance without ECG signal

### Optional ECG-assisted classical baselines (labeled separately)

- If computed: ECG summary statistics (RR interval mean/SD, QRS duration, PR interval from
  wfdb annotations if available) concatenated with the 8 metadata features above
- Labeled clearly as "ECG-summary + metadata" to distinguish from the metadata-only baseline
- Not part of primary reporting; supplementary only

---

## 8. Secondary and additional metrics (exploratory unless noted)

- macro-F1 (optimal threshold per class via val fold 9, pre-declared)
- macro-F1 (fixed threshold 0.5)
- delta_meta_auc = macro_auc(demo+anthro) − macro_auc(none) per seed
- Per-class AUC and AUPRC
- Brier score (macro)
- ECE (calibration error, 15 bins)
- Subgroup AUC: age <45 / 45–65 / >65 (thresholds: z-score equivalents of (45−62.5)/17.2 and (65−62.5)/17.2)
- Subgroup AUC: metadata-complete (all 5 values present) vs. metadata-incomplete
- Sex fairness gap: |AUC_male − AUC_female|
- MCAR missingness robustness: macro-AUC at 0/25/50/75/100% missingness (20 seeds)

**Note on HYP/LVH:** DS5 label HYP partially overlaps with LVH in clinical practice.
Per-class AUC for HYP is reported. Macro-AUC is the primary endpoint and includes HYP.
A sensitivity analysis (macro-AUC excluding HYP) is reported in supplementary materials.

---

## 9. Proof of pre-declaration

- This file is committed to the Git repository before any training run is launched.
- The Git commit SHA of this commit is the immutable timestamp of pre-declaration.
- Every `results_{run_name}.json` records `"analysis_plan_sha256": <sha256hex>` at runtime.
- `sha256(analysis_plan.md)` at evaluation time must match the value recorded in training JSONs.
- Do NOT modify this file after any training run has started.

---

*End of pre-declared analysis plan.*
