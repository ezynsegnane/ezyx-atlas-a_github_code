# Hardware & Software Provenance

## Compute Platform

| Item | Value |
|------|-------|
| Platform | Kaggle Notebooks (https://www.kaggle.com) |
| GPU | NVIDIA Tesla P100-PCIE-16GB |
| VRAM | 16 GB HBM2 |
| CPU | Intel Xeon (2 cores, Kaggle default) |
| RAM | 13 GB |
| Storage | 20 GB ephemeral (/kaggle/working) |

## Software Stack

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| PyTorch | 2.1.x (Kaggle default) |
| CUDA | 11.8 |
| NumPy | 1.26.x |
| scikit-learn | 1.3.x |
| pandas | 2.1.x |
| wfdb | 4.1.2 |
| pyarrow | 14.x |
| scipy | 1.11.x |

*Exact versions logged per run in `metadata.hardware` of each results JSON.*

## Reproducibility Protocol

1. **Seed policy**: 20 consecutive integer seeds 2024–2043 (year of study initiation).
   No seed was selected or excluded based on performance.

2. **Deterministic flags** (set before any training):
   - `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`
   - `torch.use_deterministic_algorithms(True, warn_only=True)`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
   - `torch.backends.cuda.matmul.allow_tf32 = False`

3. **DataLoader**: `num_workers=0` (eliminates multi-process shuffle non-determinism).

4. **Positive-class weights**: Computed dynamically from training-fold label
   prevalences at each run start (not hard-coded).

5. **Expected bit-level reproducibility**: Results are expected to be exactly
   reproducible when the identical script is run on the same GPU model (P100)
   with the same software versions. Variation across different GPU architectures
   is bounded by ±0.001 macro-AUC (empirically confirmed; reported SDs are
   0.0008–0.0009 across seeds, which absorbs hardware-level float rounding).

## Dataset

| Item | Value |
|------|-------|
| Dataset | PTB-XL v1.0.3 (PhysioNet) |
| DOI | 10.13026/kfzx-aw45 |
| Sampling rate | 100 Hz (10-second, 1000 time points) |
| Leads | 12 |
| Total records | 21 799 |
| Train (folds 1-8) | 17 111 |
| Validation (fold 9) | 2 193 |
| Test (fold 10) | 2 495 |

## Experiment Summary

| Group | Description | Runs |
|-------|-------------|------|
| A | Primary ablation (3 variants × 20 seeds) | 60 |
| B | GLU-width / meta_hid sensitivity (3 values × 20 seeds) | 60 |
| C | LAUC-weight sensitivity (2 values × 20 seeds) | 40 |
| D | No-augmentation sensitivity (10 seeds) | 10 |
| **Total** | | **170** |

## Estimated Total Compute

Approx. 8–10 min per run × 170 runs ≈ **24–28 GPU hours** on P100.
Spread across 2–3 Kaggle sessions (12 h/session limit).
Auto-resume ensures no work is lost across sessions.
