# EZNX-ATLAS-A

### Measuring the Incremental Contribution of Clinical Metadata to 12-Lead ECG Superclass Classification on PTB-XL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![PyTorch 2.3.1+cpu](https://img.shields.io/badge/PyTorch-2.3.1--cpu-EE4C2C.svg)](https://pytorch.org/get-started/locally/)
[![PTB-XL 1.0.3](https://img.shields.io/badge/dataset-PTB--XL%201.0.3-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)

**Ezyn SEGNANE** · Department of Mathematics and Computer Science, University of Nouakchott, Mauritania  
Submitted to *Mathematics* (MDPI) — manuscript v2.3.3 · 2026-04-25

---

## What this repository contains

This repository is the complete reproducibility package for the paper **EZNX-ATLAS-A**.
It provides:

- The full **model architecture** source code (`eznx_model_v5.py`, `eznx_loader_v2.py`)
- The **training and evaluation pipeline** (`atlas_a_v5_multiseed.py`, `run_multiseed_experiments.py`, `analyze_multiseed_results.py`)
- The **extended-analysis scripts** (`new_train_models/`) for the sensitivity sweeps and subclass analyses reported in the paper (H5 extended metrics, H7 GLU-width sweep, H8 LVH/HYP subclass AUC, M3 AUC-margin ablation, M4 training-curve figure)
- All **30 archived seed-level result JSON files** (`results/seed_json/`) — no GPU or checkpoint needed to verify the statistics
- The **paper figures** (Figures 1–6) in `figures/`
- The **LaTeX manuscript** source and compiled PDFs in `paper/`
- A complete **statistical analysis** package (`results/`) with paired Wilcoxon tests, Benjamini–Hochberg correction, bootstrap CIs, effect sizes, and integrity reports

All 30 runs were executed on **CPU only** (Intel Core i5, 8 GB RAM, PyTorch 2.3.1+cpu, no GPU, no CUDA). Total compute: **18.61 run-hours**. Anyone with a standard laptop can reproduce the full study.

---

## The scientific question

PTB-XL exposes age, sex, height, weight, and BMI alongside the ECG waveform. Do these structured variables actually improve superclass classification, once seed variance is accounted for? And if so, which type of metadata (demographic vs. anthropometric) helps which pathology class?

We run a **seed-matched three-variant ablation** across **10 random seeds**:

| Variant | What is provided to the model |
|---|---|
| `none` | ECG waveform only |
| `demo` | ECG + age + sex |
| `demo+anthro` | ECG + age + sex + height + weight + BMI |

---

## Key results

### Macro-AUC on PTB-XL fold 10 (mean ± SD, 10 seeds)

| Variant | Macro-AUC | 95% CI |
|---|---|---|
| `none` | 0.9274 ± 0.0008 | [0.9270, 0.9279] |
| `demo` | 0.9277 ± 0.0008 | [0.9271, 0.9281] |
| `demo+anthro` | 0.9289 ± 0.0009 | [0.9284, 0.9294] |

### Paired Wilcoxon tests (BH-FDR corrected, 36-test family)

| Comparison | Δ macro-AUC | 95% paired CI | BH-adj *p* | Cohen *d*z |
|---|---|---|---|---|
| `demo` − `none` (age + sex only) | +0.0002 | [−0.0002, +0.0007] | 0.626 | 0.30 |
| **`demo+anthro` − `none` (full)** | **+0.0014** | **[+0.0009, +0.0019]** | **0.028** | **1.77** |
| `demo+anthro` − `demo` (anthro only) | +0.0012 | [+0.0007, +0.0018] | 0.028 | 1.27 |

### Per-class AUC gain (`demo+anthro` − `none`)

| Class | Δ AUC | BH-adj *p* | Significant |
|---|---|---|---|
| NORM | +0.0014 | 0.167 | — |
| **MI** | **+0.0047** | **0.023** | **✓** |
| **STTC** | **+0.0013** | **0.023** | **✓** |
| CD | −0.0011 | 0.237 | — |
| HYP | +0.0008 | 0.482 | — |

**Main findings:**
- Demographics alone (age, sex) add **nothing statistically measurable** — consistent with their known recoverability from the raw ECG waveform (Attia et al. 2019).
- Anthropometrics (height, weight, BMI) provide a **small but statistically significant gain**, concentrated in **MI** and **STTC** — two morphology-rich classes where body-size calibration of waveform amplitudes is physiologically plausible.
- **HYP shows no significant effect**, despite classical LVH criteria depending on body habitus — likely due to superclass heterogeneity and limited power at 12% prevalence.
- Under full anthropometric masking at inference, macro-AUC drops by only **0.0010** — the quality-gated fusion degrades gracefully.

---

## Architecture

EZNX-ATLAS-A is a **3.82 M-parameter quality-gated dual-branch architecture**:

```
ECG waveform (12 × 1000)
    └── 1D ResNet backbone (3 stages, 0.96 M params)
            └── Temporal Statistics Pool (mean, SD, max, min) → h_ts ∈ ℝ¹⁰²⁴

Metadata (8-dim: age_z, sex01, height_z, weight_z, bmi_z, m_h, m_w, m_bmi)
    ├── DemoMLP   (4  → 64  → 64  → 64)
    └── AnthroMLP (12 → 96  → 96  → 64)
            └── MetaFusion MLP → h_m ∈ ℝ¹²⁸  (scaled by q_meta)

Availability score:  q_meta = min(1, q_d + 0.5·q_a)
Residual injection:  h_ts ← h_ts + 0.10·q_meta·W_res·h_m  (W_res init = 0)
GLU gate (2.66 M):   z = [h_ts ∥ h_m] ⊙ σ(Linear₂(ReLU(Linear₁([h_ts ∥ h_m]))))

Three heads:   ℓ_ecg,  ℓ_meta,  ℓ_fused = W_f·z + 0.05·q_meta·ℓ_meta
Inference:     p = w*·σ(ℓ_fused) + (1−w*)·σ(ℓ_ecg)    [w* = 1.0 in all 30 runs]
```

See `paper/main_en.pdf` for the full mathematical formulation and `figures/fig1_architecture.pdf` for a diagram.

---

## Repository structure

```
eznx-atlas-a/
├── eznx_model_v5.py                 # Model architecture (EZNX-ATLAS-A)
├── eznx_loader_v2.py                # PTB-XL data loader + ablation modes
├── atlas_a_v5_multiseed.py          # Single-seed training entry point
├── run_multiseed_experiments.py     # 30-run multi-seed orchestrator
├── analyze_multiseed_results.py     # Statistical analysis (BH-FDR, bootstrap, Wilcoxon)
├── index_construction.py            # PTB-XL index builder
├── run_all_experiments.sh           # Interactive shell helper
├── requirements.txt                 # pip dependencies (CPU-only)
├── environment.yml                  # Conda environment (CPU-only)
├── CITATION.cff                     # Machine-readable citation
├── LICENSE                          # MIT
│
├── scripts/
│   ├── evaluate_missingness_robustness.py   # Figure 4 source data (missingness CSV/JSON)
│   ├── render_architecture_figure.py        # Figure 1 generation
│   ├── render_manuscript_result_figures.py  # Figures 3, 4, 5, 6 generation (NOT Fig. 2)
│   ├── render_article_artifacts.py          # Table/artifact export
│   └── build_index.py                       # PTB-XL index construction helper
│
├── new_train_models/                # Extended-analysis scripts (sensitivity sweeps & subclass analyses)
│   ├── atlas_a_v5_extended.py       # Extended training entry point (H5/H7/H8/M3/M4)
│   ├── run_extended_experiments.py  # Orchestrator for the 5 extended configurations
│   ├── eznx_model_v5_extended.py    # Extended model variant
│   ├── eznx_loader_v2.py            # Data loader (copy used by extended scripts)
│   └── generate_fig2_m4.py         # Figure 2 training-curve generation (M4)
│
├── results/                         # All numerical artifacts from the paper
│   ├── statistical_analysis_full.json       # Master paired-statistics export
│   ├── statistical_analysis_report.md       # Human-readable analysis narrative
│   ├── seed_level_results.csv               # 30 rows (3 variants × 10 seeds)
│   ├── seed_level_results.md                # Markdown rendering of the above
│   ├── table_results_latex.tex              # LaTeX table fragment
│   ├── missingness_eval_demo_anthro_summary.json  # Figure 4 source data
│   ├── missingness_eval_demo_anthro_rows.csv      # Per-mask-rate rows
│   ├── dataset_integrity_report.json              # Patient-disjoint split audit
│   ├── dataset_integrity_report.md                # Human-readable integrity report
│   ├── statistical_analysis_protocol.md           # 36-test family documentation
│   └── seed_json/                           # 30 raw seed-level JSON files
│       ├── results_none_seed2024.json
│       ├── results_demo_seed2024.json
│       ├── results_demo+anthro_seed2024.json
│       └── ... (30 files total)
│
├── figures/                         # Paper figures (PDF)
│   ├── fig1_architecture.pdf
│   ├── fig2_training_curves.pdf
│   ├── fig3_per_class_delta_auc.pdf
│   ├── fig4_missingness_robustness.pdf
│   ├── fig5_per_class_heatmap.pdf
│   └── fig6_fused_vs_ecg_gap.pdf
│
└── paper/                           # LaTeX manuscript (source + compiled PDF)
    ├── main_en.tex                  # English submission (primary)
    ├── main_en.pdf                  # Compiled English PDF
    ├── main_en.bbl                  # BibTeX-compiled references
    ├── main.tex                     # French companion
    ├── main.pdf                     # Compiled French PDF
    ├── main.bbl
    ├── bibliography.bib             # BibTeX database
    ├── CHANGELOG.md                 # Package revision history
    ├── VERSION                      # Current version (2.3.3)
    ├── Definitions/                 # MDPI LaTeX style files
    └── figures/                     # Figures used by the LaTeX source
```

---

## Data download

This repository does **not** include the PTB-XL dataset (it is publicly available on PhysioNet under a Creative Commons licence).

```bash
# Option 1 — wget (Linux/macOS)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# Option 2 — PhysioNet CLI
pip install wfdb
python -c "import wfdb; wfdb.dl_database('ptb-xl', './ptb-xl')"
```

Set the environment variable to your local PTB-XL root before running any script:

```bash
export EZNX_DATA_REAL="/path/to/ptb-xl/1.0.3"    # Linux/macOS
# or
set EZNX_DATA_REAL=C:\path\to\ptb-xl\1.0.3        # Windows CMD
$env:EZNX_DATA_REAL = "C:\path\to\ptb-xl\1.0.3"  # PowerShell
```

---

## Installation

### Option A — pip (CPU-only, recommended for exact reproduction)

```bash
git clone https://github.com/ezynsegnane/ezyx-atlas-a_github_code.git
cd ezyx-atlas-a_github_code
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate eznx-atlas-a
```

> **Note:** All 30 paper runs used PyTorch 2.3.1+cpu on a CPU-only machine (no CUDA). CPU-only execution is inherently free of CUDA non-determinism, providing a stronger reproducibility guarantee.

---

## Reproduce paper results

### Step 0 — Verify statistics without any retraining (≈ 30 seconds)

The 30 raw seed-level JSON files are archived in `results/seed_json/`. You can recompute all paired Wilcoxon tests, BH-FDR corrections, bootstrap CIs, and effect sizes directly from these files:

```bash
python analyze_multiseed_results.py \
  --runs_dir results/seed_json \
  --output_dir results/recomputed \
  --n_bootstrap 10000
```

This requires no GPU, no PTB-XL data download, and completes in under a minute. Compare `results/recomputed/statistical_analysis_full.json` with the archived `results/statistical_analysis_full.json` to verify numerical identity.

### Step 1 — Build the PTB-XL index (one-time setup)

```bash
python index_construction.py \
  --data_root "$EZNX_DATA_REAL" \
  --output index_complete.parquet
```

### Step 2 — Run the full 30-run ablation (≈ 18–25 hours on CPU)

```bash
python run_multiseed_experiments.py \
  --data_root "$EZNX_DATA_REAL" \
  --index_path index_complete.parquet \
  --runs_dir runs_output \
  --seeds 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 \
  --variants none demo demo+anthro
```

For a quick sanity check with 3 seeds (≈ 5–6 hours):

```bash
python run_multiseed_experiments.py \
  --data_root "$EZNX_DATA_REAL" \
  --index_path index_complete.parquet \
  --runs_dir runs_quick \
  --seeds 2024 2025 2026 \
  --variants none demo demo+anthro
```

### Step 3 — Recompute statistics from your new runs

```bash
python analyze_multiseed_results.py \
  --runs_dir runs_output \
  --output_dir results/my_recomputed \
  --n_bootstrap 10000
```

### Step 4 — Reproduce anthropometric missingness robustness (Figure 4)

```bash
python scripts/evaluate_missingness_robustness.py \
  --runs-dir runs_output \
  --data-root "$EZNX_DATA_REAL" \
  --index-path index_complete.parquet \
  --stats-json results/statistical_analysis_full.json \
  --seeds 2024,2025,2026,2027,2028,2029,2030,2031,2032,2033 \
  --rhos 0,0.25,0.5,0.75,1.0
```

### Step 5 — Regenerate figures

```bash
# Figure 1 — architecture diagram
python scripts/render_architecture_figure.py

# Figure 2 — validation AUC mean±SD trajectory (reads results/seed_json/*.json)
python new_train_models/generate_fig2_m4.py

# Figures 3, 4, 5, 6 — per-class results and missingness robustness
# (reads from results/ by default; no PTB-XL data or retraining needed)
python scripts/render_manuscript_result_figures.py
```

---

## Compile the paper (optional)

The compiled PDFs are already included in `paper/`. To recompile from source:

```bash
cd paper
pdflatex -interaction=nonstopmode main_en.tex
bibtex main_en
pdflatex -interaction=nonstopmode main_en.tex
pdflatex -interaction=nonstopmode main_en.tex
```

Requires a standard LaTeX installation (MiKTeX or TeX Live) with the packages listed in `paper/Definitions/`.

---

## Reproducibility notes

| Aspect | Detail |
|---|---|
| Hardware | Intel Core i5, 8 GB RAM, 500 GB storage |
| OS | Windows 10/11 (local Jupyter Notebook environment) |
| PyTorch | 2.3.1+cpu (CPU-only wheel, no CUDA/GPU) |
| Determinism | CPU-only execution is free of CUDA non-determinism; CuDNN flags in the source code are present but inactive |
| Seeds | {2024, 2025, …, 2033} — 10 consecutive seeds crossed with 3 variants = 30 runs |
| Statistical protocol | Exact two-sided paired Wilcoxon signed-rank test; BH-FDR at q=0.05 across 36 confirmatory tests; 10,000-resample percentile bootstrap CIs |
| Model selection | Blend weight w* and per-class thresholds selected on fold 9, fixed for fold 10 |
| Patient-level arrays | Not archived (see Limitation 3 and 6 in the paper); seed-level summaries in `results/seed_json/` are the verification base |

---

## Citation

If you use this code or build on our results, please cite the paper:

```bibtex
@article{Segnane2026EZNXATLASA,
  author  = {SEGNANE, Ezyn},
  title   = {{EZNX-ATLAS-A}: Measuring the Incremental Contribution of
             Clinical Metadata to 12-Lead {ECG} Superclass Classification
             on {PTB-XL}},
  journal = {Mathematics},
  year    = {2026},
  publisher = {MDPI},
  note    = {Submitted}
}
```

You may also cite the software repository directly using the `CITATION.cff` file or the following:

```bibtex
@software{Segnane2026EZNXATLASA_code,
  author  = {SEGNANE, Ezyn},
  title   = {{EZNX-ATLAS-A} -- code and reproducibility package},
  year    = {2026},
  version = {2.3.3},
  url     = {https://github.com/ezynsegnane/ezyx-atlas-a_github_code},
  license = {MIT}
}
```

---

## Dataset citation

This study uses PTB-XL v1.0.3, which must be cited independently:

```bibtex
@article{Wagner2020,
  author  = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter
             and Kreiseler, Dieter and Lunze, Fatima I. and Samek, Wojciech
             and Schaeffter, Tobias},
  title   = {{PTB-XL}, a large publicly available electrocardiography dataset},
  journal = {Scientific Data},
  year    = {2020},
  volume  = {7},
  pages   = {154},
  doi     = {10.1038/s41597-020-0495-6}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code for any purpose — academic, commercial, or personal — provided you **retain the copyright notice** and **cite the paper** when publishing results derived from this work.

The PTB-XL dataset is subject to its own licence (Creative Commons Attribution 4.0 International). See [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) for details.

---

## Contact

Ezyn SEGNANE · [ezyn.segnane@univ-nkc.mr](mailto:ezyn.segnane@univ-nkc.mr) · ORCID [0009-0005-0538-4335](https://orcid.org/0009-0005-0538-4335)
