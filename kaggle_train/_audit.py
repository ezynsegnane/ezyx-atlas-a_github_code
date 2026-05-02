"""
_audit.py -- Exhaustive pre-upload audit for EZNX-ATLAS-A kaggle_train/.
Run from inside the kaggle_train/ directory.
"""
import pathlib, sys

import os as _os
W = pathlib.Path(_os.path.abspath(__file__)).parent if "__file__" in dir() else pathlib.Path(".")

def read(f):
    return (W / f).read_text(encoding="utf-8")

trn = read("atlas_a_v5_multiseed_v2.py")
orc = read("run_experiments_v2.py")
ldr = read("eznx_loader_v2.py")
mdl = read("eznx_model_v5.py")
idx = read("index_construction.py")
cal = read("compute_calibration.py")
ana = read("analyze_multiseed_v2.py")
sub = read("compute_subgroups.py")
mis = read("evaluate_missingness_v2.py")
nb  = read("kaggle_notebook.ipynb")

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def chk(section, item, ok, level=None, note=""):
    if ok:
        status = PASS
    else:
        status = level if level else FAIL
    results.append((section, item, status, note))


# ===================================================================
# 1. DÉTERMINISME GPU
# ===================================================================
S = "1.DETERMINISM"
# CUBLAS must be set BEFORE torch import
cublas_pos = trn.find("CUBLAS_WORKSPACE_CONFIG")
torch_pos  = trn.find("import torch")
chk(S, "CUBLAS_WORKSPACE_CONFIG set BEFORE import torch",
    cublas_pos != -1 and torch_pos != -1 and cublas_pos < torch_pos)
chk(S, "use_deterministic_algorithms(True)",
    "use_deterministic_algorithms(True" in trn)
chk(S, "cudnn.deterministic = True",
    "cudnn.deterministic = True" in trn)
chk(S, "cudnn.benchmark = False",
    "cudnn.benchmark = False" in trn)
chk(S, "matmul.allow_tf32 = False",
    "matmul.allow_tf32 = False" in trn)
chk(S, "cudnn.allow_tf32 = False  [future-proof for A100]",
    "cudnn.allow_tf32" in trn,
    WARN, "Not set -- harmless on P100 (no TF32 support) but needed if ever run on A100/H100")
chk(S, "num_workers=0 in ALL 3 DataLoaders",
    trn.count("num_workers=0") >= 3)
chk(S, "set_seed() called before training",
    "set_seed(cfg.seed)" in trn)
chk(S, "set_seed covers random / numpy / torch / cuda",
    "random.seed" in trn and "np.random.seed" in trn
    and "torch.manual_seed" in trn and "cuda.manual_seed_all" in trn)

# ===================================================================
# 2. COHÉRENCE DU NOMMAGE DE RUNS (5 scripts)
# ===================================================================
S = "2.RUN_NAMING"
for script_name, src in [
    ("training",     trn),
    ("orchestrator", orc),
    ("calibration",  cal),
    ("analyze",      ana),
]:
    chk(S, f"{script_name}: prefix ATLAS_A_v5_{{variant}}",
        "ATLAS_A_v5_" in src)
chk(S, "training uses metaH{cfg.meta_hid} suffix",
    "metaH{cfg.meta_hid}" in trn)
chk(S, "orchestrator uses metaH{meta_hid} suffix",
    "metaH{meta_hid}" in orc)
chk(S, "training: lauc suffix uses :g format",
    "lauc{cfg.lauc_weight:g}" in trn)
chk(S, "orchestrator: lauc suffix uses :g format",
    "lauc{lauc_weight:g}" in orc)
chk(S, "training: threshold abs(lauc-0.08)>1e-6",
    "abs(cfg.lauc_weight - 0.08) > 1e-6" in trn)
chk(S, "orchestrator: threshold abs(lauc-0.08)>1e-6",
    "abs(lauc_weight - 0.08) > 1e-6" in orc)
chk(S, "training: noaug suffix",
    "noaug" in trn)
chk(S, "orchestrator: noaug suffix",
    "noaug" in orc)
chk(S, "missingness uses Group-A naming only (no suffixes)",
    'f"ATLAS_A_v5_{variant}_seed{seed}"' in mis)
# calibration uses same naming as training
chk(S, "calibration naming matches training (metaH / lauc / noaug)",
    "metaH{meta_hid}" in cal and "lauc{lauc_weight:g}" in cal)

# ===================================================================
# 3. AUTO-RESUME
# ===================================================================
S = "3.AUTO_RESUME"
chk(S, "training skips if results JSON already exists",
    "results_path.exists()" in trn)
chk(S, "orchestrator checks result_file before launching subprocess",
    "result_file" in orc and ".exists()" in orc)
chk(S, "result filename: results_{run_name}.json  (training)",
    "results_{run_name}.json" in trn)
chk(S, "result filename: results_{name}.json  (orchestrator)",
    "results_{name}.json" in orc)
# These two patterns must name the same file
chk(S, "run_dir used consistently for all 3 output files",
    "run_dir / f" in trn and "results_" in trn
    and "best_model_" in trn and "probs_" in trn)

# ===================================================================
# 4. DATASETS / FOLDS
# ===================================================================
S = "4.DATASETS_FOLDS"
chk(S, "training folds 1--8: range(1, 9)",
    "range(1, 9)" in trn)
chk(S, "val fold hardcoded to 9",
    "fold=9" in trn)
chk(S, "test fold hardcoded to 10",
    "fold=10" in trn)
chk(S, "loader uses filename_lr for 100 Hz (col_name='filename_lr')",
    "col_name = 'filename_lr'" in ldr or "filename_lr" in ldr)
chk(S, "shuffle=True only for train_loader",
    trn.count("shuffle=True") == 1)
chk(S, "val/test loaders no shuffle (default False)",
    trn.count("shuffle=True") == 1)  # already checked above; symmetry
chk(S, "EZNXDataset receives meta_mode=cfg.variant",
    "meta_mode=cfg.variant" in trn)
chk(S, "ConcatDataset wraps 8 train folds",
    "ConcatDataset(train_datasets)" in trn)

# ===================================================================
# 5. POS_WEIGHTS DYNAMIQUES
# ===================================================================
S = "5.POS_WEIGHTS"
chk(S, "compute_pos_weights() defined",
    "def compute_pos_weights" in trn)
chk(S, "reads .y from each train dataset",
    "ds.y for ds in train_datasets" in trn)
chk(S, "pos_weight_j = neg_j / pos_j",
    "neg / np.maximum(pos, 1)" in trn or "neg_j / pos_j" in trn)
chk(S, "clipped to [0.5, 30.0]",
    "0.5, 30.0" in trn)
chk(S, "BCEWithLogitsLoss(pos_weight=pos_weights)",
    "BCEWithLogitsLoss(pos_weight=pos_weights)" in trn)
chk(S, "pos_weights saved in results JSON",
    '"pos_weights"' in trn)

# ===================================================================
# 6. LOSS WEIGHTS & LAUC NORMALISATION
# ===================================================================
S = "6.LOSS_WEIGHTS"
chk(S, "fused_w = max(0.0, 0.60 - lauc_w)",
    "max(0.0, 0.60 - lauc_w)" in trn)
chk(S, "total_w = fused_w + ecg_w + meta_w + lauc_w",
    "fused_w + ecg_w + meta_w + lauc_w" in trn)
chk(S, "fused_w /= total_w",
    "fused_w /= total_w" in trn)
chk(S, "ecg_w /= total_w",
    "ecg_w /= total_w" in trn)
chk(S, "lauc_w_n = lauc_w / total_w",
    "lauc_w_n = lauc_w / total_w" in trn)
chk(S, "lauc_w_n (normalised) used in training loss",
    "lauc_w_n *" in trn)
chk(S, "loss_weights logged to JSON",
    '"loss_weights"' in trn)
# Verify lauc=0.0 case: fused_w=0.60, total=1.0 -> no division problem
chk(S, "auc_margin_loss handles empty pos/neg gracefully",
    "pos.numel() == 0 or neg.numel() == 0" in trn)

# ===================================================================
# 7. NPZ DUMP -- clés complètes
# ===================================================================
S = "7.NPZ_DUMP"
for key in ["Y=", "P_fused=", "P_ecg=", "P_meta=", "P_blend=",
            "P_no_meta=", "ecg_id=", "patient_id=", "thresholds=",
            "w_fused=", "labels="]:
    chk(S, f"NPZ key: {key.rstrip('=')}",
        key in trn)
chk(S, "NPZ saved with np.savez_compressed",
    "savez_compressed" in trn)
chk(S, "NPZ filename: probs_{run_name}.npz",
    "probs_{run_name}.npz" in trn)

# ===================================================================
# 8. SUBGROUPE -- ordre des lignes / cohérence
# ===================================================================
S = "8.SUBGROUP_ORDER"
chk(S, "training: test_df_raw = parquet[fold==10].reset_index",
    "== 10].reset_index(drop=True)" in trn)
chk(S, "compute_subgroups: masks from parquet[fold==10].reset_index",
    "== 10].reset_index(drop=True)" in sub)
chk(S, "age thresholds identical: (45-PTB_MEAN)/PTB_SD",
    "(45  - _PTB_AGE_MEAN) / _PTB_AGE_SD" in trn or
    "(45 - _PTB_AGE_MEAN) / _PTB_AGE_SD" in trn)
chk(S, "compute_subgroups uses same PTB_AGE_MEAN=62.5, SD=17.2",
    "62.5" in sub and "17.2" in sub)
chk(S, "anthro_complete uses meta_present_strict==1",
    "meta_present_strict" in sub)
chk(S, "fairness_sex_gap computed as |male - female|",
    "fairness_sex_gap" in sub)
chk(S, "compute_subgroups reads P_blend from NPZ (not raw logits)",
    'P_blend' in sub)

# ===================================================================
# 9. MISSINGNESS -- RNG dérive (bug potentiel)
# ===================================================================
S = "9.MISSINGNESS"
chk(S, "MISS_RATES = [0.0, 0.25, 0.50, 0.75, 1.0]",
    "0.0, 0.25, 0.50, 0.75, 1.0" in mis)
chk(S, "all 3 variants evaluated",
    '"none"' in mis and '"demo"' in mis and '"demo+anthro"' in mis)
chk(S, "SEEDS_20 = range(2024,2044) used",
    "range(2024, 2044)" in mis)
chk(S, "base rng seed 42 used for MCAR masks",
    "default_rng(42" in mis)   # matches both old (42) and new (42 + offset)
chk(S, "rng is per-(variant,seed) -- NOT a single shared instance",
    # If there is only ONE call to default_rng, it's the shared-rng bug
    mis.count("default_rng(42") > 1 or "default_rng(42 +" in mis,
    FAIL,
    "BUG: single global rng shared across all seeds. If any checkpoint is "
    "missing, all subsequent seeds get different MCAR masks -> non-reproducible results.")
chk(S, "missingness uses blended probs (not raw logits_fused only)",
    "w_fused" in mis or "blend_probs" in mis,
    WARN, "Uses torch.sigmoid(logits_fused) -- equivalent to w_fused=1.0. "
    "Baseline AUC will differ slightly from results.json blend value.")
chk(S, "LaTeX table caption mentions MCAR scope limitation",
    "MCAR" in mis and "MAR" in mis)
chk(S, "model instantiated with default meta_hid=128 (Group A only)",
    "EZNX_ATLAS_A_v5(" in mis and "meta_hid" not in mis.split("EZNX_ATLAS_A_v5(")[1][:50])

# ===================================================================
# 10. MODÈLE -- dimensions et cohérence
# ===================================================================
S = "10.MODEL_DIMS"
chk(S, "meta_dim=16 passed by training script",
    "meta_dim=16" in trn)
chk(S, "META_FEATURES has 8 entries (= meta_dim//2)",
    ldr.count('"age_z"') + ldr.count('"sex01"') +
    ldr.count('"height_z"') + ldr.count('"weight_z"') +
    ldr.count('"bmi_z"') + ldr.count('"miss__height"') +
    ldr.count('"miss__weight"') + ldr.count('"miss__bmi"') == 8)
chk(S, "MASK_FEATURES has 8 entries",
    ldr.count('"mask__') == 8)
chk(S, "meta_fuse input=130 (64+64+1+1)",
    "Linear(130," in mdl)
chk(S, "demo_encoder: input = demo_value_dim*2",
    "self.demo_value_dim * 2" in mdl)
chk(S, "anthro_encoder: input = anthro_value_dim*2",
    "self.anthro_value_dim * 2" in mdl)
chk(S, "gated fusion: gate(h)*h",
    "self.gate(h)" in mdl)
chk(S, "3 output heads: head_ecg, head_meta, head_fused",
    "head_ecg" in mdl and "head_meta" in mdl and "head_fused" in mdl)
chk(S, "count_parameters() exported",
    "def count_parameters" in mdl)
chk(S, "model output returns dict with logits_fused/ecg/meta",
    '"logits_fused"' in mdl and '"logits_ecg"' in mdl and '"logits_meta"' in mdl)

# ===================================================================
# 11. CONFIGURATION D'ENTRAÎNEMENT
# ===================================================================
S = "11.TRAINING_CFG"
chk(S, "epochs=10 (budget-calibrated for 28h P100)",
    "epochs:                     int   = 10" in trn)
chk(S, "patience=25 > epochs=10  [WARNING: patience is dead parameter]",
    "patience:                   int   = 25" in trn,
    WARN, "patience(25) > epochs(10): early stopping never triggers. "
    "Not harmful -- best checkpoint is still saved. Add a code comment.")
chk(S, "gradient_accumulation_steps=2",
    "gradient_accumulation_steps: int  = 2" in trn)
chk(S, "max_grad_norm=1.0",
    "max_grad_norm:              float = 1.0" in trn)
chk(S, "AdamW with weight_decay=5e-4",
    "AdamW" in trn and "weight_decay=5e-4" in trn)
chk(S, "CosineAnnealingWarmRestarts T_0=10 T_mult=2 eta_min=1e-6",
    "T_0=10, T_mult=2, eta_min=1e-6" in trn)
chk(S, "trailing micro-batch flush at end of epoch",
    "Trailing micro-batch flush" in trn)
chk(S, "checkpoint includes thresholds + w_fused + optimizer state",
    '"thresholds"' in trn and '"w_fused"' in trn and "optimizer_state_dict" in trn)
chk(S, "no_aug: train_loader uses collate_fn_val (no augment)",
    "collate_fn_val if cfg.no_aug else collate_fn_augmented" in trn)
chk(S, "GPU memory peak logged",
    "gpu_peak_mem_mb" in trn or "max_memory_allocated" in trn)
chk(S, "hardware provenance saved in JSON",
    "get_hardware_provenance" in trn and '"hardware"' in trn)

# ===================================================================
# 12. MÉTRIQUES DANS LE JSON DE RÉSULTAT
# ===================================================================
S = "12.JSON_METRICS"
for key in ["macro_auc", "macro_auc_ecg", "macro_auc_no_meta",
            "delta_meta_auc", "macro_f1_optimal", "macro_f1_fixed_05",
            "w_fused", "thresholds"]:
    chk(S, f'test JSON contains "{key}"',
        f'"{key}"' in trn)
chk(S, "per_class dict: auc + f1 + threshold for each DS5 label",
    '"auc"' in trn and '"f1"' in trn and '"threshold"' in trn
    and '"per_class"' in trn)
chk(S, "subgroups dict saved in JSON",
    '"subgroups"' in trn)
chk(S, "training_history saved in JSON",
    '"training_history"' in trn)

# ===================================================================
# 13. ANALYSE -- métriques agrégées
# ===================================================================
S = "13.ANALYSIS_METRICS"
chk(S, "macro_auc collected for all 3 variants",
    'metric_key="macro_auc"' in ana)
chk(S, "macro_f1_optimal collected per variant",
    'metric_key="macro_f1_optimal"' in ana)
chk(S, "macro_f1_fixed_05 collected per variant",
    'metric_key="macro_f1_fixed_05"' in ana)
chk(S, "delta_meta_auc collected per variant",
    'metric_key="delta_meta_auc"' in ana)
chk(S, "95% CI via t.ppf(0.975) in summarise()",
    "t.ppf(0.975" in ana)
chk(S, "Wilcoxon exact test (method='exact' for n<=25)",
    'method="exact"' in ana or "method = \"exact\"" in ana)
chk(S, "effect size r = z/sqrt(n) computed",
    "r = abs(z)" in ana or "abs(z) / np.sqrt(n)" in ana)
chk(S, "CI interval [lo, hi] in primary LaTeX table",
    "[{lo:.4f},{hi:.4f}]" in ana)
chk(S, "Macro-F1 column in primary LaTeX table",
    "Macro-F1" in ana)
chk(S, "AUPRC in compute_calibration",
    "macro_auprc_mean" in cal and "average_precision_score" in cal)
chk(S, "per_class_auprc in calibration report",
    "per_class_auprc" in cal)
chk(S, "Brier + ECE in calibration report",
    "brier_mean" in cal and "ece_mean" in cal)

# ===================================================================
# 14. INDEX CONSTRUCTION
# ===================================================================
S = "14.INDEX_CONSTRUCTION"
chk(S, "two-step pipeline in docstring",
    "Step 1" in idx and "Step 2" in idx)
chk(S, "build_mm_core() -> index_mm_core.parquet",
    "index_mm_core.parquet" in idx)
chk(S, "build_complete() -> index_complete.parquet",
    "index_complete.parquet" in idx)
chk(S, "argparse: --data-root and --out-dir",
    "--data-root" in idx and "--out-dir" in idx)
chk(S, "integrity check: raises ValueError if required cols missing",
    "ValueError" in idx and "CRITICAL" in idx)
chk(S, "hea_path column removed before Step 2 merge (drop_cols)",
    "drop_cols" in idx)
chk(S, "inner join on ecg_id (no phantom rows)",
    'how="inner"' in idx)
chk(S, "notebook asserts exactly 21799 rows",
    "21799" in nb)

# ===================================================================
# 15. NOTEBOOK
# ===================================================================
S = "15.NOTEBOOK"
# Cell 3 clones from GitHub; Cell 4 builds index -- 3 must come first
chk(S, "Cell 3 (GitHub bootstrap) BEFORE Cell 4 (index build)",
    nb.index("Bootstrap code from GitHub") < nb.index("Build index_complete"))
chk(S, "Cell 3 clones repo and sets CODE_DIR",
    "CODE_DIR" in nb and ("git clone" in nb or "'git'" in nb or '"git"' in nb))
chk(S, "REPO_URL has a real GitHub URL",
    "github.com" in nb)
chk(S, "index_construction.py uses CODE_DIR (not hardcoded path)",
    "CODE_DIR / 'index_construction.py'" in nb or
    "str(CODE_DIR / " in nb)
chk(S, "Cell 6 uses check=False (single failure does not abort campaign)",
    "check=False" in nb)
chk(S, "all 4 eval scripts called in Cell 7 via CODE_DIR",
    "analyze_multiseed_v2.py" in nb and "compute_calibration.py" in nb
    and "compute_subgroups.py" in nb and "evaluate_missingness_v2.py" in nb)
chk(S, "PTB-XL path auto-detected via glob (no hardcode)",
    "glob.glob" in nb and "ptbxl_database.csv" in nb)
chk(S, "scp_statements.csv presence checked in Cell 2",
    "scp_statements.csv" in nb)
chk(S, "records100 and records500 presence checked",
    "records100" in nb and "records500" in nb)
chk(S, "env vars EZNX_DATA_REAL/INDEX_PATH/RUNS_DIR set for subprocesses",
    "EZNX_DATA_REAL" in nb and "EZNX_INDEX_PATH" in nb and "EZNX_RUNS_DIR" in nb)
chk(S, "Cell 8 zip: INCLUDE_NPZ flag (default False to keep size small)",
    "INCLUDE_NPZ" in nb)
chk(S, "GitHub repo has REPO_REF fallback (supports branch pinning)",
    "REPO_REF" in nb)
chk(S, "required_scripts list verified after clone",
    "required_scripts" in nb)
# CRITICAL: if GitHub repo is private or doesn't exist yet, Cell 3 will fail
chk(S, "WARNING -- GitHub repo must be public and contain kaggle_train/ BEFORE launch",
    "REPO_URL" in nb,
    WARN, "Cell 3 git-clones from GitHub. The repo must be public and "
    "pushed BEFORE launching the notebook on Kaggle.")

# ===================================================================
# PRINT RESULTS
# ===================================================================
print()
print("=" * 78)
print("AUDIT FINAL -- EZNX-ATLAS-A SCIENTIFIC REPORTS -- kaggle_train/")
print("=" * 78)

sections_dict = {}
for section, item, status, note in results:
    sections_dict.setdefault(section, []).append((item, status, note))

total_fail = total_warn = total_pass = 0
for sec, items in sections_dict.items():
    nf = sum(1 for _, s, _ in items if s == FAIL)
    nw = sum(1 for _, s, _ in items if s == WARN)
    np_ = sum(1 for _, s, _ in items if s == PASS)
    total_fail += nf; total_warn += nw; total_pass += np_
    label = "ALL OK" if nf == 0 and nw == 0 else (
        "WARN" if nf == 0 else "FAIL")
    print(f"\n{sec}  [{label}]")
    for item, status, note in items:
        sym = "OK" if status == PASS else ("WW" if status == WARN else "XX")
        print(f"  [{sym}] {item}")
        if note and status != PASS:
            # wrap long notes
            wrapped = "\n       ".join(
                [note[i:i+72] for i in range(0, len(note), 72)])
            print(f"       >> {wrapped}")

print()
print("=" * 78)
print(f"TOTAL : {total_pass+total_warn+total_fail} checks | "
      f"{total_pass} OK | {total_warn} WARN | {total_fail} FAIL")
print("=" * 78)

if total_fail > 0:
    print("STATUS: BLOCKERS FOUND -- fix before upload")
    sys.exit(1)
elif total_warn > 0:
    print("STATUS: READY WITH WARNINGS -- review WW items above")
else:
    print("STATUS: FULLY CLEAN -- ready to upload")
