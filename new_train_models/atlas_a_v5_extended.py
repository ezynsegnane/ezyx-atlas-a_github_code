# =============================================================================
# atlas_a_v5_extended.py
# Extended training script integrating all review items that require retraining.
#
# New vs. atlas_a_v5_multiseed.py:
#   H5  --compute_extended_metrics : AUPRC, Brier, DeLong CI, ECE per class
#   H7  --gate_hidden_dim INT      : GLU gate hidden width (512 / 1152 / 2048)
#   H8  (automatic when variant=demo+anthro): HYP/LVH subclass AUC
#   M3  --lauc_weight FLOAT        : AUC-margin surrogate weight (0.0 = ablation)
#   M4  (automatic): per-epoch training history always saved → mean±SD figure
#
# Usage examples:
#   H5+H8 : python atlas_a_v5_extended.py --variant demo+anthro --seed 2029
#   H7-512: python atlas_a_v5_extended.py --variant demo+anthro --seed 2026 --gate_hidden_dim 512
#   M3    : python atlas_a_v5_extended.py --variant demo+anthro --seed 2026 --lauc_weight 0.0
# =============================================================================

import os, json, random, argparse, sys, ast
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import (
    f1_score, roc_auc_score,
    average_precision_score,     # AUPRC  (H5)
    brier_score_loss,            # Brier  (H5)
)
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # ezyx-atlas-a_gihub/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import extended model and original loader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eznx_model_v5_extended import EZNX_ATLAS_A_v5, count_parameters
from eznx_loader_v2 import EZNXDataset, DS5_LABELS

DEFAULT_BLEND_CANDIDATES = (0.0, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0)
DEFAULT_DATA_ROOT  = Path(os.getenv("EZNX_DATA_REAL",  PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3"))
DEFAULT_INDEX_PATH = Path(os.getenv("EZNX_INDEX_PATH", PROJECT_ROOT / "index_complete.parquet"))
DEFAULT_RUNS_DIR   = Path(os.getenv("EZNX_RUNS_DIR",   PROJECT_ROOT / "runs_extended"))

# H8 – LVH SCP codes in PTB-XL
# LVH: left ventricular hypertrophy (voltage/morphology criteria)
# VCLVH: voltage criteria for LVH (borderline cases near the threshold boundary)
LVH_SCP_CODES = {"LVH", "VCLVH"}


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Data augmentation & collation  (unchanged from original)
# =============================================================================
class ECGAugmentation:
    @staticmethod
    def add_gaussian_noise(x, noise_level=0.02):
        if np.random.rand() > 0.3: return x
        return x + torch.randn_like(x) * noise_level

    @staticmethod
    def time_shift(x, max_shift=20):
        if np.random.rand() > 0.3: return x
        shift = np.random.randint(-max_shift, max_shift)
        return torch.roll(x, shift, dims=-1) if shift != 0 else x

    @staticmethod
    def amplitude_scale(x, scale_range=(0.95, 1.05)):
        if np.random.rand() > 0.3: return x
        return x * np.random.uniform(*scale_range)


def normalize_ts_voltage(x): return x / 5.0


def collate_fn_augmented(items):
    x_ts   = normalize_ts_voltage(torch.stack([it["x_ts"]             for it in items]))
    x_meta = torch.stack([it["x_meta"]            for it in items])
    mpm    = torch.stack([it["meta_present_mask"]  for it in items])
    y      = torch.stack([it["y"]                  for it in items])
    if np.random.rand() > 0.5:
        x_ts = ECGAugmentation.add_gaussian_noise(x_ts)
        x_ts = ECGAugmentation.time_shift(x_ts)
        x_ts = ECGAugmentation.amplitude_scale(x_ts)
    return x_ts, x_meta, mpm, y


def collate_fn_val(items):
    x_ts   = normalize_ts_voltage(torch.stack([it["x_ts"]             for it in items]))
    x_meta = torch.stack([it["x_meta"]            for it in items])
    mpm    = torch.stack([it["meta_present_mask"]  for it in items])
    y      = torch.stack([it["y"]                  for it in items])
    return x_ts, x_meta, mpm, y


# =============================================================================
# Metrics utilities
# =============================================================================
def safe_macro_auroc(Y, P):
    aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])
            if len(np.unique(Y[:, j])) >= 2]
    return float(np.mean(aucs)) if aucs else float("nan")


def find_optimal_thresholds(Y, P):
    thr = np.full(Y.shape[1], 0.5, dtype=np.float32)
    for j in range(Y.shape[1]):
        best, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 61):
            f1 = f1_score(Y[:, j], (P[:, j] >= t).astype(int), zero_division=0)
            if f1 > best: best, best_t = f1, t
        thr[j] = best_t
    return thr


def compute_metrics_per_class(Y, P, thr):
    aucs, f1s = [], []
    for j in range(Y.shape[1]):
        auc = roc_auc_score(Y[:, j], P[:, j]) if len(np.unique(Y[:, j])) >= 2 else float("nan")
        f1  = f1_score(Y[:, j], (P[:, j] >= thr[j]).astype(int), zero_division=0)
        aucs.append(auc); f1s.append(f1)
    return np.nanmean(aucs), np.nanmean(f1s), aucs, f1s


# =============================================================================
# H5 – Extended metrics
# =============================================================================

def _delong_var(y_true, p_score):
    """
    Fast vectorised DeLong variance (Sun & Xu 2014).
    Returns (auc, se).  Returns (nan, nan) if either class has < 2 samples.
    Memory: O(n_pos × n_neg); safe for fold-10 scale (~2000 samples).
    """
    pos = p_score[y_true == 1]
    neg = p_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return float("nan"), float("nan")
    # Placement matrix (m × n): 1 if pos > neg, 0.5 if tie, 0 otherwise
    gt   = (pos[:, None] > neg[None, :]).astype(np.float64)
    ties = (pos[:, None] == neg[None, :]).astype(np.float64) * 0.5
    pmat = gt + ties
    V10  = pmat.mean(axis=1)         # (m,) – each positive vs all negatives
    V01  = 1.0 - pmat.mean(axis=0)  # (n,) – each negative vs all positives
    auc  = float(V10.mean())
    var  = np.var(V10, ddof=1) / m + np.var(V01, ddof=1) / n
    return auc, float(np.sqrt(var))


def delong_ci_95(y_true, p_score):
    """
    DeLong (1988) 95% CI for binary AUC.
    Returns (auc, ci_lo, ci_hi).  NaN values when a class has no positive examples.
    """
    auc, se = _delong_var(y_true, p_score)
    if np.isnan(se):
        return float("nan"), float("nan"), float("nan")
    z = scipy_norm.ppf(0.975)   # 1.96
    return auc, float(auc - z * se), float(auc + z * se)


def compute_ece(y_true, p_score, n_bins=10):
    """
    Expected Calibration Error with equal-width bins.
    ECE = Σ_b (|B_b| / N) · |acc(B_b) – conf(B_b)|
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_score >= lo) & (p_score < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(p_score[mask].mean())
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compute_extended_metrics(Y: np.ndarray, P: np.ndarray) -> Dict[str, Any]:
    """
    Compute H5 extended evaluation metrics.

    Returns a dict with per-class and macro values for:
      - AUPRC (average precision)
      - Brier score
      - DeLong 95% CI for AUC
      - ECE (Expected Calibration Error)
    """
    results: Dict[str, Any] = {
        "per_class": {},
        "macro_auprc": None,
        "macro_brier": None,
        "macro_ece": None,
    }
    auprcs, briers, eces = [], [], []

    for j, label in enumerate(DS5_LABELS):
        y_j = Y[:, j]
        p_j = P[:, j]

        # AUPRC
        auprc = (
            float(average_precision_score(y_j, p_j))
            if len(np.unique(y_j)) >= 2 else float("nan")
        )

        # Brier (brier_score_loss expects pos_label=1, lower is better)
        brier = (
            float(brier_score_loss(y_j, p_j))
            if len(np.unique(y_j)) >= 2 else float("nan")
        )

        # DeLong 95% CI for AUC
        auc_dl, ci_lo, ci_hi = delong_ci_95(y_j, p_j)

        # ECE
        ece = compute_ece(y_j, p_j)

        results["per_class"][label] = {
            "auprc":            auprc,
            "brier":            brier,
            "auc_delong":       auc_dl,
            "auc_delong_ci_lo": ci_lo,
            "auc_delong_ci_hi": ci_hi,
            "ece":              ece,
        }

        if not np.isnan(auprc):  auprcs.append(auprc)
        if not np.isnan(brier):  briers.append(brier)
        eces.append(ece)

    results["macro_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")
    results["macro_brier"] = float(np.mean(briers)) if briers else float("nan")
    results["macro_ece"]   = float(np.mean(eces))
    return results


# =============================================================================
# H8 – HYP / LVH subclass analysis
# =============================================================================

def get_lvh_mask(df) -> np.ndarray:
    """
    Return a boolean array (len == len(df)) marking records that contain at
    least one SCP code in LVH_SCP_CODES = {LVH, VCLVH}.

    LVH  : Left Ventricular Hypertrophy (voltage / morphology criteria).
    VCLVH: Voltage Criteria for LVH (borderline cases near threshold boundary).
    """
    mask = []
    for scp_str in df["scp_codes"]:
        try:
            codes = ast.literal_eval(scp_str)
            if isinstance(codes, dict):
                codes = set(codes.keys())
            mask.append(bool(set(codes) & LVH_SCP_CODES))
        except Exception:
            mask.append(False)
    return np.array(mask, dtype=bool)


def compute_hyp_lvh_subclass(
    Y: np.ndarray,
    P: np.ndarray,
    index_path: str,
) -> Dict[str, Any]:
    """
    Compute HYP-AUC restricted to LVH-specific records (H8).

    Loads fold-10 from the index parquet (row order matches Y / P order because
    the test DataLoader is created without shuffle).
    """
    import pandas as pd

    df10 = pd.read_parquet(index_path)
    df10 = df10[df10["strat_fold"] == 10].reset_index(drop=True)

    if len(df10) != len(Y):
        return {
            "error": (
                f"Row count mismatch: index fold-10 has {len(df10)} rows "
                f"but Y has {len(Y)} rows."
            )
        }

    lvh_mask = get_lvh_mask(df10)
    n_lvh    = int(lvh_mask.sum())

    if n_lvh < 10:
        return {"n_lvh_records": n_lvh, "error": "Too few LVH records for reliable AUC."}

    HYP_IDX = DS5_LABELS.index("HYP")
    Y_lvh   = Y[lvh_mask, HYP_IDX]
    P_lvh   = P[lvh_mask, HYP_IDX]

    unique_classes = np.unique(Y_lvh)
    if len(unique_classes) < 2:
        return {
            "n_lvh_records":    n_lvh,
            "n_lvh_hyp_pos":    int(Y_lvh.sum()),
            "error":            "LVH subgroup has only one class label; AUC undefined.",
        }

    auc_lvh             = float(roc_auc_score(Y_lvh, P_lvh))
    auc_dl, ci_lo, ci_hi = delong_ci_95(Y_lvh, P_lvh)
    auprc_lvh           = float(average_precision_score(Y_lvh, P_lvh))
    brier_lvh           = float(brier_score_loss(Y_lvh, P_lvh))
    ece_lvh             = compute_ece(Y_lvh, P_lvh)

    # Full HYP AUC (all HYP-positive records, not just LVH) for reference
    auc_hyp_all = float(roc_auc_score(Y[:, HYP_IDX], P[:, HYP_IDX]))

    return {
        "n_lvh_records":         n_lvh,
        "n_lvh_hyp_positive":    int(Y_lvh.sum()),
        "n_lvh_hyp_negative":    int((~Y_lvh.astype(bool)).sum()),
        "auc_lvh":               auc_lvh,
        "auc_lvh_delong":        auc_dl,
        "auc_lvh_delong_ci_lo":  ci_lo,
        "auc_lvh_delong_ci_hi":  ci_hi,
        "auprc_lvh":             auprc_lvh,
        "brier_lvh":             brier_lvh,
        "ece_lvh":               ece_lvh,
        "auc_hyp_all_records":   auc_hyp_all,
        "lvh_scp_codes_used":    sorted(LVH_SCP_CODES),
    }


# =============================================================================
# Blending & inference  (unchanged logic from original)
# =============================================================================

def auc_margin_loss(y_true, p):
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=p.device)
    return torch.mean((1 - pos.unsqueeze(1) + neg.unsqueeze(0)).clamp(min=0))


@torch.no_grad()
def collect_branch_probs(model, loader, device, disable_meta=False):
    model.eval()
    ys, ps_fused, ps_ecg, ps_meta = [], [], [], []
    for x_ts, x_meta, mpm, y in loader:
        x_ts, x_meta, mpm = x_ts.to(device), x_meta.to(device), mpm.to(device)
        if disable_meta:
            x_meta = torch.zeros_like(x_meta)
            mpm    = torch.zeros_like(mpm)
        out = model(x_ts, x_meta, mpm)
        ps_fused.append(torch.sigmoid(out["logits_fused"]).cpu().numpy())
        ps_ecg.append(  torch.sigmoid(out["logits_ecg"]).cpu().numpy())
        ps_meta.append( torch.sigmoid(out["logits_meta"]).cpu().numpy())
        ys.append(y.numpy())
    return (np.concatenate(ys), np.concatenate(ps_fused),
            np.concatenate(ps_ecg), np.concatenate(ps_meta))


def blend_probs(p_fused, p_ecg, w_fused):
    return w_fused * p_fused + (1.0 - w_fused) * p_ecg


@torch.no_grad()
def select_best_val_blend(model, loader, device, blend_candidates=None):
    y_true, p_fused, p_ecg, p_meta = collect_branch_probs(model, loader, device)
    _, p_fused_nm, p_ecg_nm, _     = collect_branch_probs(model, loader, device, disable_meta=True)
    best = {"w_fused": 1.0, "auc": -1.0, "delta_meta": -1.0,
            "probs": p_fused, "probs_nometa": p_fused_nm,
            "p_fused": p_fused, "p_ecg": p_ecg, "p_meta": p_meta}
    if blend_candidates is None:
        blend_candidates = np.asarray(DEFAULT_BLEND_CANDIDATES, dtype=float)
    for w in blend_candidates:
        probs    = blend_probs(p_fused, p_ecg, float(w))
        probs_nm = blend_probs(p_fused_nm, p_ecg_nm, float(w))
        auc      = safe_macro_auroc(y_true, probs)
        auc_nm   = safe_macro_auroc(y_true, probs_nm)
        delta    = auc - auc_nm
        if (auc > best["auc"] + 1e-6
                or (abs(auc - best["auc"]) <= 5e-4 and delta > best["delta_meta"] + 1e-6)
                or (abs(auc - best["auc"]) <= 5e-4 and abs(delta - best["delta_meta"]) <= 5e-4
                    and float(w) > best["w_fused"])):
            best.update(w_fused=float(w), auc=auc, delta_meta=delta,
                        probs=probs, probs_nometa=probs_nm)
    return y_true, best


@torch.no_grad()
def predict_with_blend(model, loader, device, w_fused, disable_meta=False):
    y_true, p_fused, p_ecg, _ = collect_branch_probs(model, loader, device, disable_meta=disable_meta)
    return y_true, blend_probs(p_fused, p_ecg, w_fused)


# =============================================================================
# JSON export
# =============================================================================

def export_results_json(results: Dict[str, Any], output_path: Path):
    def convert(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, Path):        return str(obj)
        return obj

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {k2: convert(v2) for k2, v2 in v.items()}
        else:
            serializable[k] = convert(v)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"   Résultats exportés: {output_path}")


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    data_root:                  str   = str(DEFAULT_DATA_ROOT)
    index_path:                 str   = str(DEFAULT_INDEX_PATH)
    runs_dir:                   str   = str(DEFAULT_RUNS_DIR)
    seed:                       int   = 2026
    sampling_rate:              int   = 100
    meta_dropout_p:             float = 0.10
    batch_size:                 int   = 32
    lr:                         float = 1e-3
    epochs:                     int   = 10
    patience:                   int   = 25
    gradient_accumulation_steps: int  = 2
    max_grad_norm:              float = 1.0
    blend_candidates:           tuple = DEFAULT_BLEND_CANDIDATES
    device:                     str   = "cuda" if torch.cuda.is_available() else "cpu"
    # Extended parameters
    gate_hidden_dim:            int   = 1152   # H7: 512 / 1152 / 2048
    lauc_weight:                float = 0.08   # M3: set to 0.0 for ablation
    compute_extended_metrics:   bool  = True   # H5: always on
    hyp_subclass_analysis:      bool  = True   # H8: always on


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EZNX_ATLAS_A v5 Extended — H5/H7/H8/M3/M4"
    )
    parser.add_argument("--variant", "-variant",
                        choices=["none", "demo", "demo+anthro"],
                        default="demo+anthro")
    parser.add_argument("--seed", "-seed",       type=int,   default=2026)
    parser.add_argument("--data_root",           type=str,   default=None)
    parser.add_argument("--index_path",          type=str,   default=None)
    parser.add_argument("--runs_dir",            type=str,   default=None)
    # H7
    parser.add_argument("--gate_hidden_dim",     type=int,   default=1152,
                        help="GLU gate hidden width: 512 / 1152 / 2048 (H7)")
    # M3
    parser.add_argument("--lauc_weight",         type=float, default=0.08,
                        help="AUC-margin surrogate weight; 0.0 disables it (M3)")
    # H5 / H8 toggles (on by default)
    parser.add_argument("--no_extended_metrics", action="store_true",
                        help="Skip extended metrics (AUPRC, Brier, DeLong, ECE)")
    parser.add_argument("--no_hyp_subclass",     action="store_true",
                        help="Skip HYP/LVH subclass analysis")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed            = args.seed
    cfg.gate_hidden_dim = args.gate_hidden_dim
    cfg.lauc_weight     = args.lauc_weight
    cfg.compute_extended_metrics = not args.no_extended_metrics
    cfg.hyp_subclass_analysis    = not args.no_hyp_subclass
    if args.data_root:  cfg.data_root  = args.data_root
    if args.index_path: cfg.index_path = args.index_path
    if args.runs_dir:   cfg.runs_dir   = args.runs_dir

    set_seed(cfg.seed)
    device  = torch.device(cfg.device)
    variant = args.variant

    # Run tag encodes all varied dimensions so results never collide
    run_tag = (
        f"ATLAS_A_v5_ext_{variant}_seed{cfg.seed}"
        f"_glu{cfg.gate_hidden_dim}"
        f"_lauc{cfg.lauc_weight:.2f}"
    )
    run_dir = Path(cfg.runs_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {
        "metadata": {
            "variant":      variant,
            "seed":         cfg.seed,
            "timestamp":    datetime.now().isoformat(),
            "gate_hidden_dim": cfg.gate_hidden_dim,
            "lauc_weight":  cfg.lauc_weight,
            "config": {
                "batch_size":         cfg.batch_size,
                "lr":                 cfg.lr,
                "epochs":             cfg.epochs,
                "meta_dropout_p":     cfg.meta_dropout_p,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_grad_norm":      cfg.max_grad_norm,
            },
        },
        "validation":       {},
        "test":             {},
        "per_class":        {},
        "training_history": [],   # M4
    }

    print("=" * 80)
    print("EZNX_ATLAS_A v5 EXTENDED TRAINING")
    print("=" * 80)
    print(f"Variant:         {variant}")
    print(f"Seed:            {cfg.seed}")
    print(f"Gate hidden dim: {cfg.gate_hidden_dim}  (H7)")
    print(f"LAUC weight:     {cfg.lauc_weight:.2f}  (M3: 0.0 = ablation)")
    print(f"Extended metrics:{cfg.compute_extended_metrics}  (H5)")
    print(f"HYP subclass:    {cfg.hyp_subclass_analysis}  (H8)")
    print(f"Device:          {device}")
    print(f"Output:          {run_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    print("\n[1/5] Chargement des données...")
    train_ds = ConcatDataset([
        EZNXDataset(cfg.index_path, cfg.data_root, fold=f,
                    sampling_rate=cfg.sampling_rate, meta_mode=variant)
        for f in range(1, 9)
    ])
    val_ds  = EZNXDataset(cfg.index_path, cfg.data_root, fold=9,
                          sampling_rate=cfg.sampling_rate, meta_mode=variant)
    test_ds = EZNXDataset(cfg.index_path, cfg.data_root, fold=10,
                          sampling_rate=cfg.sampling_rate, meta_mode=variant)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn_augmented, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              collate_fn=collate_fn_val,       num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              collate_fn=collate_fn_val,       num_workers=2, pin_memory=True)

    print(f"   Train:      {len(train_ds)} samples (Folds 1-8)")
    print(f"   Validation: {len(val_ds)} samples  (Fold 9)")
    print(f"   Test:       {len(test_ds)} samples  (Fold 10)")
    all_results["metadata"]["dataset_sizes"] = {
        "train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)
    }

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    print("\n[2/5] Initialisation du modèle...")
    model = EZNX_ATLAS_A_v5(
        meta_dim=16,
        n_classes=len(DS5_LABELS),
        meta_dropout_p=cfg.meta_dropout_p,
        gate_hidden_dim=cfg.gate_hidden_dim,   # H7
    ).to(device)
    n_params = count_parameters(model)
    print(f"   Paramètres entraînables: {n_params:,}")
    print(f"   Dimensions: ECG-backbone out={model.ts.out_dim}, "
          f"fuse_dim={model.fuse_dim}, gate_hidden={model.gate_hidden_dim}")
    all_results["metadata"]["num_parameters"] = n_params

    opt       = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # 3. Loss
    # ------------------------------------------------------------------
    print("\n[3/5] Configuration de la fonction de coût...")
    pos_weights = torch.tensor([1.0, 2.3, 2.44, 2.67, 5.58], device=device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    print(f"   BCE pos_weight: {pos_weights.cpu().numpy()}")
    print(f"   LAUC weight:    {cfg.lauc_weight:.2f}"
          + ("  ← ablation (M3)" if cfg.lauc_weight == 0.0 else ""))
    all_results["metadata"]["pos_weights"] = pos_weights.cpu().numpy().tolist()

    # ------------------------------------------------------------------
    # 4. Training loop (M4: full history always saved)
    # ------------------------------------------------------------------
    print("\n[4/5] Début de l'entraînement...")
    print("-" * 80)

    best_auc        = -1.0
    best_delta_meta = -1.0
    patience_ctr    = 0
    training_history: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        opt.zero_grad()

        for batch_idx, (x_ts, x_meta, mpm, y) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100, leave=False
        )):
            x_ts, x_meta, mpm, y = (
                x_ts.to(device), x_meta.to(device), mpm.to(device), y.to(device)
            )
            out    = model(x_ts, x_meta, mpm)
            p_fused = torch.sigmoid(out["logits_fused"])

            meta_quality = torch.clamp(
                mpm[:, :2].float().mean(dim=1, keepdim=True)
                + 0.5 * mpm[:, 2:].float().mean(dim=1, keepdim=True),
                max=1.0,
            )
            meta_loss = (
                F.binary_cross_entropy_with_logits(
                    out["logits_meta"], y, reduction="none"
                ) * meta_quality
            ).mean()

            loss = (
                0.52 * criterion(out["logits_fused"], y)
                + 0.30 * criterion(out["logits_ecg"],  y)
                + 0.10 * meta_loss
                + cfg.lauc_weight * auc_margin_loss(y, p_fused)   # M3: 0.0 disables
            )
            (loss / cfg.gradient_accumulation_steps).backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        Yv, val_metrics = select_best_val_blend(
            model, val_loader, device, blend_candidates=cfg.blend_candidates
        )
        Pv       = val_metrics["probs"]
        auc_v    = safe_macro_auroc(Yv, Pv)
        f1_v     = f1_score(Yv, (Pv >= 0.5), average="macro", zero_division=0)
        delta_v  = float(val_metrics["delta_meta"])
        avg_loss = train_loss / len(train_loader)

        # M4: save full per-epoch history
        training_history.append({
            "epoch":         epoch,
            "train_loss":    avg_loss,
            "val_auc":       auc_v,
            "val_f1":        f1_v,
            "val_auc_fused": safe_macro_auroc(Yv, val_metrics["p_fused"]),
            "val_auc_ecg":   safe_macro_auroc(Yv, val_metrics["p_ecg"]),
            "val_delta_meta":delta_v,
            "w_fused":       val_metrics["w_fused"],
            "lr":            opt.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
            f"AUC_Val: {auc_v:.4f} | F1_Val: {f1_v:.4f} | "
            f"ΔMeta: {delta_v:+.4f} | w_fused: {val_metrics['w_fused']:.2f}"
        )

        # Checkpoint
        is_better = auc_v > best_auc + 1e-6
        is_tie    = (abs(auc_v - best_auc) <= 5e-4
                     and delta_v > best_delta_meta + 1e-6)
        if is_better or is_tie:
            best_auc        = auc_v
            best_delta_meta = delta_v
            patience_ctr    = 0
            thr_val         = find_optimal_thresholds(Yv, Pv)
            ckpt_path       = run_dir / f"best_model_{run_tag}.pt"
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "thresholds":        thr_val,
                "w_fused":           val_metrics["w_fused"],
                "best_auc":          best_auc,
                "best_delta_meta":   best_delta_meta,
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "seed":              cfg.seed,
                "gate_hidden_dim":   cfg.gate_hidden_dim,
                "lauc_weight":       cfg.lauc_weight,
            }, ckpt_path)
            print(f"   [BEST] Nouveau record AUC: {best_auc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"\n   Early stopping à l'epoch {epoch}")
                break

    all_results["training_history"]          = training_history
    all_results["validation"]["best_auc"]    = best_auc
    all_results["validation"]["best_delta_meta"] = best_delta_meta
    all_results["validation"]["best_epoch"]  = training_history[
        int(np.argmax([h["val_auc"] for h in training_history]))
    ]["epoch"]

    # ------------------------------------------------------------------
    # 5. Final evaluation on fold-10
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[5/5] ÉVALUATION FINALE (FOLD 10 — TEST)")
    print("=" * 80)

    ckpt_path = run_dir / f"best_model_{run_tag}.pt"
    if not ckpt_path.exists():
        print(f"ERREUR: Checkpoint introuvable: {ckpt_path}")
        all_results["test"]["error"] = "Checkpoint not found"
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        thr_final    = ckpt.get("thresholds", np.full(len(DS5_LABELS), 0.5))
        w_fused_final = float(ckpt.get("w_fused", 1.0))

        print(f"   Seuils: {thr_final}")
        print(f"   w_fused: {w_fused_final:.2f}")

        Yt, Pt         = predict_with_blend(model, test_loader, device, w_fused=w_fused_final)
        _, Pt_no_meta  = predict_with_blend(model, test_loader, device, w_fused=w_fused_final, disable_meta=True)
        _, Pt_ecg      = predict_with_blend(model, test_loader, device, w_fused=0.0)
        _, Pt_fused    = predict_with_blend(model, test_loader, device, w_fused=1.0)

        auc_m, f1_m, aucs, f1s = compute_metrics_per_class(Yt, Pt, thr_final)
        auc_no_meta     = safe_macro_auroc(Yt, Pt_no_meta)
        delta_meta_auc  = auc_m - auc_no_meta
        f1_fixed        = f1_score(Yt, (Pt >= 0.5), average="macro", zero_division=0)

        all_results["test"] = {
            "macro_auc":          auc_m,
            "macro_f1_optimal":   f1_m,
            "macro_f1_fixed":     f1_fixed,
            "auc_ecg_only":       safe_macro_auroc(Yt, Pt_ecg),
            "auc_fused_only":     safe_macro_auroc(Yt, Pt_fused),
            "auc_meta_disabled":  auc_no_meta,
            "delta_meta_auc":     delta_meta_auc,
            "w_fused":            w_fused_final,
            "thresholds":         thr_final.tolist(),
        }
        for i, lbl in enumerate(DS5_LABELS):
            all_results["per_class"][lbl] = {
                "auc":       aucs[i],
                "f1":        f1s[i],
                "threshold": float(thr_final[i]),
            }

        print(f"\nMacro AUC Test: {auc_m:.4f}")
        print(f"Delta AUC (meta active): {delta_meta_auc:+.4f}")
        print(f"{'Class':<8} | {'AUC':<8} | {'F1':<8} | {'Thr':<6}")
        print("-" * 38)
        for i, lbl in enumerate(DS5_LABELS):
            print(f"{lbl:<8} | {aucs[i]:<8.4f} | {f1s[i]:<8.4f} | {thr_final[i]:<6.3f}")

        # H5 – Extended metrics
        if cfg.compute_extended_metrics:
            print("\n--- H5: Extended metrics (AUPRC / Brier / DeLong CI / ECE) ---")
            ext = compute_extended_metrics(Yt, Pt)
            all_results["extended_metrics"] = ext
            print(f"Macro AUPRC: {ext['macro_auprc']:.4f}")
            print(f"Macro Brier: {ext['macro_brier']:.4f}")
            print(f"Macro ECE:   {ext['macro_ece']:.4f}")
            print(f"\n{'Class':<8} | {'AUPRC':<8} | {'Brier':<8} | "
                  f"{'AUC_DL':<8} | {'CI_lo':<8} | {'CI_hi':<8} | {'ECE':<8}")
            print("-" * 68)
            for lbl in DS5_LABELS:
                m = ext["per_class"][lbl]
                print(
                    f"{lbl:<8} | {m['auprc']:<8.4f} | {m['brier']:<8.4f} | "
                    f"{m['auc_delong']:<8.4f} | {m['auc_delong_ci_lo']:<8.4f} | "
                    f"{m['auc_delong_ci_hi']:<8.4f} | {m['ece']:<8.4f}"
                )

        # H8 – HYP / LVH subclass
        if cfg.hyp_subclass_analysis and variant in ("demo+anthro", "demo"):
            print("\n--- H8: HYP/LVH subclass analysis ---")
            lvh_res = compute_hyp_lvh_subclass(Yt, Pt, cfg.index_path)
            all_results["hyp_lvh_subclass"] = lvh_res
            if "error" in lvh_res:
                print(f"   Warning: {lvh_res['error']}")
            else:
                print(f"   LVH records (fold-10): {lvh_res['n_lvh_records']}")
                print(f"   HYP-positive among LVH: {lvh_res['n_lvh_hyp_positive']}")
                print(f"   AUC (LVH subset):  {lvh_res['auc_lvh']:.4f}  "
                      f"95% CI [{lvh_res['auc_lvh_delong_ci_lo']:.4f}, "
                      f"{lvh_res['auc_lvh_delong_ci_hi']:.4f}]")
                print(f"   AUPRC (LVH subset): {lvh_res['auprc_lvh']:.4f}")
                print(f"   AUC (all HYP):      {lvh_res['auc_hyp_all_records']:.4f}  "
                      f"(reference)")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    results_path = run_dir / f"results_ext_{run_tag}.json"
    export_results_json(all_results, results_path)

    print(f"\n{'=' * 80}")
    print(f"[OK] Terminé  seed={cfg.seed}  variant={variant}  "
          f"glu={cfg.gate_hidden_dim}  lauc={cfg.lauc_weight:.2f}")
    print(f"  Modèle:    {ckpt_path}")
    print(f"  Résultats: {results_path}")
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    main()
