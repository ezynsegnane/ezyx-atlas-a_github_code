"""
atlas_a_v5_multiseed_v2.py — EZNX-ATLAS-A Scientific-Reports training script.

Handles all experiment types via CLI flags:

    python atlas_a_v5_multiseed_v2.py --variant demo+anthro --seed 2024
    python atlas_a_v5_multiseed_v2.py --variant demo+anthro --seed 2024 --meta_hid 64
    python atlas_a_v5_multiseed_v2.py --variant demo+anthro --seed 2024 --lauc_weight 0.0
    python atlas_a_v5_multiseed_v2.py --variant demo+anthro --seed 2024 --no_aug

Key improvements over v1:
  • CUBLAS + cuDNN fully-deterministic flags (reproducible to bit level on same GPU+env)
  • Dynamic pos_weights computed from training-fold label frequencies at runtime
  • num_workers=0 in all DataLoaders (eliminates multi-process non-determinism)
  • NPZ dump of all branch probabilities + patient IDs after test evaluation
  • Inline subgroup AUC: sex (M/F) and age group (<45 / 45-65 / >65 yrs)
  • Hardware provenance logged in results JSON
  • Auto-resume: skip silently if results JSON already exists (idempotent)
"""

# ── Set deterministic CUBLAS workspace BEFORE any torch import ──────────────
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import random
import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eznx_loader_v2 import EZNXDataset, DS5_LABELS
from eznx_model_v5 import EZNX_ATLAS_A_v5, count_parameters

# ── Apply deterministic algorithm flags after torch is imported ──────────────
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

# PTB-XL population statistics for age group decoding
# Source: PTB-XL paper (Wagner et al. 2020): mean=62.5 yr, SD=17.2 yr
_PTB_AGE_MEAN = 62.5
_PTB_AGE_SD = 17.2

DEFAULT_BLEND_CANDIDATES = (0.0, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0)

# Environment-based default paths — overridden by CLI or Kaggle env
DEFAULT_DATA_ROOT = Path(os.getenv(
    "EZNX_DATA_REAL",
    str(PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3")
))
DEFAULT_INDEX_PATH = Path(os.getenv(
    "EZNX_INDEX_PATH",
    str(PROJECT_ROOT / "data" / "index_complete.parquet")
))
DEFAULT_RUNS_DIR = Path(os.getenv(
    "EZNX_RUNS_DIR",
    str(PROJECT_ROOT / "runs")
))


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Lock all random-number generators for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cudnn flags already set at module level


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic positive-class weights
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pos_weights(
    train_datasets: list,
    n_classes: int = 5,
    clip: Tuple[float, float] = (0.5, 30.0),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute BCEWithLogitsLoss pos_weight from training-fold label prevalences.

    For each class j:  pos_weight_j = neg_j / pos_j

    Clipped to [0.5, 30.0] to prevent extreme gradients for near-absent classes.
    Values are logged so they appear in the results JSON.
    """
    Y_list = [ds.y for ds in train_datasets]
    Y = np.concatenate(Y_list, axis=0)          # (N_train, n_classes)
    pos = Y.sum(axis=0)                           # prevalence per class
    neg = len(Y) - pos
    raw_weights = neg / np.maximum(pos, 1)
    weights = np.clip(raw_weights, *clip).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# ECG augmentation + collate functions
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_ts_voltage(x_ts: torch.Tensor) -> torch.Tensor:
    return x_ts / 5.0


class ECGAugmentation:
    """On-the-fly augmentation for 12-lead ECG signals."""

    @staticmethod
    def add_gaussian_noise(x: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
        if np.random.rand() > 0.3:
            return x
        return x + torch.randn_like(x) * noise_level

    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 20) -> torch.Tensor:
        if np.random.rand() > 0.3:
            return x
        shift = np.random.randint(-max_shift, max_shift)
        return torch.roll(x, shift, dims=-1) if shift != 0 else x

    @staticmethod
    def amplitude_scale(x: torch.Tensor,
                        scale_range: Tuple[float, float] = (0.95, 1.05)) -> torch.Tensor:
        if np.random.rand() > 0.3:
            return x
        return x * np.random.uniform(*scale_range)


def collate_fn_augmented(items):
    x_ts = torch.stack([it["x_ts"] for it in items])
    x_meta = torch.stack([it["x_meta"] for it in items])
    mpm = torch.stack([it["meta_present_mask"] for it in items])
    y = torch.stack([it["y"] for it in items])
    x_ts = normalize_ts_voltage(x_ts)
    if np.random.rand() > 0.5:
        x_ts = ECGAugmentation.add_gaussian_noise(x_ts)
        x_ts = ECGAugmentation.time_shift(x_ts)
        x_ts = ECGAugmentation.amplitude_scale(x_ts)
    return x_ts, x_meta, mpm, y


def collate_fn_val(items):
    x_ts = torch.stack([it["x_ts"] for it in items])
    x_meta = torch.stack([it["x_meta"] for it in items])
    mpm = torch.stack([it["meta_present_mask"] for it in items])
    y = torch.stack([it["y"] for it in items])
    x_ts = normalize_ts_voltage(x_ts)
    return x_ts, x_meta, mpm, y


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def safe_macro_auroc(Y: np.ndarray, P: np.ndarray) -> float:
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) < 2:
            continue
        aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def safe_auc_per_class(Y: np.ndarray, P: np.ndarray) -> List[float]:
    """Per-class AUC; NaN if only one label present."""
    return [
        float(roc_auc_score(Y[:, j], P[:, j]))
        if len(np.unique(Y[:, j])) >= 2
        else float("nan")
        for j in range(Y.shape[1])
    ]


def find_optimal_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    thr = np.full(Y.shape[1], 0.5, dtype=np.float32)
    for j in range(Y.shape[1]):
        best, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 61):
            f1 = f1_score(Y[:, j], (P[:, j] >= t).astype(np.int32), zero_division=0)
            if f1 > best:
                best, best_t = f1, t
        thr[j] = best_t
    return thr


def compute_metrics_per_class(Y: np.ndarray, P: np.ndarray,
                               thr: np.ndarray) -> Tuple[float, float,
                                                          List[float], List[float]]:
    aucs = safe_auc_per_class(Y, P)
    f1s = [
        float(f1_score(Y[:, j], (P[:, j] >= thr[j]).astype(np.int32), zero_division=0))
        for j in range(Y.shape[1])
    ]
    return float(np.nanmean(aucs)), float(np.nanmean(f1s)), aucs, f1s


# ═══════════════════════════════════════════════════════════════════════════════
# Subgroup AUC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_subgroup_aucs(
    Y: np.ndarray,
    P: np.ndarray,
    test_df,   # pd.DataFrame with age_z, sex01
) -> Dict[str, Any]:
    """
    Return macro-AUC for sex and age subgroups.

    Age-group thresholds derived from PTB-XL population statistics:
      mean=62.5 yr, SD=17.2 yr  (Wagner et al. 2020)
      < 45 yr  → age_z < (45 − 62.5)/17.2 = −1.02
      45–65 yr → −1.02 ≤ age_z < (65 − 62.5)/17.2 = 0.145
      > 65 yr  → age_z ≥ 0.145
    """
    age_z = test_df["age_z"].values
    sex01 = test_df["sex01"].values

    AGE_LT45_Z  = (45  - _PTB_AGE_MEAN) / _PTB_AGE_SD   # ≈ −1.02
    AGE_GT65_Z  = (65  - _PTB_AGE_MEAN) / _PTB_AGE_SD   # ≈  0.145

    masks = {
        "sex_male":   sex01 == 1,
        "sex_female": sex01 == 0,
        "age_lt45":   age_z < AGE_LT45_Z,
        "age_45_65":  (age_z >= AGE_LT45_Z) & (age_z < AGE_GT65_Z),
        "age_gt65":   age_z >= AGE_GT65_Z,
    }

    result: Dict[str, Any] = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        result[name] = {"n": n}
        if n >= 10:
            result[name]["macro_auc"] = safe_macro_auroc(Y[mask], P[mask])
            result[name]["per_class_auc"] = safe_auc_per_class(Y[mask], P[mask])
        else:
            result[name]["macro_auc"] = None
            result[name]["per_class_auc"] = None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AUC margin loss
# ═══════════════════════════════════════════════════════════════════════════════

def auc_margin_loss(y_true: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Pairwise AUC surrogate margin loss (batch-level approximation)."""
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=p.device)
    return torch.mean((1 - pos.unsqueeze(1) + neg.unsqueeze(0)).clamp(min=0))


# ═══════════════════════════════════════════════════════════════════════════════
# Branch-probability collection
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_branch_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    disable_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps_fused, ps_ecg, ps_meta = [], [], [], []
    for x_ts, x_meta, mpm, y in loader:
        x_ts, x_meta, mpm = x_ts.to(device), x_meta.to(device), mpm.to(device)
        if disable_meta:
            x_meta = torch.zeros_like(x_meta)
            mpm = torch.zeros_like(mpm)
        out = model(x_ts, x_meta, mpm)
        ps_fused.append(torch.sigmoid(out["logits_fused"]).cpu().numpy())
        ps_ecg.append(torch.sigmoid(out["logits_ecg"]).cpu().numpy())
        ps_meta.append(torch.sigmoid(out["logits_meta"]).cpu().numpy())
        ys.append(y.numpy())
    return (
        np.concatenate(ys),
        np.concatenate(ps_fused),
        np.concatenate(ps_ecg),
        np.concatenate(ps_meta),
    )


def blend_probs(p_fused: np.ndarray, p_ecg: np.ndarray,
                w_fused: float) -> np.ndarray:
    return w_fused * p_fused + (1.0 - w_fused) * p_ecg


@torch.no_grad()
def select_best_val_blend(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    blend_candidates=None,
) -> Tuple[np.ndarray, Dict]:
    y_true, p_fused, p_ecg, p_meta = collect_branch_probs(model, loader, device)
    _, p_fused_nm, p_ecg_nm, _ = collect_branch_probs(
        model, loader, device, disable_meta=True
    )
    best = {
        "w_fused": 1.0, "auc": -1.0, "delta_meta": -1.0,
        "probs": p_fused, "probs_nometa": p_fused_nm,
        "p_fused": p_fused, "p_ecg": p_ecg, "p_meta": p_meta,
    }
    if blend_candidates is None:
        blend_candidates = DEFAULT_BLEND_CANDIDATES
    for w in blend_candidates:
        probs = blend_probs(p_fused, p_ecg, float(w))
        probs_nm = blend_probs(p_fused_nm, p_ecg_nm, float(w))
        auc = safe_macro_auroc(y_true, probs)
        auc_nm = safe_macro_auroc(y_true, probs_nm)
        delta = auc - auc_nm
        if (auc > best["auc"] + 1e-6
                or (abs(auc - best["auc"]) <= 5e-4 and delta > best["delta_meta"] + 1e-6)
                or (abs(auc - best["auc"]) <= 5e-4
                    and abs(delta - best["delta_meta"]) <= 5e-4
                    and float(w) > best["w_fused"])):
            best.update(w_fused=float(w), auc=auc, delta_meta=delta,
                        probs=probs, probs_nometa=probs_nm)
    return y_true, best


@torch.no_grad()
def predict_with_blend(
    model: nn.Module, loader: DataLoader, device: torch.device,
    w_fused: float, disable_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true, p_fused, p_ecg, _ = collect_branch_probs(
        model, loader, device, disable_meta=disable_meta
    )
    return y_true, blend_probs(p_fused, p_ecg, w_fused)


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware provenance
# ═══════════════════════════════════════════════════════════════════════════════

def get_hardware_provenance() -> Dict[str, Any]:
    prov: Dict[str, Any] = {
        "python_version": sys.version,
        "torch_version":  torch.__version__,
        "cuda_available":  torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        prov["cuda_version"]  = torch.version.cuda
        prov["gpu_name"]      = torch.cuda.get_device_name(0)
        prov["gpu_count"]     = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        prov["gpu_total_memory_mb"] = props.total_memory // (1024 * 1024)
    try:
        import numpy as np  # noqa: F401
        prov["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import sklearn
        prov["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    return prov


# ═══════════════════════════════════════════════════════════════════════════════
# JSON export
# ═══════════════════════════════════════════════════════════════════════════════

def _convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def export_results_json(results: Dict[str, Any], output_path: Path) -> None:
    def _deep_convert(v):
        if isinstance(v, dict):
            return {k2: _deep_convert(v2) for k2, v2 in v.items()}
        if isinstance(v, (list, tuple)):
            return [_deep_convert(i) for i in v]
        return _convert(v)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_deep_convert(results), f, indent=2, ensure_ascii=False)
    print(f"   Results saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Paths (overrideable by env or CLI)
    data_root:   str = str(DEFAULT_DATA_ROOT)
    index_path:  str = str(DEFAULT_INDEX_PATH)
    runs_dir:    str = str(DEFAULT_RUNS_DIR)

    # Experiment identity
    seed:        int   = 2024
    variant:     str   = "demo+anthro"   # none | demo | demo+anthro
    meta_hid:    int   = 128             # hidden dim of meta_fuse MLP
    lauc_weight: float = 0.08           # weight of AUC-margin loss term
    no_aug:      bool  = False           # disable ECG augmentation if True

    # Architecture
    sampling_rate:    int   = 100
    meta_dropout_p:   float = 0.10

    # Optimisation
    batch_size:                 int   = 32
    lr:                         float = 1e-3
    epochs:                     int   = 10
    patience:                   int   = 25
    gradient_accumulation_steps: int  = 2
    max_grad_norm:              float = 1.0

    # Inference
    blend_candidates: Tuple[float, ...] = DEFAULT_BLEND_CANDIDATES

    # Runtime (filled automatically)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_run_name(cfg: Config) -> str:
    """
    Deterministic run-directory name that encodes all experiment-type flags.

    Examples
    --------
    ATLAS_A_v5_demo+anthro_seed2024
    ATLAS_A_v5_demo+anthro_metaH64_seed2024
    ATLAS_A_v5_demo+anthro_lauc0.16_seed2024
    ATLAS_A_v5_demo+anthro_noaug_seed2024
    """
    parts = [f"ATLAS_A_v5_{cfg.variant}"]
    if cfg.meta_hid != 128:
        parts.append(f"metaH{cfg.meta_hid}")
    if abs(cfg.lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{cfg.lauc_weight:g}")
    if cfg.no_aug:
        parts.append("noaug")
    parts.append(f"seed{cfg.seed}")
    return "_".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="EZNX-ATLAS-A v2 — Scientific Reports training script"
    )
    parser.add_argument("--variant",      type=str,   default="demo+anthro",
                        choices=["none", "demo", "demo+anthro"])
    parser.add_argument("--seed",         type=int,   default=2024)
    parser.add_argument("--meta_hid",     type=int,   default=128,
                        help="Hidden dim of metadata fusion MLP (default: 128)")
    parser.add_argument("--lauc_weight",  type=float, default=0.08,
                        help="Weight of AUC-margin loss term (default: 0.08)")
    parser.add_argument("--no_aug",       action="store_true",
                        help="Disable ECG data augmentation")
    parser.add_argument("--data_root",    type=str,   default=None)
    parser.add_argument("--index_path",   type=str,   default=None)
    parser.add_argument("--runs_dir",     type=str,   default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.seed        = args.seed
    cfg.variant     = args.variant
    cfg.meta_hid    = args.meta_hid
    cfg.lauc_weight = args.lauc_weight
    cfg.no_aug      = args.no_aug
    if args.data_root:
        cfg.data_root = args.data_root
    if args.index_path:
        cfg.index_path = args.index_path
    if args.runs_dir:
        cfg.runs_dir = args.runs_dir

    device = torch.device(cfg.device)

    # ── Run directory & auto-resume ──────────────────────────────────────────
    run_name = make_run_name(cfg)
    run_dir  = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / f"results_{run_name}.json"
    if results_path.exists():
        print(f"[AUTO-RESUME] Already complete: {results_path} — skipping.")
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)

    # ── Seed & provenance ────────────────────────────────────────────────────
    set_seed(cfg.seed)
    hw = get_hardware_provenance()

    print("=" * 80)
    print("EZNX-ATLAS-A v2 — Scientific Reports Training")
    print("=" * 80)
    print(f"  Variant:     {cfg.variant}")
    print(f"  Seed:        {cfg.seed}")
    print(f"  meta_hid:    {cfg.meta_hid}")
    print(f"  lauc_weight: {cfg.lauc_weight}")
    print(f"  no_aug:      {cfg.no_aug}")
    print(f"  Device:      {device}  ({hw.get('gpu_name', 'CPU')})")
    print(f"  Output:      {run_dir}")
    print("=" * 80)

    wall_start = time.time()

    # ── 1. Datasets ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")
    train_datasets = [
        EZNXDataset(
            index_file=cfg.index_path, data_root=cfg.data_root,
            fold=f, sampling_rate=cfg.sampling_rate, meta_mode=cfg.variant
        )
        for f in range(1, 9)
    ]
    train_ds = ConcatDataset(train_datasets)

    val_ds = EZNXDataset(
        index_file=cfg.index_path, data_root=cfg.data_root,
        fold=9, sampling_rate=cfg.sampling_rate, meta_mode=cfg.variant
    )
    test_ds = EZNXDataset(
        index_file=cfg.index_path, data_root=cfg.data_root,
        fold=10, sampling_rate=cfg.sampling_rate, meta_mode=cfg.variant
    )

    # num_workers=0: fully deterministic, no multi-process shuffle ambiguity
    _lkw = dict(num_workers=0, pin_memory=(cfg.device == "cuda"))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn_val if cfg.no_aug else collate_fn_augmented,
        **_lkw
    )
    val_loader  = DataLoader(val_ds,  batch_size=cfg.batch_size,
                             collate_fn=collate_fn_val,  **_lkw)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             collate_fn=collate_fn_val, **_lkw)

    n_train = len(train_ds)
    n_val   = len(val_ds)
    n_test  = len(test_ds)
    print(f"   Train: {n_train}  Val: {n_val}  Test: {n_test}")

    # ── 2. Pos-weights (dynamic from training fold) ──────────────────────────
    print("\n[2/6] Computing pos_weights from training fold …")
    pos_weights = compute_pos_weights(train_datasets, n_classes=5, device=device)
    print(f"   pos_weights = {pos_weights.cpu().numpy().round(4).tolist()}")

    # ── 3. Model ─────────────────────────────────────────────────────────────
    print("\n[3/6] Initialising model …")
    model = EZNX_ATLAS_A_v5(
        meta_dim=16,
        n_classes=len(DS5_LABELS),
        meta_hid=cfg.meta_hid,
        meta_dropout_p=cfg.meta_dropout_p,
    ).to(device)

    n_params = count_parameters(model)
    print(f"   Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── 4. Loss ──────────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # Loss weights (must sum to ~1): fused + ecg + meta + auc_margin
    #   Default: 0.52 + 0.30 + 0.10 + 0.08 = 1.00
    lauc_w  = cfg.lauc_weight
    fused_w = max(0.0, 0.60 - lauc_w)   # absorb lauc delta into fused term
    ecg_w   = 0.30
    meta_w  = 0.10
    # Renormalise so sum=1.0
    total_w = fused_w + ecg_w + meta_w + lauc_w
    fused_w /= total_w; ecg_w /= total_w; meta_w /= total_w; lauc_w_n = lauc_w / total_w

    print(f"   Loss weights — fused:{fused_w:.3f} ecg:{ecg_w:.3f} "
          f"meta:{meta_w:.3f} lauc:{lauc_w_n:.3f}")

    # ── 5. Training loop ─────────────────────────────────────────────────────
    print("\n[4/6] Training …")
    print("-" * 80)

    best_auc         = -1.0
    best_delta_meta  = -1.0
    patience_ctr     = 0
    history          = []

    ckpt_path = run_dir / f"best_model_{run_name}.pt"

    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        opt.zero_grad()

        for batch_idx, (x_ts, x_meta, mpm, y) in enumerate(tqdm(
            train_loader, desc=f"Ep {epoch:3d}/{cfg.epochs}", ncols=100, leave=False
        )):
            x_ts, x_meta, mpm, y = (
                x_ts.to(device), x_meta.to(device),
                mpm.to(device),  y.to(device)
            )
            out   = model(x_ts, x_meta, mpm)
            p_f   = torch.sigmoid(out["logits_fused"])

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
                fused_w  * criterion(out["logits_fused"], y)
                + ecg_w  * criterion(out["logits_ecg"],   y)
                + meta_w * meta_loss
                + lauc_w_n * auc_margin_loss(y, p_f)
            )
            (loss / cfg.gradient_accumulation_steps).backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad()

            train_loss += loss.item()

        # Trailing micro-batch flush
        n_batches = len(train_loader)
        if n_batches % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad()

        scheduler.step()

        # Validation
        Yv, vm = select_best_val_blend(model, val_loader, device, cfg.blend_candidates)
        Pv        = vm["probs"]
        auc_v     = safe_macro_auroc(Yv, Pv)
        f1_v      = f1_score(Yv, (Pv >= 0.5), average="macro", zero_division=0)
        delta_v   = float(vm["delta_meta"])
        avg_loss  = train_loss / n_batches
        ep_time   = time.time() - epoch_t0

        rec = dict(
            epoch=epoch, train_loss=avg_loss, val_auc=auc_v, val_f1=f1_v,
            val_auc_fused=safe_macro_auroc(Yv, vm["p_fused"]),
            val_auc_ecg=safe_macro_auroc(Yv, vm["p_ecg"]),
            val_delta_meta=delta_v, w_fused=vm["w_fused"],
            lr=opt.param_groups[0]["lr"], epoch_time_s=ep_time,
        )
        history.append(rec)
        print(f"Ep {epoch:3d} | loss={avg_loss:.4f} | AUC={auc_v:.4f} "
              f"| F1={f1_v:.4f} | Δmeta={delta_v:+.4f} "
              f"| w={vm['w_fused']:.2f} | {ep_time:.0f}s")

        # Checkpoint
        is_better = (auc_v > best_auc + 1e-6) or (
            abs(auc_v - best_auc) <= 5e-4
            and delta_v > best_delta_meta + 1e-6
        )
        if is_better:
            best_auc = auc_v
            best_delta_meta = delta_v
            patience_ctr = 0
            thr_val = find_optimal_thresholds(Yv, Pv)
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "thresholds": thr_val, "w_fused": vm["w_fused"],
                "best_auc": best_auc, "seed": cfg.seed,
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)
            print(f"   ★ New best AUC: {best_auc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"\n   Early stopping after epoch {epoch}")
                break

    wall_train = time.time() - wall_start

    # ── 6. Test evaluation ───────────────────────────────────────────────────
    print("\n[5/6] Final evaluation on Test fold (fold 10) …")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    thr_final   = ckpt.get("thresholds", np.full(len(DS5_LABELS), 0.5))
    w_f_final   = float(ckpt.get("w_fused", 1.0))

    # Collect ALL branch probs in one pass (deterministic)
    Yt, p_fused_t, p_ecg_t, p_meta_t = collect_branch_probs(
        model, test_loader, device
    )
    _, p_fused_nm_t, p_ecg_nm_t, _ = collect_branch_probs(
        model, test_loader, device, disable_meta=True
    )

    Pt       = blend_probs(p_fused_t,    p_ecg_t,    w_f_final)
    Pt_nm    = blend_probs(p_fused_nm_t, p_ecg_nm_t, w_f_final)
    Pt_ecg   = blend_probs(p_fused_t,    p_ecg_t,    0.0)   # ECG branch only
    Pt_fused = blend_probs(p_fused_t,    p_ecg_t,    1.0)   # fused branch only

    auc_m,  f1_m,  aucs,  f1s  = compute_metrics_per_class(Yt, Pt, thr_final)
    auc_nm  = safe_macro_auroc(Yt, Pt_nm)
    auc_ecg = safe_macro_auroc(Yt, Pt_ecg)
    auc_fus = safe_macro_auroc(Yt, Pt_fused)
    f1_fixed = float(f1_score(Yt, (Pt >= 0.5), average="macro", zero_division=0))

    # ── GPU memory peak ──────────────────────────────────────────────────────
    gpu_mem_peak_mb = (
        torch.cuda.max_memory_allocated() // (1024 * 1024)
        if torch.cuda.is_available() else 0
    )

    # ── Subgroup AUC ─────────────────────────────────────────────────────────
    print("\n[6/6] Computing subgroup AUCs …")
    import pandas as pd
    test_df_raw = pd.read_parquet(cfg.index_path)
    test_df_raw = test_df_raw[test_df_raw["strat_fold"] == 10].reset_index(drop=True)
    subgroup_aucs = compute_subgroup_aucs(Yt, Pt, test_df_raw)

    # ── NPZ dump ─────────────────────────────────────────────────────────────
    npz_path = run_dir / f"probs_{run_name}.npz"
    ecg_ids  = test_df_raw["ecg_id"].values
    pat_ids  = test_df_raw["patient_id"].values
    np.savez_compressed(
        npz_path,
        Y=Yt,
        P_fused=p_fused_t,
        P_ecg=p_ecg_t,
        P_meta=p_meta_t,
        P_blend=Pt,
        P_no_meta=Pt_nm,
        ecg_id=ecg_ids,
        patient_id=pat_ids,
        labels=np.array(DS5_LABELS),
        w_fused=np.array([w_f_final]),
        thresholds=thr_final,
    )
    print(f"   NPZ probs saved → {npz_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    wall_total = time.time() - wall_start
    print("\n" + "=" * 80)
    print(f"RESULTS — Seed: {cfg.seed} | Variant: {cfg.variant}")
    print("=" * 80)
    print(f"Macro AUC (blend):      {auc_m:.4f}")
    print(f"Macro AUC (ECG only):   {auc_ecg:.4f}")
    print(f"Macro AUC (fused only): {auc_fus:.4f}")
    print(f"Macro AUC (no meta):    {auc_nm:.4f}")
    print(f"Delta AUC (meta):       {auc_m - auc_nm:+.4f}")
    print(f"Macro F1 (optimal thr): {f1_m:.4f}")
    print(f"Macro F1 (thr=0.5):     {f1_fixed:.4f}")
    print("-" * 50)
    print(f"{'Class':<10} | {'AUC':>8} | {'F1':>8} | {'Thr':>6}")
    for i, lbl in enumerate(DS5_LABELS):
        print(f"{lbl:<10} | {aucs[i]:>8.4f} | {f1s[i]:>8.4f} | {thr_final[i]:>6.3f}")
    print("=" * 80)
    print(f"Wall time: {wall_total/60:.1f} min  |  GPU peak: {gpu_mem_peak_mb} MB")

    # ── Assemble results dict ─────────────────────────────────────────────────
    results = {
        "metadata": {
            "run_name":    run_name,
            "variant":     cfg.variant,
            "seed":        cfg.seed,
            "meta_hid":    cfg.meta_hid,
            "lauc_weight": cfg.lauc_weight,
            "no_aug":      cfg.no_aug,
            "timestamp":   datetime.now().isoformat(),
            "wall_time_s": wall_total,
            "train_time_s": wall_train,
            "gpu_peak_mem_mb": gpu_mem_peak_mb,
            "num_parameters": n_params,
            "dataset_sizes": {"train": n_train, "val": n_val, "test": n_test},
            "pos_weights": pos_weights.cpu().numpy().tolist(),
            "loss_weights": {
                "fused": fused_w, "ecg": ecg_w,
                "meta": meta_w, "lauc": lauc_w_n
            },
            "hardware": hw,
            "best_val_auc": best_auc,
            "best_val_epoch": max(
                history, key=lambda h: h["val_auc"]
            )["epoch"] if history else -1,
        },
        "test": {
            "macro_auc":          auc_m,
            "macro_auc_ecg":      auc_ecg,
            "macro_auc_fused":    auc_fus,
            "macro_auc_no_meta":  auc_nm,
            "delta_meta_auc":     auc_m - auc_nm,
            "macro_f1_optimal":   f1_m,
            "macro_f1_fixed_05":  f1_fixed,
            "w_fused":            w_f_final,
            "thresholds":         thr_final.tolist(),
        },
        "per_class": {
            lbl: {"auc": aucs[i], "f1": f1s[i], "threshold": float(thr_final[i])}
            for i, lbl in enumerate(DS5_LABELS)
        },
        "subgroups": subgroup_aucs,
        "training_history": history,
    }

    export_results_json(results, results_path)
    print(f"\n✓ Run complete: {run_name}")
    return results


if __name__ == "__main__":
    main()
