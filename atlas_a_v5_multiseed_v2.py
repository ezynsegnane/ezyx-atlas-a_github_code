"""
atlas_a_v5_multiseed_v2.py — EZNX-ATLAS-A v5 training script.

Handles all experiment groups via CLI flags:

    Group A (primary)
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --seed 2024

    Group B (fusion capacity)
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --meta_hid 64 --seed 2024

    Group C (LAUC-BCE balance)
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --lauc_weight 0.12 --seed 2024

    Group D (augmentation sensitivity)
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --no_aug --seed 2024

    Group E (architecture ablations)
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --arch_mode no_glu --seed 2024
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --meta_mask_mode trainmask --seed 2024

    Group F (multi-split sensitivity)
        python atlas_a_v5_multiseed_v2.py --variant demo+anthro --test_fold 7 --val_fold 8 --seed 2024

Design decisions (pre-declared in analysis_plan.md):
  • w_fused = 1.0 always — no blending search, no val-fold optimisation
  • Checkpoint selection = val macro-AUC only (no delta_meta tie-break)
  • pos_weights computed dynamically from training folds (not hardcoded)
  • num_workers=0 in all DataLoaders (deterministic)
  • CUBLAS env var set before any torch import
"""

# ── Set deterministic CUBLAS workspace BEFORE any torch import ──────────────
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import hashlib
import json
import random
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
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

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

from eznx_loader_v2 import EZNXDataset, DS5_LABELS
from eznx_model_v5 import EZNX_ATLAS_A_v5, count_parameters, VALID_ARCH_MODES

# ── Apply deterministic algorithm flags after torch is imported ──────────────
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False   # disable TF32 on matmul
torch.backends.cudnn.allow_tf32       = False   # disable TF32 on cuDNN (Ampere+)

# PTB-XL population statistics for age-group subgroup decoding
# Source: Wagner et al. 2020 — mean=62.5 yr, SD=17.2 yr
_PTB_AGE_MEAN = 62.5
_PTB_AGE_SD   = 17.2

# Environment-based default paths — overridden by CLI or Kaggle env vars
DEFAULT_DATA_ROOT  = Path(os.getenv("EZNX_DATA_REAL",   str(PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3")))
DEFAULT_INDEX_PATH = Path(os.getenv("EZNX_INDEX_PATH",  str(PROJECT_ROOT / "data" / "index_complete.parquet")))
DEFAULT_RUNS_DIR   = Path(os.getenv("EZNX_RUNS_DIR",    str(PROJECT_ROOT / "runs")))


# ═══════════════════════════════════════════════════════════════════════════════
# Provenance helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_git_sha() -> str:
    """Return HEAD commit SHA, or 'unknown' if git is unavailable."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_analysis_plan_sha256() -> str:
    """SHA-256 of analysis_plan.md — proof that the plan was not modified post-training."""
    plan_path = PROJECT_ROOT / "analysis_plan.md"
    if not plan_path.exists():
        return "analysis_plan.md_NOT_FOUND"
    with open(plan_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_hardware_provenance() -> Dict[str, Any]:
    prov: Dict[str, Any] = {
        "python_version": sys.version,
        "torch_version":  torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        prov["cuda_version"]       = torch.version.cuda
        prov["gpu_name"]           = torch.cuda.get_device_name(0)
        prov["gpu_count"]          = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        prov["gpu_total_memory_mb"] = props.total_memory // (1024 * 1024)
    try:
        prov["numpy_version"]   = np.__version__
    except Exception:
        pass
    try:
        import sklearn
        prov["sklearn_version"] = sklearn.__version__
    except Exception:
        pass
    return prov


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Lock all RNGs for full per-seed reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    Clipped to [0.5, 30.0] to prevent extreme gradients for rare classes.
    Computed from training folds only — never from val or test fold.
    """
    Y = np.concatenate([ds.y for ds in train_datasets], axis=0)
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    raw_weights = neg / np.maximum(pos, 1)
    weights = np.clip(raw_weights, *clip).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# ECG augmentation and collate functions
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_ts_voltage(x_ts: torch.Tensor) -> torch.Tensor:
    return x_ts / 5.0


class ECGAugmentation:
    """On-the-fly stochastic augmentation for 12-lead ECG signals."""

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
    x_ts   = torch.stack([it["x_ts"]             for it in items])
    x_meta = torch.stack([it["x_meta"]            for it in items])
    mpm    = torch.stack([it["meta_present_mask"] for it in items])
    y      = torch.stack([it["y"]                 for it in items])
    x_ts   = normalize_ts_voltage(x_ts)
    if np.random.rand() > 0.5:
        x_ts = ECGAugmentation.add_gaussian_noise(x_ts)
        x_ts = ECGAugmentation.time_shift(x_ts)
        x_ts = ECGAugmentation.amplitude_scale(x_ts)
    return x_ts, x_meta, mpm, y


def collate_fn_val(items):
    x_ts   = torch.stack([it["x_ts"]             for it in items])
    x_meta = torch.stack([it["x_meta"]            for it in items])
    mpm    = torch.stack([it["meta_present_mask"] for it in items])
    y      = torch.stack([it["y"]                 for it in items])
    x_ts   = normalize_ts_voltage(x_ts)
    return x_ts, x_meta, mpm, y


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def safe_macro_auroc(Y: np.ndarray, P: np.ndarray) -> float:
    """Macro-AUC; skips classes with only one label present."""
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) < 2:
            continue
        aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def safe_auc_per_class(Y: np.ndarray, P: np.ndarray) -> List[float]:
    return [
        float(roc_auc_score(Y[:, j], P[:, j]))
        if len(np.unique(Y[:, j])) >= 2 else float("nan")
        for j in range(Y.shape[1])
    ]


def find_optimal_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1 on the validation fold."""
    thr = np.full(Y.shape[1], 0.5, dtype=np.float32)
    for j in range(Y.shape[1]):
        best, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 61):
            f1 = f1_score(Y[:, j], (P[:, j] >= t).astype(np.int32), zero_division=0)
            if f1 > best:
                best, best_t = f1, t
        thr[j] = best_t
    return thr


def compute_metrics_per_class(
    Y: np.ndarray, P: np.ndarray, thr: np.ndarray
) -> Tuple[float, float, List[float], List[float]]:
    aucs = safe_auc_per_class(Y, P)
    f1s  = [
        float(f1_score(Y[:, j], (P[:, j] >= thr[j]).astype(np.int32), zero_division=0))
        for j in range(Y.shape[1])
    ]
    return float(np.nanmean(aucs)), float(np.nanmean(f1s)), aucs, f1s


# ═══════════════════════════════════════════════════════════════════════════════
# Subgroup AUC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_subgroup_aucs(
    Y: np.ndarray, P: np.ndarray, test_df
) -> Dict[str, Any]:
    """
    Macro-AUC for sex and age subgroups (test fold only).

    Age thresholds derived from PTB-XL population statistics
    (Wagner et al. 2020: mean=62.5 yr, SD=17.2 yr):
      < 45 yr  → age_z < (45 − 62.5)/17.2 ≈ −1.02
      45–65 yr → −1.02 ≤ age_z < (65 − 62.5)/17.2 ≈ 0.145
      > 65 yr  → age_z ≥ 0.145
    """
    age_z  = test_df["age_z"].values
    sex01  = test_df["sex01"].values

    AGE_LT45_Z = (45 - _PTB_AGE_MEAN) / _PTB_AGE_SD   # ≈ −1.02
    AGE_GT65_Z = (65 - _PTB_AGE_MEAN) / _PTB_AGE_SD   # ≈  0.145

    # metadata-complete: all 5 core values are present (age, sex, height, weight, bmi)
    # mask__* columns in MASK_FEATURES encode availability; use age+sex+height+weight+bmi
    meta_complete_mask = (
        (test_df.get("mask__age",    0) >= 0.5) &
        (test_df.get("mask__sex",    0) >= 0.5) &
        (test_df.get("mask__height", 0) >= 0.5) &
        (test_df.get("mask__weight", 0) >= 0.5) &
        (test_df.get("mask__bmi",    0) >= 0.5)
    ).values if all(c in test_df.columns for c in
                    ["mask__age", "mask__sex", "mask__height", "mask__weight", "mask__bmi"]) \
        else np.ones(len(test_df), dtype=bool)

    masks = {
        "sex_male":         sex01 == 1,
        "sex_female":       sex01 == 0,
        "age_lt45":         age_z < AGE_LT45_Z,
        "age_45_65":        (age_z >= AGE_LT45_Z) & (age_z < AGE_GT65_Z),
        "age_gt65":         age_z >= AGE_GT65_Z,
        "meta_complete":    meta_complete_mask,
        "meta_incomplete":  ~meta_complete_mask,
    }

    result: Dict[str, Any] = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        result[name] = {"n": n}
        if n >= 10:
            result[name]["macro_auc"]     = safe_macro_auroc(Y[mask], P[mask])
            result[name]["per_class_auc"] = safe_auc_per_class(Y[mask], P[mask])
        else:
            result[name]["macro_auc"]     = None
            result[name]["per_class_auc"] = None

    # Fairness gap: absolute sex AUC difference
    m_auc = result["sex_male"].get("macro_auc")
    f_auc = result["sex_female"].get("macro_auc")
    if m_auc is not None and f_auc is not None:
        result["fairness_sex_gap"] = abs(m_auc - f_auc)
    else:
        result["fairness_sex_gap"] = None

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
# Branch-probability collection  (w_fused = 1.0 always, pre-declared)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_branch_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    disable_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect Y, P_fused, P_ecg, P_meta across the loader.
    disable_meta=True: zero x_meta and mpm before inference.
    w_fused is always 1.0 — final prediction = sigmoid(logits_fused).
    """
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
    return (
        np.concatenate(ys),
        np.concatenate(ps_fused),
        np.concatenate(ps_ecg),
        np.concatenate(ps_meta),
    )


@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate on validation fold at w_fused=1.0 (pre-declared, fixed).
    Returns val macro-AUC, per-class AUC, and delta_meta (for logging only).
    Checkpoint selection uses val macro-AUC ONLY — delta_meta is not a
    tie-breaking criterion (pre-declared in analysis_plan.md).
    """
    Y, p_fused, p_ecg, p_meta = collect_branch_probs(model, loader, device)
    _, p_nm, _, _              = collect_branch_probs(model, loader, device, disable_meta=True)

    auc_val    = safe_macro_auroc(Y, p_fused)
    auc_nm     = safe_macro_auroc(Y, p_nm)
    delta_meta = auc_val - auc_nm

    return {
        "Y":          Y,
        "P":          p_fused,
        "P_ecg":      p_ecg,
        "P_meta":     p_meta,
        "auc_val":    auc_val,
        "delta_meta": delta_meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# JSON serialisation helper
# ═══════════════════════════════════════════════════════════════════════════════

def _convert(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, Path):        return str(obj)
    return obj


def export_results_json(results: Dict[str, Any], output_path: Path) -> None:
    def _deep(v):
        if isinstance(v, dict):        return {k: _deep(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)): return [_deep(i) for i in v]
        return _convert(v)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_deep(results), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)
    print(f"   Results saved → {output_path}")


def load_existing_results_if_valid(results_path: Path) -> Optional[Dict[str, Any]]:
    """Return cached results only when the JSON exists and is structurally valid."""
    if not results_path.exists():
        return None
    try:
        with open(results_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[AUTO-RESUME] Ignoring unreadable results file: {results_path.name} ({exc})")
        return None
    if not isinstance(payload, dict):
        print(f"[AUTO-RESUME] Ignoring malformed results file: {results_path.name}")
        return None
    if not isinstance(payload.get("metadata"), dict) or not isinstance(payload.get("test"), dict):
        print(f"[AUTO-RESUME] Ignoring incomplete results file: {results_path.name}")
        return None
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Paths
    data_root:  str = str(DEFAULT_DATA_ROOT)
    index_path: str = str(DEFAULT_INDEX_PATH)
    runs_dir:   str = str(DEFAULT_RUNS_DIR)

    # Experiment identity
    seed:           int   = 2024
    variant:        str   = "demo+anthro"   # none | demo | demo+anthro
    meta_hid:       int   = 128
    lauc_weight:    float = 0.08
    no_aug:         bool  = False
    arch_mode:      str   = "standard"      # Group E ablations
    meta_mask_mode: str   = "real"          # real | trainmask (Group E8)
    test_fold:      int   = 10              # Group F alternate splits
    val_fold:       int   = 9

    # Architecture
    sampling_rate:  int   = 100
    meta_dropout_p: float = 0.10

    # Optimisation
    batch_size:                  int   = 32
    lr:                          float = 1e-3
    epochs:                      int   = 10
    patience:                    int   = 25   # > epochs — always run full 10 epochs
    gradient_accumulation_steps: int   = 2
    max_grad_norm:               float = 1.0

    # Runtime (filled automatically)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_run_name(cfg: Config) -> str:
    """
    Deterministic run-directory name encoding all experiment-type flags.

    Examples
    --------
    ATLAS_A_v5_demo+anthro_seed2024
    ATLAS_A_v5_demo+anthro_metaH64_seed2024
    ATLAS_A_v5_demo+anthro_lauc0.12_seed2024
    ATLAS_A_v5_demo+anthro_noaug_seed2024
    ATLAS_A_v5_demo+anthro_no_glu_seed2024
    ATLAS_A_v5_demo+anthro_trainmask_seed2024
    ATLAS_A_v5_demo+anthro_tf7_vf8_seed2024
    """
    parts = [f"ATLAS_A_v5_{cfg.variant}"]
    if cfg.meta_hid != 128:
        parts.append(f"metaH{cfg.meta_hid}")
    if abs(cfg.lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{cfg.lauc_weight:g}")
    if cfg.no_aug:
        parts.append("noaug")
    if cfg.arch_mode != "standard":
        parts.append(cfg.arch_mode)
    if cfg.meta_mask_mode == "trainmask":
        parts.append("trainmask")
    if cfg.test_fold != 10 or cfg.val_fold != 9:
        parts.append(f"tf{cfg.test_fold}_vf{cfg.val_fold}")
    parts.append(f"seed{cfg.seed}")
    return "_".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="EZNX-ATLAS-A v5 — MDPI Mathematics training script"
    )
    parser.add_argument("--variant",         type=str,   default="demo+anthro",
                        choices=["none", "demo", "demo+anthro"])
    parser.add_argument("--seed",            type=int,   default=2024)
    parser.add_argument("--meta_hid",        type=int,   default=128)
    parser.add_argument("--lauc_weight",     type=float, default=0.08)
    parser.add_argument("--no_aug",          action="store_true")
    parser.add_argument("--arch_mode",       type=str,   default="standard",
                        choices=sorted(VALID_ARCH_MODES))
    parser.add_argument("--meta_mask_mode",  type=str,   default="real",
                        choices=["real", "trainmask"],
                        help="trainmask: zero x_meta+mpm during training (Group E8)")
    parser.add_argument("--test_fold",       type=int,   default=10,
                        help="PTB-XL test fold (default 10; Group F uses 2,3,7,8)")
    parser.add_argument("--val_fold",        type=int,   default=9,
                        help="PTB-XL validation fold (default 9)")
    parser.add_argument("--data_root",       type=str,   default=None)
    parser.add_argument("--index_path",      type=str,   default=None)
    parser.add_argument("--runs_dir",        type=str,   default=None)
    parser.add_argument("--epochs",          type=int,   default=None,
                        help="Optional override for test/smoke runs. Default keeps the pre-declared 10 epochs.")
    parser.add_argument("--batch_size",      type=int,   default=None,
                        help="Optional override for test/smoke runs. Default keeps the pre-declared batch size.")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed           = args.seed
    cfg.variant        = args.variant
    cfg.meta_hid       = args.meta_hid
    cfg.lauc_weight    = args.lauc_weight
    cfg.no_aug         = args.no_aug
    cfg.arch_mode      = args.arch_mode
    cfg.meta_mask_mode = args.meta_mask_mode
    cfg.test_fold      = args.test_fold
    cfg.val_fold       = args.val_fold
    if args.data_root:  cfg.data_root  = args.data_root
    if args.index_path: cfg.index_path = args.index_path
    if args.runs_dir:   cfg.runs_dir   = args.runs_dir
    if args.epochs is not None:      cfg.epochs = args.epochs
    if args.batch_size is not None:  cfg.batch_size = args.batch_size

    device = torch.device(cfg.device)

    # ── Run directory and auto-resume ────────────────────────────────────────
    run_name = make_run_name(cfg)
    run_dir  = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / f"results_{run_name}.json"
    cached_results = load_existing_results_if_valid(results_path)
    if cached_results is not None:
        print(f"[AUTO-RESUME] Already complete: {results_path.name} — skipping.")
        return cached_results
    if results_path.exists():
        print(f"[AUTO-RESUME] Re-running because cached results are unreadable/incomplete: {results_path.name}")

    # ── Provenance ───────────────────────────────────────────────────────────
    git_sha          = get_git_sha()
    plan_sha256      = get_analysis_plan_sha256()
    hw               = get_hardware_provenance()
    set_seed(cfg.seed)

    print("=" * 80)
    print("EZNX-ATLAS-A v5 — MDPI Mathematics Training")
    print("=" * 80)
    print(f"  variant:        {cfg.variant}")
    print(f"  seed:           {cfg.seed}")
    print(f"  meta_hid:       {cfg.meta_hid}")
    print(f"  lauc_weight:    {cfg.lauc_weight}")
    print(f"  no_aug:         {cfg.no_aug}")
    print(f"  arch_mode:      {cfg.arch_mode}")
    print(f"  meta_mask_mode: {cfg.meta_mask_mode}")
    print(f"  test_fold:      {cfg.test_fold}  val_fold: {cfg.val_fold}")
    print(f"  device:         {device}  ({hw.get('gpu_name', 'CPU')})")
    print(f"  git_sha:        {git_sha[:12]}...")
    print(f"  plan_sha256:    {plan_sha256[:16]}...")
    print(f"  output:         {run_dir}")
    print("=" * 80)

    wall_start = time.time()

    # ── 1. Datasets ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")
    all_folds    = set(range(1, 11))
    exclude_folds = {cfg.val_fold, cfg.test_fold}
    train_fold_ids = sorted(all_folds - exclude_folds)   # 8 folds

    train_datasets = [
        EZNXDataset(
            index_file=cfg.index_path, data_root=cfg.data_root,
            fold=f, sampling_rate=cfg.sampling_rate, meta_mode=cfg.variant
        )
        for f in train_fold_ids
    ]
    train_ds = ConcatDataset(train_datasets)

    val_ds = EZNXDataset(
        index_file=cfg.index_path, data_root=cfg.data_root,
        fold=cfg.val_fold, sampling_rate=cfg.sampling_rate, meta_mode=cfg.variant
    )
    test_ds = EZNXDataset(
        index_file=cfg.index_path, data_root=cfg.data_root,
        fold=cfg.test_fold, sampling_rate=cfg.sampling_rate, meta_mode=cfg.variant
    )

    _lkw = dict(num_workers=0, pin_memory=(cfg.device == "cuda"))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn_val if cfg.no_aug else collate_fn_augmented,
        **_lkw
    )
    val_loader  = DataLoader(val_ds,  batch_size=cfg.batch_size,
                             collate_fn=collate_fn_val, **_lkw)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             collate_fn=collate_fn_val, **_lkw)

    n_train = len(train_ds)
    n_val   = len(val_ds)
    n_test  = len(test_ds)
    print(f"   Train folds: {train_fold_ids}  |  n_train={n_train}  n_val={n_val}  n_test={n_test}")

    # ── 2. Pos-weights (dynamic from training folds) ─────────────────────────
    print("\n[2/6] Computing pos_weights from training folds …")
    pos_weights = compute_pos_weights(train_datasets, n_classes=5, device=device)
    print(f"   pos_weights = {pos_weights.cpu().numpy().round(4).tolist()}")

    # ── 3. Model ─────────────────────────────────────────────────────────────
    print("\n[3/6] Initialising model …")
    model = EZNX_ATLAS_A_v5(
        meta_dim=16,
        n_classes=len(DS5_LABELS),
        meta_hid=cfg.meta_hid,
        meta_dropout_p=cfg.meta_dropout_p,
        arch_mode=cfg.arch_mode,
    ).to(device)

    n_params = count_parameters(model)
    print(f"   Parameters: {n_params:,}  |  arch_mode: {cfg.arch_mode}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── 4. Loss weights (pre-declared coupling in analysis_plan.md) ──────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    lauc_w  = cfg.lauc_weight
    fused_w = max(0.0, 0.60 - lauc_w)   # coupled: fused_w decreases as lauc_w increases
    ecg_w   = 0.30
    meta_w  = 0.10
    total_w = fused_w + ecg_w + meta_w + lauc_w
    fused_w /= total_w
    ecg_w   /= total_w
    meta_w  /= total_w
    lauc_w_n = lauc_w / total_w

    print(f"   Loss weights — fused:{fused_w:.3f}  ecg:{ecg_w:.3f}  "
          f"meta:{meta_w:.3f}  lauc:{lauc_w_n:.3f}")

    # ── 5. Training loop ─────────────────────────────────────────────────────
    print("\n[4/6] Training …")
    print("-" * 80)

    # Checkpoint selection: val macro-AUC ONLY (pre-declared in analysis_plan.md)
    # delta_meta is logged but NEVER used for checkpoint selection.
    best_auc     = -1.0
    patience_ctr = 0
    history      = []
    ckpt_path    = run_dir / f"best_model_{run_name}.pt"

    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        opt.zero_grad()

        for batch_idx, (x_ts, x_meta, mpm, y) in enumerate(tqdm(
            train_loader, desc=f"Ep {epoch:3d}/{cfg.epochs}", ncols=100, leave=False
        )):
            x_ts   = x_ts.to(device)
            x_meta = x_meta.to(device)
            mpm    = mpm.to(device)
            y      = y.to(device)

            # trainmask (Group E8): zero x_meta AND mpm during training
            # meta_quality → 0 → all three metadata-dependent terms vanish
            if cfg.meta_mask_mode == "trainmask":
                x_meta = torch.zeros_like(x_meta)
                mpm    = torch.zeros_like(mpm)

            out   = model(x_ts, x_meta, mpm)
            p_f   = torch.sigmoid(out["logits_fused"])

            # Metadata quality for weighted meta loss (from mpm, not x_meta)
            meta_quality_w = torch.clamp(
                mpm[:, :2].float().mean(dim=1, keepdim=True)
                + 0.5 * mpm[:, 2:].float().mean(dim=1, keepdim=True),
                max=1.0,
            )
            meta_loss = (
                F.binary_cross_entropy_with_logits(
                    out["logits_meta"], y, reduction="none"
                ) * meta_quality_w
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

        # ── Validation (w=1.0, pre-declared) ─────────────────────────────────
        vm     = evaluate_val(model, val_loader, device)
        auc_v  = float(vm["auc_val"])
        delta_v = float(vm["delta_meta"])
        f1_v   = f1_score(vm["Y"], (vm["P"] >= 0.5), average="macro", zero_division=0)
        avg_loss = train_loss / n_batches
        ep_time  = time.time() - epoch_t0

        rec = dict(
            epoch=epoch, train_loss=avg_loss, val_auc=auc_v, val_f1=float(f1_v),
            val_auc_ecg=safe_macro_auroc(vm["Y"], vm["P_ecg"]),
            val_delta_meta=delta_v,
            lr=opt.param_groups[0]["lr"], epoch_time_s=ep_time,
        )
        history.append(rec)
        print(f"Ep {epoch:3d} | loss={avg_loss:.4f} | AUC={auc_v:.4f} "
              f"| F1={float(f1_v):.4f} | Δmeta={delta_v:+.4f} "
              f"| w=1.0 [fixed] | {ep_time:.0f}s")

        # Checkpoint: val macro-AUC ONLY — no delta_meta tie-break
        if auc_v > best_auc + 1e-6:
            best_auc = auc_v
            patience_ctr = 0
            thr_val = find_optimal_thresholds(vm["Y"], vm["P"])
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "thresholds":           thr_val,
                "w_fused":              1.0,    # always 1.0 (pre-declared)
                "best_auc":             best_auc,
                "seed":                 cfg.seed,
                "arch_mode":            cfg.arch_mode,
                "meta_mask_mode":       cfg.meta_mask_mode,
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)
            print(f"   ★ New best val AUC: {best_auc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"\n   Early stopping after epoch {epoch}")
                break

    wall_train = time.time() - wall_start

    # ── 6. Test evaluation ───────────────────────────────────────────────────
    print(f"\n[5/6] Final evaluation on test fold {cfg.test_fold} …")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    thr_final = ckpt.get("thresholds", np.full(len(DS5_LABELS), 0.5))
    w_f_final = 1.0   # always 1.0 — pre-declared

    # All branch probs (w=1.0 → P_blend = P_fused)
    Yt, p_fused_t, p_ecg_t, p_meta_t = collect_branch_probs(model, test_loader, device)
    _, p_nm_t, _, _ = collect_branch_probs(model, test_loader, device, disable_meta=True)

    # P_blend = P_fused at w=1.0 (pre-declared)
    Pt      = p_fused_t
    Pt_nm   = p_nm_t
    Pt_ecg  = p_ecg_t

    auc_m, f1_m, aucs, f1s = compute_metrics_per_class(Yt, Pt, thr_final)
    auc_nm  = safe_macro_auroc(Yt, Pt_nm)
    auc_ecg = safe_macro_auroc(Yt, Pt_ecg)
    f1_fixed = float(f1_score(Yt, (Pt >= 0.5), average="macro", zero_division=0))

    gpu_mem_peak_mb = (
        torch.cuda.max_memory_allocated() // (1024 * 1024)
        if torch.cuda.is_available() else 0
    )

    # ── Subgroup AUC ─────────────────────────────────────────────────────────
    print("\n[6/6] Computing subgroup AUCs …")
    import pandas as pd
    index_df = pd.read_parquet(cfg.index_path)
    test_df  = index_df[index_df["strat_fold"] == cfg.test_fold].reset_index(drop=True)
    subgroup_aucs = compute_subgroup_aucs(Yt, Pt, test_df)

    # ── NPZ dump (P_blend = P_fused at w=1.0) ────────────────────────────────
    npz_path = run_dir / f"probs_{run_name}.npz"
    ecg_ids  = test_df["ecg_id"].values
    pat_ids  = test_df["patient_id"].values
    np.savez_compressed(
        npz_path,
        Y=Yt,
        P_fused=p_fused_t,
        P_ecg=p_ecg_t,
        P_meta=p_meta_t,
        P_blend=Pt,          # = P_fused (w=1.0)
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
    print(f"RESULTS — seed={cfg.seed} | variant={cfg.variant} | fold={cfg.test_fold}")
    print("=" * 80)
    print(f"Macro AUC (fused, w=1.0): {auc_m:.4f}")
    print(f"Macro AUC (ECG only):     {auc_ecg:.4f}")
    print(f"Macro AUC (no meta):      {auc_nm:.4f}")
    print(f"Delta AUC (meta):         {auc_m - auc_nm:+.4f}")
    print(f"Macro F1 (optimal thr):   {f1_m:.4f}")
    print(f"Macro F1 (thr=0.5):       {f1_fixed:.4f}")
    print("-" * 50)
    print(f"{'Class':<10} | {'AUC':>8} | {'F1':>8} | {'Thr':>6}")
    for i, lbl in enumerate(DS5_LABELS):
        print(f"{lbl:<10} | {aucs[i]:>8.4f} | {f1s[i]:>8.4f} | {thr_final[i]:>6.3f}")
    print("=" * 80)
    print(f"Wall time: {wall_total/60:.1f} min  |  GPU peak: {gpu_mem_peak_mb} MB")

    # ── Assemble and export results dict ──────────────────────────────────────
    results = {
        "metadata": {
            "run_name":             run_name,
            "variant":              cfg.variant,
            "seed":                 cfg.seed,
            "meta_hid":             cfg.meta_hid,
            "lauc_weight":          cfg.lauc_weight,
            "no_aug":               cfg.no_aug,
            "arch_mode":            cfg.arch_mode,
            "meta_mask_mode":       cfg.meta_mask_mode,
            "test_fold":            cfg.test_fold,
            "val_fold":             cfg.val_fold,
            "timestamp":            datetime.now().isoformat(),
            "git_sha":              git_sha,
            "analysis_plan_sha256": plan_sha256,
            "wall_time_s":          wall_total,
            "train_time_s":         wall_train,
            "gpu_peak_mem_mb":      gpu_mem_peak_mb,
            "num_parameters":       n_params,
            "dataset_sizes":        {"train": n_train, "val": n_val, "test": n_test},
            "train_fold_ids":       train_fold_ids,
            "pos_weights":          pos_weights.cpu().numpy().tolist(),
            "loss_weights":         {
                "fused": fused_w, "ecg": ecg_w,
                "meta": meta_w, "lauc": lauc_w_n,
            },
            "hardware":             hw,
            "best_val_auc":         best_auc,
            "best_val_epoch":       max(history, key=lambda h: h["val_auc"])["epoch"]
                                    if history else -1,
        },
        "test": {
            "macro_auc":          auc_m,
            "macro_auc_ecg":      auc_ecg,
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
        "subgroups":        subgroup_aucs,
        "training_history": history,
    }

    export_results_json(results, results_path)
    print(f"\n✓ Run complete: {run_name}")
    return results


if __name__ == "__main__":
    main()
