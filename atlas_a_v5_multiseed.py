
import os
import json
import random
import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List, Any
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

DEFAULT_BLEND_CANDIDATES = (0.0, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0)
DEFAULT_DATA_ROOT = Path(os.getenv("EZNX_DATA_REAL", PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3"))
DEFAULT_INDEX_PATH = Path(os.getenv("EZNX_INDEX_PATH", PROJECT_ROOT / "index_complete.parquet"))
DEFAULT_RUNS_DIR = Path(os.getenv("EZNX_RUNS_DIR", PROJECT_ROOT / "runs"))


# ----------------------------- Reproductibilité -----------------------------
def set_seed(seed: int):
    """Configure toutes les sources d'aléatoire pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- Augmentation de Données ----------------------
class ECGAugmentation:
    """Augmentation spécialisée pour signaux ECG."""
    
    @staticmethod
    def add_gaussian_noise(x, noise_level=0.02):
        if np.random.rand() > 0.3:
            return x
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    @staticmethod
    def time_shift(x, max_shift=20):
        if np.random.rand() > 0.3:
            return x
        shift = np.random.randint(-max_shift, max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(x, scale_range=(0.95, 1.05)):
        if np.random.rand() > 0.3:
            return x
        scale = np.random.uniform(*scale_range)
        return x * scale


# ----------------------------- Utils ---------------------------------------
def normalize_ts_voltage(x_ts: torch.Tensor) -> torch.Tensor:
    return x_ts / 5.0


def collate_fn_augmented(items):
    """Collate avec augmentation de données."""
    x_ts = torch.stack([it["x_ts"] for it in items], dim=0)
    x_meta = torch.stack([it["x_meta"] for it in items], dim=0)
    mpm = torch.stack([it["meta_present_mask"] for it in items], dim=0)
    y = torch.stack([it["y"] for it in items], dim=0)
    
    x_ts = normalize_ts_voltage(x_ts)
    
    if np.random.rand() > 0.5:
        x_ts = ECGAugmentation.add_gaussian_noise(x_ts)
        x_ts = ECGAugmentation.time_shift(x_ts)
        x_ts = ECGAugmentation.amplitude_scale(x_ts)
    
    return x_ts, x_meta, mpm, y


def collate_fn_val(items):
    """Collate sans augmentation pour validation/test."""
    x_ts = torch.stack([it["x_ts"] for it in items], dim=0)
    x_meta = torch.stack([it["x_meta"] for it in items], dim=0)
    mpm = torch.stack([it["meta_present_mask"] for it in items], dim=0)
    y = torch.stack([it["y"] for it in items], dim=0)
    x_ts = normalize_ts_voltage(x_ts)
    return x_ts, x_meta, mpm, y


def safe_macro_auroc(Y: np.ndarray, P: np.ndarray) -> float:
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) < 2:
            continue
        aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def find_optimal_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    thr = np.full(Y.shape[1], 0.5, dtype=np.float32)
    for j in range(Y.shape[1]):
        best, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 61):
            pred = (P[:, j] >= t).astype(np.int32)
            f1 = f1_score(Y[:, j], pred, zero_division=0)
            if f1 > best:
                best, best_t = f1, t
        thr[j] = best_t
    return thr


def compute_metrics_per_class(Y, P, thr):
    aucs, f1s = [], []
    for j in range(Y.shape[1]):
        auc = roc_auc_score(Y[:, j], P[:, j]) if len(np.unique(Y[:, j])) >= 2 else float("nan")
        pred = (P[:, j] >= thr[j]).astype(np.int32)
        f1 = f1_score(Y[:, j], pred, zero_division=0)
        aucs.append(auc)
        f1s.append(f1)
    return np.nanmean(aucs), np.nanmean(f1s), aucs, f1s


# ----------------------------- AUC Margin Loss -----------------------------
def auc_margin_loss(y_true, p):
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=p.device)
    return torch.mean((1 - pos.unsqueeze(1) + neg.unsqueeze(0)).clamp(min=0))


# ----------------------------- Dual Prediction -----------------------------
@torch.no_grad()
def collect_branch_probs(model, loader, device, disable_meta: bool = False):
    model.eval()
    ys, ps_fused, ps_ecg, ps_meta = [], [], [], []
    for x_ts, x_meta, mpm, y in loader:
        x_ts = x_ts.to(device)
        x_meta = x_meta.to(device)
        mpm = mpm.to(device)

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


def blend_probs(p_fused: np.ndarray, p_ecg: np.ndarray, w_fused: float) -> np.ndarray:
    return w_fused * p_fused + (1.0 - w_fused) * p_ecg


@torch.no_grad()
def select_best_val_blend(model, loader, device, blend_candidates=None):
    y_true, p_fused, p_ecg, p_meta = collect_branch_probs(model, loader, device)
    _, p_fused_nometa, p_ecg_nometa, _ = collect_branch_probs(
        model, loader, device, disable_meta=True
    )
    best = {
        "w_fused": 1.0,
        "auc": -1.0,
        "delta_meta": -1.0,
        "probs": p_fused,
        "probs_nometa": p_fused_nometa,
        "p_fused": p_fused,
        "p_ecg": p_ecg,
        "p_meta": p_meta,
    }
    if blend_candidates is None:
        blend_candidates = np.asarray(DEFAULT_BLEND_CANDIDATES, dtype=float)

    for w_fused in blend_candidates:
        probs = blend_probs(p_fused, p_ecg, float(w_fused))
        probs_nometa = blend_probs(p_fused_nometa, p_ecg_nometa, float(w_fused))
        auc = safe_macro_auroc(y_true, probs)
        auc_nometa = safe_macro_auroc(y_true, probs_nometa)
        delta_meta = auc - auc_nometa
        is_better = auc > best["auc"] + 1e-6
        is_tie_with_more_meta = (
            abs(auc - best["auc"]) <= 5e-4 and delta_meta > best["delta_meta"] + 1e-6
        )
        is_tie_with_higher_w = (
            abs(auc - best["auc"]) <= 5e-4
            and abs(delta_meta - best["delta_meta"]) <= 5e-4
            and float(w_fused) > best["w_fused"]
        )
        if is_better or is_tie_with_more_meta or is_tie_with_higher_w:
            best["w_fused"] = float(w_fused)
            best["auc"] = auc
            best["delta_meta"] = delta_meta
            best["probs"] = probs
            best["probs_nometa"] = probs_nometa
    return y_true, best


@torch.no_grad()
def predict_with_blend(model, loader, device, w_fused: float, disable_meta: bool = False):
    y_true, p_fused, p_ecg, _ = collect_branch_probs(
        model, loader, device, disable_meta=disable_meta
    )
    return y_true, blend_probs(p_fused, p_ecg, w_fused)


# ----------------------------- Config --------------------------------------
@dataclass
class Config:
    # Chemins (À ADAPTER À VOTRE ENVIRONNEMENT)
    data_root: str = str(DEFAULT_DATA_ROOT)
    index_path: str = str(DEFAULT_INDEX_PATH)
    runs_dir: str = str(DEFAULT_RUNS_DIR)
    
    # Reproductibilité
    seed: int = 2026  # Sera écrasé par l'argument --seed
    
    # Architecture
    sampling_rate: int = 100
    meta_dropout_p: float = 0.10  # Conforme à l'article (Section 3.5.3)
    
    # Entraînement
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 10
    patience: int = 25
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Blending
    blend_candidates: Tuple[float, ...] = DEFAULT_BLEND_CANDIDATES
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- Results Export -----------------------------
def export_results_json(results: Dict[str, Any], output_path: Path):
    """Exporte les résultats en JSON avec conversion des types numpy."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj
    
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {k2: convert(v2) for k2, v2 in v.items()}
        else:
            serializable[k] = convert(v)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"   Résultats exportés: {output_path}")


# ----------------------------- Main ----------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='EZNX_ATLAS_A v5.1 - Entraînement Multi-Graines'
    )
    parser.add_argument(
        '-variant', '--variant', 
        type=str, 
        default='demo+anthro',
        choices=['none', 'demo', 'demo+anthro'],
        help='Variante d\'ablation des métadonnées'
    )
    parser.add_argument(
        '-seed', '--seed',
        type=int,
        default=2026,
        help='Graine aléatoire pour reproductibilité (défaut: 2026)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Chemin vers les données PTB-XL (optionnel)'
    )
    parser.add_argument(
        '--index_path',
        type=str,
        default=None,
        help='Chemin vers le fichier index parquet (optionnel)'
    )
    parser.add_argument(
        '--runs_dir',
        type=str,
        default=None,
        help='Répertoire de sortie pour les runs (optionnel)'
    )
    args = parser.parse_args()
    
    # Configuration
    cfg = Config()
    cfg.seed = args.seed
    
    # Surcharge des chemins si fournis
    if args.data_root:
        cfg.data_root = args.data_root
    if args.index_path:
        cfg.index_path = args.index_path
    if args.runs_dir:
        cfg.runs_dir = args.runs_dir
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    variant = args.variant
    
    # Répertoire de run avec seed dans le nom
    run_dir = Path(cfg.runs_dir) / f"ATLAS_A_v5_{variant}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Structure pour stocker tous les résultats
    all_results = {
        "metadata": {
            "variant": variant,
            "seed": cfg.seed,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "meta_dropout_p": cfg.meta_dropout_p,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_grad_norm": cfg.max_grad_norm,
            }
        },
        "validation": {},
        "test": {},
        "per_class": {}
    }

    print("=" * 80)
    print(f"ENTRAÎNEMENT EZNX_ATLAS_A v5.1 MULTI-GRAINES")
    print("=" * 80)
    print(f"Variante:    {variant}")
    print(f"Seed:        {cfg.seed}")
    print(f"Device:      {device}")
    print(f"Batch size:  {cfg.batch_size} (effectif: {cfg.batch_size * cfg.gradient_accumulation_steps})")
    print(f"Output:      {run_dir}")
    print("=" * 80)

    # 1. Préparation des Datasets
    print("\n[1/5] Chargement des données...")
    train_datasets = [
        EZNXDataset(
            index_file=cfg.index_path, 
            data_root=cfg.data_root, 
            fold=f,
            sampling_rate=cfg.sampling_rate, 
            meta_mode=variant
        )
        for f in range(1, 9)
    ]
    train_ds = ConcatDataset(train_datasets)
    
    val_ds = EZNXDataset(
        index_file=cfg.index_path, 
        data_root=cfg.data_root, 
        fold=9,
        sampling_rate=cfg.sampling_rate, 
        meta_mode=variant
    )
    
    test_ds = EZNXDataset(
        index_file=cfg.index_path, 
        data_root=cfg.data_root, 
        fold=10,
        sampling_rate=cfg.sampling_rate, 
        meta_mode=variant
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_augmented, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        collate_fn=collate_fn_val,
        num_workers=2, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.batch_size, 
        collate_fn=collate_fn_val,
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"   Train:      {len(train_ds)} samples (Folds 1-8)")
    print(f"   Validation: {len(val_ds)} samples (Fold 9)")
    print(f"   Test:       {len(test_ds)} samples (Fold 10)")
    
    all_results["metadata"]["dataset_sizes"] = {
        "train": len(train_ds),
        "validation": len(val_ds),
        "test": len(test_ds)
    }

    # 2. Modèle et Optimiseur
    print("\n[2/5] Initialisation du modèle...")
    
    model = EZNX_ATLAS_A_v5(
        meta_dim=16,
        n_classes=len(DS5_LABELS),
        meta_dropout_p=cfg.meta_dropout_p,  
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"   Paramètres entraînables: {num_params:,}")
    all_results["metadata"]["num_parameters"] = num_params
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 3. Fonction de coût pondérée
    print("\n[3/5] Configuration de la fonction de coût...")
    pos_weights = torch.tensor([1.0, 2.3, 2.44, 2.67, 5.58], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    print(f"   BCEWithLogitsLoss avec poids: {pos_weights.cpu().numpy()}")
    all_results["metadata"]["pos_weights"] = pos_weights.cpu().numpy().tolist()

    # 4. Boucle d'entraînement
    print("\n[4/5] Début de l'entraînement...")
    print("-" * 80)
    
    best_auc = -1.0
    best_delta_meta = -1.0
    patience_ctr = 0
    training_history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        opt.zero_grad()

        for batch_idx, (x_ts, x_meta, mpm, y) in enumerate(tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{cfg.epochs}", 
            ncols=100,
            leave=False
        )):
            x_ts, x_meta, mpm, y = (
                x_ts.to(device), 
                x_meta.to(device), 
                mpm.to(device), 
                y.to(device)
            )

            out = model(x_ts, x_meta, mpm)
            p_fused = torch.sigmoid(out["logits_fused"])
            
            meta_quality = torch.clamp(
                mpm[:, :2].float().mean(dim=1, keepdim=True)
                + 0.5 * mpm[:, 2:].float().mean(dim=1, keepdim=True),
                max=1.0,
            )
            meta_loss = F.binary_cross_entropy_with_logits(
                out["logits_meta"], y, reduction="none"
            )
            meta_loss = (meta_loss * meta_quality).mean()

            loss = (
                0.52 * criterion(out["logits_fused"], y)
                + 0.30 * criterion(out["logits_ecg"], y)
                + 0.10 * meta_loss
                + 0.08 * auc_margin_loss(y, p_fused)
            )
            
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad()

            train_loss += loss.item() * cfg.gradient_accumulation_steps

        # Stepped once per epoch, so T_0 and T_mult are expressed in epochs.
        scheduler.step()

        # ------------------------- Validation -------------------------
        Yv, val_metrics = select_best_val_blend(
            model, val_loader, device, blend_candidates=cfg.blend_candidates
        )
        Pv = val_metrics["probs"]
        auc_v = safe_macro_auroc(Yv, Pv)
        f1_v = f1_score(Yv, (Pv >= 0.5), average="macro", zero_division=0)
        auc_v_fused = safe_macro_auroc(Yv, val_metrics["p_fused"])
        auc_v_ecg = safe_macro_auroc(Yv, val_metrics["p_ecg"])
        delta_meta_v = float(val_metrics["delta_meta"])
        
        current_lr = opt.param_groups[0]['lr']
        avg_loss = train_loss / len(train_loader)

        # Enregistrement de l'historique
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auc": auc_v,
            "val_f1": f1_v,
            "val_auc_fused": auc_v_fused,
            "val_auc_ecg": auc_v_ecg,
            "val_delta_meta": delta_meta_v,
            "w_fused": val_metrics["w_fused"],
            "lr": current_lr
        }
        training_history.append(epoch_record)

        print(
            f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
            f"AUC_Val: {auc_v:.4f} | F1_Val: {f1_v:.4f} | "
            f"Delta_Meta: {delta_meta_v:+.4f} | "
            f"w_fused: {val_metrics['w_fused']:.2f}"
        )

        # Sauvegarde du meilleur modèle
        is_better_auc = auc_v > best_auc + 1e-6
        is_tie_with_better_meta = (
            abs(auc_v - best_auc) <= 5e-4 and delta_meta_v > best_delta_meta + 1e-6
        )
        
        if is_better_auc or is_tie_with_better_meta:
            best_auc = auc_v
            best_delta_meta = delta_meta_v
            patience_ctr = 0
            
            thr_val = find_optimal_thresholds(Yv, Pv)
            
            checkpoint_path = run_dir / f"best_model_v5_{variant}_seed{cfg.seed}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(), 
                "thresholds": thr_val,
                "w_fused": val_metrics["w_fused"],
                "best_auc": best_auc,
                "best_delta_meta": best_delta_meta,
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "seed": cfg.seed
            }, checkpoint_path)
            
            print(f"   ★ Nouveau record AUC: {best_auc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"\n   Early stopping après {epoch} epochs")
                break

    all_results["training_history"] = training_history
    all_results["validation"]["best_auc"] = best_auc
    all_results["validation"]["best_delta_meta"] = best_delta_meta
    all_results["validation"]["best_epoch"] = training_history[
        np.argmax([h["val_auc"] for h in training_history])
    ]["epoch"]

    # ------------------------- Évaluation Finale -------------------------
    print("\n" + "=" * 80)
    print("[5/5] ÉVALUATION FINALE (FOLD 10 - TEST)")
    print("=" * 80)

    ckpt_path = run_dir / f"best_model_v5_{variant}_seed{cfg.seed}.pt"
    
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        
        thr_final = ckpt.get("thresholds", np.full(len(DS5_LABELS), 0.5))
        w_fused_final = float(ckpt.get("w_fused", 1.0))
        
        print(f"   Seuils optimisés: {thr_final}")
        print(f"   Poids de fusion:  {w_fused_final:.2f}")

        # Inférence sur le Test
        Yt, Pt = predict_with_blend(model, test_loader, device, w_fused=w_fused_final)
        _, Pt_no_meta = predict_with_blend(
            model, test_loader, device, w_fused=w_fused_final, disable_meta=True
        )
        _, Pt_ecg = predict_with_blend(model, test_loader, device, w_fused=0.0)
        _, Pt_fused = predict_with_blend(model, test_loader, device, w_fused=1.0)

        # Métriques globales
        auc_m, f1_m, aucs, f1s = compute_metrics_per_class(Yt, Pt, thr_final)
        auc_no_meta = safe_macro_auroc(Yt, Pt_no_meta)
        auc_ecg = safe_macro_auroc(Yt, Pt_ecg)
        auc_fused = safe_macro_auroc(Yt, Pt_fused)
        delta_meta_auc = auc_m - auc_no_meta
        
        # F1 à seuil fixe 0.5
        f1_fixed = f1_score(Yt, (Pt >= 0.5), average="macro", zero_division=0)

        # Stockage des résultats test
        all_results["test"] = {
            "macro_auc": auc_m,
            "macro_f1_optimal": f1_m,
            "macro_f1_fixed": f1_fixed,
            "auc_ecg_only": auc_ecg,
            "auc_fused_only": auc_fused,
            "auc_meta_disabled": auc_no_meta,
            "delta_meta_auc": delta_meta_auc,
            "w_fused": w_fused_final,
            "thresholds": thr_final.tolist()
        }
        
        # Métriques par classe
        for i, lbl in enumerate(DS5_LABELS):
            all_results["per_class"][lbl] = {
                "auc": aucs[i],
                "f1": f1s[i],
                "threshold": float(thr_final[i])
            }

        # Affichage
        print("\n" + "=" * 80)
        print(f"RÉSULTATS FINAUX - Seed: {cfg.seed} - Variante: {variant}")
        print("=" * 80)
        print(f"Macro AUC Test (blend):     {auc_m:.4f}")
        print(f"Macro AUC Test (ECG only):  {auc_ecg:.4f}")
        print(f"Macro AUC Test (fused):     {auc_fused:.4f}")
        print(f"Delta AUC (meta active):    {delta_meta_auc:+.4f}")
        print(f"Macro F1 Test (optimal):    {f1_m:.4f}")
        print(f"Macro F1 Test (seuil 0.5):  {f1_fixed:.4f}")
        print("=" * 80)
        print(f"{'Classe':<10} | {'AUC':<10} | {'F1':<10} | {'Seuil':<10}")
        print("-" * 50)
        for i, lbl in enumerate(DS5_LABELS):
            print(f"{lbl:<10} | {aucs[i]:<10.4f} | {f1s[i]:<10.4f} | {thr_final[i]:<10.3f}")
        print("=" * 80)

    else:
        print(f"ERREUR: Checkpoint introuvable: {ckpt_path}")
        all_results["test"]["error"] = "Checkpoint not found"

    # Export JSON des résultats
    results_path = run_dir / f"results_{variant}_seed{cfg.seed}.json"
    export_results_json(all_results, results_path)
    
    print(f"\n✓ Entraînement terminé pour seed={cfg.seed}")
    print(f"  Modèle:    {ckpt_path}")
    print(f"  Résultats: {results_path}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()
