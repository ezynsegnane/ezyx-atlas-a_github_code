"""
compute_calibration.py — Cross-seed calibration metrics for Scientific Reports.

Reads NPZ probability files for the 60 PRIMARY runs (3 variants × 20 seeds)
and computes, per variant:

  • Brier score (lower = better calibrated)
  • Expected Calibration Error (ECE, 10-bin)
  • Macro-AUC (redundant with JSON but kept for cross-check)
  • Per-class AUC distribution over seeds (mean ± SD)

Output
------
  results/calibration/calibration_report.json
  results/calibration/calibration_table.tex      (ready-to-paste LaTeX table)

Usage
-----
  python compute_calibration.py --runs_dir /kaggle/working/runs \
                                --out_dir  /kaggle/working/results/calibration
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent

DS5_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANTS   = ["none", "demo", "demo+anthro"]
SEEDS      = list(range(2024, 2044))


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def brier_score_macro(Y: np.ndarray, P: np.ndarray) -> float:
    """Mean-across-classes Brier score."""
    scores = [np.mean((Y[:, j] - P[:, j]) ** 2) for j in range(Y.shape[1])]
    return float(np.mean(scores))


def ece_binary(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for a single binary class (uniform bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def ece_macro(Y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> float:
    """Mean-across-classes ECE."""
    return float(np.mean([ece_binary(Y[:, j], P[:, j], n_bins)
                           for j in range(Y.shape[1])]))


def macro_auc(Y: np.ndarray, P: np.ndarray) -> float:
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) >= 2:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_npz(runs_dir: Path, variant: str, seed: int,
             meta_hid: int = 128, lauc_weight: float = 0.08,
             no_aug: bool = False) -> Dict[str, np.ndarray]:
    """Load a single run's NPZ probability file."""
    parts = [f"ATLAS_A_v5_{variant}"]
    if meta_hid != 128:
        parts.append(f"metaH{meta_hid}")
    if abs(lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{lauc_weight:g}")
    if no_aug:
        parts.append("noaug")
    parts.append(f"seed{seed}")
    name = "_".join(parts)
    npz_path = runs_dir / name / f"probs_{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    return dict(np.load(npz_path, allow_pickle=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--out_dir",  type=str, default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir or os.getenv("EZNX_RUNS_DIR",
                    str(PROJECT_ROOT / "runs")))
    out_dir  = Path(args.out_dir or
                    PROJECT_ROOT / "results" / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {}

    for variant in VARIANTS:
        print(f"\n=== Variant: {variant} ===")
        aucs_seeds, briers, eces = [], [], []
        per_class: Dict[str, List[float]] = {lbl: [] for lbl in DS5_LABELS}

        for seed in SEEDS:
            try:
                npz = load_npz(runs_dir, variant, seed)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                continue

            Y  = npz["Y"]
            P  = npz["P_blend"]

            ma = macro_auc(Y, P)
            bs = brier_score_macro(Y, P)
            ec = ece_macro(Y, P)
            aucs_seeds.append(ma)
            briers.append(bs)
            eces.append(ec)

            for j, lbl in enumerate(DS5_LABELS):
                if len(np.unique(Y[:, j])) >= 2:
                    per_class[lbl].append(
                        float(roc_auc_score(Y[:, j], P[:, j]))
                    )

        n = len(aucs_seeds)
        if n == 0:
            print("  No completed runs found.")
            continue

        variant_report = {
            "n_seeds": n,
            "macro_auc_mean":   float(np.mean(aucs_seeds)),
            "macro_auc_sd":     float(np.std(aucs_seeds, ddof=1)) if n > 1 else 0.0,
            "brier_mean":       float(np.mean(briers)),
            "brier_sd":         float(np.std(briers,      ddof=1)) if n > 1 else 0.0,
            "ece_mean":         float(np.mean(eces)),
            "ece_sd":           float(np.std(eces,         ddof=1)) if n > 1 else 0.0,
            "per_class_auc": {
                lbl: {
                    "mean": float(np.mean(v)),
                    "sd":   float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                }
                for lbl, v in per_class.items() if v
            },
            "all_seed_aucs":    aucs_seeds,
            "all_seed_briers":  briers,
            "all_seed_eces":    eces,
        }
        report[variant] = variant_report

        print(f"  N seeds: {n}")
        print(f"  Macro AUC : {variant_report['macro_auc_mean']:.4f} "
              f"± {variant_report['macro_auc_sd']:.4f}")
        print(f"  Brier     : {variant_report['brier_mean']:.4f} "
              f"± {variant_report['brier_sd']:.4f}")
        print(f"  ECE       : {variant_report['ece_mean']:.4f} "
              f"± {variant_report['ece_sd']:.4f}")

    # Save JSON
    json_path = out_dir / "calibration_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nCalibration report → {json_path}")

    # LaTeX table
    _write_latex_table(report, out_dir / "calibration_table.tex")


def _write_latex_table(report: Dict, path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Cross-seed calibration metrics (20 seeds, test fold 10). "
        r"Macro-AUC, Brier score and Expected Calibration Error (ECE, 10 bins) "
        r"are reported as mean\,\(\pm\)\,SD across seeds.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Variant & Macro-AUC & Brier & ECE \\",
        r"\hline",
    ]
    for v in VARIANTS:
        if v not in report:
            continue
        r = report[v]
        name_map = {"none": "ECG-only", "demo": "Demo",
                    "demo+anthro": "Full (Demo+Anthro)"}
        row = (
            f"{name_map.get(v, v)} & "
            f"${r['macro_auc_mean']:.4f}\\pm{r['macro_auc_sd']:.4f}$ & "
            f"${r['brier_mean']:.4f}\\pm{r['brier_sd']:.4f}$ & "
            f"${r['ece_mean']:.4f}\\pm{r['ece_sd']:.4f}$ \\\\"
        )
        lines.append(row)
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX calibration table → {path}")


if __name__ == "__main__":
    main()
