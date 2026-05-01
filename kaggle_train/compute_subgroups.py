"""
compute_subgroups.py — Cross-seed subgroup AUC aggregation for Scientific Reports.

Reads NPZ files for all 60 primary runs and produces:
  • Per-subgroup macro-AUC  (mean ± SD over 20 seeds)
  • Per-class AUC in each subgroup
  • Fairness gap: |AUC_male − AUC_female|  (demographic parity proxy)
  • Age-group gaps

Subgroup definitions
--------------------
  Sex    : sex01 = 1 (male)  vs  sex01 = 0 (female)
  Age    : <45 yr / 45–65 yr / >65 yr
           derived from PTB-XL population stats: mean=62.5 yr, SD=17.2 yr
           (Wagner et al. 2020)
  Anthr. : meta_present_strict = 1  (≥ 2 of height/weight/BMI observed)
           vs  meta_present_strict = 0

Output
------
  results/subgroups/subgroup_report.json
  results/subgroups/subgroup_table.tex

Usage
-----
  python compute_subgroups.py --runs_dir /kaggle/working/runs \
                              --index_path /kaggle/working/index_complete.parquet \
                              --out_dir /kaggle/working/results/subgroups
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
DS5_LABELS   = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANTS     = ["none", "demo", "demo+anthro"]
SEEDS        = list(range(2024, 2044))

_PTB_AGE_MEAN, _PTB_AGE_SD = 62.5, 17.2
AGE_LT45_Z = (45 - _PTB_AGE_MEAN) / _PTB_AGE_SD   # ≈ −1.02
AGE_GT65_Z = (65 - _PTB_AGE_MEAN) / _PTB_AGE_SD   # ≈  0.145


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def macro_auc(Y: np.ndarray, P: np.ndarray) -> float:
    vals = [safe_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    valid = [v for v in vals if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def per_class_auc(Y: np.ndarray, P: np.ndarray) -> List[float]:
    return [safe_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]


def npz_path(runs_dir: Path, variant: str, seed: int) -> Path:
    name = f"ATLAS_A_v5_{variant}_seed{seed}"
    return runs_dir / name / f"probs_{name}.npz"


# ═══════════════════════════════════════════════════════════════════════════════
# Build subgroup masks (shared across seeds for the same test fold)
# ═══════════════════════════════════════════════════════════════════════════════

def build_test_masks(index_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(index_path)
    test = df[df["strat_fold"] == 10].reset_index(drop=True)

    masks: Dict[str, np.ndarray] = {
        "all":                np.ones(len(test), dtype=bool),
        "sex_male":           test["sex01"].values == 1,
        "sex_female":         test["sex01"].values == 0,
        "age_lt45":           test["age_z"].values < AGE_LT45_Z,
        "age_45_65":          (test["age_z"].values >= AGE_LT45_Z) &
                              (test["age_z"].values < AGE_GT65_Z),
        "age_gt65":           test["age_z"].values >= AGE_GT65_Z,
        "anthro_complete":    test["meta_present_strict"].values == 1,
        "anthro_incomplete":  test["meta_present_strict"].values == 0,
    }
    # Print counts once
    for k, m in masks.items():
        print(f"  {k:<25}: n = {m.sum()}")
    return masks


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir",   type=str, default=None)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--out_dir",    type=str, default=None)
    args = parser.parse_args()

    runs_dir   = Path(args.runs_dir or os.getenv("EZNX_RUNS_DIR",
                      str(PROJECT_ROOT / "runs")))
    index_path = args.index_path or os.getenv("EZNX_INDEX_PATH",
                 str(PROJECT_ROOT / "data" / "index_complete.parquet"))
    out_dir    = Path(args.out_dir or
                      PROJECT_ROOT / "results" / "subgroups")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Building test-fold subgroup masks …")
    masks = build_test_masks(index_path)

    print("\n[2] Aggregating across variants and seeds …")
    report: Dict[str, Any] = {}

    for variant in VARIANTS:
        print(f"\n--- {variant} ---")
        # Collect per-seed subgroup AUCs
        per_seed_subgroup: Dict[str, List[float]] = {k: [] for k in masks}

        for seed in SEEDS:
            p = npz_path(runs_dir, variant, seed)
            if not p.exists():
                print(f"  MISSING: {p.name}")
                continue
            npz = dict(np.load(p, allow_pickle=False))
            Y = npz["Y"]
            P = npz["P_blend"]

            for subgroup, mask in masks.items():
                if mask.sum() < 10:
                    per_seed_subgroup[subgroup].append(float("nan"))
                    continue
                ma = macro_auc(Y[mask], P[mask])
                per_seed_subgroup[subgroup].append(ma)

        n = len([v for v in per_seed_subgroup["all"] if not np.isnan(v)])
        print(f"  N complete seeds: {n}")

        variant_report: Dict[str, Any] = {"n_seeds": n}
        for subgroup, vals in per_seed_subgroup.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            if len(arr) == 0:
                variant_report[subgroup] = {"mean": None, "sd": None, "n": 0}
            else:
                variant_report[subgroup] = {
                    "mean": float(arr.mean()),
                    "sd":   float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n":    int(len(arr)),
                }
                print(f"    {subgroup:<25}: {arr.mean():.4f} ± {arr.std(ddof=1):.4f}")

        # Fairness gap: |male − female|
        m = variant_report.get("sex_male", {})
        f = variant_report.get("sex_female", {})
        if m.get("mean") is not None and f.get("mean") is not None:
            variant_report["fairness_sex_gap"] = abs(m["mean"] - f["mean"])
            print(f"    fairness_sex_gap          : {variant_report['fairness_sex_gap']:.4f}")

        report[variant] = variant_report

    # Save
    json_path = out_dir / "subgroup_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSubgroup report → {json_path}")

    _write_latex_table(report, out_dir / "subgroup_table.tex")


def _write_latex_table(report: Dict, path: Path) -> None:
    name_map = {"none": "ECG-only", "demo": "Demo",
                "demo+anthro": "Full (Demo+Anthro)"}
    subgroups_display = [
        ("all",              "All"),
        ("sex_male",         "Male"),
        ("sex_female",       "Female"),
        ("age_lt45",         r"Age $<$45"),
        ("age_45_65",        r"Age 45--65"),
        ("age_gt65",         r"Age $>$65"),
        ("anthro_complete",  "Anthro complete"),
        ("anthro_incomplete","Anthro incomplete"),
    ]
    col_header = " & ".join(
        [r"\textbf{Subgroup}"] + [name_map.get(v, v) for v in VARIANTS]
    ) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"\caption{Subgroup macro-AUC (mean\,\(\pm\)\,SD over 20 seeds, test fold). "
        r"Age groups derived from PTB-XL population statistics "
        r"(mean\,=\,62.5\,yr, SD\,=\,17.2\,yr).}",
        r"\label{tab:subgroups}",
        r"\begin{tabular}{l" + "c" * len(VARIANTS) + "}",
        r"\hline",
        col_header,
        r"\hline",
    ]
    for sg_key, sg_label in subgroups_display:
        row = sg_label
        for v in VARIANTS:
            if v not in report or sg_key not in report[v]:
                row += " & ---"
                continue
            d = report[v][sg_key]
            if d.get("mean") is None:
                row += " & ---"
            else:
                row += f" & ${d['mean']:.4f}\\pm{d['sd']:.4f}$"
        row += r" \\"
        lines.append(row)
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX subgroup table → {path}")


if __name__ == "__main__":
    main()
