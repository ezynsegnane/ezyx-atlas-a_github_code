"""
evaluate_missingness_v2.py — Metadata missingness stress-test for all 3 variants.

For each primary run (variant × seed), the saved checkpoint is reloaded and
evaluated under systematic metadata-zeroing:
  • 0 %  missing    (normal, baseline)
  • 25 % missing    (random MCAR: zero 25% of features per sample)
  • 50 % missing    (random MCAR)
  • 75 % missing    (random MCAR)
  • 100% missing    (disable_meta=True: all zeros)

"Missing" means the feature value AND its availability-mask bit are both set to 0,
mimicking the model's trained behaviour on absent demographic data.

Note on missing-data scope:
  This stress-test simulates MCAR data (features masked uniformly at random).
  The training pipeline does NOT model MAR or MNAR data-generation processes.
  Reviewers should interpret these results as architectural robustness to random
  feature dropout, not generalisation to clinically structured missingness.

Output
------
  results/missingness/missingness_report.json
  results/missingness/missingness_table.tex

Usage
-----
  python evaluate_missingness_v2.py \
      --runs_dir   /kaggle/working/runs \
      --index_path /kaggle/working/index_complete.parquet \
      --data_root  /kaggle/input/ptb-xl/1.0.3 \
      --out_dir    /kaggle/working/results/missingness
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eznx_loader_v2 import EZNXDataset, DS5_LABELS
from eznx_model_v5 import EZNX_ATLAS_A_v5
from sklearn.metrics import roc_auc_score

VARIANTS    = ["none", "demo", "demo+anthro"]
SEEDS_20    = list(range(2024, 2044))
MISS_RATES  = [0.0, 0.25, 0.50, 0.75, 1.0]


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def safe_macro_auc(Y: np.ndarray, P: np.ndarray) -> float:
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) >= 2:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def normalize_ts_voltage(x_ts: torch.Tensor) -> torch.Tensor:
    return x_ts / 5.0


def collate_fn_val(items):
    x_ts  = torch.stack([it["x_ts"] for it in items])
    x_meta = torch.stack([it["x_meta"] for it in items])
    mpm    = torch.stack([it["meta_present_mask"] for it in items])
    y      = torch.stack([it["y"] for it in items])
    x_ts   = normalize_ts_voltage(x_ts)
    return x_ts, x_meta, mpm, y


@torch.no_grad()
def evaluate_with_missingness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    miss_rate: float,
    rng: np.random.Generator,
) -> float:
    """Return macro-AUC under MCAR missingness at rate `miss_rate`."""
    model.eval()
    ys, ps = [], []
    for x_ts, x_meta, mpm, y in loader:
        x_ts   = x_ts.to(device)
        x_meta = x_meta.to(device)
        mpm    = mpm.to(device)

        if miss_rate >= 1.0:
            # Complete metadata disable
            x_meta = torch.zeros_like(x_meta)
            mpm    = torch.zeros_like(mpm)
        elif miss_rate > 0.0:
            # MCAR: randomly zero miss_rate fraction of features per sample
            bsz, fdim = x_meta.shape
            keep_prob = 1.0 - miss_rate
            # Binary keep-mask on CPU, then move to device
            keep = rng.random((bsz, fdim)) < keep_prob
            keep_t = torch.tensor(keep, dtype=x_meta.dtype, device=device)
            x_meta = x_meta * keep_t
            mpm    = mpm    * keep_t

        out = model(x_ts, x_meta, mpm)
        p_blend = torch.sigmoid(out["logits_fused"]).cpu().numpy()
        ys.append(y.numpy())
        ps.append(p_blend)

    return safe_macro_auc(np.concatenate(ys), np.concatenate(ps))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir",   type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--out_dir",    type=str, default=None)
    parser.add_argument("--seeds",      type=int, nargs="+", default=SEEDS_20,
                        help="Which seeds to evaluate (default: all 20)")
    args = parser.parse_args()

    runs_dir   = Path(args.runs_dir)
    index_path = args.index_path
    data_root  = args.data_root
    out_dir    = Path(args.out_dir or PROJECT_ROOT / "results" / "missingness")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: rng is created fresh per (variant, seed) pair — NOT shared globally.
    # This ensures that if a checkpoint is missing and a seed is skipped, all
    # other seeds still receive identical MCAR masks across re-runs of this script.
    # Formula: rng_seed = 42 + variant_index * 100_000 + seed_value
    # (large offset ensures no collision across variants)

    report: Dict[str, Any] = {}

    for variant_idx, variant in enumerate(VARIANTS):
        print(f"\n=== Variant: {variant} ===")
        # variant "none" has no metadata, so MCAR is not meaningful.
        # We still run it to show AUC is constant (invariant to metadata zeroing).

        variant_rows: List[Dict] = []

        # Load test dataset once per variant
        test_ds = EZNXDataset(
            index_file=index_path, data_root=data_root,
            fold=10, sampling_rate=100, meta_mode=variant
        )
        test_loader = DataLoader(
            test_ds, batch_size=64, collate_fn=collate_fn_val, num_workers=0
        )

        miss_aucs_by_rate: Dict[float, List[float]] = {r: [] for r in MISS_RATES}

        for seed in args.seeds:
            run_name = f"ATLAS_A_v5_{variant}_seed{seed}"
            ckpt_path = runs_dir / run_name / f"best_model_{run_name}.pt"
            if not ckpt_path.exists():
                print(f"  MISSING checkpoint: {ckpt_path.name}")
                continue

            # Fresh per-(variant, seed) rng — reproducible regardless of which
            # other seeds have completed.
            seed_rng = np.random.default_rng(42 + variant_idx * 100_000 + seed)

            model = EZNX_ATLAS_A_v5(
                meta_dim=16, n_classes=len(DS5_LABELS), meta_dropout_p=0.10
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            row: Dict[str, Any] = {"seed": seed, "variant": variant}
            for mr in MISS_RATES:
                auc = evaluate_with_missingness(model, test_loader, device,
                                                miss_rate=mr, rng=seed_rng)
                row[f"miss_{int(mr*100):03d}pct"] = auc
                miss_aucs_by_rate[mr].append(auc)

            # AUC drop from 0% to 100% missing
            row["delta_0_to_100"] = row.get("miss_000pct", np.nan) - \
                                    row.get("miss_100pct", np.nan)
            variant_rows.append(row)
            print(f"  seed={seed}  " +
                  " | ".join(f"{mr*100:.0f}%={row.get(f'miss_{int(mr*100):03d}pct', np.nan):.4f}"
                              for mr in MISS_RATES))

        # Aggregate per missingness rate
        variant_summary: Dict[str, Any] = {"n_seeds": len(variant_rows)}
        for mr in MISS_RATES:
            vals = miss_aucs_by_rate[mr]
            key  = f"miss_{int(mr*100):03d}pct"
            arr  = np.array([v for v in vals if not np.isnan(v)])
            if len(arr):
                variant_summary[key] = {
                    "mean": float(arr.mean()),
                    "sd":   float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                }
            else:
                variant_summary[key] = {"mean": None, "sd": None}

        report[variant] = {
            "summary": variant_summary,
            "per_seed": variant_rows,
        }

    # Save
    json_path = out_dir / "missingness_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nMissingness report → {json_path}")

    _write_latex_table(report, out_dir / "missingness_table.tex")


def _write_latex_table(report: Dict, path: Path) -> None:
    name_map = {"none": "ECG-only", "demo": "Demo",
                "demo+anthro": "Full (Demo+Anthro)"}
    miss_labels = {0.0: "0\\%", 0.25: "25\\%", 0.50: "50\\%",
                   0.75: "75\\%", 1.0: "100\\%"}
    col_str = "l" + "c" * len(MISS_RATES)
    header  = "Variant & " + " & ".join(miss_labels.values()) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"\caption{Metadata missingness robustness: macro-AUC (mean\,\(\pm\)\,SD, "
        r"20 seeds) under increasing MCAR missingness rates on the test fold. "
        r"Masks zero both the feature value and its availability flag. "
        r"Results reflect architectural robustness to random feature dropout; "
        r"generalisation to structured (MAR/MNAR) missingness is not evaluated.}",
        r"\label{tab:missingness}",
        rf"\begin{{tabular}}{{{col_str}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for v in VARIANTS:
        if v not in report:
            continue
        s  = report[v]["summary"]
        row = name_map.get(v, v)
        for mr in MISS_RATES:
            key = f"miss_{int(mr*100):03d}pct"
            d = s.get(key, {})
            if d.get("mean") is not None:
                row += f" & ${d['mean']:.4f}\\pm{d['sd']:.4f}$"
            else:
                row += " & ---"
        row += r" \\"
        lines.append(row)
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX missingness table → {path}")


if __name__ == "__main__":
    main()
