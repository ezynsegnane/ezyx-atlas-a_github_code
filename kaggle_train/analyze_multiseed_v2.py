"""
analyze_multiseed_v2.py — Statistical analysis of the 170-run campaign.

Reads all results JSON files, computes group-level statistics, and produces:
  • results_v2/summary_table.json   — machine-readable aggregate
  • results_v2/statistical_tests.json  — Wilcoxon signed-rank + effect sizes
  • results_v2/results_table.tex    — LaTeX table (primary ablation)
  • results_v2/sensitivity_table.tex — LaTeX table (GLU-width / LAUC sensitivity)

Seed selection policy (logged in output):
  Seeds 2024–2043  (20 consecutive, starting year of study initiation).
  No seed was selected or excluded based on performance.

Usage
-----
  python analyze_multiseed_v2.py --runs_dir /kaggle/working/runs \
                                  --out_dir  /kaggle/working/results_v2
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
DS5_LABELS   = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANTS     = ["none", "demo", "demo+anthro"]
SEEDS_20     = list(range(2024, 2044))
SEEDS_10     = list(range(2024, 2034))
META_HIDS    = [64, 128, 256]
LAUC_WS      = [0.00, 0.08, 0.16]     # 0.08 = default from Group A


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _make_run_name(variant: str, seed: int, meta_hid: int = 128,
                   lauc_weight: float = 0.08, no_aug: bool = False) -> str:
    parts = [f"ATLAS_A_v5_{variant}"]
    if meta_hid != 128:
        parts.append(f"metaH{meta_hid}")
    if abs(lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{lauc_weight:g}")
    if no_aug:
        parts.append("noaug")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def load_result(runs_dir: Path, run_name: str) -> Optional[Dict]:
    p = runs_dir / run_name / f"results_{run_name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collect_metric(
    runs_dir: Path, variant: str, seeds: List[int],
    meta_hid: int = 128, lauc_weight: float = 0.08, no_aug: bool = False,
    metric_key: str = "macro_auc",
) -> List[float]:
    vals = []
    for s in seeds:
        name = _make_run_name(variant, s, meta_hid, lauc_weight, no_aug)
        r = load_result(runs_dir, name)
        if r is None:
            continue
        v = r.get("test", {}).get(metric_key)
        if v is not None:
            vals.append(float(v))
    return vals


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def summarise(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": None, "sd": None,
                "min": None, "max": None, "median": None}
    a = np.array(vals)
    return {
        "n":      len(a),
        "mean":   float(a.mean()),
        "sd":     float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min":    float(a.min()),
        "max":    float(a.max()),
        "median": float(np.median(a)),
    }


def wilcoxon_exact(a: List[float], b: List[float]) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (exact permutation distribution, n≤20, no ties).

    method='exact' is available in scipy ≥ 1.7.
    Falls back to 'auto' if n > 25 or version is old.
    """
    if len(a) != len(b) or len(a) < 1:
        return {"stat": None, "p_value": None, "r": None, "note": "insufficient data"}
    diffs = np.array(a) - np.array(b)
    # Remove ties (zero differences)
    diffs = diffs[diffs != 0]
    n     = len(diffs)
    if n < 1:
        return {"stat": None, "p_value": None, "r": None, "note": "all differences zero"}
    method = "exact" if n <= 25 else "auto"
    try:
        res = stats.wilcoxon(np.array(a), np.array(b), method=method)
        stat, p = float(res.statistic), float(res.pvalue)
    except Exception as e:
        return {"stat": None, "p_value": None, "r": None, "note": str(e)}
    # Effect size r = z / sqrt(N)  (matched-pairs convention)
    # Approximate z from T+ statistic under normal approx
    z = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24 + 1e-12)
    r = abs(z) / np.sqrt(n)
    return {"stat": stat, "p_value": p, "r": float(r),
            "n": n, "method": method}


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
    out_dir  = Path(args.out_dir  or PROJECT_ROOT / "results_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "seed_policy": (
            "Seeds 2024–2043 (20 consecutive integers, year of study initiation). "
            "No seed selected or excluded based on performance."
        )
    }

    # ── Group A: Primary ablation ─────────────────────────────────────────────
    print("\n=== Group A: Primary ablation ===")
    group_a: Dict[str, Any] = {}
    variant_aucs: Dict[str, List[float]] = {}

    for variant in VARIANTS:
        vals = collect_metric(runs_dir, variant, SEEDS_20, metric_key="macro_auc")
        delta = collect_metric(runs_dir, variant, SEEDS_20, metric_key="delta_meta_auc")
        s = summarise(vals)
        print(f"  {variant:<15}: {s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        group_a[variant] = {
            "macro_auc": s,
            "delta_meta_auc": summarise(delta),
            "per_class_auc": {},
        }
        variant_aucs[variant] = vals

        # Per-class
        for lbl in DS5_LABELS:
            class_vals: List[float] = []
            for seed in SEEDS_20:
                name = _make_run_name(variant, seed)
                r = load_result(runs_dir, name)
                if r and lbl in r.get("per_class", {}):
                    class_vals.append(float(r["per_class"][lbl]["auc"]))
            group_a[variant]["per_class_auc"][lbl] = summarise(class_vals)

    # Pairwise Wilcoxon tests (Group A)
    print("\n  Pairwise Wilcoxon (macro-AUC, exact):")
    tests_a: Dict[str, Any] = {}
    for v1, v2 in combinations(VARIANTS, 2):
        a, b = variant_aucs.get(v1, []), variant_aucs.get(v2, [])
        n    = min(len(a), len(b))
        res  = wilcoxon_exact(a[:n], b[:n])
        key  = f"{v1}_vs_{v2}"
        tests_a[key] = res
        print(f"  {key:<40}: p={res['p_value']:.4f}  r={res['r']:.3f}")

    summary["group_A"] = {"ablation": group_a, "statistical_tests": tests_a}

    # ── Group B: GLU-width (meta_hid) sensitivity ─────────────────────────────
    print("\n=== Group B: meta_hid sensitivity ===")
    group_b: Dict[str, Any] = {}
    mh_aucs: Dict[int, List[float]] = {}

    for mh in META_HIDS:
        vals = collect_metric(runs_dir, "demo+anthro", SEEDS_20, meta_hid=mh)
        s = summarise(vals)
        print(f"  meta_hid={mh:<5}: {s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        group_b[str(mh)] = {"macro_auc": s}
        mh_aucs[mh] = vals

    # Compare 64 vs 128 and 256 vs 128
    tests_b: Dict[str, Any] = {}
    default_aucs = mh_aucs.get(128, [])
    for mh in [64, 256]:
        a, b = mh_aucs.get(mh, []), default_aucs
        n    = min(len(a), len(b))
        key  = f"metaH{mh}_vs_metaH128"
        res  = wilcoxon_exact(a[:n], b[:n])
        tests_b[key] = res
        print(f"  {key}: p={res['p_value']:.4f}  r={res['r']:.3f}")

    summary["group_B"] = {"meta_hid": group_b, "statistical_tests": tests_b}

    # ── Group C: LAUC-weight sensitivity ─────────────────────────────────────
    print("\n=== Group C: lauc_weight sensitivity ===")
    group_c: Dict[str, Any] = {}
    lauc_aucs: Dict[str, List[float]] = {}

    for lw in LAUC_WS:
        vals = collect_metric(runs_dir, "demo+anthro", SEEDS_20, lauc_weight=lw)
        s = summarise(vals)
        print(f"  lauc_w={lw}: {s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        group_c[str(lw)] = {"macro_auc": s}
        lauc_aucs[str(lw)] = vals

    tests_c: Dict[str, Any] = {}
    default_l = lauc_aucs.get("0.08", [])
    for lw in [0.00, 0.16]:
        a, b = lauc_aucs.get(str(lw), []), default_l
        n    = min(len(a), len(b))
        key  = f"lauc{lw:g}_vs_lauc0.08"
        res  = wilcoxon_exact(a[:n], b[:n])
        tests_c[key] = res
        print(f"  {key}: p={res['p_value']:.4f}  r={res['r']:.3f}")

    summary["group_C"] = {"lauc_weight": group_c, "statistical_tests": tests_c}

    # ── Group D: No-augmentation ──────────────────────────────────────────────
    print("\n=== Group D: no-aug sensitivity ===")
    aug_vals  = collect_metric(runs_dir, "demo+anthro", SEEDS_10)
    noaug_vals = collect_metric(runs_dir, "demo+anthro", SEEDS_10, no_aug=True)
    s_aug   = summarise(aug_vals)
    s_noaug = summarise(noaug_vals)
    print(f"  aug:   {s_aug['mean']:.4f} ± {s_aug['sd']:.4f}  n={s_aug['n']}")
    print(f"  noaug: {s_noaug['mean']:.4f} ± {s_noaug['sd']:.4f}  n={s_noaug['n']}")
    n = min(len(aug_vals), len(noaug_vals))
    test_d = wilcoxon_exact(aug_vals[:n], noaug_vals[:n])
    print(f"  aug_vs_noaug: p={test_d['p_value']:.4f}  r={test_d['r']:.3f}")
    summary["group_D"] = {
        "aug":   {"macro_auc": s_aug},
        "noaug": {"macro_auc": s_noaug},
        "statistical_test": test_d,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    json_path = out_dir / "summary_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_serialise(summary), f, indent=2)
    print(f"\nSummary → {json_path}")

    _write_primary_latex(summary["group_A"]["ablation"], out_dir / "results_table.tex")
    _write_sensitivity_latex(summary, out_dir / "sensitivity_table.tex")


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_primary_latex(ablation: Dict, path: Path) -> None:
    name_map = {"none": "ECG-only", "demo": "Demo",
                "demo+anthro": "Full (Demo+Anthro)"}
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Primary ablation results (macro-AUC, test fold 10). "
        r"20 consecutive seeds 2024–2043; mean\,\(\pm\)\,SD reported.}",
        r"\label{tab:primary_ablation}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Variant & Macro-AUC & $\Delta$AUC (meta) & NORM & MI \\ \hline",
    ]
    for v in VARIANTS:
        if v not in ablation:
            continue
        d = ablation[v]
        auc = d["macro_auc"]
        dm  = d["delta_meta_auc"]
        norm = d["per_class_auc"].get("NORM", {})
        mi   = d["per_class_auc"].get("MI", {})
        row = (
            f"{name_map.get(v,v)} & "
            f"${auc['mean']:.4f}\\pm{auc['sd']:.4f}$ & "
            f"${dm['mean']:+.4f}\\pm{dm['sd']:.4f}$ & "
            f"${norm.get('mean',float('nan')):.4f}\\pm{norm.get('sd',0):.4f}$ & "
            f"${mi.get('mean',float('nan')):.4f}\\pm{mi.get('sd',0):.4f}$ \\\\"
        )
        lines.append(row)
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Primary ablation table → {path}")


def _write_sensitivity_latex(summary: Dict, path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Sensitivity analysis: GLU-width (meta\_hid) and AUC-margin weight "
        r"(\(\lambda_{\text{LAUC}}\)). Full variant (Demo+Anthro), 20 seeds.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{llc}",
        r"\hline",
        r"Parameter & Value & Macro-AUC (mean\,\(\pm\)\,SD) \\ \hline",
    ]
    # meta_hid
    for mh in META_HIDS:
        d = summary.get("group_B", {}).get("meta_hid", {}).get(str(mh), {})
        auc = d.get("macro_auc", {})
        default_mark = r" $\dagger$" if mh == 128 else ""
        if auc.get("mean") is not None:
            lines.append(
                rf"meta\_hid & {mh}{default_mark} & "
                rf"${auc['mean']:.4f}\pm{auc['sd']:.4f}$ \\"
            )
    lines.append(r"\hline")
    # lauc_weight
    for lw in LAUC_WS:
        d = summary.get("group_C", {}).get("lauc_weight", {}).get(str(lw), {})
        auc = d.get("macro_auc", {})
        default_mark = r" $\dagger$" if abs(lw - 0.08) < 1e-6 else ""
        if auc.get("mean") is not None:
            lines.append(
                rf"$\lambda_{{\text{{LAUC}}}}$ & {lw}{default_mark} & "
                rf"${auc['mean']:.4f}\pm{auc['sd']:.4f}$ \\"
            )
    lines += [
        r"\hline",
        r"{\footnotesize $\dagger$ Default value.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Sensitivity table → {path}")


if __name__ == "__main__":
    main()
