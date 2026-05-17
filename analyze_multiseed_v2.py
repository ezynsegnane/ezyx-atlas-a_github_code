"""
analyze_multiseed_v2.py — Statistical analysis of the EZNX-ATLAS-A v5 campaign.

Reads all results JSON files, computes group-level statistics, and produces:
  results_v2/summary_table.json       — machine-readable aggregate
  results_v2/results_table.tex        — primary LaTeX table (Group A, confirmatory)
  results_v2/sensitivity_table.tex    — Group B / C exploratory sensitivity
  results_v2/group_e_table.tex        — Group E architecture ablations
  results_v2/group_f_table.tex        — Group F multi-split sensitivity (fold-level)

Analysis plan:
  PRIMARY ENDPOINT (confirmatory): macro-AUC Group A, demo+anthro vs none, fold 10,
  Wilcoxon signed-rank n=20 with BH-FDR correction (analysis_plan.md).
  Groups B, C, D, E: exploratory (no multiplicity-adjusted claims).
  Group F: fold-level analysis only — no statistical test across 20 runs as if independent.
  Post-hoc power analysis: labeled as such, appendix only.

Usage
-----
  python analyze_multiseed_v2.py --runs_dir /kaggle/working/runs \\
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
SEEDS_5      = list(range(2024, 2029))
META_HIDS    = [32, 64, 128, 256]
LAUC_WS      = [0.0, 0.04, 0.08, 0.12, 0.16]
ARCH_MODES   = ["meta_only", "concat", "additive", "no_glu",
                "no_residual", "no_q_meta", "no_dropout"]
GROUP_F_SPLITS = [
    {"test_fold": 2, "val_fold": 3},
    {"test_fold": 3, "val_fold": 4},
    {"test_fold": 7, "val_fold": 8},
    {"test_fold": 8, "val_fold": 9},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Run name / loader helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_run_name(
    variant: str,
    seed: int,
    meta_hid: int = 128,
    lauc_weight: float = 0.08,
    no_aug: bool = False,
    arch_mode: str = "standard",
    meta_mask_mode: str = "real",
    test_fold: int = 10,
    val_fold: int = 9,
) -> str:
    """Mirror of make_run_name() in atlas_a_v5_multiseed_v2.py."""
    parts = [f"ATLAS_A_v5_{variant}"]
    if meta_hid != 128:
        parts.append(f"metaH{meta_hid}")
    if abs(lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{lauc_weight:g}")
    if no_aug:
        parts.append("noaug")
    if arch_mode != "standard":
        parts.append(arch_mode)
    if meta_mask_mode == "trainmask":
        parts.append("trainmask")
    if test_fold != 10 or val_fold != 9:
        parts.append(f"tf{test_fold}_vf{val_fold}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def load_result(runs_dir: Path, run_name: str) -> Optional[Dict]:
    p = runs_dir / run_name / f"results_{run_name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def collect_metric(
    runs_dir: Path,
    variant: str,
    seeds: List[int],
    meta_hid: int = 128,
    lauc_weight: float = 0.08,
    no_aug: bool = False,
    arch_mode: str = "standard",
    meta_mask_mode: str = "real",
    test_fold: int = 10,
    val_fold: int = 9,
    metric_key: str = "macro_auc",
) -> List[float]:
    vals: List[float] = []
    for s in seeds:
        name = _make_run_name(variant, s, meta_hid, lauc_weight, no_aug,
                              arch_mode, meta_mask_mode, test_fold, val_fold)
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

def summarise(vals: List[float], bootstrap_n: int = 2000) -> Dict[str, Any]:
    """
    Descriptive statistics with t-CI (primary) and bootstrap CI (secondary).
    Pre-declared in analysis_plan.md: t-CI primary, bootstrap secondary.
    """
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "ci95_t": None,
                "ci95_lo": None, "ci95_hi": None,
                "ci95_boot_lo": None, "ci95_boot_hi": None,
                "min": None, "max": None, "median": None}
    a  = np.array(vals, dtype=float)
    n  = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    mean = float(a.mean())

    # t-CI (primary)
    if n > 1:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci95_t = t_crit * sd / float(np.sqrt(n))
    else:
        ci95_t = 0.0

    # Bootstrap CI (secondary, percentile method)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(a, size=n, replace=True).mean() for _ in range(bootstrap_n)
    ])
    ci95_boot_lo = float(np.percentile(boot_means, 2.5))
    ci95_boot_hi = float(np.percentile(boot_means, 97.5))

    return {
        "n":            n,
        "mean":         mean,
        "sd":           sd,
        "ci95_t":       ci95_t,
        "ci95_lo":      mean - ci95_t,
        "ci95_hi":      mean + ci95_t,
        "ci95_boot_lo": ci95_boot_lo,
        "ci95_boot_hi": ci95_boot_hi,
        "min":          float(a.min()),
        "max":          float(a.max()),
        "median":       float(np.median(a)),
    }


def wilcoxon_exact(a: List[float], b: List[float]) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test implementing the pre-declared fallback rule
    from analysis_plan.md:
      method='exact'  if n_nonzero <= 25 and no zero differences
      method='approx' with continuity_correction=True otherwise
    """
    if len(a) != len(b) or len(a) < 1:
        return {"stat": None, "p_value": None, "r": None,
                "note": "insufficient data"}

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    diffs = arr_a - arr_b
    n_zero    = int((diffs == 0).sum())
    n_nonzero = int((diffs != 0).sum())

    if n_nonzero < 1:
        return {"stat": None, "p_value": 1.0, "r": 0.0,
                "n_pairs": len(a), "n_nonzero": 0, "n_zero": n_zero,
                "method": "trivial", "note": "all differences zero (p=1.0)"}

    # Pre-declared fallback rule
    has_zeros = n_zero > 0
    if (not has_zeros) and n_nonzero <= 25:
        method = "exact"
        extra  = {}
    else:
        method = "approx"
        extra  = {"correction": True}

    try:
        res  = stats.wilcoxon(arr_a, arr_b, method=method, **extra)
        stat = float(res.statistic)
        p    = float(res.pvalue)
    except Exception as exc:
        return {"stat": None, "p_value": None, "r": None, "note": str(exc)}

    # Effect size r = |z| / sqrt(n_nonzero)
    n = n_nonzero
    mu_t  = n * (n + 1) / 4.0
    sig_t = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 + 1e-12)
    z     = (stat - mu_t) / sig_t
    r     = abs(float(z)) / float(np.sqrt(n))

    return {
        "stat":      stat,
        "p_value":   p,
        "r":         r,
        "n_pairs":   len(a),
        "n_nonzero": n_nonzero,
        "n_zero":    n_zero,
        "method":    method,
    }


def bh_fdr(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg FDR correction. Returns rejected[i] for each test."""
    m  = len(p_values)
    if m == 0:
        return []
    idx  = np.argsort(p_values)
    rank = np.arange(1, m + 1)
    thresh = alpha * rank / m
    sorted_p = np.array(p_values)[idx]
    rejected_sorted = sorted_p <= thresh
    # Accumulate: if rank k is rejected, all ranks < k are also rejected
    for i in range(m - 2, -1, -1):
        if rejected_sorted[i + 1]:
            rejected_sorted[i] = True
    # Restore original order
    out = np.zeros(m, dtype=bool)
    out[idx] = rejected_sorted
    return out.tolist()


def post_hoc_power_note() -> str:
    """
    Returns the post-hoc power note for appendix inclusion.
    Pre-declared as APPENDIX ONLY in analysis_plan.md.
    """
    return (
        "Post-hoc power analysis (appendix): computed after observing effect sizes, "
        "therefore estimates are conditional on observed data and should not be "
        "interpreted as pre-study power. Reported for descriptive purposes only."
    )


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
    out_dir  = Path(args.out_dir or PROJECT_ROOT / "results_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "analysis_plan": "analysis_plan.md (committed before any training run)",
        "primary_endpoint": (
            "macro-AUC Group A, demo+anthro vs none, test fold 10, "
            "Wilcoxon signed-rank n=20, BH-FDR alpha=0.05"
        ),
        "seed_policy": (
            "Seeds 2024–2043 (20 consecutive, year of study initiation). "
            "No seed was selected or excluded based on performance."
        ),
        "w_fused": "1.0 (fixed, pre-declared in analysis_plan.md)",
        "checkpoint_selection": "val macro-AUC only (no delta_meta tie-break, pre-declared)",
        "post_hoc_power_note": post_hoc_power_note(),
    }

    # ── Group A: Primary confirmatory ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUP A — Primary confirmatory (fold 10, 20 seeds) [CONFIRMATORY]")
    print("=" * 70)
    group_a: Dict[str, Any] = {}
    variant_aucs: Dict[str, List[float]] = {}

    for variant in VARIANTS:
        vals   = collect_metric(runs_dir, variant, SEEDS_20, metric_key="macro_auc")
        delta  = collect_metric(runs_dir, variant, SEEDS_20, metric_key="delta_meta_auc")
        f1_opt = collect_metric(runs_dir, variant, SEEDS_20, metric_key="macro_f1_optimal")
        f1_05  = collect_metric(runs_dir, variant, SEEDS_20, metric_key="macro_f1_fixed_05")
        s      = summarise(vals)
        sf1    = summarise(f1_opt)
        print(f"  {variant:<15}: AUC {s['mean']:.4f} ± {s['sd']:.4f} "
              f"[{s['ci95_lo']:.4f},{s['ci95_hi']:.4f}]  "
              f"F1 {sf1['mean']:.4f} ± {sf1['sd']:.4f}  n={s['n']}")
        group_a[variant] = {
            "macro_auc":           s,
            "delta_meta_auc":      summarise(delta),
            "macro_f1_optimal":    sf1,
            "macro_f1_fixed_05":   summarise(f1_05),
            "per_class_auc":       {},
        }
        variant_aucs[variant] = vals

        # Per-class AUC
        for lbl in DS5_LABELS:
            class_vals: List[float] = []
            for seed in SEEDS_20:
                name = _make_run_name(variant, seed)
                r = load_result(runs_dir, name)
                if r and lbl in r.get("per_class", {}):
                    class_vals.append(float(r["per_class"][lbl]["auc"]))
            group_a[variant]["per_class_auc"][lbl] = summarise(class_vals)

    # Pairwise Wilcoxon tests (confirmatory) + BH-FDR
    print("\n  Pairwise Wilcoxon (macro-AUC, pre-declared method):")
    pairs_a    = list(combinations(VARIANTS, 2))
    tests_a    = {}
    p_vals_a   = []
    for v1, v2 in pairs_a:
        a, b = variant_aucs.get(v1, []), variant_aucs.get(v2, [])
        n    = min(len(a), len(b))
        res  = wilcoxon_exact(a[:n], b[:n])
        key  = f"{v1}_vs_{v2}"
        tests_a[key] = res
        p_vals_a.append(res["p_value"] if res["p_value"] is not None else 1.0)

    # BH-FDR (pre-declared, alpha=0.05)
    rejected = bh_fdr(p_vals_a, alpha=0.05)
    for i, (pair, (v1, v2)) in enumerate(zip(pairs_a, pairs_a)):
        key  = f"{v1}_vs_{v2}"
        res  = tests_a[key]
        rej  = rejected[i]
        tests_a[key]["bh_rejected"] = rej
        p    = res["p_value"]
        r    = res["r"]
        pstr = f"{p:.4f}" if p is not None else "N/A"
        rstr = f"{r:.3f}" if r is not None else "N/A"
        print(f"  {key:<40}: p={pstr}  r={rstr}  BH-rejected={rej}")

    summary["group_A"] = {"ablation": group_a, "statistical_tests": tests_a,
                          "bh_alpha": 0.05}

    # ── Group B: Fusion capacity sensitivity ────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUP B — Fusion capacity sensitivity (meta_hid) [EXPLORATORY]")
    print("  Note: meta_hid controls both projection dim AND gate hidden dim (entangled).")
    print("=" * 70)
    group_b: Dict[str, Any] = {}
    mh_aucs:  Dict[int, List[float]] = {}
    for mh in META_HIDS:
        vals = collect_metric(runs_dir, "demo+anthro", SEEDS_10, meta_hid=mh)
        s    = summarise(vals)
        print(f"  meta_hid={mh:<5}: {s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        group_b[str(mh)] = {"macro_auc": s}
        mh_aucs[mh] = vals

    # Compare each vs default (128) — exploratory, no FDR correction
    tests_b: Dict[str, Any] = {}
    default_mh = mh_aucs.get(128, [])
    for mh in [32, 64, 256]:
        a, b = mh_aucs.get(mh, []), default_mh
        n    = min(len(a), len(b))
        key  = f"metaH{mh}_vs_metaH128"
        res  = wilcoxon_exact(a[:n], b[:n])
        tests_b[key] = res
        p = res["p_value"]; r = res["r"]
        pstr = f"{p:.4f}" if p is not None else "N/A"
        rstr = f"{r:.3f}" if r is not None else "N/A"
        print(f"  {key} [exploratory]: p={pstr}  r={rstr}")

    summary["group_B"] = {"meta_hid": group_b, "statistical_tests": tests_b,
                          "role": "exploratory"}

    # ── Group C: LAUC-BCE balance sensitivity ────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUP C — LAUC-BCE balance sensitivity [EXPLORATORY]")
    print("  Note: fused_w = max(0, 0.60 - lauc_w). Both weights change simultaneously.")
    print("=" * 70)
    group_c: Dict[str, Any] = {}
    lauc_aucs: Dict[str, List[float]] = {}
    for lw in LAUC_WS:
        vals = collect_metric(runs_dir, "demo+anthro", SEEDS_10, lauc_weight=lw)
        s    = summarise(vals)
        print(f"  lauc_w={lw}: {s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        group_c[f"{lw:g}"] = {"macro_auc": s}
        lauc_aucs[f"{lw:g}"] = vals

    tests_c: Dict[str, Any] = {}
    default_l = lauc_aucs.get("0.08", [])
    for lw in [0.0, 0.04, 0.12, 0.16]:
        a, b = lauc_aucs.get(f"{lw:g}", []), default_l
        n    = min(len(a), len(b))
        key  = f"lauc{lw:g}_vs_lauc0.08"
        res  = wilcoxon_exact(a[:n], b[:n])
        tests_c[key] = res
        p = res["p_value"]; r = res["r"]
        pstr = f"{p:.4f}" if p is not None else "N/A"
        rstr = f"{r:.3f}" if r is not None else "N/A"
        print(f"  {key} [exploratory]: p={pstr}  r={rstr}")

    summary["group_C"] = {"lauc_weight": group_c, "statistical_tests": tests_c,
                          "role": "exploratory",
                          "coupling_note": "fused_w = max(0, 0.60 - lauc_w) — pre-declared"}

    # ── Group D: Augmentation sensitivity ───────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUP D — Augmentation sensitivity [EXPLORATORY]")
    print("=" * 70)
    aug_vals   = collect_metric(runs_dir, "demo+anthro", SEEDS_10)
    noaug_vals = collect_metric(runs_dir, "demo+anthro", SEEDS_10, no_aug=True)
    s_aug      = summarise(aug_vals)
    s_noaug    = summarise(noaug_vals)
    print(f"  aug:   {s_aug['mean']:.4f} ± {s_aug['sd']:.4f}  n={s_aug['n']}")
    print(f"  noaug: {s_noaug['mean']:.4f} ± {s_noaug['sd']:.4f}  n={s_noaug['n']}")
    n      = min(len(aug_vals), len(noaug_vals))
    test_d = wilcoxon_exact(aug_vals[:n], noaug_vals[:n])
    p = test_d["p_value"]; r = test_d["r"]
    pstr = f"{p:.4f}" if p is not None else "N/A"
    rstr = f"{r:.3f}" if r is not None else "N/A"
    print(f"  aug_vs_noaug [exploratory]: p={pstr}  r={rstr}")
    summary["group_D"] = {
        "aug":   {"macro_auc": s_aug},
        "noaug": {"macro_auc": s_noaug},
        "statistical_test": test_d,
        "role": "exploratory",
    }

    # ── Group E: Architecture ablations ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUP E — Architecture ablations [EXPLORATORY]")
    print("=" * 70)
    group_e: Dict[str, Any] = {}

    # E1-E7: arch modes
    for am in ARCH_MODES:
        vals = collect_metric(runs_dir, "demo+anthro", SEEDS_10, arch_mode=am)
        s    = summarise(vals)
        print(f"  arch={am:<14}: {s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        group_e[am] = {"macro_auc": s}

    # E8: trainmask
    vals_tm = collect_metric(runs_dir, "demo+anthro", SEEDS_10,
                              meta_mask_mode="trainmask")
    s_tm = summarise(vals_tm)
    print(f"  arch=trainmask    : {s_tm['mean']:.4f} ± {s_tm['sd']:.4f}  n={s_tm['n']}")
    group_e["trainmask"] = {"macro_auc": s_tm}

    # Reference: standard (Group A, seeds 2024-2033)
    vals_std = collect_metric(runs_dir, "demo+anthro", SEEDS_10)
    s_std    = summarise(vals_std)
    print(f"  arch=standard(ref): {s_std['mean']:.4f} ± {s_std['sd']:.4f}  n={s_std['n']}")
    group_e["standard_ref"] = {"macro_auc": s_std}

    # Compare each mode vs standard [exploratory]
    tests_e: Dict[str, Any] = {}
    all_modes_e = ARCH_MODES + ["trainmask"]
    for am in all_modes_e:
        a    = group_e[am]["macro_auc"]  # summary
        vals_am = collect_metric(
            runs_dir, "demo+anthro", SEEDS_10,
            arch_mode=am if am != "trainmask" else "standard",
            meta_mask_mode="trainmask" if am == "trainmask" else "real",
        )
        n = min(len(vals_am), len(vals_std))
        res = wilcoxon_exact(vals_am[:n], vals_std[:n])
        key = f"{am}_vs_standard"
        tests_e[key] = res
        p = res["p_value"]; r = res["r"]
        pstr = f"{p:.4f}" if p is not None else "N/A"
        rstr = f"{r:.3f}" if r is not None else "N/A"
        print(f"  {key:<35} [exploratory]: p={pstr}  r={rstr}")

    summary["group_E"] = {
        "ablations": group_e,
        "statistical_tests": tests_e,
        "role": "exploratory",
    }

    # ── Group F: Multi-split sensitivity ────────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUP F — Multi-split sensitivity [CONFIRMATORY — generalization check]")
    print("  Analysis unit: fold (4 units). 5 seeds per fold.")
    print("  NO statistical test across 20 runs as if independent.")
    print("=" * 70)
    group_f: Dict[str, Any] = {}

    fold_means: List[float] = []
    for sp in GROUP_F_SPLITS:
        tf  = sp["test_fold"]
        vf  = sp["val_fold"]
        vals = collect_metric(
            runs_dir, "demo+anthro", SEEDS_5,
            test_fold=tf, val_fold=vf,
        )
        s    = summarise(vals)
        fold_key = f"tf{tf}_vf{vf}"
        group_f[fold_key] = {"test_fold": tf, "val_fold": vf, "macro_auc": s}
        print(f"  fold (tf={tf},vf={vf}): mean={s['mean']:.4f} ± {s['sd']:.4f}  n={s['n']}")
        if s["mean"] is not None:
            fold_means.append(s["mean"])

    # Group A fold 10 reference
    ref_vals = collect_metric(runs_dir, "demo+anthro", SEEDS_20)
    ref_s    = summarise(ref_vals)
    group_f["group_A_reference"] = {
        "test_fold": 10, "val_fold": 9,
        "macro_auc": ref_s,
    }
    print(f"  Group A (tf=10): mean={ref_s['mean']:.4f} ± {ref_s['sd']:.4f}  n={ref_s['n']}")

    # Inter-fold summary (fold-level, no test)
    if fold_means:
        group_f["inter_fold_summary"] = {
            "fold_means":   fold_means,
            "inter_fold_mean": float(np.mean(fold_means)),
            "inter_fold_range_lo": float(np.min(fold_means)),
            "inter_fold_range_hi": float(np.max(fold_means)),
            "note": (
                "Group A fold 10 lies within the inter-fold range if "
                f"{ref_s['mean']:.4f} is within "
                f"[{np.min(fold_means):.4f}, {np.max(fold_means):.4f}]. "
                "Reported per analysis_plan.md: 'Group F assesses whether fold 10 "
                "lies within the alternate-split performance range. If fold 10 falls "
                "outside this range, this is reported as-is and discussed.'"
            ),
        }
        print(f"\n  Inter-fold range: [{np.min(fold_means):.4f}, {np.max(fold_means):.4f}]")
        in_range = np.min(fold_means) <= ref_s["mean"] <= np.max(fold_means)
        print(f"  Group A fold-10 ({ref_s['mean']:.4f}) in range: {in_range}")

    summary["group_F"] = group_f

    # ── Serialise and write ──────────────────────────────────────────────────
    def _ser(obj):
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_ser(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    json_path = out_dir / "summary_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_ser(summary), f, indent=2)
    print(f"\nSummary JSON → {json_path}")

    _write_primary_latex(summary["group_A"]["ablation"],
                         summary["group_A"]["statistical_tests"],
                         out_dir / "results_table.tex")
    _write_sensitivity_latex(summary, out_dir / "sensitivity_table.tex")
    _write_group_e_latex(summary["group_E"]["ablations"],
                         out_dir / "group_e_table.tex")
    _write_group_f_latex(summary["group_F"], out_dir / "group_f_table.tex")

    print(f"\nAll tables written to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(mean, sd, ci_lo=None, ci_hi=None) -> str:
    if mean is None:
        return "---"
    s = f"${mean:.4f}\\pm{sd:.4f}$"
    if ci_lo is not None and ci_hi is not None:
        s += f" $[{ci_lo:.4f},{ci_hi:.4f}]$"
    return s


def _write_primary_latex(ablation: Dict, tests: Dict, path: Path) -> None:
    name_map = {"none": "ECG-only", "demo": "Demo",
                "demo+anthro": "Full (Demo\\,+\\,Anthro)"}
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Primary confirmatory results (Group~A, test fold~10, seeds 2024--2043, "
        r"$n{=}20$). Macro-AUC and Macro-F1: mean\,$\pm$\,SD with 95\%\ CI "
        r"(Student $t$-distribution). $\Delta$AUC\,(meta): gain over ECG-only variant. "
        r"Pairwise Wilcoxon signed-rank tests (exact, $n{=}20$) with "
        r"Benjamini--Hochberg FDR correction ($\alpha{=}0.05$).}",
        r"\label{tab:primary_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Variant & Macro-AUC & 95\%\ CI & Macro-F1 & $\Delta$AUC & MI-AUC \\",
        r"\hline",
    ]
    for v in VARIANTS:
        if v not in ablation:
            continue
        d    = ablation[v]
        auc  = d["macro_auc"]
        f1   = d.get("macro_f1_optimal", {})
        dm   = d["delta_meta_auc"]
        mi   = d["per_class_auc"].get("MI", {})
        mean = auc.get("mean") or 0.0
        sd   = auc.get("sd") or 0.0
        lo   = auc.get("ci95_lo") or mean
        hi   = auc.get("ci95_hi") or mean
        row = (
            f"{name_map.get(v, v)} & "
            f"${mean:.4f}\\pm{sd:.4f}$ & "
            f"$[{lo:.4f},{hi:.4f}]$ & "
            f"${(f1.get('mean') or 0):.4f}\\pm{(f1.get('sd') or 0):.4f}$ & "
            f"${(dm.get('mean') or 0):+.4f}\\pm{(dm.get('sd') or 0):.4f}$ & "
            f"${(mi.get('mean') or 0):.4f}\\pm{(mi.get('sd') or 0):.4f}$ \\\\"
        )
        lines.append(row)
    lines.append(r"\hline")
    # Pairwise test footnotes
    lines.append(r"\multicolumn{6}{l}{\footnotesize Wilcoxon BH-FDR:}")
    for key, res in tests.items():
        p   = res.get("p_value")
        rej = res.get("bh_rejected", False)
        r   = res.get("r")
        star = r"$^{*}$" if rej else ""
        if p is not None:
            key_tex = key.replace('_', r'\_')
            lines.append(
                rf"\multicolumn{{6}}{{l}}{{\footnotesize "
                rf"  {key_tex}: $p={p:.4f}${star}, $r={r:.3f}$}}"
                r"\\"
            )
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Primary table → {path}")


def _write_sensitivity_latex(summary: Dict, path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Exploratory sensitivity: fusion capacity (Group~B, meta\_hid "
        r"controls projection and gate hidden dim jointly) and LAUC-BCE balance "
        r"(Group~C; note: $w_{\text{fused}} = \max(0, 0.60 - \lambda_{\text{LAUC}})$, "
        r"both weights change simultaneously). Demo+Anthro, 10 seeds. "
        r"$\dagger$ = default value.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"Group & Parameter & Value & Macro-AUC (mean\,$\pm$\,SD) \\ \hline",
    ]
    # Group B
    for mh in META_HIDS:
        d    = summary.get("group_B", {}).get("meta_hid", {}).get(str(mh), {})
        auc  = d.get("macro_auc", {})
        dagger = r" $\dagger$" if mh == 128 else ""
        if auc.get("mean") is not None:
            lines.append(
                rf"B & meta\_hid & {mh}{dagger} & "
                rf"${auc['mean']:.4f}\pm{auc['sd']:.4f}$ \\"
            )
    lines.append(r"\hline")
    # Group C
    for lw in LAUC_WS:
        d    = summary.get("group_C", {}).get("lauc_weight", {}).get(f"{lw:g}", {})
        auc  = d.get("macro_auc", {})
        dagger = r" $\dagger$" if abs(lw - 0.08) < 1e-6 else ""
        if auc.get("mean") is not None:
            lines.append(
                rf"C & $\lambda_{{\text{{LAUC}}}}$ & {lw}{dagger} & "
                rf"${auc['mean']:.4f}\pm{auc['sd']:.4f}$ \\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Sensitivity table → {path}")


def _write_group_e_latex(ablations: Dict, path: Path) -> None:
    mode_labels = {
        "standard_ref": "Standard (reference, Group~A)",
        "meta_only":    "E1: Meta-only fusion",
        "concat":       "E2: Concatenation (no gate)",
        "additive":     "E3: Additive injection (no gate)",
        "no_glu":       "E4: ReLU gate (no sigmoid)",
        "no_residual":  "E5: No residual injection",
        "no_q_meta":    "E6: No availability signal ($q=1$)",
        "no_dropout":   "E7: No feature dropout",
        "trainmask":    "E8: Trainmask (zero meta training)",
    }
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Architecture ablations (Group~E, exploratory). "
        r"Demo+Anthro variant, 10 seeds. "
        r"E8 (trainmask): standard architecture trained with $\mathbf{x}_{\text{meta}}=\mathbf{0}$, "
        r"$\mathbf{m}=\mathbf{0}$; evaluated with real metadata.}",
        r"\label{tab:group_e}",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"Configuration & Macro-AUC (mean\,$\pm$\,SD) \\ \hline",
    ]
    order = ["standard_ref", "meta_only", "concat", "additive", "no_glu",
             "no_residual", "no_q_meta", "no_dropout", "trainmask"]
    for key in order:
        if key not in ablations:
            continue
        auc = ablations[key].get("macro_auc", {})
        label = mode_labels.get(key, key)
        if auc.get("mean") is not None:
            lines.append(
                rf"{label} & ${auc['mean']:.4f}\pm{auc['sd']:.4f}$ \\"
            )
        else:
            lines.append(rf"{label} & --- \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Group E table → {path}")


def _write_group_f_latex(group_f: Dict, path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Multi-split sensitivity analysis (Group~F, confirmatory "
        r"generalization check). Demo+Anthro model, 5 seeds per split. "
        r"Fold selection rule: two early splits (2,3) and two mid-range splits (7,8); "
        r"val\,=\,test\,+\,1. Analysis unit: fold (4 units). "
        r"No statistical test across 20 runs as if independent (pre-declared). "
        r"Group~A (fold~10) reference included for comparison.}",
        r"\label{tab:group_f}",
        r"\begin{tabular}{cclc}",
        r"\hline",
        r"Test fold & Val fold & $n$ seeds & Macro-AUC (mean\,$\pm$\,SD) \\ \hline",
    ]
    split_order = ["tf2_vf3", "tf3_vf4", "tf7_vf8", "tf8_vf9"]
    for key in split_order:
        d = group_f.get(key, {})
        auc = d.get("macro_auc", {})
        if auc.get("mean") is not None:
            lines.append(
                rf"{d.get('test_fold', '?')} & {d.get('val_fold', '?')} & "
                rf"{auc['n']} & ${auc['mean']:.4f}\pm{auc['sd']:.4f}$ \\"
            )
    lines.append(r"\hline")
    # Group A reference
    ref = group_f.get("group_A_reference", {})
    auc_ref = ref.get("macro_auc", {})
    if auc_ref.get("mean") is not None:
        lines.append(
            rf"10 (Group~A) & 9 & {auc_ref['n']} & "
            rf"${auc_ref['mean']:.4f}\pm{auc_ref['sd']:.4f}$ (reference) \\"
        )
    # Inter-fold summary
    ifs = group_f.get("inter_fold_summary", {})
    if ifs:
        lo = ifs.get("inter_fold_range_lo")
        hi = ifs.get("inter_fold_range_hi")
        mu = ifs.get("inter_fold_mean")
        if None not in (lo, hi, mu):
            lines.append(r"\hline")
            lines.append(
                rf"\multicolumn{{4}}{{l}}{{\footnotesize "
                rf"Inter-fold range: [{lo:.4f}, {hi:.4f}]; "
                rf"mean across folds: {mu:.4f}}} \\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Group F table → {path}")


if __name__ == "__main__":
    main()
