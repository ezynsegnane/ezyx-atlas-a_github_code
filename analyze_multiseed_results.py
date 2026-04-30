# ============================================================================
# analyze_multiseed_results.py - Analyse statistique des résultats multi-graines
# ============================================================================
# Calcule: moyenne ± écart-type, IC 95% (bootstrap), tests Wilcoxon
# Génère: tableaux LaTeX et Markdown pour l'article
# ============================================================================

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import warnings

import numpy as np
from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

VARIANTS = ["none", "demo", "demo+anthro"]
VARIANT_LABELS = {
    "none": "ECG seul",
    "demo": "ECG + demo",
    "demo+anthro": "ECG + complet"
}
DS5_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
TEST_PAIRS = [
    ("none", "demo"),
    ("none", "demo+anthro"),
    ("demo", "demo+anthro"),
]
KEY_METRICS = (
    ["macro_auc", "macro_f1_optimal"]
    + [f"auc_{c}" for c in DS5_LABELS]
    + [f"f1_{c}" for c in DS5_LABELS]
)

# Nombre d'itérations bootstrap
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
BH_FDR_ALPHA = 0.05
WILCOXON_MIN_PAIRS = 5


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_all_results(runs_dir: Path) -> Dict[str, Dict[int, Dict]]:
    """
    Charge tous les fichiers de résultats JSON.
    
    Returns:
        Dict[variant][seed] = results_dict
    """
    results = defaultdict(dict)
    
    for variant in VARIANTS:
        pattern = f"ATLAS_A_v5_{variant}_seed*"
        for run_dir in runs_dir.glob(pattern):
            # Extraire le seed du nom du répertoire
            try:
                seed = int(run_dir.name.split("_seed")[-1])
            except:
                continue
            
            # Chercher le fichier de résultats
            results_file = run_dir / f"results_{variant}_seed{seed}.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Vérifier que les résultats test sont présents
                if "test" in data and "macro_auc" in data["test"]:
                    metadata = data.get("metadata", {})
                    if metadata.get("variant") not in (None, variant):
                        warnings.warn(
                            f"Variant mismatch in {results_file}: "
                            f"folder={variant}, json={metadata.get('variant')}"
                        )
                    if metadata.get("seed") not in (None, seed):
                        warnings.warn(
                            f"Seed mismatch in {results_file}: "
                            f"folder={seed}, json={metadata.get('seed')}"
                        )
                    results[variant][seed] = data

        # Fallback: flat JSON layout used by the archived release (results/seed_json/).
        # File naming convention: results_{variant}_seed{seed}.json directly inside
        # runs_dir (no subdirectory wrapper).
        if not results[variant]:
            for json_file in sorted(runs_dir.glob(f"results_{variant}_seed*.json")):
                try:
                    seed = int(json_file.stem.split("_seed")[-1])
                except Exception:
                    continue
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "test" in data and "macro_auc" in data["test"]:
                    metadata = data.get("metadata", {})
                    if metadata.get("variant") not in (None, variant):
                        warnings.warn(
                            f"Variant mismatch in {json_file}: "
                            f"expected={variant}, json={metadata.get('variant')}"
                        )
                    if metadata.get("seed") not in (None, seed):
                        warnings.warn(
                            f"Seed mismatch in {json_file}: "
                            f"expected={seed}, json={metadata.get('seed')}"
                        )
                    results[variant][seed] = data

    return dict(results)


def extract_metrics(
    results: Dict[str, Dict[int, Dict]]
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Extrait les métriques pour analyse statistique.
    
    Returns:
        metrics: Dict[variant]["metric_name"] = array of values across sorted seeds
        seed_orders: Dict[variant] = sorted seeds matching each metric array
    """
    metrics = defaultdict(lambda: defaultdict(list))
    seed_orders = {}
    
    for variant, seed_results in results.items():
        seed_orders[variant] = []
        for seed, data in sorted(seed_results.items()):
            seed_orders[variant].append(seed)
            test = data.get("test", {})
            per_class = data.get("per_class", {})
            
            # Métriques globales
            metrics[variant]["macro_auc"].append(test.get("macro_auc", np.nan))
            metrics[variant]["macro_f1_optimal"].append(test.get("macro_f1_optimal", np.nan))
            metrics[variant]["macro_f1_fixed"].append(test.get("macro_f1_fixed", np.nan))
            metrics[variant]["auc_ecg_only"].append(test.get("auc_ecg_only", np.nan))
            metrics[variant]["auc_fused_only"].append(test.get("auc_fused_only", np.nan))
            metrics[variant]["delta_meta_auc"].append(test.get("delta_meta_auc", np.nan))
            metrics[variant]["w_fused"].append(test.get("w_fused", np.nan))
            
            # Métriques par classe
            for cls in DS5_LABELS:
                cls_data = per_class.get(cls, {})
                metrics[variant][f"auc_{cls}"].append(cls_data.get("auc", np.nan))
                metrics[variant][f"f1_{cls}"].append(cls_data.get("f1", np.nan))
    
    # Convertir en arrays numpy
    metrics_arrays = {
        variant: {metric: np.array(values) for metric, values in metrics_dict.items()}
        for variant, metrics_dict in metrics.items()
    }
    seed_order_arrays = {
        variant: np.array(seeds, dtype=int) for variant, seeds in seed_orders.items()
    }
    return metrics_arrays, seed_order_arrays


# ============================================================================
# STATISTIQUES
# ============================================================================

def bootstrap_ci(
    data: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float]:
    """Calcule un IC bootstrap sur des valeurs agregees au niveau des seeds."""
    if len(data) < 2 or np.all(np.isnan(data)):
        return (np.nan, np.nan)
    
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return (np.nan, np.nan)
    
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return (ci_low, ci_high)


def paired_bootstrap_ci(
    diff_values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float]:
    """IC bootstrap du gain moyen sur des differences apparieses seed-level."""
    if len(diff_values) < 2 or np.all(np.isnan(diff_values)):
        return (np.nan, np.nan)

    diff_values = diff_values[~np.isnan(diff_values)]
    if len(diff_values) < 2:
        return (np.nan, np.nan)

    bootstrap_means = np.array([
        np.mean(rng.choice(diff_values, size=len(diff_values), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return (ci_low, ci_high)


def hedges_correction_factor(n: int) -> float:
    """Petit correctif d'echantillon pour une taille d'effet standardisee appairee."""
    if n <= 1:
        return np.nan
    df = n - 1
    denom = 4 * df - 1
    if denom <= 0:
        return np.nan
    return 1.0 - 3.0 / denom


def wilcoxon_test(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """
    Test de Wilcoxon signed-rank pour données appariées.
    
    Returns:
        (statistic, p_value)
    """
    # Filtrer les NaN (doivent être appariés)
    mask = ~(np.isnan(data1) | np.isnan(data2))
    d1, d2 = data1[mask], data2[mask]
    
    if len(d1) < WILCOXON_MIN_PAIRS:
        warnings.warn("Moins de 5 paires valides pour Wilcoxon")
        return (np.nan, np.nan)
    
    try:
        stat, pval = stats.wilcoxon(
            d1,
            d2,
            alternative='two-sided',
            zero_method='wilcox',
            method='auto',
        )
        return (stat, pval)
    except Exception as exc:
        warnings.warn(f"Wilcoxon failed: {exc}")
        return (np.nan, np.nan)


def compute_statistics(
    metrics: Dict[str, Dict[str, np.ndarray]],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    bootstrap_seed: int = 2026,
) -> Dict[str, Any]:
    """
    Calcule toutes les statistiques pour chaque variante et métrique.
    """
    statistics = {}
    rng = np.random.default_rng(bootstrap_seed)
    
    for variant in VARIANTS:
        if variant not in metrics:
            continue
            
        statistics[variant] = {}
        
        for metric_name, values in metrics[variant].items():
            valid = values[~np.isnan(values)]
            
            if len(valid) == 0:
                statistics[variant][metric_name] = {
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "variance": np.nan,
                    "sem": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "min": np.nan,
                    "max": np.nan
                }
            else:
                std = np.std(valid, ddof=1) if len(valid) > 1 else 0
                variance = np.var(valid, ddof=1) if len(valid) > 1 else 0
                sem = std / np.sqrt(len(valid)) if len(valid) > 1 else 0
                ci_low, ci_high = bootstrap_ci(
                    valid,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                )
                statistics[variant][metric_name] = {
                    "n": len(valid),
                    "mean": np.mean(valid),
                    "std": std,
                    "variance": variance,
                    "sem": sem,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "min": np.min(valid),
                    "max": np.max(valid),
                    "values": valid.tolist()
                }
    
    return statistics


def compute_pairwise_tests(metrics: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict]:
    """
    Effectue les tests de Wilcoxon entre paires de variantes.
    """
    tests = {}
    pairs = [
        ("none", "demo"),
        ("none", "demo+anthro"),
        ("demo", "demo+anthro")
    ]
    
    key_metrics = ["macro_auc", "macro_f1_optimal"] + [f"auc_{c}" for c in DS5_LABELS] + [f"f1_{c}" for c in DS5_LABELS]
    
    for v1, v2 in pairs:
        if v1 not in metrics or v2 not in metrics:
            continue
            
        pair_key = f"{v1}_vs_{v2}"
        tests[pair_key] = {}
        
        for metric in key_metrics:
            if metric in metrics[v1] and metric in metrics[v2]:
                stat, pval = wilcoxon_test(metrics[v1][metric], metrics[v2][metric])
                
                # Différence moyenne
                d1, d2 = metrics[v1][metric], metrics[v2][metric]
                mask = ~(np.isnan(d1) | np.isnan(d2))
                diff = np.mean(d2[mask] - d1[mask]) if mask.any() else np.nan
                
                tests[pair_key][metric] = {
                    "statistic": stat,
                    "p_value": pval,
                    "significant_0.05": pval < 0.05 if not np.isnan(pval) else False,
                    "significant_0.01": pval < 0.01 if not np.isnan(pval) else False,
                    "mean_diff": diff
                }
    
    return tests


def align_pair_values(
    metrics: Dict[str, Dict[str, np.ndarray]],
    seed_orders: Dict[str, np.ndarray],
    v1: str,
    v2: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return metric values aligned on common seeds for paired tests."""
    seeds1 = [int(seed) for seed in seed_orders.get(v1, np.array([], dtype=int))]
    seeds2 = [int(seed) for seed in seed_orders.get(v2, np.array([], dtype=int))]
    common_seeds = sorted(set(seeds1) & set(seeds2))
    idx1 = {seed: idx for idx, seed in enumerate(seeds1)}
    idx2 = {seed: idx for idx, seed in enumerate(seeds2)}

    d1 = np.array([metrics[v1][metric][idx1[seed]] for seed in common_seeds], dtype=float)
    d2 = np.array([metrics[v2][metric][idx2[seed]] for seed in common_seeds], dtype=float)
    mask = ~(np.isnan(d1) | np.isnan(d2))
    paired_seeds = [seed for seed, keep in zip(common_seeds, mask) if keep]
    return d1[mask], d2[mask], paired_seeds


def apply_bh_fdr(tests: Dict[str, Dict], alpha: float = BH_FDR_ALPHA) -> Dict[str, Dict]:
    """Add Benjamini-Hochberg adjusted p-values across the whole test family."""
    entries = []
    for pair_key, pair_tests in tests.items():
        for metric, result in pair_tests.items():
            pval = result.get("p_value", np.nan)
            if not np.isnan(pval):
                entries.append((pair_key, metric, float(pval)))

    m = len(entries)
    if m == 0:
        return tests

    pvals = np.array([entry[2] for entry in entries], dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_min = 1.0

    for rank_from_end, original_idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        adj = min(running_min, pvals[original_idx] * m / rank)
        running_min = adj
        adjusted[original_idx] = min(adj, 1.0)

    for idx, (pair_key, metric, _) in enumerate(entries):
        tests[pair_key][metric]["p_adjusted_bh"] = adjusted[idx]
        tests[pair_key][metric]["significant_bh_0.05"] = bool(adjusted[idx] <= alpha)

    return tests


def compute_pairwise_tests_aligned(
    metrics: Dict[str, Dict[str, np.ndarray]],
    seed_orders: Dict[str, np.ndarray],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    bootstrap_seed: int = 2026,
) -> Dict[str, Dict]:
    """Wilcoxon tests on explicitly seed-aligned pairs, with BH-FDR correction."""
    tests = {}
    rng = np.random.default_rng(bootstrap_seed)

    for v1, v2 in TEST_PAIRS:
        if v1 not in metrics or v2 not in metrics:
            continue

        pair_key = f"{v1}_vs_{v2}"
        tests[pair_key] = {}

        for metric in KEY_METRICS:
            if metric in metrics[v1] and metric in metrics[v2]:
                d1, d2, paired_seeds = align_pair_values(metrics, seed_orders, v1, v2, metric)
                stat, pval = wilcoxon_test(d1, d2)
                diff_values = d2 - d1

                if len(diff_values) > 0:
                    paired_n = len(diff_values)
                    diff = np.mean(diff_values)
                    diff_std = np.std(diff_values, ddof=1) if paired_n > 1 else 0
                    diff_sem = diff_std / np.sqrt(paired_n) if paired_n > 1 else 0
                    diff_ci_low, diff_ci_high = paired_bootstrap_ci(
                        diff_values,
                        rng=rng,
                        n_bootstrap=n_bootstrap,
                        confidence=confidence,
                    )
                    cohen_dz = diff / diff_std if diff_std > 0 else np.nan
                    hedges_factor = hedges_correction_factor(paired_n)
                    hedges_gz = (
                        cohen_dz * hedges_factor
                        if not np.isnan(cohen_dz) and not np.isnan(hedges_factor)
                        else np.nan
                    )
                    n_positive = int(np.sum(diff_values > 0))
                    n_negative = int(np.sum(diff_values < 0))
                    n_zero = int(np.sum(diff_values == 0))
                else:
                    diff = np.nan
                    diff_std = np.nan
                    diff_sem = np.nan
                    diff_ci_low = np.nan
                    diff_ci_high = np.nan
                    cohen_dz = np.nan
                    hedges_gz = np.nan
                    n_positive = 0
                    n_negative = 0
                    n_zero = 0

                tests[pair_key][metric] = {
                    "statistic": stat,
                    "p_value": pval,
                    "p_adjusted_bh": np.nan,
                    "significant_0.05": pval < 0.05 if not np.isnan(pval) else False,
                    "significant_0.01": pval < 0.01 if not np.isnan(pval) else False,
                    "significant_bh_0.05": False,
                    "mean_diff": diff,
                    "diff_std": diff_std,
                    "diff_sem": diff_sem,
                    "diff_ci_low": diff_ci_low,
                    "diff_ci_high": diff_ci_high,
                    "cohen_dz": cohen_dz,
                    "hedges_gz": hedges_gz,
                    "paired_n": len(paired_seeds),
                    "paired_seeds": paired_seeds,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "n_zero": n_zero,
                }

    return apply_bh_fdr(tests, alpha=BH_FDR_ALPHA)


def build_seed_level_rows(results: Dict[str, Dict[int, Dict]]) -> List[Dict[str, Any]]:
    """Flatten seed-level JSON outputs into tabular rows for supplementary release files."""
    rows: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        for seed, data in sorted(results.get(variant, {}).items()):
            test = data.get("test", {})
            per_class = data.get("per_class", {})
            row: Dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "macro_auc": test.get("macro_auc", np.nan),
                "macro_f1_optimal": test.get("macro_f1_optimal", np.nan),
                "macro_f1_fixed": test.get("macro_f1_fixed", np.nan),
                "auc_ecg_only": test.get("auc_ecg_only", np.nan),
                "auc_fused_only": test.get("auc_fused_only", np.nan),
                "auc_meta_disabled": test.get("auc_meta_disabled", np.nan),
                "delta_meta_auc": test.get("delta_meta_auc", np.nan),
                "w_fused": test.get("w_fused", np.nan),
            }
            for cls in DS5_LABELS:
                cls_data = per_class.get(cls, {})
                row[f"auc_{cls}"] = cls_data.get("auc", np.nan)
                row[f"f1_{cls}"] = cls_data.get("f1", np.nan)
                row[f"threshold_{cls}"] = cls_data.get("threshold", np.nan)
            rows.append(row)

    return rows


# ============================================================================
# GÉNÉRATION DE TABLEAUX
# ============================================================================

def format_value(mean: float, std: float, precision: int = 4) -> str:
    """Formate une valeur avec ± écart-type."""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def format_with_significance(mean: float, std: float, pval: float, 
                             precision: int = 4, vs_baseline: bool = False) -> str:
    """Formate une valeur avec indicateur de significativité."""
    if np.isnan(mean):
        return "N/A"
    
    val = f"{mean:.{precision}f} ± {std:.{precision}f}"
    
    if vs_baseline and not np.isnan(pval):
        if pval < 0.01:
            val += "**"
        elif pval < 0.05:
            val += "*"
    
    return val


def p_value_for_stars(test_result: Dict[str, Any]) -> float:
    """Use BH-FDR adjusted p-values for display stars when available."""
    return test_result.get("p_adjusted_bh", test_result.get("p_value", np.nan))


def generate_main_table_markdown(stats: Dict, tests: Dict) -> str:
    """Génère le Tableau 1 principal en Markdown."""
    
    lines = [
        "## Tableau 1. Performances sur le jeu de test (fold 10) - Validation multi-graines",
        "",
        "| Méthode | Macro AUC | Macro F1 | n |",
        "|---------|-----------|----------|---|"
    ]
    
    for variant in VARIANTS:
        if variant not in stats:
            continue
            
        label = VARIANT_LABELS[variant]
        auc = stats[variant].get("macro_auc", {})
        f1 = stats[variant].get("macro_f1_optimal", {})
        n = auc.get("n", 0)
        
        # Récupérer p-value vs baseline (none)
        if variant == "none":
            auc_str = format_value(auc.get("mean", np.nan), auc.get("std", np.nan))
            f1_str = format_value(f1.get("mean", np.nan), f1.get("std", np.nan))
        else:
            test_key = f"none_vs_{variant}"
            pval_auc = p_value_for_stars(tests.get(test_key, {}).get("macro_auc", {}))
            pval_f1 = p_value_for_stars(tests.get(test_key, {}).get("macro_f1_optimal", {}))
            
            auc_str = format_with_significance(
                auc.get("mean", np.nan), auc.get("std", np.nan), pval_auc, vs_baseline=True
            )
            f1_str = format_with_significance(
                f1.get("mean", np.nan), f1.get("std", np.nan), pval_f1, vs_baseline=True
            )
        
        lines.append(f"| {label} | {auc_str} | {f1_str} | {n} |")
    
    lines.extend([
        "",
        "*p < 0.05; **p < 0.01 after BH-FDR correction (paired Wilcoxon vs ECG seul)",
        ""
    ])
    
    return "\n".join(lines)


def generate_perclass_table_markdown(stats: Dict, tests: Dict) -> str:
    """Génère le tableau des performances par classe en Markdown."""
    
    lines = [
        "## Tableau 2. Performances par classe - Validation multi-graines",
        "",
        "### AUC par classe",
        "",
        "| Classe | ECG seul | ECG + demo | ECG + complet |",
        "|--------|----------|------------|---------------|"
    ]
    
    for cls in DS5_LABELS:
        row = [cls]
        for variant in VARIANTS:
            if variant not in stats:
                row.append("N/A")
                continue
            
            metric = f"auc_{cls}"
            data = stats[variant].get(metric, {})
            
            if variant == "none":
                val = format_value(data.get("mean", np.nan), data.get("std", np.nan))
            else:
                test_key = f"none_vs_{variant}"
                pval = p_value_for_stars(tests.get(test_key, {}).get(metric, {}))
                val = format_with_significance(
                    data.get("mean", np.nan), data.get("std", np.nan), pval, vs_baseline=True
                )
            row.append(val)
        
        lines.append("| " + " | ".join(row) + " |")
    
    lines.extend([
        "",
        "### F1 par classe",
        "",
        "| Classe | ECG seul | ECG + demo | ECG + complet |",
        "|--------|----------|------------|---------------|"
    ])
    
    for cls in DS5_LABELS:
        row = [cls]
        for variant in VARIANTS:
            if variant not in stats:
                row.append("N/A")
                continue
            
            metric = f"f1_{cls}"
            data = stats[variant].get(metric, {})
            
            if variant == "none":
                val = format_value(data.get("mean", np.nan), data.get("std", np.nan))
            else:
                test_key = f"none_vs_{variant}"
                pval = p_value_for_stars(tests.get(test_key, {}).get(metric, {}))
                val = format_with_significance(
                    data.get("mean", np.nan), data.get("std", np.nan), pval, vs_baseline=True
                )
            row.append(val)
        
        lines.append("| " + " | ".join(row) + " |")
    
    lines.extend([
        "",
        "*p < 0.05; **p < 0.01 after BH-FDR correction (paired Wilcoxon vs ECG seul)",
        ""
    ])
    
    return "\n".join(lines)


def generate_latex_table(stats: Dict, tests: Dict) -> str:
    """Génère le Tableau 1 en LaTeX pour l'article."""
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performances sur le jeu de test (fold 10) avec validation multi-graines. "
        "Les résultats sont présentés sous forme moyenne $\\pm$ écart-type sur 10 graines aléatoires. "
        "$^*p < 0.05$; $^{**}p < 0.01$ after BH-FDR correction (paired Wilcoxon signed-rank vs ECG seul).}",
        "\\label{tab:results_multiseed}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Méthode} & \\textbf{Macro AUC} & \\textbf{Macro F1} & \\textbf{n} \\\\",
        "\\midrule"
    ]
    
    for variant in VARIANTS:
        if variant not in stats:
            continue
            
        label = VARIANT_LABELS[variant]
        auc = stats[variant].get("macro_auc", {})
        f1 = stats[variant].get("macro_f1_optimal", {})
        n = auc.get("n", 0)
        
        mean_auc = auc.get("mean", np.nan)
        std_auc = auc.get("std", np.nan)
        mean_f1 = f1.get("mean", np.nan)
        std_f1 = f1.get("std", np.nan)
        
        # Significativité
        sig_auc, sig_f1 = "", ""
        if variant != "none":
            test_key = f"none_vs_{variant}"
            pval_auc = p_value_for_stars(tests.get(test_key, {}).get("macro_auc", {}))
            pval_f1 = p_value_for_stars(tests.get(test_key, {}).get("macro_f1_optimal", {}))
            
            if not np.isnan(pval_auc):
                if pval_auc < 0.01:
                    sig_auc = "$^{**}$"
                elif pval_auc < 0.05:
                    sig_auc = "$^{*}$"
            
            if not np.isnan(pval_f1):
                if pval_f1 < 0.01:
                    sig_f1 = "$^{**}$"
                elif pval_f1 < 0.05:
                    sig_f1 = "$^{*}$"
        
        auc_str = f"${mean_auc:.4f} \\pm {std_auc:.4f}${sig_auc}" if not np.isnan(mean_auc) else "N/A"
        f1_str = f"${mean_f1:.4f} \\pm {std_f1:.4f}${sig_f1}" if not np.isnan(mean_f1) else "N/A"
        
        lines.append(f"{label} & {auc_str} & {f1_str} & {n} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_gains_summary(stats: Dict, tests: Dict) -> str:
    """Génère un résumé des gains avec significativité."""
    
    lines = [
        "## Résumé des gains (ECG + complet vs ECG seul)",
        ""
    ]
    
    if "none" not in stats or "demo+anthro" not in stats:
        return "Données insuffisantes pour calculer les gains."
    
    metrics_to_compare = [
        ("macro_auc", "Macro AUC"),
        ("macro_f1_optimal", "Macro F1"),
    ] + [(f"f1_{c}", f"F1 {c}") for c in DS5_LABELS]
    
    lines.append("| Métrique | Baseline | Complet | Gain absolu | Gain relatif | p-value |")
    lines.append("|----------|----------|---------|-------------|--------------|---------|")
    
    test_key = "none_vs_demo+anthro"
    
    for metric_key, metric_label in metrics_to_compare:
        baseline = stats["none"].get(metric_key, {})
        complete = stats["demo+anthro"].get(metric_key, {})
        test_result = tests.get(test_key, {}).get(metric_key, {})
        
        mean_base = baseline.get("mean", np.nan)
        mean_comp = complete.get("mean", np.nan)
        
        if not np.isnan(mean_base) and not np.isnan(mean_comp):
            gain_abs = mean_comp - mean_base
            gain_rel = 100 * gain_abs / mean_base if mean_base != 0 else np.nan
            pval = test_result.get("p_value", np.nan)
            
            sig = ""
            if not np.isnan(pval):
                if pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
            
            pval_str = f"{pval:.4f}{sig}" if not np.isnan(pval) else "N/A"
            
            lines.append(
                f"| {metric_label} | {mean_base:.4f} | {mean_comp:.4f} | "
                f"{gain_abs:+.4f} | {gain_rel:+.2f}% | {pval_str} |"
            )
    
    lines.extend([
        "",
        "*p < 0.05; **p < 0.01 after BH-FDR correction",
        ""
    ])
    
    return "\n".join(lines)


def generate_gains_summary(stats: Dict, tests: Dict) -> str:
    """Generate gain summary with raw and BH-FDR adjusted p-values."""

    lines = [
        "## Resume des gains (ECG + complet vs ECG seul)",
        "",
    ]

    if "none" not in stats or "demo+anthro" not in stats:
        return "Donnees insuffisantes pour calculer les gains."

    metrics_to_compare = [
        ("macro_auc", "Macro AUC"),
        ("macro_f1_optimal", "Macro F1"),
    ] + [(f"f1_{c}", f"F1 {c}") for c in DS5_LABELS]

    lines.append("| Metrique | Baseline | Complet | Gain absolu | Gain relatif | p raw | p BH-FDR |")
    lines.append("|----------|----------|---------|-------------|--------------|-------|----------|")

    test_key = "none_vs_demo+anthro"

    for metric_key, metric_label in metrics_to_compare:
        baseline = stats["none"].get(metric_key, {})
        complete = stats["demo+anthro"].get(metric_key, {})
        test_result = tests.get(test_key, {}).get(metric_key, {})

        mean_base = baseline.get("mean", np.nan)
        mean_comp = complete.get("mean", np.nan)

        if not np.isnan(mean_base) and not np.isnan(mean_comp):
            gain_abs = mean_comp - mean_base
            gain_rel = 100 * gain_abs / mean_base if mean_base != 0 else np.nan
            pval = test_result.get("p_value", np.nan)
            pval_bh = test_result.get("p_adjusted_bh", np.nan)

            sig = ""
            if not np.isnan(pval_bh):
                if pval_bh < 0.01:
                    sig = "**"
                elif pval_bh < 0.05:
                    sig = "*"

            pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
            pval_bh_str = f"{pval_bh:.4f}{sig}" if not np.isnan(pval_bh) else "N/A"

            lines.append(
                f"| {metric_label} | {mean_base:.4f} | {mean_comp:.4f} | "
                f"{gain_abs:+.4f} | {gain_rel:+.2f}% | {pval_str} | {pval_bh_str} |"
            )

    lines.extend([
        "",
        "*p < 0.05; **p < 0.01 after BH-FDR correction",
        "",
    ])

    return "\n".join(lines)


def generate_latex_table(stats: Dict, tests: Dict) -> str:
    """Generate a LaTeX summary table aligned with the release manuscript."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performances sur le jeu de test (fold 10) avec validation multi-graines. "
        "Les resultats sont presentes sous forme moyenne $\\pm$ ecart-type sur 10 graines aleatoires, "
        "avec IC bootstrap centiles a 95\\% pour les niveaux de performance et les differences appariees. "
        "Les comparaisons rapportent le $p$ brut, le $p$ ajuste par BH-FDR, le $d_z$ de Cohen et le $g_z$ "
        "de Hedges corrige petit-echantillon.}",
        "\\label{tab:results_multiseed}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Methode} & \\textbf{Macro AUC} & \\textbf{Macro F1} & \\textbf{95\\% CI} \\\\",
        "\\midrule",
    ]

    for variant in VARIANTS:
        if variant not in stats:
            continue

        label = VARIANT_LABELS[variant]
        auc = stats[variant].get("macro_auc", {})
        f1 = stats[variant].get("macro_f1_optimal", {})
        mean_auc = auc.get("mean", np.nan)
        std_auc = auc.get("std", np.nan)
        mean_f1 = f1.get("mean", np.nan)
        std_f1 = f1.get("std", np.nan)
        ci_low = auc.get("ci_low", np.nan)
        ci_high = auc.get("ci_high", np.nan)

        auc_str = f"${mean_auc:.4f} \\pm {std_auc:.4f}$" if not np.isnan(mean_auc) else "N/A"
        f1_str = f"${mean_f1:.4f} \\pm {std_f1:.4f}$" if not np.isnan(mean_f1) else "N/A"
        ci_str = f"$[{ci_low:.4f}, {ci_high:.4f}]$" if not np.isnan(ci_low) else "N/A"
        lines.append(f"{label} & {auc_str} & {f1_str} & {ci_str} \\\\")

    lines.extend([
        "\\midrule",
        "\\multicolumn{4}{l}{\\textbf{Comparaisons appariees (macro-AUC)}} \\\\",
    ])

    for pair_key, label in [
        ("none_vs_demo", "\\textsc{demo} $-$ \\textsc{none}"),
        ("demo_vs_demo+anthro", "\\textsc{demo+anthro} $-$ \\textsc{demo}"),
        ("none_vs_demo+anthro", "\\textsc{demo+anthro} $-$ \\textsc{none}"),
    ]:
        result = tests.get(pair_key, {}).get("macro_auc", {})
        if not result:
            continue

        diff = result.get("mean_diff", np.nan)
        diff_sem = result.get("diff_sem", np.nan)
        p_raw = result.get("p_value", np.nan)
        p_bh = result.get("p_adjusted_bh", np.nan)
        d_z = result.get("cohen_dz", np.nan)
        g_z = result.get("hedges_gz", np.nan)
        ci_low = result.get("diff_ci_low", np.nan)
        ci_high = result.get("diff_ci_high", np.nan)

        diff_str = f"${diff:+.4f} \\pm {diff_sem:.4f}$" if not np.isnan(diff) else "N/A"
        stats_str = (
            f"$p_{{raw}}={p_raw:.3f}$, $p_{{BH}}={p_bh:.3f}$, $d_z={d_z:.2f}$, $g_z={g_z:.2f}$"
            if not np.isnan(p_raw)
            else "N/A"
        )
        ci_str = f"IC diff. $[{ci_low:.4f}, {ci_high:.4f}]$" if not np.isnan(ci_low) else "N/A"
        lines.append(f"{label} & {diff_str} & {stats_str} & {ci_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_pairwise_summary_markdown(tests: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Compact summary of the main paired macro-level comparisons."""
    lines = [
        "## Comparaisons appariees principales",
        "",
        "| Contraste | Metrique | Diff. moyenne | IC bootstrap 95% (diff) | p raw | p BH-FDR | d_z | g_z (corrige) | n | +/-/0 |",
        "|-----------|----------|---------------|--------------------------|-------|----------|-----|----------------|---|-------|",
    ]

    for pair_key in ("none_vs_demo", "none_vs_demo+anthro", "demo_vs_demo+anthro"):
        pair_tests = tests.get(pair_key, {})
        contrast = pair_key.replace("_vs_", " vs ")
        for metric in ("macro_auc", "macro_f1_optimal"):
            result = pair_tests.get(metric, {})
            if not result:
                continue
            diff = result.get("mean_diff", np.nan)
            ci_low = result.get("diff_ci_low", np.nan)
            ci_high = result.get("diff_ci_high", np.nan)
            p_raw = result.get("p_value", np.nan)
            p_bh = result.get("p_adjusted_bh", np.nan)
            dz = result.get("cohen_dz", np.nan)
            gz = result.get("hedges_gz", np.nan)
            paired_n = result.get("paired_n", 0)
            signs = f"{result.get('n_positive', 0)}/{result.get('n_negative', 0)}/{result.get('n_zero', 0)}"
            metric_label = "Macro AUC" if metric == "macro_auc" else "Macro F1"
            diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
            ci_str = (
                f"[{ci_low:.4f}, {ci_high:.4f}]"
                if not np.isnan(ci_low) and not np.isnan(ci_high)
                else "N/A"
            )
            p_raw_str = f"{p_raw:.4f}" if not np.isnan(p_raw) else "N/A"
            p_bh_str = f"{p_bh:.4f}" if not np.isnan(p_bh) else "N/A"
            dz_str = f"{dz:.2f}" if not np.isnan(dz) else "N/A"
            gz_str = f"{gz:.2f}" if not np.isnan(gz) else "N/A"
            lines.append(
                f"| {contrast} | {metric_label} | {diff_str} | {ci_str} | "
                f"{p_raw_str} | {p_bh_str} | {dz_str} | {gz_str} | {paired_n} | {signs} |"
            )

    lines.extend([
        "",
        "Notes: the bootstrap CIs above are computed by resampling seed-level paired differences with replacement.",
        "+/-/0 reports the number of positive, negative, and exactly zero paired seed differences.",
        "",
    ])
    return "\n".join(lines)


def generate_seed_level_markdown(rows: List[Dict[str, Any]]) -> str:
    """Human-readable supplementary table with one row per completed run."""
    lines = [
        "# Seed-Level Test Results",
        "",
        "| Variant | Seed | Macro AUC | Macro F1* | ECG-only AUC | Fused-only AUC | Delta meta AUC | w_fused |",
        "|---------|------|-----------|-----------|--------------|----------------|----------------|---------|",
    ]

    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['macro_auc']:.4f} | "
            f"{row['macro_f1_optimal']:.4f} | {row['auc_ecg_only']:.4f} | "
            f"{row['auc_fused_only']:.4f} | {row['delta_meta_auc']:+.4f} | {row['w_fused']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_analysis_notes() -> str:
    """Methodological notes exported alongside the numerical outputs."""
    lines = [
        "## Analysis notes",
        "",
        "- Bootstrap confidence intervals are computed on seed-level values, not by patient-level resampling.",
        "- Pairwise inferential tests are two-sided paired Wilcoxon signed-rank tests on seed-matched variant differences.",
        "- Because blend weights and decision thresholds are selected on fold 9 and then fixed on fold 10, p-values should be interpreted conditionally on this validation-based model-selection protocol.",
        "- Cohen's d_z is reported as a descriptive paired effect size; Hedges-corrected g_z is exported alongside it for small-sample sensitivity.",
        "- Consecutive integer seeds are used as transparent run identifiers only; each run re-seeds Python, NumPy, and PyTorch independently.",
        "",
    ]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyse statistique des résultats multi-graines EZNX_ATLAS_A'
    )
    parser.add_argument(
        '--runs_dir',
        type=str,
        required=True,
        help='Répertoire contenant les runs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Répertoire de sortie pour les rapports (défaut: runs_dir)'
    )
    parser.add_argument(
        '--n_bootstrap',
        type=int,
        default=N_BOOTSTRAP,
        help=f'Nombre d\'itérations bootstrap (défaut: {N_BOOTSTRAP})'
    )
    parser.add_argument(
        '--bootstrap_seed',
        type=int,
        default=2026,
        help='Graine aleatoire pour rendre le bootstrap deterministe'
    )
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ANALYSE STATISTIQUE MULTI-GRAINES")
    print("=" * 80)
    print(f"Runs dir:    {runs_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Bootstrap:   {args.n_bootstrap} iterations")
    print(f"Seed:        {args.bootstrap_seed}")
    print("=" * 80)
    
    # 1. Charger les résultats
    print("\n[1/4] Chargement des résultats...")
    results = load_all_results(runs_dir)
    
    for variant in VARIANTS:
        n_seeds = len(results.get(variant, {}))
        seeds = sorted(results.get(variant, {}).keys())
        print(f"   {variant}: {n_seeds} seeds {seeds if n_seeds <= 10 else '...'}")
    
    if not results:
        print("\nERREUR: Aucun résultat trouvé!")
        return
    
    # 2. Extraire les métriques
    print("\n[2/4] Extraction des métriques...")
    metrics, seed_orders = extract_metrics(results)
    
    # 3. Calculer les statistiques
    print("\n[3/4] Calcul des statistiques...")
    stats = compute_statistics(
        metrics,
        n_bootstrap=args.n_bootstrap,
        confidence=CONFIDENCE_LEVEL,
        bootstrap_seed=args.bootstrap_seed,
    )
    tests = compute_pairwise_tests_aligned(
        metrics,
        seed_orders,
        n_bootstrap=args.n_bootstrap,
        confidence=CONFIDENCE_LEVEL,
        bootstrap_seed=args.bootstrap_seed,
    )
    seed_level_rows = build_seed_level_rows(results)
    
    # Afficher résumé
    print("\n" + "-" * 80)
    print("RÉSUMÉ STATISTIQUE")
    print("-" * 80)
    
    for variant in VARIANTS:
        if variant not in stats:
            continue
        print(f"\n{VARIANT_LABELS[variant]}:")
        auc = stats[variant].get("macro_auc", {})
        f1 = stats[variant].get("macro_f1_optimal", {})
        print(f"   Macro AUC: {auc.get('mean', np.nan):.4f} ± {auc.get('std', np.nan):.4f} "
              f"[{auc.get('ci_low', np.nan):.4f}, {auc.get('ci_high', np.nan):.4f}] (n={auc.get('n', 0)})")
        print(f"   Macro F1:  {f1.get('mean', np.nan):.4f} ± {f1.get('std', np.nan):.4f} "
              f"[{f1.get('ci_low', np.nan):.4f}, {f1.get('ci_high', np.nan):.4f}]")
    
    # Tests de significativité
    print("\n" + "-" * 80)
    print("TESTS DE SIGNIFICATIVITÉ (Wilcoxon signed-rank)")
    print("-" * 80)
    
    for pair_key, pair_tests in tests.items():
        v1, v2 = pair_key.replace("_vs_", " vs ").split(" vs ")
        print(f"\n{VARIANT_LABELS.get(v1, v1)} vs {VARIANT_LABELS.get(v2, v2)}:")
        
        for metric in ["macro_auc", "macro_f1_optimal"]:
            if metric in pair_tests:
                t = pair_tests[metric]
                sig = "*" if t.get("significant_bh_0.05") else ""
                print(
                    f"   {metric}: p_raw = {t.get('p_value', np.nan):.4f}, "
                    f"p_BH = {t.get('p_adjusted_bh', np.nan):.4f}{sig} "
                    f"(diff = {t.get('mean_diff', np.nan):+.4f}, "
                    f"ci = [{t.get('diff_ci_low', np.nan):+.4f}, {t.get('diff_ci_high', np.nan):+.4f}], "
                    f"sd_diff = {t.get('diff_std', np.nan):.4f}, "
                    f"dz = {t.get('cohen_dz', np.nan):.2f}, "
                    f"gz = {t.get('hedges_gz', np.nan):.2f}, "
                    f"n = {t.get('paired_n', 0)})"
                )
    
    # 4. Générer les rapports
    print("\n[4/4] Génération des rapports...")
    
    # Markdown
    md_content = [
        "# Analyse Statistique Multi-Graines - EZNX_ATLAS_A",
        "",
        f"*Généré le {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        generate_analysis_notes(),
        generate_main_table_markdown(stats, tests),
        generate_perclass_table_markdown(stats, tests),
        generate_pairwise_summary_markdown(tests),
        generate_gains_summary(stats, tests)
    ]
    
    md_path = output_dir / "statistical_analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    print(f"   Rapport Markdown: {md_path}")
    
    # LaTeX
    latex_content = generate_latex_table(stats, tests)
    latex_path = output_dir / "table_results_latex.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"   Table LaTeX: {latex_path}")

    seed_level_md_path = output_dir / "seed_level_results.md"
    with open(seed_level_md_path, 'w', encoding='utf-8') as f:
        f.write(generate_seed_level_markdown(seed_level_rows))
    print(f"   Supplement Markdown: {seed_level_md_path}")

    seed_level_csv_path = output_dir / "seed_level_results.csv"
    if seed_level_rows:
        fieldnames = list(seed_level_rows[0].keys())
        with open(seed_level_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seed_level_rows)
    else:
        with open(seed_level_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write("")
    print(f"   Supplement CSV: {seed_level_csv_path}")
    
    # JSON complet
    full_report = {
        "statistics": stats,
        "pairwise_tests": tests,
        "seed_level_rows": seed_level_rows,
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "bh_fdr_alpha": BH_FDR_ALPHA,
            "test_pairs": TEST_PAIRS,
            "key_metrics": KEY_METRICS,
            "seed_orders": seed_orders,
        }
    }
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        return obj
    
    json_path = output_dir / "statistical_analysis_full.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(full_report), f, indent=2, ensure_ascii=False)
    print(f"   Rapport JSON complet: {json_path}")
    
    print("\n" + "=" * 80)
    print("[OK] Analyse terminee.")
    print("=" * 80)
    
    # Afficher le tableau principal formaté
    print("\n" + generate_main_table_markdown(stats, tests))


if __name__ == "__main__":
    main()
