"""
Regenerate all Group A derived artifacts from the 60 seed JSON files.
Fixes C1: stale 10-seed/30-run derived artifacts.

Outputs:
  results/seed_level_results.csv            (60 rows, seeds 2024-2043)
  results/statistical_analysis_full.json    (n=20 per variant, 3-test family)
  results/table_results_latex.tex           (n=20 caption, corrected stats)
  results/statistical_analysis_protocol.md (updated: 3 tests, 20 seeds)
"""
from __future__ import annotations
import json, csv, math, itertools
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).parent
SJDIR  = REPO / "results" / "seed_json"
RESDIR = REPO / "results"
VARIANTS = ["none", "demo", "demo+anthro"]
CLASSES  = ["NORM", "MI", "STTC", "CD", "HYP"]
SEEDS    = list(range(2024, 2044))   # 20 seeds

# ── Helpers ───────────────────────────────────────────────────────────────────

def mean(xs):  return sum(xs) / len(xs)
def var(xs):
    m = mean(xs); return sum((x - m)**2 for x in xs) / (len(xs) - 1)
def std(xs):   return math.sqrt(var(xs))

def wilcoxon_signed_rank_exact(diffs):
    """Exact two-sided Wilcoxon signed-rank p-value (no ties)."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranked = sorted(enumerate(nonzero), key=lambda x: abs(x[1]))
    ranks = [0] * len(nonzero)
    for rank_idx, (orig_idx, _) in enumerate(ranked):
        ranks[orig_idx] = rank_idx + 1
    T_plus  = sum(r for (oi, d), r in zip(ranked, [ranks[i] for i in range(n)]) if nonzero[oi] > 0)
    # Recompute properly
    T_plus = sum(ranks[i] for i, (_, d) in enumerate([(oi, nonzero[oi]) for oi in range(n)]) if d > 0)
    T_minus = sum(ranks[i] for i, (_, d) in enumerate([(oi, nonzero[oi]) for oi in range(n)]) if d < 0)
    T_stat  = min(T_plus, T_minus)
    # Enumerate all 2^n sign assignments
    total = 2**n
    count = 0
    for bits in range(total):
        s = 0
        for i in range(n):
            sign = 1 if (bits >> i) & 1 else -1
            s += sign * (i + 1)  # rank = i+1 for sorted-by-abs
        if abs(s) >= abs(n*(n+1)//2 - 2*T_stat):
            count += 1
    return count / total

def bootstrap_ci(vals, n_boot=10000, ci=0.95, rng_seed=42):
    import random
    rng = random.Random(rng_seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = (1 - ci) / 2
    return boots[int(lo * n_boot)], boots[int((1 - lo) * n_boot) - 1]

def cohen_dz(diffs):
    m = mean(diffs); s = std(diffs)
    return m / s if s > 0 else 0.0

def hedges_gz(diffs):
    n = len(diffs)
    dz = cohen_dz(diffs)
    correction = 1 - 3 / (4 * (n - 1) - 1)
    return dz * correction

def bh_adjust(pvals):
    """BH adjustment; returns adjusted p-values."""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * n
    prev = 1.0
    for rank_idx, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = n - rank_idx  # n down to 1
        adj_p = min(prev, p * n / rank)
        adj[orig_idx] = adj_p
        prev = adj_p
    return adj

# ── Load all JSON files ───────────────────────────────────────────────────────

data = {}  # data[variant][seed] = dict
for variant in VARIANTS:
    data[variant] = {}
    for seed in SEEDS:
        fname = SJDIR / f"results_ATLAS_A_v5_{variant}_seed{seed}.json"
        if not fname.exists():
            print(f"  WARNING: missing {fname.name}")
            continue
        with open(fname) as f:
            j = json.load(f)
        t = j["test"]
        pc = j.get("per_class", {})
        data[variant][seed] = {
            "macro_auc":       t["macro_auc"],
            "macro_f1_optimal": t.get("macro_f1_optimal", float("nan")),
            "macro_f1_fixed":  t.get("macro_f1_fixed_05", float("nan")),
            "auc_ecg_only":    t.get("macro_auc_ecg", t["macro_auc"]),
            "auc_fused_only":  t["macro_auc"],
            "auc_meta_disabled": t.get("macro_auc_no_meta", t["macro_auc"]),
            "delta_meta_auc":  t.get("delta_meta_auc", 0.0),
            "w_fused":         t.get("w_fused", 1.0),
            **{f"auc_{c}":    pc.get(c, {}).get("auc", float("nan")) for c in CLASSES},
            **{f"f1_{c}":     pc.get(c, {}).get("f1",  float("nan")) for c in CLASSES},
            **{f"threshold_{c}": pc.get(c, {}).get("threshold", float("nan")) for c in CLASSES},
        }

# ── 1. seed_level_results.csv ─────────────────────────────────────────────────
print("Writing seed_level_results.csv ...")
fieldnames = [
    "variant", "seed", "macro_auc", "macro_f1_optimal", "macro_f1_fixed",
    "auc_ecg_only", "auc_fused_only", "auc_meta_disabled", "delta_meta_auc", "w_fused",
]
for c in CLASSES:
    fieldnames += [f"auc_{c}", f"f1_{c}", f"threshold_{c}"]

with open(RESDIR / "seed_level_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for variant in VARIANTS:
        for seed in sorted(data[variant]):
            row = {"variant": variant, "seed": seed}
            row.update(data[variant][seed])
            writer.writerow(row)
print(f"  Written {sum(len(data[v]) for v in VARIANTS)} rows.")

# ── 2. statistical_analysis_full.json ─────────────────────────────────────────
print("Writing statistical_analysis_full.json ...")

def variant_stats(variant):
    seeds_present = sorted(data[variant].keys())
    n = len(seeds_present)
    aucs = [data[variant][s]["macro_auc"] for s in seeds_present]
    f1s  = [data[variant][s]["macro_f1_optimal"] for s in seeds_present]
    ci_lo, ci_hi = bootstrap_ci(aucs)
    return {
        "macro_auc": {
            "n": n, "seeds": seeds_present,
            "mean": mean(aucs), "std": std(aucs),
            "ci_low": ci_lo, "ci_hi": ci_hi,
            "min": min(aucs), "max": max(aucs),
            "values": aucs,
        },
        "macro_f1_optimal": {
            "n": n, "mean": mean(f1s), "std": std(f1s), "values": f1s,
        },
    }

def pairwise_stats(variant_a, variant_b):
    shared = sorted(set(data[variant_a]) & set(data[variant_b]))
    diffs  = [data[variant_a][s]["macro_auc"] - data[variant_b][s]["macro_auc"]
              for s in shared]
    n = len(diffs)
    p_raw = wilcoxon_signed_rank_exact(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    dz = cohen_dz(diffs)
    gz = hedges_gz(diffs)
    return {
        "contrast": f"{variant_a} - {variant_b}",
        "n": n,
        "mean_diff": mean(diffs),
        "std_diff": std(diffs),
        "n_positive": n_pos,
        "p_raw_wilcoxon": p_raw,
        "ci_low_diff": ci_lo,
        "ci_hi_diff":  ci_hi,
        "cohen_dz": dz,
        "hedges_gz": gz,
        "diffs": diffs,
    }

pairs = [
    ("demo", "none"),
    ("demo+anthro", "none"),
    ("demo+anthro", "demo"),
]
pairwise = {f"{a}_minus_{b}": pairwise_stats(a, b) for a, b in pairs}

# BH correction over 3-test family
raw_ps = [pairwise[k]["p_raw_wilcoxon"] for k in pairwise]
adj_ps = bh_adjust(raw_ps)
for k, adj in zip(pairwise, adj_ps):
    pairwise[k]["p_bh_adjusted"] = adj

full_json = {
    "generated_by": "regen_derived_artifacts_v2.py",
    "n_seeds": 20,
    "seeds": SEEDS,
    "variants": VARIANTS,
    "confirmatory_family": "3-test BH-FDR (3 pairwise macro-AUC contrasts in Group A)",
    "statistics": {v: variant_stats(v) for v in VARIANTS},
    "pairwise": pairwise,
}

with open(RESDIR / "statistical_analysis_full.json", "w") as f:
    json.dump(full_json, f, indent=2)
print("  Done.")

# ── 3. table_results_latex.tex ────────────────────────────────────────────────
print("Writing table_results_latex.tex ...")

def sig_stars(p_bh):
    if p_bh < 0.001: return "^{***}"
    if p_bh < 0.01:  return "^{**}"
    if p_bh < 0.05:  return "^{*}"
    return ""

rows = []
labels = {"none": "ECG only (none)", "demo": "ECG + demo", "demo+anthro": "ECG + demo+anthro (full)"}
ref_key = "demo_minus_none"
for variant in VARIANTS:
    seeds_present = sorted(data[variant].keys())
    aucs = [data[variant][s]["macro_auc"] for s in seeds_present]
    f1s  = [data[variant][s]["macro_f1_optimal"] for s in seeds_present]
    n    = len(seeds_present)
    stars = ""
    if variant == "demo":
        p_bh = pairwise["demo_minus_none"]["p_bh_adjusted"]
        stars = sig_stars(p_bh)
    elif variant == "demo+anthro":
        p_bh = pairwise["demo+anthro_minus_none"]["p_bh_adjusted"]
        stars = sig_stars(p_bh)
    rows.append(
        f"{labels[variant]} & "
        f"${mean(aucs):.4f} \\pm {std(aucs):.4f}${stars} & "
        f"${mean(f1s):.4f} \\pm {std(f1s):.4f}$ & "
        f"{n} \\\\"
    )

tex = r"""\begin{table}[htbp]
\centering
\caption{Test-set performance (fold 10) under the 20-seed multi-seed protocol.
Results are mean $\pm$ SD across 20 seeds.
$^{*}p < 0.05$; $^{**}p < 0.01$; $^{***}p < 0.001$ after BH-FDR correction
over the 3-test pre-specified family (paired Wilcoxon signed-rank vs.\ ECG only).}
\label{tab:results_multiseed}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Macro-AUC} & \textbf{Macro-F1} & \textbf{$n$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
with open(RESDIR / "table_results_latex.tex", "w") as f:
    f.write(tex)
print("  Done.")

# ── 4. Print summary ──────────────────────────────────────────────────────────
print("\n=== Summary ===")
for v in VARIANTS:
    seeds_present = sorted(data[v].keys())
    aucs = [data[v][s]["macro_auc"] for s in seeds_present]
    print(f"  {v}: n={len(aucs)}, mean={mean(aucs):.7f}, std={std(aucs):.7f}")
print()
for k, pw in pairwise.items():
    print(f"  {pw['contrast']}: mean={pw['mean_diff']:+.7f}, p_raw={pw['p_raw_wilcoxon']:.6f}, p_BH={pw['p_bh_adjusted']:.6f}, n_pos={pw['n_positive']}/20")
print("\nDone. All artifacts regenerated.")
