#!/usr/bin/env python3
# =============================================================================
# run_extended_experiments.py
# Orchestrateur pour les 5 expériences supplémentaires de révision.
#
# Expériences planifiées:
#   ID        | variant      | seed | gate_dim | lauc_w | Couvre
#   --------- | ------------ | ---- | -------- | ------ | -------
#   H5+H8     | demo+anthro  | 2029 | 1152     | 0.08   | H5 métriques étendues + H8 LVH
#   H7-glu512 | demo+anthro  | 2026 | 512      | 0.08   | H7 largeur étroite
#   H7-glu1152| demo+anthro  | 2026 | 1152     | 0.08   | H7 largeur standard (référence)
#   H7-glu2048| demo+anthro  | 2026 | 2048     | 0.08   | H7 largeur large
#   M3        | demo+anthro  | 2026 | 1152     | 0.00   | M3 ablation LAUC
#
# Total: 5 runs  ≈ 3h CPU (sur Intel Core i5, 8 GB RAM, PyTorch CPU)
#
# M4 (trajectoires de validation): les 30 JSON existants ont déjà training_history
# sur 10 époques. Un script de figure séparé peut les lire directement.
# Les 5 nouveaux runs sauvegardent aussi training_history → prêts pour M4.
#
# Usage:
#   python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3
#   python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3 --dry_run
#   python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3 --only H5H8 M3
# =============================================================================

import subprocess
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Expériences définies
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "id":             "H5H8",
        "description":    "H5 métriques étendues (AUPRC/Brier/DeLong/ECE) + H8 analyse LVH",
        "variant":        "demo+anthro",
        "seed":           2029,
        "gate_hidden_dim": 1152,
        "lauc_weight":    0.08,
    },
    {
        "id":             "H7-glu512",
        "description":    "H7 balayage GLU — largeur 512 (gate étroite)",
        "variant":        "demo+anthro",
        "seed":           2026,
        "gate_hidden_dim": 512,
        "lauc_weight":    0.08,
    },
    {
        "id":             "H7-glu1152",
        "description":    "H7 balayage GLU — largeur 1152 (standard, référence seed-2026)",
        "variant":        "demo+anthro",
        "seed":           2026,
        "gate_hidden_dim": 1152,
        "lauc_weight":    0.08,
    },
    {
        "id":             "H7-glu2048",
        "description":    "H7 balayage GLU — largeur 2048 (gate large)",
        "variant":        "demo+anthro",
        "seed":           2026,
        "gate_hidden_dim": 2048,
        "lauc_weight":    0.08,
    },
    {
        "id":             "M3",
        "description":    "M3 ablation AUC-margin: lauc_weight=0.0 (suppression du terme LAUC)",
        "variant":        "demo+anthro",
        "seed":           2026,
        "gate_hidden_dim": 1152,
        "lauc_weight":    0.00,
    },
]

TRAINING_SCRIPT = str(Path(__file__).resolve().parent / "atlas_a_v5_extended.py")


# ---------------------------------------------------------------------------
def check_completed(runs_dir: Path, exp: dict) -> bool:
    """Vérifie si le run est déjà terminé (JSON de résultats présent et valide)."""
    run_tag = (
        f"ATLAS_A_v5_ext_{exp['variant']}_seed{exp['seed']}"
        f"_glu{exp['gate_hidden_dim']}"
        f"_lauc{exp['lauc_weight']:.2f}"
    )
    results_file = runs_dir / run_tag / f"results_ext_{run_tag}.json"
    if not results_file.exists():
        return False
    try:
        d = json.loads(results_file.read_text(encoding="utf-8"))
        return "test" in d and "macro_auc" in d.get("test", {})
    except Exception:
        return False


def run_experiment(
    exp: dict,
    data_root: str,
    index_path: str,
    runs_dir: str,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Lance un entraînement unique. Retourne (succès, message)."""
    cmd = [
        sys.executable, TRAINING_SCRIPT,
        "--variant",         exp["variant"],
        "--seed",            str(exp["seed"]),
        "--gate_hidden_dim", str(exp["gate_hidden_dim"]),
        "--lauc_weight",     str(exp["lauc_weight"]),
        "--data_root",       data_root,
        "--index_path",      index_path,
        "--runs_dir",        runs_dir,
    ]
    if dry_run:
        print(f"   [DRY-RUN] {' '.join(cmd)}")
        return True, "Dry-run — non exécuté"
    try:
        result = subprocess.run(cmd, capture_output=False,
                                text=True, timeout=3600 * 4)
        return (result.returncode == 0), (
            "Succès" if result.returncode == 0
            else f"Echec (exit {result.returncode})"
        )
    except subprocess.TimeoutExpired:
        return False, "Timeout (> 4 h)"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Orchestrateur des expériences de révision H5/H7/H8/M3/M4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Lancer toutes les expériences
  python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3

  # Simuler sans exécuter
  python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3 --dry_run

  # Reprendre (sauter les runs déjà complétés)
  python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3 --resume

  # Lancer uniquement certains IDs
  python run_extended_experiments.py --data_root /path/to/ptb-xl/1.0.3 --only H5H8 M3
        """
    )
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Chemin vers le répertoire PTB-XL 1.0.3")
    parser.add_argument("--index_path",  type=str,
                        default=str(Path(__file__).resolve().parent.parent / "index_complete.parquet"),
                        help="Chemin vers index_complete.parquet")
    parser.add_argument("--runs_dir",    type=str,
                        default=str(Path(__file__).resolve().parent.parent / "runs_extended"),
                        help="Répertoire de sortie pour les runs étendus")
    parser.add_argument("--resume",      action="store_true",
                        help="Ne relance pas les runs déjà complétés")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Affiche les commandes sans les exécuter")
    parser.add_argument("--only",        nargs="+", metavar="ID",
                        help="Lancer uniquement les IDs spécifiés",
                        choices=[e["id"] for e in EXPERIMENTS])
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Sélection des expériences à lancer
    experiments = EXPERIMENTS if args.only is None else [
        e for e in EXPERIMENTS if e["id"] in args.only
    ]

    print("=" * 80)
    print("ORCHESTRATEUR D'EXPÉRIENCES DE RÉVISION")
    print("=" * 80)
    print(f"Data root:  {args.data_root}")
    print(f"Index:      {args.index_path}")
    print(f"Output:     {runs_dir}")
    print(f"Mode:       {'DRY-RUN' if args.dry_run else 'EXÉCUTION'}")
    print(f"Resume:     {args.resume}")
    print()
    print(f"{'ID':<12} | {'Variant':<14} | {'Seed':<6} | {'GLU':<6} | {'LAUC':<5} | Description")
    print("-" * 90)
    for e in experiments:
        print(f"{e['id']:<12} | {e['variant']:<14} | {e['seed']:<6} | "
              f"{e['gate_hidden_dim']:<6} | {e['lauc_weight']:<5.2f} | {e['description']}")
    print("=" * 80)

    if args.resume:
        before = len(experiments)
        experiments = [e for e in experiments if not check_completed(runs_dir, e)]
        skipped = before - len(experiments)
        if skipped:
            print(f"\n{skipped} run(s) deja completes (--resume)")
        if not experiments:
            print("[OK] Toutes les experiences sont deja completees.")
            return

    print(f"\nRuns à exécuter: {len(experiments)}")
    print("-" * 80)

    log_file = runs_dir / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary  = []
    t_start  = time.time()

    for i, exp in enumerate(experiments, 1):
        t_exp = time.time()
        print(f"\n[{i}/{len(experiments)}] {exp['id']} — {exp['description']}")
        print(f"  variant={exp['variant']}, seed={exp['seed']}, "
              f"glu={exp['gate_hidden_dim']}, lauc={exp['lauc_weight']:.2f}")
        print("-" * 50)

        success, msg = run_experiment(
            exp,
            data_root=args.data_root,
            index_path=args.index_path,
            runs_dir=str(runs_dir),
            dry_run=args.dry_run,
        )
        elapsed = time.time() - t_exp
        status  = "OK" if success else "FAIL"
        print(f"  [{status}] {msg}  ({elapsed / 60:.1f} min)")

        summary.append({
            "id":         exp["id"],
            "variant":    exp["variant"],
            "seed":       exp["seed"],
            "gate_hidden_dim": exp["gate_hidden_dim"],
            "lauc_weight":exp["lauc_weight"],
            "success":    success,
            "message":    msg,
            "duration_s": elapsed,
        })
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()} | {exp['id']} | "
                f"seed={exp['seed']} | glu={exp['gate_hidden_dim']} | "
                f"lauc={exp['lauc_weight']:.2f} | {status} | {msg} | "
                f"{elapsed:.0f}s\n"
            )

    total_t   = time.time() - t_start
    successes = sum(1 for r in summary if r["success"])
    failures  = len(summary) - successes

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"Exécutés:  {len(summary)}")
    print(f"Réussites: {successes}")
    print(f"Échecs:    {failures}")
    print(f"Temps:     {total_t / 3600:.1f} h")
    print(f"Log:       {log_file}")

    if failures:
        print("\nExpériences échouées:")
        for r in summary:
            if not r["success"]:
                print(f"  - {r['id']} seed={r['seed']}: {r['message']}")

    summary_file = runs_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_file.write_text(json.dumps({
        "config": {
            "data_root":  args.data_root,
            "index_path": args.index_path,
            "runs_dir":   str(runs_dir),
        },
        "results": summary,
        "totals": {"runs": len(summary), "successes": successes,
                   "failures": failures, "total_hours": total_t / 3600},
    }, indent=2), encoding="utf-8")
    print(f"Résumé JSON: {summary_file}")
    print("=" * 80)

    if failures:
        print("\n[!] Utilisez --resume pour relancer les runs echoues.")
        sys.exit(1)
    else:
        print("\n[OK] Toutes les experiences sont terminees avec succes.")
        print("  Resultats dans:", runs_dir)


if __name__ == "__main__":
    main()
