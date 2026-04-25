#!/usr/bin/env python3
# ============================================================================
# run_multiseed_experiments.py - Orchestrateur d'expÃ©riences multi-graines
# ============================================================================
# Lance automatiquement les 3 variantes Ã— 10 seeds = 30 entraÃ®nements
# GÃ¨re les erreurs et permet de reprendre les expÃ©riences interrompues
# ============================================================================

import subprocess
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import time


# ============================================================================
# CONFIGURATION
# ============================================================================

# Seeds pour validation statistique (10 seeds recommandÃ©es)
DEFAULT_SEEDS = [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]

# Variantes d'ablation
VARIANTS = ["none", "demo", "demo+anthro"]

# Script d'entraÃ®nement
TRAINING_SCRIPT = "atlas_a_v5_multiseed.py"


def check_completed(runs_dir: Path, variant: str, seed: int) -> bool:
    """VÃ©rifie si un run est dÃ©jÃ  complÃ©tÃ© (fichier JSON existe et contient des rÃ©sultats test)."""
    results_file = runs_dir / f"ATLAS_A_v5_{variant}_seed{seed}" / f"results_{variant}_seed{seed}.json"
    
    if not results_file.exists():
        return False
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        # VÃ©rifie que les rÃ©sultats test sont prÃ©sents
        return "test" in data and "macro_auc" in data.get("test", {})
    except:
        return False


def run_experiment(
    variant: str, 
    seed: int, 
    data_root: str,
    index_path: str,
    runs_dir: str,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """Lance un entraÃ®nement unique et retourne (succÃ¨s, message)."""
    
    cmd = [
        sys.executable,
        TRAINING_SCRIPT,
        "--variant", variant,
        "--seed", str(seed),
        "--data_root", data_root,
        "--index_path", index_path,
        "--runs_dir", runs_dir
    ]
    
    if dry_run:
        print(f"   [DRY-RUN] {' '.join(cmd)}")
        return True, "Dry run - not executed"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4  # 4 heures max par run
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Exit code {result.returncode}: {result.stderr[-500:]}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout (>4h)"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Orchestrateur d\'expÃ©riences multi-graines EZNX_ATLAS_A',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Lancer toutes les expÃ©riences (30 runs)
  python run_multiseed_experiments.py --data_root /path/to/ptb-xl --runs_dir ./runs
  
  # Lancer seulement une variante
  python run_multiseed_experiments.py --variants demo+anthro --data_root /path/to/ptb-xl
  
  # Lancer avec 5 seeds seulement
  python run_multiseed_experiments.py --seeds 2024 2025 2026 2027 2028 --data_root /path/to/ptb-xl
  
  # Mode simulation (dry-run)
  python run_multiseed_experiments.py --dry-run --data_root /path/to/ptb-xl
  
  # Reprendre les expÃ©riences manquantes
  python run_multiseed_experiments.py --resume --data_root /path/to/ptb-xl
        """
    )
    
    # Arguments obligatoires
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Chemin vers le rÃ©pertoire PTB-XL'
    )
    
    # Arguments optionnels
    parser.add_argument(
        '--index_path',
        type=str,
        default='index_complete.parquet',
        help='Chemin vers le fichier index parquet (dÃ©faut: index_complete.parquet)'
    )
    parser.add_argument(
        '--runs_dir',
        type=str,
        default='./runs',
        help='RÃ©pertoire de sortie pour les runs (dÃ©faut: ./runs)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help=f'Liste des seeds Ã  utiliser (dÃ©faut: {DEFAULT_SEEDS})'
    )
    parser.add_argument(
        '--variants',
        type=str,
        nargs='+',
        choices=VARIANTS,
        default=VARIANTS,
        help=f'Variantes Ã  entraÃ®ner (dÃ©faut: {VARIANTS})'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Ne relance pas les runs dÃ©jÃ  complÃ©tÃ©s'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Affiche les commandes sans les exÃ©cuter'
    )
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Liste des expÃ©riences Ã  lancer
    experiments = []
    for variant in args.variants:
        for seed in args.seeds:
            experiments.append((variant, seed))
    
    total = len(experiments)
    
    print("=" * 80)
    print("ORCHESTRATEUR D'EXPÃ‰RIENCES MULTI-GRAINES")
    print("=" * 80)
    print(f"Data root:   {args.data_root}")
    print(f"Index:       {args.index_path}")
    print(f"Output:      {runs_dir}")
    print(f"Variantes:   {args.variants}")
    print(f"Seeds:       {args.seeds}")
    print(f"Total runs:  {total}")
    print(f"Mode:        {'DRY-RUN' if args.dry_run else 'EXECUTION'}")
    print(f"Resume:      {args.resume}")
    print("=" * 80)
    
    # Filtrer les expÃ©riences dÃ©jÃ  complÃ©tÃ©es si --resume
    if args.resume:
        pending = []
        for variant, seed in experiments:
            if not check_completed(runs_dir, variant, seed):
                pending.append((variant, seed))
            else:
                print(f"   [SKIP] {variant} seed={seed} (dÃ©jÃ  complÃ©tÃ©)")
        experiments = pending
        print(f"\nExpÃ©riences restantes: {len(experiments)}/{total}")
    
    if not experiments:
        print("\nâœ“ Toutes les expÃ©riences sont dÃ©jÃ  complÃ©tÃ©es!")
        return
    
    # Log des expÃ©riences
    log_file = runs_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # ExÃ©cution
    print("\n" + "-" * 80)
    print("LANCEMENT DES EXPÃ‰RIENCES")
    print("-" * 80)
    
    results_summary = []
    start_time = time.time()
    
    for i, (variant, seed) in enumerate(experiments, 1):
        exp_start = time.time()
        print(f"\n[{i}/{len(experiments)}] Variante: {variant}, Seed: {seed}")
        print("-" * 40)
        
        success, message = run_experiment(
            variant=variant,
            seed=seed,
            data_root=args.data_root,
            index_path=args.index_path,
            runs_dir=str(runs_dir),
            dry_run=args.dry_run
        )
        
        exp_duration = time.time() - exp_start
        status = "âœ“" if success else "âœ—"
        
        result = {
            "variant": variant,
            "seed": seed,
            "success": success,
            "message": message,
            "duration_seconds": exp_duration
        }
        results_summary.append(result)
        
        print(f"   {status} {message} ({exp_duration/60:.1f} min)")
        
        # Log en temps rÃ©el
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} | {variant} | seed={seed} | {status} | {message} | {exp_duration:.0f}s\n")
    
    # RÃ©sumÃ© final
    total_time = time.time() - start_time
    successes = sum(1 for r in results_summary if r["success"])
    failures = len(results_summary) - successes
    
    print("\n" + "=" * 80)
    print("RÃ‰SUMÃ‰ FINAL")
    print("=" * 80)
    print(f"Total exÃ©cutÃ©es: {len(results_summary)}")
    print(f"RÃ©ussites:       {successes}")
    print(f"Ã‰checs:          {failures}")
    print(f"Temps total:     {total_time/3600:.1f} heures")
    print(f"Log:             {log_file}")
    
    if failures > 0:
        print("\nExpÃ©riences Ã©chouÃ©es:")
        for r in results_summary:
            if not r["success"]:
                print(f"   - {r['variant']} seed={r['seed']}: {r['message']}")
    
    # Sauvegarder le rÃ©sumÃ© JSON
    summary_file = runs_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "config": {
                "data_root": args.data_root,
                "index_path": args.index_path,
                "runs_dir": str(runs_dir),
                "seeds": args.seeds,
                "variants": args.variants
            },
            "results": results_summary,
            "summary": {
                "total": len(results_summary),
                "successes": successes,
                "failures": failures,
                "total_time_hours": total_time / 3600
            }
        }, f, indent=2)
    
    print(f"\nRÃ©sumÃ© sauvegardÃ©: {summary_file}")
    print("=" * 80)
    
    if failures > 0:
        print("\nâš ï¸  Certaines expÃ©riences ont Ã©chouÃ©. Utilisez --resume pour les relancer.")
        sys.exit(1)
    else:
        print("\nâœ“ Toutes les expÃ©riences sont terminÃ©es avec succÃ¨s!")
        print("  Lancez maintenant: python analyze_multiseed_results.py --runs_dir", runs_dir)


if __name__ == "__main__":
    main()
