#!/bin/bash
# ============================================================================
# run_all_experiments.sh - Script shell pour lancer la validation multi-graines
# ============================================================================
# CONFIGURATION: Modifiez les chemins ci-dessous selon votre environnement
# ============================================================================

DATA_ROOT="C:\\eznx\\data\\AXIOM12L_v103\\physionet.org\\files\\ptb-xl\\1.0.3"
INDEX_PATH="index_complete.parquet"
RUNS_DIR="C:\\eznx\\runs"
PYTHON="python"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "EZNX_ATLAS_A - Validation Multi-Graines"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  DATA_ROOT:  $DATA_ROOT"
echo "  INDEX_PATH: $INDEX_PATH"
echo "  RUNS_DIR:   $RUNS_DIR"
echo ""

# Vérification des fichiers
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}ERREUR: DATA_ROOT introuvable: $DATA_ROOT${NC}"
    echo "Modifiez DATA_ROOT dans ce script."
    exit 1
fi

if [ ! -f "$INDEX_PATH" ]; then
    echo -e "${RED}ERREUR: INDEX_PATH introuvable: $INDEX_PATH${NC}"
    exit 1
fi

show_menu() {
    echo "============================================================================"
    echo "Choisissez une option:"
    echo "  1. Lancer TOUTES les expériences (30 runs, ~20-30h)"
    echo "  2. Lancer un TEST rapide (9 runs avec 3 seeds)"
    echo "  3. Reprendre les expériences interrompues"
    echo "  4. Analyser les résultats existants"
    echo "  5. Lancer UN SEUL entraînement (interactif)"
    echo "  6. Quitter"
    echo "============================================================================"
    echo ""
}

run_analysis() {
    echo ""
    echo "============================================================================"
    echo "Lancement de l'analyse statistique..."
    echo "============================================================================"
    echo ""
    $PYTHON analyze_multiseed_results.py --runs_dir "$RUNS_DIR"
    
    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}Analyse terminée. Consultez les fichiers dans: $RUNS_DIR${NC}"
    echo "  - statistical_analysis_report.md"
    echo "  - table_results_latex.tex"
    echo "  - statistical_analysis_full.json"
    echo -e "${GREEN}============================================================================${NC}"
}

while true; do
    show_menu
    read -p "Votre choix (1-6): " choice
    
    case $choice in
        1)
            echo ""
            echo -e "${YELLOW}Lancement de TOUTES les expériences (30 runs)...${NC}"
            echo "Cela prendra environ 20-30 heures sur GPU."
            echo ""
            read -p "Confirmer? (o/n): " confirm
            if [ "$confirm" = "o" ]; then
                $PYTHON run_multiseed_experiments.py \
                    --data_root "$DATA_ROOT" \
                    --index_path "$INDEX_PATH" \
                    --runs_dir "$RUNS_DIR"
                run_analysis
            fi
            ;;
        2)
            echo ""
            echo -e "${YELLOW}Lancement du test rapide (9 runs avec 3 seeds)...${NC}"
            echo ""
            $PYTHON run_multiseed_experiments.py \
                --data_root "$DATA_ROOT" \
                --index_path "$INDEX_PATH" \
                --runs_dir "$RUNS_DIR" \
                --seeds 2024 2025 2026
            run_analysis
            ;;
        3)
            echo ""
            echo -e "${YELLOW}Reprise des expériences interrompues...${NC}"
            echo ""
            $PYTHON run_multiseed_experiments.py \
                --data_root "$DATA_ROOT" \
                --index_path "$INDEX_PATH" \
                --runs_dir "$RUNS_DIR" \
                --resume
            run_analysis
            ;;
        4)
            run_analysis
            ;;
        5)
            echo ""
            echo "Lancement d'un entraînement unique"
            echo ""
            read -p "Variante (none/demo/demo+anthro): " variant
            read -p "Seed (ex: 2026): " seed
            
            $PYTHON atlas_a_v5_multiseed.py \
                --variant "$variant" \
                --seed "$seed" \
                --data_root "$DATA_ROOT" \
                --index_path "$INDEX_PATH" \
                --runs_dir "$RUNS_DIR"
            
            echo ""
            echo -e "${GREEN}Entraînement terminé.${NC}"
            ;;
        6)
            echo ""
            echo "Au revoir!"
            exit 0
            ;;
        *)
            echo -e "${RED}Choix invalide${NC}"
            ;;
    esac
    
    echo ""
    read -p "Appuyez sur Entrée pour continuer..."
    echo ""
done
