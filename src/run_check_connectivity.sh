#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=check_comm
#SBATCH --output=output_comm_%j.txt    
#SBATCH --error=errors_comm_%j.txt     
#SBATCH --mem=32G                     
#SBATCH --time=00:15:00                
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=geremia.paoletto@studenti.unipd.it
#SBATCH --mail-type=ALL

# --- ISTRUCTIONS ---

# 1. Vai nella tua cartella LFN
cd $HOME/LFN_test_300/LFN_undir

# 2. Attiva l'ambiente virtuale
source $HOME/miniconda/bin/activate geo_env

echo "=== STARTING COMMUNITY CHECK ==="

# Define filename based on request (assuming nord_est_pruned_graph.pkl)
INPUT_FILE="nord_est_with_communities.pkl"

echo "Using input: $INPUT_FILE"

# Run Script
python3 check_connectivity.py --input "$INPUT_FILE" --output_plot "smallest_communities_sizes.png"


if [ $? -ne 0 ]; then
    echo "ERROR: Community check failed."
    exit 1
fi

echo "=== CHECK COMPLETED SUCCESSFULLY ==="
