#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=check_comm
#SBATCH --output=output_comm_%j.txt    # File di log standard
#SBATCH --error=errors_comm_%j.txt     # File di log errori
#SBATCH --mem=32G                      # Should be enough for loading the graph
#SBATCH --time=00:15:00                # 15 minutes should be plenty
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=geremia.paoletto@studenti.unipd.it
#SBATCH --mail-type=ALL

# --- ISTRUZIONI ---

# 1. Vai nella tua cartella LFN
cd $HOME/LFN2

# 2. Attiva l'ambiente virtuale
source $HOME/miniconda/bin/activate geo_env

echo "=== STARTING COMMUNITY CHECK ==="

# Define filename based on request (assuming nord_est_pruned_graph.pkl)
INPUT_FILE="nord_est_pruned_graph.pkl"

echo "Using input: $INPUT_FILE"

# Run Script
python3 check_connectivity.py --input "$INPUT_FILE" --output_plot "smallest_communities_sizes.png"


if [ $? -ne 0 ]; then
    echo "ERROR: Community check failed."
    exit 1
fi

echo "=== CHECK COMPLETED SUCCESSFULLY ==="
