#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=viz_interactive
#SBATCH --output=output_viz_int_%j.txt   # File di log standard
#SBATCH --error=errors_viz_int_%j.txt    # File di log errori
#SBATCH --mem=32G                        # 32GB
#SBATCH --time=00:30:00                  # 30 minuti
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

echo "=== STARTING INTERACTIVE VISUALIZATION ==="

# Define filenames
INPUT_COMMS="nord_est_with_communities.pkl"
OUTPUT_HTML="nord_est_map.html"

echo "Using input: $INPUT_COMMS"

# Run Script
python3 visualize_clusters_interactive.py --input "$INPUT_COMMS" --output "$OUTPUT_HTML"

if [ $? -ne 0 ]; then
    echo "ERROR: Interactive visualization failed."
    exit 1
fi

echo "=== INTERACTIVE VISUALIZATION COMPLETED SUCCESSFULLY ==="
echo "Output saved to: $OUTPUT_HTML"
