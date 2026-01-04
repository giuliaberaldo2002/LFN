#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=viz_clusters
#SBATCH --output=output_viz_%j.txt       # File di log standard
#SBATCH --error=errors_viz_%j.txt        # File di log errori
#SBATCH --mem=32G                        # 32GB dovrebbero bastare per il plotting
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

echo "=== STARTING VISUALIZATION ==="

# Define filenames
INPUT_COMMS="nord_est_with_communities.pkl"
OUTPUT_IMG="nord_est_clusters.png"

echo "Using input: $INPUT_COMMS"

# Run Script
# Nota: Lo script usa matplotlib con backend 'Agg' automaticamente
python3 visualize_clusters.py --input "$INPUT_COMMS" --output "$OUTPUT_IMG"

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization failed."
    exit 1
fi

echo "=== VISUALIZATION COMPLETED SUCCESSFULLY ==="
echo "Output saved to: $OUTPUT_IMG"
