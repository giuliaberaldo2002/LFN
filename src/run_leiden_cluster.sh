#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=leiden_community
#SBATCH --output=output_leiden_%j.txt    # File di log standard
#SBATCH --error=errors_leiden_%j.txt     # File di log errori
#SBATCH --mem=120G                        
#SBATCH --time=00:30:00                    # 30 minuti
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

echo "=== STARTING LEIDEN COMMUNITY DETECTION ==="

# Define filenames
INPUT_PRUNED="nord_est_pruned_graph.pkl"     # Output della pipeline precedente
OUTPUT_COMMS="nord_est_with_communities.pkl"

echo "Using input: $INPUT_PRUNED"

# Run Script
python3 leiden_communities_undirected.py --input "$INPUT_PRUNED" --output "$OUTPUT_COMMS" --objective CPM

if [ $? -ne 0 ]; then
    echo "ERROR: Leiden detection failed."
    exit 1
fi

echo "=== LEIDEN COMPLETED SUCCESSFULLY ==="
echo "Output saved to: $OUTPUT_COMMS"
