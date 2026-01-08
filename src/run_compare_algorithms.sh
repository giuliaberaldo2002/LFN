#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=compare_algo
#SBATCH --output=output_compare_%j.txt   # File di log standard
#SBATCH --error=errors_compare_%j.txt    # File di log errori
#SBATCH --mem=64G                        # 64GB per sicurezza (metriche su grafi grandi)
#SBATCH --time=01:00:00                  # 1 ora
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

echo "=== STARTING ALGORITHM COMPARISON ==="

# Define filenames
FULL_GRAPH="nord_est_processed_graph.pkl"      # Grafo originale intero
PRUNED_GRAPH="nord_est_with_communities.pkl"   # Grafo potato con le comunità
OUTPUT_DIR="comparison_results"

echo "Full Graph: $FULL_GRAPH"
echo "Pruned Graph: $PRUNED_GRAPH"

# Run Script
python3 compare_algorithms.py \
    --full "$FULL_GRAPH" \
    --pruned "$PRUNED_GRAPH" \
    --outdir "$OUTPUT_DIR" \
    --ks 10 20 50 100 200 500 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Comparison failed."
    exit 1
fi

echo "=== COMPARISON COMPLETED SUCCESSFULLY ==="
echo "Results saved to directory: $OUTPUT_DIR"
