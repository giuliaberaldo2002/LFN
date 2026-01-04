#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=urbanity_pipeline
#SBATCH --output=output_pipeline_%j.txt    # File di log standard
#SBATCH --error=errors_pipeline_%j.txt     # File di log errori
#SBATCH --mem=64G                          # Aumentato a 64G per Nord-Est
#SBATCH --time=01:00:00                    # 1 ora (stima prudente)
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

echo "=== STARTING PIPELINE ==="

# Define filenames
INPUT_RAW="nord_est_processed_graph.pkl"  # Questo file devo esistere (da graph_init_corrected.py)
WITH_FEATS="nord_est_graph_with_features.pkl"
WEIGHTS="nord_est_urbanity_weights.json"
OUTPUT_PRUNED="nord_est_pruned_graph.pkl"

echo "Using input: $INPUT_RAW"

# 3. Compute Features (Nuovo Step)
echo "[1/3] Computing Graph Features..."
python3 compute_features.py --input "$INPUT_RAW" --output "$WITH_FEATS"
if [ $? -ne 0 ]; then
    echo "ERROR: Feature computation failed."
    exit 1
fi

# 4. Tune Urbanity
echo "[2/3] Tuning Urbanity Weights..."
# Usiamo il grafo con features come input
python3 urbanity_tuning.py --input "$WITH_FEATS" --output "$WEIGHTS" --trials 800 --sample 80000
if [ $? -ne 0 ]; then
    echo "ERROR: Tuning failed."
    exit 1
fi

# 5. Prune Graph
echo "[3/3] Pruning Graph..."
python3 urban_pruning_final.py --input "$WITH_FEATS" --weights "$WEIGHTS" --output "$OUTPUT_PRUNED"
if [ $? -ne 0 ]; then
    echo "ERROR: Pruning failed."
    exit 1
fi

echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="
echo "Output saved to: $OUTPUT_PRUNED"
