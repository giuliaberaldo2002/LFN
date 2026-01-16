#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=compare_algo
#SBATCH --output=output_compare_fix_conn_%j.txt
#SBATCH --error=errors_compare_fix_conn_%j.txt
#SBATCH --mem=500G                        
#SBATCH --time=8:00:00
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=geremia.paoletto@studenti.unipd.it
#SBATCH --mail-type=ALL

# --- INSTRUCTIONS ---

# 1. Go to project directory
cd $HOME/LFN_test_300/LFN_undir

# 2. Activate virtual environment
source $HOME/miniconda/bin/activate geo_env

echo "=== STARTING ALGORITHM COMPARISON ==="

# Define filenames
FULL_GRAPH="nord_est_processed_graph.pkl"
PRUNED_GRAPH="nord_est_with_communities_no_conn.pkl"
OUTPUT_DIR="comparison_results_500_final"

echo "Full Graph: $FULL_GRAPH"
echo "Pruned Graph: $PRUNED_GRAPH"

# Run Script
# NOTE: Ensure no spaces exist after the backslashes below
python3 compare_algorithms.py \
    --full "$FULL_GRAPH" \
    --repeats 1 \
    --pruned "$PRUNED_GRAPH" \
    --outdir "$OUTPUT_DIR" \
    --ks 10 20 50 100 200 500\
    --seed 42 43 44 45

if [ $? -ne 0 ]; then
    echo "ERROR: Comparison failed."
    exit 1
fi

echo "=== COMPARISON COMPLETED SUCCESSFULLY ==="
echo "Results saved to directory: $OUTPUT_DIR"