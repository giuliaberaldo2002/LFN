#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=viz_small
#SBATCH --output=output_viz_small_%j.txt       # Standard log file
#SBATCH --error=errors_viz_small_%j.txt        # Error log file
#SBATCH --mem=32G                        # 32GB should be sufficient for plotting
#SBATCH --time=00:30:00                  # 30 minutes
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

echo "=== STARTING VISUALIZATION OF SMALLEST COMMUNITIES ==="

# Define filenames
INPUT_COMMS="nord_est_with_communities_no_conn.pkl"
OUTPUT_IMG="nord_est_top_200_clusters_no_conn.png"

echo "Using input: $INPUT_COMMS"

# Run Script
# Note: The script uses matplotlib with 'Agg' backend automatically
python3 visualize_clusters.py --input "$INPUT_COMMS" --output "$OUTPUT_IMG" --top_k 300 

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization failed."
    exit 1
fi

echo "=== VISUALIZING PRUNED GRAPH ==="
OUTPUT_IMG_PRUNED="nord_est_pruned.png"

python3 visualize_clusters.py --input "$INPUT_COMMS" --output "$OUTPUT_IMG_PRUNED" --pruned

if [ $? -ne 0 ]; then
    echo "ERROR: Pruned visualization failed."
    exit 1
fi

echo "=== VISUALIZATION COMPLETED SUCCESSFULLY ==="
echo "Smallest clusters saved to: $OUTPUT_IMG"
echo "Pruned graph saved to: $OUTPUT_IMG_PRUNED"
