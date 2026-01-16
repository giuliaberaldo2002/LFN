#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=fix_connectivity
#SBATCH --output=output_connect_%j.txt    # Standard log file
#SBATCH --error=errors_connect_%j.txt     # Error log file
#SBATCH --mem=64G                         
#SBATCH --time=00:30:00                    # 30 minutes
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

echo "=== STARTING CONNECTIVITY FIX ==="

# Define filenames
INPUT_PRUNED="nord_est_pruned_graph.pkl"
OUTPUT_CONNECTED="nord_est_pruned_connected_graph.pkl"

echo "Input: $INPUT_PRUNED"
echo "Output: $OUTPUT_CONNECTED"

# Run Script
python3 fix_connectivity.py --input "$INPUT_PRUNED" --output "$OUTPUT_CONNECTED"

if [ $? -ne 0 ]; then
    echo "ERROR: Connectivity fix failed."
    exit 1
fi

echo "=== CONNECTIVITY FIX COMPLETED SUCCESSFULLY ==="
echo "Output saved to: $OUTPUT_CONNECTED"
