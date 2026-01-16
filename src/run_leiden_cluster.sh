#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=leiden_community
#SBATCH --output=output_leiden_%j.txt    # Standard log file
#SBATCH --error=errors_leiden_%j.txt     # Error log file
#SBATCH --mem=120G                        
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

echo "=== STARTING LEIDEN COMMUNITY DETECTION ==="

# Define filenames
INPUT_PRUNED="nord_est_pruned_graph.pkl"     # Output from previous pipeline step
OUTPUT_COMMS="nord_est_with_communities_no_conn.pkl"

echo "Using input: $INPUT_PRUNED"

# Run Script
python3 leiden_communities_undirected.py --input "$INPUT_PRUNED" --output "$OUTPUT_COMMS" --objective modularity

if [ $? -ne 0 ]; then
    echo "ERROR: Leiden detection failed."
    exit 1
fi

echo "=== LEIDEN COMPLETED SUCCESSFULLY ==="
echo "Output saved to: $OUTPUT_COMMS"
