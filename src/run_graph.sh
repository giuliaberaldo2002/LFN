#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=urban_tuning    # Updated job name
#SBATCH --output=output_urb_%j.txt # Standard log
#SBATCH --error=errors_urb_%j.txt  # Error log
#SBATCH --mem=120G                 # 120GB
#SBATCH --time=01:00:00            # Increased to 1h for safety (tuning does many trials)
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=geremia.paoletto@studenti.unipd.it
#SBATCH --mail-type=ALL

# --- INSTRUCTIONS ---

# Go to your folder
cd $HOME/LFN

# Activate environment
source $HOME/miniconda/bin/activate geo_env

# 3. UPDATED COMMAND
# The new script requires "--input" (saved graph) and "--output" (save weights)
python3 urbanity_tuning.py \
    --input bremen_pruned_with_communities.pkl \
    --output urbanity_weights.json \
    --trials 800 \
    --sample 80000