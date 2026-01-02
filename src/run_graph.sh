#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=urban_tuning    # 1. Nome aggiornato
#SBATCH --output=output_urb_%j.txt # Log standard distinto
#SBATCH --error=errors_urb_%j.txt  # Log errori distinto
#SBATCH --mem=120G                  # 120GB 
#SBATCH --time=01:00:00            # 2. Aumentato a 1h per sicurezza (il tuning fa molte prove)
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=geremia.paoletto@studenti.unipd.it
#SBATCH --mail-type=ALL

# --- ISTRUZIONI ---

# Vai nella tua cartella
cd $HOME/LFN

# Attiva l'ambiente
source $HOME/miniconda/bin/activate geo_env

# 3. COMANDO AGGIORNATO
# Il nuovo script richiede "--input" (il grafo salvato prima) e "--output" (dove salvare i pesi)
python3 urbanity_tuning.py \
    --input bremen_pruned_with_communities.pkl \
    --output urbanity_weights.json \
    --trials 800 \
    --sample 80000