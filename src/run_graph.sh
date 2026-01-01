#!/bin/bash

# --- CONFIGURAZIONE SLURM ---
#SBATCH --job-name=gen_grafo
#SBATCH --output=output_%j.txt    # File di log standard
#SBATCH --error=errors_%j.txt     # File di log errori
#SBATCH --mem=32G                 # 32GB per il Nord-Est (64G se farai Germania)
#SBATCH --time=00:30:00           # 30 minuti
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=geremia.paoletto@studenti.unipd.it
#SBATCH --mail-type=ALL

# --- ISTRUZIONI ---

# 1. Vai nella tua cartella LFN
# Nota: Usiamo $HOME che il cluster espande automaticamente nel tuo percorso utente
cd $HOME/LFN

# 2. Attiva l'ambiente virtuale che hai appena creato nello Step 1
source $HOME/miniconda/bin/activate geo_env

# 3. Esegui lo script Python per generare il grafo
python3 graph_init.py
