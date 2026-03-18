#!/bin/bash
#SBATCH --job-name=snakemake-master
#SBATCH --time=10:00:00            # wall clock for master process
#SBATCH --cpus-per-task=8          # lightweight — Snakemake itself is not heavy
#SBATCH --mem=8G
#SBATCH --output=./slurm_logs/snake.%j.out
#SBATCH --error=./slurm_logs/snake.%j.err
##SBATCH --nodelist=izbcotonou

# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

mamba activate dino_env #new_scdino

# Run Snakemake with cluster submission
snakemake -s only_downstream_snakefile all \
  --configfile="configs/only_downstream_analyses.yaml" \
  --keep-incomplete \
  --drop-metadata \
  --cores 8 \
  --jobs 1 \
  -k \
  --cluster "sbatch --time=09:30:00 \
             --gres=gpu:rtx4090:1 \
             --cpus-per-task=15 \
             --mem=125GB \
             --output=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_output_%j.txt \
             --error=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_error_%j.txt" \
  --latency-wait 90 \
  --rerun-incomplete #\
  #--forceall \
  #--verbose \
  #--unlock #\
  #--forceall