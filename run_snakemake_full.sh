#!/bin/bash
#SBATCH --job-name=snakemake-master
#SBATCH --time=72:00:00            # wall clock for master process
#SBATCH --cpus-per-task=8          # lightweight — Snakemake itself is not heavy
#SBATCH --mem=8G                   
#SBATCH --output=./slurm_logs/snake.%j.out
#SBATCH --error=./slurm_logs/snake.%j.err


# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

mamba activate dino_env

# Run Snakemake with cluster submission
snakemake -s full_pipeline_snakefile all \
  --configfile="configs/scDINO_full_pipeline.yaml" \
  --keep-incomplete \
  --drop-metadata \
  --cores 8 \
  --jobs 40 \
  -k \
  --cluster "sbatch --time=48:30:00 \
             --gres=gpu:rtx4090:1 \
             --cpus-per-task=18 \
             --mem=80GB \
             --output=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_output_%j.txt \
             --error=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_error_%j.txt" \
  --latency-wait 90 \
  --rerun-incomplete \
  --forceall