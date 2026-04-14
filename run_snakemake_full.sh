#!/bin/bash
#SBATCH --job-name=snakemake-master
#SBATCH --time=190:00:00            # wall clock for master process
#SBATCH --cpus-per-task=8          # lightweight — Snakemake itself is not heavy
#SBATCH --mem=8G
#SBATCH --output=./slurm_logs/snake.%j.out
#SBATCH --error=./slurm_logs/snake.%j.err
##SBATCH --nodelist=izbdhaka


# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

mamba activate dino_env #new_scdino

# Run Snakemake with cluster submission
snakemake -s full_pipeline_snakefile all \
  --configfile="configs/scDINO_full_pipeline.yaml" \
  --keep-incomplete \
  --drop-metadata \
  --cores 8 \
  --jobs 1 \
  -k \
  --cluster "sbatch --time=90:30:00 \
             --gres=gpu:rtx6000:2 \
             --cpus-per-task=20 \
             --mem=100GB \
             --output=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_output_%j.txt \
             --error=/home/nbahou/myimaging/scDINO/slurm_logs/slurm_error_%j.txt" \
  --latency-wait 300 #\
  #--rerun-incomplete #\
  #--forceall \
  #--verbose \
  #--unlock #\
  #--forceall