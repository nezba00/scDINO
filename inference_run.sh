#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --time=190:00:00            # wall clock for master process
#SBATCH --cpus-per-task=18          # lightweight — Snakemake itself is not heavy
#SBATCH --mem=100G
#SBATCH --output=./slurm_logs/inf.%j.out
#SBATCH --error=./slurm_logs/inf.%j.err
#SBATCH --gres=gpu:rtx4090:1


# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

mamba activate full_dino_env #new_scdino

python inference_script.py --config inference_config.yml