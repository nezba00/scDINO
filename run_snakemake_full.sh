#!/bin/bash
#SBATCH --job-name=snakemake_main       # Name for the job
#SBATCH --time=01:00:00                 # Time limit (1 hour for coordinator)
#SBATCH --mem=4GB                       # Memory for the coordinator (doesn’t need much)
#SBATCH --cpus-per-task=1               # Cores for the coordinator (no heavy computation)
#SBATCH --output=snakemake_main_%j.log  # Output file (with job ID)
#SBATCH --error=snakemake_main_%j.log   # Error file (same as output for simplicity)

# Run Snakemake with cluster submission
snakemake -s full_pipeline_snakefile all \
  --configfile="configs/scDINO_full_pipeline.yaml" \
  --keep-incomplete \
  --drop-metadata \
  --cores 1 \                # Reduced to 1 (coordinator doesn’t need parallel local cores)
  --jobs 40 \                # Max concurrent SLURM jobs allowed
  -k \                       # Keep going if some jobs fail
  --cluster "sbatch --time=03:30:00 \
             --gres=gpu:rtx4090:1 \     # GPU for individual jobs
             --cpus-per-task=16 \       # CPUs per job (matches cluster resources)
             --mem=20GB \              # Memory per job
             --output=slurm_output_%j.txt \  # Use %j for unique job IDs
             --error=slurm_error_%j.txt" \
  --latency-wait 45 \        # Wait for filesystem sync
  --rerun-incomplete \       # Rerun incomplete steps
  --forceall                 # Force all rules to run