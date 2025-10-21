#!/bin/bash

#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --job-name="pi0-rth"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /iliad/u/priyasun/miniconda3/bin/activate
cd /iliad/u/priyasun/openpi
source examples/libero/.venv/bin/activate
export HF_HOME=/iliad/u/priyasun/openpi/datasets
uv run scripts/compute_norm_stats.py --config-name pi05_libero_aug_lora_low_mem_finetune
wait
