#!/bin/bash

#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=64:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
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
export HF_HOME=/iliad/u/priyasun/huggingface_cache
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_rth_low_mem_finetune --exp_name=rth_libero --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_rth_low_mem_finetune --exp_name=rth_libero_pi05 --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_rth_lora_low_mem_finetune --exp_name=rth_libero_pi05_lora --overwrite
uv run examples/libero/augment_libero_data.py
wait
