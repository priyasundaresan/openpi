#!/bin/bash

#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --job-name="convert"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /iliad/u/priyasun/miniconda3/bin/activate
cd /iliad/u/priyasun/openpi
source examples/libero/.venv/bin/activate
export HF_HOME=/iliad/u/priyasun/openpi/datasets
uv run examples/libero/convert_libero_data_to_aug_lerobot.py --data-dir modified_libero_rlds --json-dir /iliad/u/priyasun/openpi
wait
