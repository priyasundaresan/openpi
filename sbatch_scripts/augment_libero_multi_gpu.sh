#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --job-name="libero-paraphrase"

# ----------------------------
# List of datasets (one per array task)
# ----------------------------
DATASETS=("libero_10_no_noops" "libero_goal_no_noops" "libero_object_no_noops" "libero_spatial_no_noops")

# Assign dataset based on SLURM array index
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# GPU index (SLURM allocates one GPU per task)
GPU_ID=0

echo "Processing dataset: $DATASET on GPU $GPU_ID"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"

# ----------------------------
# Environment setup
# ----------------------------
source /iliad/u/priyasun/miniconda3/bin/activate
cd /iliad/u/priyasun/openpi
source examples/libero/.venv/bin/activate
export HF_HOME=/iliad/u/priyasun/openpi/datasets

# ----------------------------
# Run Python script
# ----------------------------
uv run examples/libero/augment_libero_data.py --device $GPU_ID --dataset $DATASET

wait
