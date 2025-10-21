#!/bin/bash
#SBATCH --job-name=libero-paraphrase
#SBATCH --output=logs/libero_%A_%a.out
#SBATCH --error=logs/libero_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=sc-loprio
#SBATCH --account=iliad
#SBATCH --constraint=141G
#SBATCH --array=0-3

# ----------------------------
# Dataset per array job
# ----------------------------
DATASETS=("libero_10_no_noops" "libero_goal_no_noops" "libero_object_no_noops" "libero_spatial_no_noops")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
GPU_ID=0

echo "Processing dataset: $DATASET on GPU $GPU_ID"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "--------------------------------------"

# Environment setup
source /iliad/u/priyasun/miniconda3/bin/activate
cd /iliad/u/priyasun/openpi
source examples/libero/.venv/bin/activate
export HF_HOME=/iliad/u/priyasun/openpi/datasets

# Output directory (optional; matches default in Python)
OUTPUT_DIR="/iliad/u/priyasun/openpi/output_qwen_checkpoints"
mkdir -p "$OUTPUT_DIR"

# ----------------------------
# Preemption handling is automatic in Python
# ----------------------------

# Run the Python script
uv run examples/libero/augment_libero_data_qwen.py \
    --dataset "$DATASET" \
    --device "$GPU_ID" \
    --output_dir "$OUTPUT_DIR"

echo "Job completed for dataset: $DATASET"
echo "End time: $(date)"
echo "--------------------------------------"
