#!/bin/bash
#SBATCH --job-name=riganything_phase4
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --account=student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase4_%j.out
#SBATCH --error=logs/phase4_%j.err

# ── Setup ─────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "======================================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Directory: $(pwd)"
echo "======================================================"

nvidia-smi

source .venv/bin/activate
echo "Python:    $(which python)"
echo "------------------------------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Resume if checkpoint exists ───────────────────────────────────────────
CKPT="checkpoints/phase4/best_model.pt"
if [ -f "$CKPT" ]; then
    echo "Resuming from $CKPT"
    python phase4/train.py --epochs 50 --resume "$CKPT"
else
    echo "Starting fresh (no checkpoint found at $CKPT)"
    python phase4/train.py --epochs 50
fi

echo "======================================================"
echo "Finished:  $(date)"
echo "======================================================"
