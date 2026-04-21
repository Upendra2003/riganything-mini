#!/bin/bash
#SBATCH --job-name=riganything_phase7
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --account=student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/phase7_%j.out
#SBATCH --error=logs/phase7_%j.err

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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── Verify CUDA is actually available ────────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check PyTorch/driver version mismatch'; print('CUDA OK:', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

# ── Resume if checkpoint exists ───────────────────────────────────────────
CKPT="checkpoints/phase7/best_model.pt"
if [ -f "$CKPT" ]; then
    echo "Resuming from $CKPT"
    python phase7/train.py --epochs 50 --max_shapes 200 --resume "$CKPT"
else
    echo "Starting fresh (no checkpoint found at $CKPT)"
    python phase7/train.py --epochs 50 --max_shapes 200
fi

echo "======================================================"
echo "Finished:  $(date)"
echo "======================================================"
