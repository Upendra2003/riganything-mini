"""
Phase 6 — Configuration
========================
All hyperparameters for the Skinning Weight Prediction module.

Example:
  from phase6.config import Phase6Config
  cfg = Phase6Config(epochs=30)
"""

from dataclasses import dataclass
import torch


@dataclass
class Phase6Config:
    # ── Model ─────────────────────────────────────────────────────────
    d: int = 1024   # must match Phase 2/3/4/5 token dimension

    # ── Training ─────────────────────────────────────────────────────
    lr:               float = 1e-4
    weight_decay:     float = 1e-5
    epochs:           int   = 30
    grad_clip:        float = 1.0
    accum_steps:      int   = 4     # gradient accumulation — effective batch = 4 shapes

    # ── Hardware / precision ─────────────────────────────────────────
    device: str  = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp:    bool = True

    # ── Paths ─────────────────────────────────────────────────────────
    token_dir:      str = 'tokens/obj_remesh'
    skel_dir:       str = 'pointClouds/obj_remesh'
    train_split:    str = 'Dataset/train_final.txt'
    val_split:      str = 'Dataset/val_final.txt'
    checkpoint_dir: str = 'checkpoints/phase6'
    resume:         str = ''
