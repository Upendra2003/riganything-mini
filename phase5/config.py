"""
Phase 5 — Configuration
========================
All hyperparameters for the Connectivity Prediction modules.

Example:
  from phase5.config import Phase5Config
  cfg = Phase5Config(epochs=30)
"""

from dataclasses import dataclass
import torch


@dataclass
class Phase5Config:
    # ── Model ─────────────────────────────────────────────────────────
    d: int = 1024   # must match Phase 3/4 dimension

    # ── Training ─────────────────────────────────────────────────────
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    epochs:       int   = 30
    warmup_steps: int   = 100
    grad_clip:    float = 1.0

    # ── Diffusion (for inference only — Phase 4 config values) ────────
    M:          int = 1000
    ddim_steps: int = 50

    # ── Hardware / precision ─────────────────────────────────────────
    device: str  = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp:    bool = True

    # ── Paths ─────────────────────────────────────────────────────────
    phase3_ckpt:    str = 'checkpoints/phase3/best_model.pt'
    phase4_ckpt:    str = 'checkpoints/phase4/best_model.pt'
    token_dir:      str = 'tokens/obj_remesh'
    skel_dir:       str = 'pointClouds/obj_remesh'
    train_split:    str = 'Dataset/train_final.txt'
    val_split:      str = 'Dataset/val_final.txt'
    checkpoint_dir: str = 'checkpoints/phase5'
    resume:         str = ''
