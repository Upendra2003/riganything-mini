"""
Phase 4 — Configuration
========================
All hyperparameters for the Joint Diffusion Module in one place.

Example:
  from phase4.config import Phase4Config
  cfg = Phase4Config(epochs=100)
"""

from dataclasses import dataclass, field
import torch


@dataclass
class Phase4Config:
    # ── Model / diffusion ─────────────────────────────────────────────
    d:           int = 1024    # must match Phase 3 transformer dimension
    M:           int = 1000    # diffusion timesteps
    ddim_steps:  int = 50      # DDIM inference steps

    # ── Training ─────────────────────────────────────────────────────
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    batch_size:   int   = 1     # one shape at a time (inner loop over joints)
    epochs:       int   = 50
    warmup_steps: int   = 100
    grad_clip:    float = 1.0

    # ── Hardware / precision ─────────────────────────────────────────
    device: str  = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp:    bool = True

    # ── Paths ─────────────────────────────────────────────────────────
    phase3_ckpt:    str = 'checkpoints/phase3/best_model.pt'
    token_dir:      str = 'tokens/obj_remesh'
    skel_dir:       str = 'pointClouds/obj_remesh'
    train_split:    str = 'Dataset/train_final.txt'
    val_split:      str = 'Dataset/val_final.txt'
    checkpoint_dir: str = 'checkpoints/phase4'
    resume:         str = ''   # path to resume from; empty = start fresh
