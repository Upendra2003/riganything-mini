"""
Phase 3 — Configuration
=========================
All hyperparameters in one place.  Import Config and override fields as needed.

Example:
  from phase3.config import Config
  cfg = Config(epochs=100, batch_size=4)
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ── Model architecture ─────────────────────────────────────────
    d:          int   = 1024      # transformer dimension
    n_heads:    int   = 16        # number of attention heads
    d_k:        int   = 64        # keys/queries per head (= d // n_heads)
    ffn_dim:    int   = 4096      # feed-forward hidden dimension
    n_layers:   int   = 10         # number of transformer blocks
    L:          int   = 1024      # fixed shape-token count (matches Phase 2 output)

    # ── Training ───────────────────────────────────────────────────
    lr:               float = 1e-4
    weight_decay:     float = 1e-5
    batch_size:       int   = 2
    epochs:           int   = 50
    grad_clip:        float = 1.0     # max_norm for gradient clipping
    warmup_steps:     int   = 100     # linear LR warmup steps
    checkpoint_every: int   = 5       # save checkpoint every N epochs
    seed:             int   = 42

    # ── Paths ──────────────────────────────────────────────────────
    checkpoint_dir: str = 'checkpoints/phase3'
    tokens_dir:     str = 'tokens/obj_remesh'

    # ── Hardware / precision ───────────────────────────────────────
    device:              str  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp:             bool = True   # automatic mixed precision (fp16 forward)
    use_grad_checkpoint: bool = True   # trade ~30% speed for ~70% activation memory
