"""
Phase 5 — Connectivity Prediction Modules
==========================================
Given the context vector Z_k from Phase 3 and the ground-truth joint position
j_k (teacher-forced during training, DDIM-sampled at inference), these modules
predict which of the k-1 previously generated joints is the parent of joint k.

Architecture:
  FusingModule:       MLP(Z_k.mean + j_k + gamma(k)) → Z'_k  [d]
  ConnectivityModule: score each T_prev[i] paired with Z'_k → softmax → q_k
  connectivity_loss:  BCE over one-hot parent target

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase5/train.py --epochs 30
  .venv/bin/python tests/phase5_test.py
  .venv/bin/python phase5/inference.py --shape_id <id>
  # Resume:
  .venv/bin/python phase5/train.py --resume checkpoints/phase5/best_model.pt
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Make project root importable when this module is imported directly ─────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tokenizer import sinusoidal_embedding


# ---------------------------------------------------------------------------
# FusingModule
# ---------------------------------------------------------------------------

class FusingModule(nn.Module):
    """
    Incorporates the predicted joint position j_k and positional index k
    into the transformer context Z_k, producing a single conditioning vector.

    Input shapes:
      Z_k:    [L+k-1, d]   context vectors from Phase 3
      j_k:    [3]           joint position (ground truth or DDIM sample)
      k:      int           1-indexed joint number (≥ 2)

    Output:
      Z'_k:  [d]
    """

    def __init__(self, d: int = 1024):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(
            nn.Linear(2 * d + 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, d),
        )

    def forward(
        self,
        Z_k: torch.Tensor,
        j_k: torch.Tensor,
        k:   int,
    ) -> torch.Tensor:
        device = Z_k.device
        z_mean  = Z_k.mean(dim=0)                                  # [d]
        gamma_k = sinusoidal_embedding(k, self.d).to(device)       # [d]
        x = torch.cat([z_mean, j_k, gamma_k], dim=0)              # [2d+3]
        return self.mlp(x)                                          # [d]


# ---------------------------------------------------------------------------
# ConnectivityModule
# ---------------------------------------------------------------------------

class ConnectivityModule(nn.Module):
    """
    Scores each candidate parent token T_prev[i] against the fused context
    Z'_k and returns a probability distribution over candidate parents.

    Input shapes:
      Z_prime_k: [d]       fused context from FusingModule
      T_prev:    [k-1, d]  skeleton tokens for joints 0..k-2

    Output:
      q_k: [k-1]   softmax probability over candidate parents
    """

    def __init__(self, d: int = 1024):
        super().__init__()
        self.d = d
        self.scorer = nn.Sequential(
            nn.Linear(2 * d, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(
        self,
        Z_prime_k: torch.Tensor,
        T_prev:    torch.Tensor,
    ) -> torch.Tensor:
        k_minus_1  = T_prev.shape[0]
        z_expanded = Z_prime_k.unsqueeze(0).expand(k_minus_1, -1)  # [k-1, d]
        pairs      = torch.cat([z_expanded, T_prev], dim=-1)        # [k-1, 2d]
        scores     = self.scorer(pairs).squeeze(-1)                  # [k-1]
        return F.softmax(scores, dim=0)                              # [k-1]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def connectivity_loss(
    q_k:             torch.Tensor,
    true_parent_idx: int,
) -> torch.Tensor:
    """
    Binary cross-entropy over the one-hot parent distribution.

    Args:
        q_k:             [k-1]  predicted distribution (softmax output)
        true_parent_idx: int    0-indexed position of true parent in T_prev

    Returns:
        scalar loss tensor
    """
    k_minus_1 = q_k.shape[0]
    y_hat = torch.zeros(k_minus_1, device=q_k.device, dtype=q_k.dtype)
    y_hat[true_parent_idx] = 1.0
    loss = -(y_hat * torch.log(q_k + 1e-8) + (1.0 - y_hat) * torch.log(1.0 - q_k + 1e-8))
    return loss.sum()
