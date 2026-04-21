"""
Phase 6 — Skinning Weight Prediction
======================================
SkinningModule predicts per-point skinning weights W [L, K] from pre-tokenized
shape tokens H [L, d] and skeleton tokens T [K, d].

Architecture:
  Pairwise expansion: broadcast H and T to [L, K, 2d]
  MLP scorer:         Linear(2d, 1024) → ReLU → Linear(1024, 1) → [L, K]
  Row-wise Softmax:   rows sum to 1 (valid convex combination of joints)

Loss:
  Weighted cross-entropy where gt weights serve as both target and per-class weight:
    L_skinning = (1/L) * Σ_l Σ_k [ -ŵ_{l,k} · log(w_{l,k} + 1e-8) ]

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase6/train.py --epochs 30
  .venv/bin/python tests/phase6_test.py
  .venv/bin/python phase6/inference.py --shape_id <id>
  # Resume:
  .venv/bin/python phase6/train.py --resume checkpoints/phase6/best_model.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkinningModule(nn.Module):
    """
    Predicts skinning weights W [L, K] from shape tokens H and skeleton tokens T.

    For each (point, joint) pair the scorer computes a raw logit from the
    concatenated [H_l, T_k] representation; row-wise softmax ensures each
    point's weights form a valid probability distribution over joints.

    Parameter count (d=1024):
      Linear(2048 → 1024): 2048*1024 + 1024 = 2,098,176
      Linear(1024 → 1):    1024*1    + 1    = 1,025
      Total ≈ 2,099,201
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
        H:     torch.Tensor,
        T_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            H:     [L, d]  shape tokens (L = 1024)
            T_all: [K, d]  skeleton tokens

        Returns:
            W: [L, K]  skinning weights, rows sum to 1
        """
        L, d = H.shape
        K    = T_all.shape[0]

        h_exp = H.unsqueeze(1).expand(L, K, d)      # [L, K, d]
        t_exp = T_all.unsqueeze(0).expand(L, K, d)   # [L, K, d]
        pairs = torch.cat([h_exp, t_exp], dim=-1)     # [L, K, 2d]

        scores = self.scorer(pairs).squeeze(-1)       # [L, K]
        return F.softmax(scores, dim=-1)              # [L, K]


def skinning_loss(W: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
    """
    Weighted cross-entropy skinning loss.

    Ground-truth weights ŵ_{l,k} act as both the target AND the per-class weight,
    so bones with large influence on a point dominate the loss — mirrors the
    paper formulation.

    Args:
        W:     [L, K]  predicted skinning weights (softmax output)
        W_hat: [L, K]  ground-truth skinning weights (rows sum to 1)

    Returns:
        scalar loss tensor
    """
    L = W.shape[0]
    return (-(W_hat * torch.log(W + 1e-8)).sum()) / L
