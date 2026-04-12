"""
Phase 3 — Hybrid Attention Transformer
========================================
Autoregressive transformer that processes concatenated shape tokens (H) and
skeleton tokens (T_prev) with a hybrid attention mask:

  * Shape tokens  → full bidirectional attention.
  * Skeleton tokens → causal self-attention + full cross-attention to all shape tokens.

Forward signature:
  Z_k = model(H, T_prev)

  H:      FloatTensor [L, d]    — shape tokens from Phase 2
  T_prev: FloatTensor [k-1, d]  — skeleton tokens 0 .. k-2 (may be empty)
  Z_k:    FloatTensor [L+k-1, d]— context vectors at autoregressive step k

Parameter count (full config: d=1024, n_heads=16, ffn_dim=4096, n_layers=12):
  12 × 12,596,224 = 151,154,688
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from phase3.hybrid_mask import build_hybrid_mask


# ---------------------------------------------------------------------------
# Multi-head self-attention (unbatched: input [S, d])
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_k = d // n_heads
        self.d = d

        self.q_proj   = nn.Linear(d, d)
        self.k_proj   = nn.Linear(d, d)
        self.v_proj   = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [S, d]
        mask: [S, S]  additive bias (0 or -inf)
        Returns [S, d]
        """
        S = x.shape[0]
        h, dk = self.n_heads, self.d_k

        Q = self.q_proj(x).view(S, h, dk).transpose(0, 1)   # [h, S, dk]
        K = self.k_proj(x).view(S, h, dk).transpose(0, 1)   # [h, S, dk]
        V = self.v_proj(x).view(S, h, dk).transpose(0, 1)   # [h, S, dk]

        scale  = dk ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [h, S, S]
        scores = scores + mask.unsqueeze(0)                     # broadcast over heads
        weights = F.softmax(scores, dim=-1)                     # [h, S, S]

        out = torch.matmul(weights, V)                          # [h, S, dk]
        out = out.transpose(0, 1).contiguous().view(S, self.d)  # [S, d]
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block: Pre-LN design with GELU FFN
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.ln2  = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Hybrid Attention Transformer
# ---------------------------------------------------------------------------
class HybridTransformer(nn.Module):
    """
    Stack of TransformerBlocks with hybrid attention masking.

    Args:
        config: Config dataclass with fields d, n_heads, ffn_dim, n_layers,
                L, use_grad_checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self.d   = config.d
        self.L   = config.L
        self.use_grad_checkpoint = getattr(config, 'use_grad_checkpoint', False)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d, config.n_heads, config.ffn_dim)
            for _ in range(config.n_layers)
        ])

    def forward(self, H: torch.Tensor, T_prev: torch.Tensor) -> torch.Tensor:
        """
        H:      [L, d]    shape tokens
        T_prev: [k-1, d]  skeleton prefix (may be empty when k=1)
        Returns [L + k-1, d]
        """
        k_minus_1 = T_prev.shape[0]

        if k_minus_1 == 0:
            x = H
        else:
            x = torch.cat([H, T_prev], dim=0)    # [L + k-1, d]

        # Build hybrid mask and move to the same device / dtype as x
        mask = build_hybrid_mask(H.shape[0], k_minus_1).to(
            device=x.device, dtype=x.dtype
        )

        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = grad_checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        return x
