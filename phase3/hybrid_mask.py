"""
Phase 3 — Hybrid Attention Mask
=================================
Builds the additive attention bias for the Hybrid Attention Transformer.

Layout of the sequence [H_0 ... H_{L-1} | T_0 ... T_{k-2}]:

  Shape tokens (positions 0 .. L-1):
    Bidirectional — attend to every position in the sequence.

  Skeleton tokens (positions L .. L+k_minus_1-1):
    * Full cross-attention to all shape tokens (cols 0 .. L-1).
    * Causal self-attention among themselves:
        position L+i can attend to L+0 .. L+i (diagonal inclusive).
        position L+i cannot attend to L+i+1 .. end (upper triangle → -inf).

Usage:
  mask = build_hybrid_mask(L=1024, k_minus_1=5)  # shape (1029, 1029)
  # Then add to attention logits before softmax.
"""

import torch


def build_hybrid_mask(L: int, k_minus_1: int) -> torch.Tensor:
    """
    Build an additive attention mask for the hybrid transformer.

    Args:
        L:          Number of shape tokens (fixed; 1024 in full model).
        k_minus_1:  Number of skeleton tokens in the current prefix (T[:k-1]).
                    0 means no skeleton tokens (step k=1, first joint).

    Returns:
        FloatTensor of shape [L + k_minus_1, L + k_minus_1].
        Entries are 0.0 (attend) or -inf (masked out).
    """
    S = L + k_minus_1
    mask = torch.zeros(S, S)

    if k_minus_1 > 0:
        # Skeleton-to-skeleton block: upper triangle (diagonal=1) → -inf
        # Lower triangle (including diagonal) stays 0 (can attend).
        causal = torch.triu(
            torch.full((k_minus_1, k_minus_1), float('-inf')),
            diagonal=1,
        )
        mask[L:, L:] = causal

        # Shape-to-skeleton block: -inf so shape token representations are
        # independent of the skeleton prefix length.  This is required for
        # causal consistency: Z_k[L:L+j] must not change when more skeleton
        # tokens are appended.  Shape tokens still see ALL other shape tokens
        # (full bidirectional attention within the shape block).
        mask[:L, L:] = float('-inf')

    # Shape-to-shape block (rows :L, cols :L): all zeros → full bidirectional.
    # Skeleton-to-shape block (rows L:, cols :L): all zeros → full cross-attn.

    return mask
