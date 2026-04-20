"""
Phase 4 — Denoising MLP with AdaLN Conditioning
=================================================
Predicts the noise eps_theta(j^m | m, Z_k) used in the diffusion training
objective:
    L_joint = E_{eps, m} [ ||eps - eps_theta(j^m | m, Z_k)||^2 ]

Architecture:
  sinusoidal_time_embedding: fixed positional encoding for timestep m/M
  AdaLN:         Adaptive Layer Normalization conditioned on context
  DenoisingMLP:  2-layer MLP with AdaLN conditioning

Key design choices:
  - AdaLN uses (1 + gamma) * LN(x) + beta  (DiT-style, not plain gamma)
  - AdaLN projection is zero-initialised → identity at start of training
  - time_embed refines the fixed sinusoidal into a learned representation
  - condition = time_embed(t_sin) + context_proj(Z_k.mean)  — shared across both AdaLN
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal time embedding  (same convention as Phase 2 skeleton tokenizer)
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(m_normalized: float | torch.Tensor, d: int = 1024) -> torch.Tensor:
    """
    Fixed sinusoidal embedding for a scalar timestep value.

    Args:
        m_normalized: float or scalar tensor — timestep normalised to [0, 1]
                      (pass m / M where M is total steps)
        d:            embedding dimension (must be even)

    Returns:
        emb: [d]  float32 tensor
    """
    assert d % 2 == 0, f"d must be even, got {d}"

    if isinstance(m_normalized, torch.Tensor):
        val = m_normalized.float().item()
    else:
        val = float(m_normalized)

    half = d // 2
    # i = 0, 1, ..., half-1
    exponents = torch.arange(half, dtype=torch.float32) * (2.0 / d)
    freqs = 1.0 / (10000.0 ** exponents)    # [half]

    angles = val * freqs                     # [half]
    emb = torch.cat([angles.sin(), angles.cos()], dim=0)  # [d]
    return emb


# ---------------------------------------------------------------------------
# Adaptive Layer Normalization
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization: (1 + gamma) * LN(x) + beta

    The affine projection is zero-initialised so the module starts as a
    plain (non-adaptive) LayerNorm — gamma ≈ 0, beta ≈ 0.

    Args:
        d: feature dimension
    """

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(d, 2 * d)
        # Zero-init: at start, gamma=0, beta=0 → output = LN(x)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         [d]  feature vector
            condition: [d]  conditioning vector (time + context)

        Returns: [d]
        """
        h = self.norm(x)                              # [d]
        gamma, beta = self.proj(condition).chunk(2, dim=-1)  # each [d]
        return (1.0 + gamma) * h + beta


# ---------------------------------------------------------------------------
# Denoising MLP
# ---------------------------------------------------------------------------

class DenoisingMLP(nn.Module):
    """
    Predicts the noise eps given noisy joint j_m, timestep m, and context Z_k.

    Parameter count (d=1024):
      sinusoidal_time_embedding:  fixed (no params)
      time_embed:    Linear(d→d) + Linear(d→d)  = 2 * d*d + 2*d = 2,099,200
      context_proj:  Linear(d→d)                = d*d + d        = 1,049,600
      adaLN1.proj:   Linear(d→2d)               = d*2d + 2d      = 2,099,200
      adaLN2.proj:   Linear(d→2d)               = d*2d + 2d      = 2,099,200
      fc1:           Linear(3+d→d)              = (3+d)*d + d    = 1,050,624
      fc2:           Linear(d→d)                = d*d + d        = 1,049,600
      out:           Linear(d→3)                = d*3 + 3        = 3,075
      ─────────────────────────────────────────────────────────────────
      Total ≈ 10,450,499 params  (≈10M, within the 8–11M expected range)
    """

    def __init__(self, d: int = 1024, M: int = 1000):
        super().__init__()
        self.d = d
        self.M = M

        # Learned refinement of the fixed sinusoidal time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Project mean-pooled context to conditioning space
        self.context_proj = nn.Linear(d, d)

        # Adaptive Layer Norms
        self.adaLN1 = AdaLN(d)
        self.adaLN2 = AdaLN(d)

        # Main MLP layers
        self.fc1 = nn.Linear(3 + d, d)   # concat(j_m [3], time_emb [d])
        self.fc2 = nn.Linear(d, d)
        self.out  = nn.Linear(d, 3)

        # (out uses default init — only AdaLN projections are zero-init)

    def forward(
        self,
        j_m: torch.Tensor,
        m:   int,
        Z_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            j_m: [3]          noisy joint position at step m
            m:   int          timestep index (0-indexed)
            Z_k: [S, d]       context vectors from Phase 3 (S = L + k - 1)

        Returns:
            eps_pred: [3]   predicted noise
        """
        device = j_m.device

        # ── Time embedding ────────────────────────────────────────────
        t_sin = sinusoidal_time_embedding(m / self.M, self.d).to(device)  # [d]
        t_emb = self.time_embed(t_sin)                                      # [d]

        # ── Context ───────────────────────────────────────────────────
        z_ctx = self.context_proj(Z_k.mean(dim=0))   # [d]

        # Combined conditioning signal (shared by both AdaLN layers)
        condition = t_emb + z_ctx                    # [d]

        # ── Main forward ─────────────────────────────────────────────
        x = torch.cat([j_m, t_emb], dim=-1)          # [3 + d = 1027]
        x = self.fc1(x)                               # [d]
        x = torch.nn.functional.silu(self.adaLN1(x, condition))
        x = self.fc2(x)                               # [d]
        x = torch.nn.functional.silu(self.adaLN2(x, condition))
        return self.out(x)                            # [3]
