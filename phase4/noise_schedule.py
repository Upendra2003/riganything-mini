"""
Phase 4 — Cosine Noise Schedule, Forward Diffusion, and DDIM Sampler
======================================================================
Implements the forward diffusion process and DDIM inference for the
Joint Diffusion Module.

Cosine schedule (Nichol & Dhariwal 2021):
  f(t) = cos( ((t/M + s) / (1 + s)) * pi/2 )^2
  alpha_bar_t = f(t) / f(0)
  beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
  beta_t clamped to [0, 0.999]
"""

import math
import torch


def compute_cosine_schedule(M: int = 1000, s: float = 0.008) -> dict:
    """
    Compute the cosine noise schedule.

    Args:
        M: total diffusion timesteps
        s: small offset to prevent beta_0 being too small (default 0.008)

    Returns dict with tensors of length M:
        betas                    [M]
        alphas                   [M]
        alpha_bars               [M]
        sqrt_alpha_bars          [M]
        sqrt_one_minus_alpha_bars [M]
    """
    # f(t) = cos( ((t/M + s) / (1 + s)) * pi/2 )^2
    # Compute for t = 0..M (inclusive) to get M+1 alpha_bar values
    t_over_M = torch.arange(M + 1, dtype=torch.float64) / M
    f_t = torch.cos(((t_over_M + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2

    # alpha_bar_t = f(t) / f(0)
    alpha_bars_full = f_t / f_t[0]  # [M+1]

    # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1},  for t = 1..M
    betas = 1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1]  # [M]
    betas = betas.clamp(0.0, 0.999).float()

    alphas = 1.0 - betas                          # [M]
    alpha_bars = alpha_bars_full[1:].float()       # [M]  (index 0 = step 1, etc.)

    return {
        'betas':                     betas,
        'alphas':                    alphas,
        'alpha_bars':                alpha_bars,
        'sqrt_alpha_bars':           alpha_bars.sqrt(),
        'sqrt_one_minus_alpha_bars': (1.0 - alpha_bars).sqrt(),
    }


def forward_diffuse(
    j0: torch.Tensor,
    m: int,
    schedule: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample j^m given clean joint j^0 at timestep m.

    Args:
        j0:       [3]   clean joint position
        m:        int   timestep index (0-indexed, 0..M-1)
        schedule: dict  from compute_cosine_schedule()

    Returns:
        j_m: [3]  noisy joint at step m
        eps: [3]  the sampled noise
    """
    sqrt_ab  = schedule['sqrt_alpha_bars'][m].to(j0.device)
    sqrt_1ab = schedule['sqrt_one_minus_alpha_bars'][m].to(j0.device)

    eps  = torch.randn_like(j0)
    j_m  = sqrt_ab * j0 + sqrt_1ab * eps
    return j_m, eps


@torch.no_grad()
def ddim_sample(
    denoiser,
    Z_k: torch.Tensor,
    schedule: dict,
    ddim_steps: int = 50,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Deterministic DDIM denoising from j^M ~ N(0,I) to j^0.

    Args:
        denoiser:   DenoisingMLP — callable(j [3], m int, Z_k [S, d]) -> [3]
        Z_k:        [S, d]  context vectors from Phase 3
        schedule:   dict from compute_cosine_schedule()
        ddim_steps: number of DDIM denoising steps
        device:     torch device string

    Returns:
        j: [3]  predicted clean joint position
    """
    M = schedule['alpha_bars'].shape[0]

    # Linearly spaced timesteps from M-1 down to 0
    timesteps = torch.linspace(M - 1, 0, ddim_steps).long()

    j = torch.randn(3, device=device)

    for idx, m_val in enumerate(timesteps):
        m = int(m_val.item())
        eps_pred = denoiser(j, m, Z_k)

        alpha_bar_m = schedule['alpha_bars'][m].to(device)

        # Estimate clean signal
        j0_est = (j - alpha_bar_m.sqrt().rsqrt() * (1.0 - alpha_bar_m).sqrt() * eps_pred) \
                 / alpha_bar_m.sqrt()
        # Equivalent form: (j - sqrt(1 - ab) * eps) / sqrt(ab)
        j0_est = (j - (1.0 - alpha_bar_m).sqrt() * eps_pred) / alpha_bar_m.sqrt()
        j0_est = j0_est.clamp(-3.0, 3.0)   # numerical safety

        if m > 0:
            # Previous (smaller) timestep
            m_prev = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else 0
            alpha_bar_prev = schedule['alpha_bars'][m_prev].to(device)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        j = alpha_bar_prev.sqrt() * j0_est + (1.0 - alpha_bar_prev).sqrt() * eps_pred

    return j
