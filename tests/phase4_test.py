"""
Phase 4 Tests
==============
Structural tests for the Joint Diffusion Module — no training required.
All tests run on CPU with small tensors for speed.

Usage:
  pytest tests/phase4_test.py -v
  .venv/bin/python tests/phase4_test.py
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# ── Make project root importable ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase4.noise_schedule import compute_cosine_schedule, forward_diffuse, ddim_sample
from phase4.model          import AdaLN, DenoisingMLP, sinusoidal_time_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_denoiser(d: int = 1024, M: int = 1000) -> DenoisingMLP:
    return DenoisingMLP(d=d, M=M)


# ---------------------------------------------------------------------------
# NOISE SCHEDULE TESTS
# ---------------------------------------------------------------------------

def test_cosine_schedule_endpoints():
    sched = compute_cosine_schedule(M=1000)
    ab = sched['alpha_bars']

    # alpha_bars[0] should be close to 1 (very little noise at step 0)
    assert ab[0].item() > 0.999, \
        f'alpha_bars[0] should be ≈1.0, got {ab[0].item():.6f}'

    # alpha_bars[999] should be near 0 (essentially pure noise at last step)
    assert ab[999].item() < 0.01, \
        f'alpha_bars[999] should be < 0.01, got {ab[999].item():.6f}'


def test_cosine_schedule_strictly_decreasing():
    sched = compute_cosine_schedule(M=1000)
    ab = sched['alpha_bars']
    diffs = ab[1:] - ab[:-1]
    assert (diffs < 0).all(), \
        'alpha_bars must be strictly decreasing'


def test_forward_diffuse_at_m0():
    """At m=0 alpha_bar ≈ 1: j_m should be virtually identical to j0."""
    sched = compute_cosine_schedule(M=1000)
    torch.manual_seed(0)
    j0 = torch.randn(3)
    j_m, eps = forward_diffuse(j0, m=0, schedule=sched)

    # sqrt_alpha_bars[0] ≈ 1, sqrt_one_minus_alpha_bars[0] ≈ 0
    # So j_m ≈ j0  (within sqrt(1-alpha_bar_0) * noise tolerance)
    atol = float(sched['sqrt_one_minus_alpha_bars'][0].item()) * 3 + 1e-3
    assert torch.allclose(j_m, j0, atol=atol), \
        f'At m=0 j_m should be close to j0 (atol={atol:.4f})'


def test_forward_diffuse_at_m999():
    """At m=999 the signal is nearly pure noise; std should be ≈ 1."""
    sched = compute_cosine_schedule(M=1000)
    torch.manual_seed(42)
    j0 = torch.zeros(3)  # zero mean; result is driven entirely by noise

    # Average over many samples to check std
    samples = []
    for _ in range(2000):
        j_m, _ = forward_diffuse(j0, m=999, schedule=sched)
        samples.append(j_m)

    stacked = torch.stack(samples)  # [2000, 3]
    std = stacked.std().item()
    assert 0.8 < std < 1.2, \
        f'At m=999 j_m.std() should be ≈1.0 (pure noise), got {std:.4f}'


# ---------------------------------------------------------------------------
# MODEL TESTS
# ---------------------------------------------------------------------------

def test_denoising_mlp_output_shape():
    denoiser = make_denoiser()
    j_m = torch.randn(3)
    Z_k = torch.randn(1028, 1024)   # L + k - 1 = 1024 + 4 = 1028
    eps_pred = denoiser(j_m, 500, Z_k)
    assert eps_pred.shape == (3,), \
        f'Expected shape (3,), got {eps_pred.shape}'


def test_denoising_mlp_no_nan():
    torch.manual_seed(7)
    denoiser = make_denoiser()
    j_m = torch.randn(3)
    Z_k = torch.randn(1028, 1024)
    eps_pred = denoiser(j_m, 500, Z_k)
    assert not torch.isnan(eps_pred).any(), 'NaN in eps_pred'
    assert not torch.isinf(eps_pred).any(), 'Inf in eps_pred'


def test_adaln_identity_init():
    """
    Freshly initialised AdaLN with zero-init Linear:
    output should equal LayerNorm(x) (gamma≈0 → (1+0)*LN(x)+0 = LN(x)).
    """
    d = 64
    adaln = AdaLN(d)

    torch.manual_seed(0)
    x         = torch.randn(d)
    condition = torch.randn(d)

    out = adaln(x, condition)
    ln  = nn.LayerNorm(d, elementwise_affine=False)(x)

    assert torch.allclose(out, ln, atol=1e-5), \
        'AdaLN with zero-init should behave like plain LayerNorm at init'


def test_ddim_sample_shape():
    """DDIM sampler must return a [3] tensor without raising."""
    denoiser = make_denoiser(d=1024, M=1000)
    denoiser.eval()
    schedule = compute_cosine_schedule(M=1000)
    Z_k = torch.randn(1025, 1024)   # k=1 → L + 0 = 1024, but let's use L+1

    result = ddim_sample(denoiser, Z_k, schedule, ddim_steps=10, device='cpu')
    assert result.shape == (3,), \
        f'Expected ddim_sample output shape (3,), got {result.shape}'


def test_denoising_mlp_parameter_count():
    """
    Parameter count should be in the 8–11 million range for d=1024.
    Breakdown:
      time_embed: Linear(1024→1024) + Linear(1024→1024)  = 2*(1024*1024 + 1024) = 2,099,200
      context_proj: Linear(1024→1024)                    = 1024*1024 + 1024     = 1,049,600
      adaLN1.proj: Linear(1024→2048)                     = 1024*2048 + 2048     = 2,099,200
      adaLN2.proj: Linear(1024→2048)                     = 1024*2048 + 2048     = 2,099,200
      fc1: Linear(1027→1024)                             = 1027*1024 + 1024     = 1,052,672
      fc2: Linear(1024→1024)                             = 1024*1024 + 1024     = 1,049,600
      out: Linear(1024→3)                                = 1024*3 + 3           = 3,075
      ─────────────────────────────────────────────────────────────────
      Total = 10,452,547
    """
    denoiser = make_denoiser(d=1024, M=1000)
    total = sum(p.numel() for p in denoiser.parameters())
    assert 8_000_000 <= total <= 12_000_000, \
        f'Expected ≈8–12M params for d=1024, got {total:,}'


def test_condition_varies_with_m():
    """Same Z_k and j_m but different m → different eps_pred."""
    torch.manual_seed(1)
    denoiser = make_denoiser()
    j_m = torch.randn(3)
    Z_k = torch.randn(1024, 1024)

    eps_100 = denoiser(j_m, 100, Z_k)
    eps_900 = denoiser(j_m, 900, Z_k)

    assert not torch.allclose(eps_100, eps_900, atol=1e-4), \
        'eps_pred must differ for m=100 vs m=900 (AdaLN conditioning must act)'


def test_phase3_loads_and_freezes():
    """
    Load Phase 3 checkpoint, freeze, and verify:
      - all parameters have requires_grad=False
      - transformer is in eval mode
    """
    phase3_ckpt = os.path.join(ROOT, 'checkpoints', 'phase3', 'best_model.pt')
    if not os.path.exists(phase3_ckpt):
        pytest.skip(f'Phase 3 checkpoint not found at {phase3_ckpt}')

    from phase3.config      import Config as Phase3Config
    from phase3.transformer import HybridTransformer

    p3_cfg = Phase3Config()
    transformer = HybridTransformer(p3_cfg)

    ckpt = torch.load(phase3_ckpt, map_location='cpu', weights_only=False)
    transformer.load_state_dict(ckpt['model_state_dict'])

    transformer.eval()
    transformer.requires_grad_(False)

    # All params must be frozen
    for name, param in transformer.named_parameters():
        assert not param.requires_grad, \
            f'Parameter {name} should be frozen (requires_grad=False)'

    # Must be in eval mode
    assert not transformer.training, \
        'Transformer must be in eval mode after freezing'


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import traceback

    tests = [
        test_cosine_schedule_endpoints,
        test_cosine_schedule_strictly_decreasing,
        test_forward_diffuse_at_m0,
        test_forward_diffuse_at_m999,
        test_denoising_mlp_output_shape,
        test_denoising_mlp_no_nan,
        test_adaln_identity_init,
        test_ddim_sample_shape,
        test_denoising_mlp_parameter_count,
        test_condition_varies_with_m,
        test_phase3_loads_and_freezes,
    ]

    passed = 0
    failed = 0
    skipped = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except pytest.skip.Exception as e:
            print(f'  SKIP  {name}  ({e})')
            skipped += 1
        except Exception as e:
            print(f'  FAIL  {name}')
            traceback.print_exc()
            failed += 1

    total = passed + failed + skipped
    print(f'\n{total} tests: {passed} passed, {failed} failed, {skipped} skipped')
    sys.exit(0 if failed == 0 else 1)
