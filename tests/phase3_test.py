"""
Phase 3 Tests
==============
Covers hybrid mask, transformer forward pass, training dynamics,
checkpoint round-trips, dataset collation, and gradient checkpointing.

Usage:
  pytest tests/phase3_test.py -v
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

# ── Make project root importable ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase3.hybrid_mask  import build_hybrid_mask
from phase3.transformer  import HybridTransformer
from phase3.config       import Config
from phase3.dataset      import Phase3Dataset, phase3_collate


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def TinyConfig(**kwargs) -> Config:
    """Config with tiny dimensions so tests run fast on CPU."""
    defaults = dict(d=64, n_heads=4, d_k=16, ffn_dim=256,
                    n_layers=2, L=16, use_grad_checkpoint=False)
    defaults.update(kwargs)   # caller overrides win
    return Config(**defaults)


@pytest.fixture
def tiny_config() -> Config:
    return TinyConfig()


# ---------------------------------------------------------------------------
# MASK TESTS
# ---------------------------------------------------------------------------

def test_mask_shape():
    mask = build_hybrid_mask(L=16, k_minus_1=4)
    assert mask.shape == (20, 20)


def test_mask_shape_tokens_fully_visible():
    # Shape tokens use full bidirectional attention AMONG THEMSELVES.
    # They do NOT cross-attend to skeleton tokens (that would break causal
    # consistency: shape representations must be independent of prefix length).
    mask = build_hybrid_mask(L=16, k_minus_1=4)
    assert (mask[:16, :16] == 0).all(), \
        "shape tokens must see all other shape tokens (full bidirectional within shape block)"


def test_mask_skeleton_sees_all_shape():
    mask = build_hybrid_mask(L=16, k_minus_1=4)
    assert (mask[16:, :16] == 0).all(), \
        "skeleton tokens must see all shape tokens"


def test_mask_skeleton_causal():
    mask       = build_hybrid_mask(L=16, k_minus_1=4)
    skel_block = mask[16:, 16:]
    # Lower triangle (including diagonal) must be 0
    assert (torch.tril(skel_block) == 0).all()
    # Strictly upper-triangular elements must be -inf.
    # (torch.triu zeroes out the lower part, so we index directly.)
    upper_idx = torch.triu(torch.ones_like(skel_block, dtype=torch.bool), diagonal=1)
    assert (skel_block[upper_idx] == float('-inf')).all()


def test_mask_empty_skeleton():
    mask = build_hybrid_mask(L=16, k_minus_1=0)
    assert mask.shape == (16, 16)
    assert (mask == 0).all(), "all shape tokens, no masking needed"


def test_mask_single_skeleton():
    mask = build_hybrid_mask(L=16, k_minus_1=1)
    assert mask.shape == (17, 17)
    assert (mask[:16, :16] == 0).all()  # shape tokens see all shape tokens
    assert mask[16, 16] == 0            # T1 sees itself
    assert (mask[16, :16] == 0).all()   # T1 sees all shape tokens


# ---------------------------------------------------------------------------
# TRANSFORMER FORWARD TESTS
# ---------------------------------------------------------------------------

def test_transformer_output_shape(tiny_config):
    model = HybridTransformer(tiny_config)
    H = torch.randn(16, 64)
    T = torch.randn(3, 64)
    Z = model(H, T)
    assert Z.shape == (19, 64)


def test_transformer_empty_skeleton(tiny_config):
    model = HybridTransformer(tiny_config)
    H = torch.randn(16, 64)
    T = torch.randn(0, 64)
    Z = model(H, T)
    assert Z.shape == (16, 64)


def test_no_nan_in_output(tiny_config):
    model = HybridTransformer(tiny_config)
    H = torch.randn(16, 64)
    T = torch.randn(5, 64)
    Z = model(H, T)
    assert not torch.isnan(Z).any(),  "NaN in transformer output"
    assert not torch.isinf(Z).any(),  "Inf in transformer output"


def test_gradients_flow(tiny_config):
    model = HybridTransformer(tiny_config)
    H = torch.randn(16, 64)
    T = torch.randn(3, 64)
    Z = model(H, T)
    Z.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None,              f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(),   f"NaN gradient in {name}"


def test_causal_consistency(tiny_config):
    """Adding future tokens must not change past representations."""
    model = HybridTransformer(tiny_config)
    model.eval()
    H = torch.randn(16, 64)
    T = torch.randn(5, 64)
    with torch.no_grad():
        Z_3 = model(H, T[:3])   # [16+3, 64]
        Z_5 = model(H, T[:5])   # [16+5, 64]
    # Skeleton positions 0..2 must be identical in both outputs
    assert torch.allclose(Z_3[16:19], Z_5[16:19], atol=1e-5), \
        "Causal mask broken: future tokens leaked into past representations"


def test_shape_tokens_differ_from_input(tiny_config):
    """The transformer must actually transform tokens, not pass them through."""
    model = HybridTransformer(tiny_config)
    H = torch.randn(16, 64)
    T = torch.randn(3, 64)
    Z = model(H, T)
    assert not torch.allclose(Z[:16], H, atol=1e-3), \
        "Shape tokens unchanged — transformer doing nothing"


# ---------------------------------------------------------------------------
# PARAMETER COUNT TEST (full-size model)
# ---------------------------------------------------------------------------

def test_param_count():
    cfg   = Config()   # full-size defaults: d=1024, n_heads=16, …, n_layers=12
    model = HybridTransformer(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert total == 151_154_688, \
        f"Expected 151,154,688 params, got {total}"


# ---------------------------------------------------------------------------
# TRAINING TESTS
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """Train on 2 fake shapes for 5 steps; loss must decrease."""
    cfg   = TinyConfig()
    model = HybridTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    L = cfg.L   # 16

    H = torch.randn(L, 64)
    T = torch.randn(6, 64)
    losses = []

    for _ in range(5):
        optimizer.zero_grad()
        total_loss = torch.zeros(1)
        for k in range(2, 7):          # k=2..6  (skip k=1, no proxy loss)
            T_prev = T[:k - 1]
            Z_k    = model(H, T_prev)
            Z_skel = Z_k[L:]
            loss   = nn.functional.mse_loss(Z_skel, T_prev.detach())
            total_loss = total_loss + loss / 6
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_no_nan_during_training():
    """10 training steps must produce no NaN loss."""
    cfg   = TinyConfig()
    model = HybridTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    L = cfg.L   # 16

    H = torch.randn(L, 64)
    T = torch.randn(4, 64)

    for step in range(10):
        optimizer.zero_grad()
        total_loss = torch.zeros(1)
        for k in range(2, 5):          # k=2..4
            T_prev = T[:k - 1]
            Z_k    = model(H, T_prev)
            Z_skel = Z_k[L:]
            loss   = nn.functional.mse_loss(Z_skel, T_prev.detach())
            total_loss = total_loss + loss / 4
        total_loss.backward()
        optimizer.step()
        assert not torch.isnan(total_loss), f"NaN loss at step {step}"


def test_checkpoint_save_load():
    """Save model, load into a fresh instance, check outputs are identical."""
    cfg   = TinyConfig()
    model = HybridTransformer(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_ckpt.pt')
        torch.save({'model_state_dict': model.state_dict()}, path)

        model2 = HybridTransformer(cfg)
        model2.load_state_dict(
            torch.load(path, weights_only=True)['model_state_dict']
        )

        H = torch.randn(16, 64)
        T = torch.randn(3, 64)
        with torch.no_grad():
            Z1 = model(H, T)
            Z2 = model2(H, T)

        assert torch.allclose(Z1, Z2, atol=1e-6), \
            "Checkpoint save/load produced different outputs"


def test_grad_checkpoint_same_output():
    """Gradient checkpointing must produce identical forward-pass output."""
    cfg_no_ckpt   = TinyConfig(use_grad_checkpoint=False)
    cfg_with_ckpt = TinyConfig(use_grad_checkpoint=True)

    model1 = HybridTransformer(cfg_no_ckpt)
    model2 = HybridTransformer(cfg_with_ckpt)
    model2.load_state_dict(model1.state_dict())   # same weights

    H = torch.randn(16, 64)
    T = torch.randn(3, 64)
    model1.eval(); model2.eval()

    with torch.no_grad():
        Z1 = model1(H, T)
        Z2 = model2(H, T)

    assert torch.allclose(Z1, Z2, atol=1e-5), \
        "Grad checkpointing changed forward pass output"


# ---------------------------------------------------------------------------
# DATASET TESTS
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_tokens_dir(tmp_path):
    return str(tmp_path)


def test_dataset_loads(tmp_tokens_dir):
    """Dataset discovers all *_H.pt files in the directory."""
    for i in range(3):
        torch.save(torch.randn(1024, 1024),
                   os.path.join(tmp_tokens_dir, f'shape_{i:03d}_H.pt'))
        torch.save(torch.randn(12, 1024),
                   os.path.join(tmp_tokens_dir, f'shape_{i:03d}_T.pt'))

    ds = Phase3Dataset(tmp_tokens_dir)
    assert len(ds) == 3


def test_collate_pads_correctly():
    """Collate must pad T to max K with zeros and report correct lengths."""
    batch = [
        {'H': torch.randn(1024, 1024), 'T': torch.randn(5, 1024), 'K': 5, 'shape_id': 'a'},
        {'H': torch.randn(1024, 1024), 'T': torch.randn(8, 1024), 'K': 8, 'shape_id': 'b'},
    ]
    out = phase3_collate(batch)

    assert out['T'].shape == (2, 8, 1024)
    assert out['lengths'].tolist() == [5, 8]
    assert (out['T'][0, 5:, :] == 0).all(), "Padding must be zeros"
