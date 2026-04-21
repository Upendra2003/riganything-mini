"""
Phase 6 Tests
==============
Five checks for the Skinning Weight Prediction module.
Tests 1–3 and 5 use real shapes from the dataset.
Test 4 uses a tiny synthetic model for speed.

Usage:
  pytest tests/phase6_test.py -v
  .venv/bin/python tests/phase6_test.py
"""

import os
import sys

import pytest
import numpy as np
import torch
import torch.nn.functional as F

# ── Make project root importable ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase6.model   import SkinningModule, skinning_loss
from phase6.dataset import Phase6Dataset, _resample_skinning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOKEN_DIR = os.path.join(ROOT, 'tokens', 'obj_remesh')
SKEL_DIR  = os.path.join(ROOT, 'pointClouds', 'obj_remesh')
SPLIT     = os.path.join(ROOT, 'Dataset', 'test_final.txt')


def _first_valid_id() -> str:
    """Return the first shape ID from test_final.txt that has all required files."""
    with open(SPLIT) as f:
        ids = [l.strip().replace('.obj', '') for l in f if l.strip()]
    for sid in ids:
        h = os.path.join(TOKEN_DIR, f'{sid}_H.pt')
        t = os.path.join(TOKEN_DIR, f'{sid}_T.pt')
        s = os.path.join(SKEL_DIR,  f'{sid}_skinning.npy')
        if all(os.path.exists(p) for p in (h, t, s)):
            return sid
    pytest.skip('No valid shape found in test split')


def _load_shape(sid: str):
    H    = torch.load(os.path.join(TOKEN_DIR, f'{sid}_H.pt'), weights_only=True)
    T    = torch.load(os.path.join(TOKEN_DIR, f'{sid}_T.pt'), weights_only=True)
    skin = np.load(os.path.join(SKEL_DIR, f'{sid}_skinning.npy'))
    return H, T, skin


# ---------------------------------------------------------------------------
# Test 1: W.shape == (1024, K) for a real shape
# ---------------------------------------------------------------------------

def test_output_shape():
    sid = _first_valid_id()
    H, T, _ = _load_shape(sid)
    K   = T.shape[0]

    model = SkinningModule(d=1024)
    model.eval()
    with torch.no_grad():
        W = model(H, T)

    assert W.shape == (1024, K), \
        f'Expected W shape (1024, {K}), got {W.shape}'


# ---------------------------------------------------------------------------
# Test 2: rows sum to 1
# ---------------------------------------------------------------------------

def test_rows_sum_to_one():
    sid = _first_valid_id()
    H, T, _ = _load_shape(sid)

    model = SkinningModule(d=1024)
    model.eval()
    with torch.no_grad():
        W = model(H, T)

    row_sums = W.sum(dim=-1)   # [1024]
    assert torch.allclose(row_sums, torch.ones(1024), atol=1e-5), \
        f'Rows do not sum to 1.  max deviation: {(row_sums - 1).abs().max().item():.2e}'


# ---------------------------------------------------------------------------
# Test 3: all weights non-negative
# ---------------------------------------------------------------------------

def test_weights_non_negative():
    sid = _first_valid_id()
    H, T, _ = _load_shape(sid)

    model = SkinningModule(d=1024)
    model.eval()
    with torch.no_grad():
        W = model(H, T)

    assert W.min().item() >= -1e-7, \
        f'Negative weights found: min={W.min().item():.4e}'


# ---------------------------------------------------------------------------
# Test 4: loss decreases over 10 gradient steps (tiny synthetic model)
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """
    Loss must decrease when training on a fixed small example.
    Uses a peaked one-hot target so the initial loss is high and clearly
    improvable — avoids the degenerate case where uniform init is already
    at the minimum of the uniform target.
    """
    torch.manual_seed(0)
    d = 64
    L = 16
    K = 5

    model     = SkinningModule(d=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    H = torch.randn(L, d)
    T = torch.randn(K, d)
    # One-hot target: each point assigned entirely to joint (l % K).
    # Starting from near-uniform softmax, the initial loss ≈ log(K) ≈ 1.61
    # and must decrease as the model pushes towards the peaked target.
    W_gt = torch.zeros(L, K)
    for l in range(L):
        W_gt[l, l % K] = 1.0

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        W    = model(H, T)
        loss = skinning_loss(W, W_gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f'Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}'


# ---------------------------------------------------------------------------
# Test 5: two different shapes produce different W matrices
# ---------------------------------------------------------------------------

def test_different_shapes_different_W():
    """
    After a few gradient steps on different targets, two shapes must produce
    different W matrices.

    At random init the softmax is near-uniform for all inputs, so we must
    train briefly before checking — this tests that the model is input-sensitive
    (different H/T → different scores → different W), not just that init varies.
    """
    torch.manual_seed(7)

    with open(SPLIT) as f:
        ids = [l.strip().replace('.obj', '') for l in f if l.strip()]

    valid = []
    for sid in ids:
        h = os.path.join(TOKEN_DIR, f'{sid}_H.pt')
        t = os.path.join(TOKEN_DIR, f'{sid}_T.pt')
        s = os.path.join(SKEL_DIR,  f'{sid}_skinning.npy')
        if all(os.path.exists(p) for p in (h, t, s)):
            valid.append(sid)
        if len(valid) == 2:
            break

    if len(valid) < 2:
        pytest.skip('Need at least 2 valid shapes for this test')

    sid1, sid2 = valid
    H1, T1, skin1 = _load_shape(sid1)
    H2, T2, skin2 = _load_shape(sid2)

    K = min(T1.shape[0], T2.shape[0])
    T1, T2 = T1[:K], T2[:K]

    # Use a simple model with d=1024 but train a few steps so that the scorer
    # learns to differentiate inputs; each shape gets its own one-hot target.
    model     = SkinningModule(d=1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    W_gt1 = torch.zeros(1024, K)
    W_gt2 = torch.zeros(1024, K)
    for l in range(1024):
        W_gt1[l, l % K] = 1.0
        W_gt2[l, (l + 1) % K] = 1.0   # different target pattern for shape 2

    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        loss  = skinning_loss(model(H1, T1), W_gt1)
        loss += skinning_loss(model(H2, T2), W_gt2)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        W1 = model(H1, T1)
        W2 = model(H2, T2)

    # After training on different targets, argmax patterns must differ
    top1 = W1.argmax(dim=-1)   # [1024]
    top2 = W2.argmax(dim=-1)   # [1024]
    frac_same = (top1 == top2).float().mean().item()

    assert frac_same < 0.99, \
        f'Argmax patterns too similar after training on different targets ' \
        f'(frac_same={frac_same:.4f})'


# ---------------------------------------------------------------------------
# Bonus: _resample_skinning correctness
# ---------------------------------------------------------------------------

def test_resample_skinning_shapes():
    rng = np.random.default_rng(42)

    for V, K, L in [(2876, 14, 1024), (1014, 26, 1024), (512, 10, 1024)]:
        skin = rng.random((V, K)).astype(np.float32)
        skin /= skin.sum(axis=1, keepdims=True)   # normalise rows
        out  = _resample_skinning(skin, L)

        assert out.shape == (L, K), \
            f'resample_skinning({V},{K}) → {out.shape}, expected ({L},{K})'
        row_sums = out.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f'resample_skinning rows do not sum to 1 for V={V}'


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [
        test_output_shape,
        test_rows_sum_to_one,
        test_weights_non_negative,
        test_loss_decreases,
        test_different_shapes_different_W,
        test_resample_skinning_shapes,
    ]

    passed  = 0
    failed  = 0
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
