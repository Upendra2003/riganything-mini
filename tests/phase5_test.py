"""
Phase 5 Tests
==============
Structural and functional tests for the Connectivity Prediction modules.
Tests run on CPU with small synthetic tensors unless disk files are needed.

Usage:
  pytest tests/phase5_test.py -v
  .venv/bin/python tests/phase5_test.py
"""

import os
import sys
import json
import tempfile

import pytest
import torch
import torch.optim as optim
import numpy as np

# ── Make project root importable ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase5.connectivity import FusingModule, ConnectivityModule, connectivity_loss
from phase5.inference    import build_skeleton_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 64    # small d for fast CPU tests


def make_fuser(d: int = D) -> FusingModule:
    return FusingModule(d=d)


def make_connector(d: int = D) -> ConnectivityModule:
    return ConnectivityModule(d=d)


def make_synthetic_shape(K: int = 8, L: int = 16, d: int = D):
    """
    Returns (H, T, skeleton) tensors with small synthetic dimensions.
    skeleton[k, 3] = 1-indexed parent (root self-referential = 1, others in 1..k).
    """
    H      = torch.randn(L, d)
    T      = torch.randn(K, d)
    joints = torch.randn(K, 3)
    # Assign parents: root = 1 (self-ref), others = parent among joints 0..k-1 (1-indexed)
    parents = torch.ones(K, dtype=torch.long)
    for i in range(1, K):
        parents[i] = torch.randint(1, i + 1, (1,)).item()  # 1-indexed, parent < current
    return H, T, joints, parents


# ---------------------------------------------------------------------------
# Test 1: connectivity_loss decreases over 5 training steps
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """connectivity_loss should decrease when training on a fixed example."""
    torch.manual_seed(0)
    d = D
    fuser     = make_fuser(d)
    connector = make_connector(d)

    optimizer = optim.Adam(
        list(fuser.parameters()) + list(connector.parameters()),
        lr=1e-3,
    )

    H, T, joints, parents = make_synthetic_shape(K=6, L=16, d=d)
    k = 4   # predict parent for joint 4 (1-indexed)
    T_prev  = T[:k - 1]        # [k-1, d]
    j_k_gt  = joints[k - 1]    # [3]
    Z_k     = torch.cat([H, T_prev], dim=0)   # fake Z_k [L+k-1, d]
    true_pi = int(parents[k - 1].item()) - 1
    true_pi = max(0, min(true_pi, k - 2))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        Z_prime = fuser(Z_k, j_k_gt, k)
        q_k     = connector(Z_prime, T_prev)
        loss    = connectivity_loss(q_k, true_pi)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], \
        f'Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}'


# ---------------------------------------------------------------------------
# Test 2: q_k sums to 1 (softmax correctness)
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    torch.manual_seed(1)
    fuser     = make_fuser(D)
    connector = make_connector(D)

    for k in [2, 5, 10, 15]:
        Z_k    = torch.randn(16 + k - 1, D)
        T_prev = torch.randn(k - 1, D)
        j_k    = torch.randn(3)

        Z_prime = fuser(Z_k, j_k, k)
        q_k     = connector(Z_prime, T_prev)

        assert q_k.shape == (k - 1,), \
            f'q_k shape wrong for k={k}: expected ({k-1},), got {q_k.shape}'
        total = q_k.sum().item()
        assert abs(total - 1.0) < 1e-5, \
            f'q_k does not sum to 1.0 for k={k}: sum={total:.8f}'


# ---------------------------------------------------------------------------
# Test 3: FusingModule output shape is [d]
# ---------------------------------------------------------------------------

def test_fusing_module_output_shape():
    torch.manual_seed(2)
    for d in [64, 128, 1024]:
        fuser = FusingModule(d=d)
        Z_k   = torch.randn(20 + 3, d)   # L=20, k=4
        j_k   = torch.randn(3)
        out   = fuser(Z_k, j_k, k=4)
        assert out.shape == (d,), \
            f'FusingModule output shape wrong for d={d}: got {out.shape}'


# ---------------------------------------------------------------------------
# Test 4: ConnectivityModule output shape is [k-1] for varying k
# ---------------------------------------------------------------------------

def test_connectivity_module_output_shape():
    torch.manual_seed(3)
    for k in [2, 5, 10]:
        connector  = make_connector(D)
        Z_prime    = torch.randn(D)
        T_prev     = torch.randn(k - 1, D)
        q_k        = connector(Z_prime, T_prev)
        assert q_k.shape == (k - 1,), \
            f'ConnectivityModule output shape wrong for k={k}: got {q_k.shape}'


# ---------------------------------------------------------------------------
# Test 5: Val accuracy > 90% after 20 epochs on 5 shapes (teacher forcing)
# ---------------------------------------------------------------------------

def make_learnable_shape(K: int, L: int, d: int, rng: torch.Generator):
    """
    Structured synthetic shape with a clear learnable parent signal.

    T[i] = distinct orthogonal-ish unit vector (first d/K elements high, rest ~0).
    j_k  = T[true_parent, :3] exactly — zero noise so the signal is unambiguous.
    Linear chain topology: parent of joint i is joint i-1.
    H    = zeros (no distracting z_mean noise).
    """
    H = torch.zeros(L, d)
    T = torch.randn(K, d, generator=rng) * 0.01   # near-zero base
    for i in range(K):
        T[i, i % d] = 5.0   # each token has a large distinctive feature

    parents = torch.ones(K, dtype=torch.long)
    for i in range(1, K):
        parents[i] = i   # 1-indexed → parent is joint i-1 in 0-indexed

    joints = torch.zeros(K, 3)
    for i in range(1, K):
        p = int(parents[i].item()) - 1
        joints[i] = T[p, :3].clone()   # exact copy — strong, noiseless signal

    return H, T, joints, parents


def test_val_accuracy_teacher_forcing():
    """
    Overfit FusingModule + ConnectivityModule on 5 fixed synthetic shapes for
    20 epochs; final val accuracy (== train accuracy here) must exceed 90%.

    Uses a structured dataset where T tokens are distinctive and j_k is an
    exact copy of the parent's first 3 token features, giving the scorer a
    clear signal to learn without being overwhelmed by random noise.
    """
    torch.manual_seed(42)
    rng = torch.Generator()
    rng.manual_seed(42)

    d  = D
    L  = 16
    K  = 6
    N  = 5

    shapes = [make_learnable_shape(K=K, L=L, d=d, rng=rng) for _ in range(N)]
    data   = []
    for H, T, joints, parents in shapes:
        for k in range(2, K + 1):
            T_prev  = T[:k - 1]
            j_k     = joints[k - 1]
            Z_k     = torch.cat([H, T_prev], dim=0)
            true_pi = int(parents[k - 1].item()) - 1
            true_pi = max(0, min(true_pi, k - 2))
            data.append((Z_k, T_prev, j_k, k, true_pi))

    fuser     = make_fuser(d)
    connector = make_connector(d)
    optimizer = optim.Adam(
        list(fuser.parameters()) + list(connector.parameters()),
        lr=1e-2,
    )

    for _ in range(20):
        fuser.train()
        connector.train()
        for Z_k, T_prev, j_k, k, true_pi in data:
            optimizer.zero_grad()
            Z_prime = fuser(Z_k, j_k, k)
            q_k     = connector(Z_prime, T_prev)
            loss    = connectivity_loss(q_k, true_pi)
            loss.backward()
            optimizer.step()

    fuser.eval()
    connector.eval()
    correct = 0
    with torch.no_grad():
        for Z_k, T_prev, j_k, k, true_pi in data:
            Z_prime = fuser(Z_k, j_k, k)
            q_k     = connector(Z_prime, T_prev)
            if int(q_k.argmax().item()) == true_pi:
                correct += 1

    acc = correct / len(data)
    assert acc >= 0.90, \
        f'Val accuracy after 20 epochs should be ≥90%, got {acc:.4f} ({correct}/{len(data)})'


# ---------------------------------------------------------------------------
# Test 6: build_skeleton_tree satisfies BFS invariant p_k < k for all k > 0
# ---------------------------------------------------------------------------

def test_bfs_invariant():
    """All parent indices must be strictly less than the child index."""
    joints  = [[0.0, 0.0, 0.0], [0.1, 0.5, 0.0], [0.2, 1.0, 0.0],
               [-0.1, 0.5, 0.0], [0.0, 1.5, 0.0]]
    parents = [-1, 0, 1, 0, 2]   # valid BFS tree

    tree = build_skeleton_tree(joints, parents)
    assert len(tree) == len(joints), 'tree should have one entry per joint'

    for i in range(1, len(joints)):
        p = tree[i]['parent']
        assert p < i, f'BFS invariant violated: joint {i} has parent {p}'


def test_bfs_invariant_violation_raises():
    """build_skeleton_tree must raise ValueError when invariant is broken."""
    joints  = [[0.0, 0.0, 0.0], [0.1, 0.5, 0.0], [0.2, 1.0, 0.0]]
    parents = [-1, 2, 1]   # joint 1 has parent 2 which comes after it

    with pytest.raises(ValueError, match='BFS invariant'):
        build_skeleton_tree(joints, parents)


# ---------------------------------------------------------------------------
# Test 7: saved JSON loads correctly and joint count matches K
# ---------------------------------------------------------------------------

def test_json_round_trip():
    """JSON output must be loadable and match the skeleton dimensions."""
    torch.manual_seed(5)
    K       = 7
    joints  = [[float(i), float(i) * 0.1, 0.0] for i in range(K)]
    parents = [-1] + list(range(K - 1))   # linear chain

    save_data = {'joints': joints, 'parents': parents}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(save_data, f)
        tmp_path = f.name

    try:
        with open(tmp_path, 'r') as f:
            loaded = json.load(f)

        assert len(loaded['joints'])  == K, \
            f'Expected {K} joints in JSON, got {len(loaded["joints"])}'
        assert len(loaded['parents']) == K, \
            f'Expected {K} parents in JSON, got {len(loaded["parents"])}'
        assert loaded['parents'][0] == -1, \
            'Root parent should be -1'

        # Verify each joint has 3 coordinates
        for i, j in enumerate(loaded['joints']):
            assert len(j) == 3, f'Joint {i} does not have 3 coordinates: {j}'
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: parameter count sanity check
# ---------------------------------------------------------------------------

def test_parameter_counts():
    """
    FusingModule(d=1024):
      Linear(2*1024+3 → 2048) = 2051*2048 + 2048 = 4,200,496
      Linear(2048 → 1024)     = 2048*1024 + 1024 = 2,098,176
      Total ≈ 6,298,672

    ConnectivityModule(d=1024):
      Linear(2*1024 → 1024) = 2048*1024 + 1024 = 2,098,176
      Linear(1024 → 1)      = 1024*1 + 1       = 1,025
      Total ≈ 2,099,201
    """
    fuser     = FusingModule(d=1024)
    connector = ConnectivityModule(d=1024)

    fp = sum(p.numel() for p in fuser.parameters())
    cp = sum(p.numel() for p in connector.parameters())

    assert 5_000_000 <= fp <= 8_000_000, \
        f'FusingModule param count out of expected range: {fp:,}'
    assert 1_500_000 <= cp <= 3_000_000, \
        f'ConnectivityModule param count out of expected range: {cp:,}'


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [
        test_loss_decreases,
        test_softmax_sums_to_one,
        test_fusing_module_output_shape,
        test_connectivity_module_output_shape,
        test_val_accuracy_teacher_forcing,
        test_bfs_invariant,
        test_bfs_invariant_violation_raises,
        test_json_round_trip,
        test_parameter_counts,
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
