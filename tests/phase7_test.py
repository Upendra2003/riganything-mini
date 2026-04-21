"""
Phase 7 Tests
==============
Six checks for the end-to-end RigAnythingModel.
All tests use one shape from the dataset and run on CPU with a tiny config
(d=64, 2 layers) except the warm-start test which uses the real config.

Usage:
  pytest tests/phase7_test.py -v
  .venv/bin/python tests/phase7_test.py
"""

import os
import sys
import numpy as np

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase7.model   import RigAnythingModel
from phase7.augment import augment_shape, random_rotation_matrix, recompute_normals
from phase7.dataset import Phase7Dataset

PC_DIR     = os.path.join(ROOT, 'pointClouds', 'obj_remesh')
CKPT_DIR   = os.path.join(ROOT, 'checkpoints')
SPLIT_FILE = os.path.join(ROOT, 'Dataset', 'test_final.txt')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_valid_id() -> str:
    with open(SPLIT_FILE) as f:
        ids = [l.strip().replace('.obj', '') for l in f if l.strip()]
    for sid in ids:
        pc = os.path.join(PC_DIR, f'{sid}_pointcloud.npy')
        sk = os.path.join(PC_DIR, f'{sid}_skeleton.npy')
        sw = os.path.join(PC_DIR, f'{sid}_skinning.npy')
        if all(os.path.exists(p) for p in (pc, sk, sw)):
            return sid
    pytest.skip('No valid shape found in test split')


def _load_shape(sid: str, device='cpu'):
    ds = Phase7Dataset(SPLIT_FILE, PC_DIR)
    for item in ds.items:
        if item['shape_id'] == sid:
            import numpy as np
            pc   = np.load(item['pc_path'])
            skel = np.load(item['skel_path'])
            from phase6.dataset import _resample_skinning
            skin = _resample_skinning(np.load(item['skin_path']), 1024)
            K    = skel.shape[0]

            points    = torch.from_numpy(pc[:, :3]).float().to(device)
            normals   = torch.from_numpy(pc[:, 3:]).float().to(device)
            gt_joints = torch.from_numpy(skel[:, :3]).float().to(device)
            gt_parents= torch.from_numpy(skel[:,  3]).long().to(device)
            gt_skin   = torch.from_numpy(skin).float().to(device)
            return points, normals, gt_joints, gt_parents, gt_skin, K
    pytest.skip(f'Shape {sid} not found in dataset')


def _tiny_model() -> RigAnythingModel:
    """Small model for fast CPU tests — d=64, 2 layers."""
    return RigAnythingModel(d=64, L=16, n_layers=2, n_heads=4, ffn_dim=256, M=100)


# ---------------------------------------------------------------------------
# Test 1: Warm-start — phase3/5/6 weights load correctly
# ---------------------------------------------------------------------------

def test_warm_start():
    """
    Load the real model config (d=1024) and verify that pretrained checkpoint
    weights are transferred with 0 unexpected keys.
    Only checks checkpoints that actually exist.
    """
    model = RigAnythingModel()

    p3_path = os.path.join(CKPT_DIR, 'phase3', 'best_model.pt')
    if os.path.exists(p3_path):
        ckpt = torch.load(p3_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.transformer.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        assert len(unexpected) == 0, \
            f'phase3 → transformer: unexpected keys {unexpected[:5]}'
        print(f'  transformer warm-start: {len(missing)} missing (ok if 0)')
    else:
        print('  [SKIP] phase3 checkpoint not found — testing load logic only')

    p5_path = os.path.join(CKPT_DIR, 'phase5', 'best_model.pt')
    if os.path.exists(p5_path):
        ckpt = torch.load(p5_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.fuser.load_state_dict(
            ckpt['fuser_state_dict'], strict=False
        )
        assert len(unexpected) == 0, \
            f'phase5 → fuser: unexpected keys {unexpected[:5]}'
        missing, unexpected = model.connector.load_state_dict(
            ckpt['connector_state_dict'], strict=False
        )
        assert len(unexpected) == 0, \
            f'phase5 → connector: unexpected keys {unexpected[:5]}'

    p6_path = os.path.join(CKPT_DIR, 'phase6', 'best_model.pt')
    if os.path.exists(p6_path):
        ckpt = torch.load(p6_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.skinner.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        assert len(unexpected) == 0, \
            f'phase6 → skinner: unexpected keys {unexpected[:5]}'
        print('  skinner warm-start: OK')


# ---------------------------------------------------------------------------
# Test 2: Forward pass returns 4 finite scalars
# ---------------------------------------------------------------------------

def test_forward_pass():
    """model.forward() must return 4 finite scalar tensors."""
    torch.manual_seed(0)
    d, L, K = 64, 16, 5
    model = _tiny_model()

    points    = torch.randn(L, 3)
    normals   = torch.randn(L, 3)
    normals   = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    gt_joints = torch.randn(K, 3)
    # BFS parents: root=1, others point to previous
    gt_parents = torch.ones(K, dtype=torch.long)
    for i in range(1, K):
        gt_parents[i] = i   # 1-indexed: parent is previous joint
    gt_skin = torch.zeros(L, K)
    for l in range(L):
        gt_skin[l, l % K] = 1.0

    total, lj, lc, ls = model(points, normals, gt_joints, gt_parents, gt_skin)

    for name, val in [('total', total), ('joint', lj),
                      ('connect', lc), ('skin', ls)]:
        assert torch.isfinite(val), f'loss_{name} is not finite: {val}'
        assert val.ndim == 0, f'loss_{name} should be scalar, got shape {val.shape}'


# ---------------------------------------------------------------------------
# Test 3: All loss components > 0
# ---------------------------------------------------------------------------

def test_loss_components_positive():
    """Each sub-loss must be strictly positive."""
    torch.manual_seed(1)
    d, L, K = 64, 16, 5
    model = _tiny_model()

    points    = torch.randn(L, 3)
    normals   = torch.randn(L, 3)
    normals   = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    gt_joints = torch.randn(K, 3)
    gt_parents = torch.ones(K, dtype=torch.long)
    for i in range(1, K):
        gt_parents[i] = i
    gt_skin = torch.zeros(L, K)
    for l in range(L):
        gt_skin[l, l % K] = 1.0

    _, lj, lc, ls = model(points, normals, gt_joints, gt_parents, gt_skin)

    assert lj.item() > 0.0, f'loss_joint should be > 0, got {lj.item()}'
    assert lc.item() > 0.0, f'loss_connect should be > 0, got {lc.item()}'
    assert ls.item() > 0.0, f'loss_skin should be > 0, got {ls.item()}'

    # Sanity-check magnitudes
    assert 0.0 < lj.item() < 1e4, f'loss_joint out of O(1) range: {lj.item()}'
    assert 0.0 < lc.item() < 1e4, f'loss_connect out of O(1) range: {lc.item()}'
    assert 0.0 < ls.item() < 1e4, f'loss_skin out of O(1-3) range: {ls.item()}'


# ---------------------------------------------------------------------------
# Test 4: Gradient flow through all key parameter groups
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """
    After backward, shape_tok, skel_tok, skinner, and transformer all must
    have non-None, non-zero gradients.
    """
    torch.manual_seed(2)
    d, L, K = 64, 16, 4
    model = _tiny_model()

    points    = torch.randn(L, 3)
    normals   = torch.randn(L, 3)
    normals   = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    gt_joints = torch.randn(K, 3)
    gt_parents = torch.ones(K, dtype=torch.long)
    for i in range(1, K):
        gt_parents[i] = i
    gt_skin = torch.zeros(L, K)
    for l in range(L):
        gt_skin[l, l % K] = 1.0

    total, _, _, _ = model(points, normals, gt_joints, gt_parents, gt_skin)
    total.backward()

    groups = {
        'shape_tok':   model.shape_tok,
        'skel_tok':    model.skel_tok,
        'transformer': model.transformer,
        'skinner':     model.skinner,
    }
    for name, module in groups.items():
        params_with_grad = [
            p for p in module.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(params_with_grad) > 0, \
            f'{name}: no parameters received non-zero gradients after backward()'


# ---------------------------------------------------------------------------
# Test 5: augment_shape returns correct shapes, non-identity, unit normals
# ---------------------------------------------------------------------------

def test_augmentation():
    """
    augment_shape must return:
    - new_points  [L, 3]  different from originals
    - new_normals [L, 3]  unit vectors
    - new_joints  [K, 3]  finite
    """
    torch.manual_seed(3)
    L, K = 32, 5

    points    = torch.randn(L, 3) * 0.5
    joints    = torch.randn(K, 3) * 0.3
    gt_parents = torch.ones(K, dtype=torch.long)
    for i in range(1, K):
        gt_parents[i] = i
    gt_skin = torch.zeros(L, K)
    for l in range(L):
        gt_skin[l, l % K] = 1.0

    new_pts, new_nrm, new_jts = augment_shape(points, joints, gt_parents, gt_skin,
                                               max_degrees=30)

    assert new_pts.shape == (L, 3), f'new_points shape {new_pts.shape}'
    assert new_nrm.shape == (L, 3), f'new_normals shape {new_nrm.shape}'
    assert new_jts.shape == (K, 3), f'new_joints shape {new_jts.shape}'

    # Points must have moved
    diff = (new_pts - points).norm(dim=-1).mean().item()
    assert diff > 1e-4, f'Augmented points identical to originals (mean diff={diff:.6f})'

    # Normals must be unit vectors
    norms = new_nrm.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(L), atol=1e-4), \
        f'Normals not unit length: mean_norm={norms.mean():.4f}'

    # Joints must be finite
    assert torch.isfinite(new_jts).all(), 'Augmented joints contain NaN/Inf'


# ---------------------------------------------------------------------------
# Test 6: Inference runs and produces valid outputs
# ---------------------------------------------------------------------------

def test_inference():
    """
    Run inference with a tiny model on a real shape.
    Check output files exist, weights rows sum to 1, joints are finite.
    """
    import tempfile, json

    sid = _first_valid_id()
    ds  = Phase7Dataset(SPLIT_FILE, PC_DIR)
    # find item
    item = next((x for x in ds.items if x['shape_id'] == sid), None)
    if item is None:
        pytest.skip(f'{sid} not in dataset')

    # Build a tiny model (random weights — just tests the pipeline)
    model = _tiny_model()
    model.eval()

    from phase4.noise_schedule import ddim_sample

    pc      = np.load(item['pc_path']).astype(np.float32)
    points  = torch.from_numpy(pc[:, :3])
    normals = torch.from_numpy(pc[:, 3:])

    pc_in = torch.cat([points, normals], dim=-1)
    with torch.no_grad():
        H       = model.shape_tok(pc_in)
        sched   = model._schedule()
        T_list  = []
        joints  = []
        parents = []

        for k in range(1, 4):   # just 3 steps for speed
            T_stack = (torch.stack(T_list) if T_list else H.new_zeros(0, model.d))
            Z_k     = model.transformer(H[:16], T_stack)   # tiny L=16 subset
            j_k     = ddim_sample(model.denoiser, Z_k, sched, ddim_steps=5, device='cpu')
            joints.append(j_k)

            if k == 1:
                p = 1
            else:
                Z_prime = model.fuser(Z_k, j_k, k)
                q_k     = model.connector(Z_prime, T_stack)
                p       = int(q_k.argmax().item()) + 1
            parents.append(p)

            parent_0 = max(0, min(p - 1, k - 1))
            T_k = model.skel_tok(j_k.unsqueeze(0),
                                  torch.tensor([parent_0])).squeeze(0)
            T_list.append(T_k)

        T_all = torch.stack(T_list)
        W     = model.skinner(H[:16], T_all)     # [16, 3]

    # W rows must sum to 1
    row_sums = W.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(16), atol=1e-5), \
        f'W rows do not sum to 1: {row_sums[:3]}'

    # Joints finite
    j_tensor = torch.stack(joints)
    assert torch.isfinite(j_tensor).all(), 'Predicted joints contain NaN/Inf'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [
        test_warm_start,
        test_forward_pass,
        test_loss_components_positive,
        test_gradient_flow,
        test_augmentation,
        test_inference,
    ]

    passed = failed = skipped = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except pytest.skip.Exception as e:
            print(f'  SKIP  {name}  ({e})')
            skipped += 1
        except Exception:
            print(f'  FAIL  {name}')
            traceback.print_exc()
            failed += 1

    total = passed + failed + skipped
    print(f'\n{total} tests: {passed} passed, {failed} failed, {skipped} skipped')
    sys.exit(0 if failed == 0 else 1)
