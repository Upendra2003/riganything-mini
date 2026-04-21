"""
Phase 7 — Full Pipeline Inference
====================================
Autoregressively generates skeleton + skinning weights for a single shape
using the full RigAnythingModel:

  H = shape_tok(points, normals)
  For k = 1 .. max_joints:
    Z_k = transformer(H, T_stack)
    j_k = ddim_sample(denoiser, Z_k, 50 steps)
    if k > 1:
      Z'  = fuser(Z_k, j_k, k)
      q_k = connector(Z', T_stack)
      p_k = argmax(q_k) + 1   (1-indexed parent)
      if duplicate detection triggers: stop
    T_k = skel_tok(j_k, joints[p_k-1], k, p_k)
  W = skinner(H, T_all)

Outputs:
  output/phase7/<id>_joints.npy    [K_pred, 3]
  output/phase7/<id>_parents.npy   [K_pred]   int (1-indexed, root=1)
  output/phase7/<id>_weights.npy   [1024, K_pred]

Usage:
  .venv/bin/python phase7/inference.py --shape_id <id>
  .venv/bin/python phase7/inference.py --shape_id 10000 --max_joints 64
  .venv/bin/python phase7/inference.py --glb inference_example_datamodels/spyro_the_dragon.glb
"""

import os
import sys
import argparse

import numpy as np
import torch
import trimesh
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase7.model          import RigAnythingModel
from phase4.noise_schedule import ddim_sample
from dataset               import sample_surface


# ---------------------------------------------------------------------------
# GLB → normalised pointcloud  [1024, 6]
# ---------------------------------------------------------------------------
def glb_to_pointcloud(glb_path: str, n_points: int = 1024) -> np.ndarray:
    """
    Load a .glb file, merge all sub-meshes, normalise to unit scale
    (same convention as the RigNet training data: centred, max-dim = 1),
    then sample n_points surface points + normals.
    Returns float32 array [n_points, 6].
    """
    scene = trimesh.load(glb_path, force='scene')
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f'No triangle meshes found in {glb_path}')
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene

    verts = np.array(combined.vertices, dtype=np.float64)
    faces = np.array(combined.faces,    dtype=np.int64)

    # Normalise: centre + scale so longest bounding-box axis = 1
    bbox_min, bbox_max = verts.min(0), verts.max(0)
    centre = (bbox_min + bbox_max) / 2.0
    scale  = (bbox_max - bbox_min).max()
    if scale < 1e-8:
        raise ValueError(f'Degenerate mesh in {glb_path}')
    verts = (verts - centre) / scale

    # Convert to open3d so we can reuse sample_surface from dataset.py
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    pts, nrm, _ = sample_surface(o3d_mesh, n_points)
    return np.concatenate([pts, nrm], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device) -> RigAnythingModel:
    model = RigAnythingModel().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Autoregressive inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model:      RigAnythingModel,
    device:     torch.device,
    pc:         np.ndarray,                # [1024, 6]  float32
    max_joints: int = 64,
    ddim_steps: int = 50,
    dup_thresh: float = 0.01,
) -> dict:
    """
    Returns dict with 'joints' [K, 3], 'parents' [K] (1-indexed), 'weights' [L, K].
    """
    points  = torch.from_numpy(pc[:, :3]).to(device)       # [1024, 3]
    normals = torch.from_numpy(pc[:, 3:]).to(device)       # [1024, 3]

    # Tokenize shape
    pc_in = torch.cat([points, normals], dim=-1)
    H     = model.shape_tok(pc_in)                         # [1024, d]

    schedule = model._schedule()

    joints_list:  list[torch.Tensor] = []   # 3-vectors
    parents_list: list[int] = []            # 1-indexed
    T_list:       list[torch.Tensor] = []   # d-vectors

    for k in range(1, max_joints + 1):
        # Build T_stack
        T_stack = (torch.stack(T_list) if T_list
                   else H.new_zeros(0, model.d))            # [k-1, d]

        # Context
        Z_k = model.transformer(H, T_stack)                # [L+k-1, d]

        # Sample joint position via DDIM
        j_k = ddim_sample(
            model.denoiser, Z_k, schedule,
            ddim_steps=ddim_steps, device=str(device),
        )                                                   # [3]

        # Duplicate detection: stop if j_k is too close to an existing joint
        if joints_list:
            prev = torch.stack(joints_list)                 # [k-1, 3]
            min_dist = (prev - j_k).norm(dim=-1).min().item()
            if min_dist < dup_thresh:
                print(f'  Stopping at k={k}: duplicate joint (dist={min_dist:.4f})')
                break

        joints_list.append(j_k)

        # Connectivity
        if k == 1:
            p_k_1indexed = 1     # root is its own parent (1-indexed)
            parent_0idx  = 0
        else:
            Z_prime = model.fuser(Z_k, j_k, k)
            q_k     = model.connector(Z_prime, T_stack)     # [k-1]
            pred_0  = int(q_k.argmax().item())              # 0-indexed in T_stack
            p_k_1indexed = pred_0 + 1                       # 1-indexed
            parent_0idx  = pred_0

        parents_list.append(p_k_1indexed)

        # Skeleton token for this joint
        parent_0idx_clamped = max(0, min(parent_0idx, k - 1))
        parent_pos = joints_list[parent_0idx_clamped]
        T_k = model.skel_tok(
            torch.stack([j_k, parent_pos]),                # [2, 3]
            torch.tensor([1, 1], device=device),           # both point to index 1
        )[0]
        T_list.append(T_k)

    K_pred  = len(joints_list)
    joints  = torch.stack(joints_list)            # [K_pred, 3]
    T_all   = torch.stack(T_list)                 # [K_pred, d]

    # Skinning weights
    W = model.skinner(H, T_all)                   # [1024, K_pred]

    return {
        'joints':  joints.cpu().numpy(),
        'parents': np.array(parents_list, dtype=np.int32),
        'weights': W.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 7 Inference')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--shape_id', type=str,
                       help='Pre-processed shape ID in pointClouds/obj_remesh/')
    group.add_argument('--glb',      type=str,
                       help='Path to a .glb file — preprocessed on the fly')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/phase7/best_model.pt')
    parser.add_argument('--max_joints', type=int, default=64)
    parser.add_argument('--device',     type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    ckpt_path = resolve(args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f'ERROR: checkpoint not found: {ckpt_path}')
        sys.exit(1)

    out_dir = resolve('output/phase7')
    os.makedirs(out_dir, exist_ok=True)

    # ── Resolve shape_id and pointcloud ──────────────────────────
    if args.glb:
        glb_path = resolve(args.glb)
        if not os.path.exists(glb_path):
            print(f'ERROR: GLB file not found: {glb_path}')
            sys.exit(1)
        shape_id = os.path.splitext(os.path.basename(glb_path))[0]
        print(f'Phase 7 inference: glb={os.path.basename(glb_path)}  '
              f'shape_id={shape_id}  device={device}')
        print('  Preprocessing GLB → pointcloud ...')
        pc = glb_to_pointcloud(glb_path, n_points=1024)
        print(f'  Pointcloud ready: {pc.shape}')
    else:
        shape_id = args.shape_id
        pc_path  = resolve(os.path.join('pointClouds/obj_remesh',
                                        f'{shape_id}_pointcloud.npy'))
        if not os.path.exists(pc_path):
            print(f'ERROR: pointcloud not found: {pc_path}')
            sys.exit(1)
        print(f'Phase 7 inference: shape={shape_id}  device={device}')
        pc = np.load(pc_path).astype(np.float32)

    model  = load_model(ckpt_path, device)
    result = run_inference(model, device, pc, max_joints=args.max_joints)

    K_pred = result['joints'].shape[0]
    W      = result['weights']

    print(f'  Predicted joints: {K_pred}')
    print(f'  W shape: {W.shape}')
    print(f'  W rows sum to 1: min={W.sum(1).min():.6f}  max={W.sum(1).max():.6f}')
    print(f'  Joints finite: {np.isfinite(result["joints"]).all()}')

    np.save(os.path.join(out_dir, f'{shape_id}_joints.npy'),  result['joints'])
    np.save(os.path.join(out_dir, f'{shape_id}_parents.npy'), result['parents'])
    np.save(os.path.join(out_dir, f'{shape_id}_weights.npy'), W)
    print(f'  Saved → output/phase7/{shape_id}_{{joints,parents,weights}}.npy')


if __name__ == '__main__':
    main()
