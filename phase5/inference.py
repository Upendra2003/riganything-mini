"""
Phase 5 — Connectivity Prediction Inference
============================================
Autoregressively generates a skeleton (joint positions + parent indices) for a
single shape using the full Phase 3 → Phase 4 → Phase 5 pipeline:

  For each step k = 1, 2, ...:
    1. Z_k  = Phase 3 transformer(H, T_prev)         [L+k-1, d]
    2. j_k  = Phase 4 DDIM sampler(Z_k)              [3]
    3. Z'_k = FusingModule(Z_k, j_k, k)              [d]
    4. q_k  = ConnectivityModule(Z'_k, T_prev)       [k-1]   (if k ≥ 2)
    5. p_k  = argmax(q_k)   (0-indexed into T_prev)

Stop condition: self-loop detected (predicted parent index + 1 == k) or
max_joints reached.

Output:
  output/pred_skels/<id>.json
    {"joints": [[x,y,z], ...], "parents": [-1, 0, 0, 1, ...]}
  Root parent = -1, all others 0-indexed.

Usage:
  .venv/bin/python phase5/inference.py --shape_id <id>
  .venv/bin/python phase5/inference.py --shape_id 10532 --max_joints 80
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

# ── Make project root importable when running as a script ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase3.config       import Config as Phase3Config
from phase3.transformer  import HybridTransformer
from phase4.config       import Phase4Config
from phase4.model        import DenoisingMLP
from phase4.noise_schedule import compute_cosine_schedule, ddim_sample
from phase5.config       import Phase5Config
from phase5.connectivity import FusingModule, ConnectivityModule


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_frozen_transformer(ckpt_path: str, device: torch.device) -> HybridTransformer:
    p3_cfg      = Phase3Config()
    transformer = HybridTransformer(p3_cfg).to(device)
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt['model_state_dict'])
    transformer.eval()
    transformer.requires_grad_(False)
    return transformer


def load_frozen_denoiser(ckpt_path: str, device: torch.device) -> DenoisingMLP:
    p4_cfg   = Phase4Config()
    denoiser = DenoisingMLP(d=p4_cfg.d, M=p4_cfg.M).to(device)
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    denoiser.load_state_dict(ckpt['model_state_dict'])
    denoiser.eval()
    denoiser.requires_grad_(False)
    return denoiser


def load_phase5(ckpt_path: str, device: torch.device, d: int = 1024):
    fuser     = FusingModule(d=d).to(device)
    connector = ConnectivityModule(d=d).to(device)
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    fuser.load_state_dict(ckpt['fuser_state_dict'])
    connector.load_state_dict(ckpt['connector_state_dict'])
    fuser.eval()
    connector.eval()
    fuser.requires_grad_(False)
    connector.requires_grad_(False)
    return fuser, connector


# ---------------------------------------------------------------------------
# Skeleton tree builder
# ---------------------------------------------------------------------------

def build_skeleton_tree(joints_xyz, parents_0indexed):
    """
    Build the skeleton dict and validate BFS invariant.

    Args:
        joints_xyz:     list of [x, y, z] (one per joint, 0-indexed)
        parents_0indexed: list of int (-1 for root, else 0-indexed parent)

    Returns:
        tree: dict {joint_idx: {'pos': [x,y,z], 'parent': int, 'children': list}}

    Raises:
        ValueError if BFS invariant (p_k < k for k > 0) is violated.
    """
    tree = {}
    for i, (pos, parent) in enumerate(zip(joints_xyz, parents_0indexed)):
        if i > 0 and parent >= i:
            raise ValueError(
                f'BFS invariant violated: joint {i} has parent {parent} ≥ {i}'
            )
        children = [j for j, p in enumerate(parents_0indexed) if p == i]
        tree[i] = {'pos': pos, 'parent': parent, 'children': children}
    return tree


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    shape_id:   str,
    config:     Phase5Config,
    device:     torch.device,
    max_joints: int = 100,
) -> dict:
    """
    Returns the predicted skeleton as:
      {'joints': [[x,y,z], ...], 'parents': [-1, 0, ...]}
    """
    # ── Load pre-computed tokens ─────────────────────────────────────
    token_dir = config.token_dir
    h_path    = os.path.join(token_dir, f'{shape_id}_H.pt')
    t_path    = os.path.join(token_dir, f'{shape_id}_T.pt')

    if not os.path.exists(h_path):
        raise FileNotFoundError(f'H tokens not found: {h_path}')
    if not os.path.exists(t_path):
        raise FileNotFoundError(f'T tokens not found: {t_path}')

    H = torch.load(h_path, weights_only=True).to(device)   # [1024, 1024]
    T = torch.load(t_path, weights_only=True).to(device)   # [K, 1024]
    K_ref = T.shape[0]   # reference K from pre-computed tokens (BFS order)

    # ── Load all models ───────────────────────────────────────────────
    transformer = load_frozen_transformer(config.phase3_ckpt, device)
    denoiser    = load_frozen_denoiser(config.phase4_ckpt, device)
    fuser, connector = load_phase5(
        os.path.join(config.checkpoint_dir, 'best_model.pt'), device, config.d
    )

    schedule = compute_cosine_schedule(M=config.M)

    # ── Autoregressive generation ─────────────────────────────────────
    # We use the pre-tokenized T tokens as the skeleton token sequence.
    # At inference, T[k-1] is produced by the SkeletonTokenizer in Phase 7;
    # here we use the cached tokens as a proxy.
    joints_xyz     = []
    parents_result = []
    T_prev_list    = []

    for k in range(1, max_joints + 1):
        # Build T_prev from the cached tokens (capped at actual K)
        if k - 1 <= K_ref:
            T_prev = T[:k - 1].to(device)   # [k-1, 1024]
        else:
            # Ran out of cached tokens — stop
            break

        # Phase 3: get context
        Z_k = transformer(H, T_prev)          # [1024 + k-1, 1024]

        # Phase 4: sample joint position via DDIM
        j_k = ddim_sample(denoiser, Z_k, schedule,
                           ddim_steps=config.ddim_steps, device=str(device))  # [3]

        joints_xyz.append(j_k.cpu().tolist())

        # Phase 5: predict parent (only from k=2 onward)
        if k == 1:
            parents_result.append(-1)   # root has no parent
        else:
            Z_prime = fuser(Z_k, j_k, k)           # [d]
            q_k     = connector(Z_prime, T_prev)    # [k-1]
            pred_parent_0indexed = int(q_k.argmax().item())   # 0-indexed into T_prev

            # Stop condition: if predicted parent is the most recent joint (self-loop
            # in the sense that argmax + 1 == k), stop generating new joints.
            # (argmax is 0-indexed 0..k-2; argmax+1 is 1-indexed 1..k-1; so
            # argmax+1 == k would require argmax == k-1, which is out of range —
            # this loop therefore acts as a graceful cap via max_joints instead.)
            if pred_parent_0indexed + 1 == k:
                # Self-loop signal: break without adding this joint
                joints_xyz.pop()
                break

            parents_result.append(pred_parent_0indexed)

        # Stop once we've generated as many joints as the reference token file
        if k >= K_ref:
            break

    # ── Validate BFS invariant ────────────────────────────────────────
    tree = build_skeleton_tree(joints_xyz, parents_result)

    return {'joints': joints_xyz, 'parents': parents_result, 'tree': tree}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 5 — Connectivity Inference')
    parser.add_argument('--shape_id',   type=str, required=True)
    parser.add_argument('--max_joints', type=int, default=100)
    parser.add_argument('--device',     type=str, default=None)
    args = parser.parse_args()

    config = Phase5Config()
    if args.device is not None:
        config.device = args.device

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    config.phase3_ckpt    = resolve(config.phase3_ckpt)
    config.phase4_ckpt    = resolve(config.phase4_ckpt)
    config.token_dir      = resolve(config.token_dir)
    config.skel_dir       = resolve(config.skel_dir)
    config.checkpoint_dir = resolve(config.checkpoint_dir)

    device = torch.device(config.device)

    print(f'Running Phase 5 inference for shape: {args.shape_id}')
    result = run_inference(args.shape_id, config, device, max_joints=args.max_joints)

    out_dir  = os.path.join(ROOT, 'output', 'pred_skels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.shape_id}.json')

    # Remove non-serializable tree from output (keep joints and parents only)
    save_data = {'joints': result['joints'], 'parents': result['parents']}
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    K_pred = len(result['joints'])
    print(f'Predicted skeleton: {K_pred} joints')
    print(f'Saved to: {out_path}')


if __name__ == '__main__':
    main()
