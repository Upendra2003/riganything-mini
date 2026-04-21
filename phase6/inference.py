"""
Phase 6 — Skinning Weight Prediction Inference
===============================================
Loads pre-tokenized H and T for a single shape, runs SkinningModule,
and saves the predicted weight matrix W [1024, K] as a .npy file.

Usage:
  .venv/bin/python phase6/inference.py --shape_id <id>
  .venv/bin/python phase6/inference.py --shape_id 10000
"""

import os
import sys
import argparse

import numpy as np
import torch

# ── Make project root importable when running as a script ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase6.config import Phase6Config
from phase6.model  import SkinningModule


def load_model(ckpt_path: str, device: torch.device, d: int = 1024) -> SkinningModule:
    model = SkinningModule(d=d).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def run_inference(shape_id: str, config: Phase6Config, device: torch.device) -> np.ndarray:
    """
    Returns W [1024, K] as a numpy float32 array.
    """
    h_path = os.path.join(config.token_dir, f'{shape_id}_H.pt')
    t_path = os.path.join(config.token_dir, f'{shape_id}_T.pt')

    if not os.path.exists(h_path):
        raise FileNotFoundError(f'H tokens not found: {h_path}')
    if not os.path.exists(t_path):
        raise FileNotFoundError(f'T tokens not found: {t_path}')

    H = torch.load(h_path, weights_only=True).to(device)  # [1024, 1024]
    T = torch.load(t_path, weights_only=True).to(device)  # [K, 1024]

    ckpt_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Phase 6 checkpoint not found: {ckpt_path}')

    model = load_model(ckpt_path, device, d=config.d)
    W     = model(H, T)   # [1024, K]
    return W.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 6 — Skinning Inference')
    parser.add_argument('--shape_id', type=str, required=True)
    parser.add_argument('--device',   type=str, default=None)
    args = parser.parse_args()

    config = Phase6Config()
    if args.device is not None:
        config.device = args.device

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    config.token_dir      = resolve(config.token_dir)
    config.skel_dir       = resolve(config.skel_dir)
    config.checkpoint_dir = resolve(config.checkpoint_dir)

    device = torch.device(config.device)

    print(f'Running Phase 6 inference for shape: {args.shape_id}')
    W = run_inference(args.shape_id, config, device)

    # Validate
    K = W.shape[1]
    row_sums = W.sum(axis=1)
    print(f'  W shape:  {W.shape}')
    print(f'  W min:    {W.min():.6f}')
    print(f'  W max:    {W.max():.6f}')
    print(f'  W mean:   {W.mean():.6f}')
    print(f'  row sums: min={row_sums.min():.6f}  max={row_sums.max():.6f}  '
          f'(should be ≈1.0)')

    out_dir  = os.path.join(ROOT, 'output', 'pred_skins')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.shape_id}_weights.npy')
    np.save(out_path, W.astype(np.float32))
    print(f'  Saved → {out_path}')


if __name__ == '__main__':
    main()
