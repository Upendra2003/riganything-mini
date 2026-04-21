"""
Phase 6 — Dataset & DataLoader
================================
Loads pre-tokenized shape/skeleton tokens and ground-truth skinning weights.

Ground-truth skinning alignment:
  _skinning.npy has V rows (mesh vertex count), H has L=1024 rows (sampled points).
  Since OBJ mesh files are not available for exact nearest-neighbour matching,
  we use a simple spatial-index approach:
    V >= L: skin[:L]  (first L rows; vertex ordering has spatial coherence)
    V  < L: np.tile to reach L rows, then truncate to L
  Rows are renormalized to sum to 1 after resampling.

Expected layout:
  {token_dir}/{id}_H.pt         → FloatTensor [1024, 1024]
  {token_dir}/{id}_T.pt         → FloatTensor [K, 1024]
  {skel_dir}/{id}_skinning.npy  → float32 [V, K]
"""

import os
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

L_POINTS = 1024   # fixed number of shape tokens


def _load_split(split_file: str) -> List[str]:
    with open(split_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return [l[:-4] if l.lower().endswith('.obj') else l for l in lines]


def _resample_skinning(skin: np.ndarray, L: int) -> np.ndarray:
    """
    Resample skinning [V, K] → [L, K] preserving row normalisation.

    V >= L: take first L rows.
    V  < L: tile rows until we have at least L, then truncate.
    Final rows are re-normalised to sum to 1 to absorb any float errors.
    """
    V, K = skin.shape
    if V >= L:
        out = skin[:L].copy()
    else:
        repeats = (L + V - 1) // V
        out = np.tile(skin, (repeats, 1))[:L].copy()

    row_sum = out.sum(axis=1, keepdims=True)
    out /= np.where(row_sum < 1e-8, 1.0, row_sum)
    return out.astype(np.float32)


class Phase6Dataset(Dataset):
    """
    One item = one shape.  __getitem__ returns:
      {
        'shape_id': str,
        'H':        Tensor [1024, 1024],
        'T':        Tensor [K, 1024],
        'W_gt':     Tensor [1024, K],    ground-truth skinning weights
        'K':        int,
      }
    Shapes with missing files are silently skipped.
    """

    def __init__(self, split_file: str, token_dir: str, skel_dir: str):
        all_ids = _load_split(split_file)
        self.items: List[Dict[str, str]] = []
        skipped = 0
        for sid in all_ids:
            h_path    = os.path.join(token_dir, f'{sid}_H.pt')
            t_path    = os.path.join(token_dir, f'{sid}_T.pt')
            skin_path = os.path.join(skel_dir,  f'{sid}_skinning.npy')
            if not all(os.path.exists(p) for p in (h_path, t_path, skin_path)):
                skipped += 1
                continue
            self.items.append({
                'shape_id':  sid,
                'h_path':    h_path,
                't_path':    t_path,
                'skin_path': skin_path,
            })
        if skipped:
            logger.warning('Phase6Dataset: skipped %d shapes (missing files)', skipped)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        H    = torch.load(item['h_path'], weights_only=True)    # [1024, 1024]
        T    = torch.load(item['t_path'], weights_only=True)    # [K, 1024]
        K    = T.shape[0]

        skin = np.load(item['skin_path'])                        # [V, K_orig]
        # skinning.npy K dimension matches skeleton K (both from the same rig)
        assert skin.shape[1] == K, (
            f"{item['shape_id']}: skinning K={skin.shape[1]} != token K={K}"
        )

        W_gt = torch.from_numpy(_resample_skinning(skin, L_POINTS))  # [1024, K]

        return {
            'shape_id': item['shape_id'],
            'H':        H,
            'T':        T,
            'W_gt':     W_gt,
            'K':        K,
        }


def _collate_single(batch: List[Dict]) -> Dict[str, Any]:
    """Identity collate for batch_size=1 — shapes have variable K."""
    assert len(batch) == 1, 'Phase6 DataLoader must use batch_size=1'
    item = batch[0]
    return {
        'shape_id': item['shape_id'],
        'H':        item['H'],
        'T':        item['T'],
        'W_gt':     item['W_gt'],
        'K':        item['K'],
    }


def make_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    train_ds = Phase6Dataset(config.train_split, config.token_dir, config.skel_dir)
    val_ds   = Phase6Dataset(config.val_split,   config.token_dir, config.skel_dir)
    pin      = (config.device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=pin,
                              collate_fn=_collate_single)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=pin,
                              collate_fn=_collate_single)
    return train_loader, val_loader
