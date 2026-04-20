"""
Phase 4 — Dataset & DataLoader
================================
Loads shape IDs from official split files, then reads pre-computed token files
(Phase 2 output) and skeleton ground-truth (Phase 1 output) for each shape.

Expected layout:
  {token_dir}/{id}_H.pt       → FloatTensor [1024, 1024]
  {token_dir}/{id}_T.pt       → FloatTensor [K,    1024]
  {skel_dir}/{id}_skeleton.npy → float32 [K, 4]  (xyz + 1-indexed parent)

Split files (one shape ID per line, may have trailing '.obj'):
  Dataset/train_final.txt
  Dataset/val_final.txt
"""

import os
import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def _load_split(split_file: str) -> List[str]:
    """Read shape IDs from a split txt file, stripping '.obj' suffix if present."""
    with open(split_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    # Strip '.obj' extension that Phase 1 uses in split files
    ids = [l[:-4] if l.lower().endswith('.obj') else l for l in lines]
    return ids


class Phase4Dataset(Dataset):
    """
    One item = one shape.  __getitem__ returns:
      {
        'shape_id': str,
        'H':        Tensor [1024, 1024],   shape tokens
        'T':        Tensor [K,    1024],   skeleton tokens
        'joints':   Tensor [K, 3],         joint xyz positions
        'parents':  Tensor [K] long,       parent index (1-indexed, as in Phase 1)
        'K':        int,                   joint count
      }

    Shapes whose token files are missing are silently skipped.
    """

    def __init__(self, split_file: str, token_dir: str, skel_dir: str):
        """
        Args:
            split_file: path to train_final.txt or val_final.txt
            token_dir:  directory containing *_H.pt and *_T.pt
            skel_dir:   directory containing *_skeleton.npy
        """
        all_ids = _load_split(split_file)

        self.items: List[Dict[str, str]] = []
        skipped = 0
        for sid in all_ids:
            h_path    = os.path.join(token_dir, f'{sid}_H.pt')
            t_path    = os.path.join(token_dir, f'{sid}_T.pt')
            skel_path = os.path.join(skel_dir,  f'{sid}_skeleton.npy')
            if not os.path.exists(h_path):
                logger.warning('Skipping %s — missing %s', sid, h_path)
                skipped += 1
                continue
            if not os.path.exists(t_path):
                logger.warning('Skipping %s — missing %s', sid, t_path)
                skipped += 1
                continue
            if not os.path.exists(skel_path):
                logger.warning('Skipping %s — missing %s', sid, skel_path)
                skipped += 1
                continue
            self.items.append({
                'shape_id': sid,
                'h_path':   h_path,
                't_path':   t_path,
                'skel_path': skel_path,
            })

        if skipped:
            logger.warning('Phase4Dataset: skipped %d shapes (missing files)', skipped)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        H = torch.load(item['h_path'], weights_only=True)   # [1024, 1024]
        T = torch.load(item['t_path'], weights_only=True)   # [K,    1024]

        skel = np.load(item['skel_path'])                   # [K, 4] float32
        joints  = torch.from_numpy(skel[:, :3]).float()     # [K, 3]
        parents = torch.from_numpy(skel[:,  3]).long()      # [K]   1-indexed

        return {
            'shape_id': item['shape_id'],
            'H':        H,
            'T':        T,
            'joints':   joints,
            'parents':  parents,
            'K':        T.shape[0],
        }


def make_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders using the official split files.

    batch_size is always 1 for Phase 4 (inner loop over joints per shape).
    """
    train_ds = Phase4Dataset(config.train_split, config.token_dir, config.skel_dir)
    val_ds   = Phase4Dataset(config.val_split,   config.token_dir, config.skel_dir)

    pin = (config.device == 'cuda')

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
        collate_fn=_collate_single,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
        collate_fn=_collate_single,
    )
    return train_loader, val_loader


def _collate_single(batch: List[Dict]) -> Dict[str, Any]:
    """
    Identity collate for batch_size=1: just unwrap the list and keep tensors
    as-is (no stacking needed since each shape has different K).
    """
    assert len(batch) == 1, 'Phase 4 DataLoader must use batch_size=1'
    item = batch[0]
    return {
        'shape_id': item['shape_id'],
        'H':        item['H'],          # [1024, 1024]
        'T':        item['T'],          # [K, 1024]
        'joints':   item['joints'],     # [K, 3]
        'parents':  item['parents'],    # [K]
        'K':        item['K'],
    }
