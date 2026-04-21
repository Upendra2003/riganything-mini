"""
Phase 7 — Dataset
==================
Loads raw pointcloud, skeleton, and skinning data for end-to-end training.

Note on skinning alignment:
  _skinning.npy has V rows (mesh vertex count); pointcloud has L=1024 rows.
  V != 1024 for all shapes in the dataset, so we use _resample_skinning from
  phase6 to produce [L, K] ground-truth skinning weights.
  (The spec says to skip V != 1024, but that would leave zero shapes. We
  resample instead, which is consistent with Phase 6 training.)
"""

import os
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase6.dataset import _resample_skinning

logger = logging.getLogger(__name__)

L_POINTS = 1024


def _load_split(split_file: str) -> List[str]:
    with open(split_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return [l[:-4] if l.lower().endswith('.obj') else l for l in lines]


class Phase7Dataset(Dataset):
    """
    One item = one shape.  Returns:
      {
        'shape_id': str,
        'points':   Tensor [1024, 3],
        'normals':  Tensor [1024, 3],
        'gt_joints':  Tensor [K, 3],
        'gt_parents': Tensor [K]    long, 1-indexed
        'gt_skin':    Tensor [1024, K]
        'K':        int
      }
    """

    def __init__(
        self,
        split_file: str,
        pc_dir:     str,
        max_shapes: int | None = None,
    ):
        all_ids = _load_split(split_file)
        if max_shapes is not None:
            all_ids = all_ids[:max_shapes]

        self.items: List[Dict[str, str]] = []
        skipped = 0
        for sid in all_ids:
            pc_path   = os.path.join(pc_dir, f'{sid}_pointcloud.npy')
            skel_path = os.path.join(pc_dir, f'{sid}_skeleton.npy')
            skin_path = os.path.join(pc_dir, f'{sid}_skinning.npy')
            if not all(os.path.exists(p) for p in (pc_path, skel_path, skin_path)):
                skipped += 1
                continue
            self.items.append({
                'shape_id':  sid,
                'pc_path':   pc_path,
                'skel_path': skel_path,
                'skin_path': skin_path,
            })
        if skipped:
            logger.warning('Phase7Dataset: skipped %d shapes (missing files)', skipped)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        pc   = np.load(item['pc_path'])                        # [1024, 6]
        skel = np.load(item['skel_path'])                      # [K, 4]
        skin = np.load(item['skin_path'])                      # [V, K]

        points  = torch.from_numpy(pc[:, :3]).float()          # [1024, 3]
        normals = torch.from_numpy(pc[:, 3:]).float()          # [1024, 3]

        gt_joints  = torch.from_numpy(skel[:, :3]).float()     # [K, 3]
        gt_parents = torch.from_numpy(skel[:,  3]).long()      # [K]  1-indexed

        K = skel.shape[0]
        W = _resample_skinning(skin, L_POINTS)                  # [1024, K]
        gt_skin = torch.from_numpy(W).float()

        return {
            'shape_id':  item['shape_id'],
            'points':    points,
            'normals':   normals,
            'gt_joints': gt_joints,
            'gt_parents':gt_parents,
            'gt_skin':   gt_skin,
            'K':         K,
        }


def _collate_single(batch: List[Dict]) -> Dict[str, Any]:
    assert len(batch) == 1, 'Phase7 DataLoader requires batch_size=1'
    return batch[0]


def make_dataloaders(
    train_split: str,
    val_split:   str,
    pc_dir:      str,
    device:      str = 'cpu',
    max_shapes:  int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = Phase7Dataset(train_split, pc_dir, max_shapes)
    val_ds   = Phase7Dataset(val_split,   pc_dir)
    pin      = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=pin,
                              collate_fn=_collate_single)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=pin,
                              collate_fn=_collate_single)
    return train_loader, val_loader
