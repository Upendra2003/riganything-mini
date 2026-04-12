"""
Phase 3 — Dataset & DataLoader
================================
Loads pre-computed shape (H) and skeleton (T) tokens produced by Phase 2.

Expected directory layout:
  {tokens_dir}/{id}_H.pt   → FloatTensor [1024, 1024]
  {tokens_dir}/{id}_T.pt   → FloatTensor [K,    1024]

Because different shapes have different joint counts K, a custom collate
function pads T to the maximum K in each batch.
"""

import os
import glob
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class Phase3Dataset(Dataset):
    """
    Scans `tokens_dir` for *_H.pt files and pairs each with its *_T.pt.

    Returns dicts:
      {'H': Tensor[1024,1024], 'T': Tensor[K,1024], 'K': int, 'shape_id': str}
    """

    def __init__(self, tokens_dir: str):
        h_files = sorted(glob.glob(os.path.join(tokens_dir, '*_H.pt')))
        self.shape_ids  = [os.path.basename(f).replace('_H.pt', '') for f in h_files]
        self.tokens_dir = tokens_dir

    def __len__(self) -> int:
        return len(self.shape_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid  = self.shape_ids[idx]
        base = os.path.join(self.tokens_dir, sid)
        H = torch.load(f'{base}_H.pt', weights_only=True)   # [1024, 1024]
        T = torch.load(f'{base}_T.pt', weights_only=True)   # [K,    1024]
        return {'H': H, 'T': T, 'K': T.shape[0], 'shape_id': sid}


def phase3_collate(batch: List[Dict]) -> Dict:
    """
    Collate a list of dataset items into a batch.

    H stacks normally  → [B, 1024, 1024]
    T is padded to max K in the batch → [B, max_K, 1024] (zeros for padding)
    lengths           → LongTensor [B]  with the real K per sample
    """
    B     = len(batch)
    max_K = max(item['K'] for item in batch)
    d     = batch[0]['H'].shape[-1]

    H_stack  = torch.stack([item['H'] for item in batch])   # [B, L, d]
    T_padded = torch.zeros(B, max_K, d)

    for i, item in enumerate(batch):
        k = item['K']
        T_padded[i, :k] = item['T']

    lengths   = torch.tensor([item['K']       for item in batch], dtype=torch.long)
    shape_ids = [item['shape_id'] for item in batch]

    return {'H': H_stack, 'T': T_padded, 'lengths': lengths, 'shape_ids': shape_ids}


def make_dataloaders(config):
    """
    Build train/val DataLoaders with a 90/10 split.

    The split is reproducible (seeded) so train/val sets are stable across runs.
    """
    ds = Phase3Dataset(config.tokens_dir)

    # Reproducible, seed-based shuffle — does not touch global RNG state
    rng     = random.Random(config.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    split     = int(0.9 * len(indices))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=phase3_collate,
        pin_memory=(config.device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=phase3_collate,
        pin_memory=(config.device == 'cuda'),
    )

    return train_loader, val_loader
