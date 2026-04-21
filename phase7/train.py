"""
Phase 7 — End-to-End Joint Training
=====================================
Jointly trains all seven submodules with the combined loss:
  L_total = L_joint + L_connect + L_skinning   (equal weights)

Warm-start: Phase 3/4/5/6 pretrained weights are loaded before epoch 1.
            ShapeTokenizer and SkeletonTokenizer start from random init.

Online augmentation: random per-joint rotations + LBS deformation at every step.

Memory notes:
  * T_prev tokens are detached inside RigAnythingModel.forward() to bound the
    per-step graph size.  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    is recommended.
  * batch_size=1 (one shape per optimizer step) — K varies per shape.

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase7/train.py --epochs 50
  .venv/bin/python phase7/train.py --epochs 2 --max_shapes 5 --no_augment
  .venv/bin/python phase7/train.py --resume checkpoints/phase7/best_model.pt
  sbatch slurm/phase7_train.sh
"""

import os
import sys
import csv
import random
import argparse

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase7.model   import RigAnythingModel
from phase7.dataset import make_dataloaders
from phase7.augment import augment_shape


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Warm-start checkpoint loading
# ---------------------------------------------------------------------------
def _load_sd(path: str, device: torch.device) -> dict | None:
    if not os.path.exists(path):
        print(f'  [SKIP] checkpoint not found: {path}')
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def warm_start(model: RigAnythingModel, root: str, device: torch.device) -> None:
    """
    Load pretrained weights from Phase 3/4/5/6 checkpoints into the model.
    Missing checkpoints are skipped with a warning.
    """
    print('Warm-starting from pretrained checkpoints …')

    # Phase 3 → transformer
    ckpt3 = _load_sd(os.path.join(root, 'checkpoints/phase3/best_model.pt'), device)
    if ckpt3:
        missing, unexpected = model.transformer.load_state_dict(
            ckpt3['model_state_dict'], strict=False
        )
        print(f'  transformer  ← phase3  |  missing={len(missing)}  unexpected={len(unexpected)}')

    # Phase 4 → denoiser
    ckpt4 = _load_sd(os.path.join(root, 'checkpoints/phase4/best_model.pt'), device)
    if ckpt4:
        missing, unexpected = model.denoiser.load_state_dict(
            ckpt4['model_state_dict'], strict=False
        )
        print(f'  denoiser     ← phase4  |  missing={len(missing)}  unexpected={len(unexpected)}')

    # Phase 5 → fuser + connector
    ckpt5 = _load_sd(os.path.join(root, 'checkpoints/phase5/best_model.pt'), device)
    if ckpt5:
        missing, unexpected = model.fuser.load_state_dict(
            ckpt5['fuser_state_dict'], strict=False
        )
        print(f'  fuser        ← phase5  |  missing={len(missing)}  unexpected={len(unexpected)}')
        missing, unexpected = model.connector.load_state_dict(
            ckpt5['connector_state_dict'], strict=False
        )
        print(f'  connector    ← phase5  |  missing={len(missing)}  unexpected={len(unexpected)}')

    # Phase 6 → skinner
    ckpt6 = _load_sd(os.path.join(root, 'checkpoints/phase6/best_model.pt'), device)
    if ckpt6:
        missing, unexpected = model.skinner.load_state_dict(
            ckpt6['model_state_dict'], strict=False
        )
        print(f'  skinner      ← phase6  |  missing={len(missing)}  unexpected={len(unexpected)}')

    print('Warm-start complete.')


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    epoch: int,
    model: RigAnythingModel,
    optimizer,
    scheduler,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'val_loss':             val_loss,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(
    model:       RigAnythingModel,
    loader,
    optimizer,
    device:      torch.device,
    epoch:       int,
    use_augment: bool,
) -> tuple[float, float, float, float]:
    model.train()
    sum_total = sum_j = sum_c = sum_s = 0.0
    n = 0

    for batch in loader:
        points    = batch['points'].to(device)     # [1024, 3]
        normals   = batch['normals'].to(device)    # [1024, 3]
        gt_joints = batch['gt_joints'].to(device)  # [K, 3]
        gt_parents= batch['gt_parents'].to(device) # [K]
        gt_skin   = batch['gt_skin'].to(device)    # [1024, K]
        sid       = batch['shape_id']

        # Online pose augmentation
        if use_augment:
            try:
                new_pts, new_nrm, new_jts = augment_shape(
                    points, gt_joints, gt_parents, gt_skin
                )
                points    = new_pts
                normals   = new_nrm
                gt_joints = new_jts
            except Exception as e:
                pass  # fall back to un-augmented on rare numerical issues

        total, lj, lc, ls = model(points, normals, gt_joints, gt_parents, gt_skin)

        optimizer.zero_grad()
        total.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t = total.item(); j = lj.item(); c = lc.item(); s = ls.item()
        sum_total += t; sum_j += j; sum_c += c; sum_s += s
        n += 1

        print(f'  E{epoch:>3} | {sid} | total={t:.4f}  j={j:.4f}  c={c:.4f}  s={s:.4f}')

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    N = max(n, 1)
    return sum_total/N, sum_j/N, sum_c/N, sum_s/N


@torch.no_grad()
def val_epoch(
    model:  RigAnythingModel,
    loader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    sum_total = sum_j = sum_c = sum_s = 0.0
    n = 0

    for batch in loader:
        points    = batch['points'].to(device)
        normals   = batch['normals'].to(device)
        gt_joints = batch['gt_joints'].to(device)
        gt_parents= batch['gt_parents'].to(device)
        gt_skin   = batch['gt_skin'].to(device)

        total, lj, lc, ls = model(points, normals, gt_joints, gt_parents, gt_skin)
        sum_total += total.item(); sum_j += lj.item()
        sum_c += lc.item(); sum_s += ls.item()
        n += 1

    N = max(n, 1)
    return sum_total/N, sum_j/N, sum_c/N, sum_s/N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 7 — End-to-End Training')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--resume',     type=str,   default=None)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--max_shapes', type=int,   default=None)
    parser.add_argument('--device',     type=str,   default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    set_seed(42)

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    pc_dir       = resolve('pointClouds/obj_remesh')
    train_split  = resolve('Dataset/train_final.txt')
    val_split    = resolve('Dataset/val_final.txt')
    ckpt_dir     = resolve('checkpoints/phase7')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        train_split, val_split, pc_dir,
        device=str(device), max_shapes=args.max_shapes
    )
    print(f'Dataset: {len(train_loader.dataset)} train  {len(val_loader.dataset)} val')

    # ── Model ─────────────────────────────────────────────────────────
    model = RigAnythingModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'RigAnythingModel: {total_params:,} parameters  device={device}')

    # ── Warm-start ────────────────────────────────────────────────────
    warm_start(model, ROOT, device)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch   = 0
    best_val_loss = float('inf')

    # ── Resume ────────────────────────────────────────────────────────
    if args.resume:
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if incompatible.missing_keys:
            print(f'  [WARN] missing keys : {len(incompatible.missing_keys)}')
        if incompatible.unexpected_keys:
            print(f'  [WARN] unexpected keys (ignored): {len(incompatible.unexpected_keys)}')
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'  Resumed epoch={ckpt.get("epoch", 0)}  val_loss={best_val_loss:.6f}  '
              f'(optimizer restored: {"optimizer_state_dict" in ckpt})')

    # ── Sanity check: one forward pass before epoch 1 ─────────────────
    print('Sanity-checking forward pass …')
    model.eval()
    with torch.no_grad():
        sample = next(iter(val_loader))
        _tot, _lj, _lc, _ls = model(
            sample['points'].to(device),
            sample['normals'].to(device),
            sample['gt_joints'].to(device),
            sample['gt_parents'].to(device),
            sample['gt_skin'].to(device),
        )
    assert torch.isfinite(_tot), f'Pre-training sanity check FAILED: total_loss={_tot}'
    print(f'  OK — total={_tot:.4f}  j={_lj:.4f}  c={_lc:.4f}  s={_ls:.4f}')

    # ── CSV log ───────────────────────────────────────────────────────
    log_path = os.path.join(ckpt_dir, 'train_log.csv')
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    writer   = csv.writer(log_file)
    if not log_exists:
        writer.writerow([
            'epoch',
            'train_loss','train_loss_joint','train_loss_connect','train_loss_skin',
            'val_loss',  'val_loss_joint',  'val_loss_connect',  'val_loss_skin',
        ])

    # ── Training loop ─────────────────────────────────────────────────
    use_augment = not args.no_augment

    for epoch in range(start_epoch, args.epochs):
        tr = train_epoch(model, train_loader, optimizer, device, epoch, use_augment)
        vl = val_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f'── Epoch {epoch:>3} │ '
            f'train {tr[0]:.6f} (j={tr[1]:.4f} c={tr[2]:.4f} s={tr[3]:.4f}) │ '
            f'val {vl[0]:.6f} (j={vl[1]:.4f} c={vl[2]:.4f} s={vl[3]:.4f})'
        )
        writer.writerow([epoch, *tr, *vl])
        log_file.flush()

        if (epoch + 1) % 5 == 0:
            path = os.path.join(ckpt_dir, f'epoch_{epoch}.pt')
            save_checkpoint(path, epoch, model, optimizer, scheduler, vl[0])
            print(f'  Saved periodic → {path}')

        if vl[0] < best_val_loss:
            best_val_loss = vl[0]
            best = os.path.join(ckpt_dir, 'best_model.pt')
            save_checkpoint(best, epoch, model, optimizer, scheduler, vl[0])
            print(f'  New best val_loss={vl[0]:.6f} → {best}')

    log_file.close()
    print('Training complete.')


if __name__ == '__main__':
    main()
