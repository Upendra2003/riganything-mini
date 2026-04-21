"""
Phase 6 — Skinning Weight Prediction Training
==============================================
Trains SkinningModule to predict per-point skinning weights from pre-tokenized
shape tokens H and skeleton tokens T.

Design:
  * No frozen Phase 3 transformer — uses cached _H.pt / _T.pt directly.
  * Pairwise MLP over shape × skeleton tokens: [L, K, 2d] → W [L, K].
  * Loss: weighted cross-entropy  L = (1/L) Σ_l Σ_k [ -ŵ·log(w+1e-8) ].
  * Gradient accumulation over accum_steps=4 shapes → effective batch size 4.
  * CosineAnnealingLR with T_max = epochs.
  * AMP (fp16) when CUDA is available.
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (set before launching)

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase6/train.py --epochs 30
  .venv/bin/python phase6/train.py --resume checkpoints/phase6/best_model.pt
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

# ── Make project root importable when running as a script ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase6.config  import Phase6Config
from phase6.model   import SkinningModule, skinning_loss
from phase6.dataset import make_dataloaders


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
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    epoch: int,
    model: SkinningModule,
    optimizer,
    scheduler,
    scaler,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'val_loss':             val_loss,
    }, path)


# ---------------------------------------------------------------------------
# Training epoch  (gradient accumulation over accum_steps shapes)
# ---------------------------------------------------------------------------
def train_epoch(
    model:    SkinningModule,
    loader,
    optimizer,
    scaler:   GradScaler,
    config:   Phase6Config,
    device:   torch.device,
    epoch:    int,
) -> float:
    model.train()
    use_amp = config.amp and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    epoch_loss  = 0.0
    total_shapes = 0
    accum        = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        H    = batch['H'].to(device)     # [1024, 1024]
        T    = batch['T'].to(device)     # [K, 1024]
        W_gt = batch['W_gt'].to(device)  # [1024, K]

        with amp_ctx:
            W    = model(H, T)
            loss = skinning_loss(W, W_gt) / config.accum_steps

        scaler.scale(loss).backward()
        accum += 1

        if accum == config.accum_steps or step == len(loader) - 1:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accum = 0

        epoch_loss   += loss.item() * config.accum_steps
        total_shapes += 1

    avg_loss = epoch_loss / max(total_shapes, 1)
    print(f'Epoch {epoch:>3} | train_loss: {avg_loss:.6f}')
    return avg_loss


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_epoch(
    model:  SkinningModule,
    loader,
    config: Phase6Config,
    device: torch.device,
) -> float:
    model.eval()
    use_amp = config.amp and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    total_loss   = 0.0
    total_shapes = 0

    for batch in loader:
        H    = batch['H'].to(device)
        T    = batch['T'].to(device)
        W_gt = batch['W_gt'].to(device)

        with amp_ctx:
            W    = model(H, T)
            loss = skinning_loss(W, W_gt)

        total_loss   += loss.item()
        total_shapes += 1

    return total_loss / max(total_shapes, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 6 — Skinning Weight Training')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr',     type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = Phase6Config()
    if args.epochs is not None: config.epochs = args.epochs
    if args.lr     is not None: config.lr     = args.lr
    if args.device is not None: config.device = args.device

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    config.token_dir      = resolve(config.token_dir)
    config.skel_dir       = resolve(config.skel_dir)
    config.train_split    = resolve(config.train_split)
    config.val_split      = resolve(config.val_split)
    config.checkpoint_dir = resolve(config.checkpoint_dir)

    device = torch.device(config.device)
    set_seed(42)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(config)
    print(
        f'Dataset: {len(train_loader.dataset)} train  '
        f'{len(val_loader.dataset)} val'
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = SkinningModule(d=config.d).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'SkinningModule: {total_params:,} parameters  device={device}')

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    # ── AMP scaler ────────────────────────────────────────────────────
    use_amp_actual = config.amp and device.type == 'cuda'
    scaler         = GradScaler('cuda' if use_amp_actual else 'cpu',
                                enabled=use_amp_actual)

    start_epoch   = 0
    best_val_loss = float('inf')

    # ── Resume ────────────────────────────────────────────────────────
    resume_path = args.resume or (config.resume if config.resume else None)
    if resume_path:
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'  Resumed from epoch {ckpt["epoch"]}  val_loss={best_val_loss:.6f}')

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler,
                                 config, device, epoch)

        val_loss = val_epoch(model, val_loader, config, device)
        scheduler.step()

        print(
            f'── Epoch {epoch:>3} complete │ '
            f'train {train_loss:.6f} │ val {val_loss:.6f}'
        )

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(config.checkpoint_dir, f'epoch_{epoch}.pt')
            save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, val_loss)
            print(f'  Saved periodic checkpoint → {path}')

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, scaler, val_loss)
            print(f'  New best val loss: {val_loss:.6f} → {best_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
