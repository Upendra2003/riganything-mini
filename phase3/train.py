"""
Phase 3 — Training Loop
=========================
Trains the Hybrid Attention Transformer using a proxy MSE reconstruction loss.

Proxy loss intuition
--------------------
For each autoregressive step k (1 → K):
  1. Build T_prev = T[:k-1]   (ground-truth skeleton tokens so far)
  2. Run transformer: Z_k = model(H, T_prev)  →  [L+k-1, d]
  3. Extract skeleton portion: Z_skel = Z_k[L:]  →  [k-1, d]
  4. Loss = MSE(Z_skel, T_prev)  —  transformer should preserve & enrich token info
  5. Accumulate: total_loss += loss / K  (normalise by joint count)

This verifies that:
  * Training runs without NaN
  * Gradients flow through all 12 layers
  * Context vectors remain well-scaled

Usage:
  python phase3/train.py
  python phase3/train.py --resume checkpoints/phase3/epoch_10.pt
  python phase3/train.py --epochs 100 --batch_size 4
"""

import os
import sys
import math
import random
import argparse
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

# ── Make project root importable when running as a script ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase3.config import Config
from phase3.transformer import HybridTransformer
from phase3.dataset import make_dataloaders


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay (single LambdaLR)
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    cosine_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / cosine_steps
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler,
                    scaler, val_loss: float, config: Config) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'val_loss':             val_loss,
        'config':               asdict(config),
    }, path)


# ---------------------------------------------------------------------------
# One epoch (train or val)
# ---------------------------------------------------------------------------
def run_epoch(
    model, loader, optimizer, scaler, scheduler, config,
    device, epoch: int, is_train: bool, global_step: int,
) -> tuple[float, int]:
    """
    Returns (avg_loss, updated_global_step).
    """
    L       = config.L
    use_amp = config.use_amp and device.type == 'cuda'
    model.train() if is_train else model.eval()

    epoch_loss    = 0.0
    total_samples = 0

    # For training: do per-step backward to free each forward graph immediately.
    # This avoids holding K computation graphs in memory simultaneously.
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp) if device.type == 'cuda' \
              else torch.amp.autocast('cpu', enabled=False)

    for step, batch in enumerate(loader):
        H       = batch['H'].to(device)    # [B, L, d]
        T_pad   = batch['T'].to(device)    # [B, max_K, d]
        lengths = batch['lengths']         # [B]
        B       = H.shape[0]

        if is_train:
            optimizer.zero_grad()

        batch_loss_value = 0.0

        for b in range(B):
            H_b = H[b]                         # [L, d]
            K_b = int(lengths[b].item())
            T_b = T_pad[b, :K_b]               # [K_b, d]

            for k in range(1, K_b + 1):
                T_prev = T_b[:k - 1]           # [k-1, d]

                if is_train:
                    with amp_ctx:
                        Z_k = model(H_b, T_prev)   # [L+k-1, d]

                    if k > 1:
                        Z_skel = Z_k[L:].float()
                        # Normalise by K_b and B so gradients scale correctly
                        loss = nn.functional.mse_loss(
                            Z_skel, T_prev.detach().float()
                        ) / (K_b * B)
                        scaler.scale(loss).backward()   # frees graph immediately
                        batch_loss_value += loss.item()
                else:
                    with torch.no_grad(), amp_ctx:
                        Z_k = model(H_b, T_prev)

                    if k > 1:
                        Z_skel = Z_k[L:].float()
                        loss = nn.functional.mse_loss(
                            Z_skel, T_prev.detach().float()
                        ) / (K_b * B)
                        batch_loss_value += loss.item()

        if is_train:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch:>3} | Step {step:>4} | '
                f'Loss: {batch_loss_value:.6f} | LR: {lr:.2e}'
            )

        epoch_loss    += batch_loss_value * B
        total_samples += B

    avg_loss = epoch_loss / max(total_samples, 1)
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 3 — Hybrid Transformer Training')
    parser.add_argument('--resume',     type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--epochs',     type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    config = Config()
    if args.epochs     is not None: config.epochs     = args.epochs
    if args.batch_size is not None: config.batch_size = args.batch_size

    # Resolve relative paths against the project root so the script works
    # regardless of the current working directory (e.g. run from phase3/).
    if not os.path.isabs(config.tokens_dir):
        config.tokens_dir = os.path.join(ROOT, config.tokens_dir)
    if not os.path.isabs(config.checkpoint_dir):
        config.checkpoint_dir = os.path.join(ROOT, config.checkpoint_dir)

    device = torch.device(config.device)
    set_seed(config.seed)

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(config)
    print(
        f'Dataset: {len(train_loader.dataset)} train  '
        f'{len(val_loader.dataset)} val  '
        f'(tokens_dir={config.tokens_dir})'
    )

    # ── Model ─────────────────────────────────────────────────────
    model       = HybridTransformer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {total_params:,} parameters  device={device}')

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── LR scheduler (warmup + cosine) ────────────────────────────
    total_steps = config.epochs * max(len(train_loader), 1)
    scheduler   = build_scheduler(optimizer, config.warmup_steps, total_steps)

    # ── AMP scaler ────────────────────────────────────────────────
    use_amp_actual = config.use_amp and device.type == 'cuda'
    scaler         = GradScaler('cuda' if use_amp_actual else 'cpu',
                                enabled=use_amp_actual)

    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    # ── Resume ────────────────────────────────────────────────────
    if args.resume:
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'  Resumed from epoch {ckpt["epoch"]}  val_loss={best_val_loss:.6f}')

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, scaler, scheduler, config,
            device, epoch, is_train=True, global_step=global_step,
        )

        val_loss, _ = run_epoch(
            model, val_loader, optimizer, scaler, scheduler, config,
            device, epoch, is_train=False, global_step=global_step,
        )

        print(
            f'── Epoch {epoch:>3} complete │ '
            f'train {train_loss:.6f} │ val {val_loss:.6f}'
        )

        # Periodic checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            path = os.path.join(config.checkpoint_dir, f'epoch_{epoch}.pt')
            save_checkpoint(path, epoch, model, optimizer, scheduler,
                            scaler, val_loss, config)
            print(f'  Saved periodic checkpoint → {path}')

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(best_path, epoch, model, optimizer, scheduler,
                            scaler, val_loss, config)
            print(f'  New best val loss: {val_loss:.6f} → {best_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
