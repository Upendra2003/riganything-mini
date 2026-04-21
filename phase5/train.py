"""
Phase 5 — Connectivity Prediction Training
==========================================
Trains FusingModule + ConnectivityModule to predict the parent joint at each
autoregressive step, conditioned on frozen Phase 3 context vectors and
ground-truth joint positions (teacher forcing).

Training design:
  * Phase 3 transformer and Phase 4 denoiser are both fully frozen.
  * Teacher forcing: ground-truth j_k_gt used instead of DDIM samples.
  * Per-step backward to keep memory flat (same pattern as Phase 3/4).
  * Optimizer step once per shape after accumulating grads over all k steps.
  * Shapes with K < 2 are skipped (no connectivity steps to predict).
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (set before launching)

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase5/train.py --epochs 30
  .venv/bin/python phase5/train.py --resume checkpoints/phase5/best_model.pt
"""

import os
import sys
import math
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

from phase3.config       import Config as Phase3Config
from phase3.transformer  import HybridTransformer
from phase5.config       import Phase5Config
from phase5.connectivity import FusingModule, ConnectivityModule, connectivity_loss

# Reuse Phase 4 dataset (same inputs: H, T, joints, parents, K)
from phase4.dataset import Phase4Dataset, _collate_single
from torch.utils.data import DataLoader


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
# LR schedule: linear warmup → cosine decay
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
# Load and freeze Phase 3 transformer
# ---------------------------------------------------------------------------
def load_frozen_transformer(ckpt_path: str, device: torch.device) -> HybridTransformer:
    p3_cfg      = Phase3Config()
    transformer = HybridTransformer(p3_cfg).to(device)
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt['model_state_dict'])
    transformer.eval()
    transformer.requires_grad_(False)
    frozen_params = sum(p.numel() for p in transformer.parameters())
    print(f'Loaded frozen Phase 3 transformer from {ckpt_path}')
    print(f'  Frozen parameters: {frozen_params:,}  (requires_grad=False)')
    return transformer


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    epoch: int,
    fuser: FusingModule,
    connector: ConnectivityModule,
    optimizer,
    scheduler,
    scaler,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'fuser_state_dict':     fuser.state_dict(),
        'connector_state_dict': connector.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'val_loss':             val_loss,
    }, path)


# ---------------------------------------------------------------------------
# DataLoaders (reuse Phase 4 dataset)
# ---------------------------------------------------------------------------
def make_dataloaders(config: Phase5Config):
    train_ds = Phase4Dataset(config.train_split, config.token_dir, config.skel_dir)
    val_ds   = Phase4Dataset(config.val_split,   config.token_dir, config.skel_dir)
    pin      = (config.device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=pin, collate_fn=_collate_single)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=pin, collate_fn=_collate_single)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------
def train_epoch(
    transformer: HybridTransformer,
    fuser:       FusingModule,
    connector:   ConnectivityModule,
    loader,
    optimizer,
    scaler:      GradScaler,
    scheduler,
    config:      Phase5Config,
    device:      torch.device,
    epoch:       int,
    global_step: int,
) -> tuple[float, int]:
    fuser.train()
    connector.train()
    use_amp = config.amp and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    epoch_loss    = 0.0
    total_steps_k = 0

    for step, batch in enumerate(loader):
        H       = batch['H'].to(device)       # [1024, 1024]
        T       = batch['T'].to(device)       # [K, 1024]
        joints  = batch['joints'].to(device)  # [K, 3]
        parents = batch['parents'].to(device) # [K]   1-indexed
        K       = batch['K']
        if isinstance(K, torch.Tensor):
            K = int(K.item())

        # Skip shapes with no connectivity steps
        if K < 2:
            continue

        optimizer.zero_grad()
        shape_loss = 0.0

        for k in range(2, K + 1):
            T_prev = T[:k - 1]          # [k-1, 1024]
            j_k_gt = joints[k - 1]      # [3] ground-truth position (teacher forcing)

            # Z_k from frozen Phase 3 transformer
            with torch.no_grad():
                Z_k = transformer(H, T_prev)   # [1024 + k-1, 1024]

            # true_parent_idx: 1-indexed parent from file → 0-indexed into T_prev
            # skeleton[k-1, 3] is 1-indexed parent of the k-th joint (1-based k)
            # Converting to 0-indexed: subtract 1.
            # T_prev = T[:k-1], so T_prev[p] = T[p] for parent joint p (0-indexed).
            true_parent_idx = int(parents[k - 1].item()) - 1

            # Clamp to valid range in case of file anomalies
            true_parent_idx = max(0, min(true_parent_idx, k - 2))

            with amp_ctx:
                Z_prime = fuser(Z_k, j_k_gt, k)        # [d]
                q_k     = connector(Z_prime, T_prev)    # [k-1]
                loss    = connectivity_loss(q_k, true_parent_idx) / K

            scaler.scale(loss).backward()
            shape_loss    += loss.item()
            total_steps_k += 1

        scaler.unscale_(optimizer)
        clip_grad_norm_(
            list(fuser.parameters()) + list(connector.parameters()),
            max_norm=config.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        epoch_loss += shape_loss
        lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch:>3} | Step {step:>4} | '
            f'Loss: {shape_loss:.6f} | LR: {lr:.2e}'
        )

    avg_loss = epoch_loss / max(total_steps_k, 1)
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_epoch(
    transformer: HybridTransformer,
    fuser:       FusingModule,
    connector:   ConnectivityModule,
    loader,
    config:      Phase5Config,
    device:      torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    fuser.eval()
    connector.eval()
    use_amp = config.amp and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    total_loss    = 0.0
    total_correct = 0
    total_steps_k = 0

    for batch in loader:
        H       = batch['H'].to(device)
        T       = batch['T'].to(device)
        joints  = batch['joints'].to(device)
        parents = batch['parents'].to(device)
        K       = batch['K']
        if isinstance(K, torch.Tensor):
            K = int(K.item())

        if K < 2:
            continue

        for k in range(2, K + 1):
            T_prev = T[:k - 1]
            j_k_gt = joints[k - 1]

            Z_k = transformer(H, T_prev)

            true_parent_idx = int(parents[k - 1].item()) - 1
            true_parent_idx = max(0, min(true_parent_idx, k - 2))

            with amp_ctx:
                Z_prime = fuser(Z_k, j_k_gt, k)
                q_k     = connector(Z_prime, T_prev)
                loss    = connectivity_loss(q_k, true_parent_idx) / K

            total_loss    += loss.item()
            total_steps_k += 1

            pred = int(q_k.argmax().item())
            if pred == true_parent_idx:
                total_correct += 1

    avg_loss = total_loss / max(total_steps_k, 1)
    accuracy = total_correct / max(total_steps_k, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 5 — Connectivity Prediction Training')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr',     type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = Phase5Config()
    if args.epochs is not None: config.epochs = args.epochs
    if args.lr     is not None: config.lr     = args.lr
    if args.device is not None: config.device = args.device

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    config.phase3_ckpt    = resolve(config.phase3_ckpt)
    config.phase4_ckpt    = resolve(config.phase4_ckpt)
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

    # ── Frozen Phase 3 transformer ────────────────────────────────────
    transformer = load_frozen_transformer(config.phase3_ckpt, device)

    # ── Phase 5 models ────────────────────────────────────────────────
    fuser     = FusingModule(d=config.d).to(device)
    connector = ConnectivityModule(d=config.d).to(device)

    fuser_params     = sum(p.numel() for p in fuser.parameters())
    connector_params = sum(p.numel() for p in connector.parameters())
    print(f'FusingModule:       {fuser_params:,} parameters')
    print(f'ConnectivityModule: {connector_params:,} parameters')
    print(f'Total trainable:    {fuser_params + connector_params:,}')

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(fuser.parameters()) + list(connector.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── LR scheduler ──────────────────────────────────────────────────
    total_steps = config.epochs * max(len(train_loader), 1)
    scheduler   = build_scheduler(optimizer, config.warmup_steps, total_steps)

    # ── AMP scaler ────────────────────────────────────────────────────
    use_amp_actual = config.amp and device.type == 'cuda'
    scaler         = GradScaler('cuda' if use_amp_actual else 'cpu',
                                enabled=use_amp_actual)

    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    # ── Resume ────────────────────────────────────────────────────────
    resume_path = args.resume or (config.resume if config.resume else None)
    if resume_path:
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        fuser.load_state_dict(ckpt['fuser_state_dict'])
        connector.load_state_dict(ckpt['connector_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'  Resumed from epoch {ckpt["epoch"]}  val_loss={best_val_loss:.6f}')

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        train_loss, global_step = train_epoch(
            transformer, fuser, connector,
            train_loader, optimizer, scaler, scheduler,
            config, device, epoch, global_step,
        )

        val_loss, val_acc = val_epoch(
            transformer, fuser, connector,
            val_loader, config, device,
        )

        print(
            f'── Epoch {epoch:>3} complete │ '
            f'train {train_loss:.6f} │ val {val_loss:.6f} │ '
            f'val_acc {val_acc:.4f}'
        )

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(config.checkpoint_dir, f'epoch_{epoch}.pt')
            save_checkpoint(path, epoch, fuser, connector, optimizer, scheduler,
                            scaler, val_loss)
            print(f'  Saved periodic checkpoint → {path}')

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(best_path, epoch, fuser, connector, optimizer, scheduler,
                            scaler, val_loss)
            print(f'  New best val loss: {val_loss:.6f} → {best_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
