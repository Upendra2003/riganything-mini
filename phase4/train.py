"""
Phase 4 — Joint Diffusion Module Training
==========================================
Trains the DenoisingMLP to predict noise in joint positions, conditioned on
Z_k context vectors produced by the (frozen) Phase 3 transformer.

Training design:
  * Phase 3 transformer is loaded and immediately frozen (eval + no_grad).
  * Per-joint inner loop: for each shape, iterate k = 1..K, generating Z_k
    via the frozen transformer and computing diffusion loss for joint k.
  * Gradients are accumulated across all K joints in a shape; optimizer steps
    once per shape.
  * AMP (fp16 forward, fp32 loss) with GradScaler, same pattern as Phase 3.
  * Linear LR warmup (100 steps) → cosine decay.
  * Validation: sample 5 random joints × 5 timesteps per shape, report MSE.
- AdaLN uses (1 + gamma) * LN(x) + beta with zero-initialized projection — starts as pure LayerNorm, learns deviations
  - Cosine schedule with s=0.008 offset; alpha_bars[0] ≈ 1.0, alpha_bars[999] < 0.01
  - Per-joint inner loop: zero_grad once per shape, loss.backward() after each joint k, optimizer.step() after all K joints — avoids holding K computation graphs
  simultaneously
  - Frozen transformer: transformer.eval() + requires_grad_(False) + all Z_k generation wrapped in torch.no_grad()
  - DDIM sampler: 50 steps, deterministic denoising, j0_est clamped to [-3, 3] for numerical stability
  - Validation: 5 random joints × 5 random timesteps per shape — fast but representative
Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase4/train.py --epochs 50
  .venv/bin/python phase4/train.py --resume checkpoints/phase4/best_model.pt
"""

import os
import sys
import math
import random
import argparse
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

# ── Make project root importable when running as a script ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase3.config      import Config as Phase3Config
from phase3.transformer import HybridTransformer
from phase4.config      import Phase4Config
from phase4.model       import DenoisingMLP
from phase4.noise_schedule import compute_cosine_schedule, forward_diffuse
from phase4.dataset     import make_dataloaders


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
# Load and freeze Phase 3 transformer
# ---------------------------------------------------------------------------
def load_frozen_transformer(ckpt_path: str, device: torch.device) -> HybridTransformer:
    """Load Phase 3 checkpoint and freeze all parameters."""
    p3_cfg = Phase3Config()
    transformer = HybridTransformer(p3_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt['model_state_dict'])

    transformer.eval()
    transformer.requires_grad_(False)

    print(f'Loaded frozen Phase 3 transformer from {ckpt_path}')
    frozen_params = sum(p.numel() for p in transformer.parameters())
    print(f'  Frozen parameters: {frozen_params:,}  (all requires_grad=False)')
    return transformer


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    epoch: int,
    denoiser: DenoisingMLP,
    optimizer,
    scheduler,
    scaler,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     denoiser.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'val_loss':             val_loss,
    }, path)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------
def train_epoch(
    transformer: HybridTransformer,
    denoiser: DenoisingMLP,
    loader,
    optimizer,
    scaler: GradScaler,
    scheduler,
    schedule: dict,
    config: Phase4Config,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    """
    Returns (avg_loss_per_joint, updated_global_step).

    One optimizer step per shape (accumulate grads over all K joints first).
    """
    denoiser.train()
    use_amp = config.amp and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    epoch_loss    = 0.0
    total_joints  = 0

    for step, batch in enumerate(loader):
        H       = batch['H'].to(device)        # [1024, 1024]
        T       = batch['T'].to(device)        # [K, 1024]
        joints  = batch['joints'].to(device)   # [K, 3]
        K       = batch['K']
        if isinstance(K, torch.Tensor):
            K = int(K.item())

        optimizer.zero_grad()

        shape_loss = 0.0

        for k in range(1, K + 1):
            # Build T_prev = T[0..k-2]; empty at k=1
            T_prev = T[:k - 1]  # [k-1, 1024]; shape [0, 1024] when k=1

            # Generate Z_k with frozen transformer
            with torch.no_grad():
                Z_k = transformer(H, T_prev)    # [1024 + k-1, 1024]

            # Ground-truth joint (0-indexed: joint k is joints[k-1])
            j0 = joints[k - 1]                 # [3]

            # Sample random timestep
            m = random.randint(0, config.M - 1)

            # Forward diffuse
            j_m, eps = forward_diffuse(j0, m, schedule)
            j_m = j_m.to(device)
            eps = eps.to(device)

            # Predict noise
            with amp_ctx:
                eps_pred = denoiser(j_m, m, Z_k)
                loss = F.mse_loss(eps_pred, eps) / K

            # Backward immediately (frees computation graph, avoids OOM)
            scaler.scale(loss).backward()
            shape_loss += loss.item()

        # Gradient step after processing all K joints for this shape
        scaler.unscale_(optimizer)
        clip_grad_norm_(denoiser.parameters(), max_norm=config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        epoch_loss   += shape_loss
        total_joints += K

        lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch:>3} | Step {step:>4} | '
            f'Loss: {shape_loss:.6f} | LR: {lr:.2e}'
        )

    avg_loss = epoch_loss / max(total_joints, 1)
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_epoch(
    transformer: HybridTransformer,
    denoiser: DenoisingMLP,
    loader,
    schedule: dict,
    config: Phase4Config,
    device: torch.device,
) -> float:
    """
    For each shape, sample 5 random joints × 5 random timesteps.
    Returns mean MSE loss over all sampled (joint, timestep) pairs.
    """
    denoiser.eval()
    use_amp = config.amp and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    total_loss  = 0.0
    total_pairs = 0
    N_JOINTS    = 64
    N_STEPS     = 64

    for batch in loader:
        H      = batch['H'].to(device)
        T      = batch['T'].to(device)
        joints = batch['joints'].to(device)
        K      = batch['K']
        if isinstance(K, torch.Tensor):
            K = int(K.item())

        # Sample joints and timesteps
        joint_indices = random.choices(range(K), k=N_JOINTS)
        timesteps     = [random.randint(0, config.M - 1) for _ in range(N_STEPS)]

        for k_idx in joint_indices:
            k      = k_idx + 1       # k is 1-indexed (joint k → joints[k-1])
            T_prev = T[:k - 1]       # [k-1, 1024]

            with torch.no_grad():
                Z_k = transformer(H, T_prev)

            j0 = joints[k_idx]       # [3]

            for m in timesteps:
                j_m, eps = forward_diffuse(j0, m, schedule)
                j_m = j_m.to(device)
                eps = eps.to(device)

                with amp_ctx:
                    eps_pred = denoiser(j_m, m, Z_k)
                    loss = F.mse_loss(eps_pred, eps)

                total_loss  += loss.item()
                total_pairs += 1

    return total_loss / max(total_pairs, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 4 — Joint Diffusion Module Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = Phase4Config()
    if args.epochs is not None: config.epochs = args.epochs
    if args.device is not None: config.device = args.device

    # Resolve relative paths against project root
    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    config.phase3_ckpt    = resolve(config.phase3_ckpt)
    config.token_dir      = resolve(config.token_dir)
    config.skel_dir       = resolve(config.skel_dir)
    config.train_split    = resolve(config.train_split)
    config.val_split      = resolve(config.val_split)
    config.checkpoint_dir = resolve(config.checkpoint_dir)

    device = torch.device(config.device)
    set_seed(42)

    # ── Noise schedule (computed once, stored on CPU) ─────────────────
    schedule = compute_cosine_schedule(M=config.M)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(config)
    print(
        f'Dataset: {len(train_loader.dataset)} train  '
        f'{len(val_loader.dataset)} val'
    )

    # ── Frozen Phase 3 transformer ────────────────────────────────────
    transformer = load_frozen_transformer(config.phase3_ckpt, device)

    # ── Denoiser ──────────────────────────────────────────────────────
    denoiser = DenoisingMLP(d=config.d, M=config.M).to(device)
    total_params = sum(p.numel() for p in denoiser.parameters())
    print(f'DenoisingMLP: {total_params:,} parameters  device={device}')

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
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
        denoiser.load_state_dict(ckpt['model_state_dict'])
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
            transformer, denoiser, train_loader,
            optimizer, scaler, scheduler, schedule,
            config, device, epoch, global_step,
        )

        val_loss = val_epoch(
            transformer, denoiser, val_loader,
            schedule, config, device,
        )

        print(
            f'── Epoch {epoch:>3} complete │ '
            f'train {train_loss:.6f} │ val {val_loss:.6f}'
        )

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(config.checkpoint_dir, f'epoch_{epoch}.pt')
            save_checkpoint(path, epoch, denoiser, optimizer, scheduler,
                            scaler, val_loss)
            print(f'  Saved periodic checkpoint → {path}')

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(best_path, epoch, denoiser, optimizer, scheduler,
                            scaler, val_loss)
            print(f'  New best val loss: {val_loss:.6f} → {best_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
