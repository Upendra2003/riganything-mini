"""
Phase 7 — RigAnythingModel: End-to-End Full Model
===================================================
Wires all six submodules into one nn.Module and computes the combined loss:

  L_total = L_joint + L_connect + L_skinning   (equal weights)

Module ownership:
  shape_tok   — ShapeTokenizer     (random init → trained here)
  skel_tok    — SkeletonTokenizer  (random init → trained here)
  transformer — HybridTransformer  (warm-started from phase3 checkpoint)
  denoiser    — DenoisingMLP       (warm-started from phase4 checkpoint)
  fuser       — FusingModule       (warm-started from phase5 checkpoint)
  connector   — ConnectivityModule (warm-started from phase5 checkpoint)
  skinner     — SkinningModule     (warm-started from phase6 checkpoint)

Noise schedule is registered as non-parameter buffers so it moves with the model
to the correct device and is saved/restored with state_dict.

Memory strategy:
  * T_prev tokens are detached after each step — breaks the autoregressive
    gradient chain, bounding the per-step graph to the transformer size.
  * After the loop, T_all is recomputed WITHOUT detaching so that gradients
    flow back through SkeletonTokenizer from the skinning loss.
  * H (shape tokens) is computed once and participates in all K transformer
    calls, so ShapeTokenizer always receives gradients.

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase7/train.py --epochs 50
  .venv/bin/python phase7/inference.py --shape_id <id>
  .venv/bin/python tests/phase7_test.py
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tokenizer              import ShapeTokenizer, SkeletonTokenizer
from phase3.config          import Config as Phase3Config
from phase3.transformer     import HybridTransformer
from phase4.model           import DenoisingMLP
from phase4.noise_schedule  import compute_cosine_schedule, forward_diffuse
from phase5.connectivity    import FusingModule, ConnectivityModule, connectivity_loss
from phase6.model           import SkinningModule, skinning_loss


_p3_defaults = Phase3Config()


class RigAnythingModel(nn.Module):
    """
    Full rigging model.  All submodules are jointly optimised (none frozen).

    Args:
        d        : token / feature dimension (must be 1024 everywhere)
        L        : number of shape tokens (= 1024 sampled surface points)
        n_layers : transformer depth — defaults to phase3/config.py n_layers
        n_heads  : attention heads
        ffn_dim  : transformer FFN hidden dim
        M        : diffusion timesteps
    """

    def __init__(
        self,
        d:        int = _p3_defaults.d,
        L:        int = _p3_defaults.L,
        n_layers: int = _p3_defaults.n_layers,
        n_heads:  int = _p3_defaults.n_heads,
        ffn_dim:  int = _p3_defaults.ffn_dim,
        M:        int = 1000,
    ):
        super().__init__()
        self.d = d
        self.L = L
        self.M = M

        # ── Submodules ────────────────────────────────────────────────
        self.shape_tok   = ShapeTokenizer(d=d)
        self.skel_tok    = SkeletonTokenizer(d=d)

        p3_cfg = Phase3Config(d=d, n_heads=n_heads, ffn_dim=ffn_dim,
                              n_layers=n_layers, L=L)
        self.transformer = HybridTransformer(p3_cfg)
        self.denoiser    = DenoisingMLP(d=d, M=M)
        self.fuser       = FusingModule(d=d)
        self.connector   = ConnectivityModule(d=d)
        self.skinner     = SkinningModule(d=d)

        # ── Noise-schedule buffers (move with model, not learnable) ───
        sched = compute_cosine_schedule(M=M)
        self.register_buffer('alpha_bars',               sched['alpha_bars'])
        self.register_buffer('sqrt_alpha_bars',          sched['sqrt_alpha_bars'])
        self.register_buffer('sqrt_one_minus_alpha_bars',sched['sqrt_one_minus_alpha_bars'])
        self.register_buffer('alphas',                   sched['alphas'])
        self.register_buffer('betas',                    sched['betas'])

    # ------------------------------------------------------------------
    # Internal helper: reconstruct schedule dict from buffers
    # ------------------------------------------------------------------
    def _schedule(self) -> dict:
        return {
            'alpha_bars':               self.alpha_bars,
            'sqrt_alpha_bars':          self.sqrt_alpha_bars,
            'sqrt_one_minus_alpha_bars':self.sqrt_one_minus_alpha_bars,
            'alphas':                   self.alphas,
            'betas':                    self.betas,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        points:     torch.Tensor,   # [L, 3]
        normals:    torch.Tensor,   # [L, 3]
        gt_joints:  torch.Tensor,   # [K, 3]
        gt_parents: torch.Tensor,   # [K]   1-indexed (root=1 self-ref)
        gt_skin:    torch.Tensor,   # [L, K]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, loss_joint, loss_connect, loss_skin) as scalar tensors.
        """
        K      = gt_joints.shape[0]
        device = points.device
        sched  = self._schedule()

        # ── STEP 1: Tokenize shape ─────────────────────────────────────
        pc_in = torch.cat([points, normals], dim=-1)   # [L, 6]
        H     = self.shape_tok(pc_in)                  # [L, d]

        # ── STEP 2: Autoregressive loop (teacher-forced) ───────────────
        # T_prev_detached: used for transformer input (graph-bounded)
        # T_prev_grad:     rebuilt at end with gradient for skel_tok training
        T_prev_detached: list[torch.Tensor] = []

        loss_joint   = torch.zeros((), device=device)
        loss_connect = torch.zeros((), device=device)

        for k in range(1, K + 1):
            # Build T_stack from detached cache
            if T_prev_detached:
                T_stack = torch.stack(T_prev_detached)   # [k-1, d]
            else:
                T_stack = H.new_zeros(0, self.d)

            # Transformer context
            Z_k = self.transformer(H, T_stack)           # [L+k-1, d]

            # ── Joint diffusion loss ──────────────────────────────────
            j0_gt = gt_joints[k - 1]                     # [3]
            m     = random.randint(0, self.M - 1)
            j_m, eps = forward_diffuse(j0_gt, m, sched)
            j_m  = j_m.to(device)
            eps  = eps.to(device)
            eps_pred = self.denoiser(j_m, m, Z_k)
            loss_joint = loss_joint + F.mse_loss(eps_pred, eps) / K

            # ── Connectivity loss (skip root k=1) ─────────────────────
            if k > 1:
                Z_prime  = self.fuser(Z_k, j0_gt, k)
                q_k      = self.connector(Z_prime, T_stack)
                true_idx = int(gt_parents[k - 1].item()) - 1
                true_idx = max(0, min(true_idx, k - 2))
                loss_connect = loss_connect + connectivity_loss(q_k, true_idx) / K

            # ── Teacher-force: compute T_k, detach for next iteration ──
            parent_0idx = int(gt_parents[k - 1].item()) - 1
            parent_0idx = max(0, min(parent_0idx, k - 1))  # clamp to valid range
            parent_pos  = gt_joints[parent_0idx]           # [3]
            # Pass [j_k, parent_pos]; parent_indices=[1,1]: j_k's parent is index 1,
            # parent_pos's parent is itself (index 1) — we only use output[0].
            T_k = self.skel_tok(
                torch.stack([j0_gt, parent_pos]),          # [2, 3]
                torch.tensor([1, 1], device=device),       # both point to index 1
            )[0]                                           # [d]
            T_prev_detached.append(T_k.detach())

        # ── STEP 3: Skinning loss ──────────────────────────────────────
        # Recompute T_all WITHOUT detaching so SkeletonTokenizer receives
        # gradients through the skinning loss path.
        T_all_list = []
        for k in range(K):
            parent_0idx = int(gt_parents[k].item()) - 1
            parent_0idx = max(0, min(parent_0idx, k))
            parent_pos  = gt_joints[parent_0idx]           # [3]
            T_k_grad = self.skel_tok(
                torch.stack([gt_joints[k], parent_pos]),   # [2, 3]
                torch.tensor([1, 1], device=device),       # both point to index 1
            )[0]                                           # [d]
            T_all_list.append(T_k_grad)

        T_all  = torch.stack(T_all_list)                   # [K, d]
        W_pred = self.skinner(H, T_all)                    # [L, K]
        loss_skin = skinning_loss(W_pred, gt_skin)

        # ── STEP 4: Combine ───────────────────────────────────────────
        total_loss = loss_joint + loss_connect + loss_skin
        return total_loss, loss_joint, loss_connect, loss_skin
