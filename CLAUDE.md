# RigAnything-Mini — Claude Code Notes

## Project Overview
Deep learning pipeline for 3D character rigging prediction (RigNet-style).
Predicts skeleton (joint positions + hierarchy) and skinning weights from a raw 3D mesh.
Based on: Liu et al., ACM Transactions on Graphics, Vol. 44, August 2025.

## Environment
- Python venv at `.venv/` — always use `.venv/bin/python`
- Key packages: `open3d`, `numpy`, `matplotlib`, `torch`
- Blender is NOT installed and NOT needed — rig data comes from `rig_info_remesh/` txt files

## Dataset (`Dataset/`)
| Subfolder | Content |
|-----------|---------|
| `obj_remesh/` | Remeshed OBJ meshes (1K–5K verts) — primary mesh source |
| `rig_info_remesh/` | Rig info txt files (joints, hierarchy, skinning) |
| `obj/` | Original OBJ meshes |
| `rig_info/` | Rig info for original meshes |
| `pretrain_attention/` | Pre-computed attention supervision (Phase 3/4) |
| `volumetric_geodesic/` | Pre-computed geodesic distances (Phase 5) |
| `vox/` | Voxelized models for inside/outside checks |
| `{train,val,test}_final.txt` | Official split lists |

Total shapes: 2703. If OBJ files are empty, dataset download is incomplete.

## Rig Info Format (`rig_info_remesh/<id>.txt`)
```
joints Hips 0.0 0.92 0.0
root   Hips
hier   Hips Spine
skin   0 Hips 0.8 Spine 0.2
```
Parsed by `parse_rig_info()` in `dataset.py` — no Blender required.

## Global Dimension
`d = 1024` is the transformer dimension used throughout all phases. Every token tensor must have this width so phases compose cleanly.

---

## Phase 1 — Data Loading & Point Cloud Generation ✅
**File:** `dataset.py`  
**Input:** Raw OBJ meshes + rig info txt files  
**Output:** `pointClouds/obj_remesh/<id>_*.npy`

| File | Shape | Description |
|------|-------|-------------|
| `<id>_pointcloud.npy` | `[1024, 6]` | xyz + normals — main Phase 2 input |
| `<id>_points.npy` | `[1024, 3]` | xyz positions |
| `<id>_normals.npy` | `[1024, 3]` | outward surface normals |
| `<id>_skeleton.npy` | `[K, 4]` | joint xyz + BFS parent index (1-indexed) |
| `<id>_skinning.npy` | `[V, K]` | dense per-vertex skinning weights |

**Key details:**
- 1024 points sampled via area-weighted triangle sampling
- Skeleton serialized in BFS order — parent index always < joint index
- Root joint parent index = 1 (points to itself, 1-indexed)
- Sibling order randomized at load time for training augmentation (forces model to learn a distribution over orderings, not one fixed sequence)

**Run:**
```bash
.venv/bin/python dataset.py --split all --max_shapes 2703 --resume
.venv/bin/python tests/phase1_test.py
```

---

## Phase 2 — Shape & Skeleton Tokenizers ✅
**File:** `tokenizer.py`  
**Input:** `pointClouds/obj_remesh/<id>_pointcloud.npy` + `<id>_skeleton.npy`  
**Output:** `tokens/obj_remesh/<id>_H.pt` + `<id>_T.pt`

| File | Shape | Description |
|------|-------|-------------|
| `<id>_H.pt` | `[1024, 1024]` | Shape tokens — Phase 3 input |
| `<id>_T.pt` | `[K,    1024]` | Skeleton tokens — Phase 3 input |

**Architecture:**

ShapeTokenizer — point-wise MLP (no inter-point interaction):
```
[1024, 6] → Linear(6→512) → ReLU → Linear(512→1024) → [1024, 1024]
```

SkeletonTokenizer — encodes current joint + parent joint + positional embeddings:
```
joint_mlp:    Linear(3→512) → ReLU → Linear(512→1024)
combiner_mlp: Linear(4096→2048) → ReLU → Linear(2048→1024)
per joint k:
  jk_feat  = joint_mlp(positions[k])           # [1024]
  jpk_feat = joint_mlp(positions[parent_idx])  # [1024]
  gamma_k  = sinusoidal(k,          d=1024)    # [1024]
  gamma_pk = sinusoidal(parent_idx, d=1024)    # [1024]
  T[k] = combiner_mlp(concat([jk_feat, gamma_k, jpk_feat, gamma_pk]))
```
Sinusoidal embedding: `gamma(k)_2i = sin(k/10000^(2i/d))`, `gamma(k)_2i+1 = cos(...)`

**Key details:**
- Models are randomly initialized — Phase 2 is preprocessing only (no training yet)
- Parent index in `_skeleton.npy` is 1-indexed — convert to 0-indexed before array lookup
- Bond direction (parent→child position) encodes limb orientation; including parent position is critical
- Both H and T must be `d=1024` so they can be concatenated in Phase 3's transformer

**Run:**
```bash
.venv/bin/python tokenizer.py --max_shapes 2703 --resume
.venv/bin/python tests/phase2_test.py
```

---

## Phase 3 — Hybrid Attention Transformer ✅
**Files:** `phase3/hybrid_mask.py`, `phase3/transformer.py`, `phase3/config.py`, `phase3/dataset.py`, `phase3/train.py`  
**Input:** `tokens/obj_remesh/<id>_H.pt` + `<id>_T.pt`  
**Output:** Context vectors Z_k `[L+k-1, 1024]` per autoregressive step — **runtime only, never saved to disk**  
**Checkpoint:** `checkpoints/phase3/best_model.pt` (trained transformer weights, 1.7 GB)

**Architecture:** 12 × TransformerBlock, d=1024, n_heads=16, ffn_dim=4096 → **151,154,688 parameters**

**Hybrid attention mask** — the key design decision:
- Shape tokens (positions 0..L-1): full bidirectional self-attention among themselves; **blocked from attending to skeleton tokens** (mask[:L, L:] = -inf). This keeps shape representations prefix-independent, which is required for causal consistency.
- Skeleton tokens (positions L..L+k-2): attend to ALL shape tokens + past skeleton tokens (causal upper triangle = -inf).

```
Sequence: [H_0 ... H_{L-1} | T_0 ... T_{k-2}]
Mask:
  shape→shape:    0      (full bidirectional)
  shape→skeleton: -inf   (blocked — preserves causal consistency)
  skel→shape:     0      (full cross-attention)
  skel→skeleton:  causal (lower-tri=0, upper-tri=-inf)
```

**Proxy loss for standalone training:**
For each autoregressive step k from 2 to K:
  Z_k = transformer(H, T[:k-1]) → MSE(Z_k[L:], T[:k-1])
The transformer should preserve and enrich skeleton token information.

**Key implementation details:**
- Per-step `.backward()` in training loop (frees each computation graph immediately, avoids OOM from holding K graphs)
- AMP (fp16 forward, fp32 loss) with `torch.amp.GradScaler`
- Linear warmup (100 steps) → cosine decay via single `LambdaLR`
- Gradient checkpointing available via `config.use_grad_checkpoint=True`
- Train/val split is seeded — uses `random.Random(seed)` not global RNG

**Run:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase3/train.py --epochs 50
.venv/bin/python tests/phase3_test.py          # 19 tests, all structural (run any time)
# Resume:
.venv/bin/python phase3/train.py --resume checkpoints/phase3/best_model.pt
```

---

## Phase 4 — Joint Diffusion Module 🔜
**Role:** Instead of deterministic regression (which collapses to the mean of sibling positions), model the distribution over next joint positions using a diffusion process conditioned on Z_k.

**Why diffusion:** Sibling joints share the same transformer context. An L2-trained regressor predicts the mean of all valid positions — a point in empty space. Diffusion captures the full multimodal distribution.

**Forward diffusion (cosine noise schedule, M=1000 steps):**
```
beta_s    = cosine schedule (smoother than linear, avoids near-zero alpha_bar)
alpha_bar_m = prod_{s=1}^{m} (1 - beta_s)
j^m = sqrt(alpha_bar_m) * j^0 + sqrt(1 - alpha_bar_m) * epsilon,  epsilon ~ N(0,I)
```

**Training objective** (noise prediction, conditioned on Z_k via AdaLN):
```
L_joint = E_{eps, m} [ ||eps - eps_theta(j^m | m, Z_k)||^2 ]

AdaLN: z_ctx = MeanPool(Z_k)
       [gamma, beta] = Linear(Concat(z_ctx, time_embed(m)))
       AdaLN(x) = (1 + gamma) * LayerNorm(x) + beta

DenoisingMLP:
  x = concat(j^m, time_embed(m))     [3 + d]
  x = SiLU(AdaLN(Linear(x, d)))
  x = SiLU(AdaLN(Linear(x, d)))
  eps_pred = Linear(x, 3)            [3]
```

**Inference:** 50 DDIM steps (deterministic denoising), starting from j^M ~ N(0,I).

**Output:** Predicted joint position j_k ∈ R^3 — **runtime only, never saved**

---

## Phase 5 — Connectivity Prediction 🔜
**Role:** Given predicted joint position j_k, classify which of the k-1 previous joints is its parent. Uses teacher forcing (ground-truth j_k) during training.

**Fusing module** — incorporate j_k into the context:
```
Z'_k = MLP(concat(Z_k.mean(0), j_k, gamma(k)))    [d + 3 + d] → [d]
MLP: Linear(2d+3, 2048) → ReLU → Linear(2048, d)
```

**Connectivity module** — score each candidate parent:
```
score_i = MLP(concat(Z'_k, T_i))    [2d] → [1]   for i in 1..k-1
q_k     = Softmax([score_1, ..., score_{k-1}])    [k-1]
MLP: Linear(2d, 1024) → ReLU → Linear(1024, 1)
```

**Loss** (binary cross-entropy, teacher forced):
```
L_connect = -sum_{i<k} [ y_hat_{k,i} * log(q_{k,i}) + (1 - y_hat_{k,i}) * log(1 - q_{k,i}) ]
```
where y_hat_{k,i} = 1 only for the true parent (exactly one per step).

**Output:** Predicted parent index p_k — **runtime only**  
**Target accuracy:** >90% with teacher forcing, paper reports 96.5%.

---

## Phase 6 — Skinning Prediction 🔜
**Role:** Compute W[L, K] skinning weight matrix — influence of each joint on each surface point. Used for LBS deformation.

**SkinningModule** — pairwise MLP between shape and skeleton tokens:
```
H_exp  = H.unsqueeze(1).expand(L, K, d)     # [L, K, d]
T_exp  = T.unsqueeze(0).expand(L, K, d)     # [L, K, d]
pairs  = concat([H_exp, T_exp], dim=-1)     # [L, K, 2d]
scores = MLP(pairs).squeeze(-1)             # [L, K]
W      = Softmax(scores, dim=-1)            # [L, K] — rows sum to 1

MLP: Linear(2d, 1024) → ReLU → Linear(1024, 1)
```

**Loss** (weighted cross-entropy — gt weights are BOTH the target AND the per-class weight):
```
L_skinning = (1/L) sum_l sum_k [ -w_hat_{l,k} * log(w_{l,k}) ]
```
Strongly-influenced points get higher loss weight, diffuse points get lower weight.

**Why normals help:** Points Euclidean-close but geodesically far (e.g., inner/outer thigh) must get different weights. Normals encode local orientation as a proxy for geodesic distance.

**Output:** `output/pred_skins/<id>_weights.npy` — `[1024, K]` float32

---

## Phase 7 — End-to-End Training 🔜
**Role:** Wire all modules (Phases 2–6) into one training loop with combined loss + online pose augmentation.

**Combined loss** (equally weighted, no lambda hyperparameters):
```
L = L_joint + L_connect + L_skinning
```

**Online pose augmentation** (at every training step):
```
For each joint k: apply random rotation R_k, |angle| ≤ 45°
p'_l = sum_k w_{l,k} * (R_k * (p_l - j_k) + j'_k)    [LBS forward pass]
```
This forces generalization to arbitrary input poses, not just rest poses.

**Full model class:** `RigAnythingModel` composes ShapeTokenizer + SkeletonTokenizer + HybridTransformer + DenoisingMLP + FusingModule + ConnectivityModule + SkinningModule.

**Note:** In Phase 7 the Phase 2 tokenizers are TRAINED jointly (they were fixed/random in Phases 2–3 standalone). All modules train together from a combined loss.

**Hardware:** Single L40 GPU, batch_size=2 → 28–38 GB VRAM.  
**Checkpoint:** `checkpoints/riganything/epoch_N.pt`

---

## Phase 8 — Evaluation 🔜
**Metrics** (match definition from paper Table 3, tau=0.15 normalized units):

| Metric | Definition |
|--------|------------|
| IoU | \|matched\| / (\|pred\| + \|gt\| - \|matched\|) |
| Precision | \|matched pred bones\| / \|total pred bones\| |
| Recall | \|matched gt bones\| / \|total gt bones\| |
| CD-J2J | Bidirectional Chamfer, point-to-point |
| CD-J2B | Bidirectional Chamfer, point-to-segment |
| CD-B2B | Bidirectional Chamfer, segment-to-segment |

A bone pair is matched if min-dist(B_pred, B_gt) < 0.15.

**Targets:**

| | RigNet (baseline to beat) | Paper (full 12K shapes) | Our target (~500 shapes) |
|--|--|--|--|
| IoU | 0.456 | **0.768** | ~0.60–0.65 |
| Precision | 0.424 | 0.789 | — |
| Recall | 0.591 | 0.766 | — |
| CD-J2J | — | 0.034 | — |

**Reuse:** `utils/eval_utils.py` from RigNet repo (`calc_IoU()`, `calc_dist_score()`).

---

## Data Flow Summary

```
Dataset/obj_remesh/*.obj
  └─[Phase 1: dataset.py]──────────────► pointClouds/obj_remesh/{id}_*.npy  (saved)
                                                        │
                                          [Phase 2: tokenizer.py]
                                                        │
                                          tokens/obj_remesh/{id}_H.pt         (saved)
                                          tokens/obj_remesh/{id}_T.pt         (saved)
                                                        │
                                          [Phase 3: HybridTransformer]
                                          Z_k = transformer(H, T_prev)        (runtime only)
                                          checkpoints/phase3/best_model.pt    (saved)
                                                        │
                              ┌───────────────┬─────────┘
                              │               │
                    [Phase 4: DenoisingMLP]  [Phase 6: SkinningModule]
                    j_k ~ diffusion(Z_k)     W = skinner(H, T_all)
                    (runtime only)           output/pred_skins/*.npy (saved)
                              │
                    [Phase 5: FusingModule + ConnectivityModule]
                    p_k = argmax(scorer(Z'_k, T_prev))
                    output/pred_skels/*.json (saved)
                              │
                    [Phase 7: End-to-end training — all modules jointly]
                    checkpoints/riganything/epoch_N.pt
                              │
                    [Phase 8: Evaluation]
                    results/metrics.csv
```

## Checkpoint Quick Reference

| File | Size | Contents |
|------|------|----------|
| `checkpoints/phase3/best_model.pt` | 1.7 GB | Transformer weights (epoch 4, val_loss=0.102) |
| `checkpoints/phase3/epoch_4.pt` | 1.7 GB | Periodic checkpoint (same epoch) |

**Final training result (50 epochs):** train loss 0.000560 / val loss 0.000945 — fully converged.
