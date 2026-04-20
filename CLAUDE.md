# RigAnything-Mini — Claude Code Notes

## Project Overview
Deep learning pipeline for 3D character rigging prediction (RigNet-style).
Predicts skeleton (joint positions + hierarchy) and skinning weights from a raw 3D mesh.
Based on: Liu et al., ACM Transactions on Graphics, Vol. 44, August 2025.

## Environment
- Python venv at `.venv/` — always use `.venv/bin/python`
- Key packages: `open3d`, `numpy`, `matplotlib`, `torch`
- Blender is NOT installed — rig data comes from `rig_info_remesh/` txt files

## Dataset (`Dataset/`)
| Subfolder | Content |
|-----------|---------|
| `obj_remesh/` | Remeshed OBJ meshes (1K–5K verts) — primary mesh source |
| `rig_info_remesh/` | Rig info txt files (joints, hierarchy, skinning) |
| `pretrain_attention/` | Pre-computed attention supervision |
| `volumetric_geodesic/` | Pre-computed geodesic distances |
| `vox/` | Voxelized models for inside/outside checks |
| `{train,val,test}_final.txt` | Official split lists |

Total shapes: 2703. Split files use bare IDs (no `.obj` suffix after stripping).

## Rig Info Format (`rig_info_remesh/<id>.txt`)
```
joints Hips 0.0 0.92 0.0
root   Hips
hier   Hips Spine
skin   0 Hips 0.8 Spine 0.2
```
Parsed by `parse_rig_info()` in `dataset.py`.

## Global Dimension
`d = 1024` throughout all phases. Every token tensor must have this width so phases compose cleanly.

---

## Phase 1 — Data Loading & Point Cloud Generation ✅
**File:** `dataset.py`
**Input:** `Dataset/obj_remesh/*.obj` + `Dataset/rig_info_remesh/<id>.txt`
**Output:** `pointClouds/obj_remesh/<id>_*.npy`

| File | Shape | Description |
|------|-------|-------------|
| `<id>_pointcloud.npy` | `[1024, 6]` | xyz + normals — Phase 2 input |
| `<id>_skeleton.npy` | `[K, 4]` | joint xyz + BFS parent index (1-indexed) |
| `<id>_skinning.npy` | `[V, K]` | dense per-vertex skinning weights |

- Skeleton serialized in BFS order; root parent index = 1 (self-referential, 1-indexed)
- Sibling order randomized per load — augmentation so model learns a distribution, not a fixed sequence

```bash
.venv/bin/python dataset.py --split all --max_shapes 2703 --resume
.venv/bin/python tests/phase1_test.py
```

---

## Phase 2 — Shape & Skeleton Tokenizers ✅
**File:** `tokenizer.py`
**Input:** `<id>_pointcloud.npy` + `<id>_skeleton.npy`
**Output:** `tokens/obj_remesh/<id>_H.pt` `[1024, 1024]`, `<id>_T.pt` `[K, 1024]`

- **ShapeTokenizer:** point-wise MLP, no inter-point interaction — `[1024,6] → Linear(6→512) → ReLU → Linear(512→1024)`
- **SkeletonTokenizer:** encodes joint + parent position + sinusoidal positional embeddings for both — parent position encodes limb orientation
- Parent index in `_skeleton.npy` is 1-indexed; convert to 0-indexed for array lookup
- Tokenizers are randomly initialized here; they are trained jointly in Phase 7

```bash
.venv/bin/python tokenizer.py --max_shapes 2703 --resume
.venv/bin/python tests/phase2_test.py
```

---

## Phase 3 — Hybrid Attention Transformer ✅
**Files:** `phase3/`
**Input:** `<id>_H.pt` + `<id>_T.pt`
**Output:** `Z_k [L+k-1, 1024]` per autoregressive step — runtime only
**Checkpoint:** `checkpoints/phase3/best_model.pt` (1.7 GB, val_loss=0.000945)

12 × TransformerBlock, d=1024, n_heads=16, ffn_dim=4096 → **151,154,688 parameters**

**Hybrid mask** (the key design decision):
```
Sequence: [H_0 … H_{L-1} | T_0 … T_{k-2}]
  shape → shape:    0       full bidirectional
  shape → skeleton: -inf    blocked (keeps shape reps prefix-independent)
  skel  → shape:    0       full cross-attention
  skel  → skeleton: causal  (lower-tri=0, upper-tri=-inf)
```

**Proxy loss:** `MSE(Z_k[L:], T[:k-1])` — transformer should preserve and enrich skeleton tokens.
Per-step `.backward()` frees each graph immediately to avoid OOM.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase3/train.py --epochs 50
.venv/bin/python tests/phase3_test.py
# Resume:
.venv/bin/python phase3/train.py --resume checkpoints/phase3/best_model.pt
```

---

## Phase 4 — Joint Diffusion Module ✅
**Files:** `phase4/`
**Input:** frozen Z_k from Phase 3 + ground-truth joint positions
**Output:** predicted joint position j_k ∈ R³ — runtime only
**Checkpoint:** `checkpoints/phase4/best_model.pt` (~40 MB)

**Why diffusion:** Sibling joints share the same Z_k context. An L2 regressor predicts their mean — a point in empty space. Diffusion captures the full multimodal distribution.

**Cosine schedule** (M=1000, offset s=0.008):
```
alpha_bar_t = f(t)/f(0),  f(t) = cos(((t/M + s)/(1+s)) * π/2)²
j^m = sqrt(ᾱ_m)·j⁰ + sqrt(1−ᾱ_m)·ε,   ε ~ N(0,I)
```

**DenoisingMLP** — predicts ε given (j^m, m, Z_k):
```
t_emb     = time_embed(sinusoidal(m/M))          [d]   2-layer MLP
z_ctx     = context_proj(Z_k.mean(0))            [d]
condition = t_emb + z_ctx                        [d]

x = fc1(concat(j^m, t_emb))                      [d]
x = SiLU(AdaLN(x, condition))
x = SiLU(AdaLN(fc2(x), condition))
ε_pred = out(x)                                  [3]
```
AdaLN: `(1 + γ) * LN(x) + β`, projection zero-initialized → identity at start.
Inference: 50 DDIM steps from j^M ~ N(0,I). j0_est clamped to [-3, 3].

Phase 3 transformer is fully frozen during Phase 4 training (`eval()` + `requires_grad_(False)`).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase4/train.py --epochs 50
.venv/bin/python tests/phase4_test.py
# Resume:
.venv/bin/python phase4/train.py --resume checkpoints/phase4/best_model.pt
# Cluster (SLURM):
sbatch slurm/phase4_train.sh
```

---

## Phase 5 — Connectivity Prediction 🔜
**Role:** Given predicted j_k, classify which of the k-1 previous joints is its parent (teacher forcing during training).

**Fusing module** — incorporate j_k into the context:
```
Z'_k = MLP(concat(Z_k.mean(0), j_k, γ(k)))    [2d+3 → d]
MLP: Linear(2d+3, 2048) → ReLU → Linear(2048, d)
```

**Connectivity module** — score each candidate parent:
```
score_i = MLP(concat(Z'_k, T_i))    [2d → 1],   i in 1..k-1
q_k     = Softmax([score_1, …, score_{k-1}])
MLP: Linear(2d, 1024) → ReLU → Linear(1024, 1)
```

**Loss** (BCE, teacher forced, one true parent per step):
```
L_connect = −∑_{i<k} [ ŷ_{k,i}·log(q_{k,i}) + (1−ŷ_{k,i})·log(1−q_{k,i}) ]
```

**Output:** parent index p_k — runtime only. Target accuracy: >90% (paper: 96.5%).

---

## Phase 6 — Skinning Prediction 🔜
**Role:** Compute W `[L, K]` skinning weight matrix for LBS deformation.

**SkinningModule** — pairwise MLP over shape × skeleton tokens:
```
pairs  = concat(H.unsqueeze(1).expand(L,K,d), T.unsqueeze(0).expand(L,K,d))  [L,K,2d]
W      = Softmax(MLP(pairs).squeeze(-1), dim=-1)    [L,K],  rows sum to 1
MLP: Linear(2d, 1024) → ReLU → Linear(1024, 1)
```

**Loss** (weighted cross-entropy — gt weights are both target and per-class weight):
```
L_skinning = (1/L) ∑_l ∑_k [ −ŵ_{l,k} · log(w_{l,k}) ]
```

**Output:** `output/pred_skins/<id>_weights.npy` — `[1024, K]` float32.
Normals in H are critical: points geodesically far but Euclidean-close (e.g., inner/outer thigh) get different weights via orientation cue.

---

## Phase 7 — End-to-End Training 🔜
**Role:** Joint training of all modules (Phases 2–6) with combined loss + online pose augmentation.

```
L = L_joint + L_connect + L_skinning   (equally weighted)
```

**Online pose augmentation** (every step, forces pose generalization):
```
For each joint k: random rotation R_k, |angle| ≤ 45°
p'_l = ∑_k w_{l,k} · (R_k · (p_l − j_k) + j'_k)
```

**Full model class:** `RigAnythingModel` — ShapeTokenizer + SkeletonTokenizer + HybridTransformer + DenoisingMLP + FusingModule + ConnectivityModule + SkinningModule.
Phase 2 tokenizers are trained jointly here (random-init in Phases 2–3 standalone).

**Hardware:** Single L40 GPU, batch_size=2 → 28–38 GB VRAM.
**Checkpoint:** `checkpoints/riganything/epoch_N.pt`

---

## Phase 8 — Evaluation 🔜
Metrics from paper Table 3, τ=0.15 normalized units. Bone pair matched if min-dist < 0.15.

| Metric | Definition |
|--------|------------|
| IoU | \|matched\| / (\|pred\| + \|gt\| − \|matched\|) |
| Precision | \|matched pred bones\| / \|total pred bones\| |
| Recall | \|matched gt bones\| / \|total gt bones\| |
| CD-J2J | Bidirectional Chamfer, point-to-point |
| CD-J2B | Bidirectional Chamfer, point-to-segment |
| CD-B2B | Bidirectional Chamfer, segment-to-segment |

| | RigNet | Paper (12K) | Our target (~500) |
|--|--|--|--|
| IoU | 0.456 | **0.768** | ~0.60–0.65 |
| Precision | 0.424 | 0.789 | — |
| Recall | 0.591 | 0.766 | — |

**Reuse:** `utils/eval_utils.py` from RigNet repo (`calc_IoU()`, `calc_dist_score()`).

---

## Data Flow

```
Dataset/obj_remesh/*.obj
  └─[Phase 1]──► pointClouds/obj_remesh/{id}_*.npy
                    └─[Phase 2]──► tokens/obj_remesh/{id}_H.pt  [1024,1024]
                                   tokens/obj_remesh/{id}_T.pt  [K,1024]
                                     └─[Phase 3]──► Z_k [L+k-1,1024]  (runtime)
                                                      ├─[Phase 4]──► j_k [3]       (runtime)
                                                      │               └─[Phase 5]──► p_k int   (runtime)
                                                      └─[Phase 6]──► W [L,K]
                                                                    output/pred_skins/*.npy
                                     [Phase 7: joint training]──► checkpoints/riganything/epoch_N.pt
                                     [Phase 8: evaluation]──► results/metrics.csv
```

## Checkpoints

| File | Size | Contents |
|------|------|----------|
| `checkpoints/phase3/best_model.pt` | 1.7 GB | HybridTransformer (epoch 4, val_loss=0.000945) |
| `checkpoints/phase4/best_model.pt` | ~40 MB | DenoisingMLP (best val loss) |
| `checkpoints/phase4/epoch_N.pt` | ~40 MB | Periodic checkpoint every 5 epochs |
