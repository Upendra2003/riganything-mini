# RigAnything-Mini — Claude Code Notes

## Project Overview
Deep learning pipeline for 3D character rigging prediction (RigNet-style).
Predicts skeleton (joint positions + hierarchy) and skinning weights from a raw 3D mesh.

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
joints Hips 0.0 0.92 0.0
root   Hips
hier   Hips Spine
skin   0 Hips 0.8 Spine 0.2

Parsed by `parse_rig_info()` in `dataset.py` — no Blender required.

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
- Sibling order randomized at load time for training augmentation

**Run:**
```bash
.venv/bin/python dataset.py --split all --max_shapes 2703 --resume
.venv/bin/python tests/phase1_test.py
```

---

## Phase 2 — Shape & Skeleton Tokenizers ✅
**File:** `phase2_tokenizer.py`  
**Input:** `pointClouds/obj_remesh/<id>_pointcloud.npy` + `<id>_skeleton.npy`  
**Output:** `tokens/obj_remesh/<id>_H.pt` + `<id>_T.pt`

| File | Shape | Description |
|------|-------|-------------|
| `<id>_H.pt` | `[1024, 1024]` | Shape tokens — Phase 3 input |
| `<id>_T.pt` | `[K,    1024]` | Skeleton tokens — Phase 3 input |

**Architecture:**

ShapeTokenizer — point-wise MLP (no inter-point interaction):

[1024, 6] → Linear(6→512) → ReLU → Linear(512→1024) → [1024, 1024]
SkeletonTokenizer — encodes current joint + parent joint + positional embeddings:
joint_mlp:    Linear(3→512) → ReLU → Linear(512→1024)
combiner_mlp: Linear(4096→2048) → ReLU → Linear(2048→1024)
per joint k:
jk_feat  = joint_mlp(positions[k])           # [1024]
jpk_feat = joint_mlp(positions[parent_idx])  # [1024]
gamma_k  = sinusoidal(k,          d=1024)    # [1024]
gamma_pk = sinusoidal(parent_idx, d=1024)    # [1024]
T[k] = combiner_mlp(concat([jk_feat, gamma_k, jpk_feat, gamma_pk]))
Sinusoidal embedding: `gamma(k)_2i = sin(k/10000^(2i/d))`, `gamma(k)_2i+1 = cos(...)`

**Key details:**
- Models are randomly initialized — Phase 2 is preprocessing only (no training yet)
- Parent index in `_skeleton.npy` is 1-indexed — convert to 0-indexed before array lookup
- Both H and T must be `d=1024` so they can be concatenated in Phase 3's transformer
- `d=1024` is the global transformer dimension used throughout all phases

**Run:**
```bash
.venv/bin/python phase2_tokenizer.py --max_shapes 2703 --resume
.venv/bin/python tests/phase2_test.py
```

---

## Phase 3 — Hybrid Attention Transformer 🔜
**Input:** H `[1024, 1024]` + T `[K, 1024]` concatenated → `[1024+K, 1024]`  
Hybrid attention mask: shape tokens use full bidirectional attention; skeleton tokens use causal attention among themselves but attend to all shape tokens.  
**Output:** Context vectors Z_k `[1024+k-1, 1024]` per autoregressive step.

## Phase 4 — Joint Diffusion Module 🔜
Diffusion model conditioned on Z_k. Predicts next joint position as a distribution (not deterministic) to handle sibling ambiguity. Uses cosine noise schedule, 1000 train steps, 50 DDIM inference steps.

## Phase 5 — Connectivity Prediction 🔜
Classifies which previous joint is the parent of the current joint. Uses fusing module + scorer MLP over all candidate parents.

## Phase 6 — Skinning Prediction 🔜
Pairwise MLP between shape tokens H and skeleton tokens T to predict skinning weight matrix `[1024, K]`.

## Phase 7 — End-to-End Training 🔜
Combined loss: L_joint + L_connect + L_skinning. Online pose augmentation via LBS.

## Phase 8 — Evaluation 🔜
Metrics: IoU, Precision, Recall (bone match @ tau=0.15), CD-J2J, CD-J2B, CD-B2B.  
Target: IoU > 0.456 (RigNet baseline). Paper reports 0.768 on full dataset.