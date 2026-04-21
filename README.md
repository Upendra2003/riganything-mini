<img width="908" height="552" alt="intro" src="https://github.com/user-attachments/assets/dbd52c64-c73e-440a-a622-9a2001dc6826" />

Inference Results HTML View: [Here](https://riganything-mini-inference.netlify.app/) 

Based on the [RigNet dataset](https://github.com/zhan-xu/RigNet) (2703 preprocessed 3D character models with ground-truth rigs).

---

## Pipeline Overview

| Phase | Name | Status |
|-------|------|--------|
| **1** | Point Cloud Dataset Generation | [docs/phase_1.md](docs/phase_1.md) |
| **2** | Shape & Skeleton Tokenizers | [docs/phase_2.md](docs/phase_2.md) |
| **3** | Hybrid Attention Transformer | [docs/phase_3.md](docs/phase_3.md) |
| **4** | Joint Diffusion Module | [docs/phase_4.md](docs/phase_4.md) |
| 5 | Connectivity Prediction | [docs/phase_5.md](docs/phase_5.md) |
| 6 | Skinnign Weight Prediction | Not started |
| 7 | End-to-End Training | Not started |
| 8 | Evaluation and Metrics | Not started |

---

## Project Structure

```
riganything-mini/
├── Dataset/                        # RigNet preprocessed dataset
│   ├── obj_remesh/                 # Remeshed OBJ meshes (1K–5K verts)
│   ├── rig_info_remesh/            # Rig info txt files (joints, hierarchy, skinning)
│   ├── pretrain_attention/         # Pre-computed attention supervision (Phase 3/4)
│   ├── volumetric_geodesic/        # Pre-computed geodesic distances (Phase 5)
│   ├── vox/                        # Voxelized models
│   ├── train_final.txt
│   ├── val_final.txt
│   └── test_final.txt
├── pointClouds/
│   └── obj_remesh/                 # Phase 1 output
│       ├── <id>_pointcloud.npy     [1024, 6]  xyz + normals
│       ├── <id>_points.npy         [1024, 3]  positions
│       ├── <id>_normals.npy        [1024, 3]  outward normals
│       ├── <id>_skeleton.npy       [K, 4]     joints + BFS parent index
│       └── <id>_skinning.npy       [V, K]     skinning weights
├── tokens/
│   └── obj_remesh/                 # Phase 2 output
│       ├── <id>_H.pt               [1024, 1024]  shape tokens
│       └── <id>_T.pt               [K, 1024]     skeleton tokens
├── docs/
│   ├── phase_1.md
│   └── phase_2.md
├── dataset.py                      # Phase 1: mesh → point cloud + skeleton NPY
├── tokenizer.py                    # Phase 2: point cloud + skeleton → tokens
├── tests/
│   ├── phase1_test.py              # Phase 1 validation + visualisation
│   └── phase2_test.py              # Phase 2 validation
├── CLAUDE.md
└── README.md
```

---

## Quick Start

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install open3d numpy
```

### Phase 1 — Generate Point Clouds

```bash
# Quick test (5 shapes)
python dataset.py --max_shapes 5

# Full run (all 2703 shapes, resumes safely if interrupted)
python dataset.py --split all --max_shapes 2703 --resume
```

### Phase 1 — Validate & Visualise

```bash
# Runs all checks + saves PNG renders to tests/output/
python tests/phase1_test.py

# Test a specific shape
python tests/phase1_test.py --shape_id 10000
```

See [docs/phase_1.md](docs/phase_1.md) for a full walkthrough.

---

### Phase 2 — Tokenize Shapes & Skeletons

```bash
# Full run (all 2703 shapes, resumes safely if interrupted)
python tokenizer.py --max_shapes 2703 --resume
```

### Phase 2 — Validate

```bash
python tests/phase2_test.py
```

See [docs/phase_2.md](docs/phase_2.md) for a full walkthrough.

---

## Requirements

- Python 3.10+
- `open3d`, `numpy`
