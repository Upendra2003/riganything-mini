<img width="908" height="552" alt="intro" src="https://github.com/user-attachments/assets/dbd52c64-c73e-440a-a622-9a2001dc6826" />


Based on the [RigNet dataset](https://github.com/zhan-xu/RigNet) (2703 preprocessed 3D character models with ground-truth rigs).

---

## Pipeline Overview

| Phase | Name | Status |
|-------|------|--------|
| **1** | Point Cloud Dataset Generation | Complete |
| 2 | Point Cloud Tokenization | Not started |
| 3 | Skeleton Joint Prediction | Not started |
| 4 | Bone Connectivity Prediction | Not started |
| 5 | Skinning Weight Prediction | Not started |
| 6 | End-to-End Training | Not started |
| 7 | Evaluation & Metrics | Not started |
| 8 | Inference & Export | Not started |

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
├── docs/
│   └── phase_1.md
├── dataset.py                      # Phase 1: mesh → point cloud + skeleton NPY
├── tests/
│   └── phase1_test.py              # Phase 1 validation + visualisation
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

## Requirements

- Python 3.10+
- `open3d`, `numpy`
