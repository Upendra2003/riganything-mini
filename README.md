# RigAnything-Mini

A deep learning pipeline for automatic 3D character rigging — predicting skeleton structure and skinning weights from raw 3D mesh geometry.

Based on the [RigNet dataset](https://github.com/zhan-xu/RigNet) (2703 3D character models with ground-truth rigs).

---

## Pipeline Overview

The project is structured as 8 sequential phases. Each phase produces outputs consumed by the next.

| Phase | Name | Status |
|-------|------|--------|
| **1** | Point Cloud Dataset Generation | Partially complete — [see docs](docs/phase_1.md) |
| 2 | Point Cloud Tokenization | Not started |
| 3 | Skeleton Joint Prediction | Not started |
| 4 | Bone Connectivity Prediction | Not started |
| 5 | Skinning Weight Prediction | Not started |
| 6 | End-to-End Training | Not started |
| 7 | Evaluation & Metrics | Not started |
| 8 | Inference & Export | Not started |

---

## Phase Documentation

Detailed notes for each phase live in [`docs/`](docs/):

- [Phase 1 — Point Cloud Dataset Generation](docs/phase_1.md)

---

## Project Structure

```
riganything-mini/
├── RignetDataset/
│   ├── fbx/                    # Raw FBX models (2703 shapes)
│   ├── train_final.txt
│   ├── val_final.txt
│   └── test_final.txt
├── pointClouds/
│   └── fbx/                    # Generated point clouds (Phase 1 output)
│       ├── <id>_pointcloud.npy     [1024, 6]  xyz + normals
│       ├── <id>_points.npy         [1024, 3]  positions
│       ├── <id>_normals.npy        [1024, 3]  outward normals
│       ├── <id>_skeleton.npy       [K, 4]     joints + BFS parent index
│       └── <id>_skinning.npy       [V, K]     skinning weights
├── docs/
│   └── phase_1.md              # Phase 1 notes and walkthrough
├── phase1_dataset.py           # Phase 1 script
├── main.py                     # Training entrypoint
├── dataset_explorer.py         # Visualisation utilities
├── CLAUDE.md                   # Claude Code project notes
└── README.md
```

---

## Quick Start

### Phase 1 — Generate Point Clouds

```bash
# Install dependencies
pip install open3d trimesh fbxloader numpy

# Convert all shapes (resumes safely if interrupted)
python phase1_dataset.py --split all --max_shapes 2703 --resume

# Default run (500 shapes)
python phase1_dataset.py
```

See [docs/phase_1.md](docs/phase_1.md) for a full walkthrough of the Phase 1 code.

---

## Requirements

- Python 3.10+
- `open3d`, `trimesh`, `fbxloader`, `numpy`
- Blender (optional — required only for skeleton/skinning extraction)
