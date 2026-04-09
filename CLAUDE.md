# RigAnything-Mini — Claude Code Notes

## Project Overview
Deep learning pipeline for 3D character rigging prediction (RigNet-style).
- **Phase 1** (`dataset.py`): OBJ mesh → point cloud + skeleton NPY files
- **Phase 2** (`phase2_tokenizer.py`): tokenizes point clouds for transformer input

## Environment
- Python venv at `.venv/` — always use `.venv/bin/python`
- Key packages: `open3d`, `numpy`, `matplotlib`
- Blender is NOT installed and NOT needed — rig data comes from `rig_info_remesh/` txt files

## Dataset
Primary dataset: `Dataset/` (RigNet preprocessed)

| Subfolder | Content |
|-----------|---------|
| `obj_remesh/` | Remeshed OBJ meshes (1K–5K verts each) — primary mesh source |
| `rig_info_remesh/` | Rig info txt files (joints, hierarchy, skinning) |
| `obj/` | Original (non-remeshed) OBJ meshes |
| `rig_info/` | Rig info for original meshes |
| `pretrain_attention/` | Pre-computed attention supervision (Phase 3/4) |
| `volumetric_geodesic/` | Pre-computed geodesic distances (Phase 5) |
| `vox/` | Voxelized models for inside/outside checks |
| `{train,val,test}_final.txt` | Official split lists |

Total shapes: 2703. If OBJ files are empty, the dataset download is incomplete.

## Running Phase 1
```bash
# Full run (all 2703 shapes, resume-safe)
.venv/bin/python dataset.py --split all --max_shapes 2703 --resume

# Quick test
.venv/bin/python dataset.py --max_shapes 5

# With explicit dataset path (if needed)
.venv/bin/python dataset.py --dataset_dir Dataset --max_shapes 5
```

## Testing Phase 1
```bash
# Run all tests + generate visualisation images in tests/output/
.venv/bin/python tests/phase1_test.py

# Test a specific shape ID
.venv/bin/python tests/phase1_test.py --shape_id 10000
```

## Output Format (per shape in `pointClouds/obj_remesh/`)
| File | Shape | Description |
|------|-------|-------------|
| `<id>_pointcloud.npy` | `[1024, 6]` | xyz + normals — main Phase 2 input |
| `<id>_points.npy` | `[1024, 3]` | xyz positions |
| `<id>_normals.npy` | `[1024, 3]` | outward normals |
| `<id>_skeleton.npy` | `[K, 4]` | joint xyz + BFS parent_k |
| `<id>_skinning.npy` | `[V, K]` | dense per-vertex skinning weights |

## Rig Info Format (`rig_info_remesh/<id>.txt`)
```
joints Hips 0.0 0.92 0.0
root   Hips
hier   Hips Spine
hier   Spine Chest
skin   0 Hips 0.8 Spine 0.2
```
Parsed by `parse_rig_info()` in `dataset.py` — no Blender required.
