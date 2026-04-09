# RigAnything-Mini — Claude Code Notes

## Project Overview
Deep learning pipeline for 3D character rigging prediction (RigNet-style).
- **Phase 1** (`phase1_dataset.py`): FBX → point cloud + skeleton NPY files
- **Phase 2** (`phase2_tokenizer.py`): tokenizes point clouds for transformer input
- **Main** (`main.py`): training entrypoint

## Environment
- Python venv at `.venv/` — always use `.venv/bin/python`
- Key packages: `open3d`, `trimesh`, `fbxloader`, `numpy`, `torch`
- **Blender is NOT installed** — rig extraction via Blender subprocess is unavailable

## FBX Loading
FBX files are loaded with `fbxloader.FBXLoader` (pure Python, no Blender needed):
```python
from fbxloader import FBXLoader
loader = FBXLoader(fbx_path)
tm = loader.export_trimesh()   # returns trimesh.Trimesh
```
Then convert to Open3D for surface sampling:
```python
mesh = o3d.geometry.TriangleMesh()
mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices.astype(np.float64))
mesh.triangles = o3d.utility.Vector3iVector(tm.faces.astype(np.int32))
```

## Dataset
- FBX files: `RignetDataset/fbx/<id>.fbx`
- Split lists: `RignetDataset/{train,val,test}_final.txt`
- Total shapes across all splits: ~2703
- Some FBX files (e.g. `1101.fbx`) are empty — `process_shape` skips them via size check

## Running Phase 1
```bash
# Convert all shapes (resume-safe)
.venv/bin/python phase1_dataset.py --split all --max_shapes 10000 --resume

# Quick test (5 shapes)
.venv/bin/python phase1_dataset.py --split all --max_shapes 5
```

## Output Format (per shape in `pointClouds/fbx/`)
| File | Shape | Description |
|------|-------|-------------|
| `<id>_pointcloud.npy` | `[1024, 6]` | xyz + normals — main Phase 2 input |
| `<id>_points.npy` | `[1024, 3]` | xyz positions |
| `<id>_normals.npy` | `[1024, 3]` | outward normals |
| `<id>_skeleton.npy` | `[K, 4]` | joint xyz + BFS parent_k (requires Blender) |
| `<id>_skinning.npy` | `[V, K]` | dense skinning weights (requires Blender) |

## Skeleton / Skinning
Skeleton and skinning data require Blender + `blender_extract_rig.py` (not yet set up).
Currently only point cloud files are generated. To add rig extraction later:
1. Install Blender system-wide
2. Create `blender_extract_rig.py` that accepts `-- <fbx> <json_out>`
3. Remove `--no_rig` flag or let `extract_rig()` detect the script automatically
