# Phase 1 — Point Cloud Dataset Generation

## Overview

The [RigNet dataset](https://github.com/zhan-xu/RigNet) consists of **2703 3D models**. In this phase, each model is converted into a **point cloud of 1024 points** along with outward surface normals.

The dimension 1024 is chosen deliberately — prior work in 3D learning uses this dimension, and it is a good fit for transformer-based architectures since it provides enough resolution for stable training without being computationally prohibitive.

This phase processes `.fbx` meshes, samples 1024 surface points with outward normals, extracts skeleton + skinning via Blender subprocesses, applies coordinate fixes, and saves everything into `pointClouds/fbx/`.

---

## Running Phase 1

```bash
# Convert all 2703 shapes
python phase1_dataset.py --max_shapes 2703

# Default (500 shapes if --max_shapes is omitted)
python phase1_dataset.py

# Resume an interrupted run
python phase1_dataset.py --max_shapes 2703 --resume

# Skip rig extraction (point cloud only, faster)
python phase1_dataset.py --max_shapes 2703 --no_rig
```

---

## Output

All files are saved to `pointClouds/fbx/`:

| File | Shape | Description |
|------|-------|-------------|
| `<id>_pointcloud.npy` | `[1024, 6]` | xyz + normals — main input to Phase 2 |
| `<id>_points.npy` | `[1024, 3]` | xyz positions |
| `<id>_normals.npy` | `[1024, 3]` | outward surface normals |
| `<id>_skeleton.npy` | `[K, 4]` | joint xyz + BFS parent index |
| `<id>_skinning.npy` | `[V, K]` | dense per-vertex skinning weights |

---

## Background — File Formats

### FBX (Filmbox)
FBX is owned by Autodesk and is considered the gold standard for exchanging 3D data between software applications (3D editors, game engines).

- Supports full animation and keyframes
- Supports skeletal rigging and joints
- Binary compressed format

### OBJ (Wavefront Object)
A simpler, open standard focused strictly on the physical shape of a model.

- No animation support
- No rigging support
- Human-readable plain text

This project uses `.fbx` because it carries the full rig data needed for skeleton and skinning extraction.

### Armature
In 3D modeling, an **armature** is a skeletal framework that controls the movement and deformation of a 3D mesh — exactly like a real-world skeleton. It is a rigid structure of interconnected bones that can be posed or animated.

---

## Code Walkthrough

The script uses `argparse` for CLI flags so parameters are passed at runtime rather than being hardcoded.

Only the non-trivial functions are explained below.

---

### `extract_rig(fbx_path)`

**Purpose:** Extract skeleton data (joints, hierarchy, skinning weights) from an FBX model.

**How it works:**
1. Creates a temporary `.json` file
2. Launches Blender headlessly (`--background`) to run `blender_extract_rig.py`
3. Blender loads the FBX, extracts the rig, and writes it to the temp JSON
4. Python reads the JSON and deletes the temp file
5. Converts lists to NumPy arrays and returns the rig data

**Expected JSON written by Blender:**
```json
{
  "has_armature": true,
  "joint_pos": [[x, y, z], ...],
  "parents": [-1, 0, 1, ...],
  "joint_names": ["Hips", "Spine", ...],
  "skin_weights": {"0": {"BoneName": 0.8, ...}, ...}
}
```

**Returned dict:**
```python
{
  "joint_pos":   np.array([K, 3]),   # joint world positions
  "parents":     np.array([K]),      # parent index per joint (-1 = root)
  "joint_names": [...],
  "skin_weights": {...}
}
```

**Pseudo-code:**
```
for each FBX:
    if rig extraction disabled → skip

    create temp JSON file

    launch Blender in background:
        load FBX
        run extraction script
        write rig data → JSON

    if Blender failed or timed out → skip
    if JSON missing or empty     → skip

    read JSON
    delete temp file

    if no skeleton found → skip

    convert lists → NumPy arrays
    return rig data
```

> Skinning weights tell you how much each vertex is influenced by each joint.  
> `extract_rig()` is the ground-truth generator for the rig supervision signal.

---

### `align_joints(rig, o3d_verts)`

**Purpose:** Convert joint positions from Blender coordinate space into Open3D mesh space so that the skeleton and point cloud align correctly.

Blender uses a Z-up coordinate system; Open3D uses Y-up. This function applies the axis swap and a scale/offset correction derived from bounding box alignment.

```
Blender skeleton:
  small + rotated + misplaced

        ↓ align_joints()

Open3D mesh space:
  correctly rotated + scaled + positioned
```

**Steps:**
1. Swap Y and Z axes (Blender Z-up → Open3D Y-up)
2. If a mesh bounding box from Blender is available, compute scale and offset to map skeleton into Open3D vertex space
3. Otherwise, divide by the Blender global scale factor

---

### `dense_skin(rig, V)`

**Purpose:** Convert sparse skinning weights (stored as a dict) into a dense `[V × K]` matrix.

- `V` = number of mesh vertices  
- `K` = number of joints  

Each row is normalized so weights sum to 1.0 per vertex.

---

### `sample_surface(mesh_o3d, N=1024)`

**Purpose:** Convert a mesh into the point cloud representation that the model will learn from — `N` points uniformly sampled from the surface with outward normals.

**Why area-weighted sampling:**
A mesh is made of many triangles of varying sizes. Larger triangles cover more surface area, so they should receive proportionally more sampled points. This is achieved via area-weighted random selection.

**Barycentric sampling inside a triangle:**
Points are not just placed at corners — they are placed anywhere inside the selected triangle using barycentric coordinates with the `sqrt(r1)` correction for uniform distribution.

**Normal computation and orientation:**
- Normal = cross product of two triangle edges (perpendicular to surface)
- Some computed normals may initially point inward
- These are flipped to ensure all normals point outward

**Returns:**
```python
pts  → [N, 3]   # sampled surface points
nrm  → [N, 3]   # outward normals at those points
```

The combined `[N, 6]` array (xyz + normal) is the primary input to Phase 2.

---

### `build_bfs(names, parents)`

**Purpose:** Convert a skeleton joint tree into a BFS-ordered sequence.

Joints are visited breadth-first from the root. Each joint is assigned a BFS index `k`, and its parent's BFS index `parent_k` is guaranteed to satisfy `parent_k < k`. This ordering makes the skeleton structure compatible with sequential models (transformers, RNNs) since every parent always appears before its children.

**Saved as `[K, 4]`:** `[joint_x, joint_y, joint_z, parent_k]`

---

## Dataset Notes

The full RigNet dataset has 2703 shapes, but downloads are often incomplete:

| Status | Count |
|--------|-------|
| Empty FBX (0 bytes — incomplete download) | ~2520 |
| Corrupted FBX (truncated / parse error) | ~19 |
| Successfully converted | ~164 |

If you see a high `∅` count in the progress bar, re-download the missing FBX files from the RigNet source.
