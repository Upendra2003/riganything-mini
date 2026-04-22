# Inference Results Guide for Claude

This file is context for Claude to add new inference results to `index.html` and `viewer_data.json`.
Read this before making any changes to those files.

---

## Current Page Structure

```
index.html
├── Section 1: Adobe Model Inference    (paper reference, from inference_example_datamodels/)
│   ├── Spyro the Dragon  [original | predicted rig]
│   └── Neo Crispytyph    [original | predicted rig]
│
└── Section 2: Our Model Results        (Phase 7 end-to-end predictions)
    ├── v1 — 2 Transformer Blocks · 200 Dataset · 50 Epochs
    │   ├── Spyro  [original | v1 rig]    ← data key: spyro_the_dragon_pred_v1
    │   └── Neo    [original | v1 rig]    ← data key: neo_crispytyph_pred_v1
    └── v2 — 5 Transformer Blocks · 500 Dataset · 70 Epochs  [CURRENT]
        ├── Spyro  [original | v2 rig]    ← data key: spyro_the_dragon_pred_v2
        └── Neo    [original | v2 rig]    ← data key: neo_crispytyph_pred_v2
```

---

## viewer_data.json Schema

The JSON has these top-level keys:

| Key | Type | Description |
|-----|------|-------------|
| `spyro_original_b64` | string | Base64-encoded full-resolution Spyro GLB |
| `spyro_simp_b64` | string | Base64-encoded simplified Spyro GLB (used in rig panels) |
| `neo_original_b64` | string | Base64-encoded Neo GLB |
| `neo_simp_b64` | string | Base64-encoded Neo simplified GLB |
| `spyro_the_dragon` | object | Adobe paper reference rig data for Spyro |
| `neo_crispytyph` | object | Adobe paper reference rig data for Neo |
| `spyro_the_dragon_pred_v1` | object | Our v1 prediction for Spyro |
| `neo_crispytyph_pred_v1` | object | Our v1 prediction for Neo |
| `spyro_the_dragon_pred_v2` | object | Our v2 prediction for Spyro |
| `neo_crispytyph_pred_v2` | object | Our v2 prediction for Neo |

Each rig data object has this structure:
```json
{
  "joints":     [[x, y, z], ...],   // [K, 3] float — joint world positions
  "parents":    [0, 0, 1, ...],     // [K] int — 0-INDEXED parent for each joint
  "pointcloud": [[x, y, z], ...],   // [1024, 3] float — surface points
  "dominant":   [3, 3, 7, ...],     // [1024] int — dominant joint index (0-indexed) per point
  "num_joints": 64                  // int — total joints K
}
```

**Important:** The `.npy` files from inference use **1-indexed** parents (root = 1).
The JSON must use **0-indexed** parents (root = 0). Always subtract 1 when converting.

---

## How to Add a New Model Version (vN)

### Step 1 — Run inference

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python phase7/inference.py \
    --glb inference_example_datamodels/spyro_the_dragon.glb \
    --checkpoint checkpoints/phase7/best_model.pt \
    --out_dir output/phase7/testN

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python phase7/inference.py \
    --glb inference_example_datamodels/neo_crispytyph.glb \
    --checkpoint checkpoints/phase7/best_model.pt \
    --out_dir output/phase7/testN
```

Results land in `output/phase7/testN/`:
- `spyro_the_dragon_joints.npy`   [K, 3]
- `spyro_the_dragon_parents.npy`  [K] — 1-indexed
- `spyro_the_dragon_weights.npy`  [1024, K]
- (same pattern for neo_crispytyph)

### Step 2 — Update viewer_data.json

Run this Python snippet from the project root:

```python
import json, numpy as np

with open('viewer_data.json') as f:
    d = json.load(f)

VER = 'v3'   # change to next version
OUTDIR = 'output/phase7/testN'   # change to your output dir

for char in ['spyro_the_dragon', 'neo_crispytyph']:
    joints  = np.load(f'{OUTDIR}/{char}_joints.npy')
    parents = np.load(f'{OUTDIR}/{char}_parents.npy')   # 1-indexed
    weights = np.load(f'{OUTDIR}/{char}_weights.npy')

    # Reuse pointcloud from existing entry (same mesh = same 1024 surface points)
    pointcloud = d[f'{char}_pred_v2']['pointcloud']    # or any existing pred entry

    d[f'{char}_pred_{VER}'] = {
        'joints':     joints.tolist(),
        'parents':    (parents - 1).tolist(),           # convert to 0-indexed!
        'pointcloud': pointcloud,
        'dominant':   weights.argmax(axis=1).tolist(),
        'num_joints': int(joints.shape[0]),
    }

with open('viewer_data.json', 'w') as f:
    json.dump(d, f, separators=(',', ':'))

print("Done. New keys:", [k for k in d if 'b64' not in k])
```

### Step 3 — Add a version sub-heading in index.html

Find the `<!-- ── v2 Sub-heading ── -->` block and add a new block **after** the entire v2 grid:

```html
<!-- ── vN Sub-heading ── -->
<div class="version-heading v2">
  <h3>Run N — X Transformer Blocks · Y Dataset · Z Epochs</h3>
  <span class="version-badge v2">vN · Current</span>
  <div class="version-line"></div>
</div>

<div class="col-labels">
  <div class="col-label original"><span class="dot"></span> Original Model</div>
  <div class="col-label predicted"><span class="dot"></span> Predicted Skeleton</div>
</div>

<div class="grid">
  <!-- Row 1: Spyro — vN -->
  <div class="row">
    <div class="panel" id="panel-spyro-vN-orig">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🐉</span> Spyro the Dragon</div>
        <div class="panel-meta">Original · GLB · textured</div>
      </div>
      <div class="canvas-wrap">
        <div class="loading-overlay" id="lo-spyro-vN-orig"><div class="spinner"></div><span class="loading-text">Loading…</span></div>
        <canvas id="c-spyro-vN-orig"></canvas>
      </div>
      <div class="controls-bar">
        <button class="ctrl-btn active" onclick="toggleWireframe('spyro-vN-orig',this)">Solid</button>
        <button class="ctrl-btn" onclick="resetCamera('spyro-vN-orig')">Reset</button>
        <span class="hint">Drag · Scroll · Right-drag</span>
      </div>
    </div>
    <div class="panel" id="panel-spyro-vN-rig">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🦴</span> Spyro — vN Predicted Rig</div>
        <div class="panel-meta" id="meta-spyro-vN">Predicting…</div>
      </div>
      <div class="canvas-wrap">
        <div class="loading-overlay" id="lo-spyro-vN-rig"><div class="spinner"></div><span class="loading-text">Loading…</span></div>
        <canvas id="c-spyro-vN-rig"></canvas>
      </div>
      <div class="controls-bar">
        <button class="ctrl-btn active" onclick="toggleMesh('spyro-vN-rig',this)">Mesh</button>
        <button class="ctrl-btn active" onclick="toggleSkeleton('spyro-vN-rig',this)">Skeleton</button>
        <button class="ctrl-btn active" onclick="toggleWeights('spyro-vN-rig',this)">Weights</button>
        <button class="ctrl-btn" onclick="resetCamera('spyro-vN-rig')">Reset</button>
        <span class="hint">Drag · Scroll · Right-drag</span>
      </div>
    </div>
  </div>

  <!-- Row 2: Neo — vN -->
  <div class="row">
    <div class="panel" id="panel-neo-vN-orig">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🤖</span> Neo Crispytyph</div>
        <div class="panel-meta">Original · GLB · textured</div>
      </div>
      <div class="canvas-wrap">
        <div class="loading-overlay" id="lo-neo-vN-orig"><div class="spinner"></div><span class="loading-text">Loading…</span></div>
        <canvas id="c-neo-vN-orig"></canvas>
      </div>
      <div class="controls-bar">
        <button class="ctrl-btn active" onclick="toggleWireframe('neo-vN-orig',this)">Solid</button>
        <button class="ctrl-btn" onclick="resetCamera('neo-vN-orig')">Reset</button>
        <span class="hint">Drag · Scroll · Right-drag</span>
      </div>
    </div>
    <div class="panel" id="panel-neo-vN-rig">
      <div class="panel-header">
        <div class="panel-title"><span class="icon">🦴</span> Neo — vN Predicted Rig</div>
        <div class="panel-meta" id="meta-neo-vN">Predicting…</div>
      </div>
      <div class="canvas-wrap">
        <div class="loading-overlay" id="lo-neo-vN-rig"><div class="spinner"></div><span class="loading-text">Loading…</span></div>
        <canvas id="c-neo-vN-rig"></canvas>
      </div>
      <div class="controls-bar">
        <button class="ctrl-btn active" onclick="toggleMesh('neo-vN-rig',this)">Mesh</button>
        <button class="ctrl-btn active" onclick="toggleSkeleton('neo-vN-rig',this)">Skeleton</button>
        <button class="ctrl-btn active" onclick="toggleWeights('neo-vN-rig',this)">Weights</button>
        <button class="ctrl-btn" onclick="resetCamera('neo-vN-rig')">Reset</button>
        <span class="hint">Drag · Scroll · Right-drag</span>
      </div>
    </div>
  </div>
</div>
```

### Step 4 — Wire up the JS loader in main()

Inside the `main()` function in `index.html`, find the `tasks` array and append 4 entries for vN.
Also add a new character at the end of their respective GLB loading. Find the last `loadRigPanel` call for v2 and add after it (inside the tasks array, before the closing `]`):

```js
    // ── Our model — vN (X blocks · Y data · Z epochs) ─────────────────
    loadOrigPanel('c-spyro-vN-orig',   data.spyro_original_b64)
      .then(() => { bar.style.width = 'XX%'; }),
    loadRigPanel ('c-spyro-vN-rig',    data.spyro_simp_b64,  data.spyro_the_dragon_pred_vN,
      () => {
        const m = data.spyro_the_dragon_pred_vN;
        document.getElementById('meta-spyro-vN').textContent =
          `${m.num_joints} joints · 1 024 pts · vN`;
      }).then(() => { bar.style.width = 'XX%'; }),
    loadOrigPanel('c-neo-vN-orig',     data.neo_original_b64)
      .then(() => { bar.style.width = 'XX%'; }),
    loadRigPanel ('c-neo-vN-rig',      data.neo_simp_b64,    data.neo_crispytyph_pred_vN,
      () => {
        const m = data.neo_crispytyph_pred_vN;
        document.getElementById('meta-neo-vN').textContent =
          `${m.num_joints} joints · 1 024 pts · vN`;
      }).then(() => { bar.style.width = '96%'; }),
```

---

## Adding a New Character Model

To add a third character (e.g., `my_character`):

1. **Get the GLBs** — original + simplified (or just original if no separate simplified mesh exists).
2. **Base64-encode** the GLBs and add to viewer_data.json:
   ```python
   import base64, json
   with open('viewer_data.json') as f: d = json.load(f)
   with open('path/to/character.glb', 'rb') as f:
       d['my_character_b64'] = base64.b64encode(f.read()).decode()
   with open('viewer_data.json', 'w') as f:
       json.dump(d, f, separators=(',', ':'))
   ```
3. **Run inference** and add the rig data object following Step 2 above, using key `my_character_pred_vN`.
4. **Add HTML rows** following the Spyro/Neo pattern, choosing a new emoji icon.
5. **Wire up** in the JS `main()` tasks array with the new canvas IDs and data keys.

---

## CSS Reference — Section Styling

| Class | Use |
|-------|-----|
| `.section-heading.adobe` | Blue tint — Adobe paper reference section |
| `.section-heading` | Orange tint — our model results section |
| `.version-heading` | Grey — first/earlier runs |
| `.version-heading.v2` + `.version-badge.v2` | Green — current best run |

To mark the newest version as "current", add class `v2` to its `.version-heading` and `.version-badge`.
Remove `v2` from the previous version to downgrade it to grey.

---

## File Locations Quick Reference

| File | Purpose |
|------|---------|
| `index.html` | The viewer — HTML + Three.js rendering |
| `viewer_data.json` | All binary data (GLBs as base64) + rig data |
| `output/phase7/` | v1 inference outputs (joints/parents/weights .npy) |
| `output/phase7/test2/` | v2 inference outputs |
| `output/phase7/testN/` | Future runs go here |
| `inference_example_datamodels/` | Source GLB files for Spyro and Neo |
| `checkpoints/phase7/best_model.pt` | Latest trained checkpoint |
