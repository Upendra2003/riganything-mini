"""
blender_extract_rig.py
=======================
Runs INSIDE Blender's Python — no open3d needed here.
Extracts skeleton + skinning from ONE FBX file and writes JSON.

Called from dataset_explorer.py via subprocess:
    blender --background --python blender_extract_rig.py -- <fbx_path> <out_json>

DO NOT run this directly with your system Python.
"""

import bpy
import sys
import json
import os
import mathutils

# ── Parse args passed after "--" ──────────────────────────────
argv    = sys.argv
sep_idx = argv.index("--") if "--" in argv else len(argv)
args    = argv[sep_idx + 1:]

if len(args) < 2:
    print("Usage: blender --background --python blender_extract_rig.py -- <fbx_path> <out_json>")
    sys.exit(1)

fbx_path = args[0]
out_json = args[1]

# ── Clear default scene ────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── Import FBX ────────────────────────────────────────────────
print(f"[blender_extract] Loading: {fbx_path}")
bpy.ops.import_scene.fbx(filepath=fbx_path)

# ── Find objects ──────────────────────────────────────────────
armature_obj = None
mesh_obj     = None

for obj in bpy.data.objects:
    if obj.type == 'ARMATURE' and armature_obj is None:
        armature_obj = obj
    elif obj.type == 'MESH' and mesh_obj is None:
        mesh_obj = obj

result = {
    'fbx_path':    fbx_path,
    'has_armature': armature_obj is not None,
    'has_mesh':     mesh_obj is not None,
    'joint_names': [],
    'joint_pos':   [],       # list of [x, y, z]  rest pose world space
    'parents':     [],       # list of int, -1 for root
    'skin_weights': {},      # { vertex_id (str) : { joint_name: weight } }
    'num_vertices': 0,
}

# ── Extract skeleton ───────────────────────────────────────────
if armature_obj is not None:
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    arm_data   = armature_obj.data
    edit_bones = arm_data.edit_bones
    bone_names = [b.name for b in edit_bones]
    name_to_idx = {n: i for i, n in enumerate(bone_names)}

    result['joint_names'] = bone_names

    # Detect FBX unit scale — Blender auto-applies 0.01 when importing cm-scale FBX.
    # We store this so the main script can divide joint positions by this factor,
    # putting them back in the same coordinate space Open3D uses for mesh vertices.
    arm_sx, arm_sy, arm_sz = armature_obj.matrix_world.to_scale()
    scale = (arm_sx + arm_sy + arm_sz) / 3.0
    result['blender_scale'] = scale
    print(f"[blender_extract] Armature world scale: ({arm_sx:.6f}, {arm_sy:.6f}, {arm_sz:.6f})")

    for bone in edit_bones:
        world_head = armature_obj.matrix_world @ bone.head
        result['joint_pos'].append([world_head.x, world_head.y, world_head.z])
        if bone.parent is None:
            result['parents'].append(-1)
        else:
            result['parents'].append(name_to_idx.get(bone.parent.name, -1))

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[blender_extract] Bones found: {len(bone_names)}")

    # Grab Blender-space mesh bbox so main script can verify alignment
    if mesh_obj is not None:
        vw = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
        xs=[v.x for v in vw]; ys=[v.y for v in vw]; zs=[v.z for v in vw]
        result['blender_mesh_bbox']={'min':[min(xs),min(ys),min(zs)],'max':[max(xs),max(ys),max(zs)]}
        bb=result['blender_mesh_bbox']
        print(f"[blender_extract] Blender mesh bbox: min={[round(v,4) for v in bb['min']]} max={[round(v,4) for v in bb['max']]}")

# ── Extract skinning weights ───────────────────────────────────
if mesh_obj is not None:
    mesh_data = mesh_obj.data
    V = len(mesh_data.vertices)
    result['num_vertices'] = V

    # vertex_group name -> bone index
    vg_to_bone = {}
    for vg in mesh_obj.vertex_groups:
        if vg.name in result['joint_names']:
            vg_to_bone[vg.index] = vg.name

    skin = {}
    for vert in mesh_data.vertices:
        vid = vert.index
        weights = {}
        total   = 0.0
        for g in vert.groups:
            if g.group in vg_to_bone and g.weight > 1e-6:
                name   = vg_to_bone[g.group]
                weights[name] = g.weight
                total += g.weight
        if weights:
            # Normalize
            if total > 1e-8:
                weights = {k: v / total for k, v in weights.items()}
            skin[str(vid)] = weights

    result['skin_weights'] = skin
    print(f"[blender_extract] Vertices with skin data: {len(skin)}/{V}")

# ── Write JSON ────────────────────────────────────────────────
os.makedirs(os.path.dirname(out_json) if os.path.dirname(out_json) else '.', exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(result, f)

print(f"[blender_extract] Saved → {out_json}")