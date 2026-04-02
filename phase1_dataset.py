"""
RigAnything — Phase 1: Point Cloud Dataset Generation
======================================================
Processes FBX meshes, samples 1024 surface points with outward normals,
extracts skeleton + skinning via Blender subprocess, applies coordinate
fixes, and saves all results to pointClouds/fbx/.

Saved per shape:
  <id>_pointcloud.npy   [1024, 6]   xyz + normals  ← Phase 2 input
  <id>_points.npy       [1024, 3]   xyz positions
  <id>_normals.npy      [1024, 3]   outward normals
  <id>_skeleton.npy     [K, 4]      joint xyz + BFS parent_k
  <id>_skinning.npy     [V, K]      dense skinning weights

Usage:
    python phase1_dataset.py --max_shapes 500 --fbx_dir RignetDataset/fbx --out_dir pointClouds/fbx
    python phase1_dataset.py --max_shapes 500 --resume        # skip already done
    python phase1_dataset.py --max_shapes 500 --no_rig        # geometry only, faster
"""

import os
import sys
import json
import time
import argparse
import tempfile
import subprocess
import numpy as np
import open3d as o3d
from collections import deque

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",    default="RignetDataset")
parser.add_argument("--fbx_dir",        default="RignetDataset/fbx")
parser.add_argument("--out_dir",        default="pointClouds/fbx")
parser.add_argument("--max_shapes",     type=int, default=500)
parser.add_argument("--num_points",     type=int, default=1024)
parser.add_argument("--split",          default="train",
                    choices=["train","val","test","all"])
parser.add_argument("--blender",        default="blender")
parser.add_argument("--extract_script", default="blender_extract_rig.py")
parser.add_argument("--resume",         action="store_true",
                    help="Skip shapes that already have saved pointcloud.npy")
parser.add_argument("--no_rig",         action="store_true",
                    help="Skip Blender rig extraction (point cloud only)")
args = parser.parse_args()

# ── ANSI colours ──────────────────────────────────────────────
G="\033[92m"; Y="\033[93m"; R="\033[91m"
B="\033[94m"; C="\033[96m"; BOLD="\033[1m"; RESET="\033[0m"

def ok(m):    print(f"  {G}✓{RESET} {m}")
def warn(m):  print(f"  {Y}⚠{RESET}  {m}")
def err(m):   print(f"  {R}✗{RESET} {m}")
def info(m):  print(f"  {C}→{RESET} {m}")
def header(m):
    print(f"\n{BOLD}{B}{'═'*62}{RESET}")
    print(f"{BOLD}{B}  {m}{RESET}")
    print(f"{BOLD}{B}{'═'*62}{RESET}")

# ── Split reader ───────────────────────────────────────────────
def read_split(path):
    if not os.path.exists(path):
        return []
    ids = []
    for line in open(path):
        b = os.path.basename(line.strip()).replace('.fbx','').replace('.obj','').strip()
        if b:
            ids.append(b)
    return ids

def get_shape_ids():
    d = args.dataset_dir
    train = read_split(os.path.join(d, "train_final.txt"))
    val   = read_split(os.path.join(d, "val_final.txt"))
    test  = read_split(os.path.join(d, "test_final.txt"))
    if args.split == "train":   ids = train
    elif args.split == "val":   ids = val
    elif args.split == "test":  ids = test
    else:                       ids = train + val + test
    if not ids:
        warn("No split .txt files found — using all FBX in folder")
        ids = sorted(
            [f.replace('.fbx','') for f in os.listdir(args.fbx_dir) if f.endswith('.fbx')],
            key=lambda x: int(x) if x.isdigit() else x
        )
    return ids[:args.max_shapes]

def already_done(sid):
    return os.path.exists(os.path.join(args.out_dir, f"{sid}_pointcloud.npy"))

# ── Blender rig extraction ─────────────────────────────────────
def extract_rig(fbx_path):
    if args.no_rig or not os.path.exists(args.extract_script):
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    cmd = [
        args.blender, "--background",
        "--python", os.path.abspath(args.extract_script),
        "--", os.path.abspath(fbx_path), tmp.name
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
        return None
    try:
        data = json.load(open(tmp.name))
    except json.JSONDecodeError:
        return None
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    if not data.get('has_armature'):
        return None
    data['joint_pos'] = np.array(data['joint_pos'], dtype=np.float32)
    data['parents']   = np.array(data['parents'],   dtype=np.int32)
    return data

# ── Coordinate alignment ───────────────────────────────────────
def align_joints(rig, o3d_verts):
    """
    Fix 1 — Y/Z axis swap (Blender Z-up → Open3D Y-up)
    Fix 2 — Scale + offset via bbox alignment
    """
    jpos = rig['joint_pos'].copy()
    jpos[:, [1, 2]] = jpos[:, [2, 1]]          # Y ↔ Z swap

    if 'blender_mesh_bbox' in rig:
        bl_min = np.array(rig['blender_mesh_bbox']['min'], dtype=np.float32)
        bl_max = np.array(rig['blender_mesh_bbox']['max'], dtype=np.float32)
        bl_min[1], bl_min[2] = bl_min[2], bl_min[1]   # swap bbox too
        bl_max[1], bl_max[2] = bl_max[2], bl_max[1]

        o3d_min = o3d_verts.min(axis=0).astype(np.float32)
        o3d_max = o3d_verts.max(axis=0).astype(np.float32)
        bl_range = bl_max - bl_min
        o3_range = o3d_max - o3d_min
        valid    = np.abs(bl_range) > 1e-8
        scales   = np.where(valid, o3_range / bl_range, np.ones(3))
        u_scale  = float(np.median(scales[valid])) if valid.any() else 1.0
        jpos     = (jpos - bl_min) * u_scale + o3d_min

    elif rig.get('blender_scale', 1.0) != 1.0:
        jpos = jpos / rig['blender_scale']

    rig['joint_pos'] = jpos
    return rig

# ── Dense skinning ─────────────────────────────────────────────
def dense_skin(rig, V):
    if not rig or not rig.get('skin_weights'):
        return None
    names = rig['joint_names']; K = len(names)
    n2i   = {n: i for i, n in enumerate(names)}
    W     = np.zeros((V, K), dtype=np.float32)
    for vid_s, ws in rig['skin_weights'].items():
        vid = int(vid_s)
        if vid >= V: continue
        for bone, w in ws.items():
            if bone in n2i: W[vid, n2i[bone]] = w
    rs = W.sum(axis=1, keepdims=True)
    return W / np.where(rs < 1e-8, 1.0, rs)

# ── Barycentric sampling ───────────────────────────────────────
def sample_surface(mesh_o3d, N=1024):
    """
    Area-weighted barycentric sampling with sqrt(r1) correction.
    Triangle area:  A = 0.5 * ||(v2-v1) x (v3-v1)||
    Barycentric:    p = u*v1 + v*v2 + w*v3
                    u=1-sqrt(r1), v=sqrt(r1)*(1-r2), w=sqrt(r1)*r2
    Outward normal: n = normalize((v2-v1)x(v3-v1)), flipped if inward
    """
    verts = np.asarray(mesh_o3d.vertices,  dtype=np.float64)
    tris  = np.asarray(mesh_o3d.triangles, dtype=np.int64)
    if len(tris) == 0:
        raise ValueError("No triangles")
    v1,v2,v3 = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)
    total = areas.sum()
    probs = areas/total if total > 1e-12 else np.ones(len(areas))/len(areas)
    chosen = np.random.choice(len(tris), size=N, p=probs)
    pts = np.zeros((N,3), dtype=np.float32)
    nrm = np.zeros((N,3), dtype=np.float32)
    for i, ti in enumerate(chosen):
        t = tris[ti]; a,b,c = verts[t[0]],verts[t[1]],verts[t[2]]
        r1,r2 = np.random.random(), np.random.random()
        sr = np.sqrt(r1); u,v,w = 1-sr, sr*(1-r2), sr*r2
        pts[i] = (u*a + v*b + w*c).astype(np.float32)
        n = np.cross(b-a, c-a); nl = np.linalg.norm(n)
        nrm[i] = (n/nl if nl > 1e-10 else [0,1,0]).astype(np.float32)
    cen  = verts.mean(axis=0)
    flip = ((cen - pts)*nrm).sum(axis=1) > 0
    nrm[flip] *= -1
    return pts, nrm, int(flip.sum())

# ── BFS ordering ───────────────────────────────────────────────
def build_bfs(names, parents):
    K = len(names)
    children = {i:[] for i in range(K)}
    root = 0
    for i,p in enumerate(parents):
        if p < 0 or p == i: root = i
        else: children[p].append(i)
    seq=[]; imap={}; bfs_k=1
    q = deque([(root,-1,0)])
    while q:
        oi,poi,d = q.popleft()
        imap[oi] = bfs_k
        seq.append({'bfs_k':bfs_k,'orig_idx':oi,
                    'name':names[oi] if oi<len(names) else str(oi),
                    'parent_k':imap.get(poi,1) if poi>=0 else 1,
                    'depth':d})
        bfs_k += 1
        sibs = children[oi].copy(); np.random.shuffle(sibs)
        for ch in sibs: q.append((ch,oi,d+1))
    return seq

# ── Save ───────────────────────────────────────────────────────
def save_shape(sid, pts, nrm, bfs_seq, rig, skin):
    base = os.path.join(args.out_dir, sid)
    np.save(f"{base}_pointcloud.npy", np.concatenate([pts,nrm],axis=-1))
    np.save(f"{base}_points.npy",     pts)
    np.save(f"{base}_normals.npy",    nrm)
    if bfs_seq and rig is not None:
        jpos = rig['joint_pos']
        sk = np.array([[*jpos[e['orig_idx']], float(e['parent_k'])]
                        for e in bfs_seq if e['orig_idx']<len(jpos)],
                       dtype=np.float32)
        np.save(f"{base}_skeleton.npy", sk)
    if skin is not None:
        np.save(f"{base}_skinning.npy", skin)

# ── Test checkpoints ───────────────────────────────────────────
def run_checkpoints(pts, nrm, bfs_seq, skin, sid):
    print(f"\n{BOLD}{Y}{'─'*62}{RESET}")
    print(f"{BOLD}{Y}  TEST CHECKPOINTS — shape {sid}{RESET}")
    print(f"{BOLD}{Y}{'─'*62}{RESET}\n")
    combined = np.concatenate([pts,nrm],axis=-1)
    s = G+"✓"+RESET if combined.shape==(args.num_points,6) else R+"✗"+RESET
    print(f"  {s} TEST 1  combined shape = {combined.shape}   expect ({args.num_points},6)")
    dev = abs(np.linalg.norm(nrm,axis=1)-1).max()
    s = G+"✓"+RESET if dev<1e-4 else Y+"⚠"+RESET
    print(f"  {s} TEST 2  normal lengths ≈ 1.0   max deviation = {dev:.2e}")
    if bfs_seq:
        viol = [e for e in bfs_seq if e['bfs_k']>1 and e['parent_k']>=e['bfs_k']]
        s = G+"✓"+RESET if not viol else R+"✗"+RESET
        print(f"  {s} TEST 3  BFS p_k < k:  {len(bfs_seq)} joints,  {len(viol)} violations")
    else:
        print(f"  {Y}⚠{RESET}  TEST 3  no skeleton")
    if skin is not None:
        dev2 = abs(skin.sum(axis=1)-1).max()
        s = G+"✓"+RESET if dev2<1e-4 else Y+"⚠"+RESET
        print(f"  {s} TEST 4  skinning rows sum→1   max dev = {dev2:.2e}   {skin.shape}")
    else:
        print(f"  {Y}⚠{RESET}  TEST 4  no skinning data")
    kb = args.num_points*6*4/1024
    print(f"  {C}→{RESET} TEST 5  {kb:.1f} KB/shape   "
          f"{args.max_shapes} shapes ≈ {kb*args.max_shapes/1024:.1f} MB  ✓\n")

# ── Process one shape ──────────────────────────────────────────
def process_shape(sid, run_tests=False):
    fbx = os.path.join(args.fbx_dir, f"{sid}.fbx")
    if not os.path.exists(fbx):
        return None
    mesh = o3d.io.read_triangle_mesh(fbx)
    if len(np.asarray(mesh.vertices)) == 0:
        return None
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    rig   = extract_rig(fbx)
    bfs_seq = []; skin = None; n_joints = 0
    if rig is not None:
        rig      = align_joints(rig, verts)
        bfs_seq  = build_bfs(rig['joint_names'], rig['parents'])
        skin     = dense_skin(rig, len(verts))
        n_joints = len(bfs_seq)
    pts, nrm, n_flip = sample_surface(mesh, N=args.num_points)
    if run_tests:
        run_checkpoints(pts, nrm, bfs_seq, skin, sid)
    return dict(pts=pts, nrm=nrm, bfs_seq=bfs_seq,
                rig=rig, skin=skin, n_joints=n_joints, n_flip=n_flip)

# ── Progress bar ───────────────────────────────────────────────
def show_progress(done, total, n_ok, n_skip, n_fail, eta_s, sid):
    pct    = done / max(total,1)
    filled = int(40*pct)
    bar    = G+"█"*filled+RESET+"░"*(40-filled)
    eta    = f"{int(eta_s//60)}m{int(eta_s%60):02d}s" if eta_s>0 else "--"
    print(f"\r  [{bar}] {done}/{total}  "
          f"{G}✓{n_ok}{RESET} {Y}↷{n_skip}{RESET} {R}✗{n_fail}{RESET}  "
          f"eta={eta}  {sid:<14}", end="", flush=True)

# ── Main ──────────────────────────────────────────────────────
def main():
    header(f"RigAnything — Phase 1 Batch  ({args.max_shapes} shapes  |  split={args.split})")
    os.makedirs(args.out_dir, exist_ok=True)

    info(f"FBX dir:      {args.fbx_dir}")
    info(f"Output dir:   {args.out_dir}")
    info(f"Points/shape: {args.num_points}")
    info(f"Rig extract:  {'OFF' if args.no_rig else f'Blender ({args.blender})'}")
    info(f"Resume:       {args.resume}")

    shape_ids = get_shape_ids()
    if not shape_ids:
        err("No shapes found — check paths"); sys.exit(1)
    ok(f"Shapes queued: {len(shape_ids)}")

    # ── First shape: full test checkpoints ──
    print(f"\n{BOLD}  Shape 1/{len(shape_ids)} — running TEST CHECKPOINTS{RESET}")
    first = shape_ids[0]
    n_ok = n_skip = n_fail = 0

    if args.resume and already_done(first):
        ok(f"{first} already saved — skipping")
        n_skip = 1
    else:
        r = process_shape(first, run_tests=True)
        if r is None:
            err(f"First shape {first} failed"); sys.exit(1)
        save_shape(first, r['pts'], r['nrm'], r['bfs_seq'], r['rig'], r['skin'])
        ok(f"Saved {first}  joints={r['n_joints']}  flipped={r['n_flip']}")
        n_ok = 1

    # ── Batch loop ──
    print(f"\n{BOLD}  Processing {len(shape_ids)-1} remaining shapes{RESET}\n")
    times   = []
    t_start = time.time()

    for idx, sid in enumerate(shape_ids[1:], start=2):
        if args.resume and already_done(sid):
            n_skip += 1
            show_progress(idx, len(shape_ids), n_ok, n_skip, n_fail, 0, sid)
            continue

        t0 = time.time()
        try:
            r = process_shape(sid)
        except Exception:
            n_fail += 1
            show_progress(idx, len(shape_ids), n_ok, n_skip, n_fail, 0, sid)
            continue

        if r is None:
            n_fail += 1
        else:
            save_shape(sid, r['pts'], r['nrm'], r['bfs_seq'], r['rig'], r['skin'])
            n_ok += 1

        # Rolling ETA
        times.append(time.time() - t0)
        if len(times) > 20: times.pop(0)
        avg = sum(times)/len(times)
        eta = (len(shape_ids) - idx) * avg
        show_progress(idx, len(shape_ids), n_ok, n_skip, n_fail, eta, sid)

    print()  # newline after progress bar

    # ── Summary ──
    total_t = time.time() - t_start
    header("Phase 1 Complete")
    ok(f"Saved:   {n_ok} shapes")
    if n_skip: info(f"Skipped: {n_skip} (already done)")
    if n_fail: warn(f"Failed:  {n_fail}")
    ok(f"Time:    {total_t/60:.1f} min  ({total_t/max(n_ok,1):.1f}s per shape)")

    files  = [f for f in os.listdir(args.out_dir) if f.endswith('.npy')]
    shapes = len(set(f.split('_')[0] for f in files))
    total_b = sum(os.path.getsize(os.path.join(args.out_dir,f)) for f in files)
    ok(f"Output:  {args.out_dir}/  —  {len(files)} files  across {shapes} shapes  ({total_b/1024**2:.1f} MB)")
    print(f"\n{C}  Next → Phase 2: python phase2_tokenizer.py{RESET}\n")

if __name__ == "__main__":
    main()