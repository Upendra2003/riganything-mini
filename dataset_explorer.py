"""
RigAnything — Phase 1: Dataset Explorer + Visualizer
======================================================
Fixes the Python 3.13 / open3d conflict by keeping the two environments
completely separate:

    Your conda env (system Python)          Blender 5.1 (subprocess)
    ──────────────────────────────          ────────────────────────
    open3d  → visualization                 bpy → reads FBX armature
    numpy   → point cloud math              writes JSON to temp file
    this script                             blender_extract_rig.py

Usage:
    python dataset_explorer.py --explore
    python dataset_explorer.py --shape_id 13 --visualize
    python dataset_explorer.py --shape_id 13 --visualize --vis_normals
"""

import os, sys, json, argparse, tempfile, subprocess
import numpy as np
import open3d as o3d
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",    default="RignetDataset")
parser.add_argument("--fbx_dir",        default="RignetDataset/fbx")
parser.add_argument("--out_dir",        default="pointClouds/fbx")
parser.add_argument("--shape_id",       default="")
parser.add_argument("--explore",        action="store_true")
parser.add_argument("--visualize",      action="store_true")
parser.add_argument("--vis_normals",    action="store_true")
parser.add_argument("--num_points",     type=int, default=1024)
parser.add_argument("--blender",        default="blender")
parser.add_argument("--extract_script", default="blender_extract_rig.py")
args = parser.parse_args()

G="\033[92m"; Y="\033[93m"; R="\033[91m"; B="\033[94m"; C="\033[96m"
BOLD="\033[1m"; RESET="\033[0m"

def ok(m):      print(f"  {G}✓{RESET} {m}")
def warn(m):    print(f"  {Y}⚠{RESET}  {m}")
def err(m):     print(f"  {R}✗{RESET} {m}")
def info(m):    print(f"  {C}→{RESET} {m}")
def section(n,m): print(f"\n{BOLD}  [{n}] {m}{RESET}")
def header(m):
    print(f"\n{BOLD}{B}{'═'*62}{RESET}")
    print(f"{BOLD}{B}  {m}{RESET}")
    print(f"{BOLD}{B}{'═'*62}{RESET}")

# ── Split files ───────────────────────────────────────────────
def read_split(path):
    if not os.path.exists(path): return []
    ids = []
    for line in open(path):
        b = os.path.basename(line.strip()).replace('.fbx','').replace('.obj','').strip()
        if b: ids.append(b)
    return ids

# ── Blender subprocess ────────────────────────────────────────
def extract_rig_via_blender(fbx_path):
    if not os.path.exists(args.extract_script):
        warn(f"blender_extract_rig.py not found — put it in same folder as this script")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()

    cmd = [
        args.blender, "--background",
        "--python", os.path.abspath(args.extract_script),
        "--", os.path.abspath(fbx_path), tmp.name
    ]
    info(f"Blender cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in proc.stdout.splitlines():
            if "[blender_extract]" in line:
                info(f"  {line.strip()}")
        if proc.returncode != 0:
            warn(f"Blender exit code: {proc.returncode}")
            for line in proc.stderr.splitlines()[-8:]:
                print(f"    {Y}{line}{RESET}")
    except FileNotFoundError:
        err(f"Blender not found at: {args.blender}")
        err("Tip: find your blender path with:  which blender")
        err("     then pass it:  --blender /snap/bin/blender")
        return None
    except subprocess.TimeoutExpired:
        err("Blender timed out")
        return None

    if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
        warn("Blender produced no JSON output")
        return None

    data = json.load(open(tmp.name))
    os.unlink(tmp.name)

    if not data.get('has_armature'):
        warn("FBX has no armature / skeleton")
        return None

    data['joint_pos'] = np.array(data['joint_pos'], dtype=np.float32)
    data['parents']   = np.array(data['parents'],   dtype=np.int32)

    # ── Fix coordinate scale mismatch ──────────────────────────
    # Blender imports cm-scale FBX with a 0.01 world-scale on the armature,
    # making joint positions 100x smaller than Open3D's mesh vertices.
    # Strategy: compare Blender mesh bbox to Open3D mesh bbox and compute
    # the exact scale + offset to align them.
    blender_scale = data.get('blender_scale', 1.0)
    if blender_scale != 1.0 and abs(blender_scale) > 1e-8:
        info(f"Applying coordinate correction (Blender scale={blender_scale:.6f})")

        if 'blender_mesh_bbox' in data:
            # Use the actual mesh bboxes from both sides to compute transform
            bl_min = np.array(data['blender_mesh_bbox']['min'], dtype=np.float32)
            bl_max = np.array(data['blender_mesh_bbox']['max'], dtype=np.float32)
            info(f"Blender mesh bbox:  min={bl_min.round(4)}  max={bl_max.round(4)}")
            # We will align using Open3D mesh bbox in the caller — store for now
            data['blender_bbox_min'] = bl_min
            data['blender_bbox_max'] = bl_max
        else:
            # Simple fallback: just divide by scale factor
            data['joint_pos'] = data['joint_pos'] / blender_scale
            ok(f"Joint positions rescaled by 1/{blender_scale:.4f}")

    return data

def dense_skin(rig, V):
    if not rig or not rig.get('skin_weights'): return None
    names       = rig['joint_names']
    K           = len(names)
    n2i         = {n: i for i, n in enumerate(names)}
    W           = np.zeros((V, K), dtype=np.float32)
    for vid_s, ws in rig['skin_weights'].items():
        vid = int(vid_s)
        if vid >= V: continue
        for bone, w in ws.items():
            if bone in n2i: W[vid, n2i[bone]] = w
    rs = W.sum(axis=1, keepdims=True)
    rs = np.where(rs < 1e-8, 1.0, rs)
    return W / rs

# ── Barycentric sampling ──────────────────────────────────────
def sample_surface(mesh_o3d, N=1024):
    verts = np.asarray(mesh_o3d.vertices,  dtype=np.float64)
    tris  = np.asarray(mesh_o3d.triangles, dtype=np.int64)
    v1,v2,v3 = verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)
    tot   = areas.sum()
    probs = areas / tot if tot > 1e-12 else np.ones(len(areas))/len(areas)
    chosen = np.random.choice(len(tris), size=N, p=probs)
    pts = np.zeros((N,3), dtype=np.float32)
    nrm = np.zeros((N,3), dtype=np.float32)
    for i, ti in enumerate(chosen):
        t = tris[ti]; a,b,c = verts[t[0]],verts[t[1]],verts[t[2]]
        r1,r2 = np.random.random(), np.random.random()
        sr = np.sqrt(r1); u,v,w = 1-sr, sr*(1-r2), sr*r2
        pts[i] = (u*a + v*b + w*c).astype(np.float32)
        n = np.cross(b-a, c-a); nl = np.linalg.norm(n)
        nrm[i] = (n/nl if nl>1e-10 else [0,1,0]).astype(np.float32)
    cen = verts.mean(axis=0)
    flip = ((cen - pts) * nrm).sum(axis=1) > 0
    nrm[flip] *= -1
    return pts, nrm, int(flip.sum())

# ── BFS ───────────────────────────────────────────────────────
def build_bfs(names, parents):
    K = len(names)
    children = {i: [] for i in range(K)}
    root = 0
    for i, p in enumerate(parents):
        if p < 0 or p == i: root = i
        else: children[p].append(i)
    seq=[]; imap={}; bfs_k=1
    q = deque([(root, -1, 0)])
    while q:
        oi, poi, d = q.popleft()
        imap[oi] = bfs_k
        seq.append({'bfs_k':bfs_k,'orig_idx':oi,'name':names[oi] if oi<len(names) else str(oi),
                    'parent_k':imap.get(poi,1) if poi>=0 else 1,'depth':d})
        bfs_k += 1
        sibs = children[oi].copy(); np.random.shuffle(sibs)
        for ch in sibs: q.append((ch, oi, d+1))
    return seq

def print_bfs(seq, jpos=None):
    print(f"\n{BOLD}{B}  BFS Skeleton Traversal{RESET}")
    print(f"  {'k':>4}  {'p_k':>4}  {'d':>2}  {'joint name':<30}  {'position':>28}  valid?")
    print(f"  {'─'*82}")
    ok_all = True
    for e in seq:
        k,pk,d,name = e['bfs_k'],e['parent_k'],e['depth'],e['name']
        valid = k==1 or pk<k
        if not valid: ok_all = False
        flag  = f"{G}✓{RESET}" if valid else f"{R}✗ p_k={pk}>=k={k}{RESET}"
        pos_s = ""
        if jpos is not None and e['orig_idx']<len(jpos):
            p = jpos[e['orig_idx']]
            pos_s = f"[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]"
        indent = "  "*d
        print(f"  {k:>4}  {pk:>4}  {d:>2}  {indent}{name:<{30-2*d}}  {pos_s:>28}  {flag}")
    print()
    if ok_all: ok(f"All {len(seq)} joints satisfy p_k < k  ✓")
    else:      err("BFS violation detected")

# ── Checkpoints ───────────────────────────────────────────────
def checkpoints(pts, nrm, seq, skin, shape_id):
    print(f"\n{BOLD}{Y}{'─'*62}{RESET}")
    print(f"{BOLD}{Y}  TEST CHECKPOINTS — shape {shape_id}{RESET}")
    print(f"{BOLD}{Y}{'─'*62}{RESET}")
    combined = np.concatenate([pts,nrm], axis=-1)
    s = G+"✓"+RESET if combined.shape==(args.num_points,6) else R+"✗"+RESET
    print(f"\n  {s} TEST 1  shape = {combined.shape}   (expect ({args.num_points},6))")
    dev = abs(np.linalg.norm(nrm,axis=1)-1).max()
    s = G+"✓"+RESET if dev<1e-4 else Y+"⚠"+RESET
    print(f"  {s} TEST 2  normal lengths ≈ 1.0   max dev = {dev:.2e}")
    if seq:
        viol = [e for e in seq if e['bfs_k']>1 and e['parent_k']>=e['bfs_k']]
        s = G+"✓"+RESET if not viol else R+"✗"+RESET
        print(f"  {s} TEST 3  BFS p_k<k:  {len(seq)} joints,  {len(viol)} violations")
        print(f"     max depth = {max(e['depth'] for e in seq)}")
    else:
        print(f"  {Y}⚠{RESET}  TEST 3  no skeleton data")
    if skin is not None:
        dev2 = abs(skin.sum(axis=1)-1).max()
        s = G+"✓"+RESET if dev2<1e-4 else Y+"⚠"+RESET
        print(f"  {s} TEST 4  skinning rows sum→1   max dev = {dev2:.2e}   {skin.shape}")
    else:
        print(f"  {Y}⚠{RESET}  TEST 4  no skinning data")
    kb = args.num_points*6*4/1024
    print(f"  {C}→{RESET} TEST 5  {kb:.1f} KB/shape   500 shapes ≈ {kb*500/1024:.1f} MB  ✓\n")

# ── Visualization ─────────────────────────────────────────────
DCOLS = [
    [0.95,0.26,0.21],[1.00,0.55,0.00],[0.95,0.85,0.10],[0.20,0.70,0.30],
    [0.13,0.59,0.95],[0.61,0.15,0.69],[0.00,0.74,0.83],[1.00,0.34,0.13],
]

def make_skel_geoms(rig, seq):
    jp = rig['joint_pos']
    km = {e['bfs_k']:e['orig_idx'] for e in seq}
    spheres=[]
    for e in seq:
        i = e['orig_idx']
        if i>=len(jp): continue
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        sp.translate(jp[i].astype(np.float64))
        sp.paint_uniform_color(DCOLS[e['depth']%len(DCOLS)])
        sp.compute_vertex_normals(); spheres.append(sp)
    pts,lines,cols=[],[],[]
    for e in seq:
        if e['bfs_k']==1: continue
        ci,pi = e['orig_idx'], km.get(e['parent_k'],e['orig_idx'])
        if ci>=len(jp) or pi>=len(jp): continue
        idx=len(pts)
        pts.append(jp[ci].tolist()); pts.append(jp[pi].tolist())
        lines.append([idx,idx+1]); cols.append(DCOLS[e['depth']%len(DCOLS)])
    ls=o3d.geometry.LineSet()
    ls.points=o3d.utility.Vector3dVector(pts)
    ls.lines=o3d.utility.Vector2iVector(lines)
    ls.colors=o3d.utility.Vector3dVector(cols)
    return spheres, ls

def show(geoms, title, show_normals=False):
    print(f"\n  {BOLD}Window: {title}{RESET}")
    print(f"  {C}Close window to continue...{RESET}")
    o3d.visualization.draw_geometries(geoms, window_name=title,
        point_show_normal=show_normals, width=1024, height=768)

def visualize(mesh, pts, nrm, rig, seq, shape_id):
    mesh.compute_vertex_normals()
    # 1 — mesh
    mc=o3d.geometry.TriangleMesh(mesh); mc.paint_uniform_color([0.75,0.80,0.88])
    show([mc], f"[1/3] shape {shape_id} — mesh ({len(np.asarray(mesh.vertices))} verts)")
    # 2 — point cloud
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.normals=o3d.utility.Vector3dVector(nrm.astype(np.float64))
    pcd.paint_uniform_color([0.20,0.55,0.90])
    show([pcd], f"[2/3] shape {shape_id} — {len(pts)} point cloud (normals outward)",
         show_normals=args.vis_normals)
    # 3 — skeleton with semi-transparent mesh
    if rig and seq:
        print(f"\n  Depth colour key:  {R}red=root{RESET}  orange=1  yellow=2  {G}green=3{RESET}  {B}blue=4+{RESET}")
        print(f"  {C}Mesh rendered as wireframe so skeleton joints are visible through it{RESET}")
        spheres, ls = make_skel_geoms(rig, seq)

        # Open3D does not support true transparency on TriangleMesh in draw_geometries.
        # Best alternative: show wireframe (edges only) so joints are visible through mesh.
        # Convert mesh to wireframe lineset — this shows the mesh topology without
        # occluding the joints behind solid faces.
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wire.paint_uniform_color([0.78, 0.80, 0.84])   # light grey wireframe

        # Also keep a very faint solid mesh for shape reference using point cloud
        # sampled from mesh surface — gives shape impression without occlusion
        mesh_pcd = mesh.sample_points_uniformly(number_of_points=8000)
        mesh_pcd.paint_uniform_color([0.88, 0.90, 0.93])

        show([wire, mesh_pcd, ls] + spheres,
             f"[3/3] shape {shape_id} — skeleton ({len(seq)} joints)  |  wireframe mesh + joint spheres")
    else:
        mb=o3d.geometry.TriangleMesh(mesh); mb.paint_uniform_color([0.85,0.88,0.92])
        show([mb,pcd], f"[2→3] shape {shape_id} — mesh + point cloud (no skeleton)")

# ── Main ──────────────────────────────────────────────────────
def main():
    header("RigAnything — Phase 1: Dataset Explorer + Visualizer")

    section("ENV", "Environment")
    info(f"Python {sys.version.split()[0]}  at  {sys.executable}")
    info(f"open3d {o3d.__version__}")
    info(f"Blender subprocess: {args.blender}  (separate process — no version conflict)")

    train = read_split(os.path.join(args.dataset_dir,"train_final.txt"))
    val   = read_split(os.path.join(args.dataset_dir,"val_final.txt"))
    test  = read_split(os.path.join(args.dataset_dir,"test_final.txt"))

    section("1", "Dataset splits")
    ok(f"train={len(train)}  val={len(val)}  test={len(test)}  total={len(train)+len(val)+len(test)}")
    if train: info(f"Train IDs (first 6): {train[:6]}")

    if args.explore:
        fbx_files = [f for f in os.listdir(args.fbx_dir) if f.endswith('.fbx')]
        ok(f"FBX files: {len(fbx_files)}")
        info(f"Sample: {sorted(fbx_files)[:6]}")
        sid = train[0] if train else "13"
        print(f"\n{C}  Visualize one shape:{RESET}")
        print(f"  python dataset_explorer.py --shape_id {sid} --visualize --vis_normals")
        return

    sid = args.shape_id or (train[0] if train else "13")
    fbx = os.path.join(args.fbx_dir, f"{sid}.fbx")
    if not os.path.exists(fbx):
        m = [f for f in os.listdir(args.fbx_dir) if f.startswith(sid)]
        if m: fbx = os.path.join(args.fbx_dir, m[0]); ok(f"Found: {fbx}")
        else: err(f"Shape {sid} not found"); sys.exit(1)

    section("2", f"Loading mesh: {sid}.fbx")
    mesh = o3d.io.read_triangle_mesh(fbx)
    V    = len(np.asarray(mesh.vertices))
    T    = len(np.asarray(mesh.triangles))
    if V == 0: err("Mesh has 0 vertices"); sys.exit(1)
    ok(f"Vertices={V}  Triangles={T}")
    verts = np.asarray(mesh.vertices)
    info(f"Bbox min={verts.min(0).round(3)}  max={verts.max(0).round(3)}")

    section("3", "Extracting rig via Blender subprocess")
    rig = extract_rig_via_blender(fbx)
    skin = None
    seq  = []
    if rig:
        ok(f"Bones: {len(rig['joint_names'])}   first 5: {rig['joint_names'][:5]}")
        skin = dense_skin(rig, V)
        if skin is not None: ok(f"Skinning: {skin.shape}")

        # ── Align joint positions to Open3D mesh coordinate space ──
        # Open3D mesh bbox is known (verts loaded above)
        o3d_min = verts.min(axis=0)
        o3d_max = verts.max(axis=0)

        # ── Step 1: Swap Y and Z axes ────────────────────────────
        # Blender is Z-up:  height = Z axis
        # Open3D reads FBX as Y-up: height = Y axis
        # Simply swapping Y↔Z on joint positions fixes the axis mismatch.
        jpos = rig['joint_pos']
        jpos[:, [1, 2]] = jpos[:, [2, 1]]   # swap Y and Z columns
        rig['joint_pos'] = jpos
        info(f"Axis swap applied (Blender Z-up → Open3D Y-up)")

        if 'blender_bbox_min' in rig:
            # Also swap Y/Z on the Blender mesh bbox so alignment math is consistent
            bl_min = rig['blender_bbox_min'].copy()
            bl_max = rig['blender_bbox_max'].copy()
            bl_min[1], bl_min[2] = rig['blender_bbox_min'][2], rig['blender_bbox_min'][1]
            bl_max[1], bl_max[2] = rig['blender_bbox_max'][2], rig['blender_bbox_max'][1]

            bl_range = bl_max - bl_min
            o3_range = o3d_max - o3d_min

            # Per-axis scale — each axis scaled independently
            axis_scale = np.where(np.abs(bl_range) > 1e-8,
                                   o3_range / bl_range,
                                   np.ones(3))
            # Use median to avoid distortion from degenerate axes
            uniform_scale = float(np.median(axis_scale[np.abs(bl_range) > 1e-8]))

            # Align: map from Blender bbox space → Open3D bbox space
            rig['joint_pos'] = (rig['joint_pos'] - bl_min) * uniform_scale + o3d_min

            info(f"Blender mesh bbox (Y↔Z swapped): min={bl_min.round(4)}  max={bl_max.round(4)}")
            info(f"Open3D mesh bbox:                 min={o3d_min.round(4)}  max={o3d_max.round(4)}")
            info(f"Alignment scale: {uniform_scale:.6f}")
            info(f"Pelvis after full alignment: {rig['joint_pos'][0].round(4)}")
        elif 'blender_scale' in rig and rig['blender_scale'] != 1.0:
            s = rig['blender_scale']
            rig['joint_pos'] = rig['joint_pos'] / s
            info(f"Joints rescaled by 1/{s:.4f}")
    else:
        warn("No rig extracted — geometry only")

    section("4", "BFS sequence")
    if rig:
        seq = build_bfs(rig['joint_names'], rig['parents'])
        print_bfs(seq, rig['joint_pos'])
    else:
        warn("Skipping — no rig")

    section("5", f"Barycentric sampling → {args.num_points} points")
    pts, nrm, n_flip = sample_surface(mesh, N=args.num_points)
    ok(f"points={pts.shape}  normals={nrm.shape}  flipped={n_flip}")

    checkpoints(pts, nrm, seq, skin, sid)

    section("6", f"Saving → {args.out_dir}/")
    os.makedirs(args.out_dir, exist_ok=True)
    combined = np.concatenate([pts,nrm], axis=-1)
    np.save(f"{args.out_dir}/{sid}_pointcloud.npy", combined)
    np.save(f"{args.out_dir}/{sid}_points.npy",     pts)
    np.save(f"{args.out_dir}/{sid}_normals.npy",    nrm)
    if seq and rig is not None:
        sk = np.array([[*rig['joint_pos'][e['orig_idx']],float(e['parent_k'])]
                        for e in seq if e['orig_idx']<len(rig['joint_pos'])], dtype=np.float32)
        np.save(f"{args.out_dir}/{sid}_skeleton.npy", sk)
        ok(f"skeleton.npy: {sk.shape}")
    if skin is not None:
        np.save(f"{args.out_dir}/{sid}_skinning.npy", skin)
        ok(f"skinning.npy: {skin.shape}")
    ok(f"pointcloud.npy: {combined.shape}")

    section("7", "Visualization")
    if args.visualize:
        info("3 windows in sequence — close each to proceed")
        visualize(mesh, pts, nrm, rig, seq, sid)
        ok("Done")
    else:
        info(f"Add --visualize to open 3D windows")
        info(f"  python dataset_explorer.py --shape_id {sid} --visualize --vis_normals")

    print(f"\n{BOLD}{G}  Shape {sid} complete{RESET}\n")

if __name__ == "__main__":
    main()