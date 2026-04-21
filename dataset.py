"""
RigAnything — Phase 1: Point Cloud Dataset Generation
======================================================
Converts preprocessed RigNet OBJ meshes + rig_info txt files into
point cloud NPY arrays saved in pointClouds/obj_remesh/.

Source data (ModelResource_RigNetv1_preproccessed/):
  obj_remesh/<id>.obj       remeshed mesh (1K–5K verts)
  rig_info_remesh/<id>.txt  joints, hierarchy, root, skinning

Output per shape (pointClouds/obj_remesh/):
  <id>_pointcloud.npy  [1024, 6]  xyz + normals  ← Phase 2 input
  <id>_points.npy      [1024, 3]  xyz positions
  <id>_normals.npy     [1024, 3]  outward normals
  <id>_skeleton.npy    [K, 4]     joint xyz + BFS parent_k
  <id>_skinning.npy    [V, K]     dense per-vertex skinning weights

Usage:
  python phase1_dataset.py
  python phase1_dataset.py --split all --max_shapes 2703 --resume
"""

import os, sys, time, argparse
import numpy as np
import open3d as o3d
from collections import deque

# ── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="Dataset")
parser.add_argument("--out_dir",     default="pointClouds/obj_remesh")
parser.add_argument("--max_shapes",  type=int, default=2703)
parser.add_argument("--num_points",  type=int, default=1024)
parser.add_argument("--split",       default="all",
                    choices=["train", "val", "test", "all"])
parser.add_argument("--resume",      action="store_true",
                    help="Skip shapes that already have a saved pointcloud.npy")
args, _ = parser.parse_known_args()

# ── ANSI colours ───────────────────────────────────────────────
G="\033[92m"; Y="\033[93m"; R="\033[91m"
B="\033[94m"; C="\033[96m"; BOLD="\033[1m"; RESET="\033[0m"

def ok(m):    print(f"  {G}✓{RESET} {m}")
def warn(m):  print(f"  {Y}⚠{RESET}  {m}")
def err(m):   print(f"  {R}✗{RESET} {m}")
def info(m):  print(f"  {C}→{RESET} {m}")
def header(m):
    w = 62
    print(f"\n{BOLD}{B}{'═'*w}{RESET}")
    print(f"{BOLD}{B}  {m}{RESET}")
    print(f"{BOLD}{B}{'═'*w}{RESET}")

# ── Shape ID list ──────────────────────────────────────────────
def read_split(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        sid = os.path.basename(line.strip()).replace(".obj", "").replace(".fbx", "").strip()
        if sid:
            out.append(sid)
    return out

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
        warn("No split files found — scanning obj_remesh/ folder")
        obj_dir = os.path.join(d, "obj_remesh")
        ids = sorted(f.replace(".obj", "") for f in os.listdir(obj_dir)
                     if f.endswith(".obj"))

    return ids[:args.max_shapes]

def already_done(sid):
    return os.path.exists(os.path.join(args.out_dir, f"{sid}_pointcloud.npy"))

def is_available(sid):
    obj = os.path.join(args.dataset_dir, "obj_remesh", f"{sid}.obj")
    return os.path.exists(obj) and os.path.getsize(obj) > 0

# ── Rig info parser ────────────────────────────────────────────
def parse_rig_info(sid):
    """
    Parse rig_info_remesh/<id>.txt into a rig dict.

    File format:
      joint <name> <x> <y> <z>
      root  <name>
      hier  <parent> <child>
      skin  <vertex_id> <bone1> <w1> <bone2> <w2> ...
    """
    path = os.path.join(args.dataset_dir, "rig_info_remesh", f"{sid}.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    joints = {}     # name → [x, y, z]
    root   = None
    hier   = []     # (parent_name, child_name)
    skin   = {}     # vertex_id → {bone: weight}

    for line in open(path):
        parts = line.strip().split()
        if not parts:
            continue
        tag = parts[0]
        if tag in ("joint", "joints"):
            joints[parts[1]] = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif tag == "root":
            root = parts[1]
        elif tag == "hier":
            hier.append((parts[1], parts[2]))
        elif tag == "skin":
            vid = int(parts[1])
            pairs = {parts[i]: float(parts[i + 1]) for i in range(2, len(parts), 2)}
            skin[vid] = pairs

    if not joints or root is None:
        return None

    # Build ordered joint list (BFS from root so indices are consistent)
    children = {n: [] for n in joints}
    for p, c in hier:
        if p in children:
            children[p].append(c)

    names, order = [], []
    q = deque([root])
    while q:
        n = q.popleft()
        names.append(n)
        for ch in children.get(n, []):
            q.append(ch)

    name2idx = {n: i for i, n in enumerate(names)}
    joint_pos = np.array([joints[n] for n in names], dtype=np.float32)

    parents = np.full(len(names), -1, dtype=np.int32)
    for p, c in hier:
        if p in name2idx and c in name2idx:
            parents[name2idx[c]] = name2idx[p]

    return dict(joint_names=names, joint_pos=joint_pos,
                parents=parents, skin_weights=skin)

# ── Dense skinning matrix ──────────────────────────────────────
def dense_skin(rig, V):
    if rig is None or not rig["skin_weights"]:
        return None
    names = rig["joint_names"]
    K     = len(names)
    n2i   = {n: i for i, n in enumerate(names)}
    W     = np.zeros((V, K), dtype=np.float32)
    for vid, pairs in rig["skin_weights"].items():
        if vid >= V:
            continue
        for bone, w in pairs.items():
            if bone in n2i:
                W[vid, n2i[bone]] = w
    row_sum = W.sum(axis=1, keepdims=True)
    W /= np.where(row_sum < 1e-8, 1.0, row_sum)
    return W

# ── BFS skeleton ordering ──────────────────────────────────────
def build_bfs(rig):
    """
    Return [(bfs_k, orig_idx, parent_k), ...] with parent_k < bfs_k guaranteed.
    """
    names   = rig["joint_names"]
    parents = rig["parents"]
    K       = len(names)

    children = {i: [] for i in range(K)}
    root     = 0
    for i, p in enumerate(parents):
        if p < 0 or p == i:
            root = i
        else:
            children[p].append(i)

    seq  = []
    imap = {}
    bfs_k = 1
    q = deque([(root, -1)])
    while q:
        oi, poi = q.popleft()
        imap[oi] = bfs_k
        seq.append(dict(bfs_k=bfs_k, orig_idx=oi,
                        parent_k=imap.get(poi, 1) if poi >= 0 else 1))
        bfs_k += 1
        for ch in children[oi]:
            q.append((ch, oi))
    return seq

# ── Surface sampling ───────────────────────────────────────────
def sample_surface(mesh, N):
    """
    Area-weighted barycentric sampling → N points + outward normals.
    """
    verts = np.asarray(mesh.vertices,  dtype=np.float64)
    tris  = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        raise ValueError("mesh has no triangles")

    v1, v2, v3 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    total = areas.sum()
    probs = areas / total if total > 1e-12 else np.ones(len(areas)) / len(areas)

    chosen = np.random.choice(len(tris), size=N, p=probs)
    pts = np.zeros((N, 3), dtype=np.float32)
    nrm = np.zeros((N, 3), dtype=np.float32)

    for i, ti in enumerate(chosen):
        a, b, c = verts[tris[ti, 0]], verts[tris[ti, 1]], verts[tris[ti, 2]]
        r1, r2  = np.random.random(), np.random.random()
        sr      = np.sqrt(r1)
        pts[i]  = (( 1 - sr) * a + sr * (1 - r2) * b + sr * r2 * c).astype(np.float32)
        n       = np.cross(b - a, c - a)
        nl      = np.linalg.norm(n)
        nrm[i]  = (n / nl if nl > 1e-10 else np.array([0., 1., 0.])).astype(np.float32)

    # Flip normals that point toward mesh centroid (inward)
    cen  = verts.mean(axis=0)
    flip = ((cen - pts) * nrm).sum(axis=1) > 0
    nrm[flip] *= -1

    return pts, nrm, int(flip.sum())

# ── Save ───────────────────────────────────────────────────────
def save_shape(sid, pts, nrm, bfs_seq, rig, skin):
    base = os.path.join(args.out_dir, sid)
    np.save(f"{base}_pointcloud.npy", np.concatenate([pts, nrm], axis=-1))
    np.save(f"{base}_points.npy",     pts)
    np.save(f"{base}_normals.npy",    nrm)
    if bfs_seq is not None and rig is not None:
        jpos = rig["joint_pos"]
        sk   = np.array([[*jpos[e["orig_idx"]], float(e["parent_k"])]
                         for e in bfs_seq if e["orig_idx"] < len(jpos)],
                        dtype=np.float32)
        np.save(f"{base}_skeleton.npy", sk)
    if skin is not None:
        np.save(f"{base}_skinning.npy", skin)

# ── Test checkpoints ───────────────────────────────────────────
def run_checkpoints(pts, nrm, bfs_seq, skin, sid):
    print(f"\n{BOLD}{Y}{'─'*62}{RESET}")
    print(f"{BOLD}{Y}  TEST CHECKPOINTS — shape {sid}{RESET}")
    print(f"{BOLD}{Y}{'─'*62}{RESET}\n")

    combined = np.concatenate([pts, nrm], axis=-1)
    s = G+"✓"+RESET if combined.shape == (args.num_points, 6) else R+"✗"+RESET
    print(f"  {s} TEST 1  combined shape = {combined.shape}   expect ({args.num_points},6)")

    dev = abs(np.linalg.norm(nrm, axis=1) - 1).max()
    s = G+"✓"+RESET if dev < 1e-4 else Y+"⚠"+RESET
    print(f"  {s} TEST 2  normal unit length   max deviation = {dev:.2e}")

    if bfs_seq:
        viol = [e for e in bfs_seq if e["bfs_k"] > 1 and e["parent_k"] >= e["bfs_k"]]
        s = G+"✓"+RESET if not viol else R+"✗"+RESET
        print(f"  {s} TEST 3  BFS parent_k < k   joints={len(bfs_seq)}   violations={len(viol)}")
    else:
        print(f"  {Y}⚠{RESET}  TEST 3  no skeleton")

    if skin is not None:
        dev2 = abs(skin.sum(axis=1) - 1).max()
        s = G+"✓"+RESET if dev2 < 1e-4 else Y+"⚠"+RESET
        print(f"  {s} TEST 4  skinning rows sum→1   max dev={dev2:.2e}   shape={skin.shape}")
    else:
        print(f"  {Y}⚠{RESET}  TEST 4  no skinning")

    kb = args.num_points * 6 * 4 / 1024
    print(f"  {C}→{RESET} TEST 5  {kb:.1f} KB/shape  ✓\n")

# ── Process one shape ──────────────────────────────────────────
def process_shape(sid, run_tests=False):
    obj_path = os.path.join(args.dataset_dir, "obj_remesh", f"{sid}.obj")
    if not os.path.exists(obj_path) or os.path.getsize(obj_path) == 0:
        return "skip"

    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_triangles():
        return None

    pts, nrm, n_flip = sample_surface(mesh, args.num_points)

    rig     = parse_rig_info(sid)
    bfs_seq = build_bfs(rig) if rig else None
    skin    = dense_skin(rig, len(mesh.vertices)) if rig else None
    n_joints = len(rig["joint_names"]) if rig else 0

    if run_tests:
        run_checkpoints(pts, nrm, bfs_seq, skin, sid)

    return dict(pts=pts, nrm=nrm, bfs_seq=bfs_seq, rig=rig,
                skin=skin, n_joints=n_joints, n_flip=n_flip)

# ── Progress bar ───────────────────────────────────────────────
def show_progress(done, total, n_ok, n_skip, n_empty, n_fail, eta_s, sid):
    pct    = done / max(total, 1)
    filled = int(40 * pct)
    bar    = G + "█" * filled + RESET + "░" * (40 - filled)
    eta    = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s" if eta_s > 0 else "--"
    print(f"\r  [{bar}] {done}/{total}  "
          f"{G}✓{n_ok}{RESET} {Y}↷{n_skip}{RESET} {C}∅{n_empty}{RESET} {R}✗{n_fail}{RESET}  "
          f"eta={eta}  {sid:<14}", end="", flush=True)

# ── Main ──────────────────────────────────────────────────────
def main():
    header(f"RigAnything — Phase 1  ({args.max_shapes} shapes  |  split={args.split})")
    os.makedirs(args.out_dir, exist_ok=True)

    info(f"Dataset:      {args.dataset_dir}")
    info(f"Output:       {args.out_dir}")
    info(f"Points/shape: {args.num_points}")
    info(f"Resume:       {args.resume}")

    shape_ids = get_shape_ids()
    if not shape_ids:
        err("No shapes found — check dataset path"); sys.exit(1)

    n_valid = sum(1 for sid in shape_ids if is_available(sid))
    n_empty = len(shape_ids) - n_valid
    ok(f"Shapes queued: {len(shape_ids)}  ({n_valid} with data, {n_empty} empty)")
    if n_empty > 0:
        warn(f"{n_empty} OBJ files are empty — dataset download may be incomplete")

    # First valid shape — run test checkpoints
    first = next((sid for sid in shape_ids if is_available(sid)), None)
    if first is None:
        err("No valid OBJ files found — check dataset download"); sys.exit(1)

    print(f"\n{BOLD}  Shape 1 — running TEST CHECKPOINTS  [{first}]{RESET}")
    n_ok = n_skip = n_empty_skip = n_fail = 0

    if args.resume and already_done(first):
        ok(f"{first} already saved — skipping")
        n_skip = 1
    else:
        r = process_shape(first, run_tests=True)
        if not r or r == "skip":
            err(f"First shape {first} failed"); sys.exit(1)
        save_shape(first, r["pts"], r["nrm"], r["bfs_seq"], r["rig"], r["skin"])
        ok(f"Saved {first}   joints={r['n_joints']}   flipped={r['n_flip']}")
        n_ok = 1

    # Batch
    print(f"\n{BOLD}  Processing remaining {len(shape_ids) - 1} shapes{RESET}\n")
    times   = []
    t_start = time.time()

    for idx, sid in enumerate(shape_ids[1:], start=2):
        if args.resume and already_done(sid):
            n_skip += 1
            show_progress(idx, len(shape_ids), n_ok, n_skip, n_empty_skip, n_fail, 0, sid)
            continue

        t0 = time.time()
        try:
            r = process_shape(sid)
        except Exception:
            n_fail += 1
            show_progress(idx, len(shape_ids), n_ok, n_skip, n_empty_skip, n_fail, 0, sid)
            continue

        if r == "skip":
            n_empty_skip += 1
        elif r is None:
            n_fail += 1
        else:
            save_shape(sid, r["pts"], r["nrm"], r["bfs_seq"], r["rig"], r["skin"])
            n_ok += 1
            times.append(time.time() - t0)
            if len(times) > 50:
                times.pop(0)

        avg = sum(times) / len(times) if times else 0
        eta = (n_valid - n_ok) * avg if avg > 0 else 0
        show_progress(idx, len(shape_ids), n_ok, n_skip, n_empty_skip, n_fail, eta, sid)

    print()

    # Summary
    total_t = time.time() - t_start
    header("Phase 1 Complete")
    ok(f"Saved:   {n_ok} shapes")
    if n_skip:       info(f"Skipped: {n_skip} (already done)")
    if n_empty_skip: info(f"Empty:   {n_empty_skip} (dataset incomplete)")
    if n_fail:       warn(f"Failed:  {n_fail}")
    ok(f"Time:    {total_t / 60:.1f} min  ({total_t / max(n_ok, 1):.1f}s per shape)")

    files   = [f for f in os.listdir(args.out_dir) if f.endswith(".npy")]
    shapes  = len(set(f.split("_")[0] for f in files))
    total_b = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in files)
    ok(f"Output:  {args.out_dir}/  —  {len(files)} files  {shapes} shapes  ({total_b / 1024**2:.1f} MB)")
    print(f"\n{C}  Next → Phase 2: python phase2_tokenizer.py{RESET}\n")

if __name__ == "__main__":
    main()
