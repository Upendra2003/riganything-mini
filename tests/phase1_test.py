"""
Phase 1 — Validation & Visualisation
=====================================
Loads saved NPY files from pointClouds/obj_remesh/ and runs:
  1. Shape / dtype checks
  2. Normal unit-length check
  3. BFS parent ordering check
  4. Skinning row-sum check
  5. Renders a 3-panel PNG (point cloud, normals coloured, skeleton overlay)

Usage:
  python tests/phase1_test.py                   # tests first available shape
  python tests/phase1_test.py --shape_id 10000  # tests a specific shape
  python tests/phase1_test.py --all             # tests all shapes in pointClouds/
"""

import os, sys, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── paths ──────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC_DIR    = os.path.join(ROOT, "pointClouds", "obj_remesh")
OUT_DIR   = os.path.join(ROOT, "tests", "output")

# ── ANSI colours ───────────────────────────────────────────────
G="\033[92m"; Y="\033[93m"; R="\033[91m"; C="\033[96m"
BOLD="\033[1m"; RESET="\033[0m"

def ok(m):   print(f"  {G}PASS{RESET}  {m}")
def fail(m): print(f"  {R}FAIL{RESET}  {m}")
def warn(m): print(f"  {Y}WARN{RESET}  {m}")
def info(m): print(f"  {C}INFO{RESET}  {m}")

# ── load helpers ───────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

def files_for(sid):
    b = os.path.join(PC_DIR, sid)
    return dict(
        pointcloud = load(f"{b}_pointcloud.npy"),
        points     = load(f"{b}_points.npy"),
        normals    = load(f"{b}_normals.npy"),
        skeleton   = load(f"{b}_skeleton.npy"),
        skinning   = load(f"{b}_skinning.npy"),
    )

# ── individual checks ──────────────────────────────────────────
def check_shapes(d, sid):
    passed = True

    pc = d["pointcloud"]
    if pc is None:
        fail(f"[{sid}] pointcloud.npy missing"); return False
    if pc.shape[1] != 6:
        fail(f"[{sid}] pointcloud shape {pc.shape}  expected (N,6)"); passed = False
    else:
        ok(f"[{sid}] pointcloud shape {pc.shape}")

    for key, expected_cols in [("points", 3), ("normals", 3)]:
        arr = d[key]
        if arr is None:
            warn(f"[{sid}] {key}.npy missing")
        elif arr.shape[1] != expected_cols:
            fail(f"[{sid}] {key} shape {arr.shape}  expected (N,{expected_cols})"); passed = False
        else:
            ok(f"[{sid}] {key} shape {arr.shape}")

    if d["skeleton"] is None:
        warn(f"[{sid}] skeleton.npy missing — rig info may not exist for this shape")
    else:
        sk = d["skeleton"]
        if sk.ndim != 2 or sk.shape[1] != 4:
            fail(f"[{sid}] skeleton shape {sk.shape}  expected (K,4)"); passed = False
        else:
            ok(f"[{sid}] skeleton shape {sk.shape}  ({sk.shape[0]} joints)")

    if d["skinning"] is None:
        warn(f"[{sid}] skinning.npy missing")
    else:
        ok(f"[{sid}] skinning shape {d['skinning'].shape}")

    return passed

def check_normals(d, sid):
    nrm = d["normals"]
    if nrm is None:
        warn(f"[{sid}] skipping normal check — normals.npy missing"); return True
    dev = abs(np.linalg.norm(nrm, axis=1) - 1.0).max()
    if dev < 1e-4:
        ok(f"[{sid}] normals unit-length  max_deviation={dev:.2e}")
        return True
    else:
        fail(f"[{sid}] normals NOT unit-length  max_deviation={dev:.2e}")
        return False

def check_bfs(d, sid):
    sk = d["skeleton"]
    if sk is None:
        warn(f"[{sid}] skipping BFS check — skeleton.npy missing"); return True
    # parent_k is column 3 (1-indexed BFS order; root parent = 1 by convention)
    violations = []
    for k, row in enumerate(sk, start=1):
        parent_k = int(row[3])
        if k > 1 and parent_k >= k:
            violations.append((k, parent_k))
    if not violations:
        ok(f"[{sid}] BFS ordering  parent_k < k  ({len(sk)} joints)")
        return True
    else:
        fail(f"[{sid}] BFS violations: {violations[:5]}")
        return False

def check_skinning(d, sid):
    W = d["skinning"]
    if W is None:
        warn(f"[{sid}] skipping skinning check — skinning.npy missing"); return True
    dev = abs(W.sum(axis=1) - 1.0).max()
    if dev < 1e-4:
        ok(f"[{sid}] skinning rows sum→1  max_deviation={dev:.2e}")
        return True
    else:
        fail(f"[{sid}] skinning rows don't sum to 1  max_deviation={dev:.2e}")
        return False

# ── visualisation ──────────────────────────────────────────────
def render(d, sid, out_path):
    pts = d["points"]
    nrm = d["normals"]
    sk  = d["skeleton"]

    if pts is None:
        info(f"[{sid}] no points — skipping render"); return

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#0e0e0e")

    # ── Panel 1: raw point cloud ──
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_facecolor("#0e0e0e")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                c=pts[:, 1], cmap="plasma", s=1.5, alpha=0.85)
    ax1.set_title(f"Point Cloud\n{pts.shape[0]} pts", color="white", fontsize=10)
    _style_ax(ax1)

    # ── Panel 2: normals coloured ──
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_facecolor("#0e0e0e")
    if nrm is not None:
        # map normal direction to RGB
        rgb = (nrm * 0.5 + 0.5).clip(0, 1)
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=rgb, s=1.5, alpha=0.85)
        ax2.set_title("Normals (RGB = XYZ direction)", color="white", fontsize=10)
    else:
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="gray", s=1.5, alpha=0.5)
        ax2.set_title("Normals — missing", color="gray", fontsize=10)
    _style_ax(ax2)

    # ── Panel 3: skeleton overlay ──
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.set_facecolor("#0e0e0e")
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                c="#334455", s=1.0, alpha=0.3)

    if sk is not None and len(sk) > 0:
        jxyz = sk[:, :3]
        ax3.scatter(jxyz[:, 0], jxyz[:, 1], jxyz[:, 2],
                    c="cyan", s=30, zorder=5, label="joints")

        # draw bones (parent_k → child_k, 1-indexed)
        for k, row in enumerate(sk, start=1):
            parent_k = int(row[3])
            if parent_k > 0 and parent_k != k:
                pi = parent_k - 1   # convert to 0-indexed
                ci = k - 1
                if pi < len(jxyz):
                    ax3.plot([jxyz[pi, 0], jxyz[ci, 0]],
                             [jxyz[pi, 1], jxyz[ci, 1]],
                             [jxyz[pi, 2], jxyz[ci, 2]],
                             c="cyan", lw=1.0, alpha=0.8)

        ax3.set_title(f"Skeleton Overlay\n{len(sk)} joints", color="white", fontsize=10)
    else:
        ax3.set_title("Skeleton — missing", color="gray", fontsize=10)
    _style_ax(ax3)

    plt.suptitle(f"Shape {sid}", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    info(f"[{sid}] saved → {os.path.relpath(out_path)}")

def _style_ax(ax):
    ax.tick_params(colors="white", labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333333")
    ax.grid(False)
    ax.set_xlabel("X", color="#888888", fontsize=7)
    ax.set_ylabel("Y", color="#888888", fontsize=7)
    ax.set_zlabel("Z", color="#888888", fontsize=7)

# ── run one shape ──────────────────────────────────────────────
def test_shape(sid):
    print(f"\n{BOLD}── {sid} ──────────────────────────────────{RESET}")
    d = files_for(sid)

    results = [
        check_shapes(d, sid),
        check_normals(d, sid),
        check_bfs(d, sid),
        check_skinning(d, sid),
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    render(d, sid, os.path.join(OUT_DIR, f"{sid}_viz.png"))

    passed = all(results)
    status = f"{G}ALL PASS{RESET}" if passed else f"{R}SOME FAILED{RESET}"
    print(f"  → {status}")
    return passed

# ── discover shapes ────────────────────────────────────────────
def available_shape_ids():
    files = glob.glob(os.path.join(PC_DIR, "*_pointcloud.npy"))
    return sorted(os.path.basename(f).replace("_pointcloud.npy", "") for f in files)

# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_id", default=None,
                        help="Test a specific shape ID (e.g. 10000)")
    parser.add_argument("--all", action="store_true",
                        help="Test all shapes found in pointClouds/obj_remesh/")
    args = parser.parse_args()

    ids = available_shape_ids()
    if not ids:
        print(f"{R}No point clouds found in {PC_DIR}{RESET}")
        print("Run:  python dataset.py --max_shapes 5  first.")
        sys.exit(1)

    if args.shape_id:
        if args.shape_id not in ids:
            print(f"{R}Shape {args.shape_id} not found in {PC_DIR}{RESET}")
            sys.exit(1)
        to_test = [args.shape_id]
    elif args.all:
        to_test = ids
    else:
        to_test = [ids[0]]   # default: first available shape

    print(f"\n{BOLD}Phase 1 Tests — {len(to_test)} shape(s){RESET}")
    print(f"  Source:  {PC_DIR}")
    print(f"  Output:  {OUT_DIR}/")

    results = [test_shape(sid) for sid in to_test]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{BOLD}Summary:{RESET}  "
          f"{G}{n_pass} passed{RESET}  "
          f"{(R+str(n_fail)+' failed'+RESET) if n_fail else ''}")
    print(f"  Visualisations saved to {OUT_DIR}/\n")
    sys.exit(0 if n_fail == 0 else 1)

if __name__ == "__main__":
    main()
