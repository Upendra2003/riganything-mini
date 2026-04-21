"""
RigAnything — Phase 2: Shape & Skeleton Tokenizers
===================================================
Loads Phase 1 NPY outputs and produces token tensors (.pt files).

Input  (pointClouds/obj_remesh/):
  <id>_pointcloud.npy  [1024, 6]  xyz + normals
  <id>_skeleton.npy    [K, 4]     joint xyz + BFS parent_k (1-indexed)

Output (tokens/obj_remesh/):
  <id>_H.pt  [1024, 1024]  shape tokens
  <id>_T.pt  [K,    1024]  skeleton tokens

Usage:
  .venv/bin/python phase2_tokenizer.py --max_shapes 5
  .venv/bin/python phase2_tokenizer.py --max_shapes 2703 --resume
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

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


# ── Sinusoidal positional embedding ────────────────────────────
def sinusoidal_embedding(k: int, d: int = 1024) -> torch.Tensor:
    """
    gamma(k)_2i   = sin(k / 10000^(2i/d))
    gamma(k)_2i+1 = cos(k / 10000^(2i/d))
    Returns a [d] tensor.
    """
    i = torch.arange(d // 2, dtype=torch.float32)
    denom = torch.pow(10000.0, 2 * i / d)
    gamma = torch.zeros(d)
    gamma[0::2] = torch.sin(k / denom)
    gamma[1::2] = torch.cos(k / denom)
    return gamma


# ── ShapeTokenizer ─────────────────────────────────────────────
class ShapeTokenizer(nn.Module):
    """
    Point-wise MLP — no interaction between points.
    Input:  [N, 6]   (xyz + normals)
    Output: [N, d]
    """
    def __init__(self, d: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d),
        )
        # Zero biases so that weight×input differences dominate the output.
        # With default kaiming init the bias term otherwise dominates the mean
        # pool, making different shapes look nearly identical before training.
        for m in self.mlp:
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── SkeletonTokenizer ──────────────────────────────────────────
class SkeletonTokenizer(nn.Module):
    """
    Per-joint encoding using current joint, parent joint, and
    sinusoidal positional embeddings.
    Input:  positions [K, 3], parent_indices [K] (0-indexed)
    Output: [K, d]
    """
    def __init__(self, d: int = 1024):
        super().__init__()
        self.d = d
        self.joint_mlp = nn.Sequential(
            nn.Linear(3, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d),
        )
        self.combiner_mlp = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )

    def forward(self, positions: torch.Tensor, parent_indices: torch.Tensor) -> torch.Tensor:
        """
        positions:      [K, 3]
        parent_indices: [K]  (0-indexed; root's parent_index == 0)
        Returns:        [K, d]
        """
        K = positions.shape[0]

        # Encode all joint positions and parent positions via joint_mlp
        jk_feat  = self.joint_mlp(positions)                    # [K, d]
        jpk_feat = self.joint_mlp(positions[parent_indices])    # [K, d]

        # Sinusoidal embeddings for each joint index and its parent index
        # k is 0-indexed here (same as array index)
        gamma_k  = torch.stack([sinusoidal_embedding(k, self.d)  for k in range(K)])
        gamma_pk = torch.stack([sinusoidal_embedding(int(parent_indices[k].item()), self.d)
                                for k in range(K)])

        gamma_k  = gamma_k.to(positions.device)
        gamma_pk = gamma_pk.to(positions.device)

        combined = torch.cat([jk_feat, gamma_k, jpk_feat, gamma_pk], dim=-1)  # [K, 4*d]
        return self.combiner_mlp(combined)                                     # [K, d]


# ── I/O helpers ────────────────────────────────────────────────
def already_done(sid: str, out_dir: str) -> bool:
    base = os.path.join(out_dir, sid)
    return (os.path.exists(f"{base}_H.pt") and
            os.path.exists(f"{base}_T.pt"))


def discover_shape_ids(in_dir: str, max_shapes: int) -> list[str]:
    files = [f for f in os.listdir(in_dir) if f.endswith("_pointcloud.npy")]
    ids = sorted(f.replace("_pointcloud.npy", "") for f in files)
    return ids[:max_shapes]


def load_inputs(sid: str, in_dir: str = "pointClouds/obj_remesh"):
    base = os.path.join(in_dir, sid)
    pc_path = f"{base}_pointcloud.npy"
    sk_path = f"{base}_skeleton.npy"
    if not os.path.exists(pc_path):
        return None, None
    pc = np.load(pc_path).astype(np.float32)   # [1024, 6]
    sk = np.load(sk_path).astype(np.float32) if os.path.exists(sk_path) else None
    return pc, sk


def save_tokens(sid: str, H: torch.Tensor, T: torch.Tensor, out_dir: str):
    base = os.path.join(out_dir, sid)
    torch.save(H, f"{base}_H.pt")
    torch.save(T, f"{base}_T.pt")


# ── Process one shape ──────────────────────────────────────────
def process_shape(sid: str,
                  shape_tok: ShapeTokenizer,
                  skel_tok:  SkeletonTokenizer,
                  in_dir:    str = "pointClouds/obj_remesh"):
    """
    Returns (H, T) tensors or raises on error.
    """
    pc, sk = load_inputs(sid, in_dir)
    if pc is None:
        raise FileNotFoundError(f"pointcloud missing for {sid}")
    if sk is None:
        raise FileNotFoundError(f"skeleton missing for {sid}")

    pc_t = torch.from_numpy(pc)   # [1024, 6]

    # Shape tokens
    with torch.no_grad():
        H = shape_tok(pc_t)       # [1024, 1024]

    # Skeleton tokens
    positions = torch.from_numpy(sk[:, :3])            # [K, 3]
    parent_k  = sk[:, 3].astype(np.int32)              # 1-indexed BFS parent
    # Convert to 0-indexed; root parent_k==1 → parent_idx==0 (points to itself)
    parent_idx = torch.tensor(parent_k - 1, dtype=torch.long)

    with torch.no_grad():
        T = skel_tok(positions, parent_idx)            # [K, 1024]

    return H, T


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",     default="pointClouds/obj_remesh")
    parser.add_argument("--out_dir",    default="tokens/obj_remesh")
    parser.add_argument("--max_shapes", type=int, default=2703)
    parser.add_argument("--resume",     action="store_true",
                        help="Skip shapes that already have both _H.pt and _T.pt")
    args = parser.parse_args()

    header(f"RigAnything — Phase 2  ({args.max_shapes} shapes)")
    os.makedirs(args.out_dir, exist_ok=True)

    info(f"Input:   {args.in_dir}")
    info(f"Output:  {args.out_dir}")
    info(f"Resume:  {args.resume}")

    shape_ids = discover_shape_ids(args.in_dir, args.max_shapes)
    if not shape_ids:
        err(f"No pointcloud files found in {args.in_dir}"); sys.exit(1)

    ok(f"Shapes found: {len(shape_ids)}")

    # Build models (randomly initialised — no training yet)
    shape_tok = ShapeTokenizer().eval()
    skel_tok  = SkeletonTokenizer().eval()

    n_ok = n_skip = n_fail = 0
    t_start = time.time()

    for idx, sid in enumerate(shape_ids, start=1):
        if args.resume and already_done(sid, args.out_dir):
            n_skip += 1
            continue

        try:
            H, T = process_shape(sid, shape_tok, skel_tok, args.in_dir)
            save_tokens(sid, H, T, args.out_dir)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                warn(f"[{sid}] {e}")

        if idx % 50 == 0 or idx == len(shape_ids):
            elapsed = time.time() - t_start
            rate    = n_ok / max(elapsed, 1e-6)
            remaining = len(shape_ids) - idx
            eta   = remaining / rate if rate > 0 else 0
            eta_s = f"{int(eta // 60)}m{int(eta % 60):02d}s"
            print(f"  [{idx:>4}/{len(shape_ids)}]  "
                  f"{G}✓{n_ok}{RESET}  {Y}↷{n_skip}{RESET}  {R}✗{n_fail}{RESET}  "
                  f"eta={eta_s}", flush=True)

    total_t = time.time() - t_start
    header("Phase 2 Complete")
    ok(f"Saved:   {n_ok} shapes")
    if n_skip:  info(f"Skipped: {n_skip} (already done)")
    if n_fail:  warn(f"Failed:  {n_fail}")
    ok(f"Time:    {total_t / 60:.1f} min  ({total_t / max(n_ok, 1):.2f}s per shape)")

    files   = [f for f in os.listdir(args.out_dir) if f.endswith(".pt")]
    shapes  = len(set(f.split("_")[0] for f in files))
    total_b = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in files)
    ok(f"Output:  {args.out_dir}/  —  {len(files)} files  {shapes} shapes  "
       f"({total_b / 1024**2:.1f} MB)")
    print(f"\n{C}  Next → Phase 3: attention supervision{RESET}\n")


if __name__ == "__main__":
    main()
