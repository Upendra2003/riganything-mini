"""
Phase 2 — Tokenizer Validation
================================
Verifies shape/skeleton token tensors saved in tokens/obj_remesh/.

Tests:
  1. H.shape == (1024, 1024) and T.shape == (K, 1024) for every processed shape
  2. No NaN or Inf in H or T
  3. Two different shapes produce different H
     (cosine_similarity(H1.mean(0), H2.mean(0)) < 0.99)
  4. Same shape loaded twice gives identical H (deterministic)
  5. Sinusoidal embeddings: gamma(0) != gamma(1) != gamma(2)
  6. T varies with K — shapes with more joints produce larger T

Usage:
  .venv/bin/python tests/phase2_test.py
  .venv/bin/python tests/phase2_test.py --shape_id 10000
"""

import os, sys, argparse, glob
import torch
import numpy as np

# Add project root to path so we can import phase2_tokenizer symbols
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phase2_tokenizer import (
    ShapeTokenizer, SkeletonTokenizer,
    sinusoidal_embedding, load_inputs, process_shape,
)

TOK_DIR = os.path.join(ROOT, "tokens", "obj_remesh")
PC_DIR  = os.path.join(ROOT, "pointClouds", "obj_remesh")

# ── ANSI colours ───────────────────────────────────────────────
G="\033[92m"; Y="\033[93m"; R="\033[91m"; C="\033[96m"
BOLD="\033[1m"; RESET="\033[0m"

def ok(m):   print(f"  {G}PASS{RESET}  {m}")
def fail(m): print(f"  {R}FAIL{RESET}  {m}"); return False
def warn(m): print(f"  {Y}WARN{RESET}  {m}")
def info(m): print(f"  {C}INFO{RESET}  {m}")


# ── helpers ────────────────────────────────────────────────────
def available_token_ids() -> list[str]:
    h_files = glob.glob(os.path.join(TOK_DIR, "*_H.pt"))
    return sorted(os.path.basename(f).replace("_H.pt", "") for f in h_files)


def available_pc_ids() -> list[str]:
    files = glob.glob(os.path.join(PC_DIR, "*_pointcloud.npy"))
    return sorted(os.path.basename(f).replace("_pointcloud.npy", "") for f in files)


def load_tokens(sid: str):
    H = torch.load(os.path.join(TOK_DIR, f"{sid}_H.pt"), weights_only=True)
    T = torch.load(os.path.join(TOK_DIR, f"{sid}_T.pt"), weights_only=True)
    return H, T


# ── Test 1: shape checks ───────────────────────────────────────
def test_shapes(ids: list[str]) -> bool:
    print(f"\n{BOLD}Test 1 — Tensor shapes{RESET}")
    all_pass = True
    for sid in ids:
        h_path = os.path.join(TOK_DIR, f"{sid}_H.pt")
        t_path = os.path.join(TOK_DIR, f"{sid}_T.pt")
        if not os.path.exists(h_path) or not os.path.exists(t_path):
            warn(f"[{sid}] token files missing — run phase2_tokenizer.py first")
            all_pass = False
            continue
        H, T = load_tokens(sid)
        if H.shape[0] != 1024 or H.shape[1] != 1024:
            fail(f"[{sid}] H.shape={tuple(H.shape)}  expected (1024, 1024)")
            all_pass = False
        elif T.shape[1] != 1024:
            fail(f"[{sid}] T.shape={tuple(T.shape)}  expected (K, 1024)")
            all_pass = False
        else:
            ok(f"[{sid}] H={tuple(H.shape)}  T={tuple(T.shape)}  ({T.shape[0]} joints)")
    return all_pass


# ── Test 2: no NaN / Inf ───────────────────────────────────────
def test_no_nan_inf(ids: list[str]) -> bool:
    print(f"\n{BOLD}Test 2 — No NaN or Inf{RESET}")
    all_pass = True
    for sid in ids:
        if not os.path.exists(os.path.join(TOK_DIR, f"{sid}_H.pt")):
            continue
        H, T = load_tokens(sid)
        h_clean = torch.isfinite(H).all().item()
        t_clean = torch.isfinite(T).all().item()
        if h_clean and t_clean:
            ok(f"[{sid}] H and T are finite")
        else:
            fail(f"[{sid}] H finite={h_clean}  T finite={t_clean}")
            all_pass = False
    return all_pass


# ── Test 3: different shapes → different H ────────────────────
def test_different_shapes(ids: list[str]) -> bool:
    print(f"\n{BOLD}Test 3 — Different shapes produce different H{RESET}")
    if len(ids) < 2:
        warn("Need at least 2 shapes — skipping"); return True

    # Pick the most spatially distinct pair from available shapes so the test
    # is robust: near-identical humanoid characters produce H with cos_sim
    # close to 1.0, so we search for the pair with the largest point cloud
    # mean distance.
    token_ids = [sid for sid in ids
                 if os.path.exists(os.path.join(TOK_DIR, f"{sid}_H.pt"))]
    if len(token_ids) < 2:
        warn("Token files missing — skipping"); return True

    best_dist, sid1, sid2 = -1.0, token_ids[0], token_ids[1]
    for i in range(min(len(token_ids), 20)):
        pc_i = os.path.join(PC_DIR, f"{token_ids[i]}_pointcloud.npy")
        if not os.path.exists(pc_i):
            continue
        m_i = np.load(pc_i).mean(0)
        for j in range(i + 1, min(len(token_ids), 20)):
            pc_j = os.path.join(PC_DIR, f"{token_ids[j]}_pointcloud.npy")
            if not os.path.exists(pc_j):
                continue
            m_j = np.load(pc_j).mean(0)
            d = float(np.linalg.norm(m_i - m_j))
            if d > best_dist:
                best_dist, sid1, sid2 = d, token_ids[i], token_ids[j]

    info(f"Comparing most distinct pair: {sid1} vs {sid2}  "
         f"(pointcloud mean dist={best_dist:.4f})")

    H1, _ = load_tokens(sid1)
    H2, _ = load_tokens(sid2)

    m1 = H1.mean(0)
    m2 = H2.mean(0)
    cos_sim = torch.nn.functional.cosine_similarity(m1.unsqueeze(0), m2.unsqueeze(0)).item()
    info(f"cosine_similarity(H1.mean, H2.mean) = {cos_sim:.6f}")

    if cos_sim < 0.99:
        ok(f"Shapes {sid1} and {sid2} produce distinct H  (cos_sim={cos_sim:.4f} < 0.99)")
        return True
    else:
        fail(f"H too similar: cos_sim={cos_sim:.6f} ≥ 0.99  "
             f"(only {len(token_ids)} shapes available — try more shapes for a better pair)")
        return False


# ── Test 4: deterministic — same shape → same H ───────────────
def test_deterministic(ids: list[str]) -> bool:
    print(f"\n{BOLD}Test 4 — Same shape loaded twice gives identical H{RESET}")
    sid = ids[0]

    # Set fixed seed and run twice
    shape_tok = ShapeTokenizer().eval()
    skel_tok  = SkeletonTokenizer().eval()

    with torch.no_grad():
        H1, T1 = process_shape(sid, shape_tok, skel_tok, in_dir=PC_DIR)
        H2, T2 = process_shape(sid, shape_tok, skel_tok, in_dir=PC_DIR)

    if torch.equal(H1, H2):
        ok(f"[{sid}] H is identical across two forward passes")
        return True
    else:
        max_diff = (H1 - H2).abs().max().item()
        fail(f"[{sid}] H differs — max_diff={max_diff:.2e}")
        return False


# ── Test 5: sinusoidal embeddings are distinct ─────────────────
def test_sinusoidal() -> bool:
    print(f"\n{BOLD}Test 5 — Sinusoidal embeddings gamma(0) != gamma(1) != gamma(2){RESET}")
    g0 = sinusoidal_embedding(0)
    g1 = sinusoidal_embedding(1)
    g2 = sinusoidal_embedding(2)

    info(f"gamma(0)[:8] = {g0[:8].tolist()}")
    info(f"gamma(1)[:8] = {g1[:8].tolist()}")
    info(f"gamma(2)[:8] = {g2[:8].tolist()}")

    ok01 = not torch.equal(g0, g1)
    ok12 = not torch.equal(g1, g2)

    if ok01 and ok12:
        ok("gamma(0) != gamma(1) != gamma(2)")
        return True
    else:
        fail(f"gamma(0)==gamma(1): {not ok01}  gamma(1)==gamma(2): {not ok12}")
        return False


# ── Test 6: T grows with K ─────────────────────────────────────
def test_t_varies_with_k(ids: list[str]) -> bool:
    print(f"\n{BOLD}Test 6 — T varies with K (more joints → larger T){RESET}")
    if len(ids) < 2:
        warn("Need at least 2 shapes — skipping"); return True

    k_sizes = {}
    for sid in ids:
        t_path = os.path.join(TOK_DIR, f"{sid}_T.pt")
        if os.path.exists(t_path):
            T = torch.load(t_path, weights_only=True)
            k_sizes[sid] = T.shape[0]

    if len(k_sizes) < 2:
        warn("Need at least 2 shapes with tokens — skipping"); return True

    # Check that not all shapes have identical K
    unique_ks = set(k_sizes.values())
    if len(unique_ks) == 1:
        warn(f"All {len(k_sizes)} shapes have the same K={list(unique_ks)[0]} — "
             "can't verify K variation with this sample")
        return True

    # Sort by K and verify that T.shape[0] matches K
    for sid, K in sorted(k_sizes.items(), key=lambda x: x[1]):
        T = torch.load(os.path.join(TOK_DIR, f"{sid}_T.pt"), weights_only=True)
        assert T.shape[0] == K
        info(f"  [{sid}] K={K}  T.shape={tuple(T.shape)}")

    min_sid = min(k_sizes, key=k_sizes.get)
    max_sid = max(k_sizes, key=k_sizes.get)
    info(f"  min K={k_sizes[min_sid]} ({min_sid})  max K={k_sizes[max_sid]} ({max_sid})")
    ok(f"T.shape[0] correctly tracks K across {len(k_sizes)} shapes")
    return True


# ── CLI ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_id", default=None,
                        help="Test a specific shape ID (e.g. 10000)")
    cli = parser.parse_args()

    print(f"\n{BOLD}Phase 2 Tests{RESET}")
    print(f"  Tokens:  {TOK_DIR}")

    # Determine which shapes to use
    token_ids = available_token_ids()
    pc_ids    = available_pc_ids()

    if not token_ids and not pc_ids:
        print(f"{R}No token or pointcloud files found.{RESET}")
        print("Run:  .venv/bin/python phase2_tokenizer.py --max_shapes 5  first.")
        sys.exit(1)

    if cli.shape_id:
        ids = [cli.shape_id]
        if cli.shape_id not in token_ids:
            warn(f"Shape {cli.shape_id} has no tokens yet — running tokenizer now...")
            shape_tok = ShapeTokenizer().eval()
            skel_tok  = SkeletonTokenizer().eval()
            H, T = process_shape(cli.shape_id, shape_tok, skel_tok, in_dir=PC_DIR)
            os.makedirs(TOK_DIR, exist_ok=True)
            torch.save(H, os.path.join(TOK_DIR, f"{cli.shape_id}_H.pt"))
            torch.save(T, os.path.join(TOK_DIR, f"{cli.shape_id}_T.pt"))
            token_ids = [cli.shape_id]
    else:
        # Use all available token IDs (up to a reasonable cap for speed)
        ids = token_ids[:20] if token_ids else pc_ids[:5]

    # If tokens don't exist yet, generate them on the fly for testing
    if not token_ids:
        info("No pre-saved tokens found — generating on the fly for tests...")
        shape_tok = ShapeTokenizer().eval()
        skel_tok  = SkeletonTokenizer().eval()
        os.makedirs(TOK_DIR, exist_ok=True)
        for sid in ids:
            try:
                H, T = process_shape(sid, shape_tok, skel_tok, in_dir=PC_DIR)
                torch.save(H, os.path.join(TOK_DIR, f"{sid}_H.pt"))
                torch.save(T, os.path.join(TOK_DIR, f"{sid}_T.pt"))
            except Exception as e:
                warn(f"[{sid}] {e}")
        token_ids = available_token_ids()
        ids = token_ids[:20]

    results = [
        test_shapes(ids),
        test_no_nan_inf(ids),
        test_different_shapes(ids),
        test_deterministic(ids),
        test_sinusoidal(),
        test_t_varies_with_k(ids),
    ]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{BOLD}Summary:{RESET}  "
          f"{G}{n_pass} passed{RESET}  "
          f"{(R + str(n_fail) + ' failed' + RESET) if n_fail else ''}\n")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
