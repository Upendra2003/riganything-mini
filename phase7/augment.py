"""
Phase 7 — Online Pose Augmentation
====================================
Applies random per-joint rotations to a rigged shape, deforming the surface
points via Linear Blend Skinning (LBS).  Run once per training step to force
the model to learn pose-invariant features.

Pipeline:
  1. Sample K random rotation matrices (axis-angle, |angle| ≤ max_degrees).
  2. forward_kinematics: propagate rotations through the BFS skeleton tree
     → new joint positions + global rotation at each joint.
  3. lbs_deform: blend the per-joint rigid transforms weighted by skin weights
     → deformed surface points.
  4. Recompute normals from deformed points (KNN cross-product estimate).

LBS formula  (full rotation):
  p'_l = Σ_k  w_{l,k} · (R_k^global · (p_l − j_k) + j'_k)

where R_k^global is the global rotation accumulated from the root to joint k
following the kinematic chain.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Random rotation matrix
# ---------------------------------------------------------------------------

def random_rotation_matrix(max_degrees: float = 45.0, device=None) -> torch.Tensor:
    """
    Sample a random 3×3 rotation matrix with rotation angle ≤ max_degrees.

    Returns: [3, 3] float32 tensor
    """
    max_rad = math.radians(max_degrees)
    angle   = torch.rand(1).item() * max_rad          # uniform in [0, max_rad]

    # Random unit axis
    axis = torch.randn(3)
    axis = axis / (axis.norm() + 1e-8)

    # Rodrigues rotation formula
    ax, ay, az = axis[0].item(), axis[1].item(), axis[2].item()
    c, s = math.cos(angle), math.sin(angle)
    t    = 1.0 - c

    R = torch.tensor([
        [t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c   ],
    ], dtype=torch.float32)

    if device is not None:
        R = R.to(device)
    return R


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(
    joints:    torch.Tensor,   # [K, 3]  rest-pose joint positions
    parents:   torch.Tensor,   # [K]     1-indexed (root = 1, self-ref)
    rotations: torch.Tensor,   # [K, 3, 3]  local rotation per joint
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate local rotations through the BFS skeleton tree.

    Args:
        joints:    [K, 3]   rest-pose positions (BFS order)
        parents:   [K]      1-indexed parent indices
        rotations: [K, 3,3] local rotation matrices

    Returns:
        new_joints:  [K, 3]   posed joint positions
        global_rots: [K, 3,3] global (accumulated) rotation at each joint
    """
    K      = joints.shape[0]
    device = joints.device
    dtype  = joints.dtype

    new_joints  = torch.zeros_like(joints)
    global_rots = torch.zeros(K, 3, 3, device=device, dtype=dtype)

    # BFS is already encoded in the ordering (parent index < child index)
    for k in range(K):
        p_1indexed = int(parents[k].item())
        p          = p_1indexed - 1          # 0-indexed parent

        if k == 0 or p == k:
            # Root: keep position, global rotation = local rotation
            new_joints[k]  = joints[k]
            global_rots[k] = rotations[k]
        else:
            # Child: rotate around parent using parent's global rotation
            R_parent = global_rots[p]                           # [3, 3]
            local_offset = joints[k] - joints[p]               # [3]
            new_joints[k] = new_joints[p] + R_parent @ local_offset
            # Accumulate: global = parent_global @ local
            global_rots[k] = R_parent @ rotations[k]

    return new_joints, global_rots


# ---------------------------------------------------------------------------
# Linear Blend Skinning
# ---------------------------------------------------------------------------

def lbs_deform(
    points:       torch.Tensor,   # [L, 3]
    skin_weights: torch.Tensor,   # [L, K]
    old_joints:   torch.Tensor,   # [K, 3]
    new_joints:   torch.Tensor,   # [K, 3]
    global_rots:  torch.Tensor | None = None,  # [K, 3, 3]  optional
) -> torch.Tensor:
    """
    Deform surface points via Linear Blend Skinning.

    Full formula (when global_rots provided):
      p'_l = Σ_k  w_{l,k} · (R_k · (p_l − j_k) + j'_k)

    Simplified formula (when global_rots=None — displacement LBS):
      p'_l = p_l + Σ_k  w_{l,k} · (j'_k − j_k)

    Returns: [L, 3] deformed points
    """
    if global_rots is None:
        # Fast displacement-only LBS: O(L·K) matrix multiply
        delta = new_joints - old_joints          # [K, 3]
        return points + skin_weights @ delta     # [L, 3]

    # Full LBS with rotation
    L = points.shape[0]
    K = old_joints.shape[0]
    new_points = torch.zeros_like(points)

    for k in range(K):
        R_k    = global_rots[k]                                        # [3, 3]
        j_k    = old_joints[k]                                         # [3]
        j_new  = new_joints[k]                                         # [3]
        w_k    = skin_weights[:, k:k + 1]                              # [L, 1]
        local  = (points - j_k) @ R_k.T + j_new                       # [L, 3]
        new_points = new_points + w_k * local

    return new_points


# ---------------------------------------------------------------------------
# Recompute normals from deformed point cloud
# ---------------------------------------------------------------------------

def recompute_normals(points: torch.Tensor, k_nn: int = 4) -> torch.Tensor:
    """
    Estimate per-point outward normals via KNN cross product.

    For each point finds k_nn nearest neighbors, computes the mean normal
    from cross products of edge vectors, and flips inward-pointing normals.

    Args:
        points: [L, 3]
    Returns:
        normals: [L, 3]  (approximately unit length)
    """
    L   = points.shape[0]
    dev = points.device

    # Pairwise squared distances
    diff  = points.unsqueeze(0) - points.unsqueeze(1)   # [L, L, 3]
    dists = (diff * diff).sum(-1)                        # [L, L]
    # Set diagonal to large value to exclude self
    dists.fill_diagonal_(float('inf'))

    # k nearest neighbours (indices)
    kk     = min(k_nn, L - 1)
    _, idx = dists.topk(kk, dim=1, largest=False)       # [L, kk]

    # Edge vectors from each point to its first two neighbours
    e1 = points[idx[:, 0]] - points                     # [L, 3]
    e2 = points[idx[:, 1]] - points                     # [L, 3]

    normals = torch.linalg.cross(e1, e2)                # [L, 3]
    normals = F.normalize(normals, dim=-1, eps=1e-8)    # unit length

    # Flip normals pointing toward mesh centroid (inward)
    centroid = points.mean(dim=0, keepdim=True)         # [1, 3]
    inward   = ((centroid - points) * normals).sum(-1) > 0  # [L]
    normals[inward] = -normals[inward]

    return normals


# ---------------------------------------------------------------------------
# Full augmentation pipeline
# ---------------------------------------------------------------------------

def augment_shape(
    points:       torch.Tensor,   # [L, 3]
    joints:       torch.Tensor,   # [K, 3]
    parents:      torch.Tensor,   # [K]   1-indexed
    skin_weights: torch.Tensor,   # [L, K]
    max_degrees:  float = 45.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply random pose augmentation and recompute normals.

    Returns:
        new_points:  [L, 3]  deformed surface points
        new_normals: [L, 3]  recomputed unit normals
        new_joints:  [K, 3]  posed joint positions
    """
    K      = joints.shape[0]
    device = joints.device

    # 1. Sample K random rotation matrices
    rotations = torch.stack([
        random_rotation_matrix(max_degrees, device=device) for _ in range(K)
    ])                                                    # [K, 3, 3]

    # 2. Forward kinematics → new joint positions + global rotations
    new_joints, global_rots = forward_kinematics(joints, parents, rotations)

    # 3. LBS deformation (full rotation)
    new_points = lbs_deform(points, skin_weights, joints, new_joints, global_rots)

    # 4. Recompute normals
    new_normals = recompute_normals(new_points)

    return new_points, new_normals, new_joints
