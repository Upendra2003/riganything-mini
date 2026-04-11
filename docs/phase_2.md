# Phase 2: Shape and Skeleton Tokenizers
<img width="1137" height="884" alt="phase2_mlp" src="https://github.com/user-attachments/assets/02489d09-f355-4410-8037-92bb34d2c74c" />

## Overview

Phase 2 converts the raw point cloud and skeleton arrays from Phase 1 into fixed-size token tensors that the transformer in Phase 3 can consume directly.

Source: `pointClouds/obj_remesh/` (`_pointcloud.npy` + `_skeleton.npy`)  
Output: `tokens/obj_remesh/` (`_H.pt` and `_T.pt` per shape)

---

## Running Phase 2

```bash
# Full run (all 2703 shapes, resume-safe)
python tokenizer.py --max_shapes 2703 --resume
```

---

## Output Format

All files saved to `tokens/obj_remesh/`:

| File | Shape | Description |
|------|-------|-------------|
| `<id>_H.pt` | `[1024, 1024]` | Shape tokens — Phase 3 input |
| `<id>_T.pt` | `[K, 1024]` | Skeleton tokens — Phase 3 input |

---

## Architecture

### ShapeTokenizer

Point-wise MLP applied independently to each of the 1024 surface points. No inter-point interaction at this stage.

```
[1024, 6] → Linear(6→512) → ReLU → Linear(512→1024) → [1024, 1024]
```

### SkeletonTokenizer

Encodes each joint using its own position, its parent's position, and sinusoidal positional embeddings for both indices.

```
joint_mlp:    Linear(3→512) → ReLU → Linear(512→1024)
combiner_mlp: Linear(4096→2048) → ReLU → Linear(2048→1024)

per joint k:
  jk_feat  = joint_mlp(positions[k])           # [1024]
  jpk_feat = joint_mlp(positions[parent_idx])  # [1024]
  gamma_k  = sinusoidal(k,          d=1024)    # [1024]
  gamma_pk = sinusoidal(parent_idx, d=1024)    # [1024]
  T[k] = combiner_mlp(concat([jk_feat, gamma_k, jpk_feat, gamma_pk]))
```

**Sinusoidal embedding:**
```
gamma(k)_2i   = sin(k / 10000^(2i/d))
gamma(k)_2i+1 = cos(k / 10000^(2i/d))
```

---

## Key Details

- Models are randomly initialized — Phase 2 is preprocessing only (no training yet)
- Parent index in `_skeleton.npy` is 1-indexed — converted to 0-indexed before array lookup
- Both H and T use `d=1024` so they can be concatenated in Phase 3's transformer
- `d=1024` is the global transformer dimension used throughout all phases

---

## Parameter Count

| Module | Parameters |
|--------|-----------|
| Shape tokenizer MLP | 527,360 |
| joint_mlp (shared) | 525,824 |
| combiner_mlp | 10,485,760 |
| **Total** | **11,538,944 ≈ 11.5M** |

Breakdown:
- Shape tokenizer: `6×512 + 512×1024 = 3,072 + 524,288 = 527,360`
- joint_mlp: `3×512 + 512×1024 = 1,536 + 524,288 = 525,824`
- combiner_mlp: `4096×2048 + 2048×1024 = 8,388,608 + 2,097,152 = 10,485,760`

---

## My Learning

### Phase 2: Shape and Skeleton Tokenizers

**Why higher dimension?**

The transformer's attention mechanism requires all tokens to live in the same dimensional space, and that dimension is `d = 1024` throughout the entire model.

If your shape tokens are `[1024, 6]` and your skeleton tokens are `[K, 1024]`, you simply cannot concatenate them into a single sequence for the transformer — the dimensions don't match. The transformer has one fixed `d` it uses everywhere, and everything must conform to it.

There's also a representational reason. A raw 6-dim vector is too small to carry enough information through 12 layers of attention + FFN. The 1024-dim space gives each point enough "room" to encode rich geometric meaning that survives deep processing.

---

**Why the sinusoidal positional embedding?**

Your BFS skeleton is a sequence, and the model needs to know that joint #3 comes before joint #7. Sinusoidal embeddings encode order in a way the transformer can learn from.

The skeleton tokenizer's job is to look up the parent's position using that index, then build the full token.

---

**Why 4 × 1024 = 4096 before squishing?**

Each of the 4 ingredients is first independently projected to `d=1024`, giving `[K, 4096]` when concatenated. The `combiner_mlp` then squishes this back to `[K, 1024]`. This lets the model learn a non-linear interaction between all four pieces before handing it to the transformer.

---

**How concatenation enables cross-attention between geometry and skeleton:**

```python
x = concat([H, T], dim=0)  # [1024 + K, 1024]
```

This single sequence is what goes into the 12 transformer layers. Every token — whether it came from a surface point or a joint — is now a 1024-dim vector living in the same space.

The transformer then runs attention across all of them together:

```
shape token 5    can attend to →  shape token 200, joint token 3, joint token 7 ...
skeleton token 2 can attend to →  all shape tokens, skeleton token 1 (causal mask)
```

This is the whole point — by projecting everything to the same `d=1024` space, you allow cross-talk between geometry and skeleton inside the transformer. A joint token can ask "which surface points are near me?" and a surface point token can ask "which joints have been placed so far?" — all through the same attention mechanism.

If shape tokens were `[1024, 6]` and skeleton tokens were `[K, 4]`, you could never concatenate them and run unified attention. The tokenizers exist precisely to make this concatenation possible.

---

**Why not go directly 6 → 1024?**

You technically can. `Linear(6 → 1024)` would work. But the problem is it's a purely linear transformation — one matrix multiply. No matter how big the output is, a single linear layer can only learn linear relationships in the input.

The hidden layer with ReLU is what makes it non-linear:

```
6 → 512 → ReLU → 1024
```

The ReLU in the middle means the network can learn things like "if x is large AND normal points outward, then activate this feature" — conditional, non-linear geometric patterns. Without the hidden layer you lose all of that expressive power regardless of output size.

512 specifically is just a reasonable middle ground — large enough to capture complexity, small enough to be cheap. Going `6 → 1024 → ReLU → 1024` would also work but wastes parameters since a 6-dim input has limited information to begin with.

---

**Why 2048 and not 512 in the combiner?**

This is about information preservation. Look at what the combiner is receiving:

```python
input = [jk_feat, gamma_k, jpk_feat, gamma_pk]  →  [K, 4096]
```

All four ingredients are already rich 1024-dim vectors. You're trying to compress 4096 dims of meaningful information down to 1024. If you bottleneck through 512 first:

```
4096 → 512 → 1024   ← crushing 4096 dims through a 512 bottleneck
```

That 512 bottleneck forces the network to throw away 87.5% of the information before it even gets a chance to mix it. You'd lose most of what the joint position, parent position, and positional embeddings encoded.

Going through 2048 instead:

```
4096 → 2048 → 1024   ← gradual compression, information has room to mix
```

This is the general rule — when compressing, decrease gradually so the network can learn what to keep at each step rather than being forced to discard most of it in one brutal squeeze.
