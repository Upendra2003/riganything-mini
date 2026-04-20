## Phase 4: Joint Diffusion Module
<img width="2282" height="1408" alt="diffusion_module" src="https://github.com/user-attachments/assets/ac79a18c-265b-4352-ab9a-aac51e402d8e" />

**Files:** `phase4/`
**Input:** frozen Z_k from Phase 3 + ground-truth joint positions
**Output:** predicted joint position j_k ∈ R³ — runtime only
**Checkpoint:** `checkpoints/phase4/best_model.pt` (~40 MB)

### Why Diffusion

Sibling joints share the same Z_k context. An L2 regressor predicts their mean — a point in empty space. Diffusion captures the full multimodal distribution over valid next joint positions.

### Cosine Noise Schedule (M=1000, offset s=0.008)

```
f(t) = cos(((t/M + s) / (1 + s)) * π/2)²
alpha_bar_t = f(t) / f(0)
beta_t = 1 − alpha_bar_t / alpha_bar_{t-1}   (clamped to [0, 0.999])

Forward diffusion:
  j^m = sqrt(ᾱ_m) · j⁰ + sqrt(1 − ᾱ_m) · ε,   ε ~ N(0, I)
```

### DenoisingMLP

Predicts ε given (j^m, m, Z_k):

```
t_emb     = time_embed(sinusoidal(m/M))          [d]   2-layer MLP
z_ctx     = context_proj(Z_k.mean(0))            [d]
condition = t_emb + z_ctx                        [d]

x = fc1(concat(j^m, t_emb))                      [d]
x = SiLU(AdaLN(x, condition))
x = SiLU(AdaLN(fc2(x), condition))
ε_pred = out(x)                                  [3]
```

**AdaLN:** `(1 + γ) * LN(x) + β` — projection zero-initialized so the module starts as plain LayerNorm and learns deviations.

**Training objective:**
```
L_joint = E_{ε, m} [ ||ε − ε_θ(j^m | m, Z_k)||² ]
```

**Inference:** 50 DDIM steps (deterministic denoising) from j^M ~ N(0, I). j0_est clamped to [−3, 3] for numerical stability.

### Training Design

- Phase 3 transformer is fully frozen (`eval()` + `requires_grad_(False)`) — only DenoisingMLP is trained
- Per-joint inner loop: `zero_grad` once per shape, `loss.backward()` after each joint k, `optimizer.step()` after all K joints — avoids holding K computation graphs in memory simultaneously
- AMP (fp16 forward, fp32 loss) with `GradScaler`
- Linear warmup (100 steps) → cosine decay via `LambdaLR`
- Validation: 5 random joints × 5 random timesteps per shape

### Parameter Count (~10.4M)

| Layer | Params |
|-------|--------|
| `time_embed` (2 × Linear d→d) | 2,099,200 |
| `context_proj` (Linear d→d) | 1,049,600 |
| `adaLN1.proj` (Linear d→2d) | 2,099,200 |
| `adaLN2.proj` (Linear d→2d) | 2,099,200 |
| `fc1` (Linear 1027→d) | 1,052,672 |
| `fc2` (Linear d→d) | 1,049,600 |
| `out` (Linear d→3) | 3,075 |

### Run

```bash
# Train
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python phase4/train.py --epochs 50

# Resume
.venv/bin/python phase4/train.py --resume checkpoints/phase4/best_model.pt

# Cluster (SLURM)
sbatch slurm/phase4_train.sh

# Tests (11 structural, no training required)
.venv/bin/python tests/phase4_test.py
```
