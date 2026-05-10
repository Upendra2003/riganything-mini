import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = 'checkpoints/phase7/train_log.csv'

cols = ['epoch','train_total','train_joint','train_connect','train_skinning',
        'val_total','val_joint','val_connect','val_skinning']

rows = {}
with open(CSV_PATH) as f:
    for line in csv.reader(f):
        if len(line) < 9:
            continue
        try:
            ep = int(line[0])
        except ValueError:
            continue
        rows[ep] = [float(v) for v in line[1:9]]

epochs = sorted(rows)
data = np.array([rows[e] for e in epochs])

train_total, train_joint, train_connect, train_skinning = data[:,0], data[:,1], data[:,2], data[:,3]
val_total,   val_joint,   val_connect,   val_skinning   = data[:,4], data[:,5], data[:,6], data[:,7]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Phase 7 — End-to-End Training Losses\n(5 Transformer Blocks · 2000 Dataset · 160 Epochs)',
             fontsize=14, fontweight='bold', y=0.98)

configs = [
    (axes[0,0], 'Total Loss',        train_total, val_total),
    (axes[0,1], 'Joint Loss',        train_joint, val_joint),
    (axes[1,0], 'Connectivity Loss', train_connect, val_connect),
    (axes[1,1], 'Skinning Loss',     train_skinning, val_skinning),
]

for ax, title, train, val in configs:
    ax.plot(epochs, train, color='#4a9eff', linewidth=1.5, label='Train', alpha=0.9)
    ax.plot(epochs, val,   color='#ff6b4a', linewidth=1.5, label='Val',   alpha=0.9)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
out = 'checkpoints/phase7/training_curves.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
