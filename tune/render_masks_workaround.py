import sys, os
sys.path.append(r'P:\Nicholas\time_multiplexed_NFA\code')
os.chdir(r'P:\Nicholas\time_multiplexed_NFA\code')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import init_params, recompute_derived
from model import TimeMultiplexedClassifier

CKPT = r"P:\Nicholas\time_multiplexed_NFA\code\logs\maj_voting_C_sweep\20260917-0049-M1-C2-K3-Spacings30mm-30mm-5.0mm-30mm-batchsize12-lrslm1e-02-lrlayer1e-02-samples20000-vote\model\best.pth"
OUT_DIR = r"P:\Nicholas\time_multiplexed_NFA\code\logs\maj_voting_C_sweep\20260917-0049-M1-C2-K3-Spacings30mm-30mm-5.0mm-30mm-batchsize12-lrslm1e-02-lrlayer1e-02-samples20000-vote\test\cifar10"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = init_params()
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
config.__dict__.update(ckpt['config'])
recompute_derived(config)

model = TimeMultiplexedClassifier(config).to(device)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# --- SLM masks: rows = countries (C), cols = members (M) -- fixed orientation ---
with torch.no_grad():
    masks = (torch.sigmoid(model.slm_phases) * 2 * np.pi).cpu()
T = masks.shape[0]
M, C = model.M, model.C
rows, cols = C, M

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
im = None
m_idx = 0
for r in range(rows):
    for c in range(cols):
        ax = axes[r, c]
        if m_idx < T:
            im = ax.imshow(masks[m_idx, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
            ax.set_title(f'country {r}  member {c}')
        ax.axis('off')
        m_idx += 1
cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
cbar.set_ticks([0, np.pi, 2 * np.pi])
cbar.set_ticklabels(['0', 'pi', '2pi'])
fig.suptitle('Optimized SLM phase masks')
slm_path = os.path.join(OUT_DIR, 'slm_masks_final.png')
fig.savefig(slm_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'SLM masks saved -> {slm_path}')

# --- Diffractive layer masks (no country structure) ---
if model.num_layers > 0:
    with torch.no_grad():
        lmasks = (torch.sigmoid(model.layer_phases) * 2 * np.pi).cpu()
    K = lmasks.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3), dpi=200, squeeze=False)
    im = None
    for k in range(K):
        ax = axes[0, k]
        im = ax.imshow(lmasks[k, 0].numpy(), cmap='twilight', vmin=0, vmax=2 * np.pi)
        ax.set_title(f'layer {k}', fontsize=8)
        ax.axis('off')
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels(['0', 'pi', '2pi'])
    fig.suptitle('Diffractive layer phase masks')
    layer_path = os.path.join(OUT_DIR, 'layer_masks_final.png')
    fig.savefig(layer_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Layer masks saved -> {layer_path}')
