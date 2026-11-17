"""Write targets.npy for the in-situ loop: the checkpoint's Nf target functions on the insitu_config a grid,
min-max'd over that grid exactly as test.py does ([len(cfg.a), Nf], k = r * 64 + c).  Run once per checkpoint
from this folder, in the training environment (it imports the simulation code's config / target_functions):

    python export_targets.py --ckpt ./ckpt/best.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # code/
from config import init_params, apply_saved_config, recompute_derived
from target_functions import TargetFunctionSet
import insitu_config

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--out', default='./targets.npy')
args = p.parse_args()

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
tc = init_params()
apply_saved_config(tc, ck['config'])
tc.device = 'cpu'
recompute_derived(tc)

a = torch.as_tensor(insitu_config.init_params().a, dtype=torch.float64)
t = TargetFunctionSet(tc)(a).double()                       # [len(a), Nf]
t = (t - t.min()) / (t.max() - t.min())                     # test.py: targets min-max'd over the same grid
np.save(args.out, t.float().numpy())
print(f'wrote {args.out}: {tuple(t.shape)} targets on {len(a)} a values in [{a[0]:.3f}, {a[-1]:.3f}]')
