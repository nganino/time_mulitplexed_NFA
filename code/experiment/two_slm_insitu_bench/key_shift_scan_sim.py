"""SIM ONLY (no hardware): reward RMSE (affine) vs key shift of 0.5 / 1 / 2 PLUTO px in x and y for the bench M2 / M3 checkpoints
(linear phases), by moving the key in the spectral propagator. Writes key_shift_scan.txt. Needs a GPU and this repo's code/ folder.
Run from code/: python experiment/two_slm_insitu_bench/key_shift_scan_sim.py"""
import os, sys
import numpy as np
import torch
PKG = r'P:\Nicholas\time_multiplexed_NFA\code\experiment\two_slm_insitu_bench'
sys.path.insert(0, r'P:\Nicholas\time_multiplexed_NFA\code'); sys.path.insert(0, PKG)
from config import init_params, recompute_derived, apply_saved_config
from model import TimeMultiplexedNFA
from wave_prop import SpectralLabProp
import insitu_config
from insitu_policy import compute_reward
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(0.38, 0)
dev = torch.device('cuda')
cfg = insitu_config.init_params()
a_all = torch.as_tensor(cfg.a, dtype=torch.float32, device=dev)
T = torch.from_numpy(np.load(os.path.join(PKG, 'targets.npy'))).float().to(dev)
aff = lambda I: -compute_reward(I[None], T, cfg)[0].item()
out = []
for ck_path in ('ckpt_M2/bench_M2_bin1_ep4.pth', 'ckpt_M3/bench_M3_bin1_ep2.pth'):
    ck = torch.load(os.path.join(PKG, ck_path), map_location='cpu', weights_only=False)
    tc = init_params(); apply_saved_config(tc, ck['config']); tc.device = 'cuda'; recompute_derived(tc)
    m = TimeMultiplexedNFA(tc).to(dev); m.load_state_dict(ck['model'], strict=False); m.eval()
    g0, p1 = dict(m.spec_geom), m.spec_geom['p1']
    rd = lambda: torch.cat([m(a_all[i:i + 1]).mean(1) for i in range(len(a_all))]).reshape(len(a_all), -1)
    res = {0: aff(rd())}
    del m.spec; torch.cuda.empty_cache()
    for s in (0.5, 1, 2):
        for ax in ('x', 'y'):
            m.spec = SpectralLabProp(tc, dict(g0, **{f'key_{ax}0': g0[f'key_{ax}0'] + s * p1})).to(dev)
            res[(s, ax)] = aff(rd())
            del m.spec; torch.cuda.empty_cache()
    line = f'M{tc.M}: shift 0: {res[0]:.2e}; ' + '; '.join(f'{s} px {ax}: {res[(s, ax)]:.2e}' for s in (0.5, 1, 2) for ax in ('x', 'y'))
    print(line, flush=True); out.append(line)
    del m; torch.cuda.empty_cache()
open(os.path.join(PKG, 'key_shift_scan.txt'), 'w').write(
    'Reward (affine) RMSE vs PLUTO key shift (linear phases, sim), 100 a; 3 px gives the no-information floor 8.7e-2\n' + '\n'.join(out) + '\n')
