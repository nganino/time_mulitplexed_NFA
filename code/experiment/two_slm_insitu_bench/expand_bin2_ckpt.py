"""2 x 2 binned checkpoint -> the equivalent full-resolution checkpoint the package runs (2026-11-11).

The bin-2 models are trained with 2 x 2 macro-pixels: key 207 px of 16.155 um (PLUTO), decoder 368 px of 9 um (LUNA),
encoding bin 55.  In the model a macro-pixel is exactly 2 x 2 SLM px of equal phase, so repeating every macro-pixel 2 x 2
gives the key 414 x 414, decoder 736 x 736 and encoding bin 110 that insitu_TDNFA.py displays: no package code changes.

The expanded config uses the 1 x 1 bench geometry (key_slm_dx 8.075 um).  The exact half of 16.155 um is 8.0775 um, which
the spectral propagator cannot use (non-integer nm); the 0.03 % pitch difference is checked below by simulating both.

Run from code/:  python experiment/two_slm_insitu_bench/expand_bin2_ckpt.py <bin2 .pth> <out .pth>
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import init_params, recompute_derived, apply_saved_config
from model import TimeMultiplexedNFA
import insitu_config
from insitu_policy import compute_reward

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
c = ck['config']
assert c['phase_key_size'] == 207 and c['decoder_size_px'] == 368 and c['encoding_patch_scale'] == 55, 'not a bin-2 checkpoint'
up = lambda t: t.repeat_interleave(2, -2).repeat_interleave(2, -1)
model = dict(ck['model'])
model['slm_phases'] = up(ck['model']['slm_phases'])           # raw (pre-sigmoid) key phases [M, 1, 414, 414]
model['decoder_phases'] = up(ck['model']['decoder_phases'])   # [M, 1, 736, 736]
c2 = dict(c, key_slm_dx=8.075e-6, enc_slm_dx=4.5e-6, phase_key_size=414, decoder_size=736, decoder_size_px=736,
          encoding_patch_scale=110, expanded_from_bin2=os.path.basename(src))
torch.save({'epoch': ck['epoch'], 'best_val_loss': ck.get('best_val_loss'), 'config': c2, 'model': model,
            'readout': ck.get('readout')}, dst)
print(f'wrote {dst}: keys {tuple(model["slm_phases"].shape)}, decoders {tuple(model["decoder_phases"].shape)}, epoch {ck["epoch"]}')

# ---- check: same reward from the bin-2 model and the expanded model (linear phases, 100 a, as sim_check) ----
torch.set_grad_enabled(False)
dev = torch.device('cuda')
cfg = insitu_config.init_params()
pkg = os.path.dirname(os.path.abspath(__file__))
T = torch.from_numpy(np.load(os.path.join(pkg, 'targets.npy'))).float().to(dev)
a = torch.as_tensor(cfg.a, dtype=torch.float32, device=dev)
for name, cfgd, sd in (('bin-2 model', c, ck['model']), ('expanded', c2, model)):
    tc = init_params(); apply_saved_config(tc, cfgd); tc.device = 'cuda'; recompute_derived(tc)
    m = TimeMultiplexedNFA(tc).to(dev); m.load_state_dict(sd, strict=False); m.eval()
    I = torch.cat([m(a[i:i + 1]).mean(1) for i in range(len(a))]).reshape(len(a), -1)
    print(f'  {name:12s}: affine (reward) RMSE {-compute_reward(I[None], T, cfg)[0].item():.3e}', flush=True)
    del m; torch.cuda.empty_cache()
