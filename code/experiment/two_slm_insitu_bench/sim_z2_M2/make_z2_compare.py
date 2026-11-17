"""2026-11-13: is z2 off?  Expected camera images of the existing bench M2 masks (ckpt_M2/bench_M2_bin1_ep4.pth: Nf 4096,
key 414, decoder 736, encoding bin 110 tiled 3 x 3, trained at z1 35.87 / z2 175.35 mm) propagated to the camera at
z2 = 175.35 mm (trained) and 175.85 mm (+0.5 mm), everything else unchanged.  Camera simulation = sim_check_M2/make_sim_check.py
step 4 (spectral propagator, every camera px a 5.86 um box), RAW Basler orientation, mean over the M = 2 frames.

Writes into this folder:
  cam_z<z2>_a<+x.xxx>.npy   expected raw camera frame [1200, 1920] (float16, units of the incident beam), for comparing with
                            captured frames
  z2_compare.png            full sensor, both z2, three a values, + difference
  z2_compare_zoom.png       central 8 x 8 detectors with the detector grid, both z2
  report.txt                100-a RMSE (min-max and affine = reward) at both z2, linear and phase-table phases
Run (from this folder): /c/Anaconda/envs/qpi/python.exe make_z2_compare.py [phases: table (default) | linear]"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.abspath(os.path.join(PKG, '..', '..')))      # code/
sys.path.insert(0, PKG)
from config import init_params, recompute_derived, apply_saved_config
from model import TimeMultiplexedNFA
from wave_prop import SpectralLabProp
import insitu_config
import insitu_TDNFA as TD
from insitu_policy import compute_reward

CKPT = os.path.join(PKG, 'ckpt_M2', 'bench_M2_bin1_ep4.pth')
Z2S = (0.17535, 0.17585)
PHASES = sys.argv[1] if len(sys.argv) > 1 else 'table'
CAM_DX, NCAM = 5.86e-6, 2048
A_IDX = [0, 25, 50, 75]   # a -0.5, -0.247, +0.005, +0.258 (a = +-0.5 give the same frame: integer harmonics)
dev = torch.device('cuda')
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(0.80, 0)
TWO_PI = 2 * np.pi
cfg = insitu_config.init_params()
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
tc = init_params(); apply_saved_config(tc, ck['config']); tc.device = 'cuda'; recompute_derived(tc)
m = TimeMultiplexedNFA(tc).to(dev)
miss, _ = m.load_state_dict(ck['model'], strict=False)
assert not [k for k in miss if 'phases' in k], miss
m.eval(); M = tc.M
say(f'checkpoint {os.path.relpath(CKPT, PKG)} (epoch {ck["epoch"]}): M {M}, trained z1 {1e3 * tc.key_to_enc_spacing:g} / '
    f'z2 {1e3 * tc.last_layer_ccd_spacing:g} mm, key {tc.phase_key_size}, decoder {tc.decoder_size_px}, bin {tc.encoding_patch_scale}')
a_all = torch.as_tensor(cfg.a, dtype=torch.float32, device=dev)
T = torch.from_numpy(np.load(os.path.join(PKG, 'targets.npy'))).float().to(dev)

key_lin, inp_lin = m.key_phases, m._input_plane_px
tabP, lutP = torch.tensor(cfg.pluto_phase_table, device=dev, dtype=torch.float32), torch.tensor(cfg.pluto_lut.astype(np.int64), device=dev)
tabL, lutL = torch.tensor(cfg.luna_phase_table, device=dev, dtype=torch.float32), torch.tensor(cfg.luna_lut.astype(np.int64), device=dev)
def snap(phi, lut, tab):
    j = torch.round(torch.remainder(-phi, TWO_PI) / (TWO_PI / 4096)).long() % 4096
    return -tab[lut[j]]
def set_variant(v):
    if v == 'linear':
        m.key_phases, m._input_plane_px = key_lin, inp_lin
    else:
        m.key_phases = lambda: snap(torch.sigmoid(m.slm_phases) * TWO_PI, lutP, tabP)
        m._input_plane_px = lambda a: snap(m._encode_px(a) + m.decoder_phases_px(), lutL, tabL)
readout = lambda a: torch.cat([m(a[i:i + 1]).mean(1) for i in range(len(a))]).reshape(len(a), -1)
def rmse_minmax(I):
    I = (I - I.min()) / (I.max() - I.min()); return ((I - T) ** 2).mean(0).sqrt().mean().item()
rmse_affine = lambda I: -compute_reward(I[None], T, cfg)[0].item()

geom0, spec0 = dict(m.spec_geom), m.spec
say('100-a RMSE vs z2 (min-max / affine = reward):')
for z2 in Z2S:
    m.spec = SpectralLabProp(tc, dict(geom0, z2=z2)).to(dev)
    for v in ('linear', 'table'):
        set_variant(v); I = readout(a_all)
        say(f'  z2 {1e3 * z2:.2f} mm, {v:6s} phases: min-max {rmse_minmax(I):.3e}, affine {rmse_affine(I):.3e}')
    del m.spec; torch.cuda.empty_cache()
m.spec = None; torch.cuda.empty_cache()

@torch.no_grad()
def camera(a, z2, chunk=256):
    geom = dict(geom0, z2=z2, det_size=CAM_DX)
    geom['det_cy'] = geom['det_cx'] = (np.arange(NCAM) - (NCAM - 1) / 2) * CAM_DX
    sp = SpectralLabProp(tc, geom).to('cpu')
    V1, v1s = sp.key_field_L(torch.exp(1j * m.key_phases()[:, 0].cpu()) - 1)
    d_all = torch.exp(1j * m._input_plane_px(torch.tensor([a], device=dev)).cpu()) - 1
    acc = 0
    for mm in range(M):
        d = d_all[:, mm:mm + 1] if d_all.shape[1] > 1 else d_all
        Q, L = sp.Q2, sp.L
        DL = sp._to_L(torch.fft.fft2(torch.nn.functional.pad(d, (0, Q - d.shape[-1], 0, Q - d.shape[-2]))), sp.i2, sp.e2y, sp.e2x)
        conv = torch.fft.fft2(v1s[mm][None, None] * torch.fft.ifft2(DL)) * (L * L * sp.df2)
        sel = lambda X: X.index_select(-2, sp.iL2).index_select(-1, sp.iL2)
        A = ((sp.c1 * sel(DL) + sel(V1[mm:mm + 1])[None] + sel(conv)) * sp.H2T)[0, 0]
        del DL, conv
        AX = A @ sp.ExT
        rows = []
        for r in range(0, NCAM, chunk):
            u = sp.c12 + (sp.Ey[r * sp.ngl:(r + chunk) * sp.ngl] @ AX) * sp.df2
            I = (u.real ** 2 + u.imag ** 2).reshape(-1, sp.ngl, NCAM, sp.ngl)
            rows.append(torch.einsum('ipjq,pq->ij', I, sp.wq))
        acc = acc + torch.cat(rows).numpy()
        del A, AX
    del sp, V1, v1s
    return acc / M

def to_raw(Imod):
    Ar, Ac = cfg.det_axis_yx
    i, j = np.arange(cfg.sen_x_num), np.arange(cfg.sen_y_num)
    mr = Ar + 1023 - i if cfg.cam_flip_ud else i - Ar + 1024
    mc = Ac + 1023 - j if cfg.cam_flip_lr else j - Ac + 1024
    return Imod[mr[:, None], mc[None, :]]

set_variant(PHASES)
say(f'\ncamera images ({PHASES} phases), raw orientation, axis {cfg.det_axis_yx}, flips {cfg.cam_flip_ud, cfg.cam_flip_lr}:')
raw = {}
Ar, Ac = cfg.det_axis_yx
for z2 in Z2S:
    for ia in A_IDX:
        t0 = time.time()
        raw[z2, ia] = to_raw(camera(float(cfg.a[ia]), z2))
        np.save(os.path.join(HERE, f'cam_z{1e3 * z2:.2f}_a{cfg.a[ia]:+.3f}.npy'), raw[z2, ia].astype(np.float16))
        arr = raw[z2, ia][Ar - 254:Ar + 254, Ac - 254:Ac + 254]
        say(f'  z2 {1e3 * z2:.2f} mm, a {cfg.a[ia]:+.3f}: sensor mean {raw[z2, ia].mean():.3f}, array mean {arr.mean():.4f}, '
            f'array max {arr.max():.3f}  ({time.time() - t0:.0f} s)')
for ia in A_IDX:
    d0, d1 = [TD.captures_to_detectors(raw[z2, ia][None], (0, 0), cfg, 'cpu')[0].numpy() for z2 in Z2S]
    cc = np.corrcoef(d0, d1)[0, 1]
    a0, a1 = [raw[z2, ia][Ar - 254:Ar + 254, Ac - 254:Ac + 254].ravel() for z2 in Z2S]
    say(f'  a {cfg.a[ia]:+.3f}: correlation 175.35 vs 175.85 mm: detector readings {cc:.3f}, array pixels {np.corrcoef(a0, a1)[0, 1]:.3f}')

# ---- figures
vmax = max(np.percentile(I, 99.95) for I in raw.values())
fig, ax = plt.subplots(3, len(A_IDX), figsize=(5 * len(A_IDX), 11), constrained_layout=True)
for c, ia in enumerate(A_IDX):
    for r, z2 in enumerate(Z2S):
        im = ax[r, c].imshow(raw[z2, ia], cmap='inferno', vmin=0, vmax=vmax)
        ax[r, c].add_patch(plt.Rectangle((Ac - 254 - 0.5, Ar - 254 - 0.5), 508, 508, fill=False, ec='c', lw=0.8))
        ax[r, c].set_title(f'z2 {1e3 * z2:.2f} mm, a = {cfg.a[ia]:+.3f}', fontsize=10)
    dif = raw[Z2S[1], ia].astype(np.float32) - raw[Z2S[0], ia].astype(np.float32)
    s = np.percentile(np.abs(dif), 99.9)
    imd = ax[2, c].imshow(dif, cmap='RdBu_r', vmin=-s, vmax=s)
    ax[2, c].add_patch(plt.Rectangle((Ac - 254 - 0.5, Ar - 254 - 0.5), 508, 508, fill=False, ec='k', lw=0.8))
    ax[2, c].set_title(f'175.85 - 175.35 mm, a = {cfg.a[ia]:+.3f}', fontsize=10)
fig.colorbar(im, ax=ax[:2], shrink=0.6, label='intensity / incident beam')
fig.colorbar(imd, ax=ax[2], shrink=0.8, label='difference')
fig.suptitle(f'Expected camera, bench M2 (ckpt_M2 ep4, bin 110, trained z2 175.35 mm), {PHASES} phases, RAW Basler orientation\n'
             f'cyan: detector array (508 px) around axis {cfg.det_axis_yx}; colour clipped at the 99.95th percentile', fontsize=11)
fig.savefig(os.path.join(HERE, 'z2_compare.png'), dpi=100); plt.close(fig)

ia = A_IDX[1]
fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
for c, z2 in enumerate(Z2S):
    crop = raw[z2, ia][Ar - 254 - 8:Ar + 262, Ac - 254 - 8:Ac + 262]
    ax[c].imshow(crop, cmap='gray', interpolation='nearest')
    ax[c].add_collection(matplotlib.collections.PatchCollection(
        [plt.Rectangle((8 + 8 * cc - 0.5, 8 + 8 * r - 0.5), 4, 4) for r in range(64) for cc in range(64)],
        facecolor='none', edgecolor='cyan', linewidth=0.3))
    ax[c].set_xlim(8 + 8 * 28 - 4, 8 + 8 * 36 + 4); ax[c].set_ylim(8 + 8 * 36 + 4, 8 + 8 * 28 - 4)
    ax[c].set_title(f'z2 {1e3 * z2:.2f} mm, a = {cfg.a[ia]:+.3f}: central 8 x 8 detectors (raw camera)', fontsize=10)
fig.savefig(os.path.join(HERE, 'z2_compare_zoom.png'), dpi=110); plt.close(fig)

open(os.path.join(HERE, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
say('done')
