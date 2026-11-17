"""2 x 2 BINNED version (bin-2 checkpoints): the model runs at its native 2 x 2 macro-pixels (key 207 px of 16.155 um =
m 1.00969, decoder 368 px of 9 um), the package frames are built from the expanded checkpoint (expand_bin2_ckpt.py,
414 / 736 px) and checked against the model levels repeated 2 x 2.  Key shifts are in bench PLUTO px (8 um x m).

Sim sanity check for a collimated-bench checkpoint (2026-11-11; trained at z1 = 35.87 mm / z2 = 175.35 mm, collimated
plane wave, PLUTO pitch 8.075 um = 8 um x relay m), with the learned keys (PLUTO) and decoders (LUNA) in place.
Same checks as sim_check_M1/make_sim_check.py, trimmed to a few camera images, plus the sensitivity table of
code/TRAINING_ADJUSTMENTS_2026-11-11.md section 3.

Writes into this script's folder:
  slm_bmp/   pluto_key<m>_display.bmp, luna_dec<m>_a<+x.xxx>_display.bmp: the frames the package sends for frame m
             (display corrections ON: key shift, conjugate, phase table, PLUTO rot 180; NO collimator lens -- the bench
             copy adds collimator.json on top of the PLUTO frame)
  cmos_expected_a<+x.xxx>.npy  expected camera image, RAW Basler orientation [1200, 1920], averaged over the M frames,
             phases as realised through the phase table, units of the incident (collimated) beam intensity
  cmos_expected.png, detector_check.png, report.txt   (BMPs + array zoom: make_cmos_bmps.py, run after this)
Run (from this folder): python make_sim_check.py <native bin-2 .pth> <expanded .pth>   (paths rel. to the package)"""
import os, sys, time, copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
from PIL import Image

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

CKPT = os.path.join(PKG, sys.argv[1])                  # native bin-2 checkpoint (the model)
CKPT_X = os.path.join(PKG, sys.argv[2])                # expanded checkpoint (what the package loads and displays)
CAM_DX, NCAM = 5.86e-6, 2048                    # model-frame camera window, axis between px 1023 / 1024 (covers the sensor)
A_IDX = [0, 50, 99]                             # a values (indices into cfg.a) for the camera images
dev = torch.device('cuda')
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(0.8, 0)    # 19 GB: GPU0 is free (no training on it)
TWO_PI = 2 * np.pi
cfg = insitu_config.init_params()
os.makedirs(os.path.join(HERE, 'slm_bmp'), exist_ok=True)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)


ck = torch.load(CKPT, map_location='cpu', weights_only=False)
tc = init_params(); apply_saved_config(tc, ck['config']); tc.device = 'cuda'; recompute_derived(tc)
m = TimeMultiplexedNFA(tc).to(dev)
miss, _ = m.load_state_dict(ck['model'], strict=False)
assert not [k for k in miss if 'phases' in k], miss
m.eval()
M = tc.M
BS = 1
say(f'checkpoint {os.path.relpath(CKPT, PKG)} (epoch {ck["epoch"]}): M {M}, z1 {1e3 * tc.key_to_enc_spacing:g} mm, '
    f'z2 {1e3 * tc.last_layer_ccd_spacing:g} mm, key {tc.phase_key_size} px of {1e6 * tc.key_slm_dx:g} um, decoder '
    f'{tc.decoder_size_px} LUNA px of {1e6 * tc.enc_slm_dx:g} um, bin {tc.encoding_patch_scale}, detectors {tc.pd_size_cam} px '
    f'on a {tc.pd_pitch_cam} px pitch')
say(m.spec.describe())

a_all = torch.as_tensor(cfg.a, dtype=torch.float32, device=dev)
T = torch.from_numpy(np.load(os.path.join(PKG, 'targets.npy'))).float().to(dev)          # [100, 4096]

# ------------------------------------------------------------------------------------------- phase variants
key_lin, inp_lin = m.key_phases, m._input_plane_px
tabP, lutP = torch.tensor(cfg.pluto_phase_table, device=dev, dtype=torch.float32), torch.tensor(cfg.pluto_lut.astype(np.int64), device=dev)
tabL, lutL = torch.tensor(cfg.luna_phase_table, device=dev, dtype=torch.float32), torch.tensor(cfg.luna_lut.astype(np.int64), device=dev)
def snap(phi, lut, tab):
    """phase realised by the SLM when the package sends phi (conjugate + phase table): -table[q8(-phi)]"""
    j = torch.round(torch.remainder(-phi, TWO_PI) / (TWO_PI / 4096)).long() % 4096
    return -tab[lut[j]]
def set_variant(v):
    if v == 'linear':
        m.key_phases, m._input_plane_px = key_lin, inp_lin
    else:
        m.key_phases = lambda: snap(torch.sigmoid(m.slm_phases) * TWO_PI, lutP, tabP)
        m._input_plane_px = lambda a: snap(m._encode_px(a) + m.decoder_phases_px(), lutL, tabL)


@torch.no_grad()
def readout(a, bs=BS):
    return torch.cat([m(a[i:i + bs]).mean(1) for i in range(0, len(a), bs)]).reshape(len(a), -1)   # mean over M


def rmse_minmax(I, T):
    I = (I - I.min()) / (I.max() - I.min())
    return ((I - T) ** 2).mean(0).sqrt().mean().item()


def rmse_affine(I, T):
    return -compute_reward(I[None], T, cfg)[0].item()


# ------------------------------------------------------------------------------------------- 1. frame build check
import insitu_calib
keys_c, decs_c, _ = insitu_calib.load_trained_masks(CKPT_X)                    # expanded masks [M, 414 / 736, ...]
up2 = lambda x: x.repeat(2, 0).repeat(2, 1)
c0 = copy.copy(cfg)
c0.pluto_lut_inverted = c0.luna_lut_inverted = False
c0.pluto_flip_ud = c0.pluto_flip_lr = c0.luna_flip_ud = c0.luna_flip_lr = False
c0.use_phase_table = False
c0.phase_key_center_y, c0.phase_key_center_x = 540, 960
lev = lambda phi: (torch.round(torch.remainder(phi, TWO_PI) / (TWO_PI / 256)) % 256).cpu().numpy().astype(np.uint8)
nk, nd = tc.phase_key_size, tc.decoder_size_px
def ndiff(x, y):   # px that differ, and the largest circular level difference (float32 GPU vs float64 CPU rounding ties)
    d = np.abs(((x.astype(int) - y.astype(int) + 128) % 256) - 128)
    return f'{int((d > 0).sum())} px differ, max {int(d.max())} level'
assert nk == 207 and nd == 368 and tc.encoding_patch_scale == 55 and m.enc_px_shape == (368, 368)
for mm in range(M):
    ref = np.zeros((1080, 1920), np.uint8); ref[540 - 207:540 + 207, 960 - 207:960 + 207] = up2(lev(key_lin()[mm, 0]))
    say(f'frame check key {mm}: package pluto_frame vs model levels: {ndiff(TD.pluto_frame(keys_c[mm], c0), ref)}')
    for ia in (0, 99):
        ref = np.zeros((1080, 1920), np.uint8)
        ref[540 - 368:540 + 368, 960 - 368:960 + 368] = up2(lev(inp_lin(a_all[ia:ia + 1])[0, mm if m.decoder_per_key else 0]))
        say(f'frame check decoder {mm}, a {cfg.a[ia]:+.3f}: package luna_frame vs model levels: '
            f'{ndiff(TD.luna_frame(decs_c[mm], cfg.a[ia], c0), ref)}')

# ------------------------------------------------------------------------------------------- 2. SLM bmps
sb = lambda name, arr: Image.fromarray(np.ascontiguousarray(arr), 'L').save(os.path.join(HERE, 'slm_bmp', name))
for mm in range(M):
    sb(f'pluto_key{mm}_display.bmp', TD.pluto_frame(keys_c[mm], cfg))
    for ia in A_IDX:
        sb(f'luna_dec{mm}_a{cfg.a[ia]:+.3f}_display.bmp', TD.luna_frame(decs_c[mm], cfg.a[ia], cfg))
say(f'slm_bmp/: what the package sends for each frame m (key centre {cfg.phase_key_center_y, cfg.phase_key_center_x} model '
    f'frame, conjugate, phase table, PLUTO rot 180; no collimator lens). Frame m = pluto_key<m> together with luna_dec<m>.')

# ------------------------------------------------------------------------------------------- 3. zero-shot + sensitivity
res = {}
for v in ('linear', 'table'):
    set_variant(v)
    I = readout(a_all); res[v] = I
    say(f'zero-shot expected ({v:6s} phases), 100 a: min-max RMSE {rmse_minmax(I, T):.3e}, affine (reward) RMSE {rmse_affine(I, T):.3e}')
set_variant('linear')
geom0, spec0 = dict(m.spec_geom), m.spec
p1, nk_ = geom0['p1'], geom0['nk']
m0 = p1 / 16e-6
PX = p1 / 2                                            # one bench PLUTO px (8 um x m)
def pitch_for(mag):
    """key pitch for relay magnification mag, snapped to a whole nm (+-3 nm) giving the smallest spectral period"""
    best = None
    for pn in range(int(round(mag * 16000)) - 3, int(round(mag * 16000)) + 4):
        q = np.lcm(pn, int(round(geom0['p2'] * 1e9)))
        if best is None or q < best[0]:
            best = (q, pn)
    return best[1] * 1e-9
def with_pitch(pn):   # same key centre, new pitch
    cy, cx = geom0['key_y0'] + nk_ * p1 / 2, geom0['key_x0'] + nk_ * p1 / 2
    return dict(p1=pn, key_y0=cy - nk_ * pn / 2, key_x0=cx - nk_ * pn / 2)
say(f'sensitivity (linear phases, affine = reward RMSE; trained value {rmse_affine(res["linear"], T):.3e}):')
cases = [('z1 -0.5 mm', dict(z1=geom0['z1'] - 0.5e-3)), ('z1 +0.5 mm', dict(z1=geom0['z1'] + 0.5e-3)),
         ('z2 -0.5 mm', dict(z2=geom0['z2'] - 0.5e-3)), ('z2 +0.5 mm', dict(z2=geom0['z2'] + 0.5e-3)),
         ('key +0.5 px x', dict(key_x0=geom0['key_x0'] + 0.5 * PX)), ('key +1 px x', dict(key_x0=geom0['key_x0'] + 1 * PX)),
         ('key +2 px x', dict(key_x0=geom0['key_x0'] + 2 * PX)), ('key +1 px y', dict(key_y0=geom0['key_y0'] + 1 * PX)),
         ('key -3 px x', dict(key_x0=geom0['key_x0'] - 3 * PX)), ('key +3 px x', dict(key_x0=geom0['key_x0'] + 3 * PX)),
         ('key -3 px y', dict(key_y0=geom0['key_y0'] - 3 * PX)), ('key +3 px y', dict(key_y0=geom0['key_y0'] + 3 * PX)),
         ('pitch 16.150 (expanded file)', with_pitch(16.150e-6))]
for dm in (-0.005, +0.005):
    pn = pitch_for(m0 + dm)
    cases.append((f'm {dm:+.3f} ({pn / 8e-6:.5f})', with_pitch(pn)))
for name, upd in cases:
    m.spec = SpectralLabProp(tc, dict(geom0, **upd)).to(dev)
    I = readout(a_all)
    say(f'  {name:22s}: min-max {rmse_minmax(I, T):.3e}, affine {rmse_affine(I, T):.3e}')
    del m.spec; torch.cuda.empty_cache()
m.spec = spec0


# ------------------------------------------------------------------------------------------- 4. camera images
@torch.no_grad()
def camera(a, chunk=256):
    """Model-frame camera window [NCAM, NCAM]: every camera px is one 'detector' (5.86 um box, 4 x 4 Gauss-Legendre
    nodes); intensity averaged over the M frames."""
    geom = dict(geom0, det_size=CAM_DX)
    geom['det_cy'] = geom['det_cx'] = (np.arange(NCAM) - (NCAM - 1) / 2) * CAM_DX
    sp = SpectralLabProp(tc, geom).to('cpu')     # on the CPU: the 12 mm window's lattice does not fit next to training
    V1, v1s = sp.key_field_L(torch.exp(1j * m.key_phases()[:, 0].cpu()) - 1)
    d_all = torch.exp(1j * m._input_plane_px(torch.tensor([a], device=dev)).cpu()) - 1           # [1, F, ny, nx]
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
    del sp, V1, v1s; torch.cuda.empty_cache()
    return acc / M


def to_raw(Imod):
    """model-frame window (axis at the 1023/1024 corner) -> raw sensor frame [1200, 1920] (axis at the corner before raw
    px det_axis_yx; rotated 180 deg when cam_flip_* are True, as captured)."""
    Ar, Ac = cfg.det_axis_yx
    i, j = np.arange(cfg.sen_x_num), np.arange(cfg.sen_y_num)
    mr = Ar + 1023 - i if cfg.cam_flip_ud else i - Ar + 1024
    mc = Ac + 1023 - j if cfg.cam_flip_lr else j - Ac + 1024
    assert mr.min() >= 0 and mr.max() < NCAM and mc.min() >= 0 and mc.max() < NCAM
    return Imod[mr[:, None], mc[None, :]]


say(f'\ncamera images (phases through the phase table = what the package produces): raw orientation, '
    f'{cfg.sen_x_num} x {cfg.sen_y_num}, axis at raw corner {cfg.det_axis_yx}, cam flips {cfg.cam_flip_ud, cfg.cam_flip_lr}')
m.spec = None; torch.cuda.empty_cache()          # free the model's propagator while the camera window runs
set_variant('table')
raw = {}
for ia in A_IDX:
    t0 = time.time()
    raw[ia] = to_raw(camera(float(cfg.a[ia])))
    np.save(os.path.join(HERE, f'cmos_expected_a{cfg.a[ia]:+.3f}.npy'), raw[ia].astype(np.float16))
    say(f'  a {cfg.a[ia]:+.3f}: mean {raw[ia].mean():.3f}, max {raw[ia].max():.3f}  ({time.time() - t0:.0f} s)')
Ar, Ac = cfg.det_axis_yx
arr = np.stack([I[Ar - 254:Ar + 254, Ac - 254:Ac + 254] for I in raw.values()])
say(f'whole sensor max {max(I.max() for I in raw.values()):.2f}; inside the detector array: px max {arr.max():.2f}, '
    f'99.9th percentile {np.percentile(arr, 99.9):.2f} -> expose for the array; BMPs: make_cmos_bmps.py')

# ------------------------------------------------------------------------------------------- 5. end-to-end readout
m.spec = SpectralLabProp(tc, geom0).to(dev)
det_pkg = TD.captures_to_detectors(np.stack([raw[ia] for ia in A_IDX]), (0, 0), cfg, dev).float()
det_mod = readout(a_all[A_IDX])
rng = (det_mod.max() - det_mod.min()).item()
say(f'end-to-end: package captures_to_detectors on the raw sim images vs model detectors, {len(A_IDX)} a: '
    f'RMS diff {((det_pkg - det_mod) ** 2).mean().sqrt().item() / rng:.1e} of the readout range, '
    f'max {(det_pkg - det_mod).abs().max().item() / rng:.1e}')
set_variant('linear')

# ------------------------------------------------------------------------------------------- figures
fig, ax = plt.subplots(1, len(A_IDX), figsize=(5 * len(A_IDX), 3.9), constrained_layout=True)
vmax = max(np.percentile(I, 99.95) for I in raw.values())
for c, ia in enumerate(A_IDX):
    im = ax[c].imshow(raw[ia], cmap='inferno', vmin=0, vmax=vmax)
    ax[c].add_patch(plt.Rectangle((Ac - 254 - 0.5, Ar - 254 - 0.5), 508, 508, fill=False, ec='c', lw=0.8))
    ax[c].plot(Ac - 0.5, Ar - 0.5, 'c+', ms=8)
    ax[c].set_title(f'a = {cfg.a[ia]:+.3f}', fontsize=10)
    ax[c].set_xlabel('raw camera col'); ax[c].set_ylabel('raw camera row')
fig.colorbar(im, ax=ax, shrink=0.8, label='intensity / incident beam')
fig.suptitle(f'Expected camera image (mean of the M = {M} frames), bench M{M} 2x2-binned epoch {ck["epoch"]} (z1 35.87 mm, z2 175.35 mm), '
             f'RAW Basler orientation\ncyan: detector array and axis {cfg.det_axis_yx}; phases through the phase table; '
             f'colour clipped at the 99.95th percentile', fontsize=10)
fig.savefig(os.path.join(HERE, 'cmos_expected.png'), dpi=110)
plt.close(fig)

ia = A_IDX[1]
I = raw[ia]
y0, x0 = Ar - 254, Ac - 254
crop = I[y0 - 8:y0 + 516, x0 - 8:x0 + 516]
fig, ax = plt.subplots(1, 3, figsize=(19, 6.3), constrained_layout=True)
ax[0].imshow(crop, cmap='gray', interpolation='nearest')
ax[0].add_collection(matplotlib.collections.PatchCollection(
    [plt.Rectangle((8 + 8 * c - 0.5, 8 + 8 * r - 0.5), 4, 4) for r in range(64) for c in range(64)],
    facecolor='none', edgecolor='cyan', linewidth=0.2))
ax[0].set_title(f'raw camera crop around the detector array, a = {cfg.a[ia]:+.3f}\n(the in-situ eval_*_detgrid.png shows the same view)', fontsize=10)
ax[0].set_xlim(8 + 8 * 28 - 4, 8 + 8 * 36 + 4); ax[0].set_ylim(8 + 8 * 36 + 4, 8 + 8 * 28 - 4)
det_one = TD.captures_to_detectors(I[None], (0, 0), cfg, 'cpu')[0].numpy().reshape(64, 64)
ax[1].imshow(det_one, cmap='inferno'); ax[1].set_title('package readout (64 x 64, model orientation)', fontsize=10)
ax[2].imshow(T[ia].cpu().numpy().reshape(64, 64), cmap='inferno'); ax[2].set_title('target functions at this a (min-max over a)', fontsize=10)
fig.suptitle(f'Detector registration check, M{M} (zoom: central 8 x 8 detectors on the raw frame)', fontsize=11)
fig.savefig(os.path.join(HERE, 'detector_check.png'), dpi=110)
plt.close(fig)

open(os.path.join(HERE, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
say('done')
