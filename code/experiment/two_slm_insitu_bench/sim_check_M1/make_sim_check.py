"""Sim sanity check for the bench package (2026-11-11): the bench M1 8-bit checkpoint (ckpt_M1/bench_M1_8bit.pth, trained
at z1 = 29.7 mm / z2 = 168.8 mm), with the learned encoder (PLUTO key) and decoder (LUNA) in place.

What it checks / writes (this folder):
  1. Frame build: insitu_TDNFA.pluto_frame / luna_frame (display corrections off) vs the model's own quantised levels.
  2. slm_bmp/: the frames the package sends (display corrections ON: key shift, conjugate, phase table, PLUTO rot 180),
     plus the plain model-frame frames (linear levels, no corrections) for reference.
  3. Expected camera images, RAW camera orientation (as captured, before the 180 deg rotation), full 1920 x 1200 sensor,
     with the optical axis at the corner of raw px cfg.det_axis_yx:
        cmos_expected_<variant>_a<+x.xxx>.npy   float16 [1200, 1920], units of the incident beam intensity
        (BMPs and the detector-array zoom: make_cmos_bmps.py, run after this script)
     variant 'linear': phases exactly as trained; 'table': phases as the SLMs realise them through the measured phase
     table (q8 with use_phase_table, conjugated: realised phase = -table[level]).
  4. End-to-end readout: the package's captures_to_detectors on those raw images vs the model's detectors.
  5. Expected zero-shot numbers (what the in-situ [eval] at epoch 0 should show if the bench matched the model), on the
     package's 100-point a grid: min-max RMSE and the affine (reward) RMSE; linear vs phase table; and sensitivity to the
     measured uncertainties (z1 +-0.5 mm, z2 +-0.3 mm, key +-3 PLUTO px).
  figures: cmos_expected.png, detector_check.png; numbers: report.txt
Run from anywhere: /c/Anaconda/envs/qpi/python.exe make_sim_check.py   (needs this repo's code/ folder, a GPU)"""
import os, sys, time, copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

CKPT = os.path.join(PKG, 'ckpt_M1', 'bench_M1_8bit.pth')
CAM_DX, NCAM = 5.86e-6, 2048                    # model-frame camera window, axis between px 1023 / 1024 (covers the sensor)
A_IDX = [0, 25, 50, 74, 99]                     # a values (indices into cfg.a) for the camera images
dev = torch.device('cuda')
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(0.24, 0)   # GPU0 also runs training: stay below ~21 GB in total
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
say(f'checkpoint {os.path.relpath(CKPT, PKG)} (epoch {ck["epoch"]}): M {M}, z1 {1e3 * tc.key_to_enc_spacing:g} mm, '
    f'z2 {1e3 * tc.last_layer_ccd_spacing:g} mm, key {tc.phase_key_size} PLUTO px, decoder {tc.decoder_size_px} LUNA px, '
    f'bin {tc.encoding_patch_scale}, detectors {tc.pd_size_cam} px on a {tc.pd_pitch_cam} px pitch ({CAM_DX * 1e6} um px)')
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
def readout(a, bs=2):
    return torch.cat([m(a[i:i + bs]).mean(1) for i in range(0, len(a), bs)]).reshape(len(a), -1)   # mean over M


def rmse_minmax(I, T):
    I = (I - I.min()) / (I.max() - I.min())
    return ((I - T) ** 2).mean(0).sqrt().mean().item()


def rmse_affine(I, T):
    return -compute_reward(I[None], T, cfg)[0].item()


# ------------------------------------------------------------------------------------------- 1. frame build check
keys_c = (torch.sigmoid(m.slm_phases[:, 0]) * TWO_PI).detach().cpu()          # what insitu_calib.load_trained_masks gives
decs_c = torch.remainder(m.decoder_phases[:, 0], TWO_PI).detach().cpu()
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
assert nk == 414 and nd == 736 and tc.encoding_patch_scale == 110 and m.enc_px_shape == (736, 736)
for mm in range(M):
    ref = np.zeros((1080, 1920), np.uint8); ref[540 - 207:540 + 207, 960 - 207:960 + 207] = lev(key_lin()[mm, 0])
    say(f'frame check key {mm}: package pluto_frame vs model levels: {ndiff(TD.pluto_frame(keys_c[mm], c0), ref)}')
    for ia in (0, 37, 99):
        ref = np.zeros((1080, 1920), np.uint8)
        ref[540 - 368:540 + 368, 960 - 368:960 + 368] = lev(inp_lin(a_all[ia:ia + 1])[0, mm if m.decoder_per_key else 0])
        say(f'frame check decoder {mm}, a {cfg.a[ia]:+.3f}: package luna_frame vs model levels: '
            f'{ndiff(TD.luna_frame(decs_c[mm], cfg.a[ia], c0), ref)}')
# table variant consistency: the levels the package sends (corrections on, but no shift / flips) -> realised phase
c1 = copy.copy(cfg); c1.pluto_flip_ud = c1.pluto_flip_lr = False; c1.phase_key_center_y, c1.phase_key_center_x = 540, 960
set_variant('table')
kp = TD.pluto_frame(keys_c[0], c1)[540 - 207:540 + 207, 960 - 207:960 + 207]
dphi = np.angle(np.exp(1j * (-cfg.pluto_phase_table[kp] - m.key_phases()[0, 0].cpu().numpy())))
lp = TD.luna_frame(decs_c[0], cfg.a[37], c1)[540 - 368:540 + 368, 960 - 368:960 + 368]
dphl = np.angle(np.exp(1j * (-cfg.luna_phase_table[lp] - m._input_plane_px(a_all[37:38])[0, 0].cpu().numpy())))
say(f'phase-table variant vs package levels: max phase difference PLUTO {np.abs(dphi).max():.1e} rad, LUNA {np.abs(dphl).max():.1e} rad')
set_variant('linear')

# ------------------------------------------------------------------------------------------- 2. SLM bmps
sb = lambda name, arr: Image.fromarray(np.ascontiguousarray(arr), 'L').save(os.path.join(HERE, 'slm_bmp', name))
for mm in range(M):
    sb(f'pluto_key{mm}_display.bmp', TD.pluto_frame(keys_c[mm], cfg))
    sb(f'pluto_key{mm}_modelframe.bmp', TD.pluto_frame(keys_c[mm], c0))
    for ia in A_IDX:
        sb(f'luna_dec{mm}_a{cfg.a[ia]:+.3f}_display.bmp', TD.luna_frame(decs_c[mm], cfg.a[ia], cfg))
        sb(f'luna_dec{mm}_a{cfg.a[ia]:+.3f}_modelframe.bmp', TD.luna_frame(decs_c[mm], cfg.a[ia], c0))
say(f'slm_bmp/: *_display.bmp = what the package sends (key centre {cfg.phase_key_center_y, cfg.phase_key_center_x} model '
    f'frame, conjugate, phase table, PLUTO rot 180); *_modelframe.bmp = linear levels, no corrections')

# ------------------------------------------------------------------------------------------- 5. zero-shot numbers
res = {}
for v in ('linear', 'table'):
    set_variant(v)
    I = readout(a_all); res[v] = I
    say(f'zero-shot expected ({v:6s} phases), 100 a: min-max RMSE {rmse_minmax(I, T):.3e}, affine (reward) RMSE {rmse_affine(I, T):.3e}')
set_variant('linear')
geom0, spec0 = dict(m.spec_geom), m.spec
for name, upd in [('z1 -0.5 mm', dict(z1=geom0['z1'] - 0.5e-3)), ('z1 +0.5 mm', dict(z1=geom0['z1'] + 0.5e-3)),
                  ('z2 -0.3 mm', dict(z2=geom0['z2'] - 0.3e-3)), ('z2 +0.3 mm', dict(z2=geom0['z2'] + 0.3e-3)),
                  ('key +3 px x', dict(key_x0=geom0['key_x0'] + 3 * 8e-6)), ('key +3 px y', dict(key_y0=geom0['key_y0'] + 3 * 8e-6))]:
    m.spec = SpectralLabProp(tc, dict(geom0, **upd)).to(dev)
    I = readout(a_all)
    say(f'  sensitivity {name:12s} (linear): min-max {rmse_minmax(I, T):.3e}, affine {rmse_affine(I, T):.3e}')
    del m.spec; torch.cuda.empty_cache()
m.spec = spec0


# ------------------------------------------------------------------------------------------- 3. camera images
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


say(f'\ncamera images: raw orientation, {cfg.sen_x_num} x {cfg.sen_y_num}, axis at raw corner {cfg.det_axis_yx}, '
    f'cam flips {cfg.cam_flip_ud, cfg.cam_flip_lr}; {NCAM} x {NCAM} px model window')
m.spec = None; torch.cuda.empty_cache()          # free the model's propagator while the camera window runs
raw = {}
for v in ('linear', 'table'):
    set_variant(v)
    for ia in A_IDX:
        t0 = time.time()
        raw[v, ia] = to_raw(camera(float(cfg.a[ia])))
        say(f'  {v:6s} a {cfg.a[ia]:+.3f}: mean {raw[v, ia].mean():.3f}, max {raw[v, ia].max():.3f}  ({time.time() - t0:.0f} s)')
set_variant('linear')
for (v, ia), I in raw.items():
    np.save(os.path.join(HERE, f'cmos_expected_{v}_a{cfg.a[ia]:+.3f}.npy'), I.astype(np.float16))
Ar, Ac = cfg.det_axis_yx
arr = np.stack([I[Ar - 254:Ar + 254, Ac - 254:Ac + 254] for I in raw.values()])
say(f'whole sensor max {max(I.max() for I in raw.values()):.2f}; inside the detector array: px max {arr.max():.2f}, '
    f'99.9th percentile {np.percentile(arr, 99.9):.2f} -> expose for the array (the rim outside saturates); BMPs: make_cmos_bmps.py')

# ------------------------------------------------------------------------------------------- 4. end-to-end readout
m.spec = SpectralLabProp(tc, geom0).to(dev)
for v in ('linear', 'table'):
    set_variant(v)
    det_pkg = TD.captures_to_detectors(np.stack([raw[v, ia] for ia in A_IDX]), (0, 0), cfg, dev).float()
    det_mod = readout(a_all[A_IDX])
    rng = (det_mod.max() - det_mod.min()).item()
    say(f'end-to-end ({v}): package captures_to_detectors on the raw sim images vs model detectors, {len(A_IDX)} a: '
        f'RMS diff {((det_pkg - det_mod) ** 2).mean().sqrt().item() / rng:.1e} of the readout range, '
        f'max {(det_pkg - det_mod).abs().max().item() / rng:.1e}; reward-RMSE package {rmse_affine(det_pkg, T[A_IDX]):.3e} '
        f'vs model {rmse_affine(det_mod, T[A_IDX]):.3e}')
set_variant('linear')

# ------------------------------------------------------------------------------------------- figures
Ar, Ac = cfg.det_axis_yx
fig, ax = plt.subplots(2, len(A_IDX), figsize=(4.2 * len(A_IDX), 6.2), constrained_layout=True)
vmax = max(np.percentile(I, 99.95) for I in raw.values())
for r, v in enumerate(('linear', 'table')):
    for c, ia in enumerate(A_IDX):
        im = ax[r, c].imshow(raw[v, ia], cmap='inferno', vmin=0, vmax=vmax)
        ax[r, c].add_patch(plt.Rectangle((Ac - 254 - 0.5, Ar - 254 - 0.5), 508, 508, fill=False, ec='c', lw=0.8))
        ax[r, c].plot(Ac - 0.5, Ar - 0.5, 'c+', ms=8)
        ax[r, c].set_title(f'{v} phases, a = {cfg.a[ia]:+.3f}', fontsize=10)
        ax[r, c].set_xlabel('raw camera col'); ax[r, c].set_ylabel('raw camera row')
fig.colorbar(im, ax=ax, shrink=0.8, label='intensity / incident beam')
fig.suptitle(f'Expected camera image, bench M1 8-bit (z1 29.7 mm, z2 168.8 mm, key 414 PLUTO px, decoder 736 LUNA px), '
             f'RAW Basler orientation (1920 x 1200, 5.86 um)\ncyan: detector array (64 x 64, 4 px boxes on an 8 px pitch) '
             f'and axis {cfg.det_axis_yx}; top: trained phases, bottom: through the measured phase tables; '
             f'colour clipped at the 99.95th percentile', fontsize=11)
fig.savefig(os.path.join(HERE, 'cmos_expected.png'), dpi=110)
plt.close(fig)

ia = A_IDX[2]
I = raw['linear', ia]
y0, x0 = Ar - 254, Ac - 254
crop = I[y0 - 8:y0 + 516, x0 - 8:x0 + 516]
fig, ax = plt.subplots(1, 3, figsize=(19, 6.3), constrained_layout=True)
ax[0].imshow(crop, cmap='gray', interpolation='nearest')
ax[0].add_collection(matplotlib.collections.PatchCollection(
    [plt.Rectangle((8 + 8 * c - 0.5, 8 + 8 * r - 0.5), 4, 4) for r in range(64) for c in range(64)],
    facecolor='none', edgecolor='cyan', linewidth=0.2))
ax[0].set_title(f'raw camera crop around the detector array, a = {cfg.a[ia]:+.3f}\n(the in-situ eval_*_detgrid.png shows the same view)', fontsize=10)
ax[0].set_xlim(8 + 8 * 28 - 4, 8 + 8 * 36 + 4); ax[0].set_ylim(8 + 8 * 36 + 4, 8 + 8 * 28 - 4)
det_pkg = TD.captures_to_detectors(I[None], (0, 0), cfg, 'cpu')[0].numpy().reshape(64, 64)
f_hat = res['linear'][ia].cpu().numpy()
ax[1].imshow(det_pkg, cmap='inferno'); ax[1].set_title('package readout (64 x 64, model orientation)', fontsize=10)
ax[2].imshow(T[ia].cpu().numpy().reshape(64, 64), cmap='inferno'); ax[2].set_title('target functions at this a (min-max over a)', fontsize=10)
fig.suptitle('Detector registration check (zoom: central 8 x 8 detectors on the raw frame)', fontsize=11)
fig.savefig(os.path.join(HERE, 'detector_check.png'), dpi=110)
plt.close(fig)

open(os.path.join(HERE, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
say('done')
