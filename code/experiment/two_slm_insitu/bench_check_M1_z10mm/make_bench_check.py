"""Bench forward-model check (2026-11-10): the z2 = 10 mm M1 8-bit checkpoint (trained for z1 3 mm / z2 10 mm),
re-simulated at the locked bench distances z1 = 33 mm (PLUTO image -> LUNA, 1:1 relay) and z2 = 167 mm (LUNA -> CMOS).
Outputs, in this folder:
  pluto_key_M1.bmp, pluto_key_M1_rot180.bmp      1920 x 1080 8-bit PLUTO frames (key 120 x 120 px centred, 0 elsewhere)
  luna_M1_a<+x.xx>.bmp                            1920 x 1080 8-bit LUNA frames (encoding(a) + decoder, centred)
  cmos_sim_a<+x.xx>.npy                           simulated camera image [2048, 2048] at 5.86 um px, units of the incident
                                                  beam intensity, row 0 = top (model y), axis between px 1023 / 1024
  cmos_sim_locked.png, report.txt                 figure; numbers
Level k <-> phase 2 pi k / 256 (model._quantize; insitu_TDNFA.q8 without LUT inversion / flips).
Run: /c/Anaconda/envs/qpi/python.exe make_bench_check.py   (needs this repo's code/ folder, GPU)"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
from config import init_params, recompute_derived, apply_saved_config, mark_explicit
from model import TimeMultiplexedNFA
from wave_prop import SpectralLabProp

CKPT = os.path.join(HERE, '..', 'ckpt', 'z2_10mm_M1_8bit.pth')
Z1, Z2 = 33e-3, 167e-3
CAM_DX, NCAM = 5.86e-6, 2048                    # simulated camera window: 2048 x 2048 px = 12.0 mm (covers a 1920 x 1200 sensor)
A_BMP = [-0.5, -0.25, 0.0, 0.25, 0.5]
dev = torch.device('cuda')
torch.cuda.set_per_process_memory_fraction(0.85, 0)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)


def load(**over):
    tc = init_params()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    apply_saved_config(tc, ck['config'])
    for k, v in over.items():
        setattr(tc, k, v)
    mark_explicit(tc, list(over))
    recompute_derived(tc)
    m = TimeMultiplexedNFA(tc).to(dev)
    miss, _ = m.load_state_dict(ck['model'], strict=False)
    assert not [k for k in miss if 'phases' in k], miss
    return tc, m.eval()


@torch.no_grad()
def readout(m, a, bs=10):
    return torch.cat([m(a[i:i + bs]) for i in range(0, len(a), bs)]).reshape(len(a), -1)   # mean over M (M1: the frame)


def rmse_minmax(I, T):
    I = (I - I.min()) / (I.max() - I.min())
    return ((I - T) ** 2).mean(0).sqrt().mean().item()


# ---------------------------------------------------------------------------------------------------- design check
tc_asm, m_asm = load()
say(f'checkpoint {os.path.basename(CKPT)}: M {tc_asm.M}, z1 {1e3 * tc_asm.key_to_enc_spacing:g} mm, '
    f'z2 {1e3 * tc_asm.last_layer_ccd_spacing:g} mm, bits {tc_asm.slm_phase_bits}, key {tc_asm.phase_key_size} PLUTO px, '
    f'decoder {tc_asm.decoder_size_px} LUNA px, bin {tc_asm.encoding_patch_scale}')
a_grid = torch.linspace(-0.5, 0.5, 100, device=dev)
T = torch.from_numpy(np.load(os.path.join(HERE, '..', 'targets.npy'))).float().to(dev)
I_asm = readout(m_asm, a_grid)
tc_d, m_d = load(prop_method='spectral', spec_P_mult=1.25)
I_d = readout(m_d, a_grid)
say(f'design distances, 64 x 64 detectors, 100 a: min-max RMSE asm (as trained) {rmse_minmax(I_asm, T):.2e}, '
    f'spectral {rmse_minmax(I_d, T):.2e}; spectral vs asm readout '
    f'{((I_d - I_asm) ** 2).mean().sqrt().item() / (I_asm.max() - I_asm.min()).item():.1e} of range')
g = m_d.spec_geom
say(f'geometry: key corner {1e6 * g["key_y0"]:.2f} / {1e6 * g["key_x0"]:.2f} um ({g["nk"]} px), LUNA array corner '
    f'{1e6 * g["luna_y0"]:.2f} / {1e6 * g["luna_x0"]:.2f} um ({g["luna_ny"]} x {g["luna_nx"]} px)')
del m_asm; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------------------------------- masks + BMPs
key = m_d.key_phases()[0, 0].detach()                                   # [120, 120] quantised (rad)
lev = lambda phi: (torch.round(torch.remainder(phi, 2 * np.pi) / (2 * np.pi / 256)) % 256).cpu().numpy().astype(np.uint8)
nk, nd = key.shape[-1], tc_d.decoder_size_px
pl = np.zeros((1080, 1920), np.uint8)
pl[540 - nk // 2:540 - nk // 2 + nk, 960 - nk // 2:960 - nk // 2 + nk] = lev(key)
Image.fromarray(pl, 'L').save(os.path.join(HERE, 'pluto_key_M1.bmp'))
Image.fromarray(np.ascontiguousarray(pl[::-1, ::-1]), 'L').save(os.path.join(HERE, 'pluto_key_M1_rot180.bmp'))
# LUNA: model's displayed pattern (encoding + decoder, quantised once) on the decoder array; cross-check vs the in-situ build
ENC_IDS = np.tile([[0, 1], [2, 3]], (3, 3)).repeat(28, 0).repeat(28, 1)
dec = torch.remainder(m_d.decoder_phases[0, 0].detach(), 2 * np.pi).double().cpu().numpy()
assert m_d._input_plane_px(a_grid[:1]).shape[-1] == nd == 213 and tc_d.encoding_patch_scale == 28
for a in A_BMP:
    luna_px = m_d._input_plane_px(torch.tensor([a], device=dev))[0, 0].detach()          # [213, 213]
    sq = dec.copy(); sq[22:190, 22:190] += 2 * np.pi * np.asarray([0, 1, 2, 3])[ENC_IDS] * a
    ref = (np.round(np.mod(sq, 2 * np.pi) / (2 * np.pi / 256)) % 256).astype(np.uint8)
    ndiff = int((ref != lev(luna_px)).sum())
    lu = np.zeros((1080, 1920), np.uint8)
    lu[540 - 84 - 22:540 - 84 - 22 + nd, 960 - 84 - 22:960 - 84 - 22 + nd] = lev(luna_px)
    Image.fromarray(lu, 'L').save(os.path.join(HERE, f'luna_M1_a{a:+.2f}.bmp'))
    say(f'luna_M1_a{a:+.2f}.bmp: levels {lu.min()}..{lu.max()}, model vs in-situ luna_frame build: {ndiff} px differ')
say(f'pluto_key_M1.bmp: key rows/cols {540 - nk // 2}..{540 - nk // 2 + nk - 1} / {960 - nk // 2}..{960 - nk // 2 + nk - 1}; '
    f'LUNA decoder rows/cols {540 - 106}..{540 - 106 + nd - 1} / {960 - 106}..{960 - 106 + nd - 1} '
    f'(encoding block {540 - 84}..{540 + 83} / {960 - 84}..{960 + 83})')

# ---------------------------------------------------------------------------------------------------- camera images
@torch.no_grad()
def camera(tc, m, z1, z2, ncam, a_list, chunk=256):
    """Full camera window: every camera px is one 'detector' (5.86 um box, 4 x 4 Gauss-Legendre nodes)."""
    geom = dict(m.spec_geom, z1=z1, z2=z2, det_size=CAM_DX)
    c = (np.arange(ncam) - (ncam - 1) / 2) * CAM_DX
    geom['det_cy'] = geom['det_cx'] = c
    sp = SpectralLabProp(tc, geom).to(dev)
    say(f'  camera z1 {1e3 * z1:g} mm / z2 {1e3 * z2:g} mm, {ncam} x {ncam} px ({1e3 * ncam * CAM_DX:.2f} mm): {sp.describe()}')
    V1, v1s = sp.key_field_L(torch.exp(1j * m.key_phases()[:, 0]) - 1)
    out = []
    for a in a_list:
        d = (torch.exp(1j * m._input_plane_px(torch.tensor([a], device=dev))) - 1)[:, :1]   # [1, 1, 213, 213]
        Q, L = sp.Q2, sp.L
        Dq = torch.fft.fft2(torch.nn.functional.pad(d, (0, Q - d.shape[-1], 0, Q - d.shape[-2])))
        DL = sp._to_L(Dq, sp.i2, sp.e2y, sp.e2x)
        conv = torch.fft.fft2(v1s[None] * torch.fft.ifft2(DL)) * (L * L * sp.df2)
        sel = lambda X: X.index_select(-2, sp.iL2).index_select(-1, sp.iL2)
        A = ((sp.c1 * sel(DL) + sel(V1)[None] + sel(conv)) * sp.H2T)[0, 0]
        del Dq, DL, conv
        AX = A @ sp.ExT
        rows = []
        for r in range(0, ncam, chunk):
            u = sp.c12 + (sp.Ey[r * sp.ngl:(r + chunk) * sp.ngl] @ AX) * sp.df2
            I = (u.real ** 2 + u.imag ** 2).reshape(-1, sp.ngl, ncam, sp.ngl)
            rows.append(torch.einsum('ipjq,pq->ij', I, sp.wq))
        out.append(torch.cat(rows).cpu().numpy())
        del A, AX; torch.cuda.empty_cache()
    del sp, V1, v1s; torch.cuda.empty_cache()
    return out


say('\ncamera simulations (units: incident beam intensity; 1 = unmodulated beam)')
cam_design = camera(tc_d, m_d, tc_d.key_to_enc_spacing, tc_d.last_layer_ccd_spacing, 256, A_BMP)
cam_locked = camera(tc_d, m_d, Z1, Z2, NCAM, A_BMP)
for a, I in zip(A_BMP, cam_locked):
    np.save(os.path.join(HERE, f'cmos_sim_a{a:+.2f}.npy'), I.astype(np.float32))
    cz = I[NCAM // 2 - 128:NCAM // 2 + 128, NCAM // 2 - 128:NCAM // 2 + 128]
    say(f'  a {a:+.2f}: locked full window mean {I.mean():.3f} max {I.max():.2f}; central 1.5 mm mean {cz.mean():.3f} '
        f'max {cz.max():.2f}; design-distance central 1.5 mm mean {cam_design[A_BMP.index(a)].mean():.3f} '
        f'max {cam_design[A_BMP.index(a)].max():.2f}')

# locked distances with the trained 64 x 64 detector grid: the function readout these masks give on the bench
tc_l, m_l = load(prop_method='spectral', spec_P_mult=1.25, key_to_enc_spacing=Z1, last_layer_ccd_spacing=Z2)
I_l = readout(m_l, a_grid)
say(f'\nlocked distances, trained 64 x 64 detector grid, 100 a: min-max RMSE {rmse_minmax(I_l, T):.2e} '
    f'(design {rmse_minmax(I_d, T):.2e}) -- these masks are not expected to compute the functions at 33 / 167 mm')

# ---------------------------------------------------------------------------------------------------- figures
ext_full = np.array([-1, 1, 1, -1]) * NCAM * CAM_DX / 2 * 1e3
ext_zoom = np.array([-1, 1, 1, -1]) * 256 * CAM_DX / 2 * 1e3
vmax_l = max(np.percentile(I, 99.9) for I in cam_locked)
vmax_d = max(np.percentile(I, 99.9) for I in cam_design)
fig, ax = plt.subplots(3, len(A_BMP), figsize=(4 * len(A_BMP), 12.5), constrained_layout=True)
for j, a in enumerate(A_BMP):
    I = cam_locked[j]
    im0 = ax[0, j].imshow(I, extent=ext_full, cmap='inferno', vmin=0, vmax=vmax_l)
    ax[0, j].add_patch(plt.Rectangle((-1920 * CAM_DX / 2e-3, -1200 * CAM_DX / 2e-3), 1920 * CAM_DX * 1e3, 1200 * CAM_DX * 1e3,
                                     fill=False, ec='c', lw=0.8, ls='--'))
    ax[0, j].set_title(f'a = {a:+.2f}: locked, 12 mm window')
    zc = I[NCAM // 2 - 128:NCAM // 2 + 128, NCAM // 2 - 128:NCAM // 2 + 128]
    im1 = ax[1, j].imshow(zc, extent=ext_zoom, cmap='inferno', vmin=0, vmax=vmax_l)
    ax[1, j].set_title('locked, central 1.5 mm')
    im2 = ax[2, j].imshow(cam_design[j], extent=ext_zoom, cmap='inferno', vmin=0, vmax=vmax_d)
    ax[2, j].set_title('trained 3 / 10 mm, central 1.5 mm')
    for i in range(3):
        ax[i, j].set_xlabel('x (mm)'); ax[i, j].set_ylabel('y (mm)')
for i, im in enumerate((im0, im1, im2)):
    fig.colorbar(im, ax=ax[i, :], shrink=0.8, label='intensity / incident')
fig.suptitle('Simulated CMOS intensity, z2 = 10 mm M1 8-bit checkpoint (Np 4, Nf 4096, key 120 PLUTO px, decoder 213 LUNA px)\n'
             f'rows 1-2: locked bench z1 = {1e3 * Z1:g} mm (PLUTO image -> LUNA), z2 = {1e3 * Z2:g} mm (LUNA -> CMOS), '
             f'5.86 um px (dashed: 1920 x 1200 sensor); row 3: trained z1 3 mm / z2 10 mm\n'
             'plane-wave illumination, colour clipped at the 99.9th percentile per row', fontsize=12)
fig.savefig(os.path.join(HERE, 'cmos_sim_locked.png'), dpi=110)
open(os.path.join(HERE, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
say('done')
