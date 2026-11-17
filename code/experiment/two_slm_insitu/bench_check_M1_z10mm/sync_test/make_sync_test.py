"""PLUTO <-> LUNA sync / registration test patterns for the locked bench (z1 33 mm PLUTO image -> LUNA, z2 167 mm
LUNA -> CMOS), 635 nm. Two tests; every pattern is centred on the same axis as the trained masks (the pixel corner
between rows 539/540 and cols 959/960 on both SLMs).

TEST 1 "F imaging" (orientation + registration):
  PLUTO: a letter F (80 x 60 px = 0.64 x 0.48 mm) filled with a 0/pi stripe grating (period 2 px), flat elsewhere. The
         stripes throw the light hitting the F sideways (+-1.3 mm at the LUNA), so the F is dark.
  LUNA:  a lens f2 = z1 z2 / (z1 + z2) = 27.6 mm in a 0.6 mm-radius disk with a notch cut at the top right, 0/pi
         checkerboard elsewhere (which throws that light far off the sensor).
  Camera: the lens images the PLUTO onto the CMOS at -z2/z1 = -5.06x. You see a bright disk (~6 mm) with a dark F, both
         rotated 180 deg. The notch shows the LUNA orientation, the F the PLUTO orientation, the F's offset from the disk
         centre = -5.06 x the PLUTO-vs-LUNA misregistration (8 um -> 41 um = 7 camera px).
TEST 2 "pinhole" (pass / fail):
  PLUTO: a lens f1 = z1 = 33 mm, 1.5 mm diameter, flat elsewhere -> focuses to a ~25 um spot on the LUNA.
  LUNA:  a 10 x 10 px (45 um) open square, 0/pi checkerboard elsewhere.
  Camera: a bright smooth ~7.6 mm disk if the spot goes through the pinhole; nearly dark if the registration is off by
         more than ~30 um.

Simulation: wave_prop.SpectralLabProp, with the LUNA checkerboard modelled as opaque (its light leaves at 0.07 rad per axis,
11.8 mm off-axis at the camera, outside the simulated band). No leakage from imperfect pi levels is modelled.
Level k <-> phase 2 pi k / 256. Converging lens = exp(-i pi r^2 / (lam f)) in the model's convention; if the SLMs' sign
is opposite, nothing focuses: then try the files in inverted_lut/ (level -> (256 - level) % 256 = conjugate phase).
Run: /c/Anaconda/envs/qpi/python.exe make_sync_test.py"""
import os, sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))
from wave_prop import SpectralLabProp

LAM, Z1, Z2 = 635e-9, 33e-3, 167e-3
P1, P2, CAM_DX, NCAM = 8e-6, 4.5e-6, 5.86e-6, 2048
F2 = Z1 * Z2 / (Z1 + Z2)
dev = torch.device('cuda')
torch.cuda.set_per_process_memory_fraction(0.85, 0)
cfg = SimpleNamespace(wavelength=LAM, ridx_air=1.0, spec_P_mult=1.25)
os.makedirs(os.path.join(HERE, 'inverted_lut'), exist_ok=True)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)


def q8(phi):
    return (np.round(np.mod(phi, 2 * np.pi) / (2 * np.pi / 256)) % 256).astype(np.uint8)


def centred_coords(n, p):
    return (np.arange(n) - (n - 1) / 2) * p          # pixel centres of an even n array centred on the pixel corner


def save_frame(arr, name):
    Image.fromarray(np.ascontiguousarray(arr), 'L').save(os.path.join(HERE, name))
    Image.fromarray(np.ascontiguousarray(((256 - arr.astype(int)) % 256).astype(np.uint8)), 'L').save(
        os.path.join(HERE, 'inverted_lut', name))


def place(frame, arr, mask=None):
    n0, n1 = arr.shape
    y0, x0 = 540 - n0 // 2, 960 - n1 // 2
    sub = frame[y0:y0 + n0, x0:x0 + n1]
    if mask is None:
        sub[...] = arr
    else:
        sub[mask] = arr[mask]
    return y0, x0


# ------------------------------------------------------------------------------------------------ patterns
# TEST 1 PLUTO: F in an 80 x 80 box (cols 10..69), stripes = pi on odd absolute PLUTO columns
nk1 = 80
Fm = np.zeros((nk1, nk1), bool)
Fm[0:80, 10:26] = True; Fm[0:16, 10:70] = True; Fm[32:48, 10:54] = True
cols = np.arange(960 - nk1 // 2, 960 + nk1 // 2)
key1 = np.where(Fm, np.pi * (cols[None, :] % 2), 0.0)
# TEST 1 LUNA: lens in a 0.6 mm disk minus the notch (x > 0.30 mm, y < -0.10 mm; y down = smaller row index is up)
nl1 = 268
yy, xx = np.meshgrid(centred_coords(nl1, P2), centred_coords(nl1, P2), indexing='ij')
win1 = (yy ** 2 + xx ** 2 <= 0.6e-3 ** 2) & ~((xx > 0.30e-3) & (yy < -0.10e-3))
lens1 = -np.pi * (yy ** 2 + xx ** 2) / (LAM * F2)
# TEST 2 PLUTO: lens f1 = z1 in a 1.5 mm disk (188 px), flat outside
nk2 = 188
ky, kx = np.meshgrid(centred_coords(nk2, P1), centred_coords(nk2, P1), indexing='ij')
key2 = np.where(ky ** 2 + kx ** 2 <= 0.75e-3 ** 2, -np.pi * (ky ** 2 + kx ** 2) / (LAM * Z1), 0.0)
# TEST 2 LUNA: 10 x 10 px open (flat) pinhole
nl2 = 10
win2 = np.ones((nl2, nl2), bool)
lens2 = np.zeros((nl2, nl2))

checker = ((np.add.outer(np.arange(1080), np.arange(1920)) % 2) * 128).astype(np.uint8)
for name, key, win, lph in (('test1_F', key1, win1, lens1), ('test2_pinhole', key2, win2, lens2)):
    pl = np.zeros((1080, 1920), np.uint8); r = place(pl, q8(key))
    lu = checker.copy(); r2 = place(lu, q8(lph), win)
    save_frame(pl, f'{name}_pluto.bmp'); save_frame(lu, f'{name}_luna.bmp')
    say(f'{name}: PLUTO pattern rows {r[0]}..{r[0] + key.shape[0] - 1} cols {r[1]}..{r[1] + key.shape[1] - 1}; LUNA window '
        f'rows {r2[0]}..{r2[0] + win.shape[0] - 1} cols {r2[1]}..{r2[1] + win.shape[1] - 1}, checkerboard 0/128 elsewhere')
say(f'LUNA lens f2 = {1e3 * F2:.3f} mm (max local freq {0.6e-3 / (LAM * F2) / 1e3:.1f}/mm vs Nyquist {1 / (2 * P2) / 1e3:.0f}/mm); '
    f'PLUTO lens f1 = {1e3 * Z1:g} mm (max {0.75e-3 / (LAM * Z1) / 1e3:.1f}/mm vs {1 / (2 * P1) / 1e3:.1f}/mm)')


# ------------------------------------------------------------------------------------------------ simulation
@torch.no_grad()
def camera(key_phase, luna_phase, luna_win, shift_px=(0, 0), chunk=256):
    """Opaque-outside LUNA: u_after = t (1 + v1) e^{ikz1}, t = window e^{i luna}; spectrum c1 T + V1 (*) T."""
    nk, nl = key_phase.shape[0], luna_phase.shape[0]
    c = (np.arange(NCAM) - (NCAM - 1) / 2) * CAM_DX
    geom = dict(p1=P1, nk=nk, key_y0=-nk / 2 * P1 + shift_px[0] * P1, key_x0=-nk / 2 * P1 + shift_px[1] * P1,
                p2=P2, luna_ny=nl, luna_nx=nl, luna_y0=-nl / 2 * P2, luna_x0=-nl / 2 * P2,
                det_cy=c, det_cx=c, det_size=CAM_DX, z1=Z1, z2=Z2)
    sp = SpectralLabProp(cfg, geom).to(dev)
    kq = torch.from_numpy(q8(key_phase) * (2 * np.pi / 256)).to(dev)
    lq = torch.from_numpy(q8(luna_phase) * (2 * np.pi / 256)).to(dev)
    V1, v1s = sp.key_field_L((torch.exp(1j * kq) - 1)[None].to(torch.complex64))
    t = (torch.exp(1j * lq) * torch.from_numpy(luna_win).to(dev))[None, None].to(torch.complex64)
    Q, L = sp.Q2, sp.L
    TL = sp._to_L(torch.fft.fft2(F.pad(t, (0, Q - nl, 0, Q - nl))), sp.i2, sp.e2y, sp.e2x)
    conv = torch.fft.fft2(v1s[None] * torch.fft.ifft2(TL)) * (L * L * sp.df2)
    sel = lambda X: X.index_select(-2, sp.iL2).index_select(-1, sp.iL2)
    A = ((sp.c1 * sel(TL) + sel(conv)) * sp.H2T)[0, 0]
    del TL, conv
    AX = A @ sp.ExT
    rows = []
    for r in range(0, NCAM, chunk):
        u = (sp.Ey[r * sp.ngl:(r + chunk) * sp.ngl] @ AX) * sp.df2
        I = (u.real ** 2 + u.imag ** 2).reshape(-1, sp.ngl, NCAM, sp.ngl)
        rows.append(torch.einsum('ipjq,pq->ij', I, sp.wq))
    desc = sp.describe()
    del sp, V1, v1s, A, AX; torch.cuda.empty_cache()
    return torch.cat(rows).cpu().numpy(), desc


say('\ncamera simulations (intensity in units of the incident beam), 2048 x 2048 px window = 12 mm')
sims = {}
for name, key, win, lph, sh in (('test1_F', key1, win1, lens1, (0, 3)), ('test2_pinhole', key2, win2, lens2, (0, 4))):
    for tag, s in (('aligned', (0, 0)), (f'PLUTO_shift_x{sh[1]}px', sh)):
        I, desc = camera(key, lph, win, s)
        sims[(name, tag)] = I
        np.save(os.path.join(HERE, f'{name}_cmos_sim_{tag}.npy'), I.astype(np.float32))
        cz = I[NCAM // 2 - 700:NCAM // 2 + 700, NCAM // 2 - 700:NCAM // 2 + 700]
        say(f'  {name} {tag} (PLUTO shifted {s[1] * 8} um in x): max {I.max():.4f}, central 8.2 mm total power '
            f'{cz.sum() * CAM_DX ** 2 * 1e6:.3f} mm^2 x incident; {desc}')

# BMPs on camera px (same scale for aligned / shifted of one test: 255 = 1.1 x aligned max)
for name in ('test1_F', 'test2_pinhole'):
    vmax = 1.1 * sims[(name, 'aligned')].max()
    for (n, tag), I in sims.items():
        if n != name:
            continue
        lev = np.clip(np.round(255 * I / vmax), 0, 255).astype(np.uint8)
        c = NCAM // 2
        Image.fromarray(np.ascontiguousarray(lev[c - 600:c + 600, c - 960:c + 960]), 'L').save(
            os.path.join(HERE, f'{name}_cmos_sim_{tag}_sensor.bmp'))
    say(f'{name} camera BMPs: level = 255 x I / {vmax:.4f} (linear)')

# figure
fig, ax = plt.subplots(2, 4, figsize=(20, 10.5), constrained_layout=True)
ext = np.array([-960, 960, 600, -600]) * CAM_DX * 1e3
for i, name in enumerate(('test1_F', 'test2_pinhole')):
    pl = np.array(Image.open(os.path.join(HERE, f'{name}_pluto.bmp')))
    lu = np.array(Image.open(os.path.join(HERE, f'{name}_luna.bmp')))
    ax[i, 0].imshow(pl[540 - 120:540 + 120, 960 - 120:960 + 120], cmap='twilight', vmin=0, vmax=255,
                    extent=np.array([-120, 120, 120, -120]) * P1 * 1e3, interpolation='nearest')
    ax[i, 0].set_title(f'{name}_pluto.bmp (central 240 px)')
    ax[i, 1].imshow(lu[540 - 160:540 + 160, 960 - 160:960 + 160], cmap='twilight', vmin=0, vmax=255,
                    extent=np.array([-160, 160, 160, -160]) * P2 * 1e3, interpolation='nearest')
    ax[i, 1].set_title(f'{name}_luna.bmp (central 320 px)')
    vmax = 1.1 * sims[(name, 'aligned')].max()
    for j, tag in enumerate([t for (n, t) in sims if n == name]):
        I = sims[(name, tag)]; c = NCAM // 2
        im = ax[i, 2 + j].imshow(I[c - 600:c + 600, c - 960:c + 960], cmap='gray', vmin=0, vmax=vmax, extent=ext)
        ax[i, 2 + j].set_title(f'CMOS sim, {tag.replace("_", " ")}\n(1920 x 1200 sensor, 5.86 um px)')
        fig.colorbar(im, ax=ax[i, 2 + j], shrink=0.6, label='intensity / incident')
    for a_ in ax[i]:
        a_.set_xlabel('x (mm)'); a_.set_ylabel('y (mm)')
fig.suptitle(f'PLUTO <-> LUNA sync tests, z1 = {1e3 * Z1:g} mm, z2 = {1e3 * Z2:g} mm, 635 nm. Row 1: LUNA lens f = '
             f'{1e3 * F2:.2f} mm images the PLUTO "F" at -5.06x (notch = LUNA orientation). Row 2: PLUTO lens f = 33 mm '
             'focuses through a 45 um LUNA pinhole.\nRight column: PLUTO pattern shifted by 3 px (24 um, test 1) / 4 px '
             '(32 um, test 2) in x. Phase colour map: level 0..255 = 0..2 pi.', fontsize=11)
fig.savefig(os.path.join(HERE, 'sync_tests.png'), dpi=100)
open(os.path.join(HERE, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
say('done')
