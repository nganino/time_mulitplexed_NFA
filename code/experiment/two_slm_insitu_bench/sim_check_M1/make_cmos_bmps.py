"""Expected camera images (cmos_expected_*.npy from make_sim_check.py) -> 8-bit BMPs + a zoom figure of the detector array.

The BMPs are scaled the way the camera should be exposed: from the DETECTOR ARRAY, not the whole frame.  With the trained
masks the detectors see only 0.02-0.13 of the open beam and the array's pixels peak near 0.5-1.0, while a rim just
outside the array reaches ~7x and saturates.  level = 255 * I / SCALE, SCALE = 0.6 (open beam I = 1 -> saturated).
  cmos_expected_<variant>_a<+x.xxx>.bmp        1920 x 1200 raw camera orientation (as captured)
  cmos_expected_<variant>_a<+x.xxx>_array.bmp  the 508 x 508 px detector-array crop shown 2 x 2 per px (1016 x 1016)
  cmos_expected_array.png                      detector-array crops for all a, both variants
Run: /c/Anaconda/envs/qpi/python.exe make_cmos_bmps.py"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import insitu_config

SCALE = 0.6
cfg = insitu_config.init_params()
y0, x0 = cfg.det_axis_yx[0] - 254, cfg.det_axis_yx[1] - 254
files = sorted(glob.glob(os.path.join(HERE, 'cmos_expected_*_a*.npy')))
crops = {}
for f in files:
    I = np.load(f).astype(np.float32)
    lev = np.clip(np.round(255 * I / SCALE), 0, 255).astype(np.uint8)
    base = f[:-4]
    Image.fromarray(lev, 'L').save(base + '.bmp')
    c = lev[y0:y0 + 508, x0:x0 + 508]
    Image.fromarray(np.ascontiguousarray(c.repeat(2, 0).repeat(2, 1)), 'L').save(base + '_array.bmp')
    crops[os.path.basename(base)[len('cmos_expected_'):]] = I[y0:y0 + 508, x0:x0 + 508]
    print(os.path.basename(base), f'array px max {I[y0:y0 + 508, x0:x0 + 508].max():.2f}, '
          f'saturated: {(I > SCALE).mean() * 100:.1f} % of the frame, {(c == 255).mean() * 100:.2f} % of the array')

names = sorted(crops, key=lambda n: (n.split('_')[0], float(n.split('_a')[1])))
na = len(names) // 2
fig, ax = plt.subplots(2, na, figsize=(4 * na, 8.4), constrained_layout=True)
for k, n in enumerate(names):
    a = ax.flat[k]
    im = a.imshow(crops[n], cmap='gray', vmin=0, vmax=SCALE, extent=(x0 - 0.5, x0 + 507.5, y0 + 507.5, y0 - 0.5))
    a.set_title(n.replace('_a', ' phases, a = '), fontsize=10)
    a.set_xlabel('raw camera col'); a.set_ylabel('raw camera row')
fig.colorbar(im, ax=ax, shrink=0.7, label='intensity / incident beam (clipped at 0.6)')
fig.suptitle(f'Expected camera image inside the detector array (508 x 508 raw px = 2.98 mm, rows {y0}..{y0 + 507}, '
             f'cols {x0}..{x0 + 507}), bench M1 8-bit, raw Basler orientation', fontsize=11)
fig.savefig(os.path.join(HERE, 'cmos_expected_array.png'), dpi=100)
print('wrote cmos_expected_array.png')
