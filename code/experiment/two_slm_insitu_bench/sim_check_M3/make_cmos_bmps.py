"""Expected camera images (cmos_expected_a*.npy from make_sim_check.py) -> 8-bit BMPs + a zoom figure of the detector array.

The BMPs are scaled the way the camera should be exposed: from the DETECTOR ARRAY, not the whole frame (the open beam and
the rim around the array saturate).  level = 255 * I / SCALE, SCALE = the 99.9th percentile of the array pixels over
all a (printed), so each BMP looks like a capture exposed for the array.
  cmos_expected_a<+x.xxx>.bmp        1920 x 1200 raw camera orientation (as captured)
  cmos_expected_a<+x.xxx>_array.bmp  the 508 x 508 px detector-array crop shown 2 x 2 per px (1016 x 1016)
  cmos_expected_array.png            detector-array crops for all a
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

cfg = insitu_config.init_params()
y0, x0 = cfg.det_axis_yx[0] - 254, cfg.det_axis_yx[1] - 254
files = sorted(glob.glob(os.path.join(HERE, 'cmos_expected_a*.npy')), key=lambda f: float(f.split('_a')[-1][:-4]))
Is = {os.path.basename(f)[len('cmos_expected_'):-4]: np.load(f).astype(np.float32) for f in files}
SCALE = float(np.percentile(np.stack([I[y0:y0 + 508, x0:x0 + 508] for I in Is.values()]), 99.9))
print(f'SCALE = {SCALE:.3f} x the incident beam (99.9th percentile inside the detector array)')
for n, I in Is.items():
    lev = np.clip(np.round(255 * I / SCALE), 0, 255).astype(np.uint8)
    base = os.path.join(HERE, 'cmos_expected_' + n)
    Image.fromarray(lev, 'L').save(base + '.bmp')
    c = lev[y0:y0 + 508, x0:x0 + 508]
    Image.fromarray(np.ascontiguousarray(c.repeat(2, 0).repeat(2, 1)), 'L').save(base + '_array.bmp')
    print(n, f'array px max {I[y0:y0 + 508, x0:x0 + 508].max():.2f}, saturated: {(I > SCALE).mean() * 100:.1f} % of the frame, '
          f'{(c == 255).mean() * 100:.2f} % of the array')

fig, ax = plt.subplots(1, len(Is), figsize=(4.6 * len(Is), 4.6), constrained_layout=True)
for a, (n, I) in zip(ax, Is.items()):
    im = a.imshow(I[y0:y0 + 508, x0:x0 + 508], cmap='gray', vmin=0, vmax=SCALE, extent=(x0 - 0.5, x0 + 507.5, y0 + 507.5, y0 - 0.5))
    a.set_title(n.replace('a', 'a = ', 1), fontsize=10)
    a.set_xlabel('raw camera col'); a.set_ylabel('raw camera row')
fig.colorbar(im, ax=ax, shrink=0.8, label=f'intensity / incident beam (clipped at {SCALE:.2f})')
fig.suptitle(f'Expected camera image inside the detector array (508 x 508 raw px, rows {y0}..{y0 + 507}, cols {x0}..{x0 + 507}), '
             f'{os.path.basename(HERE)}, raw Basler orientation', fontsize=10)
fig.savefig(os.path.join(HERE, 'cmos_expected_array.png'), dpi=100)
print('wrote cmos_expected_array.png')
