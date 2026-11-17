"""Simulated CMOS images (cmos_sim_a*.npy from make_bench_check.py) -> 8-bit BMPs on the camera's own 5.86 um pixels, for
side-by-side comparison with captures. Linear, one fixed scale for every file: level = 255 * I / 1.8 (I in units of the
incident beam; the unmodulated beam is level 142, and the simulated maximum ~1.72 stays below 255).
  cmos_sim_a<+x.xx>_sensor.bmp   1920 x 1200 (full sensor, optical axis at the centre)
  cmos_sim_a<+x.xx>_center.bmp   512 x 512 central crop (3.0 mm), 1:1 pixels
  cmos_sim_a<+x.xx>_center_x3.bmp the same crop, each pixel shown 3 x 3 (easier to view)"""
import os, glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 1.8
for f in sorted(glob.glob(os.path.join(HERE, 'cmos_sim_a*.npy'))):
    I = np.load(f)
    c = I.shape[0] // 2
    lev = np.clip(np.round(255 * I / SCALE), 0, 255).astype(np.uint8)
    base = f[:-4]
    Image.fromarray(np.ascontiguousarray(lev[c - 600:c + 600, c - 960:c + 960]), 'L').save(base + '_sensor.bmp')
    crop = np.ascontiguousarray(lev[c - 256:c + 256, c - 256:c + 256])
    Image.fromarray(crop, 'L').save(base + '_center.bmp')
    Image.fromarray(np.ascontiguousarray(crop.repeat(3, 0).repeat(3, 1)), 'L').save(base + '_center_x3.bmp')
    print(os.path.basename(base), 'levels', lev.min(), '..', lev.max(), 'saturated px', int((I > SCALE).sum()))
