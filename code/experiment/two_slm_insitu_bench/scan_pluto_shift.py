"""Bench scan of the PLUTO key position (pluto_shift_yx): the hardware counterpart of key_shift_scan_sim.py.

In simulation the bench M2 / M3 masks lose most of their signal 1 PLUTO px off and all of it at 2-3 px (key_shift_scan.txt,
README section 0), and the registration (-12, +30) is only known to +-3 px.  This script finds the key position on the
bench before any in-situ training:

  for every shift (dy, dx) on a grid of +-RANGE PLUTO px (step STEP) around cfg.pluto_shift_yx:
      display the trained masks (all M key / decoder pairs, no noise) at N_A a values, capture (mean over the M frames)
      score: affine RMSE (insitu_policy.compute_reward, the in-situ reward) against the targets, taking the best detector
             axis within +-AXIS_SEARCH camera px of cfg.det_axis_yx (post-processing only, camera flips from the config)
  report the best shift, the runner-up, and the no-signal level (a readout that carries no information scores ~8.7e-2)

Shift convention = cfg.pluto_shift_yx: (rows, cols) in the MODEL frame, before the PLUTO rot180 -- the key centre is
(slm1_h // 2 + dy, slm1_w // 2 + dx).  The frames go through insitu_TDNFA.pluto_frame / luna_frame, so every display
correction in your copy of the package (collimator lens, conjugate, phase table, flips) is applied as in training.

What to expect (sim, linear phases): right position <= ~3.5e-2 (M2) / ~2.6e-2 (M3), even if 0.5 px off; 1 px off 5-6e-2;
2+ px off ~8e-2.  The phase table and distance / magnification errors raise all of these, so judge the map by contrast
(best vs runner-up vs no-signal), not by the absolute number.  A flat map at ~8.7e-2 = no key found: check z1, z2, m,
the collimator and the detector axis before going further.

Run (bench PC, from this folder):
    python scan_pluto_shift.py --ckpt ./ckpt_M2/bench_M2_bin1_ep4.pth
    options: --range 3 --step 1 --n-a 8 --axis-search 4 --repeats 1 --swap-slms --run-dir <dir>
Time: (2 RANGE / STEP + 1)^2 shifts x N_A x M displays x (slm_settle + capture_avg_n grabs) ~ 49 x 8 x M x 0.3 s
      = ~4 min (M2), ~6 min (M3) at the defaults.
Outputs (run dir, default logs/shiftscan_<timestamp>/): shift_scan.csv, shift_scan.png (reward map), best_frame.npy
(mean raw capture at the best shift, middle a), report.txt.  Paste the printed pluto_shift_yx into insitu_config.py.
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import insitu_config
import insitu_calib as C
import insitu_TDNFA as TD
from insitu_policy import compute_reward

p = argparse.ArgumentParser('Bench scan of the PLUTO key shift')
p.add_argument('--ckpt', type=str, default=None, help='trained .pth (default: config ckpt_path)')
p.add_argument('--range', type=int, default=3, help='scan +-RANGE PLUTO px around cfg.pluto_shift_yx')
p.add_argument('--step', type=int, default=1, help='grid step in PLUTO px')
p.add_argument('--n-a', type=int, default=8, help='a values per shift (spread over cfg.a)')
p.add_argument('--axis-search', type=int, default=4, help='detector axis search, +- camera px around det_axis_yx')
p.add_argument('--repeats', type=int, default=1, help='captures averaged per shift (each = M x N_A displays)')
p.add_argument('--swap-slms', action='store_true')
p.add_argument('--run-dir', type=str, default=None)
args = p.parse_args()

cfg = insitu_config.init_params()
if args.ckpt:
    cfg.ckpt_path = args.ckpt
out = args.run_dir or os.path.join('logs', time.strftime('shiftscan_%Y%m%d-%H%M%S'))
os.makedirs(out, exist_ok=True)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
keys, decoders, tc = C.load_trained_masks(cfg.ckpt_path)
M = keys.shape[0]
targets = torch.from_numpy(np.load(cfg.targets_path)).float().to(device)
assert targets.shape == (len(cfg.a), cfg.Nf), f'targets {tuple(targets.shape)} vs ({len(cfg.a)}, {cfg.Nf})'
a_idx = np.arange(args.n_a) * (len(cfg.a) // args.n_a)
T = targets[a_idx]
g = torch.Generator().manual_seed(0)
no_signal = -float(compute_reward((1 + 1e-6 * torch.randn(1, len(a_idx), cfg.Nf, generator=g)).to(device), T, cfg)[0])

base = tuple(cfg.pluto_shift_yx)
offs = np.arange(-args.range, args.range + 1, args.step)
R = args.axis_search
ax_offs = [(oy, ox) for oy in range(-R, R + 1) for ox in range(-R, R + 1)]
n_disp = len(offs) ** 2 * len(a_idx) * M * args.repeats
say(f'PLUTO shift scan: ckpt {cfg.ckpt_path} (M {M}), base pluto_shift_yx {base}, offsets {offs.tolist()} px (rows and cols), '
    f'{len(a_idx)} a values, detector axis {cfg.det_axis_yx} +-{R} camera px, {n_disp} displays '
    f'(~{n_disp * (cfg.slm_settle + 0.03 * (cfg.capture_avg_n + 1)) / 60:.0f} min)')
say(f'no-signal reference (affine RMSE of a readout without information, these a values): {no_signal:.3e}')

pluto, luna, cam = TD.init_hardware(cfg, False, args.swap_slms)

res = np.full((len(offs), len(offs)), np.nan)          # best affine RMSE per shift (over the axis search)
res_axis0 = np.full((len(offs), len(offs)), np.nan)    # at det_axis_yx exactly
best_ax = {}
rows = []
best = (np.inf, None, None)
t0 = time.time()
for iy, dy in enumerate(offs):
    for ix, dx in enumerate(offs):
        sy, sx = base[0] + int(dy), base[1] + int(dx)
        cfg.phase_key_center_y, cfg.phase_key_center_x = cfg.slm1_h // 2 + sy, cfg.slm1_w // 2 + sx
        frames = 0
        for _ in range(args.repeats):
            frames = frames + TD.capture_state(pluto, luna, cam, keys, decoders, cfg.a[a_idx], cfg)
        frames = frames / args.repeats                                                 # [N_A, H, W]
        dets, used = [], []
        for o in ax_offs:
            try:
                dets.append(TD.captures_to_detectors(frames, o, cfg, device).float()); used.append(o)
            except AssertionError:
                pass                                                                   # array off the frame
        r = -compute_reward(torch.stack(dets), T, cfg).cpu().numpy()                  # [n_axis]
        k = int(np.argmin(r))
        res[iy, ix], best_ax[(sy, sx)] = r[k], used[k]
        res_axis0[iy, ix] = r[used.index((0, 0))] if (0, 0) in used else np.nan
        rows.append((sy, sx, r[k], used[k][0], used[k][1], res_axis0[iy, ix]))
        if r[k] < best[0]:
            best = (r[k], (sy, sx), frames[len(a_idx) // 2].copy())
        print(f'  shift ({sy:+d}, {sx:+d}): affine RMSE {r[k]:.3e} (axis offset {used[k]}), at det_axis_yx {res_axis0[iy, ix]:.3e}'
              f'   [{time.time() - t0:.0f} s]', flush=True)

cfg.phase_key_center_y, cfg.phase_key_center_x = cfg.slm1_h // 2 + base[0], cfg.slm1_w // 2 + base[1]
order = sorted(rows, key=lambda r: r[2])
b, second = order[0], (order[1] if len(order) > 1 else None)
ay, ax = cfg.det_axis_yx
say(f'\nbest:      pluto_shift_yx ({b[0]}, {b[1]}), affine RMSE {b[2]:.3e}, detector axis ({ay + b[3]}, {ax + b[4]})')
if second is not None:
    say(f'runner-up: pluto_shift_yx ({second[0]}, {second[1]}), {second[2]:.3e} ({second[2] / b[2]:.2f}x the best)')
say(f'no-signal level {no_signal:.3e}: best is {no_signal / b[2]:.1f}x below it, median shift {np.nanmedian(res) / b[2]:.2f}x the best')
if max(abs(b[0] - base[0]), abs(b[1] - base[1])) >= args.range:
    say('WARNING: best shift is on the edge of the scan -- re-run with pluto_shift_yx set to it')
if no_signal / b[2] < 1.5:
    say('WARNING: the best shift is barely below the no-signal level -- no key found; check z1 / z2 / m, the collimator and '
        'the detector axis before trusting this')
say(f'\nRecommended config edit (insitu_config.py):\n    c.pluto_shift_yx = ({b[0]}, {b[1]})')
if (b[3], b[4]) != (0, 0):
    say(f'    c.det_axis_yx = ({ay + b[3]}, {ax + b[4]})      (then confirm with --calibrate-roi)')

np.savetxt(os.path.join(out, 'shift_scan.csv'), np.array(rows, dtype=float), delimiter=',', fmt='%.6g',
           header='shift_y,shift_x,affine_rmse_best_axis,axis_dy,axis_dx,affine_rmse_at_det_axis_yx', comments='')
np.save(os.path.join(out, 'best_frame.npy'), best[2].astype(np.float32))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for a_, data, ttl in ((axs[0], res, 'best detector axis (+-%d px)' % R), (axs[1], res_axis0, f'detector axis {cfg.det_axis_yx}')):
    ext = (base[1] + offs[0] - args.step / 2, base[1] + offs[-1] + args.step / 2,
           base[0] + offs[-1] + args.step / 2, base[0] + offs[0] - args.step / 2)
    im = a_.imshow(data, cmap='viridis_r', extent=ext, vmin=np.nanmin(res), vmax=max(no_signal, np.nanmax(res)))
    a_.plot(b[1], b[0], 'r+', ms=14, mew=2)
    a_.set_xlabel('pluto_shift_yx cols (PLUTO px, model frame)'); a_.set_ylabel('pluto_shift_yx rows')
    a_.set_title(ttl, fontsize=10)
fig.colorbar(im, ax=axs, label=f'affine RMSE (no signal = {no_signal:.2e})')
fig.suptitle(f'PLUTO key shift scan, M{M} ({os.path.basename(cfg.ckpt_path)}), {len(a_idx)} a values; red + = best', fontsize=11)
fig.savefig(os.path.join(out, 'shift_scan.png'), dpi=120)
plt.close(fig)
open(os.path.join(out, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
print(f'\nSaved {out}/shift_scan.png, shift_scan.csv, best_frame.npy, report.txt')
