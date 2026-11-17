"""In-situ time-multiplexed NFA fine-tuning -- main entry point.

What this script does
---------------------
Takes a trained lab checkpoint (ckpt/*.pth: M phase keys u1 for Pluto, M decoders u2 for Luna) and fine-tunes
them against the *real* optical system using policy-gradient (PPO).  Every epoch:

    1. Pick a_batch input values a, spread over cfg.a (shifted by one grid point every epoch).
    2. Sample S noisy variants of (u1, u2).
    3. For each variant, a and frame m: display key m on Pluto and (encoding(a) + decoder m) on Luna,
       capture the camera; average over the M frames and read the 64 x 64 detectors.
    4. Score each variant: -RMSE against the targets after one least-squares scale + offset
       (insitu_policy.compute_reward).
    5. Update u1, u2 via clipped PPO so they drift toward variants that
       scored higher.
Every checkpoint_every epochs: noise-free capture on all of cfg.a, logged with the test.py min-max RMSE.

Modes
-----
Training (default):     python insitu_TDNFA.py --ckpt ./ckpt/z2_5mm_M1_8bit.pth
ROI calibration:        python insitu_TDNFA.py --ckpt ./ckpt/... --calibrate-roi
Dry run (no hardware):  python insitu_TDNFA.py --ckpt ./ckpt/... --dry-run

Freezing one mask:      add --freeze-pluto or --freeze-luna.
Swap SLM display order: add --swap-slms.

Outputs land in `logs/insitu_<timestamp>/` (or wherever --run-dir points).
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches
from tqdm import tqdm

import insitu_config
import insitu_utils as U
import insitu_calib as C
from insitu_policy import PolicyGradientAgent2SLM, compute_reward


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser('In-situ QPI mask fine-tuning')
    p.add_argument('--ckpt', type=str, default=None,
                   help='Path to trained .pth (overrides config.ckpt_path)')
    p.add_argument('--iterations', type=int, default=None)
    p.add_argument('--states-num', type=int, default=None)
    p.add_argument('--sigma', type=float, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--freeze-pluto', action='store_true')
    p.add_argument('--freeze-luna',  action='store_true')
    p.add_argument('--run-dir', type=str, default=None,
                   help='Where to save logs/checkpoints (default: timestamped)')
    p.add_argument('--swap-slms', action='store_true',
                   help='If Pluto and Luna come up in the wrong display order')
    p.add_argument('--dry-run', action='store_true',
                   help='Skip hardware; sanity-check the loop on dummy captures')
    p.add_argument('--calibrate-roi', action='store_true',
                   help='ROI alignment check.  Display the centered-disk '
                        'calibration object, capture one frame, save annotated '
                        'overlays + report.  Skips training.')
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    """Mutate cfg in-place from CLI flags.  CLI takes precedence over config."""
    if args.ckpt:        cfg.ckpt_path  = args.ckpt
    if args.iterations:  cfg.iterations = args.iterations
    if args.states_num:  cfg.states_num = args.states_num
    if args.sigma:       cfg.sigma      = args.sigma
    if args.lr:          cfg.lr         = args.lr
    if args.run_dir:     cfg.run_dir    = args.run_dir
    cfg.freeze_pluto = args.freeze_pluto
    cfg.freeze_luna  = args.freeze_luna


# =====================================================================
# Hardware abstraction: dummies for --dry-run so the loop is testable
# on a dev machine without the SLM SDK / camera drivers installed.
# =====================================================================
class _DummySLM:
    def __init__(self, shape): self.shape = shape
    def showData(self, _arr): pass

class _DummyCam:
    """Fake camera that returns random uint8 frames the right shape."""
    def __init__(self, shape): self.shape = shape
    def start_capture(self): pass
    def read_next_image(self): pass
    def get_current_image(self):
        h, w = self.shape
        buf = (np.random.rand(h, w) * 50 + 100).astype(np.uint8).tobytes()
        return {"buffer": buf, "rows": h, "cols": w}


def init_hardware(cfg, dry_run, swap_slms):
    """Open SLMs (Pluto then Luna) and the camera.  In --dry-run mode,
    everything is replaced with shape-correct dummies so the loop runs
    on any machine."""
    if dry_run:
        pluto = _DummySLM((cfg.slm1_h, cfg.slm1_w))
        luna  = _DummySLM((cfg.slm2_h, cfg.slm2_w))
        cam   = _DummyCam((cfg.sen_x_num, cfg.sen_y_num))
        cam.start_capture()
        return pluto, luna, cam
    pluto, luna = U.init_two_slms()
    # The SDK opens whichever SLM the OS enumerates first.  If you wired
    # things such that Luna comes up first, this flag swaps them.
    if swap_slms:
        pluto, luna = luna, pluto
    cam = U.init_cam(cam_num=cfg.cam_num)
    cam.start_capture()
    return pluto, luna, cam


# =====================================================================
# SLM patterning + detector readout, hard-coded for the Np4 / M-key lab model (model.py layout; checked level-by-level
# against the M3 checkpoint and against the model's detector readout to 2e-7).
#   Pluto: 120 x 120 key centred, phase 0 elsewhere.
#   Luna:  2x2 patch [[0,1],[2,3]] of 28 x 28 px tiled 3x3 -> 168 x 168 encoding, centred; harmonic p has phase
#          2 pi alpha_p a; the 213 x 213 decoder is added over it (encoding starts at row / col 22 of the decoder);
#          the sum is quantised once.  Phase 0 elsewhere.
#   Camera: 64 x 64 detectors, 1 camera px each on a 2 px pitch, centred on the optical axis (det_axis_yx):
#          detector (r, c) = pixel (axis - 63 + 2r, axis - 63 + 2c), flattened k = r * 64 + c.
# =====================================================================
ENC_IDS = np.tile([[0, 1], [2, 3]], (3, 3)).repeat(28, 0).repeat(28, 1)   # [168, 168] harmonic index per Luna px


def q8(phi, inverted=False):
    """model._quantize: level k <-> phase k * 2pi / 256 (inverted LUT: 255 -> phase 0)."""
    k = np.round(np.mod(phi, 2 * np.pi) / (2 * np.pi / 256)) % 256
    return ((256 - k) % 256 if inverted else k).astype(np.uint8)


def pluto_frame(key_m, cfg):
    f = np.zeros((1080, 1920))
    y0, x0 = cfg.phase_key_center_y - 60, cfg.phase_key_center_x - 60
    f[y0:y0 + 120, x0:x0 + 120] = torch.as_tensor(key_m).detach().cpu().numpy()
    k = q8(f, cfg.pluto_lut_inverted)
    k = k[::-1] if cfg.pluto_flip_ud else k
    k = k[:, ::-1] if cfg.pluto_flip_lr else k
    return np.ascontiguousarray(k)


def luna_frame(decoder_m, a, cfg):
    sq = torch.as_tensor(decoder_m).detach().cpu().numpy().astype(np.float64)       # [213, 213]
    sq[22:190, 22:190] += 2 * np.pi * np.asarray(cfg.alpha)[ENC_IDS] * float(a)
    f = np.zeros((1080, 1920))
    y0, x0 = cfg.decoder_key_center_y - 84 - 22, cfg.decoder_key_center_x - 84 - 22
    f[y0:y0 + 213, x0:x0 + 213] = sq
    k = q8(f, cfg.luna_lut_inverted)
    k = k[::-1] if cfg.luna_flip_ud else k
    k = k[:, ::-1] if cfg.luna_flip_lr else k
    return np.ascontiguousarray(k)


def display_and_capture(pluto, luna, cam, key_m, decoder_m, a, cfg, show_pluto=True):
    """One capture: key m on Pluto (skipped if already shown), encoding(a) + decoder m on Luna, wait for the SLMs
    to settle, grab an averaged frame.  Returns the *raw* averaged frame (full camera shape)."""
    if show_pluto:
        pluto.showData(pluto_frame(key_m, cfg))
    luna.showData(luna_frame(decoder_m, a, cfg))
    # SLMs are not instantaneous -- the LCOS pixels take a few ms to settle
    # to the new phase value.  capture_avg also throws away one frame to
    # flush the camera buffer before measuring.
    time.sleep(cfg.slm_settle)
    return U.capture_avg(cam, U.caputre_img,
                         num=cfg.capture_avg_n, delay=cfg.capture_avg_delay)


def capture_state(pluto, luna, cam, keys, decoders, a_values, cfg):
    """All captures for one mask state, keys outer (Pluto changes only M times), a inner.  Returns the frames
    averaged over the M keys (the model averages intensities over frames): [len(a_values), H_cam, W_cam]."""
    acc = 0
    for m in range(keys.shape[0]):
        acc = acc + np.stack([display_and_capture(pluto, luna, cam, keys[m], decoders[m], a, cfg, show_pluto=(i == 0))
                              for i, a in enumerate(a_values)])
    return acc / keys.shape[0]


def det_origin(cfg, frame_hw, drift_shift):
    """Camera pixel of detector (0, 0): optical axis (det_axis_yx, frame centre if None) + drift - 63."""
    ay, ax = cfg.det_axis_yx if cfg.det_axis_yx is not None else (frame_hw[0] // 2, frame_hw[1] // 2)
    return ay + int(drift_shift[0]) - 63, ax + int(drift_shift[1]) - 63


def captures_to_detectors(frames, drift_shift, cfg, device):
    """Raw frames [..., H, W] (already averaged over the M keys) -> detector readings [..., 4096] on device.
    No normalisation here: the reward / metric does it (a uniform offset cancels)."""
    frames = np.asarray(frames, dtype=np.float32)
    y0, x0 = det_origin(cfg, frames.shape[-2:], drift_shift)
    assert 0 <= y0 and y0 + 127 <= frames.shape[-2] and 0 <= x0 and x0 + 127 <= frames.shape[-1],         f'detector array at ({y0}, {x0}) + 127 px falls off the frame: check det_axis_yx / drift'
    I = frames[..., y0:y0 + 128:2, x0:x0 + 128:2]                       # [..., 64, 64]
    I = I[..., ::-1, :] if cfg.cam_flip_ud else I
    I = I[..., :, ::-1] if cfg.cam_flip_lr else I
    return torch.from_numpy(np.ascontiguousarray(I).reshape(*I.shape[:-2], 4096)).to(device)


def save_detector_crop(frame, cfg, drift_shift, path, title=''):
    """Debug image: the detector region (+8 px) with every detector framed in cyan, plus the crop as .npy.
    If the squares do not sit on the bright spots, det_axis_yx (or the drift) is off."""
    y0, x0 = det_origin(cfg, frame.shape[-2:], drift_shift)
    crop = np.asarray(frame, dtype=np.float32)[max(y0 - 8, 0):y0 + 135, max(x0 - 8, 0):x0 + 135]
    np.save(os.path.splitext(path)[0] + '.npy', crop)
    oy, ox = min(y0, 8), min(x0, 8)
    gy, gx = np.meshgrid(oy + 2 * np.arange(64), ox + 2 * np.arange(64), indexing='ij')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop, cmap='gray', interpolation='nearest')
    ax.scatter(gx.ravel(), gy.ravel(), s=6, marker='s', facecolors='none', edgecolors='cyan', linewidths=0.35)
    ax.set_title(f'{title}  detector (0, 0) at camera ({y0}, {x0})', fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# ROI calibration mode (--calibrate-roi)
# =====================================================================
def calibrate_roi(cfg, agent, pluto, luna, cam, targets):
    """Find where the detector array sits on the camera (det_axis_yx) and the flips, from the trained masks.

    What it does:
      1. For each Pluto flip (4 combos), displays the trained masks (no noise) at 8 a values spread over cfg.a and
         captures them (val_capture_avg_n frames each).  Flipping Pluto alone covers every relative Pluto / Luna
         mirroring: a mirror of the whole system is just a mirrored camera image, which the camera flips cover.
      2. Intensity centroid of the mean frame -> first guess of the optical axis.
      3. Pure post-processing: tries every det_axis_yx within +-4 px of the guess and all 4 camera flips, scores each
         with compute_reward against the targets (affine RMSE, the training reward) and keeps the best.  One pixel
         off reads the gaps between detectors, so the right registration stands out sharply.
      4. Saves diagnostics and prints the values to paste into insitu_config.py.

    Run this BEFORE the first training session (and after any realignment).
    """
    a_idx = np.arange(8) * (len(cfg.a) // 8)
    keep = (cfg.pluto_flip_ud, cfg.pluto_flip_lr, cfg.det_axis_yx, cfg.cam_flip_ud, cfg.cam_flip_lr)
    print(f'\n=== ROI calibration ===')
    print(f'Trained masks at a = {np.round(cfg.a[a_idx], 3).tolist()}, {cfg.val_capture_avg_n} frames each')

    results = []                         # (rmse, pluto_ud, pluto_lr, axis, cam_ud, cam_lr)
    captured = {}                        # (pluto_ud, pluto_lr) -> (frames, mean frame, centroid)
    for p_ud in (False, True):
        for p_lr in (False, True):
            cfg.pluto_flip_ud, cfg.pluto_flip_lr = p_ud, p_lr
            frames = capture_state(pluto, luna, cam, agent.u1, agent.u2, cfg.a[a_idx], cfg)
            for _ in range(cfg.val_capture_avg_n - 1):
                frames = frames + capture_state(pluto, luna, cam, agent.u1, agent.u2, cfg.a[a_idx], cfg)
            frames = frames / cfg.val_capture_avg_n                                  # [8, H, W]

            # ---- Centroid of the mean frame (median-subtracted): first guess of the axis ----
            raw = frames.mean(0)
            H, W = raw.shape
            sig = np.maximum(raw - np.median(raw), 0).astype(np.float64)
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            cy = float((yy * sig).sum() / (sig.sum() + 1e-8))
            cx = float((xx * sig).sum() / (sig.sum() + 1e-8))
            captured[(p_ud, p_lr)] = (frames, raw, (cy, cx))

            # ---- Search the axis (+-4 px) and the camera flips ----
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    for c_ud in (False, True):
                        for c_lr in (False, True):
                            cfg.det_axis_yx = (int(round(cy)) + dy, int(round(cx)) + dx)
                            cfg.cam_flip_ud, cfg.cam_flip_lr = c_ud, c_lr
                            try:
                                det = captures_to_detectors(frames, (0, 0), cfg, targets.device)
                            except AssertionError:
                                continue                                             # array off the frame
                            rmse = -float(compute_reward(det[None], targets[a_idx], cfg)[0])
                            results.append((rmse, p_ud, p_lr, cfg.det_axis_yx, c_ud, c_lr))

    results.sort(key=lambda r: r[0])
    rmse, p_ud, p_lr, axis, c_ud, c_lr = results[0]
    second = results[1] if len(results) > 1 else None
    best_frames, best_raw, best_c = captured[(p_ud, p_lr)]
    cfg.pluto_flip_ud, cfg.pluto_flip_lr, cfg.det_axis_yx, cfg.cam_flip_ud, cfg.cam_flip_lr = \
        p_ud, p_lr, axis, c_ud, c_lr
    sat_frac = float((best_frames > 250).mean())                  # 8-bit camera: > 250 counts as saturated

    # ---- Save diagnostic figures ----
    out = cfg.run_dir
    os.makedirs(out, exist_ok=True)

    # 1) Raw mean frame with the detector-array window, the centroid and the chosen axis
    y0, x0 = det_origin(cfg, best_raw.shape, (0, 0))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(best_raw, cmap='gray')
    ax.add_patch(patches.Rectangle((x0, y0), 127, 127, linewidth=1.5, edgecolor='red', facecolor='none',
                                   label='detector array'))
    ax.plot(best_c[1], best_c[0], 'r+', markersize=22, markeredgewidth=2, label='centroid')
    ax.plot(axis[1], axis[0], 'c+', markersize=18, markeredgewidth=2, label=f'det_axis_yx {axis}')
    ax.set_title(f'ROI check  |  best affine RMSE {rmse:.3e}')
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(out, 'roi_raw_overlay.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # 2) Detector grid on the camera with the chosen settings (zoom in: squares should sit on the spots)
    save_detector_crop(best_frames[len(a_idx) // 2], cfg, (0, 0), os.path.join(out, 'roi_detector_grid.png'),
                       title=f'a = {cfg.a[a_idx[len(a_idx) // 2]]:+.3f}')

    # 3) Pixel-value histogram (saturation diagnosis)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(best_frames.ravel(), bins=64, range=(0, 256), color='gray')
    ax.axvline(250, color='red', linestyle='--', label='saturation (>250)')
    ax.set_xlabel('pixel value (0-255)')
    ax.set_ylabel('count')
    ax.set_title(f'Intensity histogram  |  saturated fraction = {sat_frac*100:.3f}%')
    ax.legend()
    fig.savefig(os.path.join(out, 'roi_histogram.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- Console report ----
    print(f'\n--- ROI calibration report ---')
    print(f'  Camera frame:        {best_raw.shape[0]} x {best_raw.shape[1]}')
    print(f'  Centroid (guess):    (y={best_c[0]:.1f}, x={best_c[1]:.1f})')
    print(f'  Best:                axis {axis}, cam flips ud={c_ud} lr={c_lr}, pluto flips ud={p_ud} lr={p_lr}'
          f'  ->  affine RMSE {rmse:.3e}')
    if second is not None:
        print(f'  Runner-up:           axis {second[3]}, cam ud={second[4]} lr={second[5]}, pluto ud={second[1]} '
              f'lr={second[2]}  ->  {second[0]:.3e}  ({second[0] / rmse:.1f}x worse; want >> 1)')
    if max(abs(axis[0] - round(best_c[0])), abs(axis[1] - round(best_c[1]))) == 4:
        print('  WARNING: best axis is on the edge of the +-4 px search -- re-run with det_axis_yx set to it')
    print(f'  Saturated pixels:    {sat_frac*100:.3f} %   (target: < 1 %)')

    print(f'\n  Recommended config edit (insitu_config.py):')
    print(f'      c.det_axis_yx = {axis}')
    print(f'      c.cam_flip_ud, c.cam_flip_lr = {c_ud}, {c_lr}')
    print(f'      c.pluto_flip_ud, c.pluto_flip_lr = {p_ud}, {p_lr}')

    print(f'\n  Saved diagnostics:')
    print(f'      {out}/roi_raw_overlay.png    (red window should cover the bright region)')
    print(f'      {out}/roi_detector_grid.png  (cyan squares should sit on the detector spots)')
    print(f'      {out}/roi_histogram.png      (peak should be mid-range, not at 255)')

    cfg.pluto_flip_ud, cfg.pluto_flip_lr, cfg.det_axis_yx, cfg.cam_flip_ud, cfg.cam_flip_lr = keep
    return results[0], sat_frac


# =====================================================================
# Main training loop
# =====================================================================
def main():
    args = parse_args()
    cfg = insitu_config.init_params()
    apply_cli_overrides(cfg, args)

    # Set up run directory + log file.  We dump config.json so the run is
    # self-documenting (you can read what hyperparameters were used months
    # later without re-deriving from logs).
    os.makedirs(cfg.run_dir, exist_ok=True)
    drift_dir = os.path.join(cfg.run_dir, 'drift')
    os.makedirs(drift_dir, exist_ok=True)
    log_file = os.path.join(cfg.run_dir, 'log.txt')
    with open(os.path.join(cfg.run_dir, 'config.json'), 'w') as f:
        json.dump({k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
                   for k, v in vars(cfg).items()}, f, indent=2)

    U.init_random(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    U.print_save(f'Device: {device}', log_file)

    # --- Load the trained masks.  The trained .pth contains the full D2NN
    # state_dict; we only need the two mask_phase tensors.  See
    # insitu_calib.load_trained_masks for how the keys are located. ---
    U.print_save(f'Loading checkpoint: {cfg.ckpt_path}', log_file)
    keys, decoders, tc = C.load_trained_masks(cfg.ckpt_path)
    U.print_save(f'  Pluto mask shape: {tuple(keys.shape)} '
                 f'(range [{keys.min():.3f}, {keys.max():.3f}])', log_file)
    U.print_save(f'  Luna  mask shape: {tuple(decoders.shape)} '
                 f'(range [{decoders.min():.3f}, {decoders.max():.3f}])', log_file)

    agent = PolicyGradientAgent2SLM(cfg, keys, decoders,
                                    freeze_pluto=cfg.freeze_pluto,
                                    freeze_luna=cfg.freeze_luna).to(device)
    U.print_save(f'Frozen: pluto={cfg.freeze_pluto}  luna={cfg.freeze_luna}',
                 log_file)

    # --- Targets: [len(cfg.a), Nf] on the cfg.a grid, min-max'd over that grid as test.py does
    # (written by export_targets.py from the same checkpoint). ---
    targets = torch.from_numpy(np.load(cfg.targets_path)).float().to(device)
    assert targets.shape == (len(cfg.a), cfg.Nf), f'targets {tuple(targets.shape)} vs ({len(cfg.a)}, {cfg.Nf})'
    U.print_save(f'Targets: {cfg.targets_path} {tuple(targets.shape)}', log_file)

    # --- Initialize hardware ---
    U.print_save('Initializing SLMs and camera...', log_file)
    pluto, luna, cam = init_hardware(cfg, args.dry_run, args.swap_slms)

    # =====================================================================
    # ROI calibration short-circuit: do the alignment check, save outputs,
    # exit before any training starts.  No checkpoints, no PPO updates.
    # =====================================================================
    if args.calibrate_roi:
        calibrate_roi(cfg, agent, pluto, luna, cam, targets)
        U.print_save('\nROI calibration done. Edit insitu_config.py with the '
                     'recommended det_axis_yx / flips, then re-run without '
                     '--calibrate-roi.', log_file)
        return

    # --- Capture drift baseline: a reference pattern we'll re-display every
    # `drift_check_every` epochs to detect when the optics has wandered.
    # The reference is a centered square; the drift monitor cross-correlates
    # against this baseline to estimate (dy, dx) shift.
    ref_pattern = U.create_pattern((cfg.slm1_h, cfg.slm1_w),
                                   square_size=min(cfg.slm1_h,
                                                   cfg.slm1_w) // 4)
    reference = U.capture_reference(cam, pluto, U.caputre_img, ref_pattern,
                                    num=10, project_delay=0.3)
    np.save(os.path.join(cfg.run_dir, 'drift_reference.npy'), reference)
    plt.imsave(os.path.join(cfg.run_dir, 'drift_reference.png'),
               reference, cmap='gray')

    # Running drift correction.  Updated every cfg.drift_check_every epochs.
    shift = np.array([0, 0])

    # =====================================================================
    # Training loop
    # =====================================================================
    rewards_history = []
    rmse_history = []
    pbar = tqdm(range(cfg.iterations))
    t0 = time.time()

    for epoch in pbar:
        agent.epoch = epoch
        agent.update_sigma(epoch)

        # --- Drift check.  If the optics has shifted, update `shift` so
        # subsequent center-crops compensate. ---
        if epoch % cfg.drift_check_every == 0:
            snr, sh = U.monitor_drift(cam, pluto, U.caputre_img,
                                      reference, ref_pattern,
                                      save_dir=drift_dir, check_id=epoch)
            shift = sh
            U.print_save(f'  [drift] epoch {epoch}: SNR={snr:.2f} dB '
                         f'shift={tuple(sh)}', log_file)

        # --- Pick a_batch a values spread over the grid, shifted by one grid point every epoch,
        # so every a in cfg.a is visited every len(cfg.a) // a_batch epochs. ---
        a_idx = (epoch + np.arange(cfg.a_batch) * (len(cfg.a) // cfg.a_batch)) % len(cfg.a)

        # --- Sample S noisy mask pairs.  u1_s, u2_s are detached samples;
        # logp1, logp2 are their per-pixel log-probabilities under the OLD
        # policy (needed for the PPO importance ratio). ---
        u1_s, u2_s, logp1, logp2 = agent.sample_states(cfg.states_num)

        # --- Display + capture, one state at a time.  This loop is the
        # wall-clock bottleneck: ~0.2 s per state in real hardware. ---
        tik = time.time()
        captures = []
        for s in range(cfg.states_num):
            img = capture_state(pluto, luna, cam, u1_s[s], u2_s[s], cfg.a[a_idx], cfg)
            captures.append(img)
        capture_t = time.time() - tik

        # --- Camera frames -> detector readings, scored against the targets. ---
        opred = captures_to_detectors(np.stack(captures), shift, cfg, device)   # [S, a_batch, Nf]
        rewards = compute_reward(opred, targets[a_idx], cfg)

        # --- PPO update.  Uses the same batch of samples for K inner steps
        # with importance-ratio clipping, then commits the new (u1, u2). ---
        avg_r = agent.update_ppo(u1_s, u2_s, logp1, logp2, rewards)

        # --- Bookkeeping. ---
        rewards_history.append(avg_r)
        pbar.set_description(f'ep{epoch} rmse={-avg_r:.3e} cap={capture_t:.1f}s')
        U.print_save(f'epoch {epoch:4d}  a_idx={a_idx[0]:3d}+  mean affine RMSE={-avg_r:.4e}  '
                     f'capture={capture_t:.1f}s', log_file)

        # --- Periodic checkpoint + a deterministic eval capture (no noise).
        # The eval capture is what you actually want to look at to track
        # convergence -- it's the masks "as they currently stand", not a
        # noisy sample. ---
        if (epoch + 1) % cfg.checkpoint_every == 0:
            agent.save(os.path.join(cfg.run_dir, f'agent_ep{epoch:04d}.pth'))
            with torch.no_grad():
                eval_img = capture_state(pluto, luna, cam, agent.u1, agent.u2, cfg.a, cfg)
                for _ in range(cfg.val_capture_avg_n - 1):
                    eval_img = eval_img + capture_state(pluto, luna, cam, agent.u1, agent.u2, cfg.a, cfg)
                eval_img = eval_img / cfg.val_capture_avg_n          # [len(cfg.a), H_cam, W_cam]
                # detector readings [len(cfg.a), Nf] (the full frames are too large to keep every checkpoint)
                eval_det = captures_to_detectors(eval_img, shift, cfg, 'cpu').numpy()
                np.save(os.path.join(cfg.run_dir, f'eval_ep{epoch:04d}_detectors.npy'), eval_det)
                # test.py metric: min-max over all detectors and the whole a grid, mean of per-function RMSE
                f_hat = (eval_det - eval_det.min()) / (eval_det.max() - eval_det.min())
                rmse = float(np.sqrt(((f_hat - targets.cpu().numpy()) ** 2).mean(0)).mean())
                rmse_history.append((epoch, rmse))
                U.print_save(f'  [eval] epoch {epoch}: min-max RMSE on all {len(cfg.a)} a = {rmse:.4e}', log_file)
                if cfg.det_debug_every and (epoch + 1) % cfg.det_debug_every == 0:
                    ia = len(cfg.a) // 2
                    save_detector_crop(eval_img[ia], cfg, shift,
                                       os.path.join(cfg.run_dir, f'eval_ep{epoch:04d}_detgrid.png'),
                                       title=f'epoch {epoch}, a = {cfg.a[ia]:+.3f}')

    # =====================================================================
    # End of training: save final masks + reward curve.
    # =====================================================================
    elapsed = time.time() - t0
    U.print_save(f'\nDone.  Total wall time: {elapsed/60:.1f} min', log_file)
    agent.save(os.path.join(cfg.run_dir, 'agent_final.pth'))

    np.save(os.path.join(cfg.run_dir, 'rewards.npy'),
            np.array(rewards_history))
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history)
    plt.xlabel('epoch'); plt.ylabel('mean reward')
    plt.title('In-situ training reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg.run_dir, 'rewards.png'), dpi=120)
    plt.close()


if __name__ == '__main__':
    main()
