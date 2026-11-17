'''
acquire.py  --  Automated SLM mask display and CCD acquisition
==============================================================
Reads the SLM mask BMPs produced by calibrate.py for one sample (or all
samples in a calibration folder), displays each on the SLM, captures and
averages N frames, then saves the results in a format reconstruct.py can
read directly.

Workflow
--------
  1. Display blank SLM  →  capture dark frame
  2. For m = 0 … M-1:
       display mask_m_slm.bmp  →  wait  →  average N frames  →  save mask_m_meas.bmp
  3. Save dark_meas.bmp
  4. Print summary

Output folder  (<sample_dir>/masks_meas/):
    mask_00_meas.bmp   — 8-bit grayscale, 1936×1216
    mask_01_meas.bmp
    ...
    dark_meas.bmp      — dark frame (blank SLM)
    acquire_log.json   — timestamps, ROI stats, settings

Usage
-----
    python acquire.py                                      # single sample (DEFAULT_SAMPLE_DIR)
    python acquire.py --sample_dir path/to/sample_0000_label7
    python acquire.py --calibration_dir path/to/calibration  # all sample_* folders
    python acquire.py --calibration_dir path/to/calibration --n_avg 5 --exposure_ms 10
'''

import os
import json
import argparse
import time
import datetime
import numpy as np
from PIL import Image

from utils import init_slm, init_cam, set_config_cam, capture_img

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  —  edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════════

# Path to a single calibration sample folder (used when --sample_dir is not given)
DEFAULT_SAMPLE_DIR = (
    r'C:\Users\Ary\Documents\Projects\Single Pixel QPI'
    r'\logs\20260524-1932-SinglePixelQPI-M10-N28-simdx4um-slm500px_slmdx8um-objdx8um'
    r'\calibration\sample_0000_label7'
)

# Path to the calibration folder containing all sample_* subdirectories.
# Set this (or pass --calibration_dir) to process every sample automatically.
DEFAULT_CALIBRATION_DIR = (
    r'E:\calibration v4'
)   

# Camera settings
EXPOSURE_MS   = 10.0    # exposure time in milliseconds — tune to your setup
GAIN          = 1       # camera gain (raw ADU units)
FPS           = 10      # frames per second (used when camera is in free-run mode)
CAM_WIDTH     = 1936    # capture width  [px]
CAM_HEIGHT    = 1216    # capture height [px]
CAM_OFFSET_X  = 0       # horizontal offset [px]
CAM_OFFSET_Y  = 0       # vertical offset   [px]

# Acquisition settings
N_AVG           = 1     # frames to average per mask (1 = single shot)
WARMUP_DELAY_S  = 0.5   # seconds to wait after displaying a new mask on the SLM
BETWEEN_FRAMES_S = 0.05 # seconds between consecutive frames within one mask capture
DISCARD_FIRST   = True  # discard one frame after each SLM update before averaging

# SLM settings
SLM_H = 1080
SLM_W = 1920

# ═══════════════════════════════════════════════════════════════════════════════


def _capture_avg(cam, n=1, between_s=0.05, discard_first=True):
    '''Capture n frames and return their float32 average.'''
    if discard_first:
        capture_img(cam)
        time.sleep(between_s)

    frames = []
    for _ in range(n):
        frame = capture_img(cam)
        frames.append(np.array(frame, dtype=np.float32))
        if len(frames) < n:
            time.sleep(between_s)

    return np.mean(np.stack(frames, axis=0), axis=0)


def _save_bmp(arr, path):
    '''Save a float32 array as an 8-bit grayscale BMP.'''
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode='L').save(path)


def _load_slm_bmp(path):
    '''Load a BMP and return a uint8 numpy array of shape (SLM_H, SLM_W).'''
    img = np.array(Image.open(path).convert('L'), dtype=np.uint8)
    if img.shape != (SLM_H, SLM_W):
        raise ValueError(
            f'Mask {path} has shape {img.shape}, expected ({SLM_H}, {SLM_W}).'
        )
    return img


def _blank_slm_pattern():
    '''Return a flat zero-phase SLM pattern (gray=255 with inverted LUT).'''
    return np.full((SLM_H, SLM_W), 255, dtype=np.uint8)


def _init_hardware(exposure_ms=EXPOSURE_MS, gain=GAIN):
    '''Initialise SLM and camera; return (slm, cam).'''
    print('Initialising SLM ...')
    slm = init_slm()
    time.sleep(2.0)

    print('Initialising camera ...')
    cam = init_cam()
    set_config_cam(
        cam,
        trigger_mode='Software',
        exposure_us=exposure_ms * 1000,
        fps=FPS,
        gain=gain,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        offset_x=CAM_OFFSET_X,
        offset_y=CAM_OFFSET_Y,
    )
    cam.StartGrabbing()
    time.sleep(1.0)
    print('Hardware ready.\n')
    return slm, cam


def _cleanup_hardware(slm, cam):
    '''Restore blank SLM and release camera.'''
    slm.showData(_blank_slm_pattern())
    cam.StopGrabbing()
    cam.Close()


def _acquire_sample(slm, cam, sample_dir, n_avg=N_AVG, out_dir=None):
    '''
    Acquire all masks for one sample_dir using already-initialised hardware.
    Returns the output directory path.
    '''
    if out_dir is None:
        out_dir = os.path.join(sample_dir, 'masks_meas')
    os.makedirs(out_dir, exist_ok=True)

    # ── Find mask BMPs ────────────────────────────────────────────────────────
    mask_paths = sorted([
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.startswith('mask_') and f.endswith('_slm.bmp')
    ])
    if not mask_paths:
        raise FileNotFoundError(
            f'No mask_*_slm.bmp files found in {sample_dir}\n'
            f'Run calibrate.py first to generate SLM masks.'
        )
    M = len(mask_paths)
    print(f'Found {M} masks in {sample_dir}')

    log = {
        'sample_dir':    sample_dir,
        'out_dir':       out_dir,
        'n_masks':       M,
        'n_avg':         n_avg,
        'discard_first': DISCARD_FIRST,
        'warmup_delay_s': WARMUP_DELAY_S,
        'start_time':    datetime.datetime.now().isoformat(),
        'masks':         [],
    }

    # ── Dark frame ────────────────────────────────────────────────────────────
    print('Capturing dark frame (blank SLM) ...')
    slm.showData(_blank_slm_pattern())
    time.sleep(WARMUP_DELAY_S)
    dark = _capture_avg(cam, n=n_avg,
                        between_s=BETWEEN_FRAMES_S,
                        discard_first=DISCARD_FIRST)
    dark_path = os.path.join(out_dir, 'dark_meas.bmp')
    _save_bmp(dark, dark_path)
    print(f'  dark mean={dark.mean():.2f}  std={dark.std():.2f}  -> {dark_path}\n')

    # ── Mask loop ─────────────────────────────────────────────────────────────
    print(f'{"Mask":<8} {"File":<30} {"Mean":>8} {"Std":>8} {"Dark-sub mean":>14}')
    print('-' * 72)

    for m, mask_path in enumerate(mask_paths):
        mask_bmp = _load_slm_bmp(mask_path)

        slm.showData(mask_bmp)
        time.sleep(WARMUP_DELAY_S)

        t0    = time.perf_counter()
        frame = _capture_avg(cam, n=n_avg,
                             between_s=BETWEEN_FRAMES_S,
                             discard_first=DISCARD_FIRST)
        elapsed = time.perf_counter() - t0

        # Dark subtraction for stats only; saved image keeps raw values
        frame_ds = np.clip(frame - dark, 0, None)

        out_name = f'mask_{m:02d}_meas.bmp'
        out_path = os.path.join(out_dir, out_name)
        _save_bmp(frame, out_path)

        print(f'mask_{m:02d}  {os.path.basename(mask_path):<30} '
              f'{frame.mean():>8.2f} {frame.std():>8.2f} '
              f'{frame_ds.mean():>14.2f}')

        log['masks'].append({
            'mask_idx':      m,
            'slm_bmp':       mask_path,
            'out_bmp':       out_path,
            'mean_raw':      float(frame.mean()),
            'std_raw':       float(frame.std()),
            'mean_dark_sub': float(frame_ds.mean()),
            'elapsed_s':     float(elapsed),
            'timestamp':     datetime.datetime.now().isoformat(),
        })

    print('-' * 72)

    # ── Save log ──────────────────────────────────────────────────────────────
    log['end_time'] = datetime.datetime.now().isoformat()
    log_path = os.path.join(out_dir, 'acquire_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\nDone. {M} masks captured.')
    print(f'Output   -> {os.path.abspath(out_dir)}')
    print(f'Log      -> {log_path}')
    print(f'  python reconstruct.py --ccd_dir "{out_dir}" --dark "{dark_path}"')
    return out_dir


def acquire(sample_dir, n_avg=N_AVG, exposure_ms=EXPOSURE_MS,
            gain=GAIN, out_dir=None):
    '''Run the full acquisition loop for one calibration sample.'''
    slm, cam = _init_hardware(exposure_ms=exposure_ms, gain=gain)
    try:
        _acquire_sample(slm, cam, sample_dir, n_avg=n_avg, out_dir=out_dir)
    finally:
        _cleanup_hardware(slm, cam)


def acquire_all(calibration_dir, n_avg=N_AVG, exposure_ms=EXPOSURE_MS, gain=GAIN):
    '''
    Discover every sample_* subfolder in calibration_dir and acquire each one
    in sequence using a single hardware initialisation.
    '''
    sample_dirs = sorted([
        os.path.join(calibration_dir, d)
        for d in os.listdir(calibration_dir)
        if d.startswith('sample_') and os.path.isdir(os.path.join(calibration_dir, d))
    ])
    if not sample_dirs:
        raise FileNotFoundError(
            f'No sample_* subdirectories found in {calibration_dir}'
        )
    print(f'Found {len(sample_dirs)} sample folders in {calibration_dir}\n')

    slm, cam = _init_hardware(exposure_ms=exposure_ms, gain=gain)
    try:
        for i, sample_dir in enumerate(sample_dirs):
            print(f'\n══ Sample {i+1}/{len(sample_dirs)}: {os.path.basename(sample_dir)} ══')
            _acquire_sample(slm, cam, sample_dir, n_avg=n_avg)
    finally:
        _cleanup_hardware(slm, cam)

    print(f'\n\nAll {len(sample_dirs)} samples acquired.')


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automated SLM mask display and CCD acquisition'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sample_dir',      default=None,
                      help='Single calibration sample folder containing mask_*_slm.bmp')
    mode.add_argument('--calibration_dir', default=None,
                      help='Calibration folder containing all sample_* subfolders '
                           '(processes every sample automatically)')
    parser.add_argument('--out_dir',      default=None,
                        help='Output folder (single-sample mode only, '
                             'default: <sample_dir>/masks_meas/)')
    parser.add_argument('--n_avg',        default=N_AVG,        type=int,
                        help=f'Frames to average per mask (default: {N_AVG})')
    parser.add_argument('--exposure_ms',  default=EXPOSURE_MS,  type=float,
                        help=f'Camera exposure in ms (default: {EXPOSURE_MS})')
    parser.add_argument('--gain',         default=GAIN,         type=int,
                        help=f'Camera gain (default: {GAIN})')
    args = parser.parse_args()

    if args.calibration_dir or DEFAULT_CALIBRATION_DIR:
        acquire_all(
            calibration_dir = args.calibration_dir or DEFAULT_CALIBRATION_DIR,
            n_avg           = args.n_avg,
            exposure_ms     = args.exposure_ms,
            gain            = args.gain,
        )
    else:
        acquire(
            sample_dir  = args.sample_dir or DEFAULT_SAMPLE_DIR,
            n_avg       = args.n_avg,
            exposure_ms = args.exposure_ms,
            gain        = args.gain,
            out_dir     = args.out_dir,
        )
