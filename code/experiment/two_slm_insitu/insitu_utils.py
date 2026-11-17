"""Hardware glue + sensor-image post-processing for the in-situ QPI loop.

Two responsibilities live here:
  1. Open/talk to the Holoeye SLM SDK and the FlyCapture camera.  These
     imports are *lazy* (inside the init functions) so the rest of the code
     can run on a dev machine without the drivers installed (--dry-run).
  2. Drift monitoring: re-display a known reference pattern every N epochs,
     compare the new capture to the baseline, and recover a (dy, dx) shift
     used to compensate for slow optical drift over the run.

All hardware-side functions take the SLM / camera handles as arguments so
the calling code (insitu_TDNFA.py) can swap in dummies for testing.
"""
import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Hardware libs are imported lazily inside init_* so that --dry-run can be run
# on a dev machine without the SDK / camera drivers installed.


# =====================================================================
# SLM / Camera initialization
# =====================================================================
def init_slm():
    """Open one SLM display window.

    The Holoeye SDK auto-selects the next free SLM among connected display
    devices.  Order is decided by the OS display enumeration, which is why
    --swap-slms exists in the main script: you can't always predict which
    physical SLM comes up first.
    """
    from holoeye import slmdisplaysdk
    slm = slmdisplaysdk.SLMInstance()
    if not slm.requiresVersion(5):
        raise RuntimeError("Holoeye SDK v5+ required")
    err = slm.open()
    if err != slmdisplaysdk.ErrorCode.NoError:
        raise RuntimeError(slm.errorString(err))
    return slm


def init_two_slms():
    """Open Pluto then Luna (in OS-display-enumeration order).

    The small sleep between calls gives the SDK time to register the first
    display before opening the second.  If Pluto and Luna come up swapped,
    flip them in the caller (one line in insitu_TDNFA.py via --swap-slms).
    """
    pluto = init_slm()
    time.sleep(0.5)
    luna  = init_slm()
    time.sleep(0.5)
    return pluto, luna


def init_cam(cam_num=0):
    """Open the GigE camera.

    We use pyflycap2 (FlyCapture2 binding).  cam_num picks among multiple
    cameras on the bus; default 0 = first found.
    """
    from pyflycap2.interface import Camera, CameraContext
    cc = CameraContext()
    cc.rescan_bus()
    cam_list = cc.get_gige_cams()
    if not cam_list:
        raise RuntimeError("No GigE camera detected")
    cam = Camera(serial=cam_list[cam_num])
    cam.connect()
    return cam


def init_random(seed):
    """Seed all RNGs so the synthetic calibration set is reproducible."""
    print(f'Random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_save(msg, file):
    """Print to stdout AND append to the log file.  Used everywhere so the
    log.txt at the end of the run captures the full session."""
    print(msg)
    with open(file, 'a') as f:
        f.write(msg + '\n')


# =====================================================================
# Camera capture
# =====================================================================
def caputre_img(cam):
    """Single frame grab.  (Typo preserved from the upstream insitu_hologram
    code so grep across both projects still works.)

    Returns a 2D uint8 array in raw camera resolution.
    """
    cam.read_next_image()
    image_dict = cam.get_current_image()
    image_data = image_dict["buffer"]
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    h, w = image_dict["rows"], image_dict["cols"]
    return image_np.reshape((h, w))


def capture_avg(cam, capture_fn, num=10, delay=0.05):
    """Average num frames.  Reduces shot noise on the captured image.

    The first capture_fn(cam) call in each iteration is THROWN AWAY -- this
    flushes any stale buffer left from a previous SLM pattern.  Without
    this, you sometimes get a one-pattern-late capture.
    """
    imgs = []
    for _ in range(num):
        _ = capture_fn(cam)              # flush
        time.sleep(delay)
        imgs.append(np.asarray(capture_fn(cam), dtype=np.float32))
    return np.mean(np.stack(imgs), axis=0)


# =====================================================================
# Drift monitoring
# =====================================================================
def create_pattern(slm_res, square_size=128):
    """Centered solid square -- used as the drift-monitor reference pattern.

    A solid square produces a localized intensity pattern at the sensor
    that's easy to cross-correlate against.  The size 128 (or whatever you
    pass) is in SLM pixels, NOT sensor pixels.
    """
    pattern = np.zeros(slm_res, dtype=np.float32)
    x0 = (slm_res[0] - square_size) // 2
    y0 = (slm_res[1] - square_size) // 2
    pattern[x0:x0+square_size, y0:y0+square_size] = 1.0
    return pattern


def capture_reference(cam, slm, capture_fn, pattern, num=10, project_delay=0.3):
    """Display the reference pattern and grab a high-averaged baseline.

    Called ONCE at the start of training to capture the "drift baseline".
    Subsequent monitor_drift() calls re-display the same pattern, capture
    a new image, and cross-correlate against this baseline.
    """
    slm.showData(pattern)
    time.sleep(project_delay)
    return capture_avg(cam, capture_fn, num=num)


def crop_center(img, crop_size=200):
    """Pull the central crop_size x crop_size square from a 2D image."""
    h, w = img.shape[:2]
    sx = w // 2 - crop_size // 2
    sy = h // 2 - crop_size // 2
    return img[sy:sy+crop_size, sx:sx+crop_size]


def compute_snr(ref, cur):
    """Mean-normalized SNR (dB) between reference and current capture.

    We scale `cur` to match `ref`'s mean intensity first, so a constant
    illumination drift doesn't get counted as "noise".  Signal power is
    defined as max(ref)^2 (so SNR scales with the brightest feature, not
    the average -- gives a more honest stability measure).
    """
    scale = np.mean(ref) / (np.mean(cur) + 1e-8)
    cur_n = scale * cur
    signal = np.max(ref) ** 2
    noise  = np.mean((ref - cur_n) ** 2) + 1e-8
    return 10 * np.log10(signal / noise)


def compute_shift(ref, cur):
    """Integer (dy, dx) shift of `cur` w.r.t. `ref` via 2D cross-correlation.

    Mean-subtraction before correlation removes the DC component, which
    otherwise dominates and biases the peak toward zero shift.
    """
    ref = ref - np.mean(ref)
    cur = cur - np.mean(cur)
    corr = correlate2d(cur, ref, mode='same', boundary='symm')
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    # zero lag of a 'same' correlation sits at (n - 1) // 2: shape // 2 gave (-1, -1) for an unmoved even-sized crop,
    # which moves the 1-camera-px detector grid onto its gaps (2026-11-05)
    center = (np.array(corr.shape) - 1) // 2
    return np.array(idx) - center


def monitor_drift(cam, slm, capture_fn, reference_img, pattern, save_dir, check_id,
                  num=10, project_delay=0.3):
    """Re-display the reference pattern, compare to baseline, save snapshots.

    Returns (snr_db, shift).  Caller uses `shift` to update the running
    drift correction applied inside captures_to_opred.

    SNR is reported as a stability indicator -- if it drops below ~25 dB,
    something has physically moved or the camera is misaligned, and the
    in-situ run probably needs to stop.
    """
    os.makedirs(save_dir, exist_ok=True)
    slm.showData(pattern)
    time.sleep(project_delay)
    cur = capture_avg(cam, capture_fn, num=num)

    # Compare only the central crop so the SNR/shift isn't dominated by the
    # camera periphery where there might be vignetting or stray light.
    ref_c = crop_center(reference_img, 200)
    cur_c = crop_center(cur, 200)
    snr = compute_snr(ref_c, cur_c)
    shift = compute_shift(ref_c, cur_c)

    np.save(os.path.join(save_dir, f'check_{check_id:03d}.npy'), cur)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(reference_img, cmap='gray'); ax[0].set_title('reference')
    ax[1].imshow(cur,            cmap='gray'); ax[1].set_title(f'check {check_id}')
    fig.suptitle(f'SNR={snr:.2f} dB, shift={shift}')
    fig.savefig(os.path.join(save_dir, f'compare_{check_id:03d}.png'))
    plt.close(fig)

    return snr, shift


# =====================================================================
# Sensor-image post-processing: shift + crop + bin
# =====================================================================
def shift_crop_bin(img_tensor, shift, crop_size, bin_factor):
    """Shift each frame by (dy, dx), center-crop, then avg-pool by bin_factor.

    This is the chain that maps raw camera frames -> sensor-coord sampling
    matching the trained model:

        raw camera (H_cam x W_cam)
            |  reflect-pad, then roll by (-dy, -dx)
            v
        shifted frame (same size)
            |  center crop to (crop_size x crop_size)
            v
        cropped frame
            |  avg pool by bin_factor (sensor px -> simulated px)
            v
        output [..., crop_size/bin_factor, crop_size/bin_factor]

    Args:
        img_tensor: [B, C, H, W]  raw camera frames stacked across states
        shift:      (dy, dx) integer shift to apply (already includes both
                    drift correction and cfg.roi_offset)
        crop_size:  side length of the central crop, BEFORE binning
        bin_factor: integer downsample factor (sensor px -> simulated px)

    Raises ValueError if crop_size doesn't fit in the raw frame, with a
    message pointing at the config knobs that control it.
    """
    import torch.nn.functional as F
    dx, dy = int(shift[0]), int(shift[1])

    # Reflect-pad first so the shift doesn't pull in zeros from outside.
    pad = (abs(dy), abs(dy), abs(dx), abs(dx))
    padded = F.pad(img_tensor, pad, mode='reflect')

    # Pull the window that corresponds to the shifted view.
    sx = abs(dx) - dx
    sy = abs(dy) - dy
    ex = sx + img_tensor.shape[-2]
    ey = sy + img_tensor.shape[-1]
    shifted = padded[..., sx:ex, sy:ey]

    _, _, H, W = shifted.shape
    if crop_size > H or crop_size > W:
        raise ValueError(
            f'shift_crop_bin: crop_size={crop_size} does not fit in frame '
            f'{H}x{W}.  Adjust cfg.sen_x_num / sen_y_num / bin_factor.')

    half = crop_size // 2
    cx, cy = H // 2, W // 2
    cropped = shifted[..., cx-half:cx+half, cy-half:cy+half]

    # Binning: average pool turns raw camera pixels into the simulated
    # pixel grid the trained model expects.
    if bin_factor > 1:
        cropped = F.avg_pool2d(cropped, kernel_size=bin_factor)
    return cropped
