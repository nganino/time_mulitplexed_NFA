"""Checkpoint extractor + synthetic calibration objects.

Why no MNIST / dataset dependency?
----------------------------------
For *calibration*, we don't need natural images.  We need phase patterns
that span a useful range of spatial frequencies and feature types so the
in-situ optimizer can detect aberrations in the optics.  Gratings test
high-frequency response; uniform disks test low-frequency; checkerboards
test edges; smooth random patterns test mid-band response.  Synthesizing
these in code keeps the folder self-contained -- no MNIST IDX files,
no external paths -- and the seed-driven random patterns are reproducible
across runs.

Why extract masks from the checkpoint manually?
-----------------------------------------------
We could `from D2NN_module_psrVacci import DiffractiveNetworkSLM`, build
the full model, load_state_dict, and read `.D2NN1[0].mask_phase`.  But that
pulls in the whole simulation propagator (CUDA kernels, kornia, etc.) and
its config namespace, which is overkill -- we only need the two phase
tensors.  Instead we read the state_dict directly and grep for the keys.
"""
import os
import numpy as np
import torch


# =====================================================================
# Checkpoint extraction
# =====================================================================
def load_trained_masks(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd, tc = ck['model'], ck['config']

    keys = sd['slm_phases'][:,0] # [M,414,414] Pluto
    keys = 2*np.pi*torch.sigmoid(keys).float()

    decoders = torch.remainder(sd['decoder_phases'][:,0], 2*np.pi).float() # [M,736,736] Luna

    assert keys.shape[0] == decoders.shape[0] == tc['M']
    return keys, decoders, tc


# =====================================================================
# Synthetic calibration phase patterns
# =====================================================================
# Each builder returns an (H, W) float32 array with values in [0, amp].
# `amp` is set to pi by default to match the trained model's input_range_max.

def _grating(H, W, period_px, angle_deg, amp=np.pi):
    """Linear sinusoidal grating.  Tests the optics' response at a specific
    spatial frequency and orientation."""
    yy, xx = np.meshgrid(np.arange(H) - H/2, np.arange(W) - W/2, indexing='ij')
    th = np.deg2rad(angle_deg)
    coord = xx * np.cos(th) + yy * np.sin(th)
    g = 0.5 * (1 + np.sin(2 * np.pi * coord / period_px))
    return (g * amp).astype(np.float32)


def _disk(H, W, radius_frac, amp=np.pi):
    """Uniform-phase disk centered in the frame.  Localized bright feature;
    useful for the ROI centroid measurement in --calibrate-roi."""
    yy, xx = np.meshgrid(np.arange(H) - H/2, np.arange(W) - W/2, indexing='ij')
    r = np.sqrt(xx**2 + yy**2)
    mask = (r <= radius_frac * min(H, W) / 2).astype(np.float32)
    return (mask * amp).astype(np.float32)


def _annulus(H, W, r_in_frac, r_out_frac, amp=np.pi):
    """Phase ring between two radii.  Tests rotational symmetry."""
    yy, xx = np.meshgrid(np.arange(H) - H/2, np.arange(W) - W/2, indexing='ij')
    r = np.sqrt(xx**2 + yy**2) / (min(H, W) / 2)
    mask = ((r >= r_in_frac) & (r <= r_out_frac)).astype(np.float32)
    return (mask * amp).astype(np.float32)


def _checkerboard(H, W, cell_px, amp=np.pi):
    """High-contrast edges at a uniform grid.  Tests edge response."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    c = ((xx // cell_px) + (yy // cell_px)) % 2
    return (c * amp).astype(np.float32)


def _smooth_random(H, W, seed, lowpass_radius=12, amp=np.pi):
    """Band-limited random phase: Gaussian noise low-passed in Fourier.

    Produces smooth "blob"-like phase variation -- closer in feature content
    to real samples than the geometric patterns above.  Seed-driven so the
    pattern is reproducible across runs (don't change the seed if you want
    reward numbers to be comparable run-to-run).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((H, W))
    F = np.fft.fftshift(np.fft.fft2(noise))
    yy, xx = np.meshgrid(np.arange(H) - H/2, np.arange(W) - W/2, indexing='ij')
    mask = (np.sqrt(xx**2 + yy**2) <= lowpass_radius).astype(np.float32)
    smooth = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    return (smooth * amp).astype(np.float32)


def build_calibration_set(obj_num, n_calib=10):
    """Return n_calib (name, gt_phase) tuples for cycling during training.

    obj_num: side length of the object window at sensor sampling.  GT
             tensors are built at this resolution -- they're what reward
             functions compare against.

    The default ten-object set is curated to span:
      - 4 gratings (varied period + orientation)
      - 2 disks    (varied size)
      - 1 annulus  (asymmetric radial test)
      - 1 checkerboard
      - 2 smooth random fields

    Order is fixed (deterministic); the in-situ loop cycles through them
    round-robin so each object gets visited ~iterations/n_calib times.
    """
    H = W = int(obj_num)
    amp = np.pi

    patterns = [
        ('grating_h_p20',    _grating(H, W, 20, 0,   amp)),
        ('grating_v_p20',    _grating(H, W, 20, 90,  amp)),
        ('grating_diag_p16', _grating(H, W, 16, 45,  amp)),
        ('grating_v_p8',     _grating(H, W,  8, 90,  amp)),
        ('disk_r05',         _disk(H, W, 0.5, amp)),
        ('disk_r03',         _disk(H, W, 0.3, amp)),
        ('annulus_03_06',    _annulus(H, W, 0.3, 0.6, amp)),
        ('checker_c10',      _checkerboard(H, W, 10, amp)),
        ('smooth_seed0',     _smooth_random(H, W, seed=0, amp=amp)),
        ('smooth_seed1',     _smooth_random(H, W, seed=1, amp=amp)),
    ]
    return patterns[:n_calib]


# =====================================================================
# Sensor-coord -> SLM-coord resampling
# =====================================================================
def pad_object_to_slm(obj_phase, slm_x_num, slm_y_num, slm_dx, sensor_dx,
                      magnification=1.0):
    """Resample a sensor-coord phase pattern onto the Pluto SLM grid.

    The trained model treats SLM1 (Pluto) as co-located with the object
    plane: at display time, Pluto shows `(obj_on_slm + u1) mod 2*pi`.  This
    helper takes a GT phase defined at sensor sampling and:
      1. Inverts the optical magnification to get the object size in SLM
         pixels (a 200-px sensor crop at 6x binning corresponds to a
         1200-sensor-px region, which at the Pluto pitch is a much smaller
         SLM-pixel footprint).
      2. Nearest-neighbor resamples to that SLM size.
      3. Center-pads with zero phase to the full SLM frame.

    The result is a (slm_x_num, slm_y_num) phase tensor that gets ADDED to
    u1 before display.  Zero-phase padding outside the object region means
    those SLM pixels only display u1 (the encoder mask) without any object
    contribution.

    Args:
        obj_phase: (H, W) numpy array, sensor-coord phase in radians
        slm_x_num, slm_y_num: Pluto SLM frame size (1080, 1920)
        slm_dx:    Pluto pixel pitch in meters
        sensor_dx: effective sensor pixel pitch in meters (raw_dx * bin_factor)
        magnification: optical mag from SLM1 to sensor.  Default 1.0 means
            "just resample by the pitch ratio".  If your trained model has
            a non-unity magnification, set it here so the displayed object
            comes out the right size on the sensor.

    Returns:
        torch.Tensor of shape (slm_x_num, slm_y_num).
    """
    import torch.nn.functional as F
    H, W = obj_phase.shape
    obj_t = torch.from_numpy(obj_phase).unsqueeze(0).unsqueeze(0).float()

    # Map sensor pixels -> SLM pixels via the pitch ratio, then invert
    # magnification (sensor object is `magnification` times the SLM object).
    scale = (sensor_dx / slm_dx) / magnification
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))
    obj_resized = F.interpolate(obj_t, size=(new_H, new_W), mode='nearest')

    # Center-pad with zero phase so the object lives in the middle of the
    # SLM and the surrounding pixels only display u1.
    pad_top    = (slm_x_num - new_H) // 2
    pad_bottom = slm_x_num - new_H - pad_top
    pad_left   = (slm_y_num - new_W) // 2
    pad_right  = slm_y_num - new_W - pad_left
    obj_on_slm = F.pad(obj_resized, (pad_left, pad_right, pad_top, pad_bottom),
                       value=0)
    return obj_on_slm.squeeze(0).squeeze(0)
