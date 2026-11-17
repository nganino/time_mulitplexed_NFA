"""In-situ QPI optimization config.

All hyperparameters live here so you can iterate on the experiment without
touching the loop / policy code.  The values returned by init_params() are
attached to a SimpleNamespace `cfg`, which is passed around the rest of the
codebase as a single dotted-access container (cfg.states_num, cfg.sigma, ...).

Physical layout assumed by the trained model (bench, measured 2026-11-10, code/RETRAINING_CHANGES.md):

    Pluto (SLM1, keys) -- 1:1 relay + [z1 = 29.7 mm] -- Luna (SLM2, encoding + decoder) -- [z2 = 168.8 mm] -- Basler

Pluto display = u1 (phase key m, 414 x 414 px, centred; flat elsewhere)
Luna  display = (encoding(a) + u2) mod 2*pi   (decoder m, 736 x 736 px, with the 660 x 660 px encoding block in its centre)

u1, u2 are the *trainable* phase masks (the "policy mean").  The in-situ loop
samples noisy variants of (u1, u2), displays them, captures, scores, and nudges
(u1, u2) toward variants that scored higher.

CLI flags in insitu_TDNFA.py can override most of these fields.
"""
import datetime
import numpy as np
from types import SimpleNamespace


def init_params():
    c = SimpleNamespace()

    # ----------------------------------------------------------------------
    # SLM geometry -- MUST match what the trained checkpoint was built for.
    # Pluto and Luna are both 1080x1920 native, with different pixel pitches.
    # Changing these here without retraining will misalign the optics.
    # ----------------------------------------------------------------------
    c.slm1_h, c.slm1_w = 1080, 1920               # Pluto pixel count (rows, cols)
    c.slm1_mask_dx = 8e-6                         # Pluto pixel pitch (m)
    c.slm2_h, c.slm2_w = 1080, 1920               # Luna pixel count (rows, cols)
    c.slm2_mask_dx = 4.5e-6                       # Luna pixel pitch (m)

    # Display-side corrections, RETRAINING_CHANGES.md section 3 (applied in insitu_TDNFA.pluto_frame / luna_frame, in this
    # order: key shift (model frame) -> conjugate -> phase table -> flips).
    # *_lut_inverted: send -phi (the measured "conjugate phase" step; True on both SLMs).
    # *_flip_ud / *_flip_lr: mirror the finished frame.  PLUTO is rotated 180 deg (= both flips) because the relay inverts.
    c.pluto_lut_inverted, c.luna_lut_inverted = True, True
    c.pluto_flip_ud, c.pluto_flip_lr = True, True
    c.luna_flip_ud,  c.luna_flip_lr  = False, False

    # Measured phase vs grey level (RETRAINING_CHANGES.md section 4, fit phi(k) = 2 pi / 256 (s k + q k^2 / 256)).  Each
    # wanted phase is sent as the level whose measured phase is nearest; phases the SLM cannot reach (about 312-360 deg)
    # land on the nearer end.  The model was trained with the linear table; use_phase_table = False sends
    # level k <-> 2 pi k / 256 instead.  To use the measured points (verify_20261110-195101/lut_used.json) instead of the
    # fit, put the 256 phases (rad, level 0..255) in *_phase_table.
    c.use_phase_table = True
    k = np.arange(256)
    c.pluto_phase_table = 2 * np.pi / 256 * (0.751 * k + 0.129 * k ** 2 / 256)
    c.luna_phase_table = 2 * np.pi / 256 * (1.445 * k - 0.576 * k ** 2 / 256)
    # level to send for each of 4096 phase bins (phase 2 pi j / 4096): the level with the nearest measured phase
    j = 2 * np.pi * np.arange(4096)[:, None] / 4096
    c.pluto_lut = np.abs(np.angle(np.exp(1j * (j - c.pluto_phase_table[None])))).argmin(1).astype(np.uint8)
    c.luna_lut = np.abs(np.angle(np.exp(1j * (j - c.luna_phase_table[None])))).argmin(1).astype(np.uint8)

    # ----------------------------------------------------------------------
    # Phase Keys (on Pluto) and Decoders (on Luna).
    # ----------------------------------------------------------------------
    c.M = 1                                       # number of keys == number of decoders (not used: M comes from the ckpt)
    c.key_size_px = 414                           # size of each phase key in terms of Pluto pixels
    c.decoder_size_px = 736                       # size of the decoder in terms of Luna pixels
    # Centre of the key on Pluto, model frame (before the 180 deg flip): the optical axis (px corner 539/540, 959/960)
    # plus the measured registration shift (-12, +30) PLUTO px (RETRAINING_CHANGES.md section 3 step 1).  The shift is
    # read as (rows, cols); if the bench record means (x, y), use (+30, -12).
    c.pluto_shift_yx = (-12, +30)
    c.phase_key_center_y = c.slm1_h // 2 + c.pluto_shift_yx[0]
    c.phase_key_center_x = c.slm1_w // 2 + c.pluto_shift_yx[1]

    # ----------------------------------------------------------------------
    # Target Functions and Input Encoding (on Luna)
    # ----------------------------------------------------------------------
    c.Np = 4                                   # number of harmonics encoded onto the input plane per key
    c.alpha = [0,1,2,3]                        # target harmonics before intensity 
    c.encoding_bin = 110                       # each encoding pixel is this many x this many Luna pixels binned together 
    c.tiling = 9                               # 3x3=9 total copies of the input patch of (sqrtNp x sqrtNp) pixels
                                               # -> 660 x 660 px block, rows / cols 38..697 of the 736 px decoder
    c.a_min = -0.5
    c.a_max = 0.5
    c.a = np.linspace(c.a_min, c.a_max, 100)
    c.a_batch = 8                              # a values shown per state each epoch (spread over c.a, shifted every epoch)
    c.targets_path = './targets.npy'           # [len(c.a), Nf] targets on c.a, min-max'd over that grid (export_targets.py)

    c.decoder_key_center_y = c.slm2_h // 2
    c.decoder_key_center_x = c.slm2_w // 2      # center of the encoding block on Luna (the decoder sits around it)
                                                # (= the optical axis, the px corner 539/540, 959/960)

    # ----------------------------------------------------------------------
    # Camera Sensor
    # ----------------------------------------------------------------------
    c.Nf = 4096                                # number of functions == # of detector regions
    c.cam_dx = 5.86e-6                         # Basler acA1920-40gm pixel (no relay to the camera: 1x)
    c.pd_size = 4 * c.cam_dx                   # size of each detector region (m): 4 x 4 camera px
    c.pd_pitch = 8 * c.cam_dx                  # pitch between each detector region (m): 8 camera px

    # Raw camera frame size (rows, cols) of the Basler acA1920-40gm, full sensor.
    c.sen_x_num = 1200
    c.sen_y_num = 1920
    c.cam_exposure_us = 5000                   # exposure (us); auto exposure / gain are switched off in init_cam.  Set it
                                               # from the DETECTOR ARRAY, not the whole frame: with the trained masks the
                                               # detectors see only 0.02-0.13 of the open beam, while a bright rim just
                                               # outside the array reaches ~7x (sim_check_M1).  Let the rim saturate;
                                               # aim the brightest detector px at ~80 % of full scale (calibrate_roi prints it)
    c.cam_pixel_format = 'Mono12'              # 12-bit (0-4095): the detector signal is a small range on a background
                                               # ('Mono8' works but leaves only tens of counts of signal range)

    # Detector readout (insitu_TDNFA.captures_to_detectors): 64 x 64 detectors of 4 x 4 camera px on an 8 px pitch,
    # symmetric about the optical axis.  det_axis_yx = (A_row, A_col): the axis is the corner between camera px A - 1 and
    # A; detector r covers camera rows A - 254 + 8 r .. A - 251 + 8 r (model frame).  Start: the measured raw axis
    # (500.1, 984.5) -> nearest corners (501, 985); --calibrate-roi refines it.  cam_flip_*: the captures are rotated
    # 180 deg vs the model (RETRAINING_CHANGES.md section 3) -> both True.  The crop is taken on the raw frame around the
    # axis, then flipped, so det_axis_yx is always in RAW camera coordinates.
    c.det_axis_yx = (497, 985)                 # 2026-11-11 with the collimator on (was (501, 985)); --calibrate-roi refines it
    c.cam_flip_ud, c.cam_flip_lr = True, True
    c.roi_search_px = 4                        # calibrate_roi searches det_axis_yx within +-this many camera px
    c.det_debug_every = 10                     # save the detector-region crop with the detector grid overlaid every
                                               # this many epochs (with the eval capture; 0 = never)



    # ----------------------------------------------------------------------
    # Policy / PPO hyperparameters.
    # states_num: number of noisy mask pairs sampled per epoch.  Each one
    #   costs an SLM write + camera capture, so this dominates wall time.
    # sigma: per-pixel Gaussian noise std (radians) used to *probe* the
    #   reward landscape.  Too small -> no signal.  Too big -> drowns out
    #   the gradient with variance.  0.04 rad (~2 deg) is a good default.
    # K: PPO inner update steps per batch of samples.  Reusing each batch
    #   K times multiplies learning per camera capture, but K too high
    #   makes the importance ratio explode.
    # ----------------------------------------------------------------------
    c.states_num = 64
    c.div_step = 32                               # memory-saving chunk size
    c.sigma = 0.04
    c.sigma_anneal = False                        # True -> sigma decays over epochs
    c.sigma_final = 0.02
    c.anneal_epochs = 100
    c.lr = 1e-2
    c.K = 4
    c.ppo_eps = 0.2                               # PPO clip range

    # ----------------------------------------------------------------------
    # Mask binning.  If you want the policy to live on a coarser grid than
    # the SLM (smaller search space, faster), set this > 1.  1 = policy
    # pixel == SLM pixel, which matches how the model was trained.
    # ----------------------------------------------------------------------
    c.mask_bin = 1

    # ----------------------------------------------------------------------
    # Calibration set: how many synthetic phase objects we cycle through.
    # Each epoch picks one (round-robin).  Generated in insitu_calib.py.
    # ----------------------------------------------------------------------
    c.n_calib = 10

    # ----------------------------------------------------------------------
    # Freeze flags (overridden by CLI --freeze-pluto / --freeze-luna).
    # Freezing a mask sets its exploration sigma to 0 and removes it from
    # the optimizer -- the other mask becomes the only thing being tuned.
    # ----------------------------------------------------------------------
    c.freeze_pluto = False
    c.freeze_luna  = False

    # ----------------------------------------------------------------------
    # Loop control / timing.
    # slm_settle: pause after slm.showData before triggering the camera.
    #   Pluto/Luna need a few ms to actually paint the new pattern; the
    #   correct value depends on the SLM and the trigger mode.  0.10 s is
    #   conservative.  Tune down to ~0.04 s once the loop runs stably.
    # capture_avg_n: frames averaged per state.  Each grab is ~30 ms.
    #   Higher reduces shot noise but lengthens every epoch.
    # ----------------------------------------------------------------------
    c.iterations = 500
    c.slm_settle = 0.10
    c.capture_avg_n = 2
    c.capture_avg_delay = 0.02
    c.drift_check_every = 10
    c.checkpoint_every = 10
    c.val_capture_avg_n = 5

    # ----------------------------------------------------------------------
    # Reward: -RMSE (mean over functions of the per-function RMSE, as test.py reports) between the detector
    # readings and the targets, after ONE least-squares scale + offset per state (the training readout; a min-max
    # over only a_batch values would be noisy).  The checkpoint eval logs the test.py min-max RMSE on all of c.a.
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Paths.
    # ckpt_path: trained .pth to initialize u1, u2 from.
    # warp_path: TPS warp parameters from warp_init.py.  Currently unused in
    #            the loop (GT lives at sensor sampling already), kept here
    #            so future code can apply the warp to GT if needed.
    # run_dir:   where logs/checkpoints/eval images get written.
    # ----------------------------------------------------------------------
    c.ckpt_path = './ckpt_M2/bench_M2_bin1_ep4.pth'   # collimated bench, z1 35.87 / z2 175.35 mm (also ckpt_M3/); ckpt_M1 is OBSOLETE
    c.warp_path = './warp_params.npy'
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    c.run_dir = f'./logs/insitu_{stamp}'

    # ----------------------------------------------------------------------
    # Misc.
    # seed: makes the synthetic calibration set deterministic.  Don't change
    #       between runs if you want reward numbers to be comparable.
    # ----------------------------------------------------------------------
    c.seed = 126
    c.cam_num = 0

    return c
