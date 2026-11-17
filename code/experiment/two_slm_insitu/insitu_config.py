"""In-situ QPI optimization config.

All hyperparameters live here so you can iterate on the experiment without
touching the loop / policy code.  The values returned by init_params() are
attached to a SimpleNamespace `cfg`, which is passed around the rest of the
codebase as a single dotted-access container (cfg.states_num, cfg.sigma, ...).

Physical layout assumed by the trained model:

    Pluto (SLM1, keys) -- [3 mm] -- Luna (SLM2, encoding + decoder) -- [z2 = 5 or 10 mm] -- camera

Pluto display = u1 (phase key m, 120 x 120 px, centred; flat elsewhere)
Luna  display = (encoding(a) + u2) mod 2*pi   (encoding block + decoder m, 213 x 213 px)

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

    # Frames are sent as 8-bit levels (level k <-> phase k*2pi/256, as the model quantises).
    # *_lut_inverted: True if that SLM's LUT maps 255 -> phase 0 (check on the bench).  *_flip_ud / *_flip_lr: mirror
    # the finished frame so the SLM shows the model's orientation (set from an asymmetric test pattern).
    c.pluto_lut_inverted, c.luna_lut_inverted = False, False
    c.pluto_flip_ud, c.pluto_flip_lr = False, False
    c.luna_flip_ud,  c.luna_flip_lr  = False, False

    # ----------------------------------------------------------------------
    # Phase Keys (on Pluto) and Decoders (on Luna).
    # ----------------------------------------------------------------------
    c.M = 3                                       # number of keys == number of decoders
    c.key_size_px = 120                           # size of each phase key in terms of Pluto pixels
    c.decoder_size_px = 213                       # size of the decoder in terms of Luna pixels
    c.phase_key_center_y = c.slm1_h // 2
    c.phase_key_center_x = c.slm1_w // 2      # center of the key region on Pluto

    # ----------------------------------------------------------------------
    # Target Functions and Input Encoding (on Luna)
    # ----------------------------------------------------------------------
    c.Np = 4                                   # number of harmonics encoded onto the input plane per key
    c.alpha = [0,1,2,3]                        # target harmonics before intensity 
    c.encoding_bin = 28                        # each encoding pixel is this many x this many Luna pixels binned together 
    c.tiling = 9                               # 3x3=9 total copies of the input patch of (sqrtNp x sqrtNp) pixels
    c.a_min = -0.5
    c.a_max = 0.5
    c.a = np.linspace(c.a_min, c.a_max, 100)
    c.a_batch = 8                              # a values shown per state each epoch (spread over c.a, shifted every epoch)
    c.targets_path = './targets.npy'           # [len(c.a), Nf] targets on c.a, min-max'd over that grid (export_targets.py)

    c.decoder_key_center_y = c.slm2_h // 2
    c.decoder_key_center_x = c.slm2_w // 2      # center of the encoding block on Luna (the decoder sits around it)

    # ----------------------------------------------------------------------
    # Camera Sensor
    # ----------------------------------------------------------------------
    c.Nf = 4096                                # number of functions == # of detector regions
    c.pd_size = 6e-6                           # size of each detector region (m)
    c.pd_pitch = 12e-6                         # pitch between each detector region (m)

    # Raw camera frame size.  IMPORTANT: set these to your actual camera
    # frame dimensions on the target machine.
    c.sen_x_num = 1536
    c.sen_y_num = 1536

    # Detector readout (insitu_TDNFA.captures_to_detectors) assumes 6 um camera px at 1x: 64 x 64 detectors of 1 px
    # on a 2 px pitch.  det_axis_yx: camera pixel (row, col) the optical axis lands on (None -> frame centre), set
    # from --calibrate-roi.  cam_flip_*: camera image mirrored vs the model (a 4f relay inverts -> both True).
    c.det_axis_yx = None
    c.cam_flip_ud, c.cam_flip_lr = False, False
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
    c.ckpt_path = './ckpt/z2_5mm_M1_8bit.pth'     # see ckpt/README.txt; the camera distance on the bench must match z2
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
