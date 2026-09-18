'''
Time-Multiplexed NFA — config parameters

NFA = Nonlinear Function Approximation. This project is transitioning from
time-multiplexed image CLASSIFICATION (MNIST/CIFAR phase objects, differential
photodiode-pair contrast, softmax cross-entropy) to time-multiplexed parallel
NONLINEAR FUNCTION APPROXIMATION, following Rahman et al., "Massively parallel
and universal approximation of nonlinear functions using diffractive
processors" (eLight 2025) -- see PAPER references below -- with one addition
of our own on top of it: a learned "phase-key" plane, time-multiplexed over
M keys, whose per-key outputs are summed before detection ("wisdom of the
crowd") to improve approximation accuracy for a FIXED set of Nf functions
(the paper instead time-multiplexes wavelength to increase the NUMBER of
functions -- see Sec. 2.2's multi-wavelength design -- which is a different
goal from ours).

Parameters below are tagged inline:
  # PAPER: <value/section>        -- value taken directly from the paper
  # NOTE: not specified in paper  -- our own choice, paper doesn't fix this
  # OBSOLETE (classification-era) -- left in place for now, safe to delete;
                                      superseded by the parameter named in
                                      the comment. Not removed here so you
                                      can track/delete these yourself.

Data paths point to the parent project directory so the same datasets are
shared. Logs are written to conv_decoder/logs/ to keep runs separate.
'''

import sys, os
# Make parent project importable (dataloader, loss, wave_prop, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as _paths   # every filesystem location resolves here

from configobj import ConfigObj
import numpy as np
import datetime


def _build_run_name(tc):
    # Spacings/pdsize are sub-micron to low-micron at the new paper-scale
    # regime (was mm/cm for the old benchtop hardware model) -- formatted in
    # um here so they don't all round to "0mm".
    datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    return (f'{datetime_str}'
            f'-Np{tc.Np}-Nf{tc.Nf}-M{tc.M}-K{tc.num_layers}'
            f'-Spacings{1e6*tc.key_to_enc_spacing:.2f}um-{1e6*tc.slm_first_layer_spacing:.2f}um-{1e6*tc.interlayer_spacing:.2f}um-{1e6*tc.last_layer_ccd_spacing:.2f}um'
            f'-batchsize{tc.batch_size}-lrkey{tc.lr_slm:.0e}-lrlayer{tc.lr_layer:.0e}-pdsize{1e9*tc.photodiode_size:.0f}nm'
            f'-trainA{tc.train_a_samples}-{tc.loss_type}')


def config_to_dict(tc):
    return {k: (v.item() if hasattr(v, 'item') else v)
            for k, v in vars(tc).items()
            if not k.startswith('_') and (isinstance(v, (int, float, str, bool, type(None)))
                                           or hasattr(v, 'item'))}


def _build_log_paths(tc):
    '''Single source of truth for where a run's output goes, derived from tc.run_name.
    '''
    tc.log_dir     = os.path.join(_paths.LOG_DIR, tc.run_name)
    tc.image_dir   = os.path.join(tc.log_dir, 'images')
    tc.model_dir   = os.path.join(tc.log_dir, 'model')
    tc.tfboard_dir = os.path.join(tc.log_dir, 'tfboard')


def recompute_derived(tc):
    '''Recompute all params that depend on base physical values.'''
    tc.slm_bin        = int(tc.slm_dx / tc.sim_dx)
    tc.slm_x_num_sim  = tc.slm_x_num * tc.slm_bin

    tc.photodiode_pixels = int(tc.photodiode_size / tc.sim_dx)

    tc.pd_row_spacing_px = int(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = int(tc.pd_col_spacing / tc.sim_dx)

    tc.layer_bin      = int(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin

    # ---- new NFA (function-approximation) derived params ------------------
    # Encoding-plane geometry: Np input pixels arranged as a
    # sqrt(Np) x sqrt(Np) square patch (paper, Sec. 2.2: "arranged
    # contiguously in a square grid"), sharing the SLM's physical pixel pitch.
    tc.encoding_bin       = int(tc.encoding_dx / tc.sim_dx)
    tc.encoding_side      = int(round(np.sqrt(tc.Np)))
    assert tc.encoding_side ** 2 == tc.Np, \
        'tc.Np must be a perfect square (square encoding patch, per the paper)'
    tc.encoding_x_num_sim = tc.encoding_side * tc.encoding_bin

    tc.N_alpha = tc.Np  # PAPER (Sec. 4.1): "We set Nalpha = Np"

    # Detector array: one intensity detector per target function (no more
    # positive/negative differential pairing -- see photodiode section below).
    assert tc.pd_num_rows * tc.pd_num_cols == tc.Nf, (
        'photodiode array size (pd_num_rows * pd_num_cols) must equal Nf '
        '-- one detector per target function'
    )
    tc.num_photodiodes = tc.pd_num_rows * tc.pd_num_cols

    # total number of learnable phase-key masks. C is OBSOLETE (frozen at 1
    # below) so this currently just reduces to T == M; kept as M*C so this
    # line doesn't need to change the moment you delete tc.C yourself.
    tc.T = int(getattr(tc, 'M', 1) * getattr(tc, 'C', 1))
    # Keep run_name (and everything derived from it) in sync with M/C/spacings/etc --
    # train.py's main() calls recompute_derived() right after applying --set overrides,
    # before training starts, so this must be refreshed here too (init_params() alone
    # only captures the *default* values in the name/paths).
    tc.run_name = _build_run_name(tc)
    _build_log_paths(tc)


def init_params():
    tc = ConfigObj()

    # ------------------------------------------------------------------ #
    #  Physical constants & optical parameters                            #
    # ------------------------------------------------------------------ #
    nm, um, mm, cm = 1e-9, 1e-6, 1e-3, 1e-2

    tc.wavelength = 550 * nm 
    tc.ridx_air   = 1.0

    tc.pixel_pitch = 300 * nm               # PAPER (Sec. 2.2): "lateral pitch
                                            # delta of 300nm (~0.55*lambda)".
                                            # Shared pitch for input pixels,
                                            # output pixels, AND diffractive-
                                            # layer feature width (all == delta
                                            # in the paper) -- see below.

    # ------------------------------------------------------------------ #
    #  Simulation grid                                                    #
    #  sim_dx == pixel_pitch so every *_bin factor below is exactly 1 (no #
    #  oversampling anywhere) -- needed to resolve the paper's sub-micron  #
    #  features at all. N_sim=256 gives a ~7.5x guard band around the     #
    #  largest single-plane aperture (~34px, see layer_size below) to     #
    #  avoid FFT wrap-around, while staying cheap (this is a MUCH smaller #
    #  grid than the old 2000x2000 -- the paper's whole system lives on a #
    #  compact, sub-100um scale, not the cm scale of the old benchtop      #
    #  hardware model this file used to describe).                        #
    # ------------------------------------------------------------------ #
    tc.sim_dx = tc.pixel_pitch
    tc.N_sim  = 256
    tc.asm_pad_factor = 1
    tc.r = 1.25 # scaling factor of the diffractive layer's feature count relative to the paper's 2*Np*Nf guideline

    # ------------------------------------------------------------------ #
    #  Nonlinear function approximation -- targets & input encoding       #
    #  See Rahman et al. eLight (2025) 5:32, Sec. 2.1/2.2/4.1.             #
    # ------------------------------------------------------------------ #
    tc.Np = 9                    # PAPER (Sec. 2.2): number of input-encoding
                                  # pixels; paper keeps this fixed at 9 across
                                  # ALL of its Nf sweeps (100 .. 1e6).
    tc.Nf = 100                  # PAPER (Fig. 2): smallest Nf they test (their
                                  # sweep goes 100 -> 1024 -> 10000 -> ~99856 ->
                                  # 1000000); we start at the smallest value.
    tc.a_min = -0.5               # PAPER (Fig. 1b/2b/2c): input domain of a
    tc.a_max = 0.5                # PAPER: same

    tc.encoding_freq_step = 1     # NOTE: not specified in paper as a separate
                                  # hyperparameter -- implicit in their
                                  # phi_in(p;a) = 2*pi*(p-1)*a, i.e. alpha_p =
                                  # (p-1)*encoding_freq_step with step fixed
                                  # at 1 (Sec. 4.1: "alpha_p = p - 1"). Exposed
                                  # here in case we ever want to change it.

    tc.func_seed = 0              # random seed for generating Nf target functinons

    # ------------------------------------------------------------------ #
    #  Phase-key plane  (hardware device -- same SLM as before, new role) #
    #  NOTE: the M-key / "wisdom of the crowd" time-multiplexing idea is  #
    #  OUR OWN addition on top of the paper -- the paper has no phase-key #
    #  plane or per-key ensembling concept, so M's role/value below is    #
    #  not paper-derived.                                                 #
    # ------------------------------------------------------------------ #
    tc.slm_dx      = tc.pixel_pitch  
    tc.slm_x_num   = int(np.ceil(np.sqrt(tc.r * 2 * tc.Np * tc.Nf / 2)))
    tc.slm_bin     = int(tc.slm_dx / tc.sim_dx)
    tc.slm_x_num_sim = tc.slm_x_num * tc.slm_bin

    tc.slm_hw_x      = 1920  # NOTE: real-device pixel count, non-binding at
    tc.slm_hw_y      = 1080  # this scale (bigger than N_sim -- gets clipped
                              # to N_sim in model.py, i.e. no additional
                              # aperture restriction beyond the sim window).
    tc.slm_bit_depth = 8

    tc.M               = 1  # NOTE: not defined in paper -- number of learned
                             # phase keys (time-multiplexed conditioning masks).
                             # Each key gives one independent estimate of f(a);
                             # summing/averaging across keys before detection is
                             # the "wisdom of the crowd" accuracy-improvement
                             # mechanism this project adds.

    tc.mask_init_method = 'normal'
    tc.mask_init_std   = 0.5

    # ------------------------------------------------------------------ #
    #  Function-input encoding plane (deterministic, NOT learned)         #
    #  Replaces the old "Object (MNIST phase images)" plane below --      #
    #  instead of an image, this plane carries phi_in(p;a) = 2*pi *       #
    #  encoding_freq_step * (p-1) * a for p = 1..Np (PAPER Sec. 2.2/4.1).  #
    # ------------------------------------------------------------------ #
    tc.encoding_dx        = tc.slm_dx  # == tc.pixel_pitch, PAPER (Sec. 2.2)
    tc.encoding_bin       = int(tc.encoding_dx / tc.sim_dx)
    tc.encoding_side      = int(round(np.sqrt(tc.Np)))
    tc.encoding_x_num_sim = tc.encoding_side * tc.encoding_bin

    # ------------------------------------------------------------------ #
    #  Diffractive Layers                                                 #
    #  Each layer is shared across all M time-multiplexed phase keys      #
    #  (unchanged role/mechanism from before).                             #
    #  PAPER (Sec. 2.2): K=2 surfaces (their main/default design; K=4 is a #
    #  deeper alternative shown to further reduce error, Fig. 3/4), with   #
    #  N ~= r * 2*Np*Nf trainable features total, distributed evenly    #
    #  over the K surfaces -- this sets layer_size (features per side of   #
    #  a square layer) below. This is a GUIDELINE, not a strict requirement#
    #  (paper's own wording) -- computed here as a starting default, but   #
    #  NOT re-derived in recompute_derived(), so you can freely --set      #
    #  layer_size to something else later without this formula clobbering #
    #  it back.                                                            #
    # ------------------------------------------------------------------ #
    tc.num_layers    = 2   # PAPER (Sec. 2.2): K
    tc.layer_dx      = tc.pixel_pitch  # PAPER: diffractive feature width == delta
    tc.layer_size    = int(np.ceil(np.sqrt(tc.r * 2 * tc.Np * tc.Nf / tc.num_layers)))
    tc.layer_bin      = int(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin

    # Axial spacing between EVERY consecutive pair of planes -- PAPER (Sec.
    # 2.2) uses one uniform value for all such gaps (input/output pixel
    # planes and diffractive surfaces alike): z = W * sqrt((2*delta/lambda)^2 - 1),
    # where W = layer_size * layer_dx is one diffractive surface's total
    # width. We apply this same z to the phase-key -> encoding-plane gap
    # too (key_to_enc_spacing) even though that gap has no paper analogue
    # (NOTE, our own choice -- picked the same value for consistency, not
    # derived from anything paper-specific).
    _W = tc.layer_size * tc.layer_dx
    _z = _W * np.sqrt((2 * tc.pixel_pitch / tc.wavelength) ** 2 - 1)
    tc.key_to_enc_spacing      = _z   # phase-key -> encoding-plane spacing (NOTE, no paper analogue)
    tc.interlayer_spacing      = _z   # PAPER
    tc.slm_first_layer_spacing = _z   # PAPER (encoding-plane -> first-layer spacing)
    tc.last_layer_ccd_spacing  = _z   # PAPER (last-layer -> detector spacing)

    # OBSOLETE -- was the no-layers-bypass propagation distance
    # (propagator_no_layers / diffraction_efficiency()), both removed from
    # model.py (num_layers > 0 is now required; still read as a fallback
    # default by wave_prop.FreeSpaceProp when called with no explicit z, but
    # nothing in this codebase calls it that way anymore).
    tc.z_slm_ccd = _z

    # ------------------------------------------------------------------ #
    #  Photodiode array  (pd_num_rows x pd_num_cols detectors)             #
    #  ONE intensity detector per target function -- detector (r,c) reads #
    #  out f_hat_k(a) for k = r*pd_num_cols + c, via direct min-max        #
    #  normalization (PAPER Eq. 9), no positive/negative differential      #
    #  pairing anymore. pd_num_rows * pd_num_cols must equal tc.Nf (asserted  #
    #  in recompute_derived).                                              #
    # ------------------------------------------------------------------ #
    tc.photodiode_size   = tc.pixel_pitch      # PAPER (Sec. 2.2): detector width == delta
    tc.photodiode_pixels = int(tc.photodiode_size / tc.sim_dx)
    tc.pd_num_rows = 10   # 10 x 10 == Nf (100)
    tc.pd_num_cols = 10
    tc.num_photodiodes = tc.pd_num_rows * tc.pd_num_cols

    # NOTE: deviates from PAPER here. The paper specifies an inter-pixel gap of
    # ~0.5*lambda (Sec. 2.2), i.e. center-to-center spacing = photodiode_size +
    # 0.5*wavelength -- but at this project's sim_dx == pixel_pitch (300nm),
    # that spacing (575nm = 1.917 sim-grid pixels) gets truncated by
    # int(pd_row_spacing / sim_dx) down to 1 pixel -- i.e. the SAME as
    # photodiode_pixels, so the detectors end up simulated as touching with NO
    # gap at all (crosstalk risk), silently defeating the paper's own
    # crosstalk-suppression intent. Using spacing = 2*photodiode_size instead
    # guarantees a whole extra sim-grid pixel of real gap between detectors
    # (spacing_px=2, photodiode_pixels=1) regardless of sim_dx.
    tc.pd_row_spacing = 4 * tc.photodiode_size  # center-to-center spacing, row direction
    tc.pd_col_spacing = 4 * tc.photodiode_size  # center-to-center spacing, column direction
    tc.pd_row_spacing_px = int(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = int(tc.pd_col_spacing / tc.sim_dx)

    # Offset of the whole array's center, in sim-grid pixels relative to the optical
    # axis (N_sim/2). Default 0 centers the array on the axis. Nonzero shifts the
    # whole array (e.g. to model misalignment).
    tc.detector_offset_y = 0
    tc.detector_offset_x = 0

    # ------------------------------------------------------------------ #
    #  Nonlinear function approximation -- sampling of `a`                #
    #  Training uses a FIXED POOL of randomly-drawn a values (Monte Carlo  #
    #  over the continuous domain, resampled/reshuffled across epochs --  #
    #  chosen instead of a fresh-every-batch infinite stream so the        #
    #  existing epoch/checkpoint/logging infra keeps working unchanged).   #
    #  Validation/test instead use a fixed, dense, EVENLY-SPACED grid, so  #
    #  the reported per-function error approximates the paper's continuous #
    #  RMSE integral (Eq. 16) with low variance, and target-vs-           #
    #  approximation curves (like Fig. 2b/2c) come out smooth.             #
    #  NOTE: none of these three sizes are specified by the paper -- their #
    #  training loss (Eq. 11) is a closed-form PSF/Fourier-coefficient fit #
    #  that never samples discrete a values at all (see design discussion);#
    #  we can't reuse that shortcut once the phase-key plane + M-key      #
    #  intensity-domain summing are added on top, so these are our own    #
    #  choices, carried over from the old train_samples/val_samples scale. #
    # ------------------------------------------------------------------ #
    tc.train_a_samples = 10000  
    tc.val_a_grid_size  = 1000   
    tc.test_a_grid_size = 1000  

    # ------------------------------------------------------------------ #
    #  Training hyper-parameters                                          #
    # ------------------------------------------------------------------ #
    tc.batch_size       = 64
    tc.test_batch_size  = 4
    tc.max_epoch        = 150
    tc.seed             = 59

    tc.num_workers = 0

    tc.lr_slm     = 1e-2   # learning rate for the phase-key plane (was: SLM mask)
    tc.lr_layer   = 1e-2   # learning rate for the diffractive layers (unchanged role)

    tc.slm_warmup_epochs = 0

    # End-to-end measurement-noise injection (SLM + decoder co-adapt to the sim->real gap).
    tc.meas_noise_std = 0

    tc.freeze_slm = False  # freezes the phase-key plane (was: SLM mask)
                                          #
    # ------------------------------------------------------------------ #
    tc.loss_type = 'mse'      # NOTE: not defined in paper -- superseded loss_mode

    # ------------------------------------------------------------------ #
    #  Logging & checkpoints        #
    # ------------------------------------------------------------------ #
    tc.checkpoint_save  = 10
    tc.checkpoint_print = 1

    tc.run_name = _build_run_name(tc)
    _build_log_paths(tc)

    tc.ckpt_to_load = None  # path to a checkpoint to load (None = start from scratch)

    # When loading ckpt_to_load, load ONLY the model weights (skip epoch/optimizer/scheduler) so
    # training starts a fresh fine-tune from epoch 0 with a new LR schedule. False = resume.
    tc.load_weights_only = False

    return tc
