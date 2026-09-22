'''
Time-Multiplexed NFA — config parameters
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
    tc.slm_bin        = round(tc.slm_dx / tc.sim_dx)
    tc.slm_x_num_sim  = tc.slm_x_num * tc.slm_bin

    tc.photodiode_pixels = round(tc.photodiode_size / tc.sim_dx)

    tc.pd_row_spacing_px = round(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = round(tc.pd_col_spacing / tc.sim_dx)

    tc.layer_bin      = round(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin

    # ---- new NFA (function-approximation) derived params ------------------
    # Encoding-plane geometry: Np input pixels arranged as a
    # sqrt(Np) x sqrt(Np) square patch (paper, Sec. 2.2: "arranged
    # contiguously in a square grid"), sharing the SLM's physical pixel pitch.
    # Unlike layer_size/slm_x_num/the spacings, encoding_dx IS recomputed here
    # (not just at init_params()) so `--set encoding_patch_scale=...` alone is
    # enough -- no separate encoding_dx/encoding_bin override needed.
    tc.encoding_dx        = tc.slm_dx * tc.encoding_patch_scale
    tc.encoding_bin       = round(tc.encoding_dx / tc.sim_dx)
    tc.encoding_side      = int(round(np.sqrt(tc.Np)))
    assert tc.encoding_side ** 2 == tc.Np, \
        'tc.Np must be a perfect square (square encoding patch, per the paper)'
    tc.encoding_x_num_sim = tc.encoding_bin * (
        tc.encoding_side + (tc.encoding_side - 1) * tc.encoding_gap_blocks)

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
    tc.r = 1.25 # scaling factor of the diffractive layer's feature count

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

    tc.encoding_opaque_background = False  

    tc.encoding_patch_scale = 1   # Sim-grid pixels spanned by EACH
                                  # encoding phase value (alpha_p)

    tc.encoding_gap_blocks = 0    # Number of BLANK bin-sized blocks
                                  # inserted between adjacent encoding pixels,
                                  # in both row and column directions -- each
                                  # blank block is the same physical size as
                                  # one pixel's own encoding_bin x
                                  # encoding_bin block, so at patch_scale=2 a
                                  # gap of 1 is itself a 2x2 blank block
                                  # separating 2x2 pixel blocks.

    tc.func_seed = 0              # random seed for generating Nf target functinons

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
    tc.M             = 1   # NOTE: not defined in paper -- number of learned
                           # phase keys (time-multiplexed conditioning masks).

    tc.num_layers    = 2   # PAPER (Sec. 2.2): K
    tc.scale_layer_with_M = False  # N = r * 2*Np*Nf*M  (True) or N = r * 2*Np*Nf (False)
    tc.N_trainable_features = int(np.ceil(
        tc.r * 2 * tc.Np * tc.Nf * (tc.M if tc.scale_layer_with_M else 1))) #total  trainable features across all layers 
    tc.layer_dx      = tc.pixel_pitch  # PAPER: diffractive feature width == delta
    tc.layer_size    = int(np.ceil(np.sqrt(tc.N_trainable_features / tc.num_layers)))
    tc.layer_bin      = round(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin

    # ------------------------------------------------------------------ #
    #  Phase-key plane  (hardware device -- same SLM as before, new role) #
    #  NOTE: the M-key / "wisdom of the crowd" time-multiplexing idea is  #
    #  OUR OWN addition on top of the paper -- the paper has no phase-key #
    #  plane or per-key ensembling concept, so M's role/value below is    #
    #  not paper-derived.                                                 #
    #                                                                      #
    #  slm_x_num == layer_size (own choice, 2026-09-18): each phase key is #
    #  sized to match exactly ONE diffractive layer. Since layer_size now  #
    #  also scales with M (2026-09-18 correction, above), total learnable  #
    #  phases across the whole system work out to                         #
    #      K * layer_size^2   (the D2NN)          ~= r*2*Np*Nf*M           #
    #    + M * layer_size^2   (the M phase keys)  ~= r*2*Np*Nf*M^2/K       #
    #    = r*2*Np*Nf*M * (1 + M/K)  -- note this now grows FASTER than     #
    #  linearly in M (M and M^2 terms), unlike the pre-correction formula  #
    #  (which only had the M*layer_size^2 term scale with M).             #
    #  Previously slm_x_num had its OWN formula (same N budget, but        #
    #  divided by a hardcoded 2 instead of tc.num_layers, and without the  #
    #  M factor) which only coincidentally matched layer_size while        #
    #  num_layers==2 and M==1 -- seek git history if you ever need that    #
    #  old, K/M-independent formula back.                                  #
    # ------------------------------------------------------------------ #
    tc.slm_dx      = tc.pixel_pitch
    tc.slm_x_num   = tc.layer_size   # phase key size matches diffractive layer size
    tc.slm_bin     = round(tc.slm_dx / tc.sim_dx)
    tc.slm_x_num_sim = tc.slm_x_num * tc.slm_bin

    tc.slm_hw_x      = 1920  # NOTE: real-device pixel count, non-binding at
    tc.slm_hw_y      = 1080  # this scale (bigger than N_sim -- gets clipped
                              # to N_sim in model.py, i.e. no additional
                              # aperture restriction beyond the sim window).
    tc.slm_bit_depth = 8

    tc.mask_init_method = 'normal'
    tc.mask_init_std   = 0.5

    # ------------------------------------------------------------------ #
    #  Function-input encoding plane (deterministic, NOT learned)         #
    #  Replaces the old "Object (MNIST phase images)" plane below --      #
    #  instead of an image, this plane carries phi_in(p;a) = 2*pi *       #
    #  encoding_freq_step * (p-1) * a for p = 1..Np (PAPER Sec. 2.2/4.1).  #
    # ------------------------------------------------------------------ #
    tc.encoding_dx        = tc.slm_dx * tc.encoding_patch_scale  # == tc.pixel_pitch
                                  # PAPER (Sec. 2.2) at the default patch_scale=1;
                                  # patch_scale > 1 widens each encoding phase
                                  # pixel beyond the paper's 1:1 pitch -- see
                                  # tc.encoding_patch_scale above.
    tc.encoding_bin       = round(tc.encoding_dx / tc.sim_dx)
    tc.encoding_side      = int(round(np.sqrt(tc.Np)))
    tc.encoding_x_num_sim = tc.encoding_bin * (
        tc.encoding_side + (tc.encoding_side - 1) * tc.encoding_gap_blocks)

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

    # ------------------------------------------------------------------ #
    #  Photodiode array  (pd_num_rows x pd_num_cols detectors)             #
    #  ONE intensity detector per target function -- detector (r,c) reads #
    #  out f_hat_k(a) for k = r*pd_num_cols + c, via direct min-max        #
    #  normalization (PAPER Eq. 9), no positive/negative differential      #
    #  pairing anymore. pd_num_rows * pd_num_cols must equal tc.Nf (asserted  #
    #  in recompute_derived).                                              #
    # ------------------------------------------------------------------ #
    tc.photodiode_size   = tc.pixel_pitch      # one detector takes up one pixel (lambda/2)
    tc.photodiode_pixels = round(tc.photodiode_size / tc.sim_dx)
    tc.pd_num_rows = int(np.sqrt(tc.Nf))   # 10 x 10 == Nf (100)
    tc.pd_num_cols = int(np.sqrt(tc.Nf))
    tc.num_photodiodes = tc.pd_num_rows * tc.pd_num_cols

    tc.pd_row_spacing = 2 * tc.photodiode_size  # center-to-center spacing, row direction
    tc.pd_col_spacing = 2 * tc.photodiode_size  # center-to-center spacing, column direction
    tc.pd_row_spacing_px = round(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = round(tc.pd_col_spacing / tc.sim_dx)

    # Offset of the whole array's center, in sim-grid pixels relative to the optical
    # axis (N_sim/2). Default 0 centers the array on the axis. Nonzero shifts the
    # whole array (e.g. to model misalignment).
    tc.detector_offset_y = 0
    tc.detector_offset_x = 0

    # ------------------------------------------------------------------ #
    #  Nonlinear function approximation -- sampling of `a`. #
    # ------------------------------------------------------------------ #
    tc.train_a_samples = 10000  # number of training samples of a
    tc.val_a_grid_size  = 1000   # number of validation samples of a (grid)
    tc.test_a_grid_size = 1000  # number of test samples of a (grid)

    # ------------------------------------------------------------------ #
    #  Training hyper-parameters                                          #
    # ------------------------------------------------------------------ #
    tc.batch_size       = 64
    tc.test_batch_size  = 4
    tc.max_epoch        = 150
    tc.seed             = 59

    tc.num_workers = 0

    tc.lr_slm     = 1e-2   # learning rate for the phase-key plane
    tc.lr_layer   = 1e-2   # learning rate for the diffractive layers

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
