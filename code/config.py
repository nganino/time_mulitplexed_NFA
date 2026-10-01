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


# Config keys renamed 2026-09-28 (old name -> new name). Checkpoints saved before
# the rename store the OLD names in their config dict, and older sweep scripts pass
# them via --set, so both paths translate through this map (_migrate_legacy_keys
# below, and train.py's _apply_overrides).
LEGACY_ALIASES = {
    'slm_x_num':     'phase_key_size',
    'slm_bin':       'phase_key_bin',
    'slm_x_num_sim': 'phase_key_size_sim',
}


def _migrate_legacy_keys(tc):
    '''Move any pre-rename keys (e.g. from a loaded checkpoint config) onto their
    new names. The loaded value wins over whatever default init_params() set.'''
    explicit = _explicit_set(tc)
    for old, new in LEGACY_ALIASES.items():
        if old in tc.__dict__:
            setattr(tc, new, tc.__dict__.pop(old))
        if old in explicit:
            explicit.discard(old)
            explicit.add(new)


# --------------------------------------------------------------------------- #
#  "Derive unless you set it"                                                  #
#                                                                              #
#  DESIGN_KEYS are re-derived from their formulas on EVERY recompute_derived() #
#  call (so `--set Np=25` alone gives the right layer size), UNLESS the key is #
#  in tc._explicit -- i.e. you passed it via --set (train.py _apply_overrides) #
#  or it came from a saved checkpoint config (apply_saved_config). Explicit    #
#  values are never overwritten. tc._explicit starts with a leading '_' so     #
#  config_to_dict() never writes it to config.json.                            #
# --------------------------------------------------------------------------- #
DESIGN_KEYS = ('N_trainable_features', 'layer_size', 'phase_key_size',
               'pd_num_rows', 'pd_num_cols')


def _explicit_set(tc):
    if not isinstance(getattr(tc, '_explicit', None), set):
        tc._explicit = set()
    return tc._explicit


def mark_explicit(tc, keys):
    '''Record keys as user-set, so recompute_derived() never re-derives them.'''
    _explicit_set(tc).update(keys)


# Behavior switches added after checkpoints already existed: a saved config that
# lacks the key was trained with the OLD behavior, so it loads with this value
# instead of the current default (keeps old checkpoints reproducible).
LEGACY_DEFAULTS = {
    'key_to_enc_exact_at_zero': False,   # added 2026-09-28; before, z=0 still ran the propagator
}


def apply_saved_config(tc, saved):
    '''Load a saved config dict (e.g. ckpt['config']) onto tc and mark EVERY
    loaded key explicit, so the rebuilt model is exactly what was trained --
    even if a formula or default has changed since (LEGACY_DEFAULTS covers
    switches the saved config predates). Call recompute_derived(tc) afterwards.'''
    tc.__dict__.update(saved)
    for key, old_value in LEGACY_DEFAULTS.items():
        if key not in saved:
            setattr(tc, key, old_value)
    mark_explicit(tc, list(saved.keys()) + list(LEGACY_DEFAULTS))


def _derive(tc, key, value):
    if key not in _explicit_set(tc):
        setattr(tc, key, value)


def describe_design(tc):
    '''One line per design value, tagged [set] or [derived] -- printed at train start.'''
    tag = lambda k: 'set' if k in _explicit_set(tc) else 'derived'
    lines = [
        f'  N_trainable_features = {tc.N_trainable_features} [{tag("N_trainable_features")}]'
        f'  (formula: ceil(r*2*Np*Nf{"*M" if tc.scale_layer_with_M else ""}), r={tc.r}, Np={tc.Np}, Nf={tc.Nf})',
        f'  layer_size     = {tc.layer_size} [{tag("layer_size")}]  (formula: ceil(sqrt(N/K)), K={tc.num_layers})',
        f'  phase_key_size = {tc.phase_key_size} [{tag("phase_key_size")}]  (formula: = layer_size'
        f'{", / phase_key_bin_scale, grid-aligned to encoding" if getattr(tc, "phase_key_bin_scale", 1) > 1 else ""})'
        f', bin {tc.phase_key_bin} -> {tc.phase_key_size_sim} sim px',
        f'  detectors      = {tc.pd_num_rows} x {tc.pd_num_cols} [{tag("pd_num_rows")}/{tag("pd_num_cols")}]'
        f'  (formula: sqrt(Nf)); size {tc.photodiode_pixels} px, pitch '
        f'{tc.pd_row_spacing_px} x {tc.pd_col_spacing_px} px',
        f'  encoding       = {tc.encoding_side} x {tc.encoding_side} px, bin {tc.encoding_bin}, '
        f'gap {tc.encoding_gap_blocks} -> {tc.encoding_x_num_sim} sim px',
        f'  key trains     = {getattr(tc, "key_train_region", "all")}'
        f'{" (key_mask_encoder_footprint)" if getattr(tc, "key_mask_encoder_footprint", False) else ""}',
    ]
    if getattr(tc, 'phase_key_bin_scale', 1) == 1 and tc.phase_key_size != tc.layer_size:
        lines.append(f'  WARNING: phase_key_size ({tc.phase_key_size}) != layer_size ({tc.layer_size})')
    return '\n'.join(lines)


def recompute_derived(tc):
    '''Recompute everything that depends on other config values. Called at the
    end of init_params(), after --set overrides (train.py), and after loading a
    saved config (test.py). See the "Derive unless you set it" block above.'''
    _migrate_legacy_keys(tc)

    # ---- design values: derived from formulas unless explicitly set --------
    _derive(tc, 'N_trainable_features', int(np.ceil(
        tc.r * 2 * tc.Np * tc.Nf * (tc.M if tc.scale_layer_with_M else 1))))
    _derive(tc, 'layer_size', int(np.ceil(np.sqrt(tc.N_trainable_features / tc.num_layers))))
    # phase_key_size is derived further down (it needs the encoding geometry)

    # detector grid: sqrt(Nf) x sqrt(Nf); if only one side is set, the other is Nf / that side
    explicit = _explicit_set(tc)
    if 'pd_num_rows' in explicit and 'pd_num_cols' not in explicit:
        _derive(tc, 'pd_num_cols', tc.Nf // tc.pd_num_rows)
    elif 'pd_num_cols' in explicit and 'pd_num_rows' not in explicit:
        _derive(tc, 'pd_num_rows', tc.Nf // tc.pd_num_cols)
    elif 'pd_num_rows' not in explicit:
        side = int(np.round(np.sqrt(tc.Nf)))
        if side * side != tc.Nf:
            raise ValueError(f'Nf={tc.Nf} is not a perfect square -- pass pd_num_rows and/or '
                             f'pd_num_cols via --set (their product must equal Nf)')
        _derive(tc, 'pd_num_rows', side)
        _derive(tc, 'pd_num_cols', side)

    # ---- pure geometry: always recomputed -----------------------------------
    tc.photodiode_pixels = round(tc.photodiode_size / tc.sim_dx)

    tc.pd_row_spacing_px = round(tc.pd_row_spacing / tc.sim_dx)
    tc.pd_col_spacing_px = round(tc.pd_col_spacing / tc.sim_dx)

    tc.layer_bin      = round(tc.layer_dx / tc.sim_dx)
    tc.layer_size_sim = tc.layer_size * tc.layer_bin

    # ---- new NFA (function-approximation) derived params ------------------
    # Encoding-plane geometry: Np input pixels arranged as a
    # sqrt(Np) x sqrt(Np) square patch (paper, Sec. 2.2: "arranged
    # contiguously in a square grid"), sharing the SLM's physical pixel pitch.
    # Unlike layer_size/phase_key_size/the spacings, encoding_dx IS recomputed here
    # (not just at init_params()) so `--set encoding_patch_scale=...` alone is
    # enough -- no separate encoding_dx/encoding_bin override needed.
    tc.encoding_dx        = tc.slm_dx * tc.encoding_patch_scale
    tc.encoding_bin       = round(tc.encoding_dx / tc.sim_dx)
    tc.encoding_side      = int(round(np.sqrt(tc.Np)))
    assert tc.encoding_side ** 2 == tc.Np, \
        'tc.Np must be a perfect square (square encoding patch, per the paper)'
    tc.encoding_x_num_sim = tc.encoding_bin * (
        tc.encoding_side + (tc.encoding_side - 1) * tc.encoding_gap_blocks)

    # Phase key. phase_key_bin_scale > 1 bins the key: each key pixel covers
    # phase_key_bin x phase_key_bin sim px. The derived key keeps the layer's PHYSICAL
    # extent (ceil(layer_size / scale) key px, i.e. ~scale^2 fewer key params), rounded
    # up until the key's bin grid lines up with the encoding patch's pixel grid (both
    # are centered in N_sim) -- at z=0 the key and encoding phases add pixel by pixel.
    scale = getattr(tc, 'phase_key_bin_scale', 1)
    tc.phase_key_bin = round(tc.slm_dx / tc.sim_dx) * scale
    n_key = tc.layer_size
    if scale > 1:
        b = tc.phase_key_bin
        n_key = -(-tc.layer_size // scale)
        enc_off = (tc.N_sim - tc.encoding_x_num_sim) // 2
        if tc.encoding_bin % b == 0:
            for _ in range(2 * b):
                if ((tc.N_sim - n_key * b) // 2 - enc_off) % b == 0:
                    break
                n_key += 1
    _derive(tc, 'phase_key_size', n_key)
    tc.phase_key_size_sim = tc.phase_key_size * tc.phase_key_bin

    tc.N_alpha = tc.Np  # PAPER (Sec. 4.1): "We set Nalpha = Np"

    # Detector array: one intensity detector per target function (no more
    # positive/negative differential pairing -- see photodiode section below).
    assert tc.pd_num_rows * tc.pd_num_cols == tc.Nf, (
        'photodiode array size (pd_num_rows * pd_num_cols) must equal Nf '
        '-- one detector per target function'
    )
    tc.num_photodiodes = tc.pd_num_rows * tc.pd_num_cols

    # total number of learnable phase-key masks
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
    tc.Np = 9                    # number of input-encoding pixels
    tc.Nf = 100                  # Number of functions = number of detectors
    tc.a_min = -0.5               #input domain of a
    tc.a_max = 0.5                

    tc.encoding_freq_step = 1     # NOTE: not specified in paper as a separate
                                  # hyperparameter -- implicit in their
                                  # phi_in(p;a) = 2*pi*(p-1)*a, i.e. alpha_p =
                                  # (p-1)*encoding_freq_step with step fixed
                                  # at 1 (Sec. 4.1: "alpha_p = p - 1"). Exposed
                                  # here in case we ever want to change it.

    tc.encoding_opaque_background = False  

    tc.encoding_patch_scale = 2   # binning of the input pixels, 
                                  # each p value takes up encoding_patch_scale x
                                  # encoding_patch_scale simulation pixels

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
    #  a square layer). This is a GUIDELINE, not a strict requirement      #
    #  (paper's own wording). N_trainable_features and layer_size are      #
    #  DERIVED in recompute_derived() from the current Np/Nf/K/M/r, unless #
    #  you --set them explicitly (then your value sticks).                 #
    # ------------------------------------------------------------------ #
    tc.M             = 1   # NOTE: not defined in paper -- number of learned
                           # phase keys (time-multiplexed conditioning masks).

    tc.num_layers    = 2   # PAPER (Sec. 2.2): K
    tc.scale_layer_with_M = False  # N = r * 2*Np*Nf*M  (True) or N = r * 2*Np*Nf (False)
    tc.layer_dx      = tc.pixel_pitch  # PAPER: diffractive feature width == delta
    # derived in recompute_derived(): N_trainable_features, layer_size, layer_bin, layer_size_sim

    # ------------------------------------------------------------------ #
    #  Phase-key plane  (hardware device -- same SLM as before, new role) #
    #  NOTE: the M-key / "wisdom of the crowd" time-multiplexing idea is  #
    #  OUR OWN addition on top of the paper -- the paper has no phase-key #
    #  plane or per-key ensembling concept, so M's role/value below is    #
    #  not paper-derived.                                                 #
    #                                                                      #
    # Size of phase key matches the size of any given diffractive layer.   #
    # ------------------------------------------------------------------ #
    tc.slm_dx      = tc.pixel_pitch
    tc.phase_key_bin_scale = 1   # >1 bins key pixels (bin x bin sim px each), keeping the
                                 # key's physical size -> ~bin^2 fewer key params
    # derived in recompute_derived(): phase_key_size (= layer_size unless --set),
    # phase_key_bin, phase_key_size_sim

    tc.slm_hw_x      = 1920  # NOTE: real-device pixel count, non-binding at
    tc.slm_hw_y      = 1080  # this scale (bigger than N_sim -- gets clipped
                              # to N_sim in model.py, i.e. no additional
                              # aperture restriction beyond the sim window).
    tc.slm_bit_depth = 8

    tc.mask_init_method = 'normal'
    tc.mask_init_std   = 1

    tc.key_mask_encoder_footprint = False # True means phase keys are NOT trainable (pinned to 0) 
                                          # inside the encoder footprint

    # Which part of the phase keys trains (added 2026-09-29):
    #   'all'               -- whole key (default)
    #   'outside_footprint' -- same as key_mask_encoder_footprint=True (footprint pinned to 0)
    #   'footprint_only'    -- stage-2 fine-tune: ONLY the encoder-footprint pixels train, the
    #                          rest of the key is frozen (layers + readout still train). Meant to
    #                          Needs to start from an 'outside_footprint' checkpoint,
    #                          The footprint is initialized at exactly phase 0  so step 0 reproduces the stage-1 model.
    tc.key_train_region = 'all' 

    # ------------------------------------------------------------------ #
    #  Function-input encoding plane (deterministic, NOT learned)         #
    #  Replaces the old "Object (MNIST phase images)" plane below --      #
    #  instead of an image, this plane carries phi_in(p;a) = 2*pi *       #
    #  encoding_freq_step * (p-1) * a for p = 1..Np (PAPER Sec. 2.2/4.1).  #
    # ------------------------------------------------------------------ #
    # derived in recompute_derived(): encoding_dx (= slm_dx * encoding_patch_scale;
    # PAPER's 1:1 pitch at patch_scale=1), encoding_bin, encoding_side,
    # encoding_x_num_sim
    #
    # the paper's layer_size-scaled formula.
    # _W = tc.layer_size * tc.layer_dx
    # _z = _W * np.sqrt((2 * tc.pixel_pitch / tc.wavelength) ** 2 - 1)
    _z = 6e-6
    tc.key_to_enc_spacing      = _z   # phase-key -> encoding-plane spacing (NOTE, no paper analogue)
    # When key_to_enc_spacing == 0, skip the propagator and form the encoding-plane
    # field as exactly exp(j*(phi_key + phi_encoding)) (the z=0 propagator is not the
    # identity -- its band-limit mask drops |f| > 1/lambda). Added 2026-09-28;
    # checkpoints saved before then load it as False (see LEGACY_DEFAULTS).
    tc.key_to_enc_exact_at_zero = True
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

    tc.photodiode_size = 3 * tc.pixel_pitch   # detector width (3 sim pixels)
    tc.pd_row_spacing  = 3 * tc.pixel_pitch   # center-to-center pitch, row direction (3 px)
    tc.pd_col_spacing  = 3 * tc.pixel_pitch   # center-to-center pitch, column direction (3 px)
    # derived in recompute_derived(): pd_num_rows/pd_num_cols (= sqrt(Nf) unless
    # --set), photodiode_pixels, pd_row/col_spacing_px, num_photodiodes

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
    tc.loss_type = 'mse'      

    # ------------------------------------------------------------------ #
    #  Logging & checkpoints        #
    # ------------------------------------------------------------------ #
    tc.checkpoint_save  = 10
    tc.checkpoint_print = 1

    tc.ckpt_to_load = None  # path to a checkpoint to load (None = start from scratch)

    # When loading ckpt_to_load, load ONLY the model weights (skip epoch/optimizer/scheduler) so
    # training starts a fresh fine-tune from epoch 0 with a new LR schedule. False = resume.
    tc.load_weights_only = False

    tc._explicit = set()    # keys set via --set / a loaded checkpoint (never re-derived)
    recompute_derived(tc)   # fills every derived value (layer_size, N_alpha, T, run_name, ...)
    return tc
