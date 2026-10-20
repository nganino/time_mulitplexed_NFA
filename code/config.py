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
    datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    return (f'{datetime_str}'
            f'-Np{tc.Np}{"-split" if getattr(tc, "encoding_mode", "constant") == "split" else ""}'
            f'-Nf{tc.Nf}-M{tc.M}-K{tc.num_layers}'
            f'-Spacings{1e6*tc.key_to_enc_spacing:.2f}um-{1e6*tc.slm_first_layer_spacing:.2f}um-{1e6*tc.interlayer_spacing:.2f}um-{1e6*tc.last_layer_ccd_spacing:.2f}um'
            f'-batchsize{tc.batch_size}-lrkey{tc.lr_slm:.0e}-lrlayer{tc.lr_layer:.0e}-pdsize{1e9*tc.photodiode_size:.0f}nm'
            f'-trainA{tc.train_a_samples}-{tc.loss_type}'
            # pitches only when they differ from the original 300 nm simulation, so old names are unchanged
            f'{"" if abs(tc.slm_dx - 300e-9) < 1e-12 else f"-slm{1e6 * tc.slm_dx:.3g}um-sim{1e6 * tc.sim_dx:.3g}um"}')

def config_to_dict(tc):
    return {k: (v.item() if hasattr(v, 'item') else v)
            for k, v in vars(tc).items()
            if not k.startswith('_') and (isinstance(v, (int, float, str, bool, type(None)))
                                           or hasattr(v, 'item'))}

def _build_log_paths(tc):
    tc.log_dir     = os.path.join(_paths.LOG_DIR, tc.run_name)
    tc.image_dir   = os.path.join(tc.log_dir, 'images')
    tc.model_dir   = os.path.join(tc.log_dir, 'model')
    tc.tfboard_dir = os.path.join(tc.log_dir, 'tfboard')


#  DESIGN_KEYS are re-derived from their formulas on EVERY recompute_derived() call                          #
DESIGN_KEYS = ('N_trainable_features', 'layer_size', 'phase_key_size')

def _explicit_set(tc):
    if not isinstance(getattr(tc, '_explicit', None), set):
        tc._explicit = set()
    return tc._explicit

def mark_explicit(tc, keys):
    '''Record keys as user-set, so recompute_derived() never re-derives them.'''
    _explicit_set(tc).update(keys)


def apply_saved_config(tc, saved):
    '''Load a saved config dict (e.g. ckpt['config']) onto tc and mark EVERY
    loaded key explicit, so the rebuilt model is exactly what was trained --
    even if a formula or default has changed since. Call recompute_derived(tc) afterwards.'''
    tc.__dict__.update(saved)
    mark_explicit(tc, list(saved.keys()))
    if 'encoding_tile_pattern' not in saved:     # saved before tiles existed (2026-10-09): one patch
        tc.encoding_tile_pattern = 'single'
        mark_explicit(tc, ['encoding_tile_pattern'])
    if 'decoder_over_encoding' not in saved:     # saved before the flag existed (2026-10-17): decoder off the encoding pixels
        tc.decoder_over_encoding = False


def _derive(tc, key, value):
    if key not in _explicit_set(tc):
        setattr(tc, key, value)


def describe_design(tc):
    '''One line per design value, tagged [set] or [derived] -- printed at train start.'''
    tag = lambda k: 'set' if k in _explicit_set(tc) else 'derived'
    lines = [
        f'  device profile = {tc.device_profile}: SLM {1e6 * tc.slm_dx:.3g} um, sim {1e6 * tc.sim_dx:.3g} um '
        f'({tc.sim_per_slm} sim px / SLM px), camera {1e6 * tc.camera_dx:.3g} um ({tc.cam_bin} sim px), '
        f'N_sim {tc.N_sim} ({1e3 * tc.N_sim * tc.sim_dx:.3g} mm), gaps '
        f'{", ".join(f"{1e3 * z:.3g}" for z in _plane_gaps(tc))} mm',
        f'  N_trainable_features = {tc.N_trainable_features} [{tag("N_trainable_features")}]'
        f'  (formula: ceil(r*2*N_alpha*Nf{"*M" if tc.scale_layer_with_M and tc.encoding_mode == "constant" else ""}), '
        f'r={tc.r}, N_alpha={tc.N_alpha}, Nf={tc.Nf})',
        (f'  layer_size     = {tc.layer_size} [{tag("layer_size")}]  (formula: ceil(sqrt(N/K)), K={tc.num_layers})'
         if tc.num_layers > 0 else '  layer_size     = 0  (K = 0: no diffractive layers, only the phase keys train)'),
        f'  phase_key_size = {tc.phase_key_size} [{tag("phase_key_size")}]  (formula: '
        f'{(str(tc.phase_key_patch_multiple) + " x encoding patch") if getattr(tc, "phase_key_patch_multiple", None) is not None else ("fixed by the " + tc.device_profile + " profile") if DEVICE_PROFILES[tc.device_profile].get("phase_key_size") is not None else "= layer_size"}'
        f' / key bin) -> {tc.phase_key_size_sim} sim px',
        f'  detectors      = {tc.pd_num_rows} x {tc.pd_num_cols}'
        f'  (formula: sqrt(Nf)); size {tc.photodiode_pixels} px, pitch '
        f'{tc.pd_row_spacing_px} x {tc.pd_col_spacing_px} px',
        f'  encoding mode  = {tc.encoding_mode}: Np = {tc.Np} px per frame, N_alpha = {tc.N_alpha} harmonics',
        f'  encoding       = {tc.encoding_rows} x {tc.encoding_cols} px (rows x cols), bin {tc.encoding_bin}, '
        f'gap {tc.encoding_gap_blocks} -> {tc.encoding_y_num_sim} x {tc.encoding_x_num_sim} sim px',
        f'  encoding tiles = {getattr(tc, "encoding_tile_pattern", "single")} '
        f'-> extent {tc.encoding_extent_sim} sim px',
    ]
    if getattr(tc, 'decoder', False):
        lines.append(f'  decoder (E+D)  = {tc.decoder_size_px} x {tc.decoder_size_px} SLM px '
                     f'({"= key size" if tc.decoder_size is None else "set"}) -> {tc.decoder_size_sim} sim px, shared by '
                     f'all M frames, trainable on every '
                     f'{"pixel (incl. encoding pixels)" if getattr(tc, "decoder_over_encoding", False) else "non-encoding pixel"}, phase init 0')
    # Patch copies outside the key are not key-modulated, outside the layer they miss layer 1.
    cover = tc.phase_key_size_sim if tc.num_layers == 0 else min(tc.phase_key_size_sim, tc.layer_size_sim)
    if tc.encoding_extent_sim > cover:
        lines.append(f'  WARNING: encoding extent {tc.encoding_extent_sim} px > key/layer '
                     f'{cover} px -- outer copies '
                     f'are not covered by the key / first layer')
    # Guard band: to prevent light from wrapping around the simulation grid due to FFT
    if tc.num_layers > 0:
        checks = [('phase key', tc.phase_key_size_sim, tc.key_to_enc_spacing),
                  ('layer', tc.layer_size_sim, max(tc.slm_first_layer_spacing, tc.interlayer_spacing,
                                                   tc.last_layer_ccd_spacing))]
    else:   # K = 0: the key / encoding region propagates straight to the camera
        checks = [('phase key', tc.phase_key_size_sim, max(tc.key_to_enc_spacing, tc.last_layer_ccd_spacing))]
    for name, size, z in checks:
        need = size + 2 * spread_px(tc, z)
        if need > tc.N_sim:
            lines.append(f'  WARNING: {name} {size} px + 2 x {spread_px(tc, z)} px spread = {need} px > '
                         f'N_sim {tc.N_sim} -- light will wrap around the grid; raise N_sim')
    # ASM transfer-function aliasing: H's phase chirp is undersampled beyond this distance
    s = min(tc.wavelength / (2 * tc.sim_dx), 0.999)
    z_alias = tc.N_sim * tc.asm_pad_factor * tc.sim_dx ** 2 / tc.wavelength * np.sqrt(1 - s * s)
    if max(_plane_gaps(tc)) > z_alias:
        lines.append(f'  WARNING: longest gap {1e3 * max(_plane_gaps(tc)):.3g} mm > ASM aliasing limit '
                     f'{1e3 * z_alias:.3g} mm (N_sim * pad * sim_dx^2 / lambda) -- raise N_sim or asm_pad_factor')
    return '\n'.join(lines)

# Device profiles: physical defaults applied in recompute_derived via _derive, so a --set or a
# value loaded from a checkpoint always wins. 'sim' = the original sub-wavelength simulation,
# 'lab' = the real devices (8 um SLM, Basler 5.86 um camera snapped to 6 um).
# camera_dx None -> camera pixel = sim pixel. N_sim 'auto' -> sized from the geometry + light spread.
# encoding_patch_scale / encoding_gap_blocks are in SLM px / bin-sized blocks. phase_key_size (SLM px) None ->
# derived (key = layer size, or 9 x patch at K = 0); a number fixes the key (and the decoder, which follows the key).
# Lab defaults locked 2026-10-17: 2 um sim px, 24 um encoding px with 72 um gaps, 400 SLM px key, 12 um detectors
# on a 24 um pitch (one detector width dark), 3 mm between planes. Sim: bin 3, gap 0 (bin 2 before 2026-10-17).
DEVICE_PROFILES = {
    'sim': dict(slm_dx=300 * 1e-9,     # = the old 300 * nm, bit-identical
                sim_per_slm=1, camera_dx=None, pd_size_cam=3, pd_pitch_cam=3,
                encoding_patch_scale=3, encoding_gap_blocks=0, phase_key_size=None,
                key_to_enc_spacing=6e-6, slm_first_layer_spacing=6e-6,
                interlayer_spacing=6e-6, last_layer_ccd_spacing=6e-6, N_sim=256),
    'lab': dict(slm_dx=8e-6, sim_per_slm=4, camera_dx=6e-6, pd_size_cam=2, pd_pitch_cam=4,
                encoding_patch_scale=3, encoding_gap_blocks=3, phase_key_size=400,
                key_to_enc_spacing=3e-3, slm_first_layer_spacing=3e-3,
                interlayer_spacing=3e-3, last_layer_ccd_spacing=3e-3, N_sim='auto'),
}

def _int_ratio(a, b, what):
    '''a / b as an int, raising if it is not (within 1e-6) a whole number.'''
    q = a / b
    if q < 1 - 1e-6 or abs(q - round(q)) > 1e-6:
        raise ValueError(f'{what}: {a:.4g} / {b:.4g} = {q:.4f} is not a whole number')
    return int(round(q))

def spread_sine(tc):
    '''Sine of the widest propagation angle that carries real light: the grid's own limit, or
    guard_orders diffraction orders of one SLM pixel, whichever is smaller.'''
    return min(tc.wavelength / (2 * tc.sim_dx), tc.guard_orders * tc.wavelength / tc.slm_dx, 0.999)

def _plane_gaps(tc):
    '''Propagation distances actually used (no layer gaps at K = 0).'''
    if tc.num_layers == 0:
        return [tc.key_to_enc_spacing, tc.last_layer_ccd_spacing]
    return [tc.key_to_enc_spacing, tc.slm_first_layer_spacing, tc.interlayer_spacing, tc.last_layer_ccd_spacing]

def spread_px(tc, z):
    '''Lateral spread in sim px of light travelling z at the spread_sine angle.'''
    s = spread_sine(tc)
    return int(np.ceil(z * s / np.sqrt(1 - s * s) / tc.sim_dx))

ENCODING_TILE_NAMES = {'single': '000010000',   # one patch in the centre (original)
                       'x':      '101010101',   # centre + 4 diagonal copies
                       'plus':   '010111010',   # centre + 4 edge copies
                       'full':   '111111111',   # all 9 cells (one continuous block at gap 0)
                       'full5':  '1' * 25}      # 5x5 grid of copies (added 2026-10-15)

def encoding_tile_cells(pattern):
    '''Tile pattern (a name in ENCODING_TILE_NAMES, or a row-major '0'/'1' string of 9 or 25 chars for
    a 3x3 or 5x5 grid centred on the original patch) -> list of (row, col) offsets where a patch copy
    sits: in {-1, 0, 1} for 3x3, {-2, ..., 2} for 5x5.'''
    bits = ENCODING_TILE_NAMES.get(str(pattern).lower(), str(pattern))
    n = int(round(np.sqrt(len(bits))))
    assert n * n == len(bits) and n in (3, 5) and set(bits) <= {'0', '1'} and '1' in bits, \
        f"encoding_tile_pattern must be one of {list(ENCODING_TILE_NAMES)} or 9 / 25 chars of 0/1, got {pattern!r}"
    h = n // 2
    return [(i // n - h, i % n - h) for i, b in enumerate(bits) if b == '1']

def recompute_derived(tc):
    '''Recompute everything that depends on other config values.'''
    # ---- device profile: physical pitches / distances (explicit values win) ----
    tc.device_profile = getattr(tc, 'device_profile', 'sim')
    assert tc.device_profile in DEVICE_PROFILES, \
        f'unknown device_profile {tc.device_profile!r}, choose from {list(DEVICE_PROFILES)}'
    for k, v in DEVICE_PROFILES[tc.device_profile].items():
        if k != 'phase_key_size':                            # design value, derived with the key below
            _derive(tc, k, v)

    # SLM pitch -> sim pitch. Keys AND layers sit on SLM pixels, so both bin by sim_per_slm.
    # Setting sim_dx directly (or loading an old checkpoint that has it) fixes sim_per_slm instead.
    if 'sim_dx' in _explicit_set(tc):
        tc.sim_per_slm = _int_ratio(tc.slm_dx, tc.sim_dx, 'slm_dx / sim_dx')
    tc.sim_dx = tc.slm_dx / tc.sim_per_slm
    # old configs carried a separate layer pitch; every run so far had it equal to the SLM pitch
    if 'layer_dx' in _explicit_set(tc) and abs(tc.layer_dx / tc.slm_dx - 1) > 1e-6:
        raise ValueError(f'layer_dx {tc.layer_dx:.4g} != slm_dx {tc.slm_dx:.4g} (layers sit on SLM pixels)')
    tc.layer_dx = tc.slm_dx

    # Camera pitch -> sim pitch; detector size / pitch are whole camera pixels
    if tc.camera_dx is None:
        tc.camera_dx = tc.sim_dx
    tc.cam_bin = _int_ratio(tc.camera_dx, tc.sim_dx, 'camera_dx / sim_dx')
    _legacy_pd = {'photodiode_size', 'pd_row_spacing', 'pd_col_spacing'} & _explicit_set(tc)
    if _legacy_pd:
        # detector size given in metres (old checkpoint / old --set): rounded to sim px as before
        tc.photodiode_pixels = round(tc.photodiode_size / tc.sim_dx)
        tc.pd_row_spacing_px = round(tc.pd_row_spacing / tc.sim_dx)
        tc.pd_col_spacing_px = round(tc.pd_col_spacing / tc.sim_dx)
    else:
        tc.photodiode_pixels = tc.pd_size_cam * tc.cam_bin
        tc.pd_row_spacing_px = tc.pd_col_spacing_px = tc.pd_pitch_cam * tc.cam_bin
        tc.photodiode_size = tc.photodiode_pixels * tc.sim_dx
        tc.pd_row_spacing = tc.pd_col_spacing = tc.pd_row_spacing_px * tc.sim_dx

    tc.encoding_mode = getattr(tc, 'encoding_mode', 'constant') # const vs split
    assert tc.encoding_mode in ('constant', 'split'), f'unknown encoding_mode {tc.encoding_mode!r}'
    tc.N_alpha = tc.Np * (tc.M if tc.encoding_mode == 'split' else 1)

    _derive(tc, 'N_trainable_features', int(np.ceil( tc.r * 2 * tc.N_alpha * tc.Nf * (tc.M if tc.scale_layer_with_M and tc.encoding_mode == 'constant' else 1))))
    _derive(tc, 'layer_size', int(np.ceil(np.sqrt(tc.N_trainable_features / tc.num_layers))) if tc.num_layers > 0 else 0)

    # detector grid: always sqrt(Nf) x sqrt(Nf), one detector per target function (Nf must be a perfect square)
    side = int(np.round(np.sqrt(tc.Nf)))
    if side * side != tc.Nf:
        raise ValueError(f'Nf={tc.Nf} is not a perfect square (detector grid is sqrt(Nf) x sqrt(Nf))')
    tc.pd_num_rows = tc.pd_num_cols = side

    # ---- pure geometry: always recomputed -----------------------------------
    tc.layer_bin      = tc.sim_per_slm                       # one layer pixel per SLM pixel
    tc.layer_size_sim = tc.layer_size * tc.layer_bin

    # ---- new NFA (function-approximation) derived params ------------------
    # Encoding-plane geometry: the patch's pixels arranged as a rows x cols grid, chooses factorization closest to square
    tc.encoding_dx        = tc.slm_dx * tc.encoding_patch_scale   # encoding_patch_scale = SLM px per encoding px
    tc.encoding_bin       = _int_ratio(tc.encoding_dx, tc.sim_dx, 'encoding_dx / sim_dx')
    # patch pixels: always Np (split mode shows a new Np-pixel block on the same patch every frame)
    n_pix = tc.Np
    tc.encoding_rows = max(d for d in range(1, int(np.sqrt(n_pix)) + 1) if n_pix % d == 0) # computes the optimal factorization for rows x cols (largest integer # of rows < sqrt(n_pix) )
    tc.encoding_cols = n_pix // tc.encoding_rows

    _g = tc.encoding_gap_blocks
    tc.encoding_y_num_sim = tc.encoding_bin * (tc.encoding_rows + (tc.encoding_rows - 1) * _g) # height in sim px of one patch copy (including gaps)
    tc.encoding_x_num_sim = tc.encoding_bin * (tc.encoding_cols + (tc.encoding_cols - 1) * _g) # width in sim px of one patch copy (including gaps)
    # Patch copies: neighbouring copies sit one patch + one gap apart along each axis (gap 0 -> touching)
    _cells = encoding_tile_cells(getattr(tc, 'encoding_tile_pattern', 'single')) # list of (row, col) offsets (3x3: {-1, 0, 1}, 5x5: {-2..2}) where a patch copy sits
    _ry = max(abs(r) for r, _ in _cells); _rx = max(abs(c) for _, c in _cells) # furthest copy from the centre along each axis
    tc.encoding_extent_sim = max(
        tc.encoding_y_num_sim + 2 * _ry * (tc.encoding_y_num_sim + tc.encoding_bin * _g),
        tc.encoding_x_num_sim + 2 * _rx * (tc.encoding_x_num_sim + tc.encoding_bin * _g)) # total extent in sim px of all patch copies in input plane

    # Phase key. Default width = one layer (layer_size_sim). phase_key_patch_multiple = X
    # instead sizes it as X x the encoding patch width
    tc.phase_key_bin = tc.sim_per_slm                        # one key pixel per SLM pixel
    # A device profile can fix the key size (lab: 400 SLM px); phase_key_patch_multiple still overrides it
    profile_key = DEVICE_PROFILES[tc.device_profile].get('phase_key_size')
    # K = 0: there is no layer size to copy, so the key defaults to 9 x the encoding patch width
    # (~3 x the full 3x3 tile block, enough to modulate every patch copy) unless set explicitly
    if profile_key is None and tc.num_layers == 0 and getattr(tc, 'phase_key_patch_multiple', None) is None:
        tc.phase_key_patch_multiple = 9
    mult = getattr(tc, 'phase_key_patch_multiple', None)
    if mult is None and profile_key is not None:
        n_key = int(profile_key)
    else:
        key_width_sim = tc.layer_size_sim if mult is None else round(mult * tc.encoding_x_num_sim)
        n_key = -(-key_width_sim // tc.phase_key_bin)      # key px covering that width
    _derive(tc, 'phase_key_size', n_key)
    tc.phase_key_size_sim = tc.phase_key_size * tc.phase_key_bin

    # Decoder (E+D architecture, model.py): learnable phase on the encoding plane's non-encoding SLM pixels,
    # decoder_size SLM px per side (None -> the phase key size)
    tc.decoder = bool(getattr(tc, 'decoder', False))
    if tc.decoder:
        tc.decoder_size_px = int(tc.decoder_size) if getattr(tc, 'decoder_size', None) else int(tc.phase_key_size)
        tc.decoder_size_sim = tc.decoder_size_px * tc.phase_key_bin

    assert tc.pd_num_rows * tc.pd_num_cols == tc.Nf, (
        'photodiode array size (pd_num_rows * pd_num_cols) must equal Nf '
        '-- one detector per target function'
    )
    tc.num_photodiodes = tc.pd_num_rows * tc.pd_num_cols

    # total number of learnable phase-key masks
    tc.T = int(tc.M)                                        # one frame per phase key

    # N_sim 'auto': largest plane + light spread over the longest gap on both sides, rounded up to 64
    if tc.N_sim == 'auto':
        det_span = (tc.pd_num_rows - 1) * tc.pd_row_spacing_px + tc.photodiode_pixels
        biggest = max(tc.phase_key_size_sim, tc.layer_size_sim, tc.encoding_extent_sim, det_span,
                      tc.decoder_size_sim if tc.decoder else 0)
        need = biggest + 2 * spread_px(tc, max(_plane_gaps(tc)))
        tc.N_sim = int(-(-need // 64) * 64)
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
    # ------------------------------------------------------------------ #
    #  Device profile & simulation grid                                  #
    # ------------------------------------------------------------------ #
    # 'sim' (300 nm SLM = sim pixel, 6 um gaps, N_sim 256) or 'lab' (8 um SLM, 1 um sim px,
    # 6 um camera, 2 mm gaps, N_sim auto). The profile fills slm_dx, sim_per_slm (sim px per
    # SLM px; sim_dx = slm_dx / sim_per_slm), camera_dx, pd_size_cam, pd_pitch_cam, the four
    # spacings and N_sim -- see DEVICE_PROFILES. Any of them can still be --set individually.
    tc.device_profile = 'sim'
    tc.guard_orders   = 2   # light spread assumed up to this many diffraction orders of one SLM pixel (N_sim auto + wrap warning)
    tc.asm_pad_factor = 1
    tc.r = 1.25 # scaling factor of the diffractive layer's feature count

    # ------------------------------------------------------------------ #
    #  Nonlinear function targets & input encoding                       #
    # ------------------------------------------------------------------ #
    tc.Np = 9                    # number of input-encoding pixels
    tc.Nf = 100                  # Number of functions = number of detectors
    tc.a_min = -0.5               #input domain of a
    tc.a_max = 0.5                

    tc.encoding_freq_step = 1     # phi_in(p;a) = 2*pi*encoding_freq_step*(p-1)*a
    # encoding_patch_scale (binning: each p value takes up bin x bin SLM pixels) and encoding_gap_blocks (zero-phase
    # bin-sized blocks between adjacent encoding pixels) come from the device profile (sim: 2 / 0, lab: 3 / 3).
    # Harmonic p sits at row p // cols, column p % cols of the rows x cols patch (p = 0 top-left).

    tc.encoding_tile_pattern = 'full'   # 'full' (default since 2026-10-17) copies the patch to a 3x3 grid around the centre, 'single' = one patch, 'full5' = 5x5 grid
    tc.encoding_mode = 'constant'       # how harmonics are spread over the M keys (constant: all keys see all Np harmonics; split: key m sees only block of harmonics Np = m*Np .. (m+1)*Np-1)
                                        # const: N_alpha = Np; split: N_alpha = Np*M and N = r*2*N_alpha*Nf grows with M
    tc.func_seed = 0                    # random seed for generating Nf target functinons

    # ------------------------------------------------------------------ #
    #  Diffractive Layers                                                #
    # ------------------------------------------------------------------ #
    tc.M             = 1   # number of phase keys
    tc.num_layers    = 2   
    tc.scale_layer_with_M = False  # N = r * 2*Np*Nf*M  (True) or N = r * 2*Np*Nf (False)
    # layers sit on SLM pixels (layer pitch = slm_dx)

    # ------------------------------------------------------------------ #
    #  Phase-key plane                                                   #   
    # ------------------------------------------------------------------ #
    tc.phase_key_patch_multiple = None  # None: key = layer size; X: key = X x encoding patch width

    tc.slm_hw_x      = 1920  # real-device pixel count
    tc.slm_hw_y      = 1080  
    tc.slm_bit_depth = 8

    # ------------------------------------------------------------------ #
    #  Decoder at the encoding plane (E+D architecture)                  #
    # ------------------------------------------------------------------ #
    tc.decoder      = False   # True: train a phase mask on every non-encoding SLM pixel of the encoding plane
    tc.decoder_size = None    # SLM px per side; None -> phase_key_size
    tc.decoder_over_encoding = True   # True (default since 2026-10-18): decoder also trains on the encoding pixels (every pixel of its square; superset of False)

    tc.mask_init_method = 'normal'
    tc.mask_init_std   = 1

    # ------------------------------------------------------------------ #
    #  Distances between planes                                          #
    # ------------------------------------------------------------------ #
    # key_to_enc_spacing, slm_first_layer_spacing, interlayer_spacing, last_layer_ccd_spacing
    # come from the device profile. key_to_enc_spacing = 0 adds the key phase directly to the
    # input encoding phase, exp(1j*(phi_key + phi_in))
    tc.key_to_enc_exact_at_zero = True

    # ------------------------------------------------------------------ #
    #  Photodiode array                                                  #
    # ------------------------------------------------------------------ #
    # Detector size / centre-to-centre pitch are whole camera pixels (pd_size_cam, pd_pitch_cam,
    # from the profile; equal -> adjacent detectors touch). photodiode_size / pd_row_spacing /
    # pd_col_spacing in metres are derived for display; --set-ting them (as old configs did)
    # overrides the camera-pixel values.

    # Offset detector array's center, relative to optical axis (in sim pixels)
    tc.detector_offset_y = 0
    tc.detector_offset_x = 0

    # ------------------------------------------------------------------ #
    #  Nonlinear function approximation -- sampling of `a`.              #
    # ------------------------------------------------------------------ #
    tc.train_a_samples = 10000  
    tc.val_a_grid_size  = 1000  
    tc.test_a_grid_size = 1000 

    # ------------------------------------------------------------------ #
    #  Training hyper-parameters                                         #
    # ------------------------------------------------------------------ #
    tc.batch_size       = 64
    tc.grad_accum_steps = 1    # split each batch into this many micro-batches (same gradient, less memory)
    tc.test_batch_size  = 4
    tc.max_epoch        = 150
    tc.seed             = 59
    tc.num_workers = 0
    tc.lr_slm     = 1e-2   # learning rate for the phase-key plane
    tc.lr_layer   = 1e-2   # learning rate for the diffractive layers
    tc.lr_decoder = 1e-2   # learning rate for the encoding-plane decoder (only used when decoder = True)
    tc.slm_warmup_epochs = 0
    tc.meas_noise_std = 0  # End-to-end measurement-noise injection (SLM + decoder co-adapt to the sim->real gap).
    tc.freeze_slm = False 
    tc.loss_type = 'mse'

    # ------------------------------------------------------------------ #
    #  Logging & checkpoints                                             #
    # ------------------------------------------------------------------ #
    tc.checkpoint_save  = 10 # every (checkpoint_save) epochs, save a checkpoint (model + config)
    tc.checkpoint_print = 1
    tc.ckpt_to_load = None 
    tc.load_weights_only = False

    tc._explicit = set()    # keys set via --set / a loaded checkpoint (never re-derived)
    recompute_derived(tc)   # fills every derived value (layer_size, N_alpha, T, run_name, ...)
    return tc
