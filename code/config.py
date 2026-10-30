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
            f'{"" if max(abs(tc.key_slm_dx - 300e-9), abs(tc.enc_slm_dx - 300e-9)) < 1e-12 else f"-key{1e6 * tc.key_slm_dx:.3g}um-enc{1e6 * tc.enc_slm_dx:.3g}um-sim{1e6 * tc.sim_dx:.3g}um"}'
            f'{"" if getattr(tc, "slm_phase_bits", "inf") == "inf" else f"-{tc.slm_phase_bits}bit"}')

def config_to_dict(tc):
    return {k: (v.item() if hasattr(v, 'item') else v)
            for k, v in vars(tc).items()
            if not k.startswith('_') and (isinstance(v, (int, float, str, bool, type(None)))
                                           or hasattr(v, 'item'))}

def _build_log_paths(tc):
    tc.log_dir     = os.path.join(_paths.LOG_DIR, tc.run_name)
    tc.model_dir   = os.path.join(tc.log_dir, 'model')
    tc.tfboard_dir = os.path.join(tc.log_dir, 'tfboard')


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
    even if a formula or default has changed since. Call recompute_derived(tc) afterwards.
    Only checkpoints that pass check_checkpoint_format() get here; the one shim: configs saved before 2026-10-25 have a
    single slm_dx (both SLMs), now key_slm_dx / enc_slm_dx.'''
    saved = dict(saved)
    if 'slm_dx' in saved and 'key_slm_dx' not in saved:
        saved['key_slm_dx'] = saved['enc_slm_dx'] = saved['slm_dx']
    for k in ('slm_dx', 'sim_per_slm', 'phase_key_bin', 'layer_bin', 'layer_dx'):
        saved.pop(k, None)
    tc.__dict__.update(saved)
    mark_explicit(tc, list(saved.keys()))


# Checkpoint format marker (2026-10-23): min-max readout and an opaque background at K > 0. Checkpoints written
# before (learned scale / bias readout, transparent background at every K) are not loadable.
CHECKPOINT_READOUT = 'minmax'

def check_checkpoint_format(ckpt, path=''):
    if ckpt.get('readout') != CHECKPOINT_READOUT:
        raise ValueError(f'{path}: checkpoint predates 2026-10-23 (learned scale / bias readout, transparent background '
                         f'at K > 0) and cannot be loaded; the model now uses a min-max readout and an opaque '
                         f'background at K > 0')


def _derive(tc, key, value):
    if key not in _explicit_set(tc):
        setattr(tc, key, value)


def describe_design(tc):
    '''One line per design value, tagged [set] or [derived] -- printed at train start.'''
    tag = lambda k: 'set' if k in _explicit_set(tc) else 'derived'
    lines = [
        f'  device profile = {tc.device_profile}: key SLM {1e6 * tc.key_slm_dx:.3g} um ({tc.key_ratio:.4g} sim px), '
        f'encoding SLM {1e6 * tc.enc_slm_dx:.3g} um ({tc.enc_ratio:.4g} sim px), sim {1e6 * tc.sim_dx:.3g} um, '
        f'camera {1e6 * tc.camera_dx:.3g} um ({tc.cam_bin} sim px), '
        f'N_sim {tc.N_sim} ({1e3 * tc.N_sim * tc.sim_dx:.3g} mm), gaps '
        f'{", ".join(f"{1e3 * z:.3g}" for z in _plane_gaps(tc))} mm',
        f'  N_trainable_features = {tc.N_trainable_features} [{tag("N_trainable_features")}]'
        f'  (formula: ceil(r*2*N_alpha*Nf{"*M" if tc.scale_layer_with_M and tc.encoding_mode == "constant" else ""}), '
        f'r={tc.r}, N_alpha={tc.N_alpha}, Nf={tc.Nf})',
        (f'  layer_size     = {tc.layer_size} [{tag("layer_size")}]  (formula: ceil(sqrt(N/K)), K={tc.num_layers})'
         if tc.num_layers > 0 else '  layer_size     = 0  (K = 0: no diffractive layers, only the phase keys train)'),
        f'  phase_key_size = {tc.phase_key_size} [{tag("phase_key_size")}]  (formula: '
        f'{("detector array / key_det_fill " + str(tc.key_det_fill)) if getattr(tc, "key_det_fill", None) is not None else (str(tc.phase_key_patch_multiple) + " x encoding patch") if getattr(tc, "phase_key_patch_multiple", None) is not None else ("fixed by the " + tc.device_profile + " profile") if DEVICE_PROFILES[tc.device_profile].get("phase_key_size") is not None else "= layer_size"}'
        f') -> {tc.phase_key_size_sim} sim px',
        f'  detectors      = {tc.pd_num_rows} x {tc.pd_num_cols}'
        f'  (formula: sqrt(Nf)); size {tc.photodiode_pixels} px, pitch '
        f'{tc.pd_row_spacing_px} x {tc.pd_col_spacing_px} px',
        f'  encoding mode  = {tc.encoding_mode}: Np = {tc.Np} px per frame, N_alpha = {tc.N_alpha} harmonics; displayed alpha = '
        f'{-tc.encoding_shift} .. {tc.N_alpha - 1 - tc.encoding_shift} (targets {"centered" if tc.harmonic_shift else "0-based"}); '
        f'background {"opaque" if tc.opaque_background else "transparent"} (background_opaque {tc.background_opaque})',
        f'  encoding       = {tc.encoding_rows} x {tc.encoding_cols} px (rows x cols), {tc.encoding_patch_scale} SLM px each, '
        f'gap {tc.encoding_gap_blocks} -> {tc.encoding_y_num_px} x {tc.encoding_x_num_px} SLM px = '
        f'{tc.encoding_y_num_sim} x {tc.encoding_x_num_sim} sim px',
        f'  encoding tiles = {getattr(tc, "encoding_tile_pattern", "single")} '
        f'-> extent {tc.encoding_extent_sim} sim px',
    ]
    bd = getattr(tc, 'slm_phase_bits', 'inf')
    lines.append(f'  SLM phase      = ' + ('continuous (slm_phase_bits inf)' if bd == 'inf' else
                                        f'{bd}-bit: {2 ** bd} levels, step 2pi/{2 ** bd} (straight-through gradient)'))
    lines.append('  readout        = val / test: f_hat = (I - Pmin) / (Pmax - Pmin), Pmin / Pmax over all detectors and the a-grid; '
                 + {'minmax': 'training loss: per-batch min-max (train_readout minmax)',
                    'globalminmax': 'training loss: global min-max, Pmin / Pmax re-measured on the a-grid every epoch '
                                    '(train_readout globalminmax)',
                    'affine': 'training loss: learned scale * I + offset, discarded after training (train_readout affine)'
                    }[getattr(tc, 'train_readout', 'affine')])
    if getattr(tc, 'decoder', False):
        lines.append(f'  decoder (E+D)  ={tc.decoder_size_px} x {tc.decoder_size_px} SLM px '
                     f'({"covers the key" if tc.decoder_size is None else "set"}) -> {tc.decoder_size_sim} sim px, '
                     f'{f"one per frame (M = {tc.M})" if getattr(tc, "decoder_per_key", False) else "shared by all M frames"}, trainable on every '
                     f'{"pixel (incl. encoding pixels)" if getattr(tc, "decoder_over_encoding", False) else "non-encoding pixel"}, phase init 0')
    # Patch copies outside the key are not key-modulated, outside the layer they miss layer 1.
    cover = tc.phase_key_size_sim if tc.num_layers == 0 else min(tc.phase_key_size_sim, tc.layer_size_sim)
    if tc.encoding_extent_sim > cover:
        lines.append(f'  WARNING: encoding extent {tc.encoding_extent_sim} px > key/layer '
                     f'{cover} px -- outer copies '
                     f'are not covered by the key / first layer')
    # Guard band: to prevent light from wrapping around the simulation grid due to FFT
    legs = _plane_legs(tc)
    if tc.num_layers > 0:
        checks = [('phase key', tc.phase_key_size_sim, [legs[0]]), ('layer', tc.layer_size_sim, legs[1:])]
    else:   # K = 0: the key / encoding region propagates straight to the camera
        checks = [('phase key', tc.phase_key_size_sim, legs)]
    for name, size, lg in checks:
        sp = max(spread_px(tc, z, p) for z, p in lg)
        if size + 2 * sp > tc.N_sim:
            lines.append(f'  WARNING: {name} {size} px + 2 x {sp} px spread = {size + 2 * sp} px > '
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
# 'lab' = the real devices (two 1920 x 1080 SLMs: Holoeye PLUTO 8 um and LUNA 4.5 um; Basler 5.86 um camera snapped to 6 um).
# key_slm_dx = pitch of the SLM showing the phase keys, enc_slm_dx = pitch of the SLM showing encoding + decoder (and the
# layers at K > 0); since 2026-10-25 the two can differ (lab default since 2026-10-26: keys on the LUNA, encoding on the
# PLUTO, handover 4as; keys on the PLUTO: --set key_slm_dx=8e-6 enc_slm_dx=4.5e-6). Every SLM pixel is rasterised onto the sim grid (sim_dx) at its exact
# physical size, area-weighted where a sim px straddles a pixel edge (model._AxisRaster); sim_dx must divide the camera px.
# camera_dx None -> camera pixel = sim pixel. N_sim 'auto' -> sized from the geometry + light spread.
# encoding_patch_scale / encoding_gap_blocks are in SLM px / bin-sized blocks. phase_key_size (SLM px) None ->
# derived (key = layer size, or 9 x patch at K = 0); a number fixes the key (the decoder covers the key's physical width).
# slm_px_max: SLM pixel count along its short side (key and decoder must fit; None = no limit).
# Lab defaults locked 2026-10-17: 24 um encoding px with 72 um gaps (now 3 SLM px + 9 SLM px gaps at the encoding pitch),
# 400 SLM px key, 12 um detectors on a 24 um pitch (one detector width dark), 3 mm between planes; sim px 2 um until
# 2026-10-25, 1 um since (PLUTO 8 px, LUNA 4.5 px, camera 6 px; 1.5 um put the PLUTO edges on 5.33 px: a moire,
# 3-6% readout error vs 0.5 um, handover 4aq). Sim (re-locked 2026-10-20, logs/sim_geom_gap_pitch):
# bin 6, gap 0 since 2026-10-25 (handover 4am: with the opaque background, larger encoding pixels pass more light;
# Np64 K2 M3 1.6e-2 -> 3.9e-7; the Np64 3x3 block is 144 px; bin 2 / gap 2 before, bin 3 / gap 2 until 10-22),
# 3 px detectors on a 5 px pitch (158 px array at Nf 1024, ~80% of the K4 layer; 2026-10-21, handover 4yy).
# Earlier: gap 0 / pitch 3 (until 10-20), pitch 6 (10-20 to 10-21; 93% of the K4 layer, too wide).
# decoder_per_key (only used with decoder = True): lab True since 2026-10-21 -- one decoder per frame, shown on the SLM
# with key m at no hardware cost; sim keeps one shared decoder.
# Every profile must list the same keys: init_params applies 'sim' first, so a key only one profile sets leaks into the other.
DEVICE_PROFILES = {
    'sim': dict(key_slm_dx=300 * 1e-9, enc_slm_dx=300 * 1e-9, sim_dx=300 * 1e-9,   # = the old 300 * nm, bit-identical
                slm_px_max=None, camera_dx=None, pd_size_cam=3, pd_pitch_cam=5,
                encoding_patch_scale=6, encoding_gap_blocks=0, phase_key_size=None,
                key_to_enc_spacing=12e-6, slm_first_layer_spacing=12e-6,     # 12 um since 2026-10-25 (6 um before;
                interlayer_spacing=12e-6, last_layer_ccd_spacing=12e-6,      # Np64 K2 M1 3.2e-3 -> 5.5e-7, handover 4ao)
                N_sim='auto', decoder_per_key=False,                         # auto: 512 at Np64 Nf1024 K2, 448 at K3/K4
                encoding_zero_based=False),   # symmetric display: the layers act as the PSF coefficient bank (theory)
    'lab': dict(key_slm_dx=4.5e-6, enc_slm_dx=8e-6, sim_dx=1e-6,     # keys LUNA, encoding + decoder PLUTO since 2026-10-26 (4as)
                slm_px_max=1080, camera_dx=6e-6, pd_size_cam=2, pd_pitch_cam=4,
                encoding_patch_scale=11, encoding_gap_blocks=0, phase_key_size=400,   # gap 0 since 2026-10-26 (4ar)
                key_to_enc_spacing=3e-3, slm_first_layer_spacing=3e-3,
                interlayer_spacing=3e-3, last_layer_ccd_spacing=3e-3, N_sim='auto', decoder_per_key=True,
                encoding_zero_based=True),    # asymmetric display since 2026-10-26: E+D (K = 0, transparent) gains from it (4ac/4ae)
}

def _int_ratio(a, b, what):
    '''a / b as an int, raising if it is not (within 1e-6) a whole number.'''
    q = a / b
    if q < 1 - 1e-6 or abs(q - round(q)) > 1e-6:
        raise ValueError(f'{what}: {a:.4g} / {b:.4g} = {q:.4f} is not a whole number')
    return int(round(q))

def sim_ratio(pitch, sim_dx, what):
    '''Sim px per device px (float; snapped to an int when within 1e-6 of one). sim_dx may not exceed the pitch.'''
    q = pitch / sim_dx
    if q < 1 - 1e-6:
        raise ValueError(f'{what}: pitch {1e6 * pitch:.4g} um < sim_dx {1e6 * sim_dx:.4g} um')
    return int(round(q)) if abs(q - round(q)) < 1e-6 else q

def footprint(n_px, ratio):
    '''Sim px spanned by n_px device px of `ratio` sim px each (whole sim px, rounded up).'''
    return int(np.ceil(n_px * ratio - 1e-6))

def spread_sine(tc, pitch):
    '''Sine of the widest propagation angle that carries real light: the grid's own limit, or
    guard_orders diffraction orders of one pixel of the source SLM (pitch), whichever is smaller.'''
    return min(tc.wavelength / (2 * tc.sim_dx), tc.guard_orders * tc.wavelength / pitch, 0.999)

def _plane_gaps(tc):
    '''Propagation distances actually used (no layer gaps at K = 0).'''
    return [z for z, _ in _plane_legs(tc)]

def _plane_legs(tc):
    '''(distance, pitch of the SLM the light leaves) for every propagation: key -> encoding plane leaves the key SLM,
    every later leg leaves the encoding SLM or a layer (layers sit at the encoding pitch).'''
    if tc.num_layers == 0:
        return [(tc.key_to_enc_spacing, tc.key_slm_dx), (tc.last_layer_ccd_spacing, tc.enc_slm_dx)]
    return [(tc.key_to_enc_spacing, tc.key_slm_dx), (tc.slm_first_layer_spacing, tc.enc_slm_dx),
            (tc.interlayer_spacing, tc.enc_slm_dx), (tc.last_layer_ccd_spacing, tc.enc_slm_dx)]

def spread_px(tc, z, pitch):
    '''Lateral spread in sim px of light travelling z at the spread_sine angle.'''
    s = spread_sine(tc, pitch)
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

    # SLM pitches -> sim px per SLM px (need not be whole: the model rasterises pixels area-weighted)
    tc.key_ratio = sim_ratio(tc.key_slm_dx, tc.sim_dx, 'key_slm_dx')
    tc.enc_ratio = sim_ratio(tc.enc_slm_dx, tc.sim_dx, 'enc_slm_dx')   # encoding + decoder, and the layers

    # Camera pitch -> sim pitch; detector size / pitch are whole camera pixels (metre values derived for display)
    if tc.camera_dx is None:
        tc.camera_dx = tc.sim_dx
    tc.cam_bin = _int_ratio(tc.camera_dx, tc.sim_dx, 'camera_dx / sim_dx')
    tc.photodiode_pixels = tc.pd_size_cam * tc.cam_bin
    tc.pd_row_spacing_px = tc.pd_col_spacing_px = tc.pd_pitch_cam * tc.cam_bin
    tc.photodiode_size = tc.photodiode_pixels * tc.sim_dx
    tc.pd_row_spacing = tc.pd_col_spacing = tc.pd_row_spacing_px * tc.sim_dx

    tc.encoding_mode = getattr(tc, 'encoding_mode', 'constant') # const vs split
    assert tc.encoding_mode in ('constant', 'split'), f'unknown encoding_mode {tc.encoding_mode!r}'
    tc.N_alpha = tc.Np * (tc.M if tc.encoding_mode == 'split' else 1)
    # harmonic i (0-based, i < N_alpha) has frequency alpha_i = (i - harmonic_shift) * encoding_freq_step
    tc.harmonic_shift = tc.N_alpha // 2 if getattr(tc, 'centered_harmonics', False) else 0
    # Displayed encoding frequencies: alpha_p = p - encoding_shift. encoding_zero_based shows p = 0 .. N_alpha-1 whatever the
    # target description; the targets are unchanged (|.|^2 removes a common frequency offset)
    tc.encoding_shift = 0 if getattr(tc, 'encoding_zero_based', False) else tc.harmonic_shift

    _derive(tc, 'N_trainable_features', int(np.ceil( tc.r * 2 * tc.N_alpha * tc.Nf * (tc.M if tc.scale_layer_with_M and tc.encoding_mode == 'constant' else 1))))
    _derive(tc, 'layer_size', int(np.ceil(np.sqrt(tc.N_trainable_features / tc.num_layers))) if tc.num_layers > 0 else 0)

    # detector grid: always sqrt(Nf) x sqrt(Nf), one detector per target function (Nf must be a perfect square)
    side = int(np.round(np.sqrt(tc.Nf)))
    if side * side != tc.Nf:
        raise ValueError(f'Nf={tc.Nf} is not a perfect square (detector grid is sqrt(Nf) x sqrt(Nf))')
    tc.pd_num_rows = tc.pd_num_cols = side

    # ---- pure geometry: always recomputed -----------------------------------
    tc.layer_size_sim = footprint(tc.layer_size, tc.enc_ratio)   # layers sit on encoding-SLM-pitch pixels

    # ---- new NFA (function-approximation) derived params ------------------
    # Encoding-plane geometry: the patch's pixels arranged as a rows x cols grid, chooses factorization closest to square
    tc.encoding_dx        = tc.enc_slm_dx * tc.encoding_patch_scale   # encoding_patch_scale = SLM px per encoding px
    tc.encoding_bin       = tc.encoding_patch_scale * tc.enc_ratio    # sim px per encoding px (display only; may be fractional)
    # patch pixels: always Np (split mode shows a new Np-pixel block on the same patch every frame)
    n_pix = tc.Np
    tc.encoding_rows = max(d for d in range(1, int(np.sqrt(n_pix)) + 1) if n_pix % d == 0) # computes the optimal factorization for rows x cols (largest integer # of rows < sqrt(n_pix) )
    tc.encoding_cols = n_pix // tc.encoding_rows

    # One patch copy in encoding-SLM px (gaps included); copies sit one patch + one gap apart (gap 0 -> touching)
    _s, _g = tc.encoding_patch_scale, tc.encoding_gap_blocks
    tc.encoding_y_num_px = _s * (tc.encoding_rows + (tc.encoding_rows - 1) * _g)
    tc.encoding_x_num_px = _s * (tc.encoding_cols + (tc.encoding_cols - 1) * _g)
    _cells = encoding_tile_cells(getattr(tc, 'encoding_tile_pattern', 'single')) # list of (row, col) offsets (3x3: {-1, 0, 1}, 5x5: {-2..2}) where a patch copy sits
    _ry = max(abs(r) for r, _ in _cells); _rx = max(abs(c) for _, c in _cells) # furthest copy from the centre along each axis
    tc.encoding_extent_px = max(tc.encoding_y_num_px + 2 * _ry * (tc.encoding_y_num_px + _s * _g),
                                tc.encoding_x_num_px + 2 * _rx * (tc.encoding_x_num_px + _s * _g))
    # the same in sim px (whole sim px spanned)
    tc.encoding_y_num_sim  = footprint(tc.encoding_y_num_px, tc.enc_ratio)
    tc.encoding_x_num_sim  = footprint(tc.encoding_x_num_px, tc.enc_ratio)
    tc.encoding_extent_sim = footprint(tc.encoding_extent_px, tc.enc_ratio)

    # Phase key. Default width = one layer (layer_size_sim). phase_key_patch_multiple = X
    # instead sizes it as X x the encoding patch width
    # A device profile can fix the key size (lab: 400 SLM px); phase_key_patch_multiple still overrides it
    profile_key = DEVICE_PROFILES[tc.device_profile].get('phase_key_size')
    # K = 0: there is no layer size to copy, so the key defaults to 9 x the encoding patch width
    # (~3 x the full 3x3 tile block, enough to modulate every patch copy) unless set explicitly
    if profile_key is None and tc.num_layers == 0 and getattr(tc, 'phase_key_patch_multiple', None) is None:
        tc.phase_key_patch_multiple = 9
    mult = getattr(tc, 'phase_key_patch_multiple', None)
    # key_det_fill = f (2026-10-26, for large Nf): key width = detector-array width / f, so the detectors fill a fraction
    # f of the key (and of the decoder, which covers the key) -- the E+D analogue of the sim's detector array at ~80% of
    # the last layer (4ab / 4yy). Overrides the profile key and phase_key_patch_multiple.
    fill = getattr(tc, 'key_det_fill', None)
    if fill:
        det_span = (tc.pd_num_rows - 1) * tc.pd_row_spacing_px + tc.photodiode_pixels
        n_key = int(np.ceil(det_span / float(fill) / tc.key_ratio - 1e-6))
    elif mult is None and profile_key is not None:
        n_key = int(profile_key)
    else:
        key_width_sim = tc.layer_size_sim if mult is None else round(mult * tc.encoding_x_num_sim)
        n_key = int(np.ceil(key_width_sim / tc.key_ratio - 1e-6))      # key px covering that width
    _derive(tc, 'phase_key_size', n_key)
    tc.phase_key_size_sim = footprint(tc.phase_key_size, tc.key_ratio)

    # Decoder (E+D architecture, model.py): learnable phase on the encoding SLM (every pixel of its square with
    # decoder_over_encoding, the default; else only the non-encoding pixels), decoder_size SLM px per side (None -> covers the key's physical width, capped at slm_px_max)
    tc.decoder = bool(getattr(tc, 'decoder', False))
    if tc.decoder:
        if getattr(tc, 'decoder_size', None):
            tc.decoder_size_px = int(tc.decoder_size)
        else:
            tc.decoder_size_px = int(round(tc.phase_key_size * tc.key_slm_dx / tc.enc_slm_dx))
            if tc.slm_px_max:
                tc.decoder_size_px = min(tc.decoder_size_px, int(tc.slm_px_max))
        tc.decoder_size_sim = footprint(tc.decoder_size_px, tc.enc_ratio)
    if tc.slm_px_max:
        for name, n in (('phase_key_size', tc.phase_key_size), ('decoder', tc.decoder_size_px if tc.decoder else 0)):
            if n > int(tc.slm_px_max):
                raise ValueError(f'{name} {n} px > slm_px_max {tc.slm_px_max} (the SLM short side)')

    assert tc.pd_num_rows * tc.pd_num_cols == tc.Nf, (
        'photodiode array size (pd_num_rows * pd_num_cols) must equal Nf '
        '-- one detector per target function'
    )

    # SLM phase bit depth: 'inf' = continuous phase (default, all runs before 2026-10-21); an int b = 2^b levels.
    # --set passes a string ('8'), so normalise here.
    bd = getattr(tc, 'slm_phase_bits', 'inf')
    if isinstance(bd, str) and bd.strip().lower() in ('inf', 'none', ''):
        tc.slm_phase_bits = 'inf'
    else:
        tc.slm_phase_bits = int(bd)
        if tc.slm_phase_bits < 1:
            raise ValueError(f"slm_phase_bits must be 'inf' or an integer >= 1, got {bd!r}")

    # Background opacity: 'auto' (default) = opaque when K > 0, transparent at K = 0; true / false force it (controls).
    # --set passes a string, so normalise here; tc.opaque_background is the resolved bool the model uses.
    bo = str(getattr(tc, 'background_opaque', 'auto')).strip().lower()
    if bo not in ('auto', 'true', 'false'):
        raise ValueError(f"background_opaque must be 'auto', 'true' or 'false', got {bo!r}")
    tc.background_opaque = bo
    tc.opaque_background = (tc.num_layers > 0) if bo == 'auto' else (bo == 'true')

    # N_sim 'auto': largest plane + light spread over the longest gap on both sides, rounded up to 64
    if tc.N_sim == 'auto':
        det_span = (tc.pd_num_rows - 1) * tc.pd_row_spacing_px + tc.photodiode_pixels
        biggest = max(tc.phase_key_size_sim, tc.layer_size_sim, tc.encoding_extent_sim, det_span,
                      tc.decoder_size_sim if tc.decoder else 0)
        need = biggest + 2 * max(spread_px(tc, z, p) for z, p in _plane_legs(tc))
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
    # 'sim' (300 nm SLMs = sim pixel, 12 um gaps, N_sim auto) or 'lab' (keys on the 4.5 um LUNA, encoding + decoder on
    # the 8 um PLUTO, 1 um sim px, 6 um camera, 3 mm gaps, N_sim auto). The profile fills key_slm_dx, enc_slm_dx,
    # sim_dx, slm_px_max, camera_dx, pd_size_cam, pd_pitch_cam, the encoding
    # bin / gap, the key size (lab), the four spacings, N_sim and decoder_per_key -- see
    # DEVICE_PROFILES. Any of them can still be --set individually.
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

    tc.encoding_freq_step = 1     # phi_in(p;a) = 2*pi*encoding_freq_step*alpha_p*a
    tc.centered_harmonics = True  # alpha = -N_alpha//2 .. N_alpha - 1 - N_alpha//2 (symmetric, as in the earlier papers; default
                                  # since 2026-10-20); False = 0 .. N_alpha-1 (every run before that date)
    tc.encoding_zero_based = False  # True: the SLM shows alpha_p = 0 .. N_alpha-1 (p = 0 .. N_alpha-1) even with centered targets.
                                    # Identical targets; puts the a-independent background (frequency 0) at the band edge (2026-10-22).
                                    # Only matters at K = 0: at K > 0 the background is opaque and both displays give identical results (4af)
    # Background (non-encoding pixels of the encoding plane), since 2026-10-23: 'auto' = OPAQUE when K > 0 (the layers are
    # the coefficient bank; no a-independent DC term), transparent at K = 0 (lab E+D needs the background light).
    # 'true' / 'false' force it -- only for controls (added 2026-10-24 for the transparent-background check)
    tc.background_opaque = 'auto'
    # encoding_patch_scale (binning: each p value takes up bin x bin SLM pixels) and encoding_gap_blocks (zero-phase
    # bin-sized blocks between adjacent encoding pixels) come from the device profile (sim: 6 / 0, lab: 11 / 0), in SLM px of the encoding SLM.
    # Harmonic p sits at row p // cols, column p % cols of the rows x cols patch (p = 0 top-left).

    tc.encoding_tile_pattern = 'full'   #  copies the patch to a 3x3 grid around the centre, 'single' = one patch
    tc.encoding_mode = 'constant'       # how harmonics are spread over the M keys (constant: all keys see all Np harmonics; split: key m sees only block of harmonics Np = m*Np .. (m+1)*Np-1)
                                        # const: N_alpha = Np; split: N_alpha = Np*M and N = r*2*N_alpha*Nf grows with M
    tc.func_seed = 0                    # random seed for generating Nf target functinons

    # ------------------------------------------------------------------ #
    #  Diffractive Layers                                                #
    # ------------------------------------------------------------------ #
    tc.M             = 1   # number of phase keys
    tc.num_layers    = 2   
    tc.scale_layer_with_M = False  # N = r * 2*Np*Nf*M  (True) or N = r * 2*Np*Nf (False)
    # layers sit on SLM pixels at the encoding pitch (enc_slm_dx)

    # ------------------------------------------------------------------ #
    #  Phase-key plane                                                   #   
    # ------------------------------------------------------------------ #
    tc.phase_key_patch_multiple = None  # None: key = layer size; X: key = X x encoding patch width
    tc.key_det_fill = None              # f: key (and decoder) width = detector-array width / f, e.g. 0.75 (overrides both above)

    tc.slm_hw_x      = 1920  # key SLM pixel count: its full area is illuminated (flat phase outside the key), clipped to N_sim
    tc.slm_hw_y      = 1080  

    # ------------------------------------------------------------------ #
    #  Decoder at the encoding plane (E+D architecture)                  #
    # ------------------------------------------------------------------ #
    tc.decoder      = False   # True: train a phase mask on every non-encoding SLM pixel of the encoding plane
    tc.decoder_size = None    # encoding-SLM px per side; None -> the key's physical width (= phase_key_size at equal pitches), capped at slm_px_max
    tc.decoder_over_encoding = True   # True (default since 2026-10-18): decoder also trains on the encoding pixels (every pixel of its square; superset of False)
    # decoder_per_key comes from the device profile (lab True: M decoders, decoder m shown with key m; sim False: one shared)

    # SLM phase quantisation (2026-10-21): 'inf' = continuous phase (default, as before); an int b = 2^b equally spaced
    # levels in [0, 2pi) for every SLM-displayed phase (keys, encoding + decoder plane, layers). Trained with a
    # straight-through estimator (forward quantised, gradient as if continuous). E.g. --set slm_phase_bits=8
    tc.slm_phase_bits = 'inf'

    tc.mask_init_method = 'normal'
    tc.mask_init_std   = 1.0   # std of the phase keys' normal init

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
    # pd_col_spacing in metres are derived for display only.

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
    tc.lr_min_ratio = 1e-3 # cosine schedules anneal each lr to lr * lr_min_ratio (1e-3: 1e-2 -> 1e-5, the value used before 2026-10-23)
    tc.meas_noise_std = 0.0  # End-to-end measurement-noise injection (SLM + decoder co-adapt to the sim->real gap).
    tc.phase_slm_noise_std = 0.0  # training only: Gaussian phase noise stdev (rad) on every displayed SLM px (keys, encoding + decoder,
                                # layers), fresh each forward pass, added after slm_phase_bits rounding (8-bit rounding ~ 2pi/256/sqrt(12)
                                # = 0.0071 rad rms); val / test noiseless. Absent in configs before 2026-10-29 -> 0
    tc.freeze_slm = False 
    tc.loss_type = 'mse'    # readout is min-max normalisation (loss.py, paper Eq. 9; since 2026-10-23) for every reported number
    tc.train_readout = 'affine'   # training loss only: 'affine' (default since 2026-10-24: learned global scale + offset, discarded
                                  # after training), 'globalminmax' (the test's min-max with Pmin / Pmax from the full a-grid,
                                  # re-measured every epoch, constant within it) or 'minmax' (per-batch; 20-680x worse at Np36,
                                  # logs/Np36_M123_ep400_*). Every
                                  # val / test number is global min-max either way

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
