'''
Time-Multiplexed NFA Model (Nonlinear Function Approximation)

Physical path, per frame m = 1..M: plane wave -> phase key m (learned, key SLM) -> propagate (key_to_enc_spacing) ->
encoding plane (encoding SLM): encoding phase phi_in(p; a) = 2*pi*alpha_p*a on the Np-pixel patch (+ copies), alpha_p =
(p - harmonic_shift) * encoding_freq_step, NOT learned (see _encode_input), plus the learned E+D decoder phase (one
shared or one per key) -> propagate -> K learned diffractive layers (shared by all frames; K = 0 in the lab setup) ->
propagate -> camera. Every SLM phase can be quantised to 2^slm_phase_bits levels (_quantize).
At K > 0 the non-encoding pixels of the encoding plane are opaque (only the encoding pixels pass light); at K = 0 they
are transparent.

Pixels -> sim grid (2026-10-25): each SLM plane is a pixel array at its own pitch (key_slm_dx for the keys, enc_slm_dx
for encoding + decoder and the layers). _AxisRaster maps it onto the sim grid at its exact physical size: a sim px that
straddles a pixel edge gets the area-weighted mix of exp(j phi) of the pixels it overlaps. With a whole number of sim
px per SLM px the weights are 0 / 1 and this is the old nearest-neighbour binning, bit for bit.

Detector: a pd_num_rows x pd_num_cols array, one intensity reading per target function (pd_num_rows * pd_num_cols ==
Nf). forward() returns the raw per-detector intensity I_vec : [B, T, pd_num_rows, pd_num_cols] (T == M); averaging
over the M frames and the readout to f_hat(a) (training: config.train_readout; val / test: min-max) happen in loss.py.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from wave_prop import FreeSpaceProp, SpectralLabProp
from config import encoding_tile_cells, footprint

# --------------------------------------------------------------------------- #
#  Pixel array -> sim grid                                                     #
# --------------------------------------------------------------------------- #

class _AxisRaster(nn.Module):
    '''One axis of a pixel array (n_px pixels of `ratio` sim px each, the first starting at sim coordinate `start`,
    in sim px, may be fractional or negative) on an N-cell sim axis. ratio >= 1, so every sim cell overlaps at most two
    pixels: cell i gets w0[i] * px[idx0[i]] + w1[i] * px[idx1[i]] (w = overlap / cell width; 0 outside the array).
    Buffers are not saved in checkpoints: the geometry is rebuilt from the config.'''

    def __init__(self, n_px, ratio, start, N):
        super().__init__()
        assert ratio >= 1 - 1e-9, f'sim px larger than the SLM px (ratio {ratio})'
        i = np.arange(N, dtype=np.float64)
        j0 = np.floor((i - start) / ratio + 1e-9).astype(np.int64)        # pixel under the cell's left edge
        w = []
        for j in (j0, j0 + 1):
            lo = np.maximum(i, start + j * ratio)
            hi = np.minimum(i + 1, start + (j + 1) * ratio)
            ov = np.clip(hi - lo, 0, 1)
            ov[(j < 0) | (j >= n_px)] = 0
            ov[np.abs(ov) < 1e-9] = 0; ov[np.abs(ov - 1) < 1e-9] = 1    # exact 0 / 1 (whole-number ratios: plain binning)
            w.append(ov)
        self.register_buffer('idx0', torch.from_numpy(np.clip(j0, 0, n_px - 1)), persistent=False)
        self.register_buffer('idx1', torch.from_numpy(np.clip(j0 + 1, 0, n_px - 1)), persistent=False)
        self.register_buffer('w0', torch.from_numpy(w[0]).float(), persistent=False)
        self.register_buffer('w1', torch.from_numpy(w[1]).float(), persistent=False)
        self.two = bool((self.w1 > 0).any())                               # False: plain binning, one gather is enough
        self.register_buffer('coverage', self.w0 + self.w1, persistent=False)   # [N] fraction of each cell covered

    def forward(self, x, dim):
        '''x [..., n_px along dim, ...] -> [..., N along dim, ...]'''
        shape = [1] * x.dim(); shape[dim] = -1
        out = x.index_select(dim, self.idx0) * self.w0.view(shape)
        if self.two:
            out = out + x.index_select(dim, self.idx1) * self.w1.view(shape)
        return out


def _raster(x, ry, rx):
    '''Pixel array x [..., ny, nx] (real or complex) -> sim grid [..., N, N], separable (rows, then columns).'''
    return rx(ry(x, -2), -1)


def _raster_phase(phi, ry, rx):
    '''Phase pixel array -> sim-grid phase for FIGURES: angle of the rasterised exp(j phi), so a cell straddling two
    pixels shows the phase of their mixed field (mixing the phase values themselves ignores the 2pi wrap); 0 off the array.'''
    U = _raster(torch.exp(1j * phi), ry, rx)
    cov = ry.coverage[:, None] * rx.coverage[None, :]
    return torch.where(cov > 0, torch.angle(U), torch.zeros_like(U.real))   # angle(-0 - 0j) = -pi off the array


# --------------------------------------------------------------------------- #
#  Main model                                                                  #
# --------------------------------------------------------------------------- #

class TimeMultiplexedNFA(nn.Module):

    def __init__(self, config):
        super(TimeMultiplexedNFA, self).__init__()

        self.M = int(config.M)
        self.T = self.M                                  # one frame per phase key
        self.N_sim         = config.N_sim
        self.sim_dx        = config.sim_dx
        self.meas_noise_std = float(getattr(config, 'meas_noise_std', 0.0))
        self.phase_slm_noise_std = float(getattr(config, 'phase_slm_noise_std', 0.0))   # rad; absent in pre-10-29 configs
        bd = getattr(config, 'slm_phase_bits', 'inf')       # 'inf' = continuous SLM phase, int b = 2^b levels
        self.slm_levels = None if bd == 'inf' else 2 ** int(bd)
        # Measured phase vs grey level (2026-11-11, config.slm_phase_table): level k of an SLM realises
        # sign * table[k] (mod 2 pi) instead of 2 pi k / 2^b; sign -1 = the bench's conjugate drive (TRAINING_ADJUSTMENTS
        # 2026-11-11). Keys use the 'pluto' table, the encoding plane (encoding + decoder) and any layers the 'luna' one.
        # pset_<slm>: the realisable phases, sorted in [0, 2 pi), padded by one wrapped value at each end (for _quantize)
        self.phase_table = getattr(config, 'slm_phase_table', None) or None
        if self.phase_table:
            import json
            assert self.slm_levels is not None, 'slm_phase_table needs an integer slm_phase_bits'
            path = self.phase_table if os.path.isabs(self.phase_table) else \
                os.path.join(os.path.dirname(os.path.abspath(__file__)), self.phase_table)
            tab = json.load(open(path))
            sign = float(getattr(config, 'slm_phase_table_sign', -1))
            for slm in ('pluto', 'luna'):
                t = np.asarray(tab[slm], dtype=np.float64)
                assert len(t) == self.slm_levels, f'{slm} table has {len(t)} levels, slm_phase_bits gives {self.slm_levels}'
                s = np.unique(np.mod(sign * t, 2 * np.pi))
                ext = np.concatenate([[s[-1] - 2 * np.pi], s, [s[0] + 2 * np.pi]])
                self.register_buffer(f'pset_{slm}', torch.from_numpy(ext).float(), persistent=False)

        # Phase-key geometry: key SLM pixels (key_slm_dx = key_ratio sim px each)
        self.phase_key_size     = config.phase_key_size # key SLM px per side
        self.phase_key_size_sim = config.phase_key_size_sim # key sim px per side
        self.key_ratio          = config.key_ratio # key SLM px / sim px
        self.enc_ratio          = config.enc_ratio # enc SLM px / sim px encoding + decoder SLM, and the layers

        # Function-input encoding-plane geometry (encoding SLM px)
        self.encoding_rows      = config.encoding_rows # # of rows in the encoding patch encoding_rows x encoding_cols = Np
        self.encoding_cols      = config.encoding_cols # # of columns in the encoding patch encoding_rows x encoding_cols = Np
        self.encoding_y_num_sim = config.encoding_y_num_sim # height in sim px of one patch copy (including gaps)
        self.encoding_x_num_sim = config.encoding_x_num_sim # width in sim px of one patch copy (including gaps)
        self.encoding_extent_sim = config.encoding_extent_sim # all patch copies together, sim px (asserts, figure zooms)
        self.encoding_freq_step = float(getattr(config, 'encoding_freq_step', 1))
        self.encoding_gap_blocks = int(getattr(config, 'encoding_gap_blocks', 0))
        assert self.encoding_extent_sim <= self.N_sim, \
            f'encoding copies span {self.encoding_extent_sim} px > N_sim {self.N_sim}'

        self.encoding_mode = getattr(config, 'encoding_mode', 'constant') # split encoding vs constant
        self.Np = int(config.Np)
        self.N_alpha = int(getattr(config, 'N_alpha', self.Np))
        self.harmonic_shift = int(getattr(config, 'harmonic_shift', 0))   # targets: alpha_i = (i - harmonic_shift) * freq_step
        # displayed encoding: alpha_p = (p - encoding_shift) * freq_step (= harmonic_shift unless encoding_zero_based)
        self.encoding_shift = int(getattr(config, 'encoding_shift', self.harmonic_shift)) # [-Np//2 ..] or [0 .. Np-1]
        self.encoding_tile_pattern = getattr(config, 'encoding_tile_pattern', 'single') # copies of the input patch placed in a 3x3 or 5x5 grid

        # Photodiode array
        self.pd_px          = config.photodiode_pixels
        self.pd_num_rows    = config.pd_num_rows
        self.pd_num_cols    = config.pd_num_cols
        self.pd_row_spacing = config.pd_row_spacing_px
        self.pd_col_spacing = config.pd_col_spacing_px
        self.det_offset_y   = getattr(config, 'detector_offset_y', 0)
        self.det_offset_x   = getattr(config, 'detector_offset_x', 0)

        # Precompute each detector's center in sim-grid pixel coordinates. Shift the
        # start back by half the array's total span so the array is centered at
        # (N_sim/2 + offset) rather than anchored there at detector (row 0, col 0).
        cy0 = self.N_sim // 2 + self.det_offset_y - ((self.pd_num_rows - 1) * self.pd_row_spacing) // 2
        cx0 = self.N_sim // 2 + self.det_offset_x - ((self.pd_num_cols - 1) * self.pd_col_spacing) // 2
        self.pd_centers_y = [cy0 + r * self.pd_row_spacing for r in range(self.pd_num_rows)]
        self.pd_centers_x = [cx0 + c * self.pd_col_spacing for c in range(self.pd_num_cols)]

        # Sanity checks
        assert self.phase_key_size_sim <= self.N_sim
        assert max(self.encoding_y_num_sim, self.encoding_x_num_sim) <= self.N_sim
        half = self.pd_px // 2
        assert min(self.pd_centers_y) - half >= 0 and max(self.pd_centers_y) + half <= self.N_sim, \
            'Photodiode array (rows) extends outside the simulation grid'
        assert min(self.pd_centers_x) - half >= 0 and max(self.pd_centers_x) + half <= self.N_sim, \
            'Photodiode array (cols) extends outside the simulation grid'

        # Sim-grid row / column index of every pixel of every detector
        win = torch.arange(self.pd_px) - half # []
        self.register_buffer('pd_rows_idx', (torch.tensor(self.pd_centers_y)[:, None] + win).reshape(-1),
                             persistent=False)
        self.register_buffer('pd_cols_idx', (torch.tensor(self.pd_centers_x)[:, None] + win).reshape(-1),
                             persistent=False)

        # Learnable SLM phase keys
        slm_init = torch.zeros(self.T, 1, config.phase_key_size, config.phase_key_size)
        if config.mask_init_method == 'normal':
            std = getattr(config, 'mask_init_std', 0.5)
            nn.init.normal_(slm_init, mean=0.0, std=std)
        self.slm_phases = nn.Parameter(slm_init)

        N = self.N_sim
        centred = lambda n, ratio: (N - footprint(n, ratio)) // 2      # first sim px of a centred n-px array
        outer = lambda r: (r.coverage[:, None] * r.coverage[None, :])[None, None]   # [1, 1, N, N] 2D coverage

        # Learnable diffractive layers (pixels at the encoding-SLM pitch; zero field outside the layer)
        self.num_layers      = getattr(config, 'num_layers', 0)
        self.layer_size_sim  = config.layer_size_sim
        layers_init = torch.zeros(self.num_layers, 1, config.layer_size, config.layer_size)
        self.layer_phases = nn.Parameter(layers_init)
        assert self.layer_size_sim <= self.N_sim, \
            'Diffractive layer extends outside the simulation grid'
        if self.num_layers > 0:
            self.ras_layer = _AxisRaster(config.layer_size, self.enc_ratio, centred(config.layer_size, self.enc_ratio), N)

        # Key SLM: the keys on their pixels; the rest of the illuminated SLM area (slm_hw_y x slm_hw_x px, clipped to the
        # grid) passes flat phase
        self.ras_key = _AxisRaster(self.phase_key_size, self.key_ratio, centred(self.phase_key_size, self.key_ratio), N)
        hw_y, hw_x = int(getattr(config, 'slm_hw_y', self.phase_key_size)), int(getattr(config, 'slm_hw_x', self.phase_key_size))
        ap_y = _AxisRaster(hw_y, self.key_ratio, centred(hw_y, self.key_ratio), N)
        ap_x = _AxisRaster(hw_x, self.key_ratio, centred(hw_x, self.key_ratio), N)
        ap = ap_y.coverage[:, None] * ap_x.coverage[None, :]
        # illum_taper = f > 0 (lab since 2026-10-30): the illuminated SLM is larger than the grid, so the grid edge would
        # be a hard-edged beam edge whose diffraction reaches the camera (11 % readout error at a 2x-key grid); a cosine
        # (Tukey) taper over the outer f / 2 of the grid on each side models a wide smooth beam instead
        # (0.34 % vs the converged estimate, logs/grid_study_2026-10-30/grid_study3*)
        f = float(getattr(config, 'illum_taper', 0.0) or 0.0)
        if f > 0:
            x = torch.arange(N, dtype=torch.float64) + 0.5
            e = f * N / 2
            t = torch.ones(N, dtype=torch.float64)
            t[x < e] = 0.5 * (1 - torch.cos(np.pi * x[x < e] / e))
            t[x > N - e] = 0.5 * (1 - torch.cos(np.pi * (N - x[x > N - e]) / e))
            ap = ap * (t[:, None] * t[None, :]).to(ap.dtype)
        self.register_buffer('slm_aperture', ap[None, None], persistent=False)
        self.register_buffer('key_coverage', outer(self.ras_key), persistent=False)

        # Encoding SLM: one pixel array holds the encoding patch (+ tile copies, all in encoding-SLM px) and the decoder.
        # The centred patch starts at sim px (N - h) // 2; the centred decoder square is shifted by < 1 SLM px so both
        # share one pixel grid (as before 2026-10-25); the array spans both.
        s = int(config.encoding_patch_scale); G = s * self.encoding_gap_blocks      # SLM px per encoding px / per gap
        ph, pw = int(config.encoding_y_num_px), int(config.encoding_x_num_px)       # one patch copy, SLM px
        shifts = [(r * (ph + G), c * (pw + G)) for r, c in encoding_tile_cells(self.encoding_tile_pattern)]
        assert all(float(v).is_integer() for d in shifts for v in d), \
            'even tile grids (half-patch offsets) need an even patch size in SLM px'
        shifts = [(int(dy), int(dx)) for dy, dx in shifts]
        self.encoding_tile_shifts = [(round(dy * self.enc_ratio), round(dx * self.enc_ratio)) for dy, dx in shifts]   # sim px (figures)
        enc0 = ((N - footprint(ph, self.enc_ratio)) // 2, (N - footprint(pw, self.enc_ratio)) // 2)
        lo = [min(d[i] for d in shifts) for i in (0, 1)]                # array bounds, SLM px from the patch origin
        hi = [max(d[i] for d in shifts) + (ph, pw)[i] for i in (0, 1)]

        # Decoder (E+D architecture): a learnable phase mask on the encoding SLM (every pixel of its square with
        # decoder_over_encoding, else only the non-encoding ones). decoder_per_key: one decoder per frame (decoder m shown
        # with key m), else one shared by all M frames. Phase stored directly (no sigmoid), initialised to 0.
        self.decoder = bool(getattr(config, 'decoder', False))
        self.decoder_per_key = self.decoder and bool(getattr(config, 'decoder_per_key', False))
        self.n_decoders = self.M if self.decoder_per_key else 1
        if self.decoder:
            D = self.decoder_size = int(config.decoder_size_px)           # SLM px per side
            self.decoder_size_sim = config.decoder_size_sim
            assert self.decoder_size_sim <= N, 'Decoder extends outside the simulation grid'
            d_org = []
            for i in (0, 1):
                d0 = (N - footprint(D, self.enc_ratio)) // 2
                o = int(np.floor((enc0[i] - d0) / self.enc_ratio + 1e-9))   # patch origin - decoder origin, SLM px
                if enc0[i] - o * self.enc_ratio + D * self.enc_ratio > N + 1e-6:
                    o += 1                                                  # keep the decoder on the grid
                d_org.append(-o); lo[i] = min(lo[i], -o); hi[i] = max(hi[i], -o + D)
        ny, nx = hi[0] - lo[0], hi[1] - lo[1]
        self.enc_px_shape = (ny, nx)
        self.ras_enc_y = _AxisRaster(ny, self.enc_ratio, enc0[0] + lo[0] * self.enc_ratio, N)
        self.ras_enc_x = _AxisRaster(nx, self.enc_ratio, enc0[1] + lo[1] * self.enc_ratio, N)
        self.register_buffer('enc_coverage', self.ras_enc_y.coverage[:, None] * self.ras_enc_x.coverage[None, :],
                             persistent=False)
        # enc_src[y, x] = harmonic p shown by that SLM px (Np = none); harmonic p sits at row p // cols, column p % cols
        src = torch.full((ny, nx), self.Np, dtype=torch.long)
        for dy, dx in shifts:
            for i in range(self.encoding_rows):
                for j in range(self.encoding_cols):
                    y, x = dy - lo[0] + i * (s + G), dx - lo[1] + j * (s + G)
                    src[y:y + s, x:x + s] = i * self.encoding_cols + j
        self.register_buffer('enc_src', src.reshape(-1), persistent=False)
        self.register_buffer('enc_mask_px', (src < self.Np).float(), persistent=False)   # [ny, nx] 1 on encoding px
        if self.decoder:
            self.dec_y0, self.dec_x0 = d_org[0] - lo[0], d_org[1] - lo[1]   # decoder square in the array, SLM px
            dm = torch.zeros(ny, nx)
            dm[self.dec_y0:self.dec_y0 + D, self.dec_x0:self.dec_x0 + D] = 1.0
            if not getattr(config, 'decoder_over_encoding', False):   # True: decoder also adds phase on the encoding pixels
                dm[src < self.Np] = 0.0
            self.register_buffer('decoder_mask_px', dm, persistent=False)
            self.decoder_phases = nn.Parameter(torch.zeros(self.n_decoders, 1, D, D))

        # Background (2026-10-23, fixed): with K > 0 every encoding-plane pixel that is NOT an encoding pixel (in any tile
        # copy) is opaque, so only the encoding pixels pass light and there is no a-independent DC term; the layers do the
        # rest. At K = 0 (lab E+D) the background stays transparent. background_amp (sim grid, figures) is None then.
        self.opaque_background = bool(getattr(config, 'opaque_background', self.num_layers > 0))   # config.background_opaque
        if self.opaque_background:
            if self.decoder and not getattr(config, 'decoder_over_encoding', False):
                raise ValueError('decoder=True with decoder_over_encoding=False at K > 0: the background is opaque, so this '
                                 'decoder would act only on blocked pixels; set decoder_over_encoding=True')
            self.register_buffer('background_amp', _raster(self.enc_mask_px, self.ras_enc_y, self.ras_enc_x)[None, None],
                                 persistent=False)   # [1, 1, N_sim, N_sim], 1 on encoding px
        else:
            self.background_amp = None

        # 2026-11-10: prop_method 'spectral' (K = 0 E+D, transparent background): per-plane spectral propagator
        # (wave_prop.SpectralLabProp) in place of the sim-grid ASM; the sim grid is then only used for figures
        self.prop_method = getattr(config, 'prop_method', 'asm')
        if self.prop_method == 'spectral':
            assert self.num_layers == 0 and not self.opaque_background and self.decoder, \
                "prop_method 'spectral' supports only the K = 0 E+D model with a transparent background"
            dx, ks = self.sim_dx, centred(self.phase_key_size, self.key_ratio)
            n = np.arange(self.pd_num_rows)
            geom = dict(p1=config.key_slm_dx, nk=self.phase_key_size,
                        key_y0=(ks - N / 2) * dx, key_x0=(ks - N / 2) * dx,
                        p2=config.enc_slm_dx, luna_ny=ny, luna_nx=nx,
                        luna_y0=(enc0[0] + lo[0] * self.enc_ratio - N / 2) * dx,
                        luna_x0=(enc0[1] + lo[1] * self.enc_ratio - N / 2) * dx,
                        det_cy=(n - (self.pd_num_rows - 1) / 2) * config.det_pitch_m + self.det_offset_y * dx,
                        det_cx=(np.arange(self.pd_num_cols) - (self.pd_num_cols - 1) / 2) * config.det_pitch_m
                               + self.det_offset_x * dx,
                        det_size=config.det_size_m, z1=config.key_to_enc_spacing, z2=config.last_layer_ccd_spacing)
            adj = {k.strip(): float(v) for k, v in (kv.split('=') for kv in str(getattr(config, 'spec_adjust', '') or '').split(',') if kv.strip())}
            if adj:   # 2026-11-16: fitted bench geometry (config.spec_adjust)
                bad = set(adj) - {'key_dy', 'key_dx', 'luna_dy', 'luna_dx', 'det_dy', 'det_dx', 'cam_mag'}
                assert not bad, f'unknown spec_adjust keys {bad}'
                cm, cdx = 1 + adj.get('cam_mag', 0.0), config.camera_dx
                geom.update(key_y0=geom['key_y0'] + adj.get('key_dy', 0.0) * geom['p1'],
                            key_x0=geom['key_x0'] + adj.get('key_dx', 0.0) * geom['p1'],
                            luna_y0=geom['luna_y0'] + adj.get('luna_dy', 0.0) * geom['p2'],
                            luna_x0=geom['luna_x0'] + adj.get('luna_dx', 0.0) * geom['p2'],
                            det_cy=np.asarray(geom['det_cy']) * cm + adj.get('det_dy', 0.0) * cdx,
                            det_cx=np.asarray(geom['det_cx']) * cm + adj.get('det_dx', 0.0) * cdx,
                            det_size=geom['det_size'] * cm)
            self.luna_fringe = float(getattr(config, 'luna_fringe', 0.0) or 0.0)
            self.spec_geom = geom
            self.spec = SpectralLabProp(config, geom)
            return

        # Free-space propagators
        self.prop_key_to_enc = FreeSpaceProp(config, z=config.key_to_enc_spacing)
        self.key_to_enc_exact = (float(config.key_to_enc_spacing) == 0.0
                                 and bool(getattr(config, 'key_to_enc_exact_at_zero', True)))
        if self.num_layers > 0:
            self.prop_to_layer1  = FreeSpaceProp(config, z=config.slm_first_layer_spacing)
            self.prop_interlayer = FreeSpaceProp(config, z=config.interlayer_spacing)
        # last layer -> camera; at K = 0 this is the encoding plane -> camera distance
        self.prop_to_ccd = FreeSpaceProp(config, z=config.last_layer_ccd_spacing)

    # ---------------------------------------------------------------------- #

    def _embed_in_sim(self, x):
        # zero pad all around x to reach N_sim x N_sim, keeps x centered
        h, w       = x.shape[-2], x.shape[-1]
        pad_h      = self.N_sim - h
        pad_w      = self.N_sim - w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def _quantize(self, phi, slm='luna'):
        # SLM bit depth: wrap to [0, 2pi) and round to slm_levels equally spaced levels. Straight-through estimator:
        # the forward pass sees the quantised phase, the gradient passes as if continuous. No rounding when slm_levels is None.
        # With a measured phase table (slm_phase_table): round to the nearest (on the circle) phase that SLM `slm`
        # ('pluto' keys, 'luna' encoding plane / layers) can realise instead; phases in its unreachable band snap to the
        # nearer end. Same straight-through gradient.
        # Training only (phase_slm_noise_std > 0): fresh Gaussian phase noise (rad) on every displayed SLM px, after the
        # rounding, at any bit depth; val / test / figures see the noiseless phase.
        noise = torch.randn_like(phi) * self.phase_slm_noise_std if self.training and self.phase_slm_noise_std > 0 else 0.0
        if self.slm_levels is None:
            return phi + noise
        if self.phase_table:
            ext = getattr(self, f'pset_{slm}')
            x = torch.remainder(phi.detach(), 2 * np.pi).float().contiguous()
            i = torch.searchsorted(ext, x).clamp(1, len(ext) - 1)
            lo, hi = ext[i - 1], ext[i]
            q = torch.remainder(torch.where(x - lo <= hi - x, lo, hi), 2 * np.pi).to(phi.dtype)
            return phi + (q - phi).detach() + noise
        step = 2 * np.pi / self.slm_levels
        q = torch.remainder(torch.round(torch.remainder(phi, 2 * np.pi) / step), self.slm_levels) * step
        return phi + (q - phi).detach() + noise

    def key_phases(self):
        return self._quantize(torch.sigmoid(self.slm_phases) * (2 * np.pi), 'pluto')

    def _get_slm_field(self):
        # key phases on the sim grid [M, 1, N_sim, N_sim] (figures; 0 outside the key, area-mixed at pixel edges)
        return _raster_phase(self.key_phases(), self.ras_key, self.ras_key)

    def _key_field(self):
        # key plane field [M, 1, N_sim, N_sim]: exp(j key) on the key pixels, flat phase on the rest of the illuminated SLM
        U = _raster(torch.exp(1j * self.key_phases()), self.ras_key, self.ras_key)
        return self.slm_aperture * (U + (1 - self.key_coverage))

    def incident_energy(self):
        # Sums the incident intensity over the larger of the phase-key and first-layer footprints.
        side = min(max(self.phase_key_size_sim, self.layer_size_sim), self.N_sim)
        footprint = self._embed_in_sim(self.slm_aperture.new_ones(1, 1, side, side))
        return (footprint * self.slm_aperture).sum()

    def key_intensity_at_encoding(self):
        # returns intensity distribution at the input encoding plane per phase key
        # Returns tensor [T, N_sim, N_sim]
        U = self._key_to_encoding(self._key_field())[:, 0]
        return U.abs().pow(2), U.angle()

    def _key_to_encoding(self, U_key):
        # Propagate the phase_key to the input encoding plane. If z = 0, no propagator (copy key field to input plane) so input plane field is exp(j*(phi_key + phi_encoding))
        return U_key if self.key_to_enc_exact else self.prop_key_to_enc(U_key)

    def _encode_px(self, a):
        # Encoding phases on the encoding-SLM pixel array [B, F, ny, nx] (0 off the encoding pixels): F = 1 in 'constant'
        # mode (every frame shows all Np harmonics, broadcast over the M keys in forward), F = M in 'split' mode (frame m
        # shows harmonics m*Np .. (m+1)*Np-1 only). Harmonic p has phase 2*pi*alpha_p*a and sits at row p // cols,
        # column p % cols of the rows x cols patch, in every tile copy (enc_src).
        B = a.shape[0]
        rows, cols = self.encoding_rows, self.encoding_cols
        n_harm = self.N_alpha if self.encoding_mode == 'split' else rows * cols
        p_idx = torch.arange(n_harm, device=a.device, dtype=a.dtype) - self.encoding_shift   # centered: -N_alpha//2 ..; else 0 ..
        alphas = p_idx * self.encoding_freq_step                     # [n_harm]
        phi_h = 2 * np.pi * torch.einsum('b,p->bp', a, alphas)       # [B, n_harm]: phase 2*pi*alpha_i*a of each harmonic

        if self.encoding_mode == 'constant':
            phi_flat = phi_h[:, None, :]                             # [B, 1, Np]: same input for every frame
        else:
            # split: frame m shows its own Np harmonics on the same Np-pixel patch
            phi_flat = phi_h.reshape(B, self.M, rows * cols)         # [B, M, Np]
        phi_flat = torch.cat([phi_flat, phi_flat.new_zeros(*phi_flat.shape[:2], 1)], -1)   # index Np = no harmonic
        return phi_flat[..., self.enc_src].reshape(*phi_flat.shape[:2], *self.enc_px_shape)

    def _raster_enc_mask(self):
        return _raster(self.enc_mask_px, self.ras_enc_y, self.ras_enc_x)     # [N, N] encoding-pixel coverage

    def _encode_input(self, a):
        # encoding phases on the sim grid [B, F, N_sim, N_sim] (figures)
        return _raster_phase(self._encode_px(a), self.ras_enc_y, self.ras_enc_x) * (self._raster_enc_mask() > 0)

    def decoder_phases_px(self):
        # Decoder phases on the encoding-SLM pixel array [1, n_decoders, ny, nx] (n_decoders = M per key, else 1), zero
        # outside the decoder square (and on the encoding pixels unless decoder_over_encoding)
        ny, nx = self.enc_px_shape
        D, y0, x0 = self.decoder_size, self.dec_y0, self.dec_x0
        phi = F.pad(self.decoder_phases, (x0, nx - x0 - D, y0, ny - y0 - D))   # [n_decoders, 1, ny, nx]
        return (phi * self.decoder_mask_px).transpose(0, 1)

    def _input_plane_px(self, a):
        # Total phase shown by the encoding SLM [B, F, ny, nx]: encoding phases (+ decoder phases, if enabled).
        # F = 1 (constant, shared decoder) or M (split, and/or one decoder per key): [B, 1 or M] + [1, 1 or M] broadcasts
        phi = self._encode_px(a)
        if self.decoder:
            phi = phi + self.decoder_phases_px()
        return self._quantize(phi)              # the SLM displays encoding + decoder as one quantised pattern

    def _fringe(self, phi):
        # 2026-11-16: LUNA pixel crosstalk (config.luna_fringe w; spectral path only): each displayed phase becomes
        # (1 - 4w) itself + w each of its 4 neighbours (edge pixels replicate). Identity when w = 0.
        w = getattr(self, 'luna_fringe', 0.0)
        if not w:
            return phi
        k = phi.new_tensor([[0, w, 0], [w, 1 - 4 * w, w], [0, w, 0]])
        sh = phi.shape
        return F.conv2d(F.pad(phi.reshape(-1, 1, *sh[-2:]), (1, 1, 1, 1), mode='replicate'), k[None, None]).reshape(sh)

    def input_plane_field(self, a):
        # Encoding-plane modulation on the sim grid [B, F, N_sim, N_sim]. K > 0: opaque background, only the encoding
        # pixels pass light; K = 0: transparent (unit field off the encoding SLM's pixel array)
        E = torch.exp(1j * self._input_plane_px(a))
        if self.opaque_background:
            return _raster(E * self.enc_mask_px, self.ras_enc_y, self.ras_enc_x)
        return _raster(E, self.ras_enc_y, self.ras_enc_x) + (1 - self.enc_coverage)

    def _layer_phase_px(self, k):
        return self._quantize(torch.sigmoid(self.layer_phases[k:k+1]) * (2 * np.pi))   # [1, 1, layer_size, layer_size]

    def _get_layer_phase(self, k):
        # kth layer's phase on the sim grid [1, 1, N_sim, N_sim] (figures)
        return _raster_phase(self._layer_phase_px(k), self.ras_layer, self.ras_layer)

    def _layer_field(self, k):
        # kth layer's transmission on the sim grid [1, 1, N_sim, N_sim] (zero outside the layer)
        return _raster(torch.exp(1j * self._layer_phase_px(k)), self.ras_layer, self.ras_layer)

    def _integrate_photodiode_array(self, I_ccd):
        # Using pre computed detector center locations, averages intensity over each detector's pd_px x pd_px footprint
        #I_ccd : [B, T, N_sim, N_sim] raw sensor plane intensity
        #returns [B, T, pd_num_rows, pd_num_cols]
        B, T = I_ccd.shape[:2]
        win = I_ccd[:, :, self.pd_rows_idx[:, None], self.pd_cols_idx[None, :]]   # [B, T, rows*pd, cols*pd] readout from all detector pixels of interest
        win = win.reshape(B, T, self.pd_num_rows, self.pd_px, self.pd_num_cols, self.pd_px)
        return win.mean(dim=(3, 5))                          # [B, T, pd_num_rows, pd_num_cols] average over each detectors pd_px x pd_px footprint

    def _add_meas_noise(self, I_vec):
        #Adds gaussian noise to each pixel of the photodiode array. StDev given by self.meas_noise_std * std(I_vec) per sample
        if self.training and self.meas_noise_std > 0:
            B    = I_vec.shape[0]
            flat = I_vec.reshape(B, -1)                     # per-sample std over all M x rows x cols
            level = torch.rand(B, 1, device=I_vec.device, dtype=I_vec.dtype)
            sigma = (level * self.meas_noise_std) * flat.std(dim=1, keepdim=True)
            noise = torch.randn_like(flat) * sigma
            I_vec = I_vec + noise.reshape(I_vec.shape)
        return I_vec

    # ---------------------------------------------------------------------- #

    def forward(self, a):
        '''
        a : [B] real tensor of scalar function-input values.
        Returns I_vec : [B, T, pd_num_rows, pd_num_cols]  (T == M phase keys); with meas_noise_std > 0 and in training
        mode, Gaussian detector noise is added (_add_meas_noise).
        '''
        B = a.shape[0]
        if self.prop_method == 'spectral':
            # keys and LUNA on their own pixels (e^{i phi} - 1: the flat surroundings are the analytic plane wave)
            V1, v1s = self.spec.key_field_L(torch.exp(1j * self.key_phases()[:, 0]) - 1)          # [T, L, L] each
            d = torch.exp(1j * self._fringe(self._input_plane_px(a))) - 1                         # [B, F, ny, nx]
            I_vec = self.spec(V1, v1s, d.expand(-1, self.T, -1, -1))
            return self._add_meas_noise(I_vec)
        # Phase-key plane -> input plane
        U_key        = self._key_field() # [T, 1, N_sim, N_sim]
        U_at_enc     = self._key_to_encoding(U_key) # [T, 1, N_sim, N_sim] either propagated or copied (z=0) to the input plane
        U_at_enc_exp = U_at_enc[:, 0, :, :].unsqueeze(0) # [1, T, N_sim, N_sim]

        # Input plane perturbation: encoding (+ decoder) [B, F, N_sim, N_sim], F = 1 (constant, same encoding for all M
        # keys) or M (split and / or per-key decoders); K > 0: opaque background, only encoding px pass
        U_input = self.input_plane_field(a)
        U_input_exp = U_input.expand(-1, self.T, -1, -1) # [B, T, N_sim, N_sim]; F = 1 broadcasts, F = M is per frame
        U_flat = (U_at_enc_exp * U_input_exp).reshape(B * self.T, 1, self.N_sim, self.N_sim)

        # Diffractive layers -> detector plane
        if (self.num_layers == 0):
            U_ccd_flat = self.prop_to_ccd(U_flat)
        else:
            field = self.prop_to_layer1(U_flat)
            for k in range(self.num_layers):
                field = self._layer_field(k) * field
                if k < self.num_layers - 1:
                    field = self.prop_interlayer(field)
                else:
                    U_ccd_flat = self.prop_to_ccd(field)

        I_ccd = U_ccd_flat.abs().pow(2)[:, 0].reshape(B, self.T, self.N_sim, self.N_sim)
        I_vec = self._integrate_photodiode_array(I_ccd)
        return self._add_meas_noise(I_vec)              # no-op unless training with meas_noise_std > 0