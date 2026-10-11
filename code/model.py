'''
Time-Multiplexed NFA Model (Nonlinear Function Approximation)

Physical path: phase-key plane (learned, T == M keys, time-multiplexed) ->
propagate (key_to_enc_spacing) -> function-input encoding plane
(deterministic, phi_in(p;a) = 2*pi*encoding_freq_step*(p-1)*a over an
Np-pixel patch, NOT learned -- see _encode_input) -> propagate
(slm_first_layer_spacing) -> K learned diffractive layers (shared across all
M keys) -> propagate -> detector plane.

Detector: a pd_num_rows x pd_num_cols photodiode array, one intensity
reading per target function (pd_num_rows * pd_num_cols == config.Nf -- no
more positive/negative differential pairing, that was classification-era,
see loss.py). forward()/measure() return the raw per-detector intensity
array I_vec : [B, T, pd_num_rows, pd_num_cols]; summing over the M (== T)
phase-key axis and normalizing into f_hat(a) is done downstream in loss.py.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from wave_prop import FreeSpaceProp
from config import encoding_tile_cells

# --------------------------------------------------------------------------- #
#  Main model                                                                  #
# --------------------------------------------------------------------------- #

class TimeMultiplexedNFA(nn.Module):

    def __init__(self, config):
        super(TimeMultiplexedNFA, self).__init__()

        self.M = int(config.M)
        self.C = int(getattr(config, 'C', 1))
        self.T = int(getattr(config, 'T', self.M * self.C))
        self.N_sim         = config.N_sim
        self.sim_dx        = config.sim_dx
        # additive measurement noise for noise-robust end-to-end training (0 = off);
        # max fraction of per-sample std(I_vec). See _add_meas_noise / config.meas_noise_std.
        self.meas_noise_std = float(getattr(config, 'meas_noise_std', 0.0))

        # Phase-key geometry (the key is displayed on the SLM, hence slm_dx/slm_hw_*)
        self.phase_key_size     = config.phase_key_size
        self.phase_key_bin      = config.phase_key_bin
        self.phase_key_size_sim = config.phase_key_size_sim
        self.slm_hw_x      = getattr(config, 'slm_hw_x', config.phase_key_size)
        self.slm_hw_y      = getattr(config, 'slm_hw_y', config.phase_key_size)

        # Function-input encoding-plane geometry (deterministic, NOT learned --
        # see _encode_input). Replaces the old "object" (image) plane.
        self.encoding_bin       = config.encoding_bin


        self.encoding_rows      = config.encoding_rows # # of rows in the encoding patch encoding_rows x encoding_cols = Np
        self.encoding_cols      = config.encoding_cols # # of columns in the encoding patch encoding_rows x encoding_cols = Np
        self.encoding_y_num_sim = config.encoding_y_num_sim # height in sim px of one patch copy (including gaps)
        self.encoding_x_num_sim = config.encoding_x_num_sim # width in sim px of one patch copy (including gaps)
        self.encoding_freq_step = float(getattr(config, 'encoding_freq_step', 1))
        self.encoding_opaque_background = bool(getattr(config, 'encoding_opaque_background', False))
        self.encoding_gap_blocks = int(getattr(config, 'encoding_gap_blocks', 0))
        # Harmonics are always placed row-major (raster). The 'reversed' / 'spiral' layouts were
        # removed 2026-10-10; refuse old configs that used them rather than rebuild them wrongly.
        if (getattr(config, 'encoding_order', 'raster') != 'raster'
                or bool(getattr(config, 'encoding_reverse_order', False))):
            raise NotImplementedError('encoding_order / encoding_reverse_order support was removed '
                                      '2026-10-10 (raster layout only)')
        # Encoding mode (config.encoding_mode): 'constant' = all Np harmonics every frame;
        # 'split' = frame m shows harmonics m*Np .. (m+1)*Np-1 only (N_alpha = Np*M).
        self.encoding_mode = getattr(config, 'encoding_mode', 'constant')
        self.split_patch = getattr(config, 'split_patch', 'frame')
        self.Np = int(config.Np)
        self.N_alpha = int(getattr(config, 'N_alpha', self.Np))
        if self.encoding_mode == 'split':
            assert self.T == self.M, "encoding_mode='split' needs T == M (C == 1)"
        # Identical patch copies (config.encoding_tile_pattern): sim-px shifts (dy, dx) of each
        # copy from the centred patch; neighbouring copies are one patch + one gap apart,
        # separately along y (rows) and x (cols) for a rectangular patch.
        self.encoding_tile_pattern = getattr(config, 'encoding_tile_pattern', 'single')
        gap_px = self.encoding_bin * self.encoding_gap_blocks
        pitch_y, pitch_x = self.encoding_y_num_sim + gap_px, self.encoding_x_num_sim + gap_px # seperation between patch copes in sim px
        self.encoding_tile_shifts = [(r * pitch_y, c * pitch_x)
                                     for r, c in encoding_tile_cells(self.encoding_tile_pattern)] # locations of all patch copies in sim px relative to the centred patch
        reach_y = max(abs(dy) for dy, _ in self.encoding_tile_shifts)
        reach_x = max(abs(dx) for _, dx in self.encoding_tile_shifts)
        # extent = larger of the copies' total height / width (used for asserts and figure zooms)
        self.encoding_extent_sim = max(self.encoding_y_num_sim + 2 * reach_y,
                                       self.encoding_x_num_sim + 2 * reach_x) # maximum extent in sim px of all patch copies together in input plane
        assert self.encoding_extent_sim <= self.N_sim, \
            f'encoding copies span {self.encoding_extent_sim} px > N_sim {self.N_sim}'

        # Photodiode array (pd_num_rows x pd_num_cols detectors, replacing the single
        # photodiode). The array is centered on (N_sim/2 + detector_offset_y/x): with
        # the default offset of 0 it's centered on the optical axis
        # A nonzero offset shifts the whole array (e.g. to model real misalignment).
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

        # Sim-grid row / column index of every pixel of every detector, for the vectorised
        # readout in _integrate_photodiode_array: detector r covers rows
        # pd_centers_y[r] - half ... + pd_px - 1 (same window the old per-detector loop
        # sliced), so pd_rows_idx = [rows*pd_px] lists detector 0's pd_px rows, then
        # detector 1's, ... Non-persistent: not saved in checkpoints (rebuilt from config).
        win = torch.arange(self.pd_px) - half # []
        self.register_buffer('pd_rows_idx', (torch.tensor(self.pd_centers_y)[:, None] + win).reshape(-1),
                             persistent=False)
        self.register_buffer('pd_cols_idx', (torch.tensor(self.pd_centers_x)[:, None] + win).reshape(-1),
                             persistent=False)

        # Learnable SLM phase masks (total masks = M * C == T)
        slm_init = torch.zeros(self.T, 1, config.phase_key_size, config.phase_key_size)
        if config.mask_init_method == 'normal':
            std = getattr(config, 'mask_init_std', 0.5)
            nn.init.normal_(slm_init, mean=0.0, std=std)
        self.slm_phases = nn.Parameter(slm_init)

        # The whole key always trains. Partial-key training (key_train_region /
        # key_mask_encoder_footprint, stage-1/stage-2 fine-tunes) was removed 2026-10-09;
        # refuse old checkpoints that used it rather than rebuild them wrongly.
        if (getattr(config, 'key_train_region', 'all') != 'all'
                or bool(getattr(config, 'key_mask_encoder_footprint', False))):
            raise NotImplementedError('key_train_region / key_mask_encoder_footprint support '
                                      'was removed 2026-10-09 (whole key always trains)')

        # Learnable diffractive layers
        self.num_layers      = getattr(config, 'num_layers', 0)
        assert self.num_layers > 0, 'TimeMultiplexedNFA requires num_layers > 0'
        self.layer_bin       = getattr(config, 'layer_bin', 1)
        self.layer_size_sim  = getattr(config, 'layer_size_sim', config.layer_size)
        layers_init = torch.zeros(self.num_layers, 1, config.layer_size, config.layer_size)
        self.layer_phases = nn.Parameter(layers_init)
        assert self.layer_size_sim <= self.N_sim, \
            'Diffractive layer extends outside the simulation grid'

        # SLM aperture (current just an array full of ones)
        ap_h = min(self.slm_hw_y * self.phase_key_bin, self.N_sim)
        ap_w = min(self.slm_hw_x * self.phase_key_bin, self.N_sim)
        aperture = torch.zeros(1, 1, self.N_sim, self.N_sim)
        y0 = (self.N_sim - ap_h) // 2;  y1 = y0 + ap_h
        x0 = (self.N_sim - ap_w) // 2;  x1 = x0 + ap_w
        aperture[..., y0:y1, x0:x1] = 1.0
        self.register_buffer('slm_aperture', aperture)

        # Layer aperture
        layer_ap = torch.zeros(1, 1, self.N_sim, self.N_sim)
        ly0 = (self.N_sim - self.layer_size_sim) // 2
        ly1 = ly0 + self.layer_size_sim
        layer_ap[..., ly0:ly1, ly0:ly1] = 1.0   # square aperture, same centering both axes
        self.register_buffer('layer_aperture', layer_ap)

        # Function-input encoding-plane aperture
        # Transparent background: all ones. Opaque: 1 only on the encoding pixels of every
        # patch copy (same gap / bin / centring / tiling as the phases, via _patch_to_sim).
        if self.encoding_opaque_background:
            enc_ap = self._patch_to_sim(torch.ones(1, 1, self.encoding_rows, self.encoding_cols))
        else:
            enc_ap = torch.ones(1, 1, self.N_sim, self.N_sim)
        self.register_buffer('encoding_aperture', enc_ap)

        # Free-space propagators
        self.prop_key_to_enc = FreeSpaceProp(config, z=config.key_to_enc_spacing)
        self.key_to_enc_exact = (float(config.key_to_enc_spacing) == 0.0
                                 and bool(getattr(config, 'key_to_enc_exact_at_zero', True)))
        self.prop_to_layer1  = FreeSpaceProp(config, z=config.slm_first_layer_spacing)
        self.prop_interlayer = FreeSpaceProp(config, z=config.interlayer_spacing)
        self.prop_to_ccd     = FreeSpaceProp(config, z=config.last_layer_ccd_spacing)

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

    def key_phases(self):
        return torch.sigmoid(self.slm_phases) * (2 * np.pi)

    def _get_slm_field(self):
        #embeds the M learned phase keys into sim grid, and performs binning if necessary
        phi = self.key_phases()
        if self.phase_key_bin != 1:
            phi = F.interpolate(
                phi,
                size=(self.phase_key_size_sim, self.phase_key_size_sim),
                mode='nearest'
            )
        return self._embed_in_sim(phi)

    def incident_energy(self):
        # Sums the incident intensity over the larger of the phase-key and first-layer
        # footprints. The key plane is lit across the whole slm_aperture (outside the key the
        # SLM shows phase 0), so with a key smaller than the layers the first layer collects
        # light from its whole aperture -- a key-only footprint undercounted that and gave
        # efficiencies > 1 (changed 2026-10-07; identical to before whenever key == layer).
        # With |Uin|=1 this returns max(phase_key_size_sim, layer_size_sim)^2.
        side = min(max(self.phase_key_size_sim, self.layer_size_sim), self.N_sim)
        footprint = self._embed_in_sim(self.slm_aperture.new_ones(1, 1, side, side))
        return (footprint * self.slm_aperture).sum()

    def key_intensity_at_encoding(self):
        #returns intensity distribution at the input encoding plane per phase key
        # Returns tensor [T, N_sim, N_sim]
        U = self.slm_aperture * torch.exp(1j * self._get_slm_field())
        return self._key_to_encoding(U)[:, 0].abs().pow(2), self._key_to_encoding(U)[:, 0].angle()

    def _key_to_encoding(self, U_key):
        # Propagate the phase_key to the input encoding plane.
        # If z = 0, no propagator (copy key field to input plane) such that field at input plane is exactly exp(j*(phi_key + phi_encoding))
        return U_key if self.key_to_enc_exact else self.prop_key_to_enc(U_key)

    def _encode_input(self, a):
        # Returns encoding-plane phases [B, F, N_sim, N_sim]: F = 1 in 'constant' mode (every
        # frame shows all Np harmonics, broadcast over the M keys in forward), F = M in 'split'
        # mode (frame m shows harmonics m*Np .. (m+1)*Np-1 only).
        # Harmonic p has phase 2*pi*p*freq_step*a and sits at row p // cols, column p % cols
        # (raster) of the rows x cols patch; _patch_to_sim then adds gaps, bins and embeds.
        B = a.shape[0]
        rows, cols = self.encoding_rows, self.encoding_cols
        n_harm = self.N_alpha if self.encoding_mode == 'split' else rows * cols
        p_idx = torch.arange(n_harm, device=a.device, dtype=a.dtype)
        alphas = p_idx * self.encoding_freq_step                     # [n_harm]
        phi_h = 2 * np.pi * torch.einsum('b,p->bp', a, alphas)       # [B, n_harm]: [2pi(0)a, 2pi(1)a, ...] phase correspodning to each harmonic

        if self.encoding_mode == 'constant':
            phi_flat = phi_h[:, None, :]                             # [B, 1, Np]: same input for every frame
        elif self.split_patch == 'frame':
            # frame m shows its own Np harmonics on the Np-pixel patch
            phi_flat = phi_h.reshape(B, self.M, rows * cols)         # [B, M, Np]
        else:
            # 'full': N_alpha-pixel patch, only frame m's block active, the rest phase 0
            frame_of = torch.arange(n_harm, device=a.device) // self.Np          # frame owning harmonic p
            active = frame_of[None, :] == torch.arange(self.M, device=a.device)[:, None]   # [M, N_alpha]
            phi_flat = phi_h[:, None, :] * active                    # [B, M, N_alpha]
        Fr = phi_flat.shape[1]
        phi_px = phi_flat.reshape(B, Fr, rows, cols)                  # [B, F, rows, cols], raster order
        return self._patch_to_sim(phi_px)

    def _patch_to_sim(self, x):
        # x: [B, F, rows, cols], one value per encoding pixel. Inserts gap zeros between pixels,
        # bins each pixel to encoding_bin x encoding_bin sim px (gaps scale too), centres the
        # patch in the N_sim grid and adds the tile copies. Returns [B, F, N_sim, N_sim].
        rows, cols = x.shape[-2:]
        gap = self.encoding_gap_blocks
        if gap > 0:
            cell = x.new_zeros(*x.shape[:-2], rows + (rows - 1) * gap, cols + (cols - 1) * gap)
            ir = torch.arange(rows, device=x.device) * (1 + gap)      # where the real pixels go
            ic = torch.arange(cols, device=x.device) * (1 + gap)
            cell[..., ir[:, None], ic[None, :]] = x                   # zeros left in between
            x = cell
        if self.encoding_bin != 1:
            x = F.interpolate(x, size=(self.encoding_y_num_sim, self.encoding_x_num_sim), mode='nearest')
        # everything above creates a single patch of size encoding_y_num_sim x encoding_x_num_sim with correct gaps
        # below then embeds the patch in sim and adds the tile copies 
        return self._tile_encoding(self._embed_in_sim(x))

    def _tile_encoding(self, x):
        # x: [..., N_sim, N_sim] with the patch centred. Returns the sum of shifted copies,
        # one per tile in encoding_tile_shifts ('single' -> x unchanged). Copies never
        # overlap (pitch >= patch) and never wrap (extent <= N_sim, asserted in __init__).
        if len(self.encoding_tile_shifts) == 1 and self.encoding_tile_shifts[0] == (0, 0):
            return x
        return sum(torch.roll(x, shifts=s, dims=(-2, -1)) for s in self.encoding_tile_shifts) # shift each copy by (dy, dx) in sim px and sum

    def _get_layer_phase(self, k):
        # returns kth layer's phase, embeds in sim with binning if necessary
        phi = torch.sigmoid(self.layer_phases[k:k+1]) * (2 * np.pi)   # [1, 1, layer_size, layer_size]
        if self.layer_bin != 1:
            phi = F.interpolate(
                phi,
                size=(self.layer_size_sim, self.layer_size_sim),
                mode='nearest'
            )
        return self._embed_in_sim(phi)                                 # [1, 1, N_sim, N_sim]

    def _integrate_photodiode_array(self, I_ccd):
        # Using pre computed detector center locations, averages intensity over each detector's pd_px x pd_px footprint
        #I_ccd : [B, T, N_sim, N_sim] raw sensor plane intensity
        #returns [B, T, pd_num_rows, pd_num_cols]
        # Vectorised (2026-10-06): one gather of all detector pixels, then a per-detector mean.
        # Bit-identical to the old per-detector slicing loop (outputs and gradients), which
        # cost ~0.3-0.7 s per batch at 1024 detectors (Python loop of 1024 slices fwd+bwd).
        B, T = I_ccd.shape[:2]
        win = I_ccd[:, :, self.pd_rows_idx[:, None], self.pd_cols_idx[None, :]]   # [B, T, rows*pd, cols*pd] readout from all detector pixels of interest
        win = win.reshape(B, T, self.pd_num_rows, self.pd_px, self.pd_num_cols, self.pd_px)
        return win.mean(dim=(3, 5))                          # [B, T, pd_num_rows, pd_num_cols] average over each detectors pd_px x pd_px footprint

    def _add_meas_noise(self, I_vec):
        #Adds gaussian noise to each pixel of the photodiode array
        # st. deviation given by self.meas_noise_std * std(I_vec) per sample
        
        if self.training and self.meas_noise_std > 0:
            B    = I_vec.shape[0]
            flat = I_vec.reshape(B, -1)                     # per-sample std over all M x rows x cols
            level = torch.rand(B, 1, device=I_vec.device, dtype=I_vec.dtype)
            sigma = (level * self.meas_noise_std) * flat.std(dim=1, keepdim=True)
            noise = torch.randn_like(flat) * sigma
            I_vec = I_vec + noise.reshape(I_vec.shape)
        return I_vec

    # ---------------------------------------------------------------------- #

    def forward(self, a, return_field=False):
        '''
        a : [B] real tensor of scalar function-input values.
        Returns I_vec : [B, T, pd_num_rows, pd_num_cols]  (T == M, no more countries)
        '''
        B = a.shape[0]
        phi_in_sim  = self._encode_input(a) # encoding plane phases [B, F, N_sim, N_sim], F = 1 (constant) or M (split)
        phi_slm_sim = self._get_slm_field() # key plane phases [M, 1, N_sim, N_sim]

        # Phase-key plane -> input plane
        U_key        = self.slm_aperture * torch.exp(1j * phi_slm_sim) # [T, 1, N_sim, N_sim]
        U_at_enc     = self._key_to_encoding(U_key) # [T, 1, N_sim, N_sim] either propagated or copied (z=0) to the input plane
        U_at_enc_exp = U_at_enc[:, 0, :, :].unsqueeze(0) # [1, T, N_sim, N_sim]

        # Input plane perturbation
        U_input = self.encoding_aperture * torch.exp(1j * phi_in_sim)  # [B, F, N_sim, N_sim]
        U_input_exp = U_input.expand(-1, self.T, -1, -1) # [B, T, N_sim, N_sim]; F = 1 broadcasts, F = M is per frame
        U_flat = (U_at_enc_exp * U_input_exp).reshape(B * self.T, 1, self.N_sim, self.N_sim)

        # Diffractive layers -> detector plane
        field = self.prop_to_layer1(U_flat)
        for k in range(self.num_layers):
            layer_phase = self._get_layer_phase(k)
            field = self.layer_aperture * torch.exp(1j * layer_phase) * field
            if k < self.num_layers - 1:
                field = self.prop_interlayer(field)
            else:
                U_ccd_flat = self.prop_to_ccd(field)

        I_ccd = U_ccd_flat.abs().pow(2)[:, 0].reshape(B, self.T, self.N_sim, self.N_sim)
        I_vec = self._integrate_photodiode_array(I_ccd)
        if return_field:
            return I_vec, I_ccd
        return I_vec