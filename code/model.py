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
        self.encoding_side      = config.encoding_side
        self.encoding_x_num_sim = config.encoding_x_num_sim
        self.encoding_freq_step = float(getattr(config, 'encoding_freq_step', 1))
        self.encoding_opaque_background = bool(getattr(config, 'encoding_opaque_background', False))
        self.encoding_gap_blocks = int(getattr(config, 'encoding_gap_blocks', 0))

        # Photodiode array (pd_num_rows x pd_num_cols detectors, replacing the single
        # photodiode). The array is centered on (N_sim/2 + detector_offset_y/x): with
        # the default offset of 0 it's centered on the optical axis -- for the default
        # 4x5 array that puts the axis exactly on the middle column (col 2) and exactly
        # between the two middle rows (rows 1, 2), i.e. between r2c3 and r3c3 in 1-indexed
        # terms (up to an unavoidable sub-pixel rounding when rows*spacing is odd).
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
        assert self.encoding_x_num_sim <= self.N_sim
        half = self.pd_px // 2
        assert min(self.pd_centers_y) - half >= 0 and max(self.pd_centers_y) + half <= self.N_sim, \
            'Photodiode array (rows) extends outside the simulation grid'
        assert min(self.pd_centers_x) - half >= 0 and max(self.pd_centers_x) + half <= self.N_sim, \
            'Photodiode array (cols) extends outside the simulation grid'

        # Learnable SLM phase masks (total masks = M * C == T)
        slm_init = torch.zeros(self.T, 1, config.phase_key_size, config.phase_key_size)
        if config.mask_init_method == 'normal':
            std = getattr(config, 'mask_init_std', 0.5)
            nn.init.normal_(slm_init, mean=0.0, std=std)
        self.slm_phases = nn.Parameter(slm_init)

        # Which part of the phase keys trains (config.key_train_region):
        #   'all'               : whole key.
        #   'outside_footprint' : (or key_mask_encoder_footprint=True) the encoder footprint
        #                          is pinned to zero phase 
        #   'footprint_only'    : only the footprint trains; a gradient hook zeroes the
        #                         gradient elsewhere (Adam then leaves those pixels untouched).
        self.key_train_region = getattr(config, 'key_train_region', 'all')
        if self.key_train_region == 'all' and bool(getattr(config, 'key_mask_encoder_footprint', False)):
            self.key_train_region = 'outside_footprint'
        assert self.key_train_region in ('all', 'outside_footprint', 'footprint_only'), \
            f'unknown key_train_region {self.key_train_region!r}'

        footprint = torch.zeros(1, 1, config.phase_key_size, config.phase_key_size)
        if self.key_train_region != 'all':
            assert self.phase_key_bin == 1, 'key_train_region != all requires phase_key_bin == 1'
        if self.phase_key_bin == 1:
            key_off = (self.N_sim - self.phase_key_size_sim) // 2      # same centering as _embed_in_sim
            e0 = (self.N_sim - self.encoding_x_num_sim) // 2
            e1 = e0 + self.encoding_x_num_sim
            k0, k1 = max(e0 - key_off, 0), min(e1 - key_off, self.phase_key_size_sim)
            footprint[..., k0:k1, k0:k1] = 1.0
        self.register_buffer('key_footprint', footprint, persistent=False)
        self.register_buffer('key_train_mask', self._region_mask())

        # Constant phase subtracted in key_phases(). Zero except in a footprint_only
        # fine-tune, where start_footprint_finetune() sets it to pi on the footprint.
        self.register_buffer('key_phase_offset', torch.zeros_like(footprint))

        if self.key_train_region == 'footprint_only':
            self.slm_phases.register_hook(lambda g: g * self.key_footprint)

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
        enc_ap = torch.ones(1, 1, self.N_sim, self.N_sim)
        if self.encoding_opaque_background:
            enc_ap.zero_()
            side, gap = self.encoding_side, self.encoding_gap_blocks
            cell_side = side + (side - 1) * gap
            if gap > 0:
                cell_mask = torch.zeros(1, 1, cell_side, cell_side)
                idx = torch.arange(side) * (1 + gap)
                cell_mask[..., idx[:, None], idx[None, :]] = 1.0
                if self.encoding_bin != 1:
                    cell_mask = F.interpolate(
                        cell_mask,
                        size=(self.encoding_x_num_sim, self.encoding_x_num_sim),
                        mode='nearest'
                    )
            else:
                cell_mask = torch.ones(1, 1, self.encoding_x_num_sim, self.encoding_x_num_sim)
            ey0 = (self.N_sim - self.encoding_x_num_sim) // 2
            ey1 = ey0 + self.encoding_x_num_sim
            enc_ap[..., ey0:ey1, ey0:ey1] = cell_mask
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
        return torch.sigmoid(self.slm_phases) * (2 * np.pi) * self.key_train_mask - self.key_phase_offset

    def _region_mask(self):
        # 0 where the key is pinned to phase 0 (only for 'outside_footprint'), else 1
        if self.key_train_region == 'outside_footprint':
            return 1.0 - self.key_footprint
        return torch.ones_like(self.key_footprint)

    def reset_key_region(self):
        # Re-apply this config's region mask. Needed after loading a checkpoint from a
        # different region (a stage-1 'outside_footprint' checkpoint carries a mask with
        # the footprint zeroed, which would otherwise keep it pinned in stage 2).
        self.key_train_mask.copy_(self._region_mask())

    @torch.no_grad()
    def start_footprint_finetune(self):
        # Stage-2 start from an 'outside_footprint' checkpoint. The footprint was pinned to
        # phase 0 there, but its raw slm_phases are still the untrained random init. Reset
        # them to 0 (sigmoid(0)*2pi = pi, steepest point of the sigmoid) and subtract pi
        # there, so the footprint starts at exactly phase 0 (the model at step 0 equals the
        # stage-1 model) and trains over (-pi, pi). Outside the footprint nothing changes.
        fp = self.key_footprint.expand_as(self.slm_phases).bool()
        self.slm_phases[fp] = 0.0
        self.key_phase_offset.copy_(np.pi * self.key_footprint)

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
        #sums intensity over the size phase key footprint,
        # with |Uin|=1, this returns phase_key_size_sim^2
        footprint = self._embed_in_sim(self.slm_aperture.new_ones(
            1, 1, self.phase_key_size_sim, self.phase_key_size_sim))
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
        # for a given a, computes [2pi(0)a, 2pi(1)a, ..., 2pi(Np-1)a] and transforms into 2d block of size encoding_side x encoding_side
        # if gap != 0, inserts zeros between blocks
        # embeds encoding block into sim, and performs binning if necessary
        B = a.shape[0]
        side = self.encoding_side
        gap  = self.encoding_gap_blocks
        p_idx = torch.arange(side * side, device=a.device, dtype=a.dtype)
        alphas = p_idx * self.encoding_freq_step                     # [Np]
        phi_flat = 2 * np.pi * torch.einsum('b,p->bp', a, alphas)    # outer product--[B, Np], for a given a:[2pi(0)a, 2pi(1)a, ..., 2pi(Np-1)a]
        phi_px = phi_flat.view(B, side, side) # B is the number of a samples (batch size), transforms 1d flat grid into 2d grid of size side x side

        if gap > 0:
            cell_side = side + (side - 1) * gap # size of new block with inserted zeros
            cell_phi = phi_px.new_zeros(B, cell_side, cell_side) 
            idx = torch.arange(side, device=a.device) * (1 + gap) # where to place real pixels
            cell_phi[:, idx[:, None], idx[None, :]] = phi_px # inserts real pixels in the zero array, leaving zeros in between
        else:
            cell_phi = phi_px
        phi_patch = cell_phi.unsqueeze(1)                             # [B, 1, cell_side, cell_side]

        # encoding bin is directly related to the encoding_patch_scale
        # non unity encoding_bin also scales gap sizes
        if self.encoding_bin != 1: 
            phi_patch = F.interpolate(
                phi_patch,
                size=(self.encoding_x_num_sim, self.encoding_x_num_sim),
                mode='nearest'
            )
        return self._embed_in_sim(phi_patch)

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
        
        half = self.pd_px // 2
        rows = []
        for cy in self.pd_centers_y:
            y0 = cy - half
            cols = [
                I_ccd[:, :, y0:y0 + self.pd_px, cx - half:cx - half + self.pd_px].mean(dim=[-2, -1])
                for cx in self.pd_centers_x
            ]
            rows.append(torch.stack(cols, dim=-1))          # [B, M, pd_num_cols]
        return torch.stack(rows, dim=-2)                     # [B, M, pd_num_rows, pd_num_cols]

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
        phi_in_sim  = self._encode_input(a) # encoding plane phases [B, 1, N_sim, N_sim]
        phi_slm_sim = self._get_slm_field() # key plane phases [M, 1, N_sim, N_sim]

        # Phase-key plane -> input plane
        U_key        = self.slm_aperture * torch.exp(1j * phi_slm_sim) # [T, 1, N_sim, N_sim]
        U_at_enc     = self._key_to_encoding(U_key) # [T, 1, N_sim, N_sim] either propagated or copied (z=0) to the input plane
        U_at_enc_exp = U_at_enc[:, 0, :, :].unsqueeze(0) # [1, T, N_sim, N_sim]

        # Input plane perturbation
        U_input = self.encoding_aperture * torch.exp(1j * phi_in_sim)  # [B, 1, N_sim, N_sim]
        U_input_exp = U_input.expand(-1, self.T, -1, -1) # [B, T, N_sim, N_sim]
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