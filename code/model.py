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

        # SLM geometry
        self.slm_x_num     = config.slm_x_num
        self.slm_bin       = config.slm_bin
        self.slm_x_num_sim = config.slm_x_num_sim
        self.slm_hw_x      = getattr(config, 'slm_hw_x', config.slm_x_num)
        self.slm_hw_y      = getattr(config, 'slm_hw_y', config.slm_x_num)

        # Function-input encoding-plane geometry (deterministic, NOT learned --
        # see _encode_input). Replaces the old "object" (image) plane.
        self.encoding_bin       = config.encoding_bin
        self.encoding_side      = config.encoding_side
        self.encoding_x_num_sim = config.encoding_x_num_sim
        self.encoding_freq_step = float(getattr(config, 'encoding_freq_step', 1))

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
        assert self.slm_x_num_sim <= self.N_sim
        assert self.encoding_x_num_sim <= self.N_sim
        half = self.pd_px // 2
        assert min(self.pd_centers_y) - half >= 0 and max(self.pd_centers_y) + half <= self.N_sim, \
            'Photodiode array (rows) extends outside the simulation grid'
        assert min(self.pd_centers_x) - half >= 0 and max(self.pd_centers_x) + half <= self.N_sim, \
            'Photodiode array (cols) extends outside the simulation grid'

        # Learnable SLM phase masks (total masks = M * C == T)
        slm_init = torch.zeros(self.T, 1, config.slm_x_num, config.slm_x_num)
        if config.mask_init_method == 'normal':
            std = getattr(config, 'mask_init_std', 0.5)
            nn.init.normal_(slm_init, mean=0.0, std=std)
        self.slm_phases = nn.Parameter(slm_init)

        # Learnable diffractive layers (required -- the no-layers bypass case
        # is no longer supported, see forward()).
        self.num_layers      = getattr(config, 'num_layers', 0)
        assert self.num_layers > 0, 'TimeMultiplexedNFA requires num_layers > 0'
        self.layer_bin       = getattr(config, 'layer_bin', 1)
        self.layer_size_sim  = getattr(config, 'layer_size_sim', config.layer_size)
        layers_init = torch.zeros(self.num_layers, 1, config.layer_size, config.layer_size)
        self.layer_phases = nn.Parameter(layers_init)
        assert self.layer_size_sim <= self.N_sim, \
            'Diffractive layer extends outside the simulation grid'

        # SLM aperture
        ap_h = min(self.slm_hw_y * self.slm_bin, self.N_sim)
        ap_w = min(self.slm_hw_x * self.slm_bin, self.N_sim)
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

        # Free-space propagators (key -> encoding plane, then through the
        # diffractive layers to the detector -- no no-layers bypass anymore)
        self.prop_key_to_enc = FreeSpaceProp(config, z=config.key_to_enc_spacing)
        self.prop_to_layer1  = FreeSpaceProp(config, z=config.slm_first_layer_spacing)
        self.prop_interlayer = FreeSpaceProp(config, z=config.interlayer_spacing)
        self.prop_to_ccd     = FreeSpaceProp(config, z=config.last_layer_ccd_spacing)

    # ---------------------------------------------------------------------- #

    def _embed_in_sim(self, x):
        h, w       = x.shape[-2], x.shape[-1]
        pad_h      = self.N_sim - h
        pad_w      = self.N_sim - w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def _get_slm_field(self):
        phi = torch.sigmoid(self.slm_phases) * (2 * np.pi)
        if self.slm_bin != 1:
            phi = F.interpolate(
                phi,
                size=(self.slm_x_num_sim, self.slm_x_num_sim),
                mode='nearest'
            )
        return self._embed_in_sim(phi)

    def _encode_input(self, a):
        '''
        Deterministic function-input encoding (NOT learned) -- PAPER Sec.
        2.2/4.1: phi_in(p;a) = 2*pi*alpha_p*a, alpha_p = (p-1)*encoding_freq_step,
        over an encoding_side x encoding_side square patch of Np pixels
        (row-major: p - 1 = row*encoding_side + col).

        a : [B] real tensor of scalar input values.
        returns phi_in_sim : [B, 1, N_sim, N_sim] real phase field (radians),
        zero everywhere outside the Np-pixel encoding patch.
        '''
        B = a.shape[0]
        side = self.encoding_side
        p_idx = torch.arange(side * side, device=a.device, dtype=a.dtype)
        alphas = p_idx * self.encoding_freq_step                     # [Np]
        phi_flat = 2 * np.pi * torch.einsum('b,p->bp', a, alphas)    # [B, Np]
        phi_patch = phi_flat.view(B, 1, side, side)

        if self.encoding_bin != 1:
            phi_patch = F.interpolate(
                phi_patch,
                size=(self.encoding_x_num_sim, self.encoding_x_num_sim),
                mode='nearest'
            )
        return self._embed_in_sim(phi_patch)

    def _get_layer_phase(self, k):
        phi = torch.sigmoid(self.layer_phases[k:k+1]) * (2 * np.pi)   # [1, 1, layer_size, layer_size]
        if self.layer_bin != 1:
            phi = F.interpolate(
                phi,
                size=(self.layer_size_sim, self.layer_size_sim),
                mode='nearest'
            )
        return self._embed_in_sim(phi)                                 # [1, 1, N_sim, N_sim]

    def _integrate_photodiode_array(self, I_ccd):
        '''
        Mean intensity captured by each detector in the pd_num_rows x pd_num_cols array.

        I_ccd : [B, T, N_sim, N_sim]
        returns [B, T, pd_num_rows, pd_num_cols]
        '''
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
        """Additive measurement noise for noise-robust end-to-end training.

        Draws a per-sample noise level uniformly from [0, meas_noise_std], then adds Gaussian
        noise with std = level * per-sample std(I_vec). Only active in training mode
        (model.train()); measurement/eval passes stay clean. No multiplicative jitter — additive
        measurement noise only. Set config.meas_noise_std to a chosen maximum, or to the estimate
        printed by experimental_decoder.ipynb (NOISE_ADD, ~0.2).
        """
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

        Returns the raw per-detector measurement array — there is no decoder /
        readout head here. See loss.py for how I_vec is summed over the M
        phase-key axis, normalized, and turned into f_hat(a).

        I_vec : [B, T, pd_num_rows, pd_num_cols]  (T == M, no more countries)
        '''
        B = a.shape[0]
        phi_in_sim  = self._encode_input(a)
        phi_slm_sim = self._get_slm_field()

        # Phase-key plane -> propagate across key_to_enc_spacing to reach the
        # encoding plane. The key field doesn't depend on B, so propagate once
        # (per T) and expand after.
        U_slm        = self.slm_aperture * torch.exp(1j * phi_slm_sim) # [T, 1, N_sim, N_sim]
        U_at_enc     = self.prop_key_to_enc(U_slm) # [T, 1, N_sim, N_sim]
        U_at_enc_exp = U_at_enc[:, 0, :, :].unsqueeze(0) # [1, T, N_sim, N_sim]

        U_in_exp = torch.exp(1j * phi_in_sim).expand(-1, self.T, -1, -1) # [B, T, N_sim, N_sim]
        U_flat = (U_at_enc_exp * U_in_exp).reshape(B * self.T, 1, self.N_sim, self.N_sim)

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