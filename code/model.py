'''
Time-Multiplexed Classifier Model

Detector: a pd_num_rows x pd_num_cols photodiode array (default 4x5 = 20 detectors)
replaces the single-pixel photodiode. There are T = M * C learnable SLM masks total
(M "members" per country, C "countries" -- see loss.py), each still measured
separately (time multiplexing is unchanged) — only the spatial read-out at the
detector plane changed, from one integrated region to a grid of them. forward()/
measure() therefore return the raw per-detector intensity array
I_vec : [B, T, pd_num_rows, pd_num_cols] instead of a decoded image; splitting T
into countries/members and turning that into the 10-way differential class scores
(rows 0..pd_num_rows/2-1 positive, pd_num_rows/2..end negative, paired by column)
is done downstream in loss.py.

NOTE: self.decoder_type / self.output_phase_max below are vestigial from the old
image-reconstruction decoder path and are intentionally left alone here — the owner
of loss.py is removing that path separately.
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

class TimeMultiplexedClassifier(nn.Module):

    def __init__(self, config):
        super(TimeMultiplexedClassifier, self).__init__()

        # Number of learnable masks: M per country, C countries -> total T masks
        self.M = int(config.M)                     # masks per country (members)
        self.C = int(getattr(config, 'C', 1))
        # total number of displayed masks (T = M * C).
        self.T = int(getattr(config, 'T', self.M * self.C))
        self.N             = config.N
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

        # Object geometry
        self.obj_bin       = config.obj_bin
        self.obj_x_num_sim = config.obj_x_num_sim

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
        assert self.obj_x_num_sim <= self.N_sim
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

        # Learnable diffractive layers
        self.num_layers      = getattr(config, 'num_layers', 0)
        self.layer_bin       = getattr(config, 'layer_bin', 1)
        self.layer_size_sim  = getattr(config, 'layer_size_sim', config.layer_size)
        layers_init = torch.zeros(self.num_layers, 1, config.layer_size, config.layer_size)
        self.layer_phases = nn.Parameter(layers_init)
        if self.num_layers > 0:
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

        # Free-space propagator(s)
        self.propagator_no_layers = FreeSpaceProp(config) # straight from slm to ccd (no layers)
        self.prop_obj_to_slm = FreeSpaceProp(config, z=config.object_slm_spacing)
        if self.num_layers > 0:
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

    def _get_obj_field(self, phi_obj):
        if self.obj_bin != 1:
            phi_obj = F.interpolate(
                phi_obj,
                size=(self.obj_x_num_sim, self.obj_x_num_sim),
                mode='bilinear',
                align_corners=False
            )
        return self._embed_in_sim(phi_obj)

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

    @torch.no_grad()
    def diffraction_efficiency(self):
        '''Fraction of each mask's total output power captured by the photodiode array
        (summed over all detectors, averaged over the M masks).'''
        # NOT UPDATED TO NEW TIME-MULTIPLEXED FORMAT
        phi_slm = self._get_slm_field()
        U_slm   = self.slm_aperture * torch.exp(1j * phi_slm)
        U_ccd   = self.propagator_no_layers(U_slm)
        I_ccd   = U_ccd.abs().pow(2)[:, 0]
        total   = I_ccd.sum(dim=[-2, -1])

        half     = self.pd_px // 2
        captured = torch.zeros_like(total)
        for cy in self.pd_centers_y:
            y0 = cy - half
            for cx in self.pd_centers_x:
                x0 = cx - half
                captured = captured + I_ccd[:, y0:y0 + self.pd_px, x0:x0 + self.pd_px].sum(dim=[-2, -1])
        return ((captured / total.clamp(min=1e-12)).mean()).item()

    def measure(self, phi_obj):
        '''
        Returns
        -------
        I_vec : [B, T, pd_num_rows, pd_num_cols]  mean intensity per detector, per SLM mask
        I_ccd : [B, T, N_sim, N_sim]               full detector-plane intensity field
                (T = M * C total masks)
        '''
        B = phi_obj.shape[0]
        phi_obj_sim = self._get_obj_field(phi_obj)
        phi_slm_sim = self._get_slm_field()

        # SLM plane -> propagate across object_slm_spacing to reach the object plane.
        # The SLM field doesn't depend on B, so propagate once (per T) and expand after.
        U_slm        = self.slm_aperture * torch.exp(1j * phi_slm_sim)
        U_at_obj     = self.prop_obj_to_slm(U_slm)
        U_at_obj_exp = U_at_obj[:, 0, :, :].unsqueeze(0)

        U_obj_exp = torch.exp(1j * phi_obj_sim).expand(-1, self.T, -1, -1)
        U_flat = (U_at_obj_exp * U_obj_exp).reshape(B * self.T, 1, self.N_sim, self.N_sim)
        U_ccd_flat = self.propagator_no_layers(U_flat)
        I_ccd = U_ccd_flat.abs().pow(2)[:, 0].reshape(B, self.T, self.N_sim, self.N_sim)
        I_vec = self._integrate_photodiode_array(I_ccd)
        return I_vec, I_ccd

    def forward(self, phi_obj, return_field=False):
        '''
        Returns the raw per-detector measurement array — there is no decoder /
        classification head here. See loss.py for how I_vec is turned into the
        10-way differential class scores.

        I_vec : [B, T, pd_num_rows, pd_num_cols]  (T = M * C total masks)
        '''
        B = phi_obj.shape[0]
        phi_obj_sim = self._get_obj_field(phi_obj)
        phi_slm_sim = self._get_slm_field()

        # SLM plane -> propagate across object_slm_spacing to reach the object plane.
        # The SLM field doesn't depend on B, so propagate once (per T) and expand after.
        U_slm        = self.slm_aperture * torch.exp(1j * phi_slm_sim) # [T, 1, N_sim, N_sim]
        U_at_obj     = self.prop_obj_to_slm(U_slm) # [T, 1, N_sim, N_sim]
        U_at_obj_exp = U_at_obj[:, 0, :, :].unsqueeze(0) # [1, T, N_sim, N_sim]

        U_obj_exp = torch.exp(1j * phi_obj_sim).expand(-1, self.T, -1, -1) # [B, T, N_sim, N_sim]
        U_flat = (U_at_obj_exp * U_obj_exp).reshape(B * self.T, 1, self.N_sim, self.N_sim)

        if self.num_layers > 0:
            # Propagate through the diffractive layers before the CCD.
            field = self.prop_to_layer1(U_flat)

            for k in range(self.num_layers):
                layer_phase = self._get_layer_phase(k)
                field = self.layer_aperture * torch.exp(1j * layer_phase) * field
                if k < self.num_layers - 1:
                    field = self.prop_interlayer(field)
                else:
                    U_ccd_flat = self.prop_to_ccd(field)

        else:
            U_ccd_flat = self.propagator_no_layers(U_flat)

        I_ccd = U_ccd_flat.abs().pow(2)[:, 0].reshape(B, self.T, self.N_sim, self.N_sim)
        I_vec = self._integrate_photodiode_array(I_ccd)
        if return_field:
            return I_vec, I_ccd
        return I_vec