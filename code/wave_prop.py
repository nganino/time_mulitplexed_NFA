'''
Single-Pixel QPI — wave propagation
Angular Spectrum Method (ASM) free-space propagator.

The transfer function H is pre-computed on the (optionally zero-padded)
simulation grid and stored as a register_buffer so it moves to the correct
device automatically.

Zero-padding (config.asm_pad_factor > 1):
  The FFT-based ASM assumes periodic boundary conditions, which can cause
  energy from one side of the simulation to wrap to the other.  Setting
  asm_pad_factor = 2 doubles the grid before the FFT and crops it back
  afterwards, fully eliminating this circular artefact at the cost of
  ~4× memory and ~2× compute.  With asm_pad_factor = 1 the zero-margin
  that already surrounds the SLM aperture (N_sim - slm_x_num) / 2 pixels
  on each side usually provides sufficient guard-band.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreeSpaceProp(nn.Module):
    '''
    Input / output shape: [B, 1, N_sim, N_sim]  (complex64)
    '''

    def __init__(self, config, z=None):
        super(FreeSpaceProp, self).__init__()

        self.N_sim      = config.N_sim
        self.pad_factor = int(getattr(config, 'asm_pad_factor', 1))

        wlength_eff = config.wavelength / config.ridx_air
        dx = config.sim_dx          # simulation / SLM pixel pitch
        z  = config.z_slm_ccd if z is None else z

        # H is built on the zero-padded grid so that a single precomputed
        # buffer can be used in forward() without re-computing each call.
        N_pad = self.N_sim * self.pad_factor
        dfx   = 1.0 / (N_pad * dx)

        fx, fy = torch.meshgrid(
            (torch.arange(N_pad) - N_pad / 2) * dfx,
            (torch.arange(N_pad) - N_pad / 2) * dfx,
            indexing='ij'
        )

        f0 = 1.0 / wlength_eff   # spatial-frequency cut-off (diffraction limit)

        # Band-limit mask: zero out evanescent (non-propagating) modes
        Q = (fx ** 2 + fy ** 2) <= (f0 ** 2)

        # ASM transfer function phase: 2π/λ · z · √(1 - (λfx)² - (λfy)²)
        prop_arg = Q * (fx ** 2 + fy ** 2) * (wlength_eff ** 2)
        phase    = 2 * np.pi * f0 * z * torch.sqrt(
            torch.clamp(1.0 - prop_arg, min=0.0)
        )

        H = torch.complex(torch.cos(phase), torch.sin(phase)) * Q
        # Pre-shift DC to [0, 0] to match the output layout of fft2
        H = torch.fft.ifftshift(H)[None, None]   # [1, 1, N_pad, N_pad]

        self.register_buffer('H', H)

    def forward(self, u):
        '''
        u   : complex field  [B, 1, N_sim, N_sim]
        out : propagated complex field  [B, 1, N_sim, N_sim]
        '''
        if self.pad_factor > 1:
            p = self.N_sim * (self.pad_factor - 1) // 2
            u = F.pad(u, (p, p, p, p))          # → [B, 1, N_pad, N_pad]

        U   = torch.fft.fft2(u)
        U_z = self.H * U
        out = torch.fft.ifft2(U_z)

        if self.pad_factor > 1:
            p = self.N_sim * (self.pad_factor - 1) // 2
            out = out[..., p : p + self.N_sim, p : p + self.N_sim]

        return out
