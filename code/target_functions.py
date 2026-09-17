'''
Target nonlinear functions for parallel function approximation (NFA).

Generates Nf synthetic bandlimited nonlinear functions f_k(a), k = 1..Nf,
following Rahman et al., eLight (2025) 5:32, Sec. 4.1 / Eq. 10:

    f_k(a) = | sum_{i=1}^{Nalpha} F_k(alpha_i) * exp(j*2*pi*alpha_i*a) |^2

where:
  - F_k(alpha_i) are complex Fourier coefficients, modulus ~ Uniform(0,1) and
    phase ~ Uniform(0, 2*pi) (PAPER, Eq. 10's surrounding text).
  - alpha_i = (i-1) * encoding_freq_step for i = 1..Nalpha, Nalpha == Np
    (PAPER, Sec. 4.1: "We set Nalpha = Np and alpha_i = (i-1)").
  - A single min-max normalization is applied JOINTLY across all Nf functions
    (not per-function) so the global min/max over (k, a) becomes 0/1 (PAPER,
    Sec. 4.1's normalization step, restated for Eq. 10's already-squared form).

All randomness is controlled by config.func_seed, NOTE: not a literal seed
value given in the paper -- they only specify the sampling *distributions*.
This is our own reproducibility knob so the same Nf target functions persist
across a training run (and across resumes from a checkpoint).

This module never touches the optical simulation -- it's pure closed-form
math, intentionally cheap to evaluate for any batch of `a` values (that's
what makes it usable as ground truth for both the training pool and the
dense validation/test grids in dataloader.py).
'''

import torch
import numpy as np


class TargetFunctionSet:
    '''
    Holds Nf fixed target nonlinear functions and evaluates them in closed
    form (no optical simulation) for any batch of input values `a`.
    '''

    def __init__(self, config):
        self.Nf       = int(config.Nf)
        self.N_alpha  = int(config.N_alpha)   # == Np, see config.recompute_derived
        self.freq_step = float(getattr(config, 'encoding_freq_step', 1))

        # alpha_i = (i-1) * freq_step, i = 1..N_alpha  (PAPER Sec. 4.1)
        self.alphas = torch.arange(self.N_alpha, dtype=torch.float64) * self.freq_step

        gen = torch.Generator().manual_seed(int(config.func_seed))
        modulus = torch.rand(self.Nf, self.N_alpha, generator=gen, dtype=torch.float64)
        phase   = torch.rand(self.Nf, self.N_alpha, generator=gen, dtype=torch.float64) * (2 * np.pi)
        self.coeffs = modulus * torch.exp(1j * phase)   # [Nf, N_alpha] complex128

        # ---- global min-max normalization (PAPER Eq. 10) -------------------
        # The paper normalizes using the true min/max of f_k(a) over the whole
        # continuous domain and all Nf functions jointly. We approximate that
        # with a dense grid (NOTE: grid resolution not specified in paper --
        # our own choice, generous enough for a stable estimate).
        calib_a    = torch.linspace(config.a_min, config.a_max, 2001, dtype=torch.float64)
        calib_vals = self._raw(calib_a)   # [2001, Nf]
        self.f_min = calib_vals.min().item()
        self.f_max = calib_vals.max().item()
        assert self.f_max > self.f_min, (
            'degenerate target-function range (f_max == f_min) -- check '
            'config.func_seed / config.Nf / config.Np'
        )

    def _raw(self, a):
        '''
        a : real tensor, any shape [...]
        returns : [..., Nf] UN-normalized f_k(a) = |sum_i F_k(alpha_i) exp(j*2*pi*alpha_i*a)|^2
        '''
        a = a.to(torch.float64)
        phase = 2 * np.pi * torch.einsum('...,i->...i', a, self.alphas)   # [..., N_alpha]
        basis = torch.exp(1j * phase)                                      # [..., N_alpha] complex
        series = torch.einsum('...i,ki->...k', basis, self.coeffs)         # [..., Nf] complex
        return series.abs() ** 2                                           # [..., Nf] real

    def __call__(self, a):
        '''
        a : real tensor of input values, any shape [...] (e.g. [B])
        returns : [..., Nf] float32 tensor, globally min-max normalized to [0, 1]
                  (PAPER Eq. 10's normalization, applied jointly across all Nf)
        '''
        normed = (self._raw(a) - self.f_min) / (self.f_max - self.f_min)
        return normed.to(torch.float32)
