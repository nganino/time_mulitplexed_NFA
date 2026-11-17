'''
Time-Multiplexed NFA — wave propagation
Angular Spectrum Method (ASM) free-space propagator.

The transfer function H is pre-computed on the (optionally zero-padded)
simulation grid and stored as a register_buffer so it moves to the correct
device automatically.

Zero-padding (config.asm_pad_factor > 1):
  The FFT-based ASM assumes periodic boundary conditions, which can cause
  energy from one side of the simulation to wrap to the other.  Setting
  asm_pad_factor = 2 doubles the grid before the FFT and crops it back
  afterwards, fully eliminating this circular artefact at the cost of
  ~4× memory and ~2× compute.  With asm_pad_factor = 1 light leaving the grid
  wraps back in: trained lab designs exploited it (7.7e-8 -> 2.4e-2 on a padded
  grid, 2026-10-30), so the lab profile uses pad 2 + the band limit below.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreeSpaceProp(nn.Module):
    '''
    Input / output shape: [B, 1, N_sim, N_sim]  (complex64)
    '''

    def __init__(self, config, z):
        super(FreeSpaceProp, self).__init__()

        self.N_sim      = config.N_sim
        self.pad_factor = int(getattr(config, 'asm_pad_factor', 1))

        wlength_eff = config.wavelength / config.ridx_air
        dx = config.sim_dx          # simulation grid pitch

        # H is built on the zero-padded grid so that a single precomputed
        # buffer can be used in forward() without re-computing each call.
        # Built in float64: the phase k*z*sqrt(...) reaches ~1e4 rad (lab 3 mm), where float32 rounding
        # alone gives up to 3.5e-3 rad of phase error. Cast to complex64 only at the end.
        N_pad = self.N_sim * self.pad_factor
        dfx   = 1.0 / (N_pad * dx)

        fx, fy = torch.meshgrid(
            (torch.arange(N_pad, dtype=torch.float64) - N_pad / 2) * dfx,
            (torch.arange(N_pad, dtype=torch.float64) - N_pad / 2) * dfx,
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

        H = torch.complex(torch.cos(phase), torch.sin(phase)) * Q   # complex128
        # Band-limited ASM (Matsushima & Shimobaba 2009; config.asm_band_limit, lab since 2026-10-30): drop the angles whose
        # H phase chirp is undersampled on this grid, |fx|, |fy| >= 1 / (lambda sqrt((2 dfx z)^2 + 1)) -- light that would
        # otherwise alias / wrap; with pad 2 it is the light leaving the window (logs/grid_study_2026-10-30)
        if bool(getattr(config, 'asm_band_limit', False)):
            f_lim = 1.0 / (wlength_eff * np.sqrt((2 * dfx * z) ** 2 + 1))
            H = H * ((fx.abs() < f_lim) & (fy.abs() < f_lim))
        # Pre-shift DC to [0, 0] to match the output layout of fft2
        H = torch.fft.ifftshift(H)[None, None].to(torch.complex64)   # [1, 1, N_pad, N_pad]

        # Not saved in checkpoints: H is fully determined by the config, and older checkpoints' stored
        # (float32-built) H is ignored on load (strict=False), so every model uses this build.
        self.register_buffer('H', H, persistent=False)

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


# --------------------------------------------------------------------------------------------------------------------- #
#  Spectral per-plane propagator (2026-11-10, config.prop_method = 'spectral'; K = 0 E+D lab model only)               #
# --------------------------------------------------------------------------------------------------------------------- #
# No common spatial grid: every SLM keeps its own pixels and the whole path is done on one frequency lattice
# f_n = n / P (P = a whole number of both SLM pitches), so long distances (cm) need no huge 0.5 um grid.
# Fourier convention v(x) = int V(f) e^{2 pi i f x} df.
#   key plane   u0 = 1 + c,  c = e^{i phi_key} - 1 on the key px:  C(f) = DFT(px) * p1^2 sinc(p1 fx) sinc(p1 fy)  (exact)
#   leg 1       u1 = e^{ikz1} + v1,  V1 = C H(z1) T1   (T1: raised-cosine taper to 0 at |fx|, |fy| = F1)
#   LUNA        L = 1 + d,  d = e^{i phi_luna} - 1 on its px:  D(f) exact the same way
#               u1 L = e^{ikz1} + e^{ikz1} d + v1 + v1 d,  spectrum of v1 d = V1 (*) D (lattice convolution, by FFT)
#   leg 2       u2(x) = e^{ik(z1+z2)} + sum_{|f| <= F2} [e^{ikz1} D + V1 + V1 (*) D] H(z2) T2 e^{2 pi i f x} df^2
#   detectors   u2 at ngl x ngl Gauss-Legendre nodes inside each box (separable inverse DFT), box mean of |u2|^2
# The incident plane wave is infinite and flat (no slm_aperture / taper). Validated in 1D against a 0.125 um brute force
# (logs/multigrid_study_2026-11-10): F1 = F2 = 1.5 x the geometric reach, P = 1 x Pmin, taper 0.3 -> 6e-5 readout error.

def _fast_len(n):
    '''Smallest m >= n with no prime factor above 7 (fast FFT size).'''
    m = int(n)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def _taper1d(f, F, tau):
    a = np.abs(f); f0 = F * (1 - tau)
    t = np.zeros_like(a)
    t[a <= f0] = 1.0
    mid = (a > f0) & (a < F)
    t[mid] = 0.5 * (1 + np.cos(np.pi * (a[mid] - f0) / (F - f0)))
    return t


def _H_float64(fy, fx, z, lam):
    '''ASM transfer function on the lattice (float64 phase, complex64 out); 0 for evanescent waves.'''
    kz2 = 1.0 / lam ** 2 - fy[:, None] ** 2 - fx[None, :] ** 2
    ph = 2 * np.pi * z * np.sqrt(np.clip(kz2, 0, None))
    return torch.from_numpy(np.where(kz2 > 0, np.exp(1j * ph), 0).astype(np.complex64))


class SpectralLabProp(nn.Module):
    '''Key SLM -> z1 -> LUNA (encoding + decoder) -> z2 -> detector boxes, see the block comment above.
    geom: dict with (metres) p1, key_y0, key_x0 (corner of key px 0), nk; p2, luna_y0, luna_x0, luna_ny, luna_nx;
    det_cy, det_cx (detector centre coordinates, 1D arrays), det_size; z1, z2.'''

    def __init__(self, config, geom):
        super().__init__()
        lam = config.wavelength / config.ridx_air
        g = geom
        p1, p2, z1, z2 = g['p1'], g['p2'], g['z1'], g['z2']
        tau = float(getattr(config, 'spec_taper', 0.3))
        ngl = int(getattr(config, 'spec_gl', 4))

        # geometric reach -> bands and period (per axis, the larger of y / x)
        r_key = max(abs(g['key_y0']), abs(g['key_x0']), abs(g['key_y0'] + g['nk'] * p1), abs(g['key_x0'] + g['nk'] * p1))
        r_luna = max(abs(g['luna_y0']), abs(g['luna_x0']), abs(g['luna_y0'] + g['luna_ny'] * p2),
                     abs(g['luna_x0'] + g['luna_nx'] * p2))
        r_det = max(np.abs(g['det_cy']).max(), np.abs(g['det_cx']).max()) + g['det_size'] / 2
        # band = mult x geometric reach + kappa / sqrt(lam z): the second term is the Fresnel (stationary-phase) width,
        # which dominates when the apertures are small for the distance (Fresnel number < 1)
        kap = float(getattr(config, 'spec_kappa', 4.0))
        F1 = min(float(getattr(config, 'spec_F1_mult', 1.25)) * (r_key + r_luna) / (lam * z1) + kap / np.sqrt(lam * z1), 0.999 / lam)
        F2 = min(float(getattr(config, 'spec_F2_mult', 1.25)) * (r_luna + r_det) / (lam * z2) + kap / np.sqrt(lam * z2), 0.999 / lam)
        # period: a band-limited source of radius r spreads over lam z F, so a copy one period away must stay off the
        # receiving region on every leg (key -> LUNA, LUNA -> camera, key -> camera for the unmodulated key light)
        Pmin = max(r_key + r_luna + lam * z1 * F1, r_luna + r_det + lam * z2 * F2, r_key + r_det + lam * (z1 + z2) * F2)
        q = np.lcm(int(round(p1 * 1e9)), int(round(p2 * 1e9))) * 1e-9          # P = whole number of both pitches
        P = np.ceil(float(getattr(config, 'spec_P_mult', 1.25)) * Pmin / q) * q
        self.Q1, self.Q2 = int(round(P / p1)), int(round(P / p2))
        df = 1.0 / P
        N2 = int(np.ceil(F2 / df)); N1 = max(int(np.ceil(F1 / df)), N2)
        S = N1 + N2
        L = _fast_len(2 * S + 1)
        assert g['nk'] <= self.Q1 and max(g['luna_ny'], g['luna_nx']) <= self.Q2
        self.N1, self.N2, self.S, self.L, self.P, self.F1, self.F2 = N1, N2, S, L, P, F1, F2
        self.df2 = df * df
        self.c1 = complex(np.exp(1j * 2 * np.pi / lam * z1))
        self.c12 = complex(np.exp(1j * 2 * np.pi / lam * (z1 + z2)))

        # L-grid position k <-> lattice index n (k <= L//2: n = k, else n = k - L)
        k = np.arange(L)
        nL = np.where(k <= L // 2, k, k - L)
        fL = nL * df

        def env(n_mask, p, x0, Q):
            # DFT index into the size-Q FFT of the px array, and p sinc(p f) e^{-2 pi i f x_centre0} (0 where masked)
            e = p * np.sinc(p * fL) * np.exp(-2j * np.pi * fL * (x0 + p / 2)) * n_mask
            return torch.from_numpy(np.mod(nL, Q)), torch.from_numpy(e.astype(np.complex64))

        m1 = (np.abs(nL) <= N1).astype(float)
        mS = (np.abs(nL) <= S).astype(float)
        i1, e1y = env(m1, p1, g['key_y0'], self.Q1); _, e1x = env(m1, p1, g['key_x0'], self.Q1)
        i2, e2y = env(mS, p2, g['luna_y0'], self.Q2); _, e2x = env(mS, p2, g['luna_x0'], self.Q2)
        for name, t in (('i1', i1), ('e1y', e1y), ('e1x', e1x), ('i2', i2), ('e2y', e2y), ('e2x', e2x)):
            self.register_buffer(name, t, persistent=False)
        # leg 1 transfer x taper on the L-grid (0 beyond F1)
        T1 = _taper1d(fL, F1, tau) * m1
        self.register_buffer('H1T', _H_float64(fL, fL, z1, lam) * torch.from_numpy((T1[:, None] * T1[None, :]).astype(np.float32)),
                             persistent=False)
        # output lattice |n| <= N2: L-grid positions, leg 2 transfer x taper
        n2 = np.arange(-N2, N2 + 1)
        self.register_buffer('iL2', torch.from_numpy(np.mod(n2, L)), persistent=False)
        f2 = n2 * df
        T2 = _taper1d(f2, F2, tau)
        self.register_buffer('H2T', _H_float64(f2, f2, z2, lam) * torch.from_numpy((T2[:, None] * T2[None, :]).astype(np.float32)),
                             persistent=False)
        # detector Gauss-Legendre nodes and the separable inverse DFT to them
        t, w = np.polynomial.legendre.leggauss(ngl)
        yq = (np.asarray(g['det_cy'])[:, None] + g['det_size'] / 2 * t[None, :]).ravel()
        xq = (np.asarray(g['det_cx'])[:, None] + g['det_size'] / 2 * t[None, :]).ravel()
        self.register_buffer('Ey', torch.from_numpy(np.exp(2j * np.pi * np.outer(yq, f2)).astype(np.complex64)), persistent=False)
        self.register_buffer('ExT', torch.from_numpy(np.exp(2j * np.pi * np.outer(f2, xq)).astype(np.complex64)), persistent=False)
        self.register_buffer('wq', torch.from_numpy((np.outer(w, w) / 4).astype(np.float32)), persistent=False)
        self.ngl, self.ny_det, self.nx_det = ngl, len(g['det_cy']), len(g['det_cx'])

    def describe(self):
        return (f'spectral propagator: P {1e3 * self.P:.2f} mm, F1 {self.F1 / 1e3:.0f}/mm (N1 {self.N1}), '
                f'F2 {self.F2 / 1e3:.1f}/mm (N2 {self.N2}), conv FFT {self.L}^2, key FFT {self.Q1}^2, LUNA FFT {self.Q2}^2, '
                f'{self.ngl}x{self.ngl} nodes / detector')

    def _to_L(self, X, idx, ey, ex):
        # [..., Q, Q] FFT of a px array -> its exact spectrum on the L-grid (0 outside the kept band)
        X = X.index_select(-2, idx).index_select(-1, idx)
        return X * ey[:, None] * ex[None, :]

    def key_field_L(self, c_px):
        '''c_px [M, nk, nk] = e^{i key} - 1 -> (V1 on the L-grid [M, L, L], v1 in space on the L-grid [M, L, L])'''
        Q = self.Q1
        X = torch.fft.fft2(F.pad(c_px, (0, Q - c_px.shape[-1], 0, Q - c_px.shape[-2])))
        V1 = self._to_L(X, self.i1, self.e1y, self.e1x) * self.H1T
        return V1, torch.fft.ifft2(V1)

    def forward(self, V1, v1s, d_px):
        '''V1, v1s [T, L, L] (key_field_L); d_px [B, T, ny, nx] = e^{i luna} - 1 -> detector means [B, T, ny_det, nx_det]'''
        Q, L = self.Q2, self.L
        Dq = torch.fft.fft2(F.pad(d_px, (0, Q - d_px.shape[-1], 0, Q - d_px.shape[-2])))
        DL = self._to_L(Dq, self.i2, self.e2y, self.e2x)                        # [B, T, L, L]
        conv = torch.fft.fft2(v1s[None] * torch.fft.ifft2(DL)) * (L * L * self.df2)   # sum_m V1(f_m) D(f_n - f_m) df^2
        sel = lambda X: X.index_select(-2, self.iL2).index_select(-1, self.iL2)  # |n| <= N2
        A = (self.c1 * sel(DL) + sel(V1)[None] + sel(conv)) * self.H2T
        u2 = self.c12 + (self.Ey @ A @ self.ExT) * self.df2                        # [B, T, ny_det*ngl, nx_det*ngl]
        I = (u2.real ** 2 + u2.imag ** 2).reshape(*u2.shape[:2], self.ny_det, self.ngl, self.nx_det, self.ngl)
        return torch.einsum('ntipjq,pq->ntij', I, self.wq)
