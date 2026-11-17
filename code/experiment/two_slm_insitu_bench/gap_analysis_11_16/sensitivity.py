"""Sim-vs-bench gap: how much each non-ideality degrades the pfaff models, per M, and which one makes the sim look like the bench.

Run from the bench package folder (code/experiment/two_slm_insitu_bench) with the qpi env:
    python gap_analysis_11_16/sensitivity.py
Writes gap_analysis_11_16/sensitivity.csv.

Metrics on the bench's 25-a grid (cfg.a[::4]), per-function min-max readout, as on the bench:
  nmse   mean over functions of MSE / max(var_a(target), 4e-4)  (ddof 0)
  rmse   mean per-function RMSE
  r_t    median correlation (sim detectors vs target)
  r_b    median correlation (sim detectors vs bench detectors, 2-pass mean of the 11-14 zero-shot)
Perturbations (one at a time):
  geometry:  z1, z2, detector grid / LUNA / key shifts, relay and camera magnification   (via SpectralLabProp geometry)
  SLM:       u = unmodulated fraction  (field (1-u) e^{i phi} + u),
             s = LUNA phase scale (displayed phase x s),
             fringe = LUNA nearest-neighbour blur of the displayed phase (weight w to each of the 4 neighbours)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

PKG = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(PKG, '..', '..')))
sys.path.insert(0, PKG)
from config import init_params, recompute_derived, apply_saved_config
from model import TimeMultiplexedNFA
from wave_prop import SpectralLabProp
import insitu_config

torch.set_grad_enabled(False)
dev = torch.device('cuda:0')
OUT = os.path.join(PKG, 'gap_analysis_11_16')
CK = {1: 'ckpt_M1_z1759_pfaff/bench_M1_bin1_z1759_pfaff_ep50.pth',
      2: 'ckpt_M2_z1759_pfaff/bench_M2_bin1_z1759_pfaff_ep1.pth',
      3: 'ckpt_M3_z1759_pfaff/bench_M3_bin1_z1759_pfaff_ep0.pth',
      5: 'ckpt_M5_z1759_pfaff/bench_M5_bin1_z1759_pfaff_ep0.pth'}
cfg = insitu_config.init_params()
A = torch.as_tensor(np.asarray(cfg.a)[::4], dtype=torch.float32, device=dev)
T = np.load(os.path.join(PKG, 'targets.npy')).astype(np.float64)[::4]          # [25, 4096]


def metrics(d, bench=None):
    lo, hi = d.min(0), d.max(0)
    f = T.min(0) + (d - lo) / np.maximum(hi - lo, 1e-30) * (T.max(0) - T.min(0))
    mse = ((f - T) ** 2).mean(0)
    def corr(x, y):
        xc, yc = x - x.mean(0), y - y.mean(0)
        return (xc * yc).sum(0) / np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum(0) + 1e-300)
    out = dict(nmse=(mse / np.maximum(T.var(0), 4e-4)).mean(), rmse=np.sqrt(mse).mean(), r_t=np.median(corr(d, T)))
    if bench is not None:
        out['r_b'] = np.median(corr(d, bench))
    return out


def run(m, g, u1=0.0, u2=0.0, s2=1.0, fringe=0.0):
    m.spec = SpectralLabProp(m.tc, g).to(dev)
    V1, v1s = m.spec.key_field_L((1 - u1) * (torch.exp(1j * m.key_phases()[:, 0]) - 1))
    rows = []
    for i in range(len(A)):
        phi = m._input_plane_px(A[i:i + 1])                                     # [1, F, ny, nx], quantised, in [0, 2pi)
        if s2 != 1.0:
            phi = torch.remainder(phi, 2 * np.pi) * s2
        if fringe > 0:
            p = torch.remainder(phi, 2 * np.pi)
            k = torch.tensor([[0, fringe, 0], [fringe, 1 - 4 * fringe, fringe], [0, fringe, 0]], device=dev, dtype=p.dtype)
            sh = p.shape
            phi = F.conv2d(F.pad(p.reshape(-1, 1, *sh[-2:]), (1, 1, 1, 1), mode='replicate'), k[None, None]).reshape(sh)
        d = (1 - u2) * (torch.exp(1j * phi) - 1)
        rows.append(m.spec(V1, v1s, d.expand(-1, m.T, -1, -1)).mean(1).reshape(-1).double().cpu().numpy())
    return np.stack(rows)                                                        # [25, 4096]


def main():
    lines = ['M,perturbation,nmse,rmse,median_r_target,median_r_bench']
    for M, path in CK.items():
        t0 = time.time()
        ck = torch.load(path, map_location='cpu', weights_only=False)
        tc = init_params(); apply_saved_config(tc, ck['config']); tc.device = 'cuda'; recompute_derived(tc)
        m = TimeMultiplexedNFA(tc).to(dev); m.tc = tc
        miss, _ = m.load_state_dict(ck['model'], strict=False)
        assert not [k for k in miss if 'phases' in k], miss
        m.eval()
        g0 = dict(m.spec_geom)
        cam = 5.86e-6
        bench = np.mean([np.load(os.path.join(OUT, 'bench_data', f'Nf4096_pfaff_M{M}_p{p}.npy')).astype(np.float64)
                         for p in (1, 2)], 0)
        P = [('nominal', {}, {}),
             ('z2 +0.1 mm', dict(z2=g0['z2'] + 1e-4), {}), ('z2 -0.1 mm', dict(z2=g0['z2'] - 1e-4), {}),
             ('z2 +0.3 mm', dict(z2=g0['z2'] + 3e-4), {}),
             ('z1 +0.5 mm', dict(z1=g0['z1'] + 5e-4), {}),
             ('det +0.25 cam px', dict(det_cx=g0['det_cx'] + 0.25 * cam), {}),
             ('det +0.5 cam px', dict(det_cx=g0['det_cx'] + 0.5 * cam), {}),
             ('LUNA +0.25 px', dict(luna_y0=g0['luna_y0'] + 0.25 * g0['p2']), {}),
             ('LUNA +0.5 px', dict(luna_y0=g0['luna_y0'] + 0.5 * g0['p2']), {}),
             ('key +1 px', dict(key_x0=g0['key_x0'] + g0['p1']), {}),
             ('relay mag +0.2 %', dict(p1=g0['p1'] * 1.002, key_x0=g0['key_x0'] * 1.002, key_y0=g0['key_y0'] * 1.002), {}),
             ('camera mag +0.2 %', dict(det_cx=g0['det_cx'] * 1.002, det_cy=g0['det_cy'] * 1.002), {}),
             ('u LUNA 0.07', {}, dict(u2=0.07)), ('u LUNA 0.15', {}, dict(u2=0.15)),
             ('u PLUTO 0.15', {}, dict(u1=0.15)), ('u both 0.10', {}, dict(u1=0.1, u2=0.1)),
             ('LUNA phase scale 0.97', {}, dict(s2=0.97)), ('LUNA phase scale 0.92', {}, dict(s2=0.92)),
             ('LUNA fringe w 0.05', {}, dict(fringe=0.05)), ('LUNA fringe w 0.15', {}, dict(fringe=0.15))]
        for name, gch, kw in P:
            res = metrics(run(m, dict(g0, **gch), **kw), bench)
            lines.append(f"{M},{name},{res['nmse']:.4g},{res['rmse']:.4g},{res['r_t']:.4f},{res['r_b']:.4f}")
            print(lines[-1], flush=True)
        b = metrics(bench)
        lines.append(f"{M},BENCH (11-14 zero-shot),{b['nmse']:.4g},{b['rmse']:.4g},{b['r_t']:.4f},1")
        print(lines[-1], f'  [{time.time() - t0:.0f} s]', flush=True)
        del m; torch.cuda.empty_cache()
    open(os.path.join(OUT, 'sensitivity.csv'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
