"""Fit the bench's non-idealities to the 11-14 zero-shot detector data (pfaff models), then test the fit on the other M.

Run from code/experiment/two_slm_insitu_bench with the qpi env:
    python gap_analysis_11_16/fit_bench.py [fit_M] [maxfev]
Objective: mean over functions of the correlation (over the 25 a) between the sim and the bench detectors (readout-free,
so the bench's gain / offset / per-detector response does not matter). Stage 1: a coarse key-shift scan (the key moved the
M1 correlation most in sensitivity.py). Stage 2: bounded Powell over all parameters. Then the fitted parameters are applied
to every M (held-out check) and each model's sim NMSE vs the target is reported at the fitted geometry.
Writes gap_analysis_11_16/fit_M<fit_M>.txt and fit_M<fit_M>.npz.
"""
import os, sys, time
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.getcwd(), 'gap_analysis_11_16'))
import sensitivity as S                                    # model loading, run(), metrics()

FIT_M = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAXFEV = int(sys.argv[2]) if len(sys.argv) > 2 else 500
OUT = S.OUT
# name, unit label, lower, upper  (all offsets relative to the trained geometry)
PAR = [('key_dx', 'PLUTO px', -4, 4), ('key_dy', 'PLUTO px', -4, 4),
       ('luna_dx', 'LUNA px', -3, 3), ('luna_dy', 'LUNA px', -3, 3),
       ('det_dx', 'cam px', -1.5, 1.5), ('det_dy', 'cam px', -1.5, 1.5),
       ('dz1', 'mm', -2, 2), ('dz2', 'mm', -1, 1),
       ('relay_mag', '%', -1, 1), ('cam_mag', '%', -1, 1),
       ('u_pluto', '', 0, 0.4), ('u_luna', '', 0, 0.4), ('fringe', 'w', 0, 0.2), ('luna_scale', '', 0.8, 1.1)]
X0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
# best point of the first 11-16 run (eval 251), which stalled on an unsnapped relay pitch
START = np.array([0.493, 0.767, -0.892, -0.438, 0.32, 0.342, -0.0715, 0.329, -0.193, 0.21, 0.0151, 0.000293, 0.0846, 1.0])     if os.environ.get('FIT_WARM') else None
CAM = 5.86e-6


def load(M):
    ck = torch.load(S.CK[M], map_location='cpu', weights_only=False)
    tc = S.init_params(); S.apply_saved_config(tc, ck['config']); tc.device = 'cuda'; S.recompute_derived(tc)
    m = S.TimeMultiplexedNFA(tc).to(S.dev); m.tc = tc
    m.load_state_dict(ck['model'], strict=False); m.eval()
    m.g0 = dict(m.spec_geom)
    bench = np.mean([np.load(os.path.join(OUT, 'bench_data', f'Nf4096_pfaff_M{M}_p{p}.npy')).astype(np.float64)
                     for p in (1, 2)], 0)
    return m, bench


def sim(m, x):
    kx, ky, lx, ly, dx, dy, z1, z2, rm, cm, u1, u2, fr, s2 = x
    g0 = m.g0
    c = 1 + cm / 100
    # relay magnification: the key pitch must be a whole nm with a small lcm with the LUNA pitch (README 0f, pitch_for),
    # otherwise the spectral propagator's lattice explodes.  Snap to the best whole-nm pitch within +-1 nm.
    want = g0['p1'] * (1 + rm / 100) * 1e9
    p2n = int(round(g0['p2'] * 1e9))
    cands = [pn for pn in range(int(round(want)) - 1, int(round(want)) + 2)]
    pn = min(cands, key=lambda q: (np.lcm(q, p2n), abs(q - want)))
    r = pn * 1e-9 / g0['p1']
    g = dict(g0, z1=g0['z1'] + z1 * 1e-3, z2=g0['z2'] + z2 * 1e-3,
             p1=pn * 1e-9, key_x0=g0['key_x0'] * r + kx * g0['p1'], key_y0=g0['key_y0'] * r + ky * g0['p1'],
             luna_x0=g0['luna_x0'] + lx * g0['p2'], luna_y0=g0['luna_y0'] + ly * g0['p2'],
             det_cx=np.asarray(g0['det_cx']) * c + dx * CAM, det_cy=np.asarray(g0['det_cy']) * c + dy * CAM,
             det_size=g0['det_size'] * c)
    return S.run(m, g, u1=u1, u2=u2, s2=s2, fringe=fr)


def corr_mean(d, b):
    dc, bc = d - d.mean(0), b - b.mean(0)
    r = (dc * bc).sum(0) / np.sqrt((dc ** 2).sum(0) * (bc ** 2).sum(0) + 1e-300)
    return r.mean(), np.median(r)


def main():
    t0 = time.time()
    log = []
    def say(s):
        print(s, flush=True); log.append(s)
    m, bench = load(FIT_M)
    lo = np.array([p[2] for p in PAR], float); hi = np.array([p[3] for p in PAR], float)
    n = [0]; best = [None, -2]
    def f(x):
        x = np.clip(x, lo, hi)
        r, _ = corr_mean(sim(m, x), bench)
        n[0] += 1
        if r > best[1]:
            best[:] = [x.copy(), r]
            say(f'  eval {n[0]:4d}  mean r {r:.4f}  ' + ' '.join(f'{p[0]} {v:+.3g}' for p, v in zip(PAR, x) if abs(v - X0[PAR.index(p)]) > 1e-9))
        return -r
    say(f'fit on M{FIT_M}; start (trained geometry, ideal SLMs): mean r {corr_mean(sim(m, X0), bench)[0]:.4f}')
    if START is not None:
        f(START)
        say(f'warm start from the stalled 11-16 fit ({n[0]} evals)')
    else:  # stage 1: coarse key scan
        for kx in range(-3, 4):
            for ky in range(-3, 4):
                x = X0.copy(); x[0], x[1] = kx, ky
                f(x)
        say(f'stage 1 done ({n[0]} evals, {time.time() - t0:.0f} s)')
    res = minimize(f, best[0], method='Powell', bounds=list(zip(lo, hi)),
                   options=dict(maxfev=MAXFEV, xtol=1e-2, ftol=1e-4, direc=np.diag([1, 1, 0.5, 0.5, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03])))
    xf = best[0]
    say(f'stage 2 done ({n[0]} evals, {time.time() - t0:.0f} s): mean r {best[1]:.4f}')
    say('fitted parameters: ' + ', '.join(f'{p[0]} {v:+.3g} {p[1]}' for p, v in zip(PAR, xf)))
    del m; torch.cuda.empty_cache()

    # held-out check on every M
    say('\nM | mean r / median r sim-vs-bench: trained geometry -> fitted | sim NMSE vs target at fitted (trained) | bench NMSE')
    for M in S.CK:
        m, bench = load(M)
        d0, d1 = sim(m, X0), sim(m, xf)
        r0, r1 = corr_mean(d0, bench), corr_mean(d1, bench)
        say(f'M{M} | {r0[0]:.3f} / {r0[1]:.3f} -> {r1[0]:.3f} / {r1[1]:.3f} | {S.metrics(d1)["nmse"]:.3f} ({S.metrics(d0)["nmse"]:.2e}) '
            f'| {S.metrics(bench)["nmse"]:.3f}' + ('   <- fitted on this M' if M == FIT_M else ''))
        del m; torch.cuda.empty_cache()
    open(os.path.join(OUT, f'fit_M{FIT_M}.txt'), 'w').write('\n'.join(log) + '\n')
    np.savez(os.path.join(OUT, f'fit_M{FIT_M}.npz'), x=xf, names=[p[0] for p in PAR])


if __name__ == '__main__':
    main()
