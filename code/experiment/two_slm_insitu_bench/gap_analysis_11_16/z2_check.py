"""Fit objective (mean corr sim vs bench, pfaff M2 and held-out M3) at the fitted parameters with only z2 changed."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), 'gap_analysis_11_16'))
import fit_bench as FB
x = np.load('gap_analysis_11_16/fit_M2.npz')['x']
for M in (2, 3):
    m, bench = FB.load(M)
    for z2 in (175.9, 176.45, 176.9):
        xx = x.copy(); xx[7] = z2 - 175.9
        print(f'M{M} z2 {z2:.2f} mm: mean r {FB.corr_mean(FB.sim(m, xx), bench)[0]:.4f}', flush=True)
    del m; torch.cuda.empty_cache()
