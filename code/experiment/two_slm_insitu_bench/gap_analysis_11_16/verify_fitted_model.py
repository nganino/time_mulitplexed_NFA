"""Check the patched model (luna_fringe / spec_adjust / fitted z1 z2 pitch) reproduces fit_bench.py on the pfaff M2 ckpt,
and print the --set string for training at the fitted bench."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), 'gap_analysis_11_16'))
import sensitivity as S

x = dict(zip([str(n) for n in np.load('gap_analysis_11_16/fit_M2.npz')['names']], np.load('gap_analysis_11_16/fit_M2.npz')['x']))
p1_0, p2n = 8.075e-6, 4500
want = p1_0 * (1 + x['relay_mag'] / 100) * 1e9
pn = min(range(int(round(want)) - 1, int(round(want)) + 2), key=lambda q: (np.lcm(q, p2n), abs(q - want)))
SETS = dict(key_to_enc_spacing=round(0.03587 + x['dz1'] * 1e-3, 7), last_layer_ccd_spacing=round(0.1759 + x['dz2'] * 1e-3, 7),
            key_slm_dx=pn * 1e-9, luna_fringe=round(x['fringe'], 4),
            spec_adjust=','.join(f'{k}={x[n]:.3f}' for k, n in (('key_dy', 'key_dy'), ('key_dx', 'key_dx'), ('luna_dy', 'luna_dy'),
                                                                 ('luna_dx', 'luna_dx'), ('det_dy', 'det_dy'), ('det_dx', 'det_dx')))
                        + f",cam_mag={x['cam_mag'] / 100:.5f}")
print('SETS', ' '.join(f'{k}={v}' for k, v in SETS.items()))
open('gap_analysis_11_16/fitted_sets.txt', 'w').write(' '.join(f'{k}={v}' for k, v in SETS.items()) + '\n')

for label, extra in (('trained geometry', {}), ('fitted bench', SETS)):
    ck = torch.load(S.CK[2], map_location='cpu', weights_only=False)
    tc = S.init_params(); S.apply_saved_config(tc, ck['config'])
    for k, v in extra.items():
        setattr(tc, k, v)
    tc.device = 'cuda'; S.recompute_derived(tc)
    m = S.TimeMultiplexedNFA(tc).to(S.dev); m.load_state_dict(ck['model'], strict=False); m.eval()
    d = np.stack([m(S.A[i:i + 1]).mean(1).reshape(-1).double().cpu().numpy() for i in range(len(S.A))])
    print(label, 'NMSE', round(S.metrics(d)['nmse'], 4), 'p1', m.spec_geom['p1'], 'fringe', m.luna_fringe)
