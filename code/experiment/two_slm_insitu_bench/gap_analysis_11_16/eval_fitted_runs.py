"""Bench-style NMSE (25 a, per-function min-max) of checkpoints, each in its own saved config (fitted runs: the fitted bench),
plus the old pfaff M2 evaluated at the fitted bench for reference.  Usage: python eval_fitted_runs.py <ckpt> ..."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), 'gap_analysis_11_16'))
import sensitivity as S
FIT = dict(kv.split('=', 1) for kv in open('gap_analysis_11_16/fitted_sets.txt').read().split())

def ev(path, extra=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    tc = S.init_params(); S.apply_saved_config(tc, ck['config'])
    for k, v in (extra or {}).items():
        setattr(tc, k, type(getattr(tc, k))(v) if getattr(tc, k) is not None and not isinstance(getattr(tc, k), str) else v)
    tc.device = 'cuda'; S.recompute_derived(tc)
    m = S.TimeMultiplexedNFA(tc).to(S.dev); m.load_state_dict(ck['model'], strict=False); m.eval()
    d = np.stack([m(S.A[i:i + 1]).mean(1).reshape(-1).double().cpu().numpy() for i in range(len(S.A))])
    r = S.metrics(d)
    return f"NMSE {r['nmse']:.4f}  RMSE {r['rmse']:.3e}  median r {r['r_t']:.4f}  (epoch {ck.get('epoch')}, fringe {m.luna_fringe})"

for Mo in [int(x) for x in os.environ.get('REF_M', '2').split(',')]:
    print(f'old pfaff M{Mo} at the fitted bench:', ev(S.CK[Mo], FIT), flush=True)
for p in sys.argv[1:]:
    print(os.path.relpath(p, '../../logs'), ':', ev(p), flush=True)
