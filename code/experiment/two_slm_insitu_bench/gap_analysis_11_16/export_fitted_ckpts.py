"""Copy the current best.pth of the fitted-bench runs (logs/bench_fitted_2026-11-16) into ckpt_M<M>_fitted/ here, keeping only
what the bench package reads (model, config, epoch, readout marker): no optimiser state, so they fit on vsilkog.
Run from code/experiment/two_slm_insitu_bench.  Re-run later to refresh M3 / M5 (still training)."""
import os, torch
RUNS = {1: 'M1_8bit_bin1_ep50_pfaff', 2: 'M2_8bit_bin1_ep30_pfaff', 3: 'M3_8bit_bin1_pfaff', 5: 'M5_8bit_bin1_pfaff'}
for M, run in RUNS.items():
    src = f'../../logs/bench_fitted_2026-11-16/{run}/model/best.pth'
    ck = torch.load(src, map_location='cpu', weights_only=False)
    keep = {k: ck[k] for k in ck if k in ('model', 'config', 'epoch', 'readout')}
    keep['model'] = {k: v for k, v in ck['model'].items()}
    keep['source'] = f'logs/bench_fitted_2026-11-16/{run}/model/best.pth (epoch {ck.get("epoch")})'
    d = f'ckpt_M{M}_fitted'; os.makedirs(d, exist_ok=True)
    out = f'{d}/bench_M{M}_bin1_fitted_ep{ck.get("epoch")}.pth'
    torch.save(keep, out)
    print(M, sorted(ck.keys()), '->', out, round(os.path.getsize(out) / 2**20, 1), 'MB')
