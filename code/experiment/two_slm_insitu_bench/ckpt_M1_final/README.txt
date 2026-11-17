Bench checkpoint M1, FINAL epoch (not best), collimated bench (2026-11-13 copy).
bench_M1_bin1_ep99_final.pth = code/logs/bench_collimated_2026-11-11/M1_8bit_bin1_seed1234/model/epoch=099.pth
(last epoch of the full 100-epoch run, lr 1e-2 cosine). NOTE: seed 1234, a different design from ckpt_M1 (seed 59, ep 0;
that run stopped at epoch 24). This is the only baseline M1 trained the full 100 epochs; its best val was epoch 80.
Geometry identical to ../ckpt_M2/README.txt (collimated bench 2026-11-11, z1 35.87 / z2 175.35 mm, key 414 PLUTO px at 8.075 um,
decoder 736 LUNA px, encoding Np 4 bin 110 tiled 3 x 3, 64 x 64 detectors 4 px on an 8 px pitch, 8-bit linear phases).
Same bench code as ckpt_M1/M2/M3: no edits, only --ckpt / ckpt_path changes.

val min-max RMSE: ep0 5.03e-2, ep2 6.81e-2 (worst), ep40 5.05e-2, ep80 4.72e-2 (best), ep99 4.98e-2
Sim test (test.py, 1000 a, mean per-function RMSE), this file: min-max 4.98e-2, global affine (in-situ reward) 9.55e-3,
per-detector affine 8.43e-3.   For comparison: seed1234 best (ep80) 4.61e-2 / 9.88e-3; ckpt_M1 (seed 59 ep0) 4.83e-2 / 2.70e-2.
Run: python insitu_TDNFA.py --ckpt ./ckpt_M1_final/bench_M1_bin1_ep99_final.pth
