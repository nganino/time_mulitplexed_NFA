Bench checkpoint M3, FINAL epoch (not best), collimated bench (2026-11-13 copy).
bench_M3_bin1_ep6_final.pth = code/logs/bench_collimated_2026-11-11/M3_8bit_bin1_ep7/model/epoch=006.pth
(last epoch of the 7-epoch run; lr still ~1e-2 on the 100-epoch cosine). Same run as ckpt_M3 (best, ep 2), 4 epochs later.
Geometry identical to ../ckpt_M2/README.txt (collimated bench 2026-11-11, z1 35.87 / z2 175.35 mm, key 414 PLUTO px at 8.075 um,
decoder 736 LUNA px, encoding Np 4 bin 110 tiled 3 x 3, 64 x 64 detectors 4 px on an 8 px pitch, 8-bit linear phases).
Same bench code as ckpt_M1/M2/M3: no edits, only --ckpt / ckpt_path changes.

val min-max RMSE by epoch: 8.58e-3, 7.56e-4, 4.91e-4 (ep2 = ckpt_M3), 6.09e-4, 8.90e-4, 9.86e-4, 1.18e-3 (ep6 = this file)
Sim test (test.py, 1000 a, mean per-function RMSE), this file: min-max 9.88e-4, global affine (in-situ reward) 4.96e-4,
per-detector affine 4.45e-4.   ckpt_M3 (ep2): 4.83e-4 / 2.61e-4 / 2.29e-4.
Run: python insitu_TDNFA.py --ckpt ./ckpt_M3_final/bench_M3_bin1_ep6_final.pth
