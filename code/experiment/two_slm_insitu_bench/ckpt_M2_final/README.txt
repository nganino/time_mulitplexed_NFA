Bench checkpoint M2, FINAL epoch (not best), collimated bench (2026-11-13 copy).
bench_M2_bin1_ep6_final.pth = code/logs/bench_collimated_2026-11-11/M2_8bit_bin1_ep7/model/epoch=006.pth
(last epoch of the 7-epoch run; lr still ~1e-2 on the 100-epoch cosine). Same run as ckpt_M2 (best, ep 4), 2 epochs later.
Geometry identical to ../ckpt_M2/README.txt (collimated bench 2026-11-11, z1 35.87 / z2 175.35 mm, key 414 PLUTO px at 8.075 um,
decoder 736 LUNA px, encoding Np 4 bin 110 tiled 3 x 3, 64 x 64 detectors 4 px on an 8 px pitch, 8-bit linear phases).
Same bench code as ckpt_M1/M2/M3: no edits, only --ckpt / ckpt_path changes.

val min-max RMSE by epoch: 3.39e-2, 1.04e-2, 3.88e-3, 1.61e-3, 1.25e-3 (ep4 = ckpt_M2), 1.31e-3, 2.57e-3 (ep6 = this file)
Sim test (test.py, 1000 a, mean per-function RMSE), this file: min-max 2.94e-3, global affine (in-situ reward) 8.30e-4,
per-detector affine 7.23e-4.   ckpt_M2 (ep4): 1.20e-3 / 5.00e-4 / 4.16e-4.
Run: python insitu_TDNFA.py --ckpt ./ckpt_M2_final/bench_M2_bin1_ep6_final.pth
