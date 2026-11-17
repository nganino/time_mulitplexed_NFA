Bench checkpoint M2, z2 = 175.9 mm (2026-11-13).
bench_M2_bin1_z1759_ep5.pth = code/logs/bench_z2_1759_2026-11-13/M2_8bit_bin1_ep7/model/best.pth (epoch 5, best val of the 7-epoch run).
Geometry identical to ../ckpt_M2/README.txt (collimated bench, z1 35.87 mm, key 414 PLUTO px at 8.075 um, decoder 736 LUNA px,
encoding Np 4 bin 110 tiled 3 x 3, 64 x 64 detectors 4 px on an 8 px pitch, 8-bit linear phases, seed 59) EXCEPT
z2 = 175.9 mm (was 175.35; see ../sim_z2_M2/report.txt). Bench code unchanged: only --ckpt / ckpt_path changes.

Training: lr 1e-2 cosine over 100 epochs, stopped after epoch 6, val every epoch.
val min-max RMSE by epoch: 3.11e-2, 1.14e-2, 4.27e-3, 1.74e-3, 1.35e-3, 1.29e-3 (ep5, kept), 2.31e-3
Sim test (test.py, 1000 a, mean per-function RMSE): min-max 1.26e-3, global affine (in-situ reward) 6.63e-4, per-detector affine 5.58e-4.
Run: python insitu_TDNFA.py --ckpt ./ckpt_M2_z1759/bench_M2_bin1_z1759_ep5.pth
