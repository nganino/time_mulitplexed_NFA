Bench checkpoint M5, z2 = 175.9 mm (2026-11-13). NEW: 5 key / decoder pairs (bench loops over keys.shape[0], no code change).
bench_M5_bin1_z1759_ep2.pth = code/logs/bench_z2_1759_2026-11-13/M5_8bit_bin1_ep7/model/best.pth (epoch 2, best val of the 7-epoch run).
Geometry identical to ../ckpt_M2/README.txt (collimated bench, z1 35.87 mm, key 414 PLUTO px at 8.075 um, decoder 736 LUNA px,
encoding Np 4 bin 110 tiled 3 x 3, 64 x 64 detectors 4 px on an 8 px pitch, 8-bit linear phases, seed 59) EXCEPT
z2 = 175.9 mm (was 175.35; see ../sim_z2_M2/report.txt). Bench code unchanged: only --ckpt / ckpt_path changes.

Training: lr 1e-2 cosine over 100 epochs, stopped after epoch 6, val every epoch (micro-batch 1 x 8 accumulation).
val min-max RMSE by epoch: 1.01e-3, 3.48e-4, 3.09e-4 (ep2, kept), 3.40e-4, 3.96e-4, 4.19e-4, 4.39e-4
Sim test (test.py, 1000 a, mean per-function RMSE): min-max 3.60e-4, global affine (in-situ reward) 1.72e-4, per-detector affine 1.54e-4.
Bench: 5 PLUTO frames per a, so capture time is 2.5x M2.
Run: python insitu_TDNFA.py --ckpt ./ckpt_M5_z1759/bench_M5_bin1_z1759_ep2.pth
