Bench checkpoint M2, z2 = 175.9 mm, trained with PER-FUNCTION AFFINE readout (train_readout perfuncaffine), 2026-11-14.
bench_M2_bin1_z1759_pfaff_ep1.pth = code/logs/bench_z2_1759_2026-11-13/M2_8bit_bin1_ep7_pfaff/model/best.pth (epoch 1; best of the 7-epoch run; val rose to ~3.3e-3 by epoch 6).
Geometry identical to ../ckpt_M2_z1759 (baseline key 414 / decoder 736, bin 110 tiled 3 x 3, Nf 4096, z1 35.87 / z2 175.9 mm).
Bench code unchanged: only --ckpt / ckpt_path changes.

Training: learned scale + offset PER FUNCTION (4096 each, training only, discarded) against per-function min-max'd targets;
validation = per-function min-max over the a-grid (picks best.pth).
Sim test, 1000 a, mean per-function RMSE:
  PER-FUNCTION min-max (bench readout): 1.09e-3    (global-norm-trained ckpt_M2_z1759: 4.35e-3)
  global min-max:                       1.26e-1    -> USE THE PER-FUNCTION READOUT ON THE BENCH; global min-max is meaningless here
Run: python insitu_TDNFA.py --ckpt ./ckpt_M2_z1759_pfaff/bench_M2_bin1_z1759_pfaff_ep1.pth
