Bench checkpoint M1, z2 = 175.9 mm, trained with PER-FUNCTION AFFINE readout (train_readout perfuncaffine), 2026-11-14.
bench_M1_bin1_z1759_pfaff_ep50.pth = code/logs/bench_z2_1759_2026-11-13/M1_8bit_bin1_pfaff/model/best.pth (epoch 50; 100-epoch run STILL TRAINING at epoch ~50; re-copy when done).
Geometry identical to ../ckpt_M1_z1759 (baseline key 414 / decoder 736, bin 110 tiled 3 x 3, Nf 4096, z1 35.87 / z2 175.9 mm).
Bench code unchanged: only --ckpt / ckpt_path changes.

Training: learned scale + offset PER FUNCTION (4096 each, training only, discarded) against per-function min-max'd targets;
validation = per-function min-max over the a-grid (picks best.pth).
Sim test, 1000 a, mean per-function RMSE:
  PER-FUNCTION min-max (bench readout): 1.58e-2    (global-norm-trained ckpt_M1_z1759: 1.49e-1)
  global min-max:                       8.50e-2    -> USE THE PER-FUNCTION READOUT ON THE BENCH; global min-max is meaningless here
Run: python insitu_TDNFA.py --ckpt ./ckpt_M1_z1759_pfaff/bench_M1_bin1_z1759_pfaff_ep50.pth
