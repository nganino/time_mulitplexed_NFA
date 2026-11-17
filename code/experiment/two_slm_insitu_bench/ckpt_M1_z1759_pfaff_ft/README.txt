Bench checkpoint M1, z2 = 175.9 mm, per-function-affine model FINE-TUNED with the phase keys FROZEN, 2026-11-15.
bench_M1_bin1_z1759_pfaff_ft_ep0.pth = code/logs/bench_z2_1759_2026-11-13/M1_8bit_bin1_pfaff_ft2_frozen3/model/best.pth (fine-tune epoch 0; later epochs only drifted worse).
Started from M1_8bit_bin1_pfaff best (epoch 99 of 100), test 1.50e-2.
Fine-tune: freeze_slm (PLUTO keys unchanged), decoders + per-function readout at lr 3e-4, fresh Adam, cosine over 10 epochs,
batch 8, train_readout perfuncaffine; validation per-function min-max picks best.pth.
Geometry identical to ../ckpt_M1_z1759 (key 414 / decoder 736, bin 110 tiled 3 x 3, Nf 4096, z1 35.87 / z2 175.9 mm).
Bench code unchanged: only --ckpt / ckpt_path changes.

Sim test, 1000 a, mean per-function RMSE:
  PER-FUNCTION min-max (bench readout): 1.49e-2    (before fine-tuning, ../ckpt_M1_z1759_pfaff: 1.58e-2)
  global min-max:                       8.33e-2    -> use the per-function readout on the bench
Run: python insitu_TDNFA.py --ckpt ./ckpt_M1_z1759_pfaff_ft/bench_M1_bin1_z1759_pfaff_ft_ep0.pth
