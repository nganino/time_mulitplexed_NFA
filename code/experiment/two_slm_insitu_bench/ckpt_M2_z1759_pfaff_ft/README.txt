Bench checkpoint M2, z2 = 175.9 mm, per-function-affine model FINE-TUNED with the phase keys FROZEN, 2026-11-15.
bench_M2_bin1_z1759_pfaff_ft_ep0.pth = code/logs/bench_z2_1759_2026-11-13/M2_8bit_bin1_pfaff_ft2_frozen3/model/best.pth (fine-tune epoch 0; later epochs only drifted worse).
Started from M2_8bit_bin1_pfaff_ft_lr3e-4 best (its epoch 0), val 9.49e-4.
Fine-tune: freeze_slm (PLUTO keys unchanged), decoders + per-function readout at lr 3e-4, fresh Adam, cosine over 3 epochs,
batch 8, train_readout perfuncaffine; validation per-function min-max picks best.pth.
Geometry identical to ../ckpt_M2_z1759 (key 414 / decoder 736, bin 110 tiled 3 x 3, Nf 4096, z1 35.87 / z2 175.9 mm).
Bench code unchanged: only --ckpt / ckpt_path changes.

Sim test, 1000 a, mean per-function RMSE:
  PER-FUNCTION min-max (bench readout): 9.07e-4    (before fine-tuning, ../ckpt_M2_z1759_pfaff: 1.09e-3)
  global min-max:                       1.26e-1    -> use the per-function readout on the bench
Run: python insitu_TDNFA.py --ckpt ./ckpt_M2_z1759_pfaff_ft/bench_M2_bin1_z1759_pfaff_ft_ep0.pth
