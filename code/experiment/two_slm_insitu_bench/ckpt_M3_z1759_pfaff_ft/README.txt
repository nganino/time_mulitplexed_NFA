Bench checkpoint M3, z2 = 175.9 mm, per-function-affine model FINE-TUNED with the phase keys FROZEN, 2026-11-15.
bench_M3_bin1_z1759_pfaff_ft_ep0.pth = code/logs/bench_z2_1759_2026-11-13/M3_8bit_bin1_pfaff_ft2_frozen/model/best.pth (fine-tune epoch 0; later epochs only drifted worse).
Started from M3_8bit_bin1_pfaff_ft_lr3e-4 best (its epoch 0), test 5.97e-4.
Fine-tune: freeze_slm (PLUTO keys unchanged), decoders + per-function readout at lr 3e-4, fresh Adam, cosine over 10 epochs,
batch 8, train_readout perfuncaffine; validation per-function min-max picks best.pth.
Geometry identical to ../ckpt_M3_z1759 (key 414 / decoder 736, bin 110 tiled 3 x 3, Nf 4096, z1 35.87 / z2 175.9 mm).
Bench code unchanged: only --ckpt / ckpt_path changes.

Sim test, 1000 a, mean per-function RMSE:
  PER-FUNCTION min-max (bench readout): 5.52e-4    (before fine-tuning, ../ckpt_M3_z1759_pfaff: 6.45e-4)
  global min-max:                       1.83e-1    -> use the per-function readout on the bench
Run: python insitu_TDNFA.py --ckpt ./ckpt_M3_z1759_pfaff_ft/bench_M3_bin1_z1759_pfaff_ft_ep0.pth
