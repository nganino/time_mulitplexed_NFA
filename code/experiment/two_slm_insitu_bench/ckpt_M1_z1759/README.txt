Bench checkpoint M1, z2 = 175.9 mm (2026-11-13). STILL TRAINING (100 epochs): this is best.pth as of epoch ~45.
bench_M1_bin1_z1759_ep0.pth = code/logs/bench_z2_1759_2026-11-13/M1_8bit_bin1/model/best.pth (epoch 0).
Geometry identical to ../ckpt_M2/README.txt (collimated bench, z1 35.87 mm, key 414 PLUTO px at 8.075 um, decoder 736 LUNA px,
encoding Np 4 bin 110 tiled 3 x 3, 64 x 64 detectors 4 px on an 8 px pitch, 8-bit linear phases, seed 59) EXCEPT
z2 = 175.9 mm (was 175.35; see ../sim_z2_M2/report.txt). Bench code unchanged: only --ckpt / ckpt_path changes.

Training: lr 1e-2 cosine, 100 epochs, val every epoch to 10 then every 5.
val min-max RMSE: ep0 4.93e-2 (best so far), ep1-10 5.5e-2..7.2e-2, ep15 5.21e-2, ep25 5.02e-2, ep45 5.74e-2
Sim test: not yet (runs automatically after epoch 99). Re-copy best.pth when the run completes.
Run: python insitu_TDNFA.py --ckpt ./ckpt_M1_z1759/bench_M1_bin1_z1759_ep0.pth
