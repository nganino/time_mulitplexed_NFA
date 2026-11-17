Bench checkpoint BIGGER DESIGN, Nf 4096, M2, collimated bench (copied 2026-11-13).
bench_big_M2_ep3.pth = code/logs/bench_big_2026-11-12/M2_8bit_bin1_ep7_big/model/best.pth (epoch 3, best validation).
bench_big_M2_config.json = the run's config, train_LOG.txt = its training log at copy time.

BIGGER key, decoder, encoding block and detectors than ckpt_M1/M2/M3 and ckpt_M*_enc55 -> the package code needs the
edits in ../README_updated.md section 0c (key / decoder sizes, encoding map, detector readout). Targets: ../targets.npy
(Nf 4096, unchanged).

Geometry (code/TRAINING_ADJUSTMENTS_2026-11-11.md), collimated plane wave:
  635 nm; PLUTO -> relay (m = 1.0096) -> z1 = 35.87 mm -> LUNA 4.5 um -> z2 = 175.35 mm -> Basler 5.86 um
  key 520 x 520 PLUTO px (4.20 mm), modelled at 8.075 um (8 um x relay m); the bench shows it at the real 8 um
  decoder 930 x 930 LUNA px (4.19 mm), one per key (M = 2 key / decoder pair(s), camera intensity averaged over the frames)
  encoding Np 4, 2 x 2 px of 55 LUNA px, tiled 8 x 8 -> 880 x 880 px block (3.96 mm) at rows / cols 25..904 of the decoder
  detectors 64 x 64, 7 x 7 camera px on a 10 px pitch (41.0 / 58.6 um), array 637 camera px (3.73 mm), symmetric about
  the axis; function k = 64 r + c
  8-bit LINEAR phase in training (no phase table, no randomisation, 1 x 1 pixels)

Training: lr 1e-2 cosine over 100 epochs, seed 59, affine training readout, min-max validation, epochs 0-6 only (stop_epoch 6), 100-epoch lr schedule.
  val min-max RMSE per validated epoch (0, 1, 2, ...; every 5 after epoch 10 for M1): 1.971e-03, 8.468e-04, 8.791e-04, 8.159e-04, 1.163e-03, 1.572e-03, 2.039e-03

Sim test (code/test.py, 999 a, min-max readout): mean RMSE 8.34e-4
