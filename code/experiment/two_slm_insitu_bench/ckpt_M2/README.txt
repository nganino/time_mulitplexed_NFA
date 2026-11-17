Bench checkpoint M2, collimated bench (2026-11-11).
bench_M2_bin1_ep4.pth = code/logs/bench_collimated_2026-11-11/M2_8bit_bin1_ep7/model/epoch=004.pth (best validation of the
7-epoch run; = that run's best.pth). bench_M2_bin1_config.json = the run's config, train_LOG.txt = its training log.

Geometry (code/TRAINING_ADJUSTMENTS_2026-11-11.md), collimated plane wave:
  635 nm; PLUTO -> relay (m = 1.0096) -> z1 = 35.87 mm -> LUNA 4.5 um -> z2 = 175.35 mm -> Basler 5.86 um
  key 414 x 414 PLUTO px, modelled at 8.075 um (= 8 um x 1.00935, the PLUTO pixel as seen at the LUNA; 8.077 um is not
  usable in the spectral propagator). The bench shows the key at the real 8 um pitch: NO resampling needed.
  decoder 736 x 736 LUNA px, one per key (M = 2 key / decoder pairs, camera intensity averaged over the 2 frames)
  encoding Np 4, 2 x 2 px of 110 LUNA px, tiled 3 x 3; detectors 64 x 64, 4 x 4 camera px on an 8 px pitch
  8-bit LINEAR phase in training (no phase table, no randomisation, 1 x 1 pixels)

Training: lr 1e-2 cosine over 100 epochs, stopped after epoch 6 (stop_epoch 6, validation every epoch).
  val min-max RMSE by epoch: 3.39e-2, 1.04e-2, 3.88e-3, 1.61e-3, 1.25e-3 (ep 4, kept), 1.31e-3, 2.57e-3

Sim test (code/test.py, 1000 a, mean per-function RMSE): min-max 1.20e-3, one global scale + offset (the in-situ reward)
5.0e-4, per-detector affine 4.2e-4.

Bench numbers, sensitivity and expected camera images: ../sim_check_M2/report.txt (and README.md section 0).
Run: python insitu_TDNFA.py --ckpt ./ckpt_M2/bench_M2_bin1_ep4.pth   (the config default)
