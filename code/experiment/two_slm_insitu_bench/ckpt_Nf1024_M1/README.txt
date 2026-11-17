Bench checkpoint Nf 1024, M1, collimated bench (copied 2026-11-12).
bench_Nf1024_M1_ep1.pth = code/logs/bench_nf1024_2026-11-11/M1_8bit_bin1_p16s12/model/best.pth (epoch 1, best
validation). bench_Nf1024_M1_config.json = the run's config, train_LOG.txt = its training log at copy time.

EASIER PROBLEM than ckpt_M1/M2/M3: Nf 1024 functions (32 x 32 detectors) and LARGER detectors, 12 x 12 camera px on a
16 px pitch (70.3 / 93.8 um, gap 23.4 um). The array is still 508 camera px (2.977 mm) wide, so the optical layout,
key, decoder, encoding and distances are IDENTICAL to ckpt_M2/M3. Only the camera readout and the targets differ:
use ../targets_Nf1024.npy and the code changes in ../README_updated.md section 0b.

Geometry (code/TRAINING_ADJUSTMENTS_2026-11-11.md), collimated plane wave:
  635 nm; PLUTO -> relay (m = 1.0096) -> z1 = 35.87 mm -> LUNA 4.5 um -> z2 = 175.35 mm -> Basler 5.86 um
  key 414 x 414 PLUTO px, modelled at 8.075 um (8 um x relay m); the bench shows it at the real 8 um, no resampling
  decoder 736 x 736 LUNA px, one per key (M = 1 key / decoder pairs, camera intensity averaged over the 1 frames)
  encoding Np 4, 2 x 2 px of 110 LUNA px, tiled 3 x 3
  detectors 32 x 32, 12 x 12 camera px on a 16 px pitch, symmetric about the axis; function k = 32 r + c
  8-bit LINEAR phase in training (no phase table, no randomisation, 1 x 1 pixels)

Training: lr 1e-2 cosine over 100 epochs, seed 59, affine training readout, min-max validation every epoch early on.
  val min-max RMSE, epochs 0-5: 1.30e-3, 4.20e-4 (ep 1, kept), 7.0e-4, 1.47e-3, 1.95e-3, 2.08e-3
  The validation minimum is at epoch 1 in every Nf 1024 run; later epochs drift up to ~1e-3 at the lr peak.

Sim test (code/test.py, 1000 a, mean per-function RMSE): min-max 3.72e-4, one global scale + offset (the in-situ reward)
2.28e-4, per-detector affine 2.0e-4.

Run: python insitu_TDNFA.py --ckpt ./ckpt_Nf1024_M1/bench_Nf1024_M1_ep1.pth   (after the section 0b edits)
