Lab-profile checkpoints for the bench (copied 2026-11-05 from logs/final_Np4_Nf4096/*/model/best.pth).
All: 635 nm, Np 4 (alpha 0..3), Nf 4096 (64 x 64 detectors, 6 um on a 12 um pitch), 8-bit SLM phase,
keys on PLUTO (120 x 120 px, 8 um), encoding + decoder on LUNA (213 x 213 px, 4.5 um), key -> Luna 3 mm.
Only the Luna -> camera distance z2 differs.  M comes from the checkpoint (cfg.M is not used).

file                                   z2      M   sim test RMSE (min-max, mean over functions)   source run
z2_5mm_M1_8bit.pth                     5 mm    1   4.00e-3                                         M1_8bit_s10000_wl635
z2_5mm_M2_8bit.pth                     5 mm    2   2.55e-3                                         M2_8bit_s10000_wl635
z2_5mm_M3_8bit.pth                     5 mm    3   1.75e-3                                         M3_8bit_s10000_wl635
z2_5mm_M5_8bit.pth                     5 mm    5   1.27e-3                                         M5_8bit_s10000_wl635
z2_10mm_M1_8bit.pth                    10 mm   1   6.28e-3                                         M1_8bit_s10000_wl635_z2_10mm
z2_10mm_M2_8bit.pth                    10 mm   2   2.57e-3                                         M2_8bit_s10000_wl635_z2_10mm
z2_10mm_M3_8bit.pth                    10 mm   3   1.56e-3                                         M3_8bit_s10000_wl635_z2_10mm
z2_10mm_M5_8bit.pth                    10 mm   5   1.23e-3                                         M5_8bit_s10000_wl635_z2_10mm

../targets.npy is valid for every checkpoint here (same Np / Nf / alpha; exported targets are identical).
Run:  python insitu_TDNFA.py --ckpt ./ckpt/z2_5mm_M3_8bit.pth [--calibrate-roi]
