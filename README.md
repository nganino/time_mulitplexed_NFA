# time-multiplexed-NFA

**NFA = Nonlinear Function Approximation.** A diffractive optical processor computes `Nf` independent nonlinear
functions `f_k(a)` of one scalar input `a` in parallel, following Rahman et al., *"Massively parallel and universal
approximation of nonlinear functions using diffractive processors,"* eLight (2025) 5:32 (DOI 10.1186/s43593-025-00113-w).

Our additions:
- **Time multiplexing:** `M` learned phase keys (frames), one per exposure; the detector intensities are averaged over
  the `M` frames before the readout. More frames add representational capacity, not only noise averaging.
- **E+D architecture:** a learned decoder phase mask on the encoding plane. It gives ~10^4x over keys alone at K = 0
  and is the paper architecture. In the lab profile each frame has its own decoder (`decoder_per_key`, since
  2026-10-21: decoder m is shown with key m, no hardware cost); sim and older runs share one decoder.
- **Split encoding (the paper's main result):** frame `m` shows only its own block of `Np` harmonics, so
  `N_alpha = Np * M` harmonics are represented with `Np` pixels per frame. Each key conditions the same static
  diffractive network to act as the coefficient bank for its own frequency block: the per-key output contains only that
  block's frequencies (plus within-frame differences below `Np`), see `key_evolution.png`.
- **Bench experiment:** `code/experiment/two_slm_insitu/` displays the trained keys and decoders on the two real SLMs,
  reads the 64 x 64 detectors off the camera and fine-tunes the masks in situ (see "Experiment" below).

The experiment log (what was run, what was learned) is `code/logs/SWEEP_HANDOVER.txt`; the paper-figure plan and the
split-encoding status are in `code/logs/PAPER_FIGURES_PLAN.md`; main figures and their generators are in `code/logs/figures/`; runs and summaries not behind a main figure are in `code/logs/archive/`.

## Current state (2026-11-06)

All numbers are the test RMSE: min-max readout, mean over the `Nf` functions.

- **Split encoding, core result.** Sim, Nf 1024, 400 epochs, transparent background, zero-based display, single block,
  gap 0. The encoding footprint is held fixed: the bin grows with M as 12, 17, 21, 24, 27 at M = 1 to 5, so every frame
  lights the M1 area. Figure: `logs/figures/fig28_split_core.png`.

  | N_alpha (K2) | M1 | M2 | M3 | M4 | M5 | runs |
  |---|---|---|---|---|---|---|
  | 60 | 7.2e-5 | 1.5e-5 | 3.3e-7 | 2.5e-7 | 4.0e-7 | `logs/split_Na60_2026-11-05` |
  | 80 | 1.2e-3 | 7.3e-5 | - | 3.3e-6 | 4.8e-7 | `logs/split_Na80_2026-11-06` |
  | 100 | 8.4e-3 | 1.2e-3 | - | 7.6e-5 | 7.3e-5 | `logs/split_Na100_2026-11-05` |

  - **N_alpha 80:** improves about 2500x from M1 to M5 and only reaches the ~2-4e-7 floor at M5.
  - **K3 and K4:** the same 13 runs at each K started 11-06 (`logs/split_Na_K34_2026-11-06`, done ~11-07).
- **Open issue: high-M plateaus.**
  - **Cases:** N_alpha 100 is flat from M4 to M5. N_alpha 64 at M8 never gets below 7.5e-5 (bin scan in
    `logs/split_Np64fp_2026-11-03`, fig26).
  - **What the runs have in common:** in each, the min-max test readout is ~25x worse than the best per-detector affine
    fit of the same intensities (fig27 b). The usual gap is 1-8x. So the optics learn, and the train-affine / test-min-max
    readout mismatch is the main suspect.
- **Constant encoding (sim; `logs/main_Nf1024_2026-11-01`, 150 epochs).** More frames help most when the network is
  short of capacity.
  - **Np64 K2:** M1 3.0e-3, M2 3.6e-4, M3 4.7e-5, M5 1.7e-6.
  - **Np36:** every K reaches the ~1e-7 floor from M2.
- **Lab curves for the bench** (`logs/final_Np4_Nf4096`: Np4, Nf 4096, 8-bit SLMs, 635 nm, 100 epochs):

  | z2 (E+D -> camera) | M1 | M2 | M3 | M5 |
  |---|---|---|---|---|
  | 5 mm | 4.0e-3 | 2.55e-3 | 1.75e-3 | 1.27e-3 |
  | 10 mm | 6.3e-3 | 2.6e-3 | 1.56e-3 | 1.23e-3 |

  - **Use on the bench:** these checkpoints are copied to `code/experiment/two_slm_insitu/ckpt/`. The 5 mm M2
    checkpoint is copied automatically when it finishes.
  - **Floor:** with continuous phase (no bit-depth limit) the 5 mm setup reaches 8.3e-8 / 4.7e-8 / 4.0e-8 at
    M1 / M3 / M5, so 8-bit quantisation sets the lab floor.

## Planned figures

Main text, in order. The `figNN` numbers refer to `code/logs/figures/` (older ones to `code/logs/archive/summary_<date>/`).

| # | Figure | Content | Status |
|---|---|---|---|
| 1 | **Architecture** | key m -> encoding plane -> K static layers -> detector array, with the time-multiplexed keys; constant vs split encoding drawn side by side | masks exported (`archive/summary_2026-11-02/system_diagram_Np64_K3_M3`); schematic to draw |
| 2 | **Core split result** | test RMSE vs M for N_alpha 60 / 80 / 100, one panel per K = 2 / 3 / 4 | K2 done (fig28); K3 / K4 running |
| 3 | **How split works** | Np16 M4 example. Per key: the phase key, its encoding frame (harmonic block), its own contribution c_m(a) and the spectrum of c_m (only its block's frequencies). Last panel: the sum of the contributions against the target | done: fig25 + `figures/split_panels/panels/`; spectra in `key_evolution.png` |
| 4 | **Constant encoding** | RMSE vs M for Np 36 / 49 / 64 at K 2 / 3 / 4 (each frame shows all harmonics): more frames add capacity | done (fig19) |
| 5 | **Experiment** | lab-matched Np4 Nf4096 E+D, 8-bit, RMSE vs M in simulation (fig17), then the bench: zero-shot and after in-situ fine-tuning | sim done except 5 mm M2; bench pending |

Supplementary:
- **Split design choices:** the footprint-matched bin vs a fixed bin (fig24), the N_alpha comparison including the
  readout cost (fig27), the M8 bin scan (fig26), and the encoding layout pilot (fig18).
- **True-scale plane layouts:** `archive/summary_2026-11-01/sim_layout`.
- **Learned masks and key similarity:** test.py outputs.
- **Lab design studies:** bit depth (fig11), z1 / z2 distances (fig15 / fig16), grid convergence
  (`logs/archive/grid_study_2026-10-30`).

Still to decide or run:
- **Shared-key ablation for split:** one key for all frames, to show the improvement comes from the keys and not the
  partition alone. Needs a `share_keys` flag (`code/logs/PAPER_FIGURES_PLAN.md` section 5).
- **Equal-photon-budget comparison:** needs a test-time noise model.
- **Seed error bars.**

`code/logs/PAPER_FIGURES_PLAN.md` has the full plan and the locked settings.

## Physical model

```
phase key m (learned, one per frame)          [M, key, key]  on the SLM
   | free space, key_to_enc_spacing
   v
encoding plane: phi_in(p; a) = 2*pi*p*a on Np pixels (sim: one block; lab: 3x3 copies of the patch)
                + decoder phase (learned; E+D: one per frame in the lab profile, else shared)
                non-encoding pixels: OPAQUE when K > 0, transparent when K = 0
   | free space, slm_first_layer_spacing (or last_layer_ccd_spacing when K = 0)
   v
K learned diffractive layers (shared; K = 0 in the lab setup)
   | free space, last_layer_ccd_spacing
   v
camera: sqrt(Nf) x sqrt(Nf) detectors, detector k -> function k
   -> mean over the M frames P_k(a) -> min-max readout f_hat_k(a) = (P_k(a) - Pmin) / (Pmax - Pmin)
      -> MSE against f(a)
```

**Readout (since 2026-10-23, paper Eq. 9):** `Pmin = min_k min_a P_k(a)`, `Pmax = max_k max_a P_k(a)`, jointly over all
detectors, so the output is normalised the same way as the targets. Every val / test number uses it, with Pmin / Pmax
over the full dense a-grid. **Training** (`train_readout`, default `affine` since 2026-10-24) fits a learned global scale
and offset, `scale * P + bias`, to the targets; the two numbers are discarded after training (their lr is
`lr_readout`, default = the key lr; 1e-3 and 1e-1 were both worse for split M8). Training directly with
min-max (`train_readout=minmax`: per batch, prediction and batch targets both min-max normalised) was 20-680x worse at
sim Np36 and lost the gain from more frames, although min-max itself costs nothing at test. The mismatch between the
trained affine and the test min-max usually costs ~1-8x (`readout_calibration.png`), but ~25x in the split runs that
plateau at high M (N_alpha 100 M5, N_alpha 64 M8).

**Background (since 2026-10-23):** with K > 0 the layers act as the coefficient bank and every encoding-plane pixel that
is not an encoding pixel is opaque, which removes the large a-independent DC term. With K = 0 (lab E+D) the background
stays transparent. At K > 0 a decoder can only act on the encoding pixels (`decoder_over_encoding` must be True).

`model.py` returns the raw per-frame detector intensities `[B, M, rows, cols]`; frame averaging, readout and loss
live in `loss.py`. Targets (`target_functions.py`) are `f_k(a) = |sum_i F_k(alpha_i) exp(j 2 pi alpha_i a)|^2`
with random `F_k` (paper Eq. 10, seeded by `func_seed`), min-max normalised jointly over all `Nf` functions; val / test
re-normalise the targets over their own a-grid, like `f_hat` (since 2026-10-24: the 2001-point calibration grid missed
the test grid's range by up to 0.17%, a fixed ~1.8e-4 RMSE floor at Np64 even for a perfect model).
Harmonics are centered, `alpha_i = -N_alpha//2 .. N_alpha - 1 - N_alpha//2` (`centered_harmonics=True`, default since
2026-10-20; earlier runs used `0 .. N_alpha - 1`). The encoding pixels use the same `alpha_i` unless
`encoding_zero_based`. The targets are identical in both conventions. With a transparent background (K = 0) the optics
are not: light that passes outside the encoding pixels interferes with the encoding light and creates terms at the
`alpha_i` themselves, so shifting them changes what the system has to fit. With the opaque background (K > 0) the two
conventions give identical results. Training uses the paper's forward-simulate-and-backprop route, not its closed-form
Fourier fit: the keys and intensity averaging make the system non-linear in the field.

## Locked defaults (lab 2026-10-30/31, sim 2026-11-01)

Only `Np`, `Nf`, `M` and the encoding mode are meant to change. `device_profile` selects one of two physical setups
(`config.py` `DEVICE_PROFILES`); each value can still be `--set`.

**Lab** (`device_profile=lab`, the real devices; paper setup with `num_layers=0 decoder=true`):

Two 1920 x 1080 Holoeye SLMs, LOCKED 2026-10-30 (wavelength and camera distance updated 2026-10-31): the phase keys on
the PLUTO (`key_slm_dx` 8 um), encoding + decoder on the LUNA (`enc_slm_dx` 4.5 um). (Before 10-30 the keys were on the
LUNA and the encoding on the PLUTO, at sim 1 um, 3 / 3 mm, pad 1.) Each SLM pixel is drawn on the sim grid at its exact
physical size (`model._AxisRaster`: a sim px straddling a pixel edge gets the area-weighted mix of the pixels it
covers). `sim_dx` = 0.5 um is the only pitch that divides 8, 4.5 and 6 um; at 1 um the trained designs read out 10-14%
wrong (`logs/archive/grid_study_2026-10-30`). The FFT grid is padded 2x (`asm_pad_factor` 2): at pad 1 the trained designs
used light that wrapped around the FFT grid (Np4 M1 7.7e-8 on the training grid -> 2.4e-2 on a padded grid). Values
for the final Np4 Nf4096 curves:

| plane | parameter | value |
|---|---|---|
| source | wavelength | 635 nm (lab laser) |
| key SLM (PLUTO) | pixel / key | 8 um; key = detector array / `key_det_fill` 0.8 -> 120 x 120 px = 0.96 mm at Nf 4096 |
| encoding SLM (LUNA) | pixel | 4.5 um |
| encoding | decoder (`decoder_per_key` True) | one per frame, covers the key: 213 x 213 LUNA px |
| encoding | pixel (`encoding_patch_scale` `fit`) | largest bin whose 3x3-tiled block fits in the detector array: Np4 -> 28 LUNA px (126 um); 2x2 patch x 3x3 tiles = 168 px |
| encoding | gap / display | gap 0; asymmetric alpha 0 .. N_alpha-1 (`encoding_zero_based` True); transparent background (K = 0) |
| distances | key -> E+D (z1) / E+D -> camera (z2) | 3 mm / 5 mm (`last_layer_ccd_spacing`; 10 mm backup). z2 is the effective distance after a 1:1 4f relay (2 cm direct: NA too low, M1 never trained) |
| camera | pixel / detectors | 6 um (Basler 5.86 um, snapped); detectors 1 camera px (6 um) on a 2 px (12 um) pitch, 64 x 64 at Nf 4096 (0.77 mm) |
| grid | `sim_dx` / `N_sim` | 0.5 um; `grid_key_mult` 1.5 x the key = 2880 (1.44 mm), FFT 5760 (`asm_pad_factor` 2), band-limited ASM, `illum_taper` 0.25 (cosine taper of the illumination over the outer 12.5% of the grid; a hard beam edge gave 11% error) |
| SLM phase | `slm_phase_bits` | 8 for the bench runs (each bit removed doubles the RMSE) |

**Sim** (`device_profile=sim`, default; 300 nm pixels everywhere, 1 sim px = 1 SLM px = 1 camera px):
LOCKED 2026-11-01 for the paper runs (`logs/PAPER_FIGURES_PLAN.md`): single encoding block, bin 12, gap 0 (96 px block at Np 64; 3x3 tiles bin 6 from 10-25, bin 2 / gap 2 until 10-25, bin 3 until 10-22), 2 px detectors on a 4 px pitch (126 px array at Nf 1024; 3 / 5 from 10-21, pitch 3 until 10-20, pitch 6 until 10-21), 150 epochs, all gaps 12 um (6 um until 2026-10-25; 12 um reaches the floor at M1 for Np64 K2), `N_sim` auto (512 at Np64 Nf 1024 K2, 448 at K3; was 256 / 384),
K = 2 layers. Key = layer size = `ceil(sqrt(N/K))` with `N = ceil(r*2*N_alpha*Nf)`, r = 1.25 (paper guideline).
Non-encoding pixels of the encoding plane are opaque (every K > 0 run, since 2026-10-23); the lab profile (K = 0) keeps
them transparent.

**Other defaults:** constant encoding, min-max readout (no setting), `decoder_over_encoding=True` (decoder trains on
every pixel of its square, encoding pixels included), `M` 1, `Nf` 100, `Np` 9, `a` in [-0.5, 0.5], `train_a_samples`
10000 (random pool), val/test on 1000-point evenly spaced grids (val: linspace with endpoints; test, since 2026-10-27: the 999 midpoints
of that grid, held out from val; earlier test numbers used the val grid itself), `max_epoch` 150, Adam lr 1e-2 for keys, layers and
decoder with cosine annealing to 1e-5, `seed` 59, `mask_init_std` 1, `batch_size` 64 (lab bench runs: 8 with `grad_accum_steps` 4, 100 epochs),
`grad_accum_steps` 1, `train_readout` affine, `slm_phase_bits` inf (lab bench runs: 8).

Lab bench run (as in `logs/final_Np4_Nf4096/run2.sh`; ~8.5 min / epoch at M2, 9 GB):
```
python train.py --set device_profile=lab num_layers=0 decoder=true decoder_over_encoding=true Nf=4096 Np=4 M=3 \
    slm_phase_bits=8 train_a_samples=10000 batch_size=8 grad_accum_steps=4 test_batch_size=2 max_epoch=100 \
    [last_layer_ccd_spacing=1e-2]
python test.py --ckpt logs/<run>/model/best.pth            # outputs in logs/<run>/test/
```
Sim constant-encoding run (locked 2026-11-01), and a core split run (N_alpha 80, M4: Np 20 per frame, footprint bin
12*sqrt(M) = 24; bins 12 / 17 / 21 / 24 / 27 at M = 1..5; see `logs/split_Na_K34_2026-11-06/run_split_Na_K34.sh`):
```
python train.py --set device_profile=sim Np=64 Nf=1024 num_layers=2 M=3
python train.py --set device_profile=sim Np=20 Nf=1024 num_layers=2 M=4 encoding_mode=split background_opaque=false \
    encoding_zero_based=true encoding_patch_scale=24 encoding_gap_blocks=0 encoding_tile_pattern=single max_epoch=400
```

## Main config keys

| key | meaning |
|---|---|
| `Np`, `Nf`, `M`, `num_layers` (K) | encoding pixels per frame, functions = detectors (perfect square), frames, layers |
| `encoding_mode` | `constant`: every frame shows all `Np` harmonics (`N_alpha = Np`); `split`: frame m shows harmonics `m*Np .. (m+1)*Np-1` on the same patch (`N_alpha = Np*M`). Patch = closest-to-square rows x cols |
| `encoding_tile_pattern` | from the device profile (sim `single` since 2026-11-01, lab `full` = 3x3 copies), `single`, `x`, `plus`, `full5` (5x5), or a 9/25-char 0/1 string |
| `device_profile`, `key_slm_dx`, `enc_slm_dx`, `sim_dx` | `sim` / `lab`; pitch of the SLM showing the keys, of the SLM showing encoding + decoder (and the layers), and of the sim grid. The SLM pitches need not be whole multiples of `sim_dx` (area-weighted pixel edges); `sim_dx` must divide the camera px. Checkpoints saved before 2026-10-25 (one `slm_dx`) load with both pitches = `slm_dx` |
| `decoder`, `decoder_size`, `decoder_over_encoding`, `lr_decoder` | E+D decoder: on/off, side in encoding-SLM px (None = covers the key's physical width, capped at `slm_px_max`), also train on the encoding pixels, its lr |
| `decoder_per_key` | one decoder per frame (lab profile default) or one shared (sim default). Shared and per-key checkpoints cannot be loaded into each other (train.py raises) |
| `phase_key_size`, `phase_key_patch_multiple`, `key_det_fill` | key side in SLM px (lab: 400; sim: = layer size; K = 0 sim: 9 x patch); or set as a multiple of the patch width; or `key_det_fill` f: key (and the decoder covering it) = detector-array width / f (e.g. 0.75 for large Nf; the E+D analogue of the sim detector array at ~80% of the last layer) |
| `key_to_enc_spacing` | 0 = key phase added exactly to the encoding phase (no propagation) |
| `train_readout` | training loss only: `affine` (default since 2026-10-24: a learned global scale + offset, discarded after training), `globalminmax` (the test's global min-max, Pmin / Pmax re-measured on the full a-grid at the start of every epoch and held fixed within it) or `minmax` (per-batch min-max; 20-680x worse at sim Np36, the frames stop helping). Every test.py number is global min-max either way; `readout_calibration.png` shows the train / test mismatch. Since 2026-11-13/14, two options match the bench's per-function readout. `perfuncaffine` is the current bench models: a learned scale + offset per function against per-function-normalised targets. `perfuncminmax` min-maxes each function over the micro-batch, which fails at micro-batch 1. With either, validation and `best.pth` use per-function min-max; `logs/bench_z2_1759_2026-11-13/eval_perfunc.py` gives the per-function test RMSE |
| `background_opaque` | `auto` (default): non-encoding pixels opaque when K > 0, transparent at K = 0; `true` / `false` force it (for controls only) |
| `lr_min_ratio` | every cosine schedule anneals its lr to `lr * lr_min_ratio` (default 1e-3: 1e-2 -> 1e-5, as before 2026-10-23) |
| `grad_accum_steps` | split each batch into k micro-batches (for memory). With `affine` / `perfuncaffine` / `globalminmax` the gradient is identical to the full batch (checked 2026-11-14, cosine 1.000000); with `minmax` / `perfuncminmax` each micro-batch normalises with its own range, so it is not |
| `prop_method`, `spec_P_mult` | `asm` (sim grid, default) or `spectral` (K = 0 E+D only, the bench models: each SLM on its own pixels, exact detector boxes, `wave_prop.SpectralLabProp`; geometry in `model.spec_geom`, swappable without retraining); spectral lattice period multiplier (bench 1.0) |
| `camera_dx`, `pd_pitch_cam`, `pd_size_cam` | camera pixel size; detector pitch and side in camera px (bench: 5.86 µm, 8, 4) |
| `checkpoint_save`, `val_dense_epochs`, `stop_epoch` | validate + save every n epochs; also every epoch while epoch < n; stop after this epoch but keep the `max_epoch` lr schedule |
| `slm_phase_bits` | SLM phase bit depth: `inf` (default, continuous, all runs before 2026-10-21) or an int b = 2^b levels on every SLM-displayed phase (keys, encoding + decoder plane, layers), straight-through gradient. Use for lab runs, e.g. `--set slm_phase_bits=8` |
| `centered_harmonics`, `encoding_zero_based` | targets use alpha = -N_alpha/2 .. (symmetric, default) or 0 .. N_alpha-1; `encoding_zero_based=True` displays alpha_p = p = 0 .. N_alpha-1 on the SLM even with symmetric targets (identical targets). Only matters at K = 0 (transparent background): asymmetric display puts the a-independent background at the band edge, up to ~10^3x better for lab K = 0 E+D (handover 4ac/4ae). At K > 0 (opaque background) both displays give identical results |
| `lr_readout` | lr of the training-only affine scale / offset (None = the key lr) |
| `asm_pad_factor`, `asm_band_limit`, `guard_orders`, `grid_key_mult`, `illum_taper` | FFT padding (lab 2: pad 1 lets light wrap around the grid), band-limited ASM, diffraction orders of one SLM pixel kept on the grid, `N_sim` as a multiple of the key, cosine illumination taper as a fraction of the grid |
| `meas_noise_std`, `freeze_slm`, `ckpt_to_load`, `load_weights_only` | training-time detector noise (up to this fraction of the per-sample readout std), frozen keys, resume from a checkpoint (continues at the next epoch) or warm start (weights only) |

### How `--set` works ("derive unless you set it")

`python train.py --set KEY=VALUE ...` overrides any `config.py` value (type preserved). `recompute_derived()` then
re-derives the profile values and the design values (`N_trainable_features`, `layer_size`, `phase_key_size`) unless
you set them yourself, and always recomputes pure geometry (bins, sim-pixel sizes, `N_alpha`, `run_name`, paths).
Design values are printed at train start and written to `LOG.txt`, tagged `[set]` or `[derived]`, with warnings
when a plane plus its light spread exceeds `N_sim`, the tiles exceed the key / layer, or a gap exceeds the ASM
aliasing limit. `test.py` rebuilds checkpoints exactly as trained: `apply_saved_config()` marks every saved key
explicit. Checkpoints written before 2026-10-23 (learned scale / bias readout, transparent background at K > 0) carry no
`readout: minmax` marker and are rejected by `train.py` and `test.py`; there is no backward compatibility. The ASM
transfer function is not stored in checkpoints; it is rebuilt from the config in float64.

### Bench models: training and the forward model

The current bench models are Np4, Nf 4096, E+D, K = 0, 8-bit, at z1 35.87 mm / z2 175.9 mm, trained with
`train_readout=perfuncaffine`. They are in `code/experiment/two_slm_insitu_bench/ckpt_M*_z1759_pfaff_ft`.

**Training command:** for M2, epochs 0–6, run from `code/`:
```
python logs/lab_ED_Np4_Nf4096_05um/capmem.py train.py --log_dir logs/<run_name> --set device_profile=lab num_layers=0 \
  decoder=true decoder_over_encoding=true Nf=4096 Np=4 seed=59 slm_phase_bits=8 prop_method=spectral spec_P_mult=1.0 \
  camera_dx=5.86e-6 pd_pitch_cam=8 pd_size_cam=4 key_det_fill=0.9 key_to_enc_spacing=0.03587 last_layer_ccd_spacing=0.1759 \
  key_slm_dx=8.075e-6 enc_slm_dx=4.5e-6 phase_key_size=414 decoder_size=736 encoding_patch_scale=110 \
  M=2 max_epoch=100 train_a_samples=10000 batch_size=8 grad_accum_steps=2 test_batch_size=4 \
  checkpoint_save=1 stop_epoch=6 train_readout=perfuncaffine
```
The script that built these runs is `code/logs/bench_z2_1759_2026-11-13/run_bench.sh`; `ft2.sh` there holds the
frozen-key fine-tunes.

**Forward model:** loading a checkpoint, changing the geometry (z1, z2, shifts, magnifications) without retraining,
simulated camera frames, and a flag-by-flag explanation are all in section 0f of
`code/experiment/two_slm_insitu_bench/README_updated.md`.

## Files (`code/`)

| file | role |
|---|---|
| `config.py` | all parameters, `DEVICE_PROFILES`, `recompute_derived()`, `describe_design()` |
| `model.py` | `TimeMultiplexedNFA`: keys -> encoding (+ decoder) -> layers -> detector windows; `K = 0` supported. Every SLM plane is a pixel array at its own pitch, put on the sim grid by `_AxisRaster` (exact pixel edges, area-weighted) |
| `wave_prop.py` | angular-spectrum propagator (evanescent waves cut; no Matsushima band limit on H, aliasing checked by `describe_design`) (`FreeSpaceProp`) |
| `loss.py` | `FunctionApproxLoss`: frame mean, min-max readout (`minmax_range`, `normalize`), MSE; `per_function_rmse`, `diffraction_efficiency` (diagnostic) |
| `target_functions.py` | `TargetFunctionSet`: closed-form targets, shared by train/val/test |
| `dataloader.py` | random training pool + dense val/test grids |
| `train.py` | trainer: separate Adam + cosine schedule per group (keys, layers, decoder), eval every 10 epochs and at the last epoch (min-max over the full val grid), `best.pth` by val loss, tensorboard |
| `test.py` | test pass + figures; `--csv/--sweep/--label` append a results row |
| `paths.py` | filesystem locations; runs go to `code/logs/` (gitignored); sweeps use per-folder `logs/<sweep>/analyze_*.py` scripts |
| `experiment/two_slm_insitu/` | bench code: display trained masks on the two SLMs, camera readout, in-situ PPO fine-tuning (next section) |

`test.py` writes to `<run>/test/`: `test_summary.png` (error histogram, input plane, per-detector RMSE map,
best/middle/worst fits), `plane_layout.png` (all planes on the sim grid), `phase_keys_final.png`,
`layer_masks_final.png`, `decoder_mask.png` (1 row, one panel per decoder), `phase_key_similarity.png`,
`key_evolution.png` (per key: the key; the intensity and phase of its field at the encoding plane; its own contribution
`c_m(a) = (I_m - Pmin) / (M (Pmax - Pmin))` to the output, which sums over m to f_hat exactly; and the spectrum of `c_m`,
with frame m's harmonic block shaded in split mode), `encoding_frames.png` (split mode, encoding phase only),
`readout_calibration.png` (2-D histogram of f_hat against the target, the points that set Pmin / Pmax, and the test RMSE
of three readouts of the same intensities: min-max, best global scale + offset, best per-detector scale + offset) and
`per_function_rmse.npy`. With `--csv`, the results row also gets
`rmse_global_affine`, `rmse_detector_affine`, `gain_median` and `depth_median`. Training logs `minmax/*` scalars to
tensorboard: per-epoch mean and spread of the batch Pmin / Pmax, the batch target extremes, and the val-grid Pmin / Pmax.
The input-plane panels show the encoding alone (never the decoder); at K > 0 the
opaque background is drawn black and every figure title says whether the background is opaque or transparent.

Tensor shapes: `a [B]` -> targets `[B, Nf]`; key phases `[M, 1, N_sim, N_sim]`; encoding phase `[B, 1 or M, N_sim, N_sim]`;
`model(a)` -> `[B, M, rows, cols]`; loss -> `f_hat [B, Nf]`.

## Experiment (`code/experiment/two_slm_insitu/`)

Starter code for the bench (PPO in-situ fine-tuning, Holoeye SDK, FLIR camera), adapted to this model with as few
changes as possible. It loads a lab checkpoint, shows key m on the PLUTO and (encoding(a) + decoder m) on the LUNA as
8-bit frames, captures the camera for every key, averages over the M frames and reads 64 x 64 detectors. The reward is
the same metric test.py reports, after one least-squares scale + offset per noisy state. Details, the bench checklist
and the run order are in that folder's `README.md`.

| file | role |
|---|---|
| `insitu_config.py` | SLM / camera geometry, flips and LUT flags, `a` grid, detector registration (`det_axis_yx`, `cam_flip_*`), PPO settings |
| `insitu_TDNFA.py` | main loop: frame building (`pluto_frame`, `luna_frame`), `capture_state`, `captures_to_detectors`, `calibrate_roi`, evaluation |
| `insitu_policy.py` | PPO agent over keys (u1) and decoders (u2); `compute_reward` (affine-fit RMSE) |
| `insitu_calib.py` | `load_trained_masks`: keys = 2 pi sigmoid(raw), decoders in radians, from a lab checkpoint |
| `insitu_utils.py` | SLM / camera glue, frame averaging, drift monitor |
| `export_targets.py` | writes `targets.npy` [100 a values, 4096] from a checkpoint (one file is valid for every Np4 Nf4096 checkpoint) |
| `ckpt/` | the 8-bit lab checkpoints (z2 5 mm and 10 mm, M1-M5) with `README.txt` listing their sim RMSE |

Checked in software against the model: frames within 1 grey level, detector readout to 2e-7, eval metric identical to
test.py, and `calibrate_roi` recovers a known axis offset and set of flips from simulated frames.

## Practical notes

- **Memory:** at Nf 1024 in the old 2 um lab profile (`N_sim` 2048, batch 16; before 2026-10-25), constant M5 needs 13.5 GB and constant M8
  20.3 GB; split needs more per frame (M4 15.2 GB). Keep the peak <= ~21 GB on a 24 GB card that drives a display, and
  use `grad_accum_steps` rather than a smaller batch. Wall time grows ~linearly with M (lab constant M5 ~5.7 h).
- **Intermittent segfaults:** `train.py` / `test.py` occasionally segfault at startup or teardown on this Windows
  machine. A rerun succeeds; sweep scripts retry (see the handover lessons). Long runs are launched detached (WMI
  `Win32_Process Create`) so they survive the session; `logs/*/run*.sh` resume from `last.pth`.
- **Comparability:** compare runs only at the same batch size and epoch count (batch alone shifts ~3x at the 1e-7
  level; `max_epoch` stretches the cosine schedule). Results from before 2026-10-23 (learned readout, transparent
  background at K > 0) are not comparable with newer runs.
- **Not implemented:** the paper's diffraction-efficiency loss term (Eq. 13/14); only MSE.

## Provenance

The original files came from [`ary-portes/single-pixel-qpi-paper`](https://github.com/ary-portes/single-pixel-qpi-paper)
(commit `f5b388c`, 29 Aug 2026), were pivoted to time-multiplexed image classification (Sep 2026) and then to this
function-approximation design (17 Sep 2026). Lab profile and E+D added 13-16 Oct 2026; defaults locked 17-18 Oct 2026;
min-max readout and opaque K > 0 background (no backward compatibility) 23 Oct 2026; lab profile re-locked for the
bench (keys on the PLUTO, 0.5 um grid, pad 2, 635 nm) 30-31 Oct 2026; sim paper settings locked 1 Nov 2026; bench
experiment code added 4-5 Nov 2026; split encoding with a footprint-matched bin made the main result 5-6 Nov 2026.
