# time-multiplexed-NFA

**NFA = Nonlinear Function Approximation.** A diffractive optical processor computes `Nf` independent nonlinear
functions `f_k(a)` of one scalar input `a` in parallel, following Rahman et al., *"Massively parallel and universal
approximation of nonlinear functions using diffractive processors,"* eLight (2025) 5:32 (DOI 10.1186/s43593-025-00113-w).

Our additions:
- **Time multiplexing:** `M` learned phase keys (frames), one per exposure; the detector intensities are averaged over
  the `M` frames before the readout. More frames add representational capacity, not only noise averaging.
- **E+D architecture:** a learned decoder phase mask on the encoding plane, shared by all frames. It gives ~10^4x
  over keys alone at K = 0 and is the paper architecture.
- **Split encoding:** frame `m` shows only its own block of `Np` harmonics, so `N_alpha = Np * M` harmonics are
  represented with `Np` pixels per frame.

The experiment log (what was run, what was learned) is `code/logs/SWEEP_HANDOVER.txt`.

## Physical model

```
phase key m (learned, one per frame)          [M, key, key]  on the SLM
   | free space, key_to_enc_spacing
   v
encoding plane: phi_in(p; a) = 2*pi*p*a on Np pixels (3x3 copies of the patch)
                + decoder phase (learned, shared by all frames; E+D)
   | free space, slm_first_layer_spacing (or last_layer_ccd_spacing when K = 0)
   v
K learned diffractive layers (shared; K = 0 in the lab setup)
   | free space, last_layer_ccd_spacing
   v
camera: sqrt(Nf) x sqrt(Nf) detectors, detector k -> function k
   -> mean over the M frames -> learned affine readout (scale, bias) -> f_hat(a), MSE against f(a)
```

`model.py` returns the raw per-frame detector intensities `[B, M, rows, cols]`; frame averaging, readout and loss
live in `loss.py`. Targets (`target_functions.py`) are `f_k(a) = |sum_i F_k(alpha_i) exp(j 2 pi alpha_i a)|^2`
with random `F_k` (paper Eq. 10, seeded by `func_seed`), min-max normalised jointly over all `Nf` functions;
`alpha_i = 0 .. N_alpha - 1`. Training uses the paper's forward-simulate-and-backprop route, not its closed-form
Fourier fit: the keys and intensity averaging make the system non-linear in the field.

## Locked defaults (2026-10-17/18)

Only `Np`, `Nf`, `M` and the encoding mode are meant to change. `device_profile` selects one of two physical setups
(`config.py` `DEVICE_PROFILES`); each value can still be `--set`.

**Lab** (`device_profile=lab`, the real devices; paper setup with `num_layers=0 decoder=true`):

| plane | parameter | value | sim px (2 um) |
|---|---|---|---|
| SLM | pixel (keys, decoder, layers) | 8 um | 4 |
| camera | pixel | 6 um (Basler 5.86 um, snapped) | 3 |
| key plane | phase key = decoder | 400 x 400 SLM px = 3.2 mm (fixed by the profile) | 1600 |
| encoding | pixel (`encoding_patch_scale` 3) | 3 SLM px = 24 um | 12 |
| encoding | gap (`encoding_gap_blocks` 3) | 9 SLM px = 72 um, also between tile copies | 36 |
| encoding | 3x3 block, varies with Np | Np 36: 1.66 mm, Np 49: 1.94 mm, Np 64: 2.23 mm | 828 / 972 / 1116 |
| camera | detector / pitch (`pd_size_cam` 2, `pd_pitch_cam` 4) | 12 um on 24 um (one width dark) | 6 / 12 |
| camera | detector array, varies with Nf | Nf 100: 0.23 mm, Nf 1024: 0.76 mm | 114 / 378 |
| distances | key->encoding, encoding->camera (and layer gaps) | 3 mm | 1500 |
| grid | `N_sim` | `auto` -> 2048 (4.1 mm) | |

**Sim** (`device_profile=sim`, default; 300 nm pixels everywhere, 1 sim px = 1 SLM px = 1 camera px):
bin 3, gap 0, 3x3 tiles, 3 px detectors touching (pitch 3 px), all gaps 6 um, `N_sim` 256 (set 384 at Nf 1024),
K = 2 layers. Key = layer size = `ceil(sqrt(N/K))` with `N = ceil(r*2*N_alpha*Nf)`, r = 1.25 (paper guideline).

**Other defaults:** constant encoding, `decoder_over_encoding=True` (decoder trains on every pixel of its square,
encoding pixels included), `M` 1, `Nf` 100, `Np` 9, `a` in [-0.5, 0.5], `train_a_samples` 10000 (random pool),
val/test on 1000-point evenly spaced grids, `max_epoch` 150, Adam lr 1e-2 for keys, layers, decoder and readout with
cosine annealing to 1e-5, `seed` 59, `mask_init_std` 1, `batch_size` 64 (lab runs use 16), `grad_accum_steps` 1.

Lab paper run:
```
python train.py --set device_profile=lab num_layers=0 decoder=true Nf=1024 Np=36 M=3 batch_size=16
python train.py --set device_profile=lab num_layers=0 decoder=true Nf=1024 encoding_mode=split Np=8 M=8 batch_size=16 grad_accum_steps=2
python test.py --ckpt logs/<run>/model/best.pth            # outputs in logs/<run>/test/
```

## Main config keys

| key | meaning |
|---|---|
| `Np`, `Nf`, `M`, `num_layers` (K) | encoding pixels per frame, functions = detectors (perfect square), frames, layers |
| `encoding_mode` | `constant`: every frame shows all `Np` harmonics (`N_alpha = Np`); `split`: frame m shows harmonics `m*Np .. (m+1)*Np-1` on the same patch (`N_alpha = Np*M`). Patch = closest-to-square rows x cols |
| `encoding_tile_pattern` | `full` (3x3 copies, default), `single`, `x`, `plus`, `full5` (5x5), or a 9/25-char 0/1 string |
| `decoder`, `decoder_size`, `decoder_over_encoding`, `lr_decoder` | E+D decoder: on/off, side in SLM px (None = key size), also train on the encoding pixels, its lr |
| `phase_key_size`, `phase_key_patch_multiple` | key side in SLM px (lab: 400; sim: = layer size; K = 0 sim: 9 x patch); or set as a multiple of the patch width |
| `key_to_enc_spacing` | 0 = key phase added exactly to the encoding phase (no propagation) |
| `grad_accum_steps` | split each batch into k micro-batches with the same gradient (for memory) |
| `meas_noise_std`, `freeze_slm`, `ckpt_to_load`, `load_weights_only` | training-time noise, frozen keys, warm start |

### How `--set` works ("derive unless you set it")

`python train.py --set KEY=VALUE ...` overrides any `config.py` value (type preserved). `recompute_derived()` then
re-derives the profile values and the design values (`N_trainable_features`, `layer_size`, `phase_key_size`) unless
you set them yourself, and always recomputes pure geometry (bins, sim-pixel sizes, `N_alpha`, `run_name`, paths).
Design values are printed at train start and written to `LOG.txt`, tagged `[set]` or `[derived]`, with warnings
when a plane plus its light spread exceeds `N_sim`, the tiles exceed the key / layer, or a gap exceeds the ASM
aliasing limit. `test.py` rebuilds checkpoints exactly as trained: `apply_saved_config()` marks every saved key
explicit; keys added later load with their old behaviour (no `encoding_tile_pattern` -> `single`, no
`decoder_over_encoding` -> False).

## Files (`code/`)

| file | role |
|---|---|
| `config.py` | all parameters, `DEVICE_PROFILES`, `recompute_derived()`, `describe_design()` |
| `model.py` | `TimeMultiplexedNFA`: keys -> encoding (+ decoder) -> layers -> detector windows; `K = 0` supported |
| `wave_prop.py` | band-limited angular-spectrum propagator (`FreeSpaceProp`) |
| `loss.py` | `FunctionApproxLoss`: frame mean, learned scale/bias readout, MSE; `per_function_rmse`, `diffraction_efficiency` (diagnostic) |
| `target_functions.py` | `TargetFunctionSet`: closed-form targets, shared by train/val/test |
| `dataloader.py` | random training pool + dense val/test grids |
| `train.py` | trainer: separate Adam + cosine schedule per group (keys, layers, readout, decoder), eval every 10 epochs and at the last epoch, `best.pth` by val loss, tensorboard |
| `test.py` | test pass + figures; `--csv/--sweep/--label` append a results row |
| `plot_sweep.py`, `plot_sweep_grid.py` | older generic sweep plots (current sweeps use per-folder `analyze_*.py` scripts) |
| `paths.py` | filesystem roots (`SPQPI_*` env overrides); runs go to `code/logs/` (gitignored) |

`test.py` writes to `<run>/test/`: `test_summary.png` (error histogram, input plane, per-detector RMSE map,
best/middle/worst fits), `plane_layout.png` (all planes on the sim grid), `phase_keys_final.png`,
`layer_masks_final.png`, `decoder_mask.png`, `phase_key_similarity.png`, `key_evolution.png` (per-key field at the
encoding plane and the cumulative fit), `encoding_frames.png` (split mode) and `per_function_rmse.npy`.

Tensor shapes: `a [B]` -> targets `[B, Nf]`; key phases `[M, 1, N_sim, N_sim]`; encoding phase `[B, 1 or M, N_sim, N_sim]`;
`model(a)` -> `[B, M, rows, cols]`; loss -> `f_hat [B, Nf]`.

## Practical notes

- **Memory:** at Nf 1024 in the lab profile (`N_sim` 2048, batch 16), constant M5 needs 13.5 GB and constant M8
  20.3 GB; split needs more per frame (M4 15.2 GB). Keep the peak <= ~21 GB on a 24 GB card that drives a display, and
  use `grad_accum_steps` rather than a smaller batch. Wall time grows ~linearly with M (lab constant M5 ~5.7 h).
- **Intermittent segfaults:** `train.py` / `test.py` occasionally segfault at startup or teardown on this Windows
  machine. A rerun succeeds; sweep scripts retry (see the handover lessons).
- **Comparability:** compare runs only at the same batch size and epoch count (batch alone shifts ~3x at the 1e-7
  level; `max_epoch` stretches the cosine schedule).
- **Not implemented:** the paper's diffraction-efficiency loss term (Eq. 13/14); only MSE.

## Provenance

The original files came from [`ary-portes/single-pixel-qpi-paper`](https://github.com/ary-portes/single-pixel-qpi-paper)
(commit `f5b388c`, 29 Aug 2026), were pivoted to time-multiplexed image classification (Sep 2026) and then to this
function-approximation design (17 Sep 2026). Lab profile and E+D added 13-16 Oct 2026; defaults locked 17-18 Oct 2026.
