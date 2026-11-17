# In-situ time-multiplexed NFA: bench package (2026-11-11)

> **Simulating the bench or training: see section 0f.** It covers running the forward model on a checkpoint, changing
> z1 / z2 / shifts / magnification without retraining, simulated camera frames, and every training flag. The current
> models are `ckpt_M*_z1759_pfaff_ft` (section 0e); read them out with per-function min-max.

This package runs the new bench checkpoints on the two-SLM setup and fine-tunes them there with PPO. It targets the
measured geometry in `code/RETRAINING_CHANGES.md`.

It is a copy of `../two_slm_insitu/`, the package for the old z1 = 3 mm / z2 = 5–10 mm lab design. That copy is kept
unchanged. Here the hard-coded layout, the display-side corrections and the camera code are updated for the bench. The
loop, the PPO agent and the reward are unchanged.

```
635 nm, wide collimated beam
 -> PLUTO (8 um px): key m, 414 x 414 px (3.31 mm)
 -> 1:1 4f relay; z1 = 29.7 mm from the PLUTO image to the LUNA
 -> LUNA (4.5 um px): decoder m, 736 x 736 px, plus the encoding:
        Np 4 = 2 x 2 encoding px of 110 x 110 LUNA px, phase 2 pi p a (p = 0..3), tiled 3 x 3
        -> a 660 x 660 px block at rows / cols 38..697 of the decoder
        display = encoding + decoder, mod 2 pi, quantised once
 -> z2 = 168.8 mm free space (no relay)
 -> Basler acA1920-40gm (5.86 um px, 1920 x 1200): 64 x 64 detectors of 4 x 4 camera px on an 8 px pitch,
        symmetric about the optical axis; function k = 64 r + c
readout: mean over the M frames -> one scale + offset (reward) / min-max over all detectors and a (eval)
```

## 0. Update 2026-11-11 (afternoon): collimated bench, M2 and M3

The beam is now collimated (`code/TRAINING_ADJUSTMENTS_2026-11-11.md`): z1 = **35.87 mm**, z2 = **175.35 mm**, relay
m = 1.0096. `ckpt_M1/` (29.7 / 168.8 mm) is **obsolete**: at the true distances its masks carry no information
(`sim_check_M1_z35.87_z175.35/`). Sections 1–5 below were written for M1. The code, layout, display corrections and
run order still apply; only the distances and checkpoints changed.

**New in this folder:**

| folder / file | content |
|---|---|
| `ckpt_M2/` | `bench_M2_bin1_ep4.pth`: M2, epoch 4 of a 7-epoch run (best validation), plus config, training log and README |
| `ckpt_M3/` | `bench_M3_bin1_ep2.pth`: M3, epoch 2 (best validation), plus config, training log and README |
| `sim_check_M2/`, `sim_check_M3/` | SLM frames and expected camera images (below), `report.txt` |
| `key_shift_scan.txt` | reward error vs PLUTO key shift (sim, from `key_shift_scan_sim.py`) |
| `scan_pluto_shift.py` | **bench** scan of `pluto_shift_yx` (below) |
| `insitu_config.py` | `det_axis_yx` (501, 985) → **(497, 985)** (collimator on); default `ckpt_path` → `ckpt_M2/` |

No other code changed. **When porting, merge rather than overwrite:**
- Your bench copy adds the collimator lens (`collimator.json`) to every PLUTO frame. This package has no collimator code,
  so keep yours.
- If your bench copy loads the measured `phase_tables.json`, keep that too. This package still uses the earlier quadratic
  fit: 315° / 312° at level 255, against about 301° / 305° in the new tables.

**Models:** both use 1 × 1 pixels, linear 8-bit phase, and a PLUTO pixel of 8.075 µm (8 µm × relay m). The bench
displays at the real 8 µm, so no resampling is needed.

**Calibration images** (`sim_check_M<M>/`, simulated with a collimated plane wave):
- `slm_bmp/pluto_key<m>_display.bmp` and `luna_dec<m>_a<+x.xxx>_display.bmp`: the frames the package sends for frame m,
  for a = −0.5, 0, +0.5. Display corrections are applied; the collimator lens is **not**.
- `cmos_expected_a<+x.xxx>.bmp`: the expected capture, mean of the M frames, through the phase table, in raw Basler
  orientation. `_array.bmp` is the detector-array zoom. The `.png` figures show the same images.
- BMP scale: exposed for the detector array, so 99.9% of the array px are below saturation (scale 0.83 for M2, 0.77 for
  M3, in units of the incident beam). The open beam and the rim saturate.

**What the camera should show:**
- a dark square about 3 mm across;
- a bright rim around it, peaking at 2.5–2.6× the beam;
- the open beam everywhere else;
- inside the square, the detector array (raw rows 243..750, cols 731..1238), with fine-grained light up to about 1.2×
  the beam.

**Checks:** the package's frames match the model's levels except for a few ±1-level rounding ties. Its detector readout
of the simulated captures matches the model to within 5e-6 of the range.

**Expected zero-shot numbers and sensitivity** (sim, 100 a; reward = affine RMSE, one scale + offset; a readout with no
signal scores 8.7e-2):

| case | M2 reward | M3 reward |
|---|---|---|
| as trained (linear phase) | 5.0e-4 | 2.6e-4 |
| **through the phase table (fit)** | **9.1e-3** | **8.4e-3** |
| z1 ± 0.5 mm | 5.5e-2 | 4.5e-2 |
| z2 ± 0.5 mm | 2.6e-2 | 1.9e-2 |
| m ± 0.005 | 5.6–6.0e-2 | 4.6–5.1e-2 |
| key shift 0.5 / 1 / 2 / 3 PLUTO px (x or y) | 3.5e-2 / 6.4e-2 / 8.2e-2 / 8.7e-2 | 2.6e-2 / 5.4e-2 / 7.9e-2 / 8.6e-2 |

**What this means for the bench:**
- **Key position is critical.** These masks need the key placed to within about 0.5 PLUTO px (4 µm). At 1 px most of the
  signal is gone; at 2–3 px none is left. Before any in-situ training, run
  `python scan_pluto_shift.py --ckpt ./ckpt_M2/bench_M2_bin1_ep4.pth`:
  - it steps `pluto_shift_yx` in 1 px steps over ±3 px around (−12, +30), 49 positions;
  - at each it captures 8 a values and scores the affine reward, also searching the detector axis ±4 camera px;
  - it takes about 4 min (M2) or 6 min (M3), then prints the best shift and saves `shift_scan.png`.

  The right position should stand out: at most 3.5e-2, against about 6e-2 or worse at its neighbours. A flat map near
  8.7e-2 means no key was found.
- **Within-uncertainty errors stack:** z1 ± 0.5 mm or m ± 0.005 alone costs about 100×. Expect the zero-shot reward to be
  close to the no-signal value unless the geometry is better than the stated uncertainties.
- **Phase table:** 18–32×. The masks were trained linear, as agreed.
- **Robustness:** these numbers argue for training the next models with key-position and distance robustness (or 2 × 2
  binning; those runs are in progress).

**Run:**
1. `python scan_pluto_shift.py --ckpt ./ckpt_M2/bench_M2_bin1_ep4.pth`, then paste the printed `pluto_shift_yx` into
   the config.
2. `python insitu_TDNFA.py --ckpt ./ckpt_M2/bench_M2_bin1_ep4.pth --calibrate-roi`.
3. The same command without `--calibrate-roi`.

The same steps apply to `ckpt_M3/`. A measurement takes M camera frames, so a PPO epoch takes about 2× (M2)
or 3× (M3) as long as at M1.

**Contents (M1, before the update)**
- **Checkpoint:** `ckpt_M1/` (obsolete, see above).
- **Sanity check:** `sim_check_M1/` holds the expected camera images with the M1 masks in place.

## 0b. Update 2026-11-12: Nf 1024 models with larger detectors

An easier problem for the bench: **1024 functions** (32 × 32 detectors) and **detectors 3× wider**: 12 × 12 camera px
on a 16 px pitch (70.3 µm boxes, 93.8 µm pitch, 23.4 µm gap), against 4 × 4 px on an 8 px pitch before.
- The detector array is still **508 camera px (2.977 mm)** wide. Everything optical is identical to `ckpt_M2/`/`ckpt_M3/`:
  - key 414 PLUTO px, decoder 736 LUNA px, encoding bin 110 tiled 3 × 3;
  - z1 35.87 mm, z2 175.35 mm, relay m, collimated beam;
  - linear 8-bit phase, no randomisation.
- The PLUTO / LUNA frames, the display corrections, the detector axis (`det_axis_yx`), the ROI calibration and the
  reward are therefore unchanged.
- Only the **camera readout** and the **targets** change.

Why: at Nf 4096 each detector (23 µm) was smaller than one diffraction spot (~34 µm at 175 mm), so it sampled speckle.
The 70 µm detectors cover about two spots each, with ~2.8 resolvable spots per pitch (was 1.4).

**Checkpoints** (all epoch 1, the validation minimum; every Nf 1024 run is best at epoch 1 and drifts up afterwards):

| folder | file | val min-max RMSE (ep 1) | sim test (min-max / reward) |
|---|---|---|---|
| `ckpt_Nf1024_M1/` | `bench_Nf1024_M1_ep1.pth` | 4.20e-4 | 3.72e-4 / **2.28e-4** |
| `ckpt_Nf1024_M2/` | `bench_Nf1024_M2_ep1.pth` | 3.40e-4 | pending |
| `ckpt_Nf1024_M3/` | `bench_Nf1024_M3_ep1.pth` | 2.04e-4 | pending |

For comparison, the Nf 4096 bench M1 at the same distances never got below ~5e-2, and `ckpt_M2/` tests at 1.20e-3 /
5.0e-4.
- M2 and M3 were still training (100 epochs) when copied. Their `best.pth` is epoch 1 and is not expected to change.
- `sim_check_*` and the sensitivity table have **not** been made for these checkpoints yet.

**Targets:** `targets_Nf1024.npy` is [100, 1024]: the same 100 a values in [−0.5, 0.5], min-max'd over that grid as
`test.py` does, with function k = 32 r + c.
- It is built from the runs' saved config, like `export_targets.py`. It is identical for M1, M2 and M3.
- It is a **different function set**, not the first 1024 columns of `targets.npy`.
- `targets.npy` (4096 functions) is kept for the old checkpoints.
- `targets_Nf1024.png` shows samples.
- `python export_targets.py --ckpt ./ckpt_Nf1024_M1/bench_Nf1024_M1_ep1.pth --out ./targets_Nf1024.npy` regenerates it.

### Code edits to run the Nf 1024 checkpoints

**Not applied in this package** (it still runs the Nf 4096 checkpoints). Make these edits in your bench copy, or in a copy
of this folder; switching back means reverting them. They are hard-coded like the rest of the package.

**1. `insitu_config.py`, camera section:**
```python
c.Nf = 1024                                # was 4096: 32 x 32 detectors
c.pd_size = 12 * c.cam_dx                  # was 4 * c.cam_dx: 12 x 12 camera px (70.3 um)
c.pd_pitch = 16 * c.cam_dx                 # was 8 * c.cam_dx: 16 camera px (93.8 um)
c.targets_path = './targets_Nf1024.npy'    # was './targets.npy'
c.ckpt_path = './ckpt_Nf1024_M1/bench_Nf1024_M1_ep1.pth'   # or _M2 / _M3
```
The detector-row comment becomes: detector r covers camera rows A − 254 + 16 r .. A − 243 + 16 r (model frame).

**2. `insitu_TDNFA.py`, `captures_to_detectors` (about lines 209 and 212):**
```python
I = I.reshape(*I.shape[:-2], 32, 16, 32, 16)[..., :12, :, :12].mean(axis=(-3, -1))   # [..., 32, 32] 12 x 12 px box means
...
return torch.from_numpy(np.ascontiguousarray(I).reshape(*I.shape[:-2], 1024)).to(device)
```
(was `64, 8, 64, 8` / `:4` / `4096`). The 512 px crop and the `y0 + 508` assert stay: 31 × 16 + 12 = 508 = 63 × 8 + 4.

**3. `insitu_TDNFA.py`, `save_detector_crop` (about line 226):**
```python
[patches.Rectangle((ox + 16 * c - 0.5, oy + 16 * r - 0.5), 12, 12) for r in range(32) for c in range(32)],
```

**Unchanged:**
- `det_origin` (axis − 254): the array is still 508 px, symmetric about the axis corner.
- The 180° camera flips: the 12 px boxes on a 16 px pitch are symmetric under the flip, as the 4 / 8 px boxes were.
- `calibrate_roi` (its 508 px box), the frames, `q8`, PPO and the reward. `scan_pluto_shift.py` reads the detectors
  through `captures_to_detectors`, so it follows edit 2.
- The comments at the top of `insitu_TDNFA.py` (lines 11, 134–135) still say 64 × 64. Update them if you like.

**Checked:** on a synthetic frame (each 12 × 12 box holding its own index, rotated 180° as the camera sees it, placed at
`det_axis_yx`), edit 2 returns exactly k = 0..1023 in model order and reads no gap px.

**Bench notes:**
- **Exposure:** the 12 px boxes collect about 9× the light of the 4 px boxes, and the diffraction efficiency is about
  13–17% (Nf 4096: ~4–6%). Re-set `cam_exposure_us` from the array with `--calibrate-roi`. The old setting will probably
  saturate.
- **Key position:** the key-shift sensitivity has not been re-measured. Larger detectors should tolerate more, but still
  run `scan_pluto_shift.py` first.
- **Phase table:** the models were trained linear, as before. Expect the phase table to cost something at zero shot.

## 0f. Running the forward model (for alignment) and training (2026-11-16)

This section is the reference for simulating the bench with the trained model, e.g. to fit the real alignment by
comparing simulated and captured detector readings or camera frames.

### Environment

- Python: the conda env `qpi`, `C:\Anaconda\envs\qpi\python.exe` (Git Bash: `/c/Anaconda/envs/qpi/python.exe`).
- The model code is in `code/` (two levels up from this folder): `config.py`, `model.py`, `wave_prop.py`, `loss.py`.
  Scripts in this package put `code/` on `sys.path` themselves (see below).
- GPU: about 3.5 GB for M2 at one a value per call. There are two GPUs of 24 GB each; one drives the display. Keep the
  total on that one under about 21 GB, and check `nvidia-smi` first in case a training run is using a GPU.

### What the forward model is

`TimeMultiplexedNFA` with `prop_method='spectral'` (`wave_prop.SpectralLabProp`). It takes the checkpoint's keys and
decoders and gives the 64 × 64 detector intensities for each frame m. The detectors are exact 4 × 4 camera-pixel boxes on
an 8 px pitch, function k = 64 r + c. In order, it models:

1. A collimated, uniform, unit-amplitude plane wave at 635 nm.
2. The PLUTO key at the relayed plane: an ideal 4f image with magnification m = 1.00935, i.e. a pitch of 8.075 µm
   (`key_slm_dx`). 414 × 414 px, centred on the optical axis.
3. z1 = 35.87 mm of free space.
4. The LUNA, 4.5 µm px: 736 × 736 px holding encoding(a) + decoder m, centred on the axis. Pixels outside the key and the
   LUNA region are flat, so the light there passes as the plane wave.
5. z2 = 175.9 mm of free space.
6. Camera detector boxes, centred on the axis.

Phases are exact pixel arrays with hard edges, 8-bit straight-through in training. **Not modelled:**
- rotation of either SLM or the camera;
- tilt, wavefront curvature or aberration of the beam or relay;
- SLM pixel crosstalk or fringing and flicker;
- camera noise and saturation;
- the measured phase tables. The model is linear 8-bit; the phase-table variant below adds them.

### Run it

Run from this folder. The checkpoint path is relative to it.

```python
import os, sys, numpy as np, torch
PKG = os.getcwd()                                                       # this folder
sys.path.insert(0, os.path.abspath(os.path.join(PKG, '..', '..')))     # code/
sys.path.insert(0, PKG)
from config import init_params, recompute_derived, apply_saved_config
from model import TimeMultiplexedNFA
from wave_prop import SpectralLabProp
import insitu_config

torch.set_grad_enabled(False); dev = torch.device('cuda')
ck = torch.load('ckpt_M2_z1759_pfaff_ft/bench_M2_bin1_z1759_pfaff_ft_ep0.pth', map_location='cpu', weights_only=False)
tc = init_params(); apply_saved_config(tc, ck['config']); tc.device = 'cuda'; recompute_derived(tc)   # exact training config
m = TimeMultiplexedNFA(tc).to(dev)
miss, _ = m.load_state_dict(ck['model'], strict=False)          # the propagator is rebuilt from the config, not stored
assert not [k for k in miss if 'phases' in k], miss
m.eval()

cfg = insitu_config.init_params()
a = torch.as_tensor(cfg.a, dtype=torch.float32, device=dev)     # the bench's 100 a values
T = torch.from_numpy(np.load('targets.npy')).float().to(dev)    # [100, 4096] targets for those a

I_frames = m(a[:1])                                             # [B, M, 64, 64]: detector intensity per frame m
def readout(a):                                                 # [len(a), 4096]: mean over the M frames, k = 64 r + c
    return torch.cat([m(a[i:i + 1]).mean(1) for i in range(len(a))]).reshape(len(a), -1)

def rmse_perfunc(I, T):                                         # the bench readout: each function min-max'd over a
    n = lambda X: (X - X.amin(0)) / (X.amax(0) - X.amin(0)).clamp_min(1e-12)
    return ((n(I) - n(T)) ** 2).mean(0).sqrt().mean().item()

print(rmse_perfunc(readout(a), T))                              # 8.3e-4 for this checkpoint (100 a)
```

100 a values take about 3 s at M2. Intensities are in units of the incident beam intensity.

### Changing the geometry without retraining

`m.spec_geom` holds the geometry dict. All values are in metres, in the model frame, with the optical axis at 0:

| key | meaning | value (z1759 models) |
|---|---|---|
| `z1`, `z2` | PLUTO image → LUNA, LUNA → camera | 0.03587, 0.1759 |
| `p1`, `nk` | key pitch at the relayed plane (= 8 µm × relay magnification), key size in px | 8.075e-6, 414 |
| `key_y0`, `key_x0` | corner of key px 0 (centred: −nk·p1/2) | −1.672e-3 each |
| `p2`, `luna_ny`, `luna_nx` | LUNA pitch and the size of the displayed region | 4.5e-6, 736, 736 |
| `luna_y0`, `luna_x0` | corner of LUNA region px 0 | −1.656e-3 each |
| `det_cy`, `det_cx` | 64 detector-centre coordinates per axis (pitch 8 × 5.86 µm) | centred arrays |
| `det_size` | detector box side (4 × 5.86 µm) | 2.344e-5 |

To simulate a misaligned bench, replace the propagator and read out again. This takes a few seconds; nothing is
retrained:

```python
g0 = dict(m.spec_geom)
m.spec = SpectralLabProp(tc, dict(g0, z2=g0['z2'] + 0.3e-3, key_x0=g0['key_x0'] + 2 * g0['p1'])).to(dev)
I = readout(a)
m.spec = SpectralLabProp(tc, g0).to(dev)                       # back to the trained geometry
```

- **Shifts:** `key_*0` / `luna_*0` move the whole key or LUNA region; adding a constant to `det_cy` / `det_cx` moves the
  detector grid.
- **Magnifications:** `p1` sets the relay magnification. Scaling `det_cy` / `det_cx` and `det_size` together sets the
  camera magnification.
- **Coordinate frames:** these are model-frame coordinates. The package maps them to the raw SLM and camera frames with
  the flips, `phase_key_center_*`, `pluto_shift_yx` and `det_axis_yx` in `insitu_config.py`. `make_sim_check.py`
  (`to_raw`) shows the mapping for the camera.

**How sensitive the readout is** (M2 `ckpt_M2_z1759_pfaff_ft`, per-function RMSE over the 100 a; trained geometry
8.3e-4):

| change | RMSE |
|---|---|
| z2 + 0.3 mm | 4.1e-2 |
| detector grid + 1 camera px in x | 1.1e-1 |
| LUNA + 1 px in y | 9.3e-2 |

`sim_check_M2/report.txt` has z1 ± 0.5 mm, key ± 3 px and relay m ± 0.005 for the older M2. All are 50–150× above
the trained value. The design tolerates sub-pixel and sub-mm errors only, so fit the alignment before judging the masks.

### Phases through the phase tables, and full camera frames

- **Phase-table variant:** `set_variant('table')` in `sim_check_M2/make_sim_check.py` swaps `m.key_phases` and
  `m._input_plane_px` for the phases the SLMs actually produce: conjugate + `pluto_phase_table` / `luna_phase_table`
  with the LUTs. This uses the same frame building as the package.
- **Full camera frames:** `camera(a)` in the same script gives the full 2048 × 2048 model-frame camera window, with every
  camera px as a box and the mean over the M frames. It runs on the CPU, about 20–30 s per a value. `to_raw()` turns it
  into the raw Basler frame [1200, 1920], with the same flips and `det_axis_yx` as captures.
- `TD.captures_to_detectors(raw[None], (0, 0), cfg, dev)` then reads it exactly as on the bench. It matches the model's
  detectors to 1e-6 of the readout range.
- For a full run, `make_sim_check.py <ckpt path>` writes SLM BMPs, expected camera images and a report. To scan z2 on
  camera frames, use `sim_z2_M2/make_z2_compare.py`. Edit `CKPT` / `Z2S` at the top; they hold the old checkpoint
  and z2 values.

### Training: how the bench models were made

Run from `code/`. The scripts are `code/logs/bench_z2_1759_2026-11-13/run_bench.sh` (training),
`ft2.sh` / `run_bench_ft.sh` (fine-tunes) and `eval_perfunc.py` (per-function test RMSE).

The z1759 per-function-affine run for M2 (epochs 0–6), as `run_bench.sh` builds it:

```
cd code
/c/Anaconda/envs/qpi/python.exe logs/lab_ED_Np4_Nf4096_05um/capmem.py train.py --log_dir logs/<run_name> --set \
  device_profile=lab num_layers=0 decoder=true decoder_over_encoding=true Nf=4096 Np=4 seed=59 \
  slm_phase_bits=8 prop_method=spectral spec_P_mult=1.0 camera_dx=5.86e-6 pd_pitch_cam=8 pd_size_cam=4 key_det_fill=0.9 \
  key_to_enc_spacing=0.03587 last_layer_ccd_spacing=0.1759 \
  key_slm_dx=8.075e-6 enc_slm_dx=4.5e-6 phase_key_size=414 decoder_size=736 encoding_patch_scale=110 \
  M=2 max_epoch=100 train_a_samples=10000 batch_size=8 grad_accum_steps=2 test_batch_size=4 \
  checkpoint_save=1 stop_epoch=6 train_readout=perfuncaffine
```

`capmem.py` caps the process at 85% of the GPU memory (`GPU_MEM_FRAC`) and then runs the script.

The flags, by group:

- **Profile:** `device_profile=lab` brings the lab defaults: 635 nm, one decoder per frame (`decoder_per_key`), the
  encoding tiled 3 × 3, and zero-based encoding (displayed phase 2π·p·a, p = 0..3).
  - `num_layers=0`: no diffractive layers.
  - `decoder=true decoder_over_encoding=true`: a learned decoder on the LUNA, also over the encoding pixels.
- **Task:** `Nf=4096` functions on 64 × 64 detectors; `Np=4` encoding pixels (2 × 2); `M` frames.
- **SLMs and camera:**
  - `slm_phase_bits=8`: 8-bit phases with a straight-through gradient.
  - `prop_method=spectral spec_P_mult=1.0`: the propagator above.
  - `camera_dx=5.86e-6 pd_pitch_cam=8 pd_size_cam=4`: Basler pixels; 4 px detectors on an 8 px pitch.
  - `key_det_fill=0.9`: no effect here, because `phase_key_size` is set explicitly.
- **Geometry:**
  - `key_to_enc_spacing` = z1 and `last_layer_ccd_spacing` = z2, in metres.
  - `key_slm_dx`: PLUTO pitch × relay magnification. `enc_slm_dx`: LUNA pitch.
  - `phase_key_size` / `decoder_size`: in PLUTO / LUNA px.
  - `encoding_patch_scale=110`: each encoding px is 110 × 110 LUNA px; tiled 3 × 3, that is 660 px.
- **Optimisation:**
  - Separate Adam optimisers for keys (`lr_slm`), decoders (`lr_decoder`) and the training readout (`lr_readout`,
    default = `lr_slm`), all at 1e-2.
  - Cosine to `lr × lr_min_ratio` (1e-3) over `max_epoch`; `train_a_samples` a values per epoch; `seed` 59.
  - `batch_size=8` with `grad_accum_steps` micro-batches (M1 8 → 1, M2 4 → 2, M3 2 → 4, M5 1 → 8 steps). The gradient
    is identical to the full batch (checked, `check_accum_grad.py`).
- **Saving and stopping:** `checkpoint_save=1` validates and saves every epoch. `stop_epoch=6` stops after epoch 6
  but keeps the 100-epoch lr schedule; M1 ran all 100 epochs without it. `best.pth` is the best validation.
- **Training readout (`train_readout`):**
  - `perfuncaffine`: a learned scale + offset per function against per-function-normalised targets. Validation is
    per-function min-max. This is what the `_pfaff` models use.
  - `affine`: a single global scale + offset; the default, used by the `ckpt_M*_z1759` models.
  - Both are training-only and are discarded. `test.py` always reports global min-max; per-function numbers come from
    `eval_perfunc.py`.
- **Fine-tune (the `_pfaff_ft` models):** append
  `load_weights_only=true ckpt_to_load=<run>/model/best.pth freeze_slm=true lr_slm=3e-4 lr_decoder=3e-4 max_epoch=3`
  (M3: `max_epoch=10`).
  - `load_weights_only`: weights only, with a fresh optimiser and schedule (without it, a run resumes where it
    stopped).
  - `freeze_slm`: the PLUTO keys stay fixed.

**To train for a measured geometry** (e.g. a fitted z2 or relay magnification), change only the geometry flags.
- Recompute `key_slm_dx` as 8 µm × m, rounded so that its lcm with 4.5 µm stays small; see `pitch_for()` in
  `make_sim_check.py`.
- Training time per epoch on one GPU (10000 a values): about 3.5 min at M1, 8 min at M2, 14 min at M3 and 40 min at
  M5, plus about 1 min of validation. The M5 run, at micro-batch 1, used about 11 GB.
- Long runs: launch detached (WMI `Win32_Process Create`) so they survive the session. `train.py` and `test.py` can
  segfault at start or exit on this machine; the `run_bench*.sh` scripts retry and resume from `last.pth`.

## 0e. Update 2026-11-15: per-function-affine models, fine-tuned (`ckpt_M*_z1759_pfaff_ft`)

These are the same z2 = 175.9 mm design as `ckpt_M*_z1759`, trained with a learned scale and offset per function (`ckpt_M*_z1759_pfaff`). They were then fine-tuned with the **PLUTO keys frozen**: only the LUNA decoders and the readout trained, at lr 3e-4. The keys are therefore identical to the starting model's (for M2, M3 and M5, the model after the first plain fine-tune). Every fine-tune gained only in its first epoch, so all four checkpoints are fine-tune epoch 0. **Use the per-function min-max readout on the bench.** The bench code is unchanged; only `--ckpt` / `ckpt_path` changes.

| folder | checkpoint | sim test, per-function min-max | before (`ckpt_M*_z1759_pfaff`) |
|---|---|---|---|
| `ckpt_M1_z1759_pfaff_ft/` | `bench_M1_bin1_z1759_pfaff_ft_ep0.pth` | 1.49e-2 | 1.58e-2 (ep50; ep99 1.50e-2) |
| `ckpt_M2_z1759_pfaff_ft/` | `bench_M2_bin1_z1759_pfaff_ft_ep0.pth` | 9.07e-4 | 1.09e-3 |
| `ckpt_M3_z1759_pfaff_ft/` | `bench_M3_bin1_z1759_pfaff_ft_ep0.pth` | 5.52e-4 | 6.45e-4 |
| `ckpt_M5_z1759_pfaff_ft/` | `bench_M5_bin1_z1759_pfaff_ft_ep0.pth` | 4.08e-4 | 4.53e-4 |

Source: `code/logs/bench_z2_1759_2026-11-13/*_pfaff_ft2_frozen*`. The folder READMEs give the full lineage.

## 0d. Update 2026-11-13 (evening): FINAL models at z2 = 175.9 mm (`ckpt_M*_z1759`)

The baseline design (key 414, decoder 736, encoding bin 110 tiled 3 × 3, Nf 4096, z1 35.87 mm) retrained with z2 = 175.9 mm instead of 175.35 mm. The simulation in `sim_z2_M2/` showed a 0.5 mm z2 error costs about 50× in RMSE. **The bench code is unchanged:** only `ckpt_path` / `--ckpt` changes. M comes from the checkpoint, so M5 runs as is, with 5 PLUTO frames per a.

| folder | checkpoint | sim test min-max | global affine (reward) | same model at z2 175.35 (`ckpt_M*`) |
|---|---|---|---|---|
| `ckpt_M1_z1759/` | `bench_M1_bin1_z1759_ep0.pth` | still training (val 4.93e-2) | – | 4.83e-2 / 2.70e-2 |
| `ckpt_M2_z1759/` | `bench_M2_bin1_z1759_ep5.pth` | 1.26e-3 | 6.63e-4 | 1.20e-3 / 5.0e-4 |
| `ckpt_M3_z1759/` | `bench_M3_bin1_z1759_ep1.pth` | 6.25e-4 | 2.41e-4 | 4.83e-4 / 2.6e-4 |
| `ckpt_M5_z1759/` | `bench_M5_bin1_z1759_ep2.pth` | 3.60e-4 | 1.72e-4 | – |

Source: `code/logs/bench_z2_1759_2026-11-13/`. M1 runs the full 100 epochs; re-copy its `best.pth` when it finishes.

`ckpt_M*_final/` hold the LAST-epoch checkpoints of the z2 175.35 models, not the best ones. They are for checking whether longer training gives a more robust design on the bench. Their READMEs give the numbers.

## 0c. Update 2026-11-13: bigger bench design, Nf 4096 (`ckpt_big_M1/M2/M3`)

Everything on the two SLMs and the camera grows. The distances, beam, wavelength, Np 4, Nf 4096 and `targets.npy` stay the same.

| | `ckpt_M*`, `ckpt_M*_enc55` | **`ckpt_big_M*`** |
|---|---|---|
| key (PLUTO) | 414 px, 3.34 mm | **520 px, 4.20 mm** |
| decoder (LUNA) | 736 px, 3.31 mm | **930 px, 4.19 mm** |
| encoding | bin 110 × 3 × 3 / bin 55 × 6 × 6, 660 px at decoder rows 38..697 | **bin 55 × 8 × 8, 880 px (3.96 mm) at decoder rows 25..904** |
| detectors, 64 × 64 | 4 px on an 8 px pitch, array 508 px (2.98 mm) | **7 px on a 10 px pitch, array 637 px (3.73 mm)** |

Why:
- The decoder sets the smallest spot on the camera, about λ·z2 / D: 34 µm for a 736 px decoder, 27 µm for 930 px.
- At an 8 px pitch the detectors were only about 1.4 spots apart; at 10 px they are 2.2 spots apart.
- The detector array stays inside the encoding block and the decoder, so the corner detectors are not starved.
- About 2.3× more of the light lands on the detectors: diffraction efficiency about 0.12, against about 0.05 for `ckpt_M*_enc55`.

**Checkpoints** (best validation of each run, min-max readout):

| folder | file | val RMSE | sim test (min-max) | for comparison |
|---|---|---|---|---|
| `ckpt_big_M1/` | `bench_big_M1_ep2.pth` | 1.19e-3 | pending (100-epoch run still training) | `ckpt_M1_enc55` 1.68e-2 |
| `ckpt_big_M2/` | `bench_big_M2_ep3.pth` | 8.16e-4 | 8.34e-4 | `ckpt_M2_enc55` 1.04e-3 |
| `ckpt_big_M3/` | `bench_big_M3_ep2.pth` | 4.56e-4 | 3.94e-4 | `ckpt_M3_enc55` 5.17e-4 |

- M2 and M3 ran epochs 0–6 only.
- M1 is a full 100-epoch run and is still training. Its best so far is epoch 2; it will be re-copied if a later epoch takes over.
- `sim_check_*` has not been made for these checkpoints.

### Code edits to run the big checkpoints

**Not applied to `insitu_TDNFA.py` in this folder**, which still runs the 414 / 736 checkpoints.
- **`big_design_edits/insitu_TDNFA.py`** is the edited copy. To use it, copy it over `insitu_TDNFA.py` in your bench copy of the package.
- `scan_pluto_shift.py` and the rest import `insitu_TDNFA`, so they follow automatically.

The 14 code changes, all in `insitu_TDNFA.py`; the comments in the copy are updated too:

| where | was | becomes |
|---|---|---|
| `ENC_IDS` | `np.tile(..., (3, 3)).repeat(110, 0).repeat(110, 1)` | `np.tile(..., (8, 8)).repeat(55, 0).repeat(55, 1)` (880 × 880) |
| `pluto_frame` | `- 207` ×2, `y0 + 414`, `x0 + 414` | `- 260` ×2, `y0 + 520`, `x0 + 520` |
| `luna_frame` | `sq[38:698, 38:698]` | `sq[25:905, 25:905]` |
| `luna_frame` | `- 368` ×2, `y0 + 736`, `x0 + 736` | `- 465` ×2, `y0 + 930`, `x0 + 930` |
| `det_origin` | `- 254` ×2 | `- 318` ×2 |
| `captures_to_detectors` | assert `+ 508` ×2; crop `+ 512` ×2 | assert `+ 640` ×2; crop `+ 640` ×2 |
| `captures_to_detectors` | `reshape(..., 64, 8, 64, 8)[..., :4, :, :4]` | `reshape(..., 64, 10, 64, 10)[..., :7, :, :7]` |
| `save_detector_crop` | crop `+ 516` ×2; rectangles `8 * c`, `8 * r`, size `4, 4` | crop `+ 648` ×2; rectangles `10 * c`, `10 * r`, size `7, 7` |
| `calibrate_roi` | `ya + 508`, `xa + 508`; red box `508, 508` | `ya + 637`, `xa + 637`; red box `637, 637` |

In `insitu_config.py` only the documentation fields change, plus `ckpt_path`:
- `key_size_px` 520, `decoder_size_px` 930, `encoding_bin` 55, `tiling` 64;
- `pd_size = 7 * c.cam_dx`, `pd_pitch = 10 * c.cam_dx`;
- `ckpt_path = './ckpt_big_M2/bench_big_M2_ep3.pth'`, or `_M1` / `_M3`.

The code reads the checkpoint, not these fields. `Nf`, `targets_path`, the distances, `det_axis_yx`, the flips and the LUT stay as they are.

**Checked** in a scratch copy against the `ckpt_big_M2` model, with the same check as `sim_check` step 1:
- `pluto_frame` and `luna_frame` reproduce the model's 8-bit levels. At most 12 px differ, by 1 level each; these are float32/float64 rounding ties, as in `sim_check_M1`.
- The model's encoding map equals the new `ENC_IDS` at rows 25..904 for all three checkpoints.
- On a synthetic camera frame (each 7 × 7 box holding its own index, rotated 180°), `captures_to_detectors` returns exactly k = 0..4095 in model order and reads no gap pixels.

**Half-pixel offset of the detector array (unavoidable at this pitch):**
- 63 × 10 + 7 = 637 px is odd, so the model's array, which is symmetric about the axis corner, starts at axis − 318.5 camera px.
- The package uses axis − 318, so every box sits 0.5 camera px (2.9 µm) off the trained position.
- That is 7% of a 41 µm box, against a 27 µm spot. I expect it to cost little, but it is a fixed sub-pixel error that `--calibrate-roi`, which searches whole pixels, cannot remove.
- A future retrain at an even span (for example 8 px boxes on a 10 px pitch, 638 px) would avoid it.

**Bench notes:**
- **SLM illumination:** the beam must cover the 4.2 mm key on the PLUTO and the 4.2 mm decoder on the LUNA. Before, these were 3.3 mm each.
- **Camera sensor:** the 640 px crop around `det_axis_yx` (497, 985) spans rows 179..818 and cols 667..1306 of the 1200 × 1920 sensor.
- **Exposure:** the 7 px boxes collect about 3× the light of the 4 px boxes, at about 2.3× the diffraction efficiency. Re-set `cam_exposure_us` with `--calibrate-roi`.
- **Phase table / key shift:** the models were trained linear, as before. Re-run `scan_pluto_shift.py` with the bigger key.

## 1. Script changes vs `two_slm_insitu/`

These changes are already made in this package. Each script is edited in place, with no new modules.

| file | change | why |
|---|---|---|
| `insitu_config.py` | `key_size_px` 120 → **414**; `decoder_size_px` 213 → **736**; `encoding_bin` 28 → **110**; `M` 1 (informational) | new design |
| `insitu_config.py` | `pd_size` / `pd_pitch` = **4 / 8 camera px** (23.4 / 46.9 µm); new `cam_dx` = 5.86 µm | Basler, detector design |
| `insitu_config.py` | `sen_x_num, sen_y_num` = **1200, 1920** (rows, cols); new `cam_exposure_us` and `cam_pixel_format = 'Mono12'` | Basler full sensor; weak detector signal (§3) |
| `insitu_config.py` | `pluto_lut_inverted = luna_lut_inverted = True` | RETRAINING §3 step 2: send −φ (conjugate phase) |
| `insitu_config.py` | `pluto_flip_ud = pluto_flip_lr = True` | §3 step 4: PLUTO rotated 180° (the relay inverts) |
| `insitu_config.py` | new `pluto_shift_yx = (-12, +30)`; `phase_key_center_y/x` = axis + shift | §3 step 1: PLUTO registration |
| `insitu_config.py` | new `use_phase_table = True`, `pluto/luna_phase_table` (the fit from §4), `pluto/luna_lut` (phase → level) | §3 step 3: measured phase vs level |
| `insitu_config.py` | `det_axis_yx = (501, 985)`; `cam_flip_ud = cam_flip_lr = True`; new `roi_search_px = 4` | measured axis at raw (500.1, 984.5); captures rotated 180° |
| `insitu_config.py` | `ckpt_path = ./ckpt_M1/bench_M1_8bit.pth` | |
| `insitu_TDNFA.py` | `ENC_IDS` 28 → 110 px bins (660 × 660); `pluto_frame` 414 px key; `luna_frame` 736 px decoder with the encoding at offset 38 | new layout |
| `insitu_TDNFA.py` | `q8(phi, inverted, lut)`: conjugates, then sends the level whose measured phase is nearest. A 4096-bin lookup is built in the config. | phase table |
| `insitu_TDNFA.py` | `det_origin` / `captures_to_detectors`: array origin = axis − 254 px; 4 × 4 px box mean on an 8 px pitch (was 1 px on a 2 px pitch) | new detectors |
| `insitu_TDNFA.py` | `calibrate_roi` searches around `cfg.det_axis_yx` (the measured axis), not the intensity centroid. At 169 mm the pattern is not centred on the axis. Search range `roi_search_px`; overlay box 508 px. | |
| `insitu_TDNFA.py` | `save_detector_crop`: 4 px boxes on an 8 px pitch | |
| `insitu_TDNFA.py` | `calibrate_roi`: saturation measured inside the detector array only (8 / 12 bit); prints the detector counts | the rim and open surround saturate by design |
| `insitu_TDNFA.py` | `_DummyCam` / `init_hardware`: pypylon interface, no `start_capture()` | Basler |
| `insitu_utils.py` | `init_cam`: **pypylon** instead of pyflycap2. Auto exposure, gain and gamma off; exposure and pixel format set; `StartGrabbing(LatestImageOnly)`. `caputre_img` uses `RetrieveResult`. | Basler |
| `insitu_calib.py` | comments only (mask shapes) | |
| `targets.npy` | re-exported from the M1 checkpoint (`export_targets.py`); identical to the old file | same Np, Nf, alpha and a grid |

`insitu_policy.py` (PPO and reward) and `export_targets.py` are unchanged.

## 2. Still to do or check on the bench

Nothing in the code needs changing for M1. These are settings and checks.

1. **Install:** `pip install pypylon` (Basler pylon), torch, numpy, scipy, matplotlib, tqdm and pillow, plus the
   Holoeye SDK (vendored in `holoeye/`). pyflycap2 is no longer needed.
2. **Camera node names:** `init_cam` sets whichever of `ExposureTimeAbs`/`ExposureTime` and `GainRaw`/`Gain` exists. Check
   the printed line, and in pylon Viewer that auto exposure, auto gain and gamma are off.
3. **Exposure (`cam_exposure_us`):** set it so the brightest spot of the trained masks reads about 200 counts.
   Set it from the **detector array**, not the frame. The detectors see only 0.02–0.13 of the open beam, while the rim
   just outside reaches 7–8× (§3). Let the rim and the open surround saturate, and aim the brightest array px at about
   80% of full scale. `--calibrate-roi` prints the array saturation (under 1%) and the detector counts.
4. **PLUTO shift order:** check that `pluto_shift_yx = (-12, +30)` means (rows, cols) in the model frame, as read here.
   If the bench record (`FORWARD_MODEL_CHANGES.md`) means (x, y), use (+30, −12). The sync test or the zero-shot reward
   will show which.
5. **Phase table:** the config uses the quadratic fit. To use the measured points in
   `verify_20261110-195101/lut_used.json`, set `c.pluto_phase_table` / `c.luna_phase_table` to the 256 measured phases
   (rad, level 0..255). The lookup rebuilds from them. `use_phase_table = False` sends linear levels.
6. **SLM order:** if the keys come up on the LUNA, add `--swap-slms`.
7. **Unmodulated light (u ≈ 0.17):** it adds a nearly uniform background and scales the pattern (camera ≈ 0.83 I + 0.17).
   The affine reward ignores it, but the captures will have less contrast than the sim images.
   If it is a polarization effect, an analyzer before the camera removes it.
8. **Relay magnification:** still unverified (RETRAINING §5). A non-1:1 relay scales the key at the LUNA, and the in-situ
   loop cannot fix that.

## 3. Sim sanity check (`sim_check_M1/`)

`make_sim_check.py` loads the M1 checkpoint with the encoder and decoder phases in place and runs at the training
distances. It writes:

**Learned masks, placed as the bench shows them** (`slm_bmp/`, full 1920 × 1080 frames for the Holoeye software)
- `pluto_key0_display.bmp`, `luna_dec0_a<+x.xxx>_display.bmp`: exactly what the package sends. Display corrections:
  - key shift (−12, +30);
  - conjugate phase;
  - phase table;
  - PLUTO rotated 180°.
  In the model frame the key is centred at row 528, col 990; after the flip it sits at row 552, col 930.
- `*_modelframe.bmp`: the plain trained pattern (linear levels, no corrections), for reference.
- **Frame check:** the package's `pluto_frame` / `luna_frame` reproduce the model's own levels. At most 21 of 171k key px
  and 4–7 of 542k LUNA px differ, each by 1 level: rounding ties, float32 on GPU vs float64 on CPU.

**Expected camera images** (`cmos_expected_<linear|table>_a<+x.xxx>.*`, RAW Basler orientation: 1920 × 1200, as captured)
- `.npy`: intensity in units of the open beam (float16).
- `.bmp`: level = 255 · I / 0.6, i.e. exposed for the detector array. The open-beam surround saturates.
- `_array.bmp`: the 508 × 508 px detector-array crop, 2× enlarged.
- `linear` = the phases as trained. `table` = the phases the SLMs actually produce through the measured tables; this one
  should match a capture.
- **Figures:** `cmos_expected.png` (whole sensor, with the detector array outlined), `cmos_expected_array.png` (array
  zoom), `detector_check.png` (4 px detector boxes on the spot grid, package readout vs targets).

What the camera should show:
- **Overall:** a dark square shadow about 3 mm across, with the detector array inside it (raw rows 247..754, cols
  731..1238).
- **Inside the shadow:** a grid of spots on an 8 px pitch, at **0.02–0.13 of the open beam per detector** (pixels up to
  about 1.0).
- **Rim:** a bright speckled rim just outside the array, reaching **7–8× the open beam**.
- **Surround:** the open beam (1.0) everywhere else.
- **Exposure:** set it from the array, not the frame. With an exposure that keeps the rim unsaturated, the detectors get
  only about 4 counts of 255. So `cam_pixel_format = 'Mono12'` is now the default, and `--calibrate-roi` measures
  saturation inside the array only and prints the detector counts.
- **Unmodulated light (u ≈ 0.17):** this adds about 0.17 to every pixel. That is more than the detector signal itself,
  so captures will look lower in contrast than the BMPs. The affine reward is unaffected.

**End-to-end readout check:** the package's `captures_to_detectors` on the simulated raw frames reproduces the model's
detectors to within 2e-4 of the readout range (RMS 1.6e-5), with the identical reward RMSE. The crop, the 180° flips,
the 4 × 4 boxes and the axis convention are therefore right.

**Expected zero-shot numbers:** what the in-situ eval should show if the bench matched the model. 100 a values;
"affine" = one scale + offset, the reward.

| case | min-max RMSE | affine (reward) RMSE |
|---|---|---|
| phases as trained (linear) | 3.56e-2 | 7.6e-3 |
| **through the measured phase tables** (what the bench shows) | 6.29e-2 | **2.70e-2** (3.5× worse) |
| z1 ± 0.5 mm (measurement uncertainty) | 1.1–1.4e-1 | 8.7e-2 (11×) |
| z2 ± 0.3 mm | 5.2e-2 / 1.0e-1 | 3.7e-2 (5×) |
| key shifted 3 PLUTO px (x or y) | 1.1e-1 | 8.9e-2 (12×) |

These masks are very sensitive. Every error within the measured uncertainties costs 5–12×, and the phase table alone
costs 3.5×. A poor zero-shot number on the bench is therefore expected; it is what the in-situ loop has to recover.
Two cheap steps before PPO:
- scan `pluto_shift_yx` by ±3 px and keep the best reward;
- a sim fine-tune of M1 with the measured phase table (and the RETRAINING §2 randomisation) would remove most of the 3.5×.

**Regenerate:** `make_sim_check.py` (GPU, about 10 min; the camera window runs on the CPU), then `make_cmos_bmps.py`.
Numbers are in `report.txt`.

## 4. Run order

1. **Dry run** (no hardware): `python insitu_TDNFA.py --dry-run --iterations 10 --states-num 2`
   (and `--calibrate-roi --dry-run`). Both were run here and passed.
2. **Sync tests:** check SLM registration first with the sync tests in `../two_slm_insitu/bench_check_M1_z10mm/sync_test/`.
   Those were made for 33 / 167 mm without the display corrections; re-run them at 29.7 / 168.8 mm if quantitative.
3. **Static check:** show `sim_check_M1/slm_bmp/pluto_key0_display.bmp` and `luna_dec0_a+0.005_display.bmp` in the Holoeye
   software. Compare the capture with `sim_check_M1/cmos_expected_table_a+0.005.bmp` and `_array.bmp`, both in raw camera
   orientation. You should see the dark square, the spot grid inside it, and the bright rim.
4. **Register the detectors:** `python insitu_TDNFA.py --calibrate-roi`. Paste the printed `det_axis_yx` / flips into the
   config. Check:
   - the runner-up should be several times worse than the best;
   - the best should not sit on the search edge;
   - in `roi_detector_grid.png` the cyan boxes should sit where `sim_check_M1/detector_check.png` puts them on the pattern.
5. **Zero-shot:** `python insitu_TDNFA.py --iterations 10`. Compare the epoch-0 `affine RMSE` and the epoch-9
   `[eval] min-max` with the sim numbers in §3.
6. **In-situ training:** `python insitu_TDNFA.py`.
   - Time per epoch: `states_num` 64 × `a_batch` 8 × M displays at about 0.25 s each, so about 2 min per epoch at M1.
     The dry run measured 0.29 s per display, including the 0.1 s settle.
   - To speed it up once stable: `slm_settle` 0.04, `capture_avg_n` 1.

## 5. Putting tuned masks back into a checkpoint

```python
import torch, numpy as np
ck = torch.load('./ckpt_M1/bench_M1_8bit.pth', map_location='cpu', weights_only=False)
ag = torch.load('./logs/insitu_<ts>/agent_final.pth', map_location='cpu')
p = (ag['u1'] % (2 * np.pi)) / (2 * np.pi)
ck['model']['slm_phases'] = torch.logit(p.clamp(1e-6, 1 - 1e-6))[:, None]   # [M, 1, 414, 414]; key = 2 pi sigmoid(raw)
ck['model']['decoder_phases'] = ag['u2'][:, None] % (2 * np.pi)            # [M, 1, 736, 736]
torch.save(ck, './ckpt_M1/bench_M1_8bit_insitu.pth')
```
Then `python test.py --ckpt ...` (from `code/`) evaluates the bench-tuned masks in simulation. The tuned masks absorb the
phase table, so simulate them with the table: `sim_check_M1/make_sim_check.py`, `set_variant('table')`.
