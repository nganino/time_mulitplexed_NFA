# In-situ time-multiplexed NFA (two-SLM bench)

Runs a trained lab checkpoint on the real optics and fine-tunes it there. The keys go on the PLUTO, the encoding +
decoder on the LUNA, and the camera reads 64 x 64 detectors. Fine-tuning is PPO over the M keys (u1) and the M
decoders (u2): the loop displays noisy variants, scores the camera readout against the targets, and moves the masks
toward the variants that scored better.

The code started as a PPO in-situ QPI starter script and was changed as little as possible to match this model.
Values are hard-coded where that keeps the script easier to read and debug.

## What the bench must reproduce

The checkpoints are trained for this geometry (`device_profile=lab` in `code/config.py`):

```
635 nm, wide collimated beam (flat over the key; no hard aperture near it)
 -> PLUTO (8 um px): key m, 120 x 120 px (0.96 mm), centred; flat phase elsewhere
 -> z1 = 3 mm free space (with reflective SLMs: a 1:1 4f relay imaging the PLUTO to a plane 3 mm before the LUNA)
 -> LUNA (4.5 um px): decoder m, 213 x 213 px, centred, plus the encoding:
        Np 4 = 2 x 2 encoding pixels of 28 x 28 LUNA px each, phase 2 pi * p * a for p = 0..3,
        tiled 3 x 3 -> 168 x 168 px block in the decoder centre (display = encoding + decoder, mod 2 pi, 8 bit)
 -> z2 = 5 mm (or 10 mm) free space, realised by a 1:1 4f relay in front of the camera
 -> camera, 6 um px at 1x: detector (r, c) = the single camera px at (axis - 63 + 2r, axis - 63 + 2c),
        r, c = 0..63, function k = 64 r + c
readout: mean over the M frames -> min-max over all detectors and all a (test.py metric)
```

The 8-bit level k corresponds to phase k * 2 pi / 256. Both SLMs need a LUT that is linear over 0-2 pi at 635 nm.

## Files

| file | role |
|---|---|
| `insitu_TDNFA.py` | entry point: CLI, frame building (`pluto_frame`, `luna_frame`, `q8`), `capture_state` (M outer, a inner, mean over M), `captures_to_detectors`, `calibrate_roi`, PPO loop, evaluation |
| `insitu_config.py` | everything you edit: SLM sizes / pitches, LUT and flip flags, key / decoder / encoding sizes, `a` grid, camera frame size, detector registration (`det_axis_yx`, `cam_flip_*`), PPO and timing settings, `ckpt_path` |
| `insitu_policy.py` | `PolicyGradientAgent2SLM` (PPO over u1, u2) and `compute_reward` |
| `insitu_calib.py` | `load_trained_masks(ckpt)`: keys = 2 pi sigmoid(raw `slm_phases`) [M, 120, 120]; decoders = `decoder_phases` mod 2 pi [M, 213, 213] |
| `insitu_utils.py` | SLM / camera init (Holoeye SDK, pyflycap2), frame capture and averaging, drift monitor |
| `export_targets.py` | writes `targets.npy` [100, 4096]: the checkpoint's targets on `cfg.a`, min-max'd over that grid as test.py does |
| `targets.npy` | targets for every checkpoint in `ckpt/` (all are Np4 Nf4096, alpha 0..3, so the targets are identical) |
| `ckpt/` | 8-bit lab checkpoints, z2 = 5 mm and 10 mm, M1-M5; `ckpt/README.txt` lists their sim test RMSE |
| `holoeye/` | vendored Holoeye SDK (do not modify) |

M is taken from the checkpoint; `cfg.M` is not used. Change the model with `--ckpt`.

## Reward and evaluation

- **Reward** (`compute_reward`): for each noisy state, fit one scale and one offset (least squares, shared by all
  4096 detectors and the `a_batch` = 8 a values of that epoch), then take -mean_k RMSE_k. Laser power, exposure and
  the camera dark level do not affect it. One shared scale + offset matches the model's training readout.
- **Evaluation** (every `checkpoint_every` epochs, noise-free masks, all 100 a values): the test.py metric, min-max
  over all detectors and a, mean of the per-function RMSE. Epoch 0 is the zero-shot number, to compare with
  `ckpt/README.txt`.

## Run order

1. **Install** on the bench PC: Holoeye SLM Display SDK, FLIR FlyCapture2, then
   `pip install torch numpy scipy matplotlib tqdm pyflycap2 configobj`. `export_targets.py` (and the dry run with a
   checkpoint) also needs this repo's `code/` folder next to `experiment/`.
2. **Dry run**, no hardware:
   `python insitu_TDNFA.py --ckpt ./ckpt/z2_5mm_M1_8bit.pth --dry-run --iterations 2 --states-num 4`
3. **Edit `insitu_config.py`:**
   - `sen_x_num` / `sen_y_num` (still 1536 placeholders) and `cam_num`;
   - the SLM resolutions;
   - `pluto_lut_inverted` / `luna_lut_inverted`.
   Turn off the camera's auto-exposure, auto-gain and gamma.
4. **Check the display order:** add `--swap-slms` if the keys come up on the LUNA.
5. **Register the detectors:**
   `python insitu_TDNFA.py --ckpt ./ckpt/z2_5mm_M1_8bit.pth --calibrate-roi`.
   This shows the trained masks at 8 a values for each of the 4 PLUTO flips. It then searches the optical axis
   (+-4 px around the intensity centroid) and the 4 camera flips, scoring each with `compute_reward`. It prints the
   `det_axis_yx`, `cam_flip_*` and `pluto_flip_*` lines to paste into the config. Check:
   - the runner-up should be many times worse than the best;
   - the best axis must not sit on the edge of the search (if it does, re-run with that axis set);
   - fewer than 1% of pixels may be saturated;
   - in `roi_detector_grid.png` the cyan squares should sit on the spots.
   Repeat after any realignment and when switching between 5 mm and 10 mm.
6. **Zero-shot evaluation:** a short run (e.g. `--iterations 10`). Read the `[eval]` min-max RMSE at epoch 0 for M1, M2,
   M3 and M5 at 5 mm.
7. **In-situ training:** `python insitu_TDNFA.py --ckpt ./ckpt/z2_5mm_M3_8bit.pth`.
   - **Time per epoch** is `states_num` x `a_batch` x M displays at ~0.2 s each (`slm_settle` 0.1 s + 2 frames):
     ~1.7 min at M1, 3.4 min at M2, 5 min at M3, 8.5 min at M5.
   - **To speed it up** once the loop is stable: lower `slm_settle` to ~0.04 and `capture_avg_n` to 1. Fewer
     `--states-num`, or `--freeze-pluto`, also cut the time.

Suggested order: M1 at 5 mm first (simplest model, clearest spots, best for calibration), then M2, M3 and M5. The 10 mm
set is the fallback if the 5 mm relay is impractical.

### Not calibrated in software

`calibrate_roi` finds only the camera registration and the flips. Two things must be right optically:
- **Key-to-encoding registration:** the centre of the key image on the LUNA plane must land on the encoding-block
  centre, laterally and at 3 mm. If the zero-shot RMSE is poor, scan `phase_key_center_x/y` by a few PLUTO px.
- **The two distances (3 mm and z2).**

## CLI flags

```
--ckpt PATH        checkpoint (overrides cfg.ckpt_path)
--iterations N     epochs (default 500)
--states-num N     noisy states per epoch (default 64)
--sigma F          exploration noise in radians (default 0.04)
--lr F             PPO lr (default 1e-2)
--freeze-pluto     do not update the keys
--freeze-luna      do not update the decoders
--run-dir PATH     output folder (default ./logs/insitu_<timestamp>)
--swap-slms        swap the SDK enumeration order of the two SLMs
--dry-run          dummy SLMs and camera
--calibrate-roi    detector registration only (no training)
```

## Outputs (`logs/insitu_<timestamp>/`)

```
config.json, log.txt                   effective config; per-epoch reward, drift, [eval] RMSE
agent_epNNNN.pth, agent_final.pth      u1 (keys, rad) [M,120,120], u2 (decoders, rad) [M,213,213], sigmas, optimizer
eval_epNNNN_detectors.npy              noise-free detector readings [100, 4096] at each checkpoint
eval_epNNNN_detgrid.png / .npy         detector-region crop with the detector grid (every det_debug_every epochs)
rewards.npy / rewards.png              reward curve
drift_reference.*, drift/              drift monitor
roi_raw_overlay.png, roi_detector_grid.png, roi_histogram.png     (--calibrate-roi)
```

## Putting tuned masks back into a checkpoint

```python
import torch, numpy as np
ck = torch.load('./ckpt/z2_5mm_M3_8bit.pth', map_location='cpu', weights_only=False)
ag = torch.load('./logs/insitu_<ts>/agent_final.pth', map_location='cpu')
p = (ag['u1'] % (2 * np.pi)) / (2 * np.pi)
ck['model']['slm_phases'] = torch.logit(p.clamp(1e-6, 1 - 1e-6))[:, None]   # model: key = 2 pi sigmoid(raw)
ck['model']['decoder_phases'] = ag['u2'][:, None] % (2 * np.pi)
torch.save(ck, './ckpt/z2_5mm_M3_8bit_insitu.pth')
```
Then `python test.py --ckpt ...` (from `code/`) shows how the bench-tuned masks perform in simulation.
