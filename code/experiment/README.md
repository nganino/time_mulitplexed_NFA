# SLM + camera code from the single-pixel / single-shot QPI setups

Three independent pieces of code, taken from our setup at UCLA.
Each folder runs on its own, but all of them need hardware drivers and settings for your own system.

```
for_colleague/
  sgd_hologram_project.py      1. SLM phase pattern that draws a target (e.g. the detector) on the camera
  single_pixel_acquire/        2. one SLM: for each object, display every mask and save the captures
      acquire.py
      utils.py                    SLM / camera helpers (Holoeye + Basler)
      holoeye/                    Python wrapper of the Holoeye SDK
  two_slm_insitu/              3. two SLMs (Pluto + Luna) + camera, used for in-situ fine-tuning
      insitu_qpi.py               main script
      insitu_utils.py             SLM / camera helpers (two Holoeye SLMs + FLIR camera)
      insitu_config.py            all settings
      insitu_policy.py            the optimiser (PPO) of the two masks
      insitu_calib.py             loads the trained masks, makes calibration objects
      README.md                   detailed guide for this folder
      holoeye/                    Python wrapper of the Holoeye SDK
```

---

## What you need to install

| What | Needed by | Notes |
|---|---|---|
| Python 3.9+, `numpy scipy matplotlib pillow torch torchvision tqdm` | all | a CUDA GPU makes 1. much faster |
| **Holoeye SLM Display SDK** | 2, 3 | The `holoeye/` folders are only the Python wrapper. It loads the SDK's DLL from the folder in the environment variable `HEDS_3_2_PYTHON`, which the SDK installer sets. Without the SDK installed, `import holoeye` works but opening the SLM fails. |
| **Basler pylon** + `pip install pypylon` | 2 | camera used in `single_pixel_acquire/utils.py` |
| **FLIR FlyCapture2** + `pip install pyflycap2` | 3 | camera used in `two_slm_insitu/insitu_utils.py` |

**Different camera?** Rewrite the camera functions only: `init_cam`, `set_config_cam` and `capture_img` in `utils.py`, or `init_cam` and `caputre_img` in `insitu_utils.py`.
The rest of the code only needs a function that returns one frame as a 2D uint8 array.

**Different SLM?** Replace `init_slm()` and the `slm.showData(...)` calls with your SLM's API.
The scripts send one full-resolution (1080 x 1920) array per pattern.

---

## 1. `sgd_hologram_project.py` — pattern that shows the detector on the camera

Finds one phase pattern for the SLM so that the light intensity at the camera, a distance `FOCUS_DIST_M` behind the SLM, looks like a target image.
We use it to light up the detector region(s) on the camera, so you can see where the detector is.

**How it works**

1. Learn a phase over the full SLM, `phase = (sin(latent) + 1) / 2 * 2 pi`, rounded to 8-bit (256 levels) in the forward pass. Gradients pass through the rounding (straight-through).
2. Propagate the field to the camera with the angular-spectrum method. The SLM is zero-padded to twice its size, and spatial frequencies are limited by `THETA0`.
3. Loss = MSE(camera intensity, target) − `LAMBDA_ENERGY` × (light inside the target area / all light). The second term pushes light into the target area.
4. Optimiser: AdamW, `LR` 1e-3, `NUM_EPOCHS` 5000.

**Set these at the top of the file**

| Setting | Our value | Meaning |
|---|---|---|
| `TARGET_IMAGE` | `sensor_target_3x3_9_0_700_700.png` | grayscale target image, white = light. **Not included: make your own** (e.g. white squares at the detector positions on black). Centred on the camera; cropped if larger than the SLM. |
| `WLENGTH_VC` | 635e-9 | wavelength (m) |
| `FOCUS_DIST_M` | 0.0396 | SLM -> camera distance (m) |
| `DX`, `DY` | 8e-6 | SLM pixel pitch (m). The target is assumed on the same 8 um grid, so rescale it if the camera pixels differ. |
| `SLM_H`, `SLM_W` | 1080, 1920 | SLM resolution |
| `THETA0` | 75 | largest propagation angle kept (deg) |
| `LAMBDA_ENERGY` | 0.1 | weight of the "light on the target" term |

**Run:** `python sgd_hologram_project.py`

**Output** (`sgd_hologram_data/<time>/`):
- `final_slm_phase.bmp`: the 8-bit pattern to display (0 = phase 0, 255 = phase ~2 pi).
- `target.png`, `loss_curve.png`, and every 500 epochs the phase (`phase/`) and the simulated camera image (`simulated/`).

**Check on your SLM:**
- **Orientation:** the pattern is saved as computed, without any flip. If the image on the camera comes out mirrored, flip the BMP.
- **Phase mapping (LUT):** on our single-pixel setup the SLM LUT was inverted (255 = phase 0, see 2.).

---

## 2. `single_pixel_acquire/acquire.py` — display masks, capture, one folder per object

For each object (sample) folder, it displays every phase mask on the SLM, captures the camera, and saves the images inside that object's folder.

**Input layout**

```
calibration/                     <- --calibration_dir (all sample_* folders are processed)
  sample_0000_label7/            <- --sample_dir (one object)
    mask_00_slm.bmp              1080 x 1920 8-bit grayscale SLM pattern per mask
    mask_01_slm.bmp
    ...
  sample_0001_label2/
    ...
```

**What it does, per object**

1. Displays a blank SLM (all 255 = phase 0 with our inverted LUT) and captures a dark/reference frame.
2. For m = 0 ... M−1:
   - displays `mask_m_slm.bmp`;
   - waits `WARMUP_DELAY_S`;
   - throws away one frame (`DISCARD_FIRST`);
   - averages `N_AVG` frames;
   - saves the result.

**Output** (`<sample folder>/masks_meas/`, or `--out_dir`):

```
mask_00_meas.bmp, mask_01_meas.bmp, ...   8-bit camera frames (full frame, raw values)
dark_meas.bmp                              frame with the blank SLM
acquire_log.json                           settings, timestamps, image statistics per mask
```

**Run**

```
python acquire.py --sample_dir      D:\data\calibration\sample_0000_label7
python acquire.py --calibration_dir D:\data\calibration
python acquire.py --calibration_dir D:\data\calibration --n_avg 5 --exposure_ms 10
```

**Settings at the top of `acquire.py`:**
- **Camera:** `EXPOSURE_MS`, `GAIN`, `FPS`, `CAM_WIDTH`/`CAM_HEIGHT` (ours 1936 x 1216) and the offsets.
- **Timing:** `N_AVG`, `WARMUP_DELAY_S`, `BETWEEN_FRAMES_S`, `DISCARD_FIRST`.
- **SLM:** `SLM_H`, `SLM_W`.
- **Default folders:** `DEFAULT_SAMPLE_DIR` and `DEFAULT_CALIBRATION_DIR` point to our machine, so **change them, or always pass the folder on the command line**.

`acquire.py` and `utils.py` must stay in the same folder, with `holoeye/` next to them.

---

## 3. `two_slm_insitu/` — two SLMs + camera

The code we use to drive two Holoeye SLMs (Pluto and Luna) and the camera together, inside an in-situ fine-tuning loop (PPO) of a two-mask diffractive QPI system.
**Read `two_slm_insitu/README.md`.** It explains the setup, a dry run without hardware, ROI calibration and every option.

**Parts you can reuse on their own** (in `insitu_utils.py` / `insitu_qpi.py`):

| Function | What it does |
|---|---|
| `init_two_slms()` | opens the two SLMs, in the order the OS lists the displays. If they come up swapped, use `--swap-slms` or swap the two names. |
| `init_cam()`, `caputre_img(cam)` | opens the FLIR GigE camera, grabs one frame (the typo is in the original code) |
| `make_slm_pattern(u, obj_phase)` | radians -> the [0, 1] float array the Holoeye SDK expects (0 = phase 0, 1 = 2 pi) |
| `display_and_capture(...)` | puts one pattern on each SLM, waits, returns an averaged camera frame |

**To run the full loop you also need:**
- A trained two-mask checkpoint (`ckpt/epoch=1440.pth`).
- Optionally `warp_params.npy` (the warp between the two SLMs).

Neither is included, since both are specific to our system. Set the paths in `insitu_config.py`.
A dry run with no hardware still needs the checkpoint:

```
python insitu_qpi.py --ckpt ./ckpt/epoch=1440.pth --dry-run --iterations 2 --states-num 4
```

**For a plain two-SLM acquisition (no optimisation):**
1. Open both SLMs with `init_two_slms()` and the camera with `init_cam()`.
2. For each pair of patterns, call `pluto.showData(p)` and `luna.showData(l)`.
3. Wait about 0.5 s, then average a few `caputre_img(cam)` frames and save them.

This is the same loop as in `acquire.py` (2.), with a second `showData`.

---

## Things that differ between our setup and yours

- **Hardware:** SLM resolution and pixel pitch (ours 1920 x 1080, 8 um), camera model and frame size, and wavelength (635 nm).
- **SLM phase LUT:** inverted on our single-pixel setup (255 = 0 phase). Check it on yours before trusting any pattern.
- **Orientation:** mirror or flip between the simulated pattern and the real SLM, and between the SLM and the camera.
- **Timing:** SLM settling time (`WARMUP_DELAY_S`, 0.5 s here) and camera exposure (avoid saturation).
- **Paths:** all default paths in the scripts point to our machines.
