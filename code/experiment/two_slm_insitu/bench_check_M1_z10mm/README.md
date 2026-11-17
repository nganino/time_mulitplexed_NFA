# Bench forward-model check (2026-11-10)

This folder holds bench test material for the locked two-SLM setup: SLM frames as `.bmp` files, plus simulated camera images
to compare with captures. It is a **pre-training sanity check**. The new long-distance models (`logs/bench_spectral_2026-11-10/`)
are not launched until the bench reproduces these simulations.

> **Superseded distances (2026-11-10, later).** The bench measurements in `code/RETRAINING_CHANGES.md` give z1 = 29.7 mm
> and z2 = 168.8 mm, with the camera a Basler acA1920-40gm (1920 x 1200, 5.86 um). The files here were made for 33 / 167 mm.
> Re-run both scripts with the new `Z1` / `Z2` before comparing captures quantitatively.
> The display-side corrections (PLUTO shift, conjugate phase, phase table, 180 deg rotations) are not applied to these BMPs.

## Locked bench geometry (what everything here assumes)

```
635 nm collimated beam
 -> PLUTO (SLM1, 8 um px, 1080 x 1920)
 -> 1:1 relay; z1 = 33 mm from the PLUTO image plane to the LUNA
 -> LUNA (SLM2, 4.5 um px, 1080 x 1920)
 -> z2 = 167 mm free space (no relay)
 -> CMOS (5.86 um px; sensor assumed 1920 x 1200, not confirmed)
```

- **Optical axis:** the pixel corner between rows 539/540 and cols 959/960 on both SLMs. Every pattern here is centred
  there, the same as the trained masks.
- **Grey level:** level k means phase 2πk/256. Both SLMs need a linear 0–2π LUT at 635 nm.
- **Image orientation:** the simulations use model coordinates, with row 0 at the top. They apply no relay inversion and
  no camera-mounting flips.

## Part 1: trained M1 masks re-simulated (`make_bench_check.py`)

**Checkpoint:** `../ckpt/z2_10mm_M1_8bit.pth`.
- It was trained for z1 = 3 mm and z2 = 10 mm, with a 120 × 120 PLUTO-px key and a 213 × 213 LUNA-px decoder.
- The 2 × 2 encoding uses 28-px bins, tiled 3 × 3.

**Purpose:** a check that the simulation predicts what the camera actually sees. At 33 / 167 mm these masks do **not**
compute the functions: the min-max RMSE is 0.20, against 6e-3 at the trained distances.

**Output files**

| File | What it is |
|---|---|
| `pluto_key_M1.bmp` | full PLUTO frame; key at rows 480–599, cols 900–1019, level 0 elsewhere |
| `pluto_key_M1_rot180.bmp` | the same rotated 180°, for when the relay inverts the image |
| `luna_M1_a{-0.50,-0.25,+0.00,+0.25,+0.50}.bmp` | full LUNA frame per input a: encoding(a) + decoder, quantised once; decoder at rows 434–646, cols 854–1066 |
| `cmos_sim_a*.npy` | simulated camera intensity [2048, 2048] at 5.86 µm px (12 mm window), in units of the incident beam; the axis sits between px 1023 and 1024 |
| `cmos_sim_a*_sensor.bmp` | 1920 × 1200 crop |
| `cmos_sim_a*_center.bmp` | 512 × 512 crop |
| `cmos_sim_a*_center_x3.bmp` | the 512 × 512 crop upscaled 3× |
| `cmos_sim_locked.png` | overview figure |
| `report.txt` | the numbers behind the checks |

- **BMP brightness scale:** linear, level = 255 · I / 1.8, so the open beam sits at level 142. Produced by
  `make_cmos_bmps.py`, which reads the `.npy` files.
- **a = ±0.5:** the +0.5 and −0.5 LUNA frames are identical, because the phase wraps.
- **Expected camera image:** a dark square shadow about 1.2 mm across, with a bright rim at about 1.5× the beam intensity
  and fringes around it. Changing a alters the pattern only about 9% of its own contrast.

**How the masks are built**
- Masks come from the model itself: `model.key_phases()` and `model._input_plane_px(a)`, both already 8-bit.
- They were cross-checked against the in-situ `luna_frame` construction and agree on all but one px per frame.

**Checks**
- At 3 / 10 mm the spectral propagator reproduces the checkpoint: RMSE 6.09e-3 vs the ASM simulation's 5.95e-3, and a
  readout difference of 2.4e-3 of the range.

## Part 2: sync / registration tests (`sync_test/make_sync_test.py`)

Unlearned patterns that give an unambiguous camera image. Run test 1 first.

**Test 1 `test1_F_*`: orientation + coarse registration**
- **PLUTO:** an 80 × 60 px letter F filled with 0/π stripes (period 2 px), flat elsewhere. The stripes throw the light
  ±1.3 mm sideways at the LUNA, so the F appears dark.
- **LUNA:** a lens with f = z1·z2/(z1 + z2) = 27.555 mm, inside a 0.6 mm-radius disk with a notch cut out (x > 0.3 mm,
  y < −0.1 mm, i.e. the top right). Elsewhere it shows a 0/π checkerboard.
- **What it does:** the lens images the PLUTO onto the CMOS at −z2/z1 = −5.06×.
- **Camera:** a ringed bright disk about 6 mm across with a dark F, both rotated 180°.
  - **Notch:** shows the LUNA orientation.
  - **F:** shows the PLUTO orientation.
  - **F offset from the disk centre:** −5.06 × the PLUTO-vs-LUNA misregistration.
  - **Rings:** real, caused by the pixelated lens's phase wraps.

**Test 2 `test2_pinhole_*`: pass / fail fine registration**
- **PLUTO:** a lens with f = z1 = 33 mm, 1.5 mm across. It focuses to a spot about 25 µm across on the LUNA.
- **LUNA:** a 10 × 10 px (45 µm) open pinhole, checkerboard elsewhere.
- **Camera:** a smooth bright disk about 7 mm across when aligned; about 30× less light at a 32 µm offset.

**Output files**
- `test*_pluto.bmp`, `test*_luna.bmp`: full SLM frames.
- `inverted_lut/`: the same frames with level → (256 − level) % 256, i.e. the conjugate phase.
  - Use these if nothing focuses, which means the SLM phase sign is opposite to the model's.
  - If they work, the in-situ config needs `*_lut_inverted = True`.
- `test*_cmos_sim_{aligned,PLUTO_shift_x3px|x4px}.npy` and `_sensor.bmp`: simulated camera images.
  - BMP scale: 255 = 1.1 × the aligned maximum. The value is in `report.txt`.
- `sync_tests.png`: overview figure.

## Simulation method

- **Propagator:** `code/wave_prop.py` `SpectralLabProp`, the exact per-plane spectral method.
  - Pixel spectra are computed as DFT × sinc on a lattice whose period is a whole number of both pitches.
  - The method is validated against brute-force propagation (see memory / `logs/multigrid_study_2026-11-10`).
- **Camera:** each camera pixel is treated as one detector box (5.86 µm, 4 × 4 Gauss-Legendre nodes). The field is
  evaluated with a separable inverse DFT, in row chunks.
- **Part 1:** the LUNA outside the decoder is transparent, written as 1 + d. The full forward includes the unmodulated
  plane wave, so the background is 1 everywhere.
- **Part 2:** the LUNA checkerboard is modelled as **opaque**. Its light leaves at 0.07 rad per axis, about 11.8 mm
  off-axis at the camera, which is outside the simulated band.
  - So the LUNA field is t = window · e^{iφ}, with spectrum c1·T + V1 ⊛ T.
  - It drops the V1 and c12 terms of `SpectralLabProp.forward`; the custom forward is in `camera()`.
- **Sign convention:** the propagator uses H = exp(+i 2π z kz), so a converging lens is exp(−iπr²/(λf)).
- **Not modelled:**
  - the finite beam profile;
  - SLM fringing and fill factor;
  - leakage from imperfect π levels, which shows up as a faint even background;
  - the relay aperture;
  - polarisation.

## Regenerating / changing

- **Interpreter:** run with `/c/Anaconda/envs/qpi/python.exe`. Plain `python` is the Windows Store alias.
- **Requirements:** a GPU, and this repo's `code/` folder; the scripts add it to `sys.path`.
- **Run times:** a few minutes each script; GPU memory is a few GB (the large arrays are about 8000² complex64), capped at
  0.85 of the GPU.
- **Distances:** set by `Z1` and `Z2` at the top of each script.
- **Camera window:** `NCAM` (default 2048 px = 12 mm).
- **Other a values:** edit `A_BMP` in `make_bench_check.py`.
- **Test-pattern registration shifts:** the `shift_px` argument of `camera()`, in PLUTO px, moves the PLUTO pattern.
- **Other checkpoints:** change `CKPT`. The LUNA BMP build hard-codes the 213 px decoder and 28 px bin (asserted), so a
  new design needs those placement lines updated.
