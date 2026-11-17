# Retraining changes for the vsilkog bench

These are the changes to make before retraining, based on measurements made on 2026-11-10 (PLUTO-2.1, LUNA, Basler
acA1920-40gm, 635 nm). The full measurement record, with methods and uncertainties, is in `FORWARD_MODEL_CHANGES.md`.
The data are in `C:\nicholas\bench_check\runs\` on the bench.

## 1. Forward-model changes for training

| # | parameter | old value | new value | source / uncertainty |
|---|---|---|---|---|
| 1 | PLUTO image -> LUNA distance, z1 (`key_to_enc_spacing`) | 3 mm (checkpoints) / 33 mm (bench README) | **29.7 mm** | PLUTO Fresnel-lens focus (z1 + z2 = 198.6 mm) minus z2; +-0.5 mm; assumes an exact 1:1 relay |
| 2 | LUNA -> camera distance, z2 (`last_layer_ccd_spacing`) | 5-10 mm / 167 mm | **168.8 mm** | LUNA Fresnel-lens focus sweep; +-0.3 mm |
| 3 | camera pixel (`camera_dx`) | 6 um | **5.86 um** | Basler acA1920-40gm |

Unchanged: wavelength 635 nm, PLUTO pitch 8 um, LUNA pitch 4.5 um, SLMs 1080 x 1920, ideal 1:1 relay. Training keeps
the linear 8-bit quantizer (level k = 2 pi k / 256); the measured phase table (former item 4) is deferred, see section 4.

| | pi reached at level | phase at level 255 |
|---|---|---|
| LUNA | 106 | about 312 deg (strongly nonlinear) |
## 2. Training robustness (suggested)

Randomize within the measured uncertainties:

- z1 +-0.5 mm
- z2 +-0.3 mm
- PLUTO key position +-3 PLUTO px
- phase scale +-3 %

## 3. Display and camera side (not in the model)

Apply these when showing the trained masks and reading the camera:

| step | PLUTO | LUNA |
|---|---|---|
| 1 | shift by (-12, +30) PLUTO px (model frame) | - |
| 2 | conjugate phase (send -phi) | conjugate phase (send -phi) |
| 3 | measured phase table -> device level | measured phase table -> device level |
| 4 | rotate the frame 180 deg (the relay inverts it) | no flip |

- **Camera:** rotate captures 180 deg into the model frame. The optical axis is at raw camera px (500.1, 984.5).
- **Registration:** with the shift, the PLUTO and LUNA centers moved from 271 um apart to 23 um (residual about
  +-3 PLUTO px).

## 4. Deferred to after training (they would slow convergence)

Add these to the forward model when evaluating or fine-tuning the trained masks, not during training:

| effect | measured | how to include later |
|---|---|---|
| unmodulated light | u = 0.155 (LUNA), 0.194 (PLUTO) | I = (1 - u) \|F(phi_PLUTO, phi_LUNA)\|^2 + u \|F(0, 0)\|^2, u ~ 0.17; or remove it in hardware with an analyzer before the camera, if it is a polarization effect |
| pixel crosstalk (fringing) | zero order at pi: 1-px checkerboard 0.41 (LUNA) / 0.55 (PLUTO) | Gaussian blur of the displayed phase, sigma ~ 0.4 SLM px |
| illumination | finite beam overfilling the LUNA; flat needs ~0.5 s | measured flat as the beam profile; full unmodulated surround |
| phase vs grey level (former item 4) | measured phase table per SLM (below) | in the model's 8-bit quantizer, map level k to the measured phase phi_dev(k) instead of 2 pi k / 256, so the optimizer only uses phases the SLMs can produce |

### Measured phase vs level (former item 4)

| | pi reached at level | phase at level 255 |
|---|---|---|
| LUNA | 106 | about 312 deg (strongly nonlinear) |
| PLUTO | 155 | about 315 deg |
| old model | 128 | 360 deg |

- **Tables:** `verify_20261110-195101/lut_used.json` (interpolated measured points, the recommended source).
- **Raw points:** `cal_20261110-190603/analysis/results.json` (`lut_<slm>.k`, `lut_<slm>.fit.phi_meas_deg`).
- **Fit:** phi(k) = 2 pi / 256 (s k + q k^2 / 256); LUNA s = 1.445, q = -0.576; PLUTO s = 0.751, q = +0.129.
- **Consequence of deferring:** the masks are trained with linear 0-2 pi phases. At display, section 3 step 3 maps each
  phase to the device level with the nearest measured phase. Phases between about 312 deg (LUNA) / 315 deg (PLUTO) and
  360 deg cannot be produced (about 13 % of the circle), so those pixels land on the nearest reachable end. Evaluate the
  trained masks with this mapping before the bench, and fine-tune with it (in sim or in situ) if the loss is large.

## 5. Still to verify

| item | why it matters |
|---|---|
| relay magnification | sets the PLUTO pixel size at the LUNA, and scales z1; test with two PLUTO lenses a known distance apart (beam tilt cancels) |
| beam collimation | would bias both distance results |
| phase past 2 pi | the tables assume the phase does not wrap past 2 pi |
