# Training adjustments for the vsilkog bench (2026-11-11)

These changes are for the next bench checkpoints (M1, then M2 and M3). They replace the distances in
`RETRAINING_CHANGES.md` (2026-11-10).

## Why the distances changed

The illumination reaching the SLMs was **not collimated**. It diverged astigmatically, with a virtual source about 5.8 m
(x) and 3.5 m (y) behind the LUNA. The Fresnel-lens focus tests of 2026-11-10 therefore measured plane-wave-*equivalent*
distances (29.7 / 168.8 mm), not physical ones. A single curvature cannot make both of M1's distances match, so M1 had no
correlation with the bench.

**Fix (bench side, already in place).** A converging astigmatic lens over the whole PLUTO frame collimates the beam:
- focal lengths f_y = 3.531 m, f_x = 5.667 m;
- stored in `collimator.json` and added to every PLUTO frame by the in-situ package;
- with it, Fresnel-lens pairs on the LUNA image at a scale of 1.0000 in both axes.

**Every model from now on should assume a collimated plane wave at the distances below.**

**Validation of the forward model at the new distances.** M1's masks were propagated at 35.87 / 175.35 mm and compared
with bench captures of the same frames, collimator on:

| | correlation with the bench |
|---|---|
| model at the old distances (29.7 / 168.8 mm) | 0.01 (none) |
| model at the new distances (35.87 / 175.35 mm) | **0.31** in the detector array, **0.34** for the 4096 detectors, whole frame 0.72 |

- The match is the same for all 5 a values tested.
- There is no residual scale error (at most 0.5 %), and the residual shift is a uniform (−4, 0) camera px, which the
  in-situ ROI calibration absorbs.
- The remaining gap to 1.0 is consistent with effects deliberately left out of training (section 4) and with the small
  relay magnification.
- M1 is not usable on the bench: at the true distances its own sim reward is 8.9e-2, which carries no information.

## 1. Parameter changes for training

| parameter (config name) | M1 value | **new value** | uncertainty | source |
|---|---|---|---|---|
| PLUTO image → LUNA, z1 (`key_to_enc_spacing`) | 29.7 mm | **35.87 mm** | ±0.5 mm | PLUTO lens focus sweep, collimated beam, z1 + z2 = m² f |
| LUNA → camera, z2 (`last_layer_ccd_spacing`) | 168.8 mm | **175.35 mm** | ±0.5 mm | LUNA lens focus sweep, collimated beam |
| relay magnification, PLUTO → LUNA plane | 1 (implicit) | **1.0096** (x 1.012, y 1.007) | ±0.003 | PLUTO lens pairs 4.0 / 2.88 mm apart |
| PLUTO pixel at the LUNA plane (`key_slm_dx` as seen by the LUNA) | 8 µm | **8.077 µm** (= 8 µm × 1.0096) | | from m |
| phase vs grey level (8-bit quantizer) | linear, 2πk/256 | **measured table per SLM** (`phase_tables.json`) | ±3 % scale | 32-level sweep |
| illumination | plane wave | plane wave (keep; now true with the collimator) | | |

**Relay magnification, two options.**
- **(a) Preferred, if the model supports it:** set the key pixel pitch to 8.077 µm. Anisotropic pitches would be
  8.096 µm (x) and 8.056 µm (y); only use them if the code allows separate values.
- **(b) Otherwise:** keep 8 µm. The bench package will resample the key by 1/m on the display. If you choose this,
  report it with the checkpoint.

**Phase table.**
- `phase_tables.json` lists the measured phase in radians for device levels 0..255, per SLM. In brief, the PLUTO
  reaches π at level 144 and the LUNA at level 109; at level 255 they reach about 301° and 305°.
- In the quantizer, map level k to `table[k]` instead of 2πk/256, so the optimizer only uses phases the SLMs can make.
- **Sign (corrected 2026-11-11 evening):** both SLMs are driven with the conjugate phase, so in the MODEL's sign
  convention level k produces **−table[k] (mod 2π)**, not +table[k]. Train with the set {−table[k] mod 2π}: the
  unreachable band is then 0°…59° (PLUTO) / 0°…55° (LUNA) in model phase, not 301°…360°. This is exactly what
  `make_sim_check.py`'s `snap()` models (realised phase = −table[q8(−φ)]); the package's display (`q8` with
  `*_lut_inverted`) sends the level whose table phase is nearest to −φ.
- M1 was trained with linear levels, which costs about 3.5× in the sim at the bench.

**Unchanged:**
- wavelength 635 nm;
- PLUTO 8 µm and LUNA 4.5 µm pitch, both 1080 × 1920;
- key 414 × 414 PLUTO px (3.31 mm) and decoder 736 × 736 LUNA px, one decoder per key;
- encoding Np = 4: 2 × 2 patch of 110 LUNA px, tiled 3 × 3 (660 px);
- camera pixel 5.86 µm;
- detectors 64 × 64, each 4 × 4 camera px on an 8 px pitch, symmetric about the axis;
- `prop_method` 'spectral', training readout 'affine', targets / Np / Nf / alpha / a grid (`targets.npy` stays valid).

**Note on the sim window.** z2 grew from 168.8 to 175.35 mm, so check that the spectral-propagator window and padding
still cover the camera area. The rerun of the sim check at these distances used P 9.50 mm and worked.

## 2. Optional: 2×2 pixel binning

No randomization during training: varying z1 or z2 would prevent convergence. Key position and phase scale are
handled after training (section 4).

Train with 2 × 2 macro-pixels on both SLMs:
- PLUTO: 207 × 207 key macro-pixels of 16 µm, still 3.31 mm.
- LUNA: 368 × 368 decoder macro-pixels of 9 µm; the 110-px encoding bins become 55 macro-pixels.
- Every size divides evenly.

What it buys:
- **Lateral and magnification tolerance about 2×:** the edge error per pixel halves.
- **Depth tolerance about 4×:** it scales with p²/λ.
- **Much less fringing loss:** 1–2 px features lose most of their modulation on these SLMs.

The cost is 4× fewer free parameters and half the maximum diffraction angle.

If you try it, train M1 both ways (1×1 and 2×2) and pick the variant whose accuracy holds best under the sensitivity
test below.

## 3. What to report with each checkpoint

1. The checkpoint, its config JSON, and the run log (as in `ckpt_M1/`).
2. Sim test: min-max RMSE and the affine (reward) RMSE on 1000 a, as before.
3. **Sensitivity table** (affine RMSE), evaluation only, not used in training. It tells us how much the bench will
   need to absorb.
   - z1 ± 0.5 mm
   - z2 ± 0.5 mm
   - key ± 3 px in x and in y
   - m ± 0.005
   - linear vs measured phase table
4. Expected camera images for the bench check (`make_sim_check.py`), at the trained distances, with a collimated plane
   wave.

## 4. Deferred until after training (do not add during training)

| effect | measured | later use |
|---|---|---|
| unmodulated light u | 0.155 (LUNA), 0.194 (PLUTO) | I = (1 − u)·\|F(φ)\|² + u·\|F(0)\|², for evaluation or fine-tuning |
| pixel crosstalk (fringing) | Gaussian phase blur σ ≈ 0.4 SLM px | for evaluation or fine-tuning; less important with 2×2 binning |
| illumination profile | flat over the 3.3 mm key / decoder area | measured flat if needed |
| key lateral position | registered to ±3 PLUTO px | key shift on the display; in-situ fine-tuning |
| phase scale | ±3 % | phase table on the display; in-situ fine-tuning |

## 5. Bench side, for reference only (not in the model)

- **PLUTO frame:** collimator phase over the whole frame, then key shift (−12, +30) PLUTO px (model frame), then
  conjugate, then measured table, then rot180.
- **LUNA frame:** conjugate, then measured table; no flip.
- **Camera:** captures rotated 180°.
- **Detector axis:** about raw (497, 985) with the collimator on (was (501, 985)); `--calibrate-roi` refines it.
- **Bench records:**
  - `runs/mag_20261111-102647`: divergence discovered;
  - `runs/colli_20261111-104051`: collimator and distances;
  - `runs/sanity_20261111-104219`: validation captures.
