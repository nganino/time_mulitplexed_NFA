# Migration: reconstruction → differential-detection classification

Tracks the pivot away from single-pixel intensity measurement + image-reconstruction
decoder, toward a photodiode array + softmax-cross-entropy classification head.

New target task: 10 classes, computed from 20 intensity measurements as
`class = argmax_i( score_i )`, where `score_i` is a normalized differential contrast
(see the 2026-09-03 loss.py entry below for the exact formula — refined from the
original raw-difference description).

## Status

| Piece | State |
|---|---|
| `config.py` — photodiode array geometry | ✅ done |
| `model.py` — array read-out in the physics sim | ✅ done |
| `loss.py` — softmax cross-entropy over the 10 differential pairs | ✅ done |
| `train.py` — training loop rewritten for classification | ✅ done |
| `train.py` — `save_images()` classification diagnostic figure | ✅ done |
| `test.py` — `_eval_loop`/`_print_summary`/`_save_per_class` | ✅ done |
| `test.py` — `evaluate()` / `_write_csv` | ✅ done |
| `test.py` — `_build_test_loader` FashionMNIST bug | ❌ known issue, unfixed (see below) |
| `test.py` — confusion matrix / misclassified-examples / measurement-diversity figures | ⏳ not started |
| `finetune.py` | ⏳ owned by user, not started |
| `model.py`/`wave_prop.py`/`config.py` — diffractive layers (`num_layers`, Option A propagators) | ✅ done, K=0 verified identical to before |
| `train.py` — `layer_phases` optimizer/scheduler wiring | ✅ done |
| `train.py` — `save_layer_masks()` | ✅ done |
| `test.py` — dataset default derived from checkpoint's config, not hardcoded | ✅ done |
| `train.py`/`model.py` — layer init strategy, `freeze_layers`, `measure()`/`diffraction_efficiency()` layer support, `layer_gnorm` | ⏳ not started |

## Background: why train.py couldn't run before this pass

`model.py` already called `self._decode(I_vec, B)` in `measure()`/`forward()`, but had
no `self.decoder` and no `_decode` method — both were apparently already stripped out
in anticipation of this pivot. `config.py` also had no `decoder_type`/`hidden_dim`. So
the reconstruction pipeline (`train.py`, `loss.py`'s `ReconstructionLoss`, `test.py`)
was already non-functional going into this change; this pass doesn't introduce that
breakage, it's pre-existing.

## 2026-09-03 — Photodiode array geometry (config.py + model.py)

**Decisions confirmed with user before implementing:**
- Time multiplexing (`config.M`, currently 20 learnable SLM masks) is **kept as-is**.
  Each mask is still measured separately; only the *spatial* read-out at the detector
  plane changed (one integrated region → a grid of 20). So the model's measurement
  output grew from `[B, M]` to `[B, M, pd_num_rows, pd_num_cols]` — nothing is summed
  or reduced across `M` in `model.py`.
- Detector → class pairing: **rows 0..(pd_num_rows/2 - 1) are the positive detectors**,
  rows `pd_num_rows/2..end` are the negative detectors, paired by column. With the
  default 4×5 array that's rows (0,1) = positive, rows (2,3) = negative, giving 10
  column-wise pairs. The exact per-pair combination rule (offset pairing, normalized
  contrast) was pinned down in the loss.py pass below.
- Scope for this pass: **config.py and model.py only**. `loss.py` was picked up in the
  very next pass (below); `train.py`/`test.py`/`finetune.py` are still the user's own.

**`config.py` changes** (the user had already started this edit; this pass added the
spacing/offset variables to match):
- `tc.pd_num_rows = 4`, `tc.pd_num_cols = 5`, `tc.num_photodiodes = pd_num_rows * pd_num_cols`
  (added by user)
- `tc.photodiode_size` (existing, `.1mm`) is reused as each detector's footprint —
  0.1mm × 0.1mm, matching the spec.
- New: `tc.pd_row_spacing`, `tc.pd_col_spacing` (physical units, meters; default `.1mm`
  each, i.e. detectors touching edge-to-edge) with derived pixel versions
  `pd_row_spacing_px` / `pd_col_spacing_px` computed in `recompute_derived()` (same
  pattern as `photodiode_pixels`).
- `tc.detector_offset_y` / `tc.detector_offset_x` (existing, pixel units, default 0) are
  now documented as the **first detector's** (row 0, col 0) center offset from the
  optical axis — the rest of the array is laid out from there via the spacings above.

**`model.py` changes:**
- `self.pd_px` (existing, from `photodiode_pixels`) is now the per-detector footprint
  size instead of the single detector's size — same variable, new meaning.
- Added `self.pd_num_rows`, `self.pd_num_cols`, `self.pd_row_spacing`,
  `self.pd_col_spacing`, and precomputed per-detector center pixel coordinates
  `self.pd_centers_y` / `self.pd_centers_x` in `__init__`.
- Sanity-check assertion generalized from "does the single detector fit in the sim
  grid" to "does the full array (all 20 detector footprints) fit in the sim grid".
- `_integrate_photodiode()` (single region, `[B, M] → scalar`) replaced by
  `_integrate_photodiode_array()`, which loops over the `pd_num_rows × pd_num_cols`
  grid and returns `[B, M, pd_num_rows, pd_num_cols]` (mean intensity per detector, per
  mask — same "mean over footprint" convention as before, just once per detector).
- `_add_meas_noise()` generalized to flatten over all non-batch dims (`M × rows × cols`)
  before computing the per-sample noise std, instead of assuming a 2D `[B, M]` input.
  Behavior on the old 2D shape is unchanged.
- `diffraction_efficiency()` generalized to sum captured power over all 20 detector
  regions instead of one.
- `measure()` and `forward()`: removed the dead `self._decode(...)` call (see
  Background above). They now return the raw measurement array directly:
  - `measure(phi_obj)` → `(I_vec, I_ccd)` — **signature changed**, was
    `(phi_pred, I_vec, I_ccd)`. `I_vec` is `[B, M, pd_num_rows, pd_num_cols]`.
  - `forward(phi_obj, return_field=False)` → `I_vec` (or `(I_vec, I_ccd)` if
    `return_field=True`). Previously returned `phi_pred`.
- `self.decoder_type` / `self.output_phase_max` in `__init__` are vestigial leftovers
  from the old decoder path — **left alone intentionally**, per the user's request not
  to touch the reconstruction-path cleanup.

**Known breakage this pass does NOT fix** (pre-existing; `loss.py` is now handled —
see the 2026-09-03 loss.py entry below):
- `train.py`: still imports `ReconstructionLoss` (no longer exists in `loss.py` —
  see below), reads `model.decoder`,
  `config.hidden_dim`, `config.lr_decoder`, `config.decoder_type` — none of which exist
  post-cleanup. Needs a full rewrite for classification (cross-entropy loss, accuracy
  metric, no decoder optimizer/checkpoint/image-saving).
- `test.py:256`: calls `_, I_vec, _ = model.measure(phi_obj)` — will now raise
  `ValueError: too many values to unpack` since `measure()` returns a 2-tuple. Needs
  updating to `I_vec, _ = model.measure(phi_obj)` when `test.py` is migrated.
- `finetune.py`: has its own standalone `_decode(model, config, device, vec)` function
  (separate from the model class) that likely also needs rework for the new pipeline.

## 2026-09-03 — Classification loss (loss.py)

User's file was already mid-edit: `ReconstructionLoss` had been renamed to
`ClassificationLoss` and `PCC`/`PurityLoss`/`tv_loss` deleted, but the body still
called them (broken — `NameError` on construction). Replaced the whole file with the
working differential-contrast classification loss below; the old reconstruction
classes are gone now (not just deferred) since the user's own edit had already
committed to removing them.

**Resolves both "open questions" from the previous entry** — the user clarified the
exact scheme:

1. **Reduction across `M`**: sum `I_vec` over the mask dimension before scoring —
   `I_sum = I_vec.sum(dim=1)`, `[B, M, rows, cols] → [B, rows, cols]`. Physically: the
   CCD integrates the signal while all `M` phase biases are displayed in sequence, so
   one mask alone misclassifying gets averaged out by the ensemble. Noted in the
   docstring: because the per-class score is a normalized ratio (below), summing vs.
   averaging over `M` produce numerically identical scores — the `1/M` factor is on
   both the numerator and denominator and cancels exactly. Sum was used since it
   matches "integrate" literally, but it's equivalent either way.
2. **+/− pairing**: row `r` (0-indexed, positive half) pairs with row `r + rows/2`
   (negative half) at the same column — e.g. for the default 4×5 array, row0↔row2 and
   row1↔row3, each across all 5 columns → 10 pairs. This is the same "positive half /
   negative half" convention from the config.py/model.py pass, now made precise (it's
   an *offset* pairing, not adjacent-row or sum-then-subtract).
3. **Per-pair class score** — normalized contrast, not a raw difference:
   `score = (I+ − I−) / (I+ + I− + eps)`, bounded to `[-1, 1]`. Flattened to a
   `[B, num_classes]` logit vector via class index `r * pd_num_cols + c`, then
   `nn.CrossEntropyLoss` (softmax cross-entropy) against the integer label.

**Temperature `T`** (originally proposed as a `logit_scale` multiplier, then changed to
a temperature *divisor* per user request — same idea, inverse convention). Flagged as a
concern, not silently decided: because the raw contrast is confined to `[-1, 1]`, the
softmax distribution's confidence is capped regardless of how well-separated the
underlying classes are (a full-scale spread of 2 across 10 classes tops out around
~45% max softmax probability). `scores()` now divides the contrast by `self.T` before
returning it:
```python
contrast = (I_pos - I_neg) / (I_pos + I_neg + self.eps)   # in [-1, 1]
contrast = contrast / self.T
```
`T` is a **fixed constant, not trained** — `config.T`, added to `config.py` right next
to `num_classes` since it's a classification-head hyperparameter, default `0.1` (i.e.
divides contrast by 0.1 = multiplies by 10, widening the effective logit range before
softmax). Smaller `T` → sharper/more confident predictions for the same physical
contrast; `T=1` recovers the raw, un-widened contrast. Read via
`getattr(config, 'T', 0.1)` in `loss.py` so it degrades gracefully if `config.T` is
ever missing.

**Still open / worth double-checking when you wire up `train.py`:**
- `train.py`'s loop currently discards the label (`for phi_obj, _ in pbar`). It'll need
  to pass that label through as `ClassificationLoss.forward(I_vec, target)`'s `target`.
- `ClassificationLoss.forward()` returns `(loss, logits, pred)` — `pred` is
  `argmax(logits)`, handy for an accuracy metric (`(pred == target).float().mean()`)
  without recomputing anything in `train.py`.
- Early in training, when the SLM hasn't learned to route light anywhere in particular,
  both detectors in a pair can read near-zero — `eps=1e-8` prevents a divide-by-zero
  but the resulting contrast can be a noisy, low-magnitude signal until the SLM masks
  start concentrating light. Expected, not a bug — just don't be surprised by noisy
  early-epoch logits.
- Asserts `pd_num_rows % 2 == 0` and `(pd_num_rows/2) * pd_num_cols == config.num_classes`
  at construction — will raise immediately (not silently misbehave) if the array
  geometry and `num_classes` ever drift out of sync.

## 2026-09-03 — Training loop rewrite (train.py)

Full rewrite for classification. Decisions confirmed with user first (five questions,
answered 1-5):

1. **`freeze_slm` kept**, but fixed. It previously froze `slm_phases` and fell back to
   training `model.decoder` — decoder's gone, and today `slm_phases` is the model's
   *only* learnable parameter, so `freeze_slm=True` now means nothing trains this run.
   Kept intentionally (not deleted) because diffractive layers (planned: `num_layers`
   1..8, replacing plain free-space propagation to the CCD) will give this flag a real
   second thing to leave trainable. Documented with a NOTE comment at the point it's
   set in `_init_optimizers()`.
2. **Dropped `slm_warmup_epochs`** entirely (`train_decoder`/`phase_tag` split in
   `main()`, and the `train_decoder` param on `train_step()`) — there's no decoder left
   to warm up *toward*. SLM now trains from epoch 0 whenever it isn't frozen.
3. **Best-checkpoint criterion switched to highest validation accuracy** —
   `best_val_loss`/`val_loss <` became `best_val_acc`/`val_acc >`.
4. **`save_images()` left defined but disabled** — it plots predicted-vs-ground-truth
   *phase images*, which the classification pipeline no longer produces (`valid_step`
   returns `(loss, accuracy)`, not an image pair). Kept the method body verbatim (not
   deleted) with a NOTE comment explaining why, and removed its call site from
   `main()`'s validation block so it's simply unused for now. User will specify
   classification-appropriate replacement figures later.
5. Cosmetic `"ConvDecoder"` naming (run-name string, wandb project) — left untouched,
   user is fixing those themselves.

**Mechanical changes needed regardless of the above (fixing the hard blockers from
"Known breakage"):**
- `from loss import ReconstructionLoss` → `from loss import ClassificationLoss`.
- Removed `_bin_target()` (pooled ground-truth images to `N×N` — no image target
  anymore) and its only import, `torch.nn.functional as F`.
- `_init_optimizers()`: dropped `opt_dec`/`sched_dec`/`config.lr_decoder` entirely
  (referenced `model.decoder`, which doesn't exist). Now returns just
  `(opt_slm, sched_slm)` — `__init__`, `_checkpoint_dict()`, and `load()` updated to
  match (no more `opt_dec`/`sched_dec` checkpoint keys).
- `train_step(phi_obj, target)` / `valid_step(phi_obj, target)`: both now take the
  integer class label (`main()`'s loop previously discarded it — `for phi_obj, _ in
  pbar`, now `for phi_obj, label in pbar`). Both call `self.model(phi_obj)` → `I_vec`
  (single return, no more `(phi_pred, I_ccd)` / `return_field`/`purity_fn` machinery)
  → `self.criterion(I_vec, target)` → `(loss, logits, pred)`; accuracy computed as
  `(pred == target).float().mean()`. `train_step` returns `(loss, acc, slm_gnorm)`;
  `valid_step` returns `(loss, acc)`.
- `main()`: replaced all `mse`/`pcc`/`purity`/`dec_gnorm` accumulation and
  `writer.add_scalars('mse'/'pcc', ...)` calls with `accuracy` tracking/logging.
  `wandb.init(...)`'s `config=` dict no longer references `config.hidden_dim` (crashed
  immediately before) — now reports `pd_num_rows`/`pd_num_cols`/`num_classes`/`T`.

**`test.py` / `finetune.py`**: still not touched — `test.py:256`'s
`_, I_vec, _ = model.measure(phi_obj)` 3-tuple unpack and `finetune.py`'s standalone
`_decode()` helper are exactly as described in the earlier entry, unaffected by this
pass.

## 2026-09-03 — Classification-appropriate `save_images()` (train.py)

Replaced the disabled reconstruction-era `save_images()` (was: dead code, per the
previous entry) with the classification diagnostic the user requested. One figure per
checkpoint, up to 4 rows (samples) x 4 columns:
- **col 0**: input phase image, titled with the true label.
- **col 1**: grouped bar chart, one group per class -- raw positive-detector vs
  negative-detector intensity (`I_vec.sum(dim=1)`, i.e. summed over `M` the same way
  `loss.py` does before scoring). Same class ordering/flattening as `loss.py`
  (`r * pd_num_cols + c`), so bars line up 1:1 with col 2.
- **col 2**: bar chart of the normalized differential contrast per class
  (`(I+ - I-)/(I+ + I- + eps)`, recovered from `ClassificationLoss` by undoing the `/T`
  temperature scaling -- `contrast = logits * self.criterion.T` -- so this panel shows
  the physical, T-independent quantity rather than the softmax input). True class bar
  colored orange, predicted class marked with an X (green if correct, black if wrong).
- **col 3**: `I_ccd.sum(dim=1)` (the CCD-plane intensity integrated over the `M` phase
  biases -- the same quantity the classification actually reduces over), cropped to a
  window around the detector array (the full 2000x2000 grid would make 25-50px
  detectors nearly invisible), with each of the 20 photodiode footprints drawn as a
  `Rectangle` outline: green for positive detectors (`row < pd_num_rows/2`), red for
  negative.

Implementation notes:
- `save_images(phi_obj, label, tag='val', n_show=4)` now does its own forward pass
  with `return_field=True` (rather than reusing `valid_step`'s), since it's called
  once per `checkpoint_save` interval, not per batch -- no reason to complicate the
  hot validation loop's return signature for a diagnostic that runs rarely.
  Temporarily flips the model to `eval()`/back if it was mid-training, matching the
  guard pattern already used in `train_step`/`valid_step`.
- Re-enabled the call site in `main()`'s validation block:
  `trainer.save_images(phi_obj, label, tag='val')`, reusing the *last* validation
  batch's `phi_obj`/`label` (the loop variables persist after the `for` loop ends --
  same trick the old code used with `last_pred`/`last_gt`).
- Output filename changed from `phase_{tag}_epoch{N}.png` to
  `classification_{tag}_epoch{N}.png` to avoid confusion with the old reconstruction
  figures if a log directory has both from different runs.
- Added `from matplotlib.patches import Rectangle` import.

## 2026-09-03 — test.py: `_eval_loop` / `_print_summary` / `_save_per_class` (partial pass)

First of what will be several test.py passes (user is rewriting this file
incrementally themselves, editing between passes -- e.g. already fixed
`SinglePixelQPI`→`TimeMultiplexedClassifier` and `ReconstructionLoss`→`ClassificationLoss`
at the import, and deleted several reconstruction-only helpers -- so this file
should be read fresh each session, not assumed to match the last-known state).

**Naming**: renamed `logits`→`class_scores` throughout (`loss.py`'s
`ClassificationLoss.forward()` return value, its docstring, and the two remaining
consumers -- `train.py`'s `save_images()` and this pass's new test.py code).
`model.py`'s docstring comment updated to match ("class scores" instead of "class
logits"). Rationale: `ClassificationLoss.scores()`/`forward()` never produced logits in
the usual learned-linear-layer sense -- they're physically-measured, temperature-scaled
differential contrast values -- so `class_scores` is more accurate to what the numbers
actually are.

**`_eval_loop`**: rewritten for classification. Dropped the `pcc_fn` parameter (PCC no
longer exists) and the whole `error_map`/`vis_preds`/`vis_gts` collection (those fed
the now-deleted `_save_side_by_side`/`_save_error_map`, image-reconstruction figures
with no classification equivalent). New body: `I_vec = model(phi_obj)` →
`loss, class_scores, pred = criterion(I_vec, labels)`, accumulating per-sample
`labels`/`preds`/`class_scores` and per-batch `loss` into `agg`. Also deleted the now
provably-dead `_bin_target()` helper (and its only import, `torch.nn.functional as F`)
since nothing calls it anymore once `_eval_loop` no longer needs to pool a ground-truth
image.

**`_print_summary`**: now reports overall accuracy (`(preds==labels).mean()`) and mean
cross-entropy loss, replacing the MSE/PCC/PSNR/SSIM/MAE printout.

**`_save_per_class`**: replaced the PSNR/SSIM/MAE-per-class bar chart (and its
grating-period label formatting, `period_labels` param) with per-class
precision/recall/F1, hand-rolled from `agg['labels']`/`agg['preds']` via TP/FP/FN
counts per class -- no `sklearn` dependency added (not in `requirements.txt`; project
already dropped `scikit-image` when the user removed `_ssim_batch`). `recall` here is
mathematically identical to "per-class accuracy" (fraction of that class's true samples
correctly predicted) -- documented as such in the docstring so it isn't read as a
separate, unrelated metric.

**Deliberately left alone this pass** (matches the user's "let's start with just these
three" scope): `_save_side_by_side`, `_save_error_map`, `_collect_meas_vectors`,
`_save_meas_diversity`, `_estimate_dominant_period_um`, `_save_period_vs_psnr`,
`_write_csv`, `_build_test_loader`, and `evaluate()` itself are all still exactly as
described in the previous entry's function-by-function plan -- several have dangling
calls to now-deleted functions (`_save_side_by_side`, `_estimate_dominant_period_um`,
`_psnr`/`_ssim_batch` were deleted by the user but are still called from `evaluate()`),
and the `from loss import ClassificationLoss, PCC` import will `ImportError` the moment
`evaluate()` (which still does `pcc_fn = PCC()`) is touched. Left as-is rather than
partially patched, since a one-line fix here wouldn't actually make anything runnable
until `evaluate()` itself is rewritten.

## 2026-09-03 — test.py: `evaluate()` + `_write_csv` (next incremental pass)

- Dropped `PCC` from `from loss import ...` (it no longer exists in `loss.py`).
- **`evaluate()`** rewritten to only call what actually exists now:
  `agg = _eval_loop(model, test_loader, criterion, device, desc='Testing')` →
  `_print_summary(agg)` → `_write_csv(...)` (if `csv_path` given) →
  `_save_per_class(agg, ..., config=config)` → `_save_slm_masks(...)`. Removed the
  `error_map`/`vis_preds`/`vis_gts`/`psnr_vis`/`period_vis` machinery and the calls to
  `_save_side_by_side`/`_estimate_dominant_period_um`/`_save_period_vs_psnr` (all
  reconstruction/grating-only, no longer produced by `_eval_loop`), and the
  `_collect_meas_vectors`/`_save_meas_diversity` call pair (kept *defined*, just not
  called yet -- still needs reworking for `I_vec`'s new
  `[B, M, pd_num_rows, pd_num_cols]` shape before it can run; a `NOTE` comment in
  `evaluate()` and the module docstring both point at this).
- **`_save_error_map`** deleted outright (not just deferred) -- with `_eval_loop` no
  longer producing an `error_map`, it had zero callers left; it was already on the
  "delete outright" list from the earlier function-by-function plan.
- **`_write_csv`** rewritten: `psnr_mean/std`, `ssim_mean/std`, `mae_mean/std`,
  `pcc_mean`, `cv_intra_mean`, `pw_dist_mean` → `n_samples`, `accuracy`, `loss_mean`,
  `loss_std`, `macro_f1`. Dropped the `diversity` param entirely (nothing currently
  produces a `diversity` dict to pass it -- add back if/when
  `_save_meas_diversity` gets reworked and wired back in).
- **New shared helper `_per_class_prf(labels, preds, num_classes)`**: factored the
  TP/FP/FN → precision/recall/F1 computation out of `_save_per_class` (previous pass)
  so `_write_csv`'s `macro_f1` (mean F1 across classes) reuses the exact same
  computation instead of a second copy.
- Rewrote the module-level docstring (output-file list) to match what `evaluate()`
  actually produces now, and to flag what's still pending inline.

## 2026-09-03 — TensorBoard/wandb logging fix (train.py)

User reported TensorBoard showing nothing after epoch 0 finished. Root cause found by
inspecting the actual run directory on disk: `writer.add_scalars('loss', {'train':
...}, epoch)` (plural, tensorboardX) doesn't write to the main `tfboard/` event file --
it creates a **separate subdirectory per tag** (`tfboard/loss/train/`,
`tfboard/loss/val/`, `tfboard/accuracy/train/`, `tfboard/accuracy/val/`, each with its
own event file), leaving the top-level `tfboard/events.out.tfevents...` permanently
0 bytes. Data was actually being written, just fragmented across 4 subdirectories
instead of one file -- confusing for TensorBoard's run selector even though it should
still recurse into them.

**Fix**: switched every `writer.add_scalars(tag, {...}, epoch)` call to
`writer.add_scalar('tag/train'|'tag/val', value, epoch)` -- slash-delimited tag,
singular `add_scalar`. Keeps everything in one event file at the top of `tfboard/`
while still grouping train/val onto the same chart. Takes effect on the *next*
training run (a running process keeps writing to the old fragmented layout since
Python doesn't hot-reload).

**Also added `wandb.log(...)`** (previously flagged as initialized-but-never-called):
now logs `loss/train`, `accuracy/train`, `slm_gnorm/train` every epoch, and
`loss/val`, `accuracy/val` every `checkpoint_save` epochs -- mirroring exactly what
goes to TensorBoard, same values, same epoch numbers (`step=epoch`). `slm_gnorm`
(the SLM gradient-norm diagnostic that was computed every epoch but never logged
anywhere) is included now that the logging call sites were already being touched.

## 2026-09-03 — Diffractive layers (model.py, wave_prop.py, config.py)

User's next planned feature (their idea, their initiative): replace the single
free-space hop with a stack of learnable diffractive phase layers between the SLM and
the CCD (`num_layers` config, 0 = original single-hop behavior unchanged). User wrote
a first draft themselves; I reviewed it twice (once mid-draft, once after their own
partial fix) and it had several blocking bugs -- this entry covers the version I fixed
directly, at their request, after walking through the options for the propagation-
distance design with them first.

**Root design issue**: `FreeSpaceProp` precomputes its transfer function `H` for one
fixed `z` at construction time; it can't take a different `z` per `forward()` call.
The draft tried to reuse a single `self.propagator` across the SLM->layer1,
interlayer, and layer->CCD hops (three different distances) by passing `z` as a second
argument to `.forward()`, and separately by trying to pass `z` to `FreeSpaceProp()`
without actually storing multiple instances -- neither works, since `.forward(self,
u)` never accepts a `z`, and one instance can only ever represent one distance.
Discussed three options (multiple precomputed instances / compute H on-demand from a
per-call z, optionally cached / one instance per gap for non-uniform spacing) and
implemented **Option A** per the user's choice: since their config already implies
exactly three distinct distances (`slm_first_layer_spacing`, one shared
`interlayer_spacing` reused for every inter-layer gap, `last_layer_ccd_spacing`), build
exactly three additional `FreeSpaceProp` instances up front (`prop_to_layer1`,
`prop_interlayer`, `prop_to_ccd`), only when `num_layers > 0`, and call the right one
at each stage of the loop instead of passing `z` at call time.

**`wave_prop.py`**: `FreeSpaceProp.__init__(self, config, z=None)` -- `z` defaults to
`config.z_slm_ccd` when omitted, so the one pre-existing call site
(`self.propagator = FreeSpaceProp(config)`) is unaffected; pass an explicit `z=` to get
a propagator for a different fixed distance.

**`config.py`**: added `tc.layer_bin = int(layer_dx / sim_dx)` and
`tc.layer_size_sim = layer_size * layer_bin` (both in `init_params()` and
`recompute_derived()`, mirroring `slm_bin`/`slm_x_num_sim`) -- `layer_dx` was already
defined but never actually used anywhere before this. Also fixed a stray trailing `'`
on the "Diffractive Layers" section-header comment (harmless, cosmetic).

**`model.py`** -- three separate fixes, all previously flagged as broken in review:
1. `self.layer_aperture` now properly `register_buffer`'d. The user's own attempted
   fix (`self.layer_aperture = torch.ones(...)` immediately followed by
   `self.register_buffer('layer_aperture', self.layer_aperture)`) would have raised
   `KeyError: attribute 'layer_aperture' already exists` -- `register_buffer` rejects
   re-registering a name that's already a plain attribute. Fixed by calling
   `register_buffer` directly with no preceding assignment.
2. `self.layer_phases` changed from `nn.ParameterList(layers_init)` (iterating a 4D
   `[num_layers,1,H,W]` tensor along dim 0 silently drops a dimension, leaving each
   `layer_phases[k]` 3D -- `F.interpolate(..., mode='bilinear')` requires 4D input, so
   this would have errored the first time `_get_layer_phase` ran) to a single
   `nn.Parameter(layers_init)` (mirroring `slm_phases`'s pattern exactly), indexed with
   a `[k:k+1]` slice (preserves the dim) instead of plain `[k]` (drops it).
3. `_get_layer_phase` rewritten to mirror `_get_slm_field`'s three-step pattern, which
   it previously skipped entirely: `sigmoid(...) * 2pi` (bounds the raw parameter to a
   physical `[0, 2pi)` phase -- previously used raw/unbounded, inconsistent with how
   `slm_phases` is handled), `mode='nearest'` interpolation up to `layer_size_sim`
   (previously `mode='bilinear'` straight to `N_sim` in one step -- physically wrong
   for a discrete flat-phase pixel, same reasoning `_get_slm_field` already documents),
   then `_embed_in_sim` to zero-pad/center into the full `N_sim` grid (previously
   skipped -- the layer pattern filled the entire simulation window edge-to-edge with
   no guard band at all, unlike every other plane in the system).
4. `forward()`'s layered branch rewired to call `prop_to_layer1` (once, before the
   loop), `prop_interlayer` (between layers), and `prop_to_ccd` (after the last
   layer's modulation) instead of the single shared `self.propagator` -- this is
   Option A itself.
5. Added an assert (`layer_size_sim <= N_sim`, only checked when `num_layers > 0`)
   matching the existing SLM/object/detector geometry sanity checks.

**Verified K=0 still degrades identically**: the `else` branch is untouched --
`U_ccd_flat = self.propagator(U_flat)`, same single call, same `self.propagator`
(built with `z=config.z_slm_ccd` by default) as the pre-diffractive-layer code. Traced
it byte-for-byte against the original single-hop `forward()`.

**Not yet updated** (flagged, not fixed -- lower priority, not on the current
training/eval path): `measure()` and `diffraction_efficiency()` both still do a single
unconditional `self.propagator(...)` call with no layers at all, regardless of
`num_layers` -- they'll keep describing the old single-hop physics even once layers are
trained. Also still true from the last review: `layer_phases` isn't in any optimizer
yet (`train.py`'s `_init_optimizers()` only builds `opt_slm`) -- the layers would train
`forward()` correctly now, but nothing would actually update them without a `train.py`
change too.

## 2026-09-03 — Layer optimizer/scheduler wiring (train.py)

User added `opt_layers`/`sched_layers` themselves (mirroring `opt_slm`/`sched_slm`).
Reviewed and found six bugs; user had already fixed the first (`config.lr_layers` ->
`config.lr_layer` name mismatch) before I got to it. Fixed the remaining five:

1. **`train_step()` never touched `opt_layers` at all.** It built the optimizer but
   never called `zero_grad()`/`.step()` on it -- so even with #1 fixed, gradients on
   `layer_phases` would accumulate forever (never zeroed) and never actually get
   applied (no `.step()`), leaving the layers frozen at zero-init regardless of how
   long training ran. Added `if self.opt_layers is not None: self.opt_layers.zero_grad(...)`
   / `.step()` right alongside the existing `opt_slm` calls.
2. **`sched_layers.step()` was never called** in `main()`'s epoch loop -- only
   `sched_slm.step()` was. Added the matching call, same guard pattern.
3. **`_checkpoint_dict()` didn't save `opt_layers`/`sched_layers` state** -- resuming
   a run would silently lose the layer optimizer's Adam momentum/schedule (restart
   from scratch) while the SLM's resumed correctly. Added both keys, same
   `if X else None` pattern as the SLM ones.
4. **`load()` didn't restore them either** -- added matching restore blocks (same
   try/except-and-warn pattern as `opt_slm`/`sched_slm`).
5. **`config.K` doesn't exist** (only `config.num_layers`) in the wandb config dict --
   caught by the surrounding `try/except`, so it silently disabled wandb every run
   rather than crashing. Changed to `config.num_layers`.

All five verified by re-reading the full file after editing -- `opt_layers`/
`sched_layers` now flow consistently through construction, `train_step`, `_checkpoint_dict`,
and `load`, matching the `opt_slm`/`sched_slm` pattern at every step.

**Still open, not part of this pass** (flagged during review, not requested yet):
layer-phase initialization is asymmetric with the SLM's (`layer_phases` starts at
exact zero -> every layer starts as a spatially-uniform, optically-inert `pi` phase
piston, vs. the SLM's random-normal init); no `freeze_layers` flag exists (the
original motivation for keeping `freeze_slm` around); `measure()`/
`diffraction_efficiency()` still don't route through the layer stack; no `layer_gnorm`
diagnostic (parity with `slm_gnorm`); the `interlayer_spacing = 15um` magnitude
question from a few passes back is still unresolved.

## 2026-09-03 — `save_layer_masks()` (train.py) + test.py dataset-default fix

**`train.py`**: added `save_layer_masks()`, mirroring `save_slm_masks()` exactly --
visualizes each learned diffractive-layer phase (`sigmoid(layer_phases)*2pi`) as a
`twilight`-colormap image, one per layer, saved to
`layer_masks_epoch{N}.png`. No-ops immediately if `num_layers == 0` (nothing to show).
Wired into `main()`'s checkpoint block right alongside `save_slm_masks()`.

**`test.py`**: user had already band-aided the "silently evaluates a FashionMNIST
checkpoint against real MNIST" bug by changing the hardcoded defaults from `'mnist'`
to `'fashion'` (both `evaluate()`'s `dataset` param and `__main__`'s `DATASET`). That
fixes today's common case but is still a hardcoded guess -- a checkpoint trained on
anything *other* than FashionMNIST (`mnist_grating`, `cifar10`, ...) would now silently
default to the wrong dataset instead, just a different wrong one. Replaced both
hardcoded defaults with `None`, and added real default-resolution logic in
`evaluate()`: once the checkpoint's saved config is loaded (so `config.dataset`
reflects what the checkpoint was *actually* trained on), an unset `dataset` arg is
resolved from `config.dataset` via a small alias map (`{'FashionMNIST': 'fashion'}`,
identity otherwise -- `mnist_grating`/`cifar10`/`tinyimagenet` already match
`_build_test_loader`'s branch names directly). Explicit `--dataset` still overrides
this, preserving the existing (and useful) ability to force evaluation against a
different dataset than training for generalization checks.

## 2026-09-04 — `num_workers` config knob (config.py, dataloader.py, test.py)

User hit a training crash mid-validation:
`OSError: [WinError 1114] ... loading torch\lib\shm.dll` inside a
`multiprocessing.spawn` child process. Root cause: `DataLoader(num_workers=4)`
(hardcoded, all three loaders in `dataloader.py`'s `get_dataloaders()`, plus
`test.py`'s `_build_test_loader`) uses Windows' `spawn` start method, which
re-imports `train.py` from scratch in each new worker process -- including its
top-level `from tensorboardX import SummaryWriter`, which imports `torch` a second
time in that fresh process. On this machine that second import intermittently fails
loading `shm.dll`. Not a tensorboardX bug and not a bug in this codebase -- a
Windows/torch DLL-loading issue triggered by spawning new worker processes.

Added `config.num_workers` (default `0` -- loads in the main process, sidesteps the
spawn-triggered re-import entirely) and wired it into all four `DataLoader(...)`
calls that were hardcoded to `num_workers=4` (three in `dataloader.py`, one in
`test.py`, the latter via `getattr(config, 'num_workers', 0)` for backward
compatibility with checkpoints saved before this config key existed). Dataset
loading isn't the bottleneck here anyway (FFT propagation dominates every step), so
there's little performance cost to defaulting to 0. Override with
`--set num_workers=4` on a machine that doesn't hit the DLL issue.

## 2026-09-04 — `_apply_overrides` now rejects unknown `--set` keys (train.py)

User ran `--set train_samples=1000 max_epochs=10` and the epoch count didn't change.
Root cause: `max_epochs` (typo -- real attribute is `max_epoch`, singular). Since
`_apply_overrides` used `getattr(config, key, None)` unconditionally, a typo'd key
looked identical to a legitimate `None`-default attribute (`train_samples`,
`mnist_cap`) -- it fell into that branch, got coerced to `int(10)`, and
`setattr(config, 'max_epochs', 10)` created a brand-new, never-read attribute instead
of erroring. It even printed `Override: max_epochs = 10`, which reads as confirmation
but isn't -- the real `config.max_epoch` (what `train.py`'s epoch loop and both LR
schedulers actually read) was untouched, silently defaulting to whatever `config.py`
sets it to.

Fixed by adding a `hasattr(config, key)` check before that branch: every real config
attribute is set in `init_params()`, even the `None`-default ones, so `hasattr` is
only ever `False` for a genuine typo. Now raises `ValueError` immediately naming the
bad key, instead of a misleadingly-successful-looking no-op.

## Known issue, not touched this pass (flagged in the earlier function-by-function
plan too): `_build_test_loader`'s default (`else`) branch always reads
`MNISTPhaseDataset(config, is_training=False)` off `config.data_path` (real MNIST
digits), regardless of `config.dataset == 'FashionMNIST'` -- unlike
`dataloader.get_dataloaders()`, which was fixed for this earlier. Since both
`evaluate()`'s default `dataset='mnist'` param and the `__main__` block's
`DATASET = 'mnist'` route through that branch, **running `python test.py` with no
`--dataset` flag right now will silently evaluate a FashionMNIST-trained model against
real MNIST digit images** -- it won't crash, it'll just produce meaningless numbers.
Worth fixing before trusting any output from a default run; pass `--dataset fashion`
in the meantime (goes through the other, correct branch) as a workaround.

## 2026-09-03 — Bug fix: photodiode array wasn't actually centered (model.py + config.py)

User sanity-checked the geometry: with the default `detector_offset_y/x = 0`, the
4×5 array should be centered on the simulation grid such that the optical axis falls
on the middle column (col 2, 0-indexed) and exactly between the two middle rows
(rows 1 and 2, 0-indexed) — i.e. between r2c3 and r3c3 in 1-indexed terms.

**It wasn't.** The original `cy0/cx0` formula (`N_sim//2 + offset`, then
`cy0 + r*spacing`) put detector **(row 0, col 0)** — not the array's center — at
`N_sim//2` when `offset=0`. The whole array extended down-and-right from the grid
center instead of straddling it. This was a consequence of how I'd originally defined
`detector_offset_y/x` (as "first detector's position") — a valid design, just never
actually centered by default, and nobody had explicitly asked for centering until now.

**Fix** — `model.py`'s `__init__`, centers now computed as:
```python
cy0 = N_sim//2 + det_offset_y - ((pd_num_rows - 1) * pd_row_spacing) // 2
cx0 = N_sim//2 + det_offset_x - ((pd_num_cols - 1) * pd_col_spacing) // 2
```
i.e. shift the starting corner back by half the array's total span, so the array is
centered *at* `(N_sim/2 + offset)` instead of anchored there at its first detector.
`detector_offset_y/x` now means **the array's center offset** from the optical axis
(re-documented in both `config.py`'s comment and `model.py`'s inline comment) —
previously "first detector (row 0, col 0) position". Verified against the actual
config values (`N_sim=2000`, `pd_row_spacing_px=pd_col_spacing_px=50` since
`pd_row_spacing`/`pd_col_spacing` are `.2mm` / `sim_dx=4um`): columns land at
`[900, 950, 1000, 1050, 1100]` — middle column exactly on `1000 = N_sim/2`; rows land
at `[925, 975, 1025, 1075]` — midpoint of the two middle rows is exactly `1000` too.
(Both land exactly on-pixel here because `50` is even; the `//2` floor-division in the
formula only introduces a sub-pixel rounding when `(pd_num_rows-1)*spacing` or
`(pd_num_cols-1)*spacing` is odd — negligible at `sim_dx=4um` resolution even then.)

## 2026-09-04 — "Countries and citizens" multi-country voting (loss.py, model.py, train.py, config.py)

**Feature.** Instead of one integration period showing all `M` learnable SLM phase
biases, the model now shows `C` independent integration periods ("countries"), each
with its own `M` learnable phase biases ("members" / "citizens"), for a total of
`T = M * C` learnable masks. `loss.py`'s `ClassificationLoss.scores()` reshapes
`I_vec : [B, T, pd_num_rows, pd_num_cols]` to `[B, C, M, rows, cols]`, integrates each
country's `M` members separately, computes each country's own differential class-score
vote `per_country_scores : [B, C, num_classes]`, then sums the countries' votes into
`aggregated_scores : [B, num_classes]` for the actual cross-entropy loss and the final
`argmax` prediction — "wisdom of the crowd" over independently-optimized voting blocs.
`C=1` exactly reproduces the original single-integration-period behavior.

This was implemented by a different coding assistant before this session; the user
asked for a careful understanding + consistency/bug-hunt pass across every touched
file before trusting it. Found and fixed:

1. **`train.py`'s `_init_model` crashed on every startup.** `getattr(model, 'T',
   getattr(self.config, 'T', self.config.M * self.config.C))` — Python evaluates all
   arguments eagerly, including the fallback-of-a-fallback, so `self.config.C` was
   dereferenced unconditionally even though `config.C` didn't exist yet (see #2).
   `AttributeError` before a single batch ran. Fixed by simplifying to `model.T`
   directly (the model always sets `self.T` in `__init__`).
2. **`config.C` didn't exist at all.** Wired through `model.py`, `loss.py`,
   `test.py`, and `train.py`'s wandb config dict, but never added to `config.py` —
   so the feature couldn't be turned on, and the `wandb.init(config={'C': config.C})`
   call would also have raised (silently disabling wandb again, same failure mode as
   the earlier `config.K` bug). **Fixed directly by the user** in `config.py`:
   added `tc.M = 10` (masks per country), `tc.C = 1` (countries, default off), and
   `tc.T = tc.M * tc.C` (recomputed in both `init_params()` and `recompute_derived()`,
   so `--set M=... C=...` stays consistent automatically). `tc.T` is a pure derived
   value now — never a manual `--set T=...` target.
3. **Root cause / worst bug: `config.T` name collision.** `config.T = 0.1` already
   existed and meant *softmax temperature* (the original spec). The other assistant
   reused the same name for *total mask count* in `model.py`
   (`self.T = int(getattr(config, 'T', self.M * self.C))`) and `loss.py`
   (`self.T_total = ...` the same way). Since `config.T` existed, the `M * C`
   fallback never fired — `self.T` silently became `int(0.1) = 0`: zero learnable
   SLM masks, model unusable, no crash to announce it. **Fixed by renaming the
   temperature to `config.softmax_T`** (user's choice — keeps `T` meaning "total
   mask count" everywhere, which is what `model.py`/`loss.py`/`test.py` already
   assumed). Also fixed the mirror-image version of the same bug in `loss.py`:
   `self.softmax_T = float(getattr(config, 'softmax_T', getattr(config, 'T', 0.1)))`
   would have quietly read the *mask count* as the temperature on any config missing
   `softmax_T` — removed that fallback entirely.
4. **Dead/duplicated code in `train.py`'s `save_images()`.** The diagnostic-figure
   forward pass + loss call was present twice in a row, back to back, with the first
   copy's results (`class_scores`/`contrast`/`pred`) computed then immediately
   discarded and recomputed by an identical second copy — a leftover merge artifact
   that silently doubled the compute cost of every checkpoint's diagnostic image.
   Removed the first (dead) copy.
5. **Stale docstrings** in `model.py` (module docstring, `measure()`, `forward()`)
   and `loss.py` (module docstring, `forward()`) still said `I_vec : [B, M, ...]` —
   updated to `[B, T, ...]` (`T = M * C`) throughout, and `save_images()`'s docstring
   updated to describe all 5 current columns (it only documented 4; column 4,
   per-country scores, was undocumented).

**Defaults after this pass**: `M=10`, `C=1` (country-splitting off by default,
`T=10` — half the previous `M=20` baseline; `C` was defaulted to `1`, not `4`, so
this is *not* the same total mask budget as before. Flagged to the user — pass
`--set C=...` explicitly to opt into multi-country voting).

**Not touched / open questions, deliberately left for the user to decide:**
- `train.py`'s `save_images()` col 4 shows all `C` countries as one grouped bar
  chart rather than `C` separate bar-plot panels — a reasonable reading of "C bar
  plots of the 20 differential measurements" but not the only one; revisit if the
  user wants literal per-country subplots instead.
- Col 1 ("Detector Signal") still shows raw intensity summed over *all* `T` masks
  (not split per country) — unchanged from before this feature existed.

## 2026-09-04 (cont'd) — missing checkpoint methods, save_images redesign (train.py, loss.py, config.py)

**Found on a second review pass, before the user's own two follow-up requests could
even be tested**: `main()` calls `trainer.save_slm_masks()` and
`trainer.save_layer_masks()` every `checkpoint_save` epochs, but neither method
existed anywhere on `TimeMultiplexedClassifierTrainer` — training would have crashed
with `AttributeError` at epoch 0. Missed in the first review pass (only traced
docstrings/data flow, didn't check every call site resolved to a real method). Added
both: `save_slm_masks()` lays out one row per country (row r = country r's M members,
matching `test.py`'s existing `_save_slm_masks` convention); `save_layer_masks()`
mirrors the old design (no country structure -- layers are shared hardware), no-op
when `num_layers == 0`.

Also confirmed by full trace (object phase + per-mask bias -> propagation -> 4x5
detection -> per-country integration over M -> per-country class scores -> sum ->
loss/pred) that gradients flow correctly end-to-end to both `slm_phases` and
`layer_phases`, and that `C=1` exactly reproduces the pre-country pipeline
numerically for any `M` (verified algebraically, not just by inspection).

User then adjusted `config.py` directly: `M=5` (was `M=10` from the previous entry).

**`save_images` redesign**, per user request:
1. Input-image colormap fixed to `vmin=0, vmax=input_phase_max` (was hardcoded
   `2*np.pi`, stretching the colormap over double the image's actual `[0, π]` range
   and making it look artificially dim) — reverses the earlier 2026-09-03 change
   where 0..2π had been explicitly requested.
2. Layout changed from 1 row x 5 cols to `(C+1)` rows x 4 cols: row `r` (0..C-1) now
   shows country `r`'s own detector signal / class-score / I_ccd panels (each
   computed from only that country's `M` members, not summed across all `T`), plus a
   repeated input-image panel for layout symmetry. Final row spans all 4 columns and
   shows the aggregated (summed-over-countries) class scores with the true class
   highlighted and the prediction marked -- same content the old col 2 showed, now
   promoted to its own row instead of being one column among five.
3. To support per-country raw intensities, `loss.py`'s `ClassificationLoss.scores()`
   now also returns `I_pos, I_neg` (`[B, C, half, cols]`, pre-ratio per-country
   detector intensities) alongside `aggregated_scores`/`per_country_scores` --
   `forward()` updated to unpack the 4-tuple and discard the two it doesn't need;
   no other call site touches `.scores()` directly so this was a safe signature change.

Note: the aggregate row's y-axis is intentionally *not* clamped to `[-1, 1]` like the
per-country panels are -- summing `C` independent `[-1,1]`-bounded contrasts can
reach `±C`, so that panel autoscales.
