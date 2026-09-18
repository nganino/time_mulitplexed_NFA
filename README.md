# time-multiplexed-NFA

**NFA = Nonlinear Function Approximation.** This repo optically approximates `Nf`
independent, arbitrary nonlinear functions `f_k(a)`, k = 1..Nf, of a single scalar
input `a`, all in parallel, using a diffractive optical processor -- following
Rahman et al., *"Massively parallel and universal approximation of nonlinear
functions using diffractive processors,"* eLight (2025) 5:32 (DOI
10.1186/s43593-025-00113-w) -- **plus one addition of our own**: a learned,
time-multiplexed **phase-key plane** placed before the input encoding. The idea is
"wisdom of the crowd": each of `M` phase keys gives one independent (imperfect)
optical estimate of `f(a)`; averaging the `M` raw detector intensities before
normalizing is meant to reduce approximation error for a *fixed* set of `Nf`
functions. This is a different goal from the paper's own time-multiplexing (which
uses multiple wavelengths to *increase the number* of functions computed in
parallel, not to improve accuracy on a fixed set).

This project was pivoted (Sep 2026, in a single working session) from an earlier,
structurally similar codebase that did time-multiplexed **image classification**
(MNIST/CIFAR phase objects -> differential photodiode-pair contrast -> softmax
cross-entropy). That pivot is essentially complete as of this README (config,
model, loss, dataloader, train, test all rewritten), but see **"Known open
items"** at the bottom -- there is unresolved training instability, some
paper-guideline values that were adopted as defaults without dedicated tuning, and
a couple of pre-existing-but-still-inaccurate leftover comments to be aware of.

## Physical architecture

```
Phase-key plane (LEARNED, T == M keys, time-multiplexed)
   phi_key ~ sigmoid(slm_phases) * 2*pi          [T, 1, slm_x_num, slm_x_num]
        |
        | propagate across key_to_enc_spacing  (FreeSpaceProp)
        v
Function-input encoding plane (DETERMINISTIC, NOT learned)
   phi_in(p; a) = 2*pi * encoding_freq_step * (p-1) * a
   for p = 1..Np, arranged as an encoding_side x encoding_side square patch
   (encoding_side = sqrt(Np)); field there = exp(i*phi_key_propagated) * exp(i*phi_in(p;a))
        |
        | propagate across slm_first_layer_spacing
        v
K learned diffractive layers (shared across all M keys)
   phi_layer_k ~ sigmoid(layer_phases[k]) * 2*pi   [1, 1, layer_size, layer_size]
   interleaved with propagation across interlayer_spacing
        |
        | propagate across last_layer_ccd_spacing
        v
Detector plane: pd_num_rows x pd_num_cols intensity detectors
   ONE detector per target function: detector (r,c) -> function k = r*pd_num_cols + c
   pd_num_rows * pd_num_cols MUST equal Nf (asserted in config.recompute_derived)
        |
        | (everything above is model.py; everything below is loss.py)
        v
Average raw intensity over the M phase keys -> [B, Nf]
        |
        v
Learned affine readout: f_hat = I_summed * scale + bias (scale/bias are
   nn.Parameters, trained by backprop like everything else -- see "Known
   open items" for why this replaced an EMA-tracked running Pmin/Pmax)
        |
        v
f_hat(a)  in [0,1]^Nf (approximately; not hard-clamped) <-- compared to the target f(a) via MSE
```

Both `object_slm_spacing`-style planes from the old classifier (a learned SLM mask,
then a separate "object" image plane) map directly onto this: the phase-key plane
plays the SLM's old structural role (learned, propagated, then multiplicatively
combined with the next plane), and the encoding plane plays the object's old
structural role (deterministic content, multiplicatively combined) -- **but the
object's content used to be an image; now it's a formula in `a`.**

## Why this can't reuse the paper's efficient training trick

The paper's own training loss (its Eq. 11) never simulates a specific `a` at all --
it fits the diffractive stack's coherent point-spread function directly to each
target function's Fourier coefficients in closed form, because with nothing in
front of the encoding plane, the system is purely linear (coherent field in, coherent
field out) and matching coefficients guarantees the fit for *every* `a`
simultaneously. Our design breaks that shortcut in two ways: (1) the phase-key
plane sits *before* the encoding plane and gets diffraction-propagated onto it, so
its effect is a fixed-but-nontrivial per-pixel complex weighting that only becomes
useful with `M > 1` *different* keys; (2) the `M`-key combination happens by
averaging **intensities** (post-detection), which is a nonlinear (incoherent)
combination with no closed-form Fourier-coefficient fit. So training here is
standard forward-simulate-then-backprop (`loss.py`'s MSE against `target_functions`'
closed-form values), not the paper's shortcut. This is a deliberate, understood
trade-off, not an oversight.

## Terminology / glossary

| term | meaning |
|---|---|
| `a` | scalar input to the target functions, sampled/gridded over `[a_min, a_max]` = `[-0.5, 0.5]` |
| `Np` | number of input-encoding pixels (paper default 9, a 3x3 patch); must be a perfect square |
| `Nf` | number of target functions approximated in parallel = number of detectors |
| `M` | number of learned phase keys (time-multiplexed, our addition, not in the paper) |
| `K` / `num_layers` | number of learned diffractive surfaces |
| `T` | total learnable phase-key masks; `T == M` now (see "C is gone" below) |
| phase key | one learned mask on the plane *before* the encoding plane; there are `M` of them |
| encoding plane | deterministic plane carrying `phi_in(p;a)`; NOT learned, replaces the old "object" image plane |
| `f_k(a)` | the k-th true target function, closed-form via `target_functions.TargetFunctionSet` |
| `f_hat(a)` | the model's optically-simulated, key-averaged, normalized approximation of `f(a)` |
| "wisdom of the crowd" | averaging M independent (noisy/imperfect) key-conditioned estimates before normalizing, to reduce error |
| `C` / "countries" | **GONE.** Old classifier concept (independent majority-vote groups); has no meaning for regression. `config.C` was deleted; some code still reads it via `getattr(..., 1)` fallback (harmless, always resolves to 1) |

## Config parameter reference (`code/config.py`)

Every parameter is tagged inline in the file itself:
- `# PAPER: <value/section>` -- taken directly from the paper
- `# NOTE: not specified in paper` -- our own choice, paper doesn't fix this
- `# OBSOLETE` -- classification-era, left in place, safe to delete, not currently read (or only read via a harmless fallback)

Current defaults (paper-scale regime, adopted 2026-09-17):

| category | key params | default | source |
|---|---|---|---|
| optics | `wavelength` | 550 nm | PAPER Sec. 2.2 |
| optics | `pixel_pitch` | `0.55 * wavelength` ~= 302.5 nm | PAPER Sec. 2.2 (shared pitch for input px, output px, AND diffractive feature width) |
| sim grid | `sim_dx` | `= pixel_pitch` (bin factor 1 everywhere) | derived |
| sim grid | `N_sim` | 256 | our choice: ~7.5x guard band around the largest single-plane aperture (~34px), FFT-friendly power of 2 |
| targets | `Np` | 9 (3x3 patch) | PAPER Sec. 2.2, fixed across all their Nf sweeps |
| targets | `Nf` | 100 | PAPER Fig. 2's smallest tested value (they go up to 1e6) |
| targets | `a_min`, `a_max` | -0.5, 0.5 | PAPER |
| targets | `func_seed` | 0 | our reproducibility knob (paper only specifies the sampling distributions, Eq. 10) |
| phase key | `M` | 1 | our addition, not paper-derived; **the whole point of this project is to sweep this** |
| layers | `num_layers` (K) | 2 | PAPER's default/main design (K=4 is their deeper alt.) |
| layers | `layer_size` | `ceil(sqrt(1.25*2*Np*Nf/K))` = 34 | PAPER's guideline `N ~= 1.25*2*Np*Nf` total features -- computed ONCE as a starting default in `init_params()`, NOT re-derived in `recompute_derived()`, so `--set layer_size=X` sticks |
| spacings | `key_to_enc_spacing`, `slm_first_layer_spacing`, `interlayer_spacing`, `last_layer_ccd_spacing` | all == `z = W*sqrt((2*pixel_pitch/wavelength)^2 - 1)` ~= 4.71 um, `W = layer_size*layer_dx` | PAPER uses ONE uniform value for every plane-to-plane gap (Sec. 2.2); we apply the same value to the key->encoding gap too even though it has no paper analogue |
| detector | `pd_num_rows`, `pd_num_cols` | 10, 10 (== Nf) | must satisfy `rows*cols == Nf`, asserted |
| detector | `photodiode_size` | `= pixel_pitch` | PAPER: detector width == delta |
| detector | `pd_row/col_spacing` | `photodiode_size + 0.5*wavelength` | PAPER: "inter-pixel spacing of ~0.5*lambda" |
| sampling | `train_a_samples` | 20000 | fixed pool, resampled/reshuffled each epoch (NOT regenerated -- drawn once, seeded by `config.seed`) |
| sampling | `val_a_grid_size`, `test_a_grid_size` | 1000, 1000 | dense EVENLY-SPACED grids (not random) -- approximates the paper's continuous RMSE integral (Eq. 16) with low variance and gives gap-free plots |
| loss | `loss_type` | `'mse'` | only `'mse'` is implemented |
| loss | `norm_momentum` | 0.1 | **OBSOLETE** -- was the EMA rate for the old running Pmin/Pmax buffers; `loss.py` now uses a learned scale/bias instead (see "Known open items") |
| training | `batch_size`, `max_epoch`, `lr_slm`, `lr_layer` | 12, 100, 1e-2, 1e-2 | `lr_slm` applies to the phase-key plane (name kept from the old SLM-mask era); also reused as the LR for `loss.py`'s learned scale/bias (`opt_readout` in `train.py`) |

`recompute_derived(config)` must be called after any `--set` override that changes
a base physical value (`train.py`/`test.py` already do this) -- it recomputes bin
factors, `encoding_side`, `T`, `run_name`/log paths, and asserts `Np` is a perfect
square and `pd_num_rows*pd_num_cols == Nf`.

## File-by-file reference

### `config.py`
Single source of truth for every parameter (see table above) plus `run_name`/
log-path construction. `init_params()` builds defaults; `recompute_derived(tc)`
recomputes everything that depends on base physical values (call this after
applying `--set` overrides). `config_to_dict(tc)` flattens to a JSON-serializable
dict for checkpoints/`config.json`.

### `target_functions.py`
`TargetFunctionSet(config)`: generates `Nf` fixed random Fourier-coefficient sets
(seeded by `config.func_seed`, PAPER Eq. 10) and evaluates `f_k(a)` in closed form
(no optical simulation) for any batch of `a`. Calibrates its own global min/max
(over a dense 2001-point grid) at construction time so `__call__(a)` always
returns values in `[0,1]`. This is the ONLY source of ground truth; it's shared
(same object) across train/val/test splits so they all see the same `Nf`
functions.

### `dataloader.py`
`FunctionApproxDataset(a_values, target_fn)`: thin wrapper, precomputes targets
once at construction. `get_function_approx_dataloaders(config)` returns
`(train_loader, val_loader, test_loader, target_fn)` -- train is a fixed random
pool (`train_a_samples`, seeded by `config.seed`, reshuffled every epoch by
`shuffle=True`); val/test are dense `torch.linspace` grids (`shuffle=False`,
`drop_last=False` so the full grid is always covered, no gaps).
Old image-dataset classes (MNIST/FashionMNIST/CIFAR10/TinyImageNet/grating) have
been deleted from this file; a few now-unused imports (`os`, `random`, `Path`,
`F`, `np`, `torchvision`, `transforms`) are leftover clutter, harmless.

### `model.py`
`TimeMultiplexedNFA(config)` (renamed from `TimeMultiplexedClassifier`):
implements exactly the physical path in the diagram above. Key methods:
- `_encode_input(a)`: builds `phi_in(p;a)` for a batch `a : [B]`, returns
  `[B, 1, N_sim, N_sim]` (deterministic, no learnable parameters, zero outside the
  Np-pixel patch)
- `_get_slm_field()`: the learned phase-key field (still named "slm" internally --
  same physical SLM hardware, new logical role; NOT renamed to avoid unnecessary
  attribute churn)
- `_get_layer_phase(k)`: the k-th learned diffractive layer's field
- `forward(a, return_field=False) -> I_vec [B, T, pd_num_rows, pd_num_cols]` (or
  `(I_vec, I_ccd)` if `return_field=True`): the ONLY forward path now -- there is
  no more "no diffractive layers" bypass (`num_layers > 0` is asserted in
  `__init__`; `measure()` and `diffraction_efficiency()` were deleted along with
  it, since both existed solely to support that bypass and neither is called
  anywhere else)

**model.py returns the RAWEST per-detector intensity. No averaging over M, no
normalization, no target comparison happens here -- that's all loss.py's job**
(explicit design decision).

### `loss.py`
`FunctionApproxLoss(config)`, an `nn.Module` with two of its own learnable
parameters (`_raw_scale`, `bias` -- a global affine readout, see "Known open
items" below for why). Old `ClassificationLoss` has been deleted (was already
broken -- referenced config attributes the user had removed).
- `forward(I_vec, target) -> (loss, f_hat)`: averages `I_vec` over the T/M axis
  (`I_vec.mean(dim=1)`, chosen over `sum` so the intensity scale stays
  M-invariant -- important since M is the primary thing this project sweeps),
  reshapes `[B, rows, cols] -> [B, Nf]` (row-major, `k = r*cols + c`), applies
  `f_hat = I_summed * scale + bias` (learned, `scale = softplus(_raw_scale)` to
  stay positive), then MSE against `target`
- `per_function_rmse(f_hat, target)` (`@staticmethod`): `[N, Nf] -> [Nf]`, the
  diagnostic/plotting metric (paper's Eq. 16-style), NOT the training loss
- `scale`/`bias` are trained by their own optimizer (`train.py`'s
  `opt_readout`/`sched_readout`, reusing `lr_slm`'s LR) and round-trip through
  checkpoints via `criterion.state_dict()`

### `train.py`
`TimeMultiplexedNFATrainer`, rewritten in place (old classification version not
kept side-by-side -- a single-entry-point script can't sensibly host two parallel
`main()`s). **wandb support removed entirely per explicit request -- tensorboard
only.** Key pieces:
- `_set_mode(training)`: toggles `.train()`/`.eval()` on `model` and `criterion`
  together (kept as good habit -- no longer functionally required now that
  `criterion` has no training-mode-dependent state, see loss.py note above)
- `_init_optimizers()`: THREE optimizer/scheduler pairs now --
  `opt_slm`/`sched_slm` (phase-key plane), `opt_layers`/`sched_layers`
  (diffractive layers, always present), `opt_readout`/`sched_readout`
  (loss.py's scale/bias, reuses `lr_slm`'s LR) -- `self.criterion` is
  constructed BEFORE `_init_optimizers()` is called (order matters:
  `opt_readout` needs `self.criterion.parameters()` to already exist)
- `train_step(a, target)`: one optimizer step across all three groups, returns
  `(loss, key_gnorm, layer_gnorm)`
- `evaluate(loader, tag='val', plot=True, n_show=4)`: single pass over a whole
  loader (typically a dense grid) in eval mode; returns `(loss,
  per_function_rmse)` and optionally saves a target-vs-approximation plot for 4
  representative functions (best-fit / worst-fit / two mid-error, paper Fig.
  2b/2c-style) -- this replaced the old per-batch `valid_step` AND the old
  classification `save_images` diagnostic (deleted, not applicable to regression)
- checkpoints (`_checkpoint_dict`/`save`/`save_best`/`load`) also save/restore
  `criterion.state_dict()` (the learned scale/bias) and `opt_readout`/
  `sched_readout` so a resumed run doesn't lose its calibration; tracks
  `best_val_loss` (lower is better), not `best_val_acc`
- `save_phase_keys()` (renamed from `save_slm_masks`): one row of M keys, no more
  country/member grid
- `save_layer_masks()`: unchanged in spirit, docstring updated

Usage: `python train.py --set M=5 lr_slm=5e-3` etc. (`--set KEY=VALUE`, any type,
preserved).

### `test.py`
Standalone eval script, also rewritten in place. `evaluate(ckpt_path, out_dir,
csv_path, sweep_name, run_label)`:
- `_eval_loop`: single pass over the (dense) test loader, returns `agg` dict with
  `a`/`f_hat`/`target` sorted ascending by `a`, plus per-batch `loss` list
- `_print_summary` / `_write_csv`: loss + per-function RMSE (mean/max/min) instead
  of accuracy/confusion-matrix/per-class-F1
- `_save_error_distribution`: histogram of per-function RMSE (paper Fig. 2a-style)
- `_save_function_curves`: target-vs-approx curves, same best/worst/mid-error
  selection as `train.py`'s `evaluate()`
- `_save_error_heatmap` (added 2026-09-18): spatial map of per-function RMSE laid
  out on the physical detector plane -- reshapes the `[Nf]` RMSE vector back into
  the `pd_num_rows x pd_num_cols` grid (`k = r*cols + c`, same indexing as
  `model.py`), positions it using the true `pd_row_spacing`/`pd_col_spacing`, and
  lets `imshow(..., interpolation='bilinear')` smooth between the discrete
  detectors purely for legibility (real detector centers are overlaid as dots so
  the interpolated fill is never mistaken for an actual measurement). Deliberately
  dependency-light: matplotlib's own `'hot'` colormap and bilinear resampling, no
  `scipy`/custom colormap.
- `_save_mask_similarity`: pairwise CIRCULAR phase similarity between the M keys
  (`|mean(exp(i*(phi_i - phi_j)))|`, invariant to a constant phase offset since
  that cancels under `|field|^2`) -- **low off-diagonal similarity means the keys
  are learning genuinely different things (good -- the ensembling is doing
  something); high similarity means the keys are redundant (the M-averaging isn't
  buying anything).** This is the most direct diagnostic for whether the whole
  phase-key idea is working.
- Dropped entirely: dataset selection (`_build_test_loader`, `--dataset` flag --
  there's no image dataset anymore, the test set is fully determined by config's
  `Np`/`Nf`/`func_seed`/`test_a_grid_size`), the legacy checkpoint-shape-repair
  logic (T inferred from `slm_phases.shape[0]` -- was a safety net for older
  classifier checkpoints, doesn't apply to the new format), the hardcoded stale
  classifier checkpoint path in `__main__` (now `None`, requires `--ckpt`)

Usage: `python test.py --ckpt logs/<run>/model/best.pth` (outputs default to
`logs/<run>/test/`; add `--csv path.csv --sweep NAME --label NAME` to append a
summary row for sweeps across multiple runs).

### `wave_prop.py`
`FreeSpaceProp(config, z=None)`: band-limited angular-spectrum free-space
propagator, unchanged by the NFA pivot (already fully general over
`N_sim`/`sim_dx`/`z`). `z=None` falls back to `config.z_slm_ccd`, which is now
OBSOLETE (nothing in the current codebase calls `FreeSpaceProp` without an
explicit `z`) -- harmless dead branch.

### `paths.py`
Unchanged. Resolves `REPO_ROOT`/`DATA_ROOT`/`CKPT_ROOT`/`LOG_DIR`, overridable via
`SPQPI_*` env vars. `LOG_DIR = code/logs/` (gitignored). No datasets are needed
anymore (no image dataset), so `DATA_ROOT` is effectively unused by this pivot,
but left as-is since other sibling projects may share it.

## Tensor shape cheat-sheet

```
a (dataloader)                     [B]
target (dataloader, TargetFunctionSet)   [B, Nf]           in [0,1]
model._encode_input(a)             [B, 1, N_sim, N_sim]    real phase (radians)
model._get_slm_field()             [T, 1, N_sim, N_sim]    real phase (radians)
model.forward(a) -> I_vec          [B, T, pd_num_rows, pd_num_cols]   raw intensity
loss._detector_to_function(I_vec)  [B, Nf]                  averaged over T, reshaped
loss.normalize(...)  -> f_hat      [B, Nf]                  ~[0,1], learned affine, NOT hard-clamped
loss.forward(...) -> (loss, f_hat) scalar, [B, Nf]
per_function_rmse(f_hat, target)   [Nf]
```

## Known open items (as of this README)

- ~~Training instability~~ **RESOLVED (2026-09-17).** A 60-epoch smoke run (M=5)
  had val loss improve then get noticeably worse. A 5-point LR sweep at M=1
  (`logs/lr_sweep_M1/`, 1e-2 down to 1e-4) showed the SAME qualitative failure
  at every LR (violent spikes at high LR -- val loss hit 146 at one point,
  impossible for a bounded-[0,1] MSE -- down to a slower but still-present
  late-training climb at 1e-4), which ruled out "LR too high" as the sole
  cause. Loading the saved checkpoints and inspecting the old
  `criterion.running_min`/`running_max` directly showed why: raw detector
  power collapsed by 3-6 orders of magnitude over training and swung by up to
  ~1000x between checkpoints just a few epochs apart (nothing in the loss
  constrained overall optical power), and the EMA-based running Pmin/Pmax
  normalizer (fixed momentum, out-of-loop) couldn't track those sudden swings,
  so its denominator was sometimes badly mismatched with the current scale --
  producing exactly this kind of blow-up.

  **Fix:** `loss.py`'s running Pmin/Pmax buffers were replaced with a LEARNED
  affine readout (`scale`/`bias`, trained by backprop via `train.py`'s new
  `opt_readout`, see loss.py's module docstring for the full writeup). A
  second, otherwise-identical LR sweep (`logs/lr_sweep_M1_learnedscale/`)
  confirmed the fix: every one of the 5 LRs now converges smoothly and
  monotonically over all 150 epochs -- zero spikes, train/val loss track each
  other closely throughout, and results now order sanely by LR (higher LR
  reaches a better minimum within the fixed epoch budget: lr=1e-2 final loss
  0.00206 vs the old EMA version's best-epoch loss of 0.0188, roughly a 9x
  improvement, with lr=1e-4 at the other end at 0.0173). Side-by-side plot:
  `logs/lr_sweep_before_after.png`.

  **Gotcha for future readers:** both of these LR sweeps (`logs/lr_sweep_M1/`
  and `logs/lr_sweep_M1_learnedscale/`) were actually run with
  `train_a_samples=512`, `batch_size=16` (32 batches/epoch) -- NOT the
  config.py defaults (20000 / 12). This wasn't a deliberate choice, it just
  wasn't noticed until the much-larger-scale sweep below made the mismatch
  obvious. Don't treat those two sweeps' absolute loss/RMSE numbers as
  representative of the real default sample budget -- check each run's own
  `config.json` before comparing across sweeps in this repo.
- **`train_a_samples` sweep (2026-09-18, M=1, lr=1e-2, `batch_size=64`, 150
  epochs, `logs/trainA_sweep_M1/`)**: 20000/30000/40000/50000 samples/epoch
  gave rmse_mean 0.00454 / 0.00328 / 0.00337 / 0.00282 respectively (plot:
  `logs/trainA_sweep_M1/trainA_sweep_rmse.png`). Two takeaways: (1) the big
  win is going from a badly-undersampled regime (the 512-sample LR sweeps
  above, rmse_mean 0.044) up to ~20k -- roughly 10x -- but (2) *within*
  20k-50k, further increases give small, noisy returns (max RMSE isn't even
  monotonic). `train_a_samples` looks like a mostly-exhausted lever now; best
  result so far (50k) is still ~4 orders of magnitude off the paper's
  reported ~1e-7 RMSE for Nf=100. See `code/logs/SWEEP_HANDOVER.txt` for the
  fuller experiment log and suggested next directions (deviating from the
  paper's spacing defaults, then sweeping `M`).
- **GPU utilization / `batch_size`**: the config default `batch_size=12` only
  reaches ~36-39% utilization on an RTX 4090 for this model size (kernel-launch
  overhead dominates at such a small batch) -- `batch_size=64` was found to
  give roughly 4.5x the throughput (~5700 vs ~1250 samples/s) and is now the
  batch size used for sweeps in this repo, though `config.py`'s own default is
  left at 12 (not changed without being asked). Worth re-benchmarking batch
  size again if the model's shape (`Np`/`Nf`/`layer_size`/`M`) changes a lot.
- **`layer_size`/`slm_x_num` are computed once from the paper's guideline
  formula and NOT re-derived in `recompute_derived()`** -- if you `--set Np=...`
  or `--set Nf=...`, you must manually recompute/re-set `layer_size` and
  `slm_x_num` yourself; they will NOT automatically track the new Np/Nf.
- **Diffraction-efficiency loss penalty (paper Eq. 13/14, `LDE`) was
  deliberately NOT implemented** -- explicit decision to keep the first version
  to plain MSE only; revisit once the core M-key accuracy idea is validated.
- **`M` default is 1** (no ensembling) -- sweeping M upward is the actual point
  of this project and hasn't been systematically explored yet (all sweeps so
  far, incl. the ones above, are still M=1 groundwork); `phase_key_similarity.png`
  (test.py) is the primary diagnostic for whether increasing M is actually
  buying diversity or just redundant keys.
- **`key_to_enc_spacing` sweep (2026-09-18, complete, M=1, 20k samples,
  `logs/keyspacing_sweep_M1/`)**: deliberately deviating from the paper's
  uniform-spacing choice for just this one gap (phase-key -> encoding plane,
  which has no paper analogue anyway), scaling the paper-derived default
  (~4.71 um) by 0.5x/1x/2x/3x/5x. Result: a clear non-monotonic dip at 2x
  (rmse_mean 0.00338, rmse_max 0.01089 -- best of the 5) with both smaller
  (0.5x/1x) and larger (3x/5x) spacings worse on both metrics -- looks like
  a real interior optimum, not a "more/less is always better" trend. Plot:
  `logs/keyspacing_sweep_M1/keyspacing_sweep_rmse.png`. See
  `code/logs/SWEEP_HANDOVER.txt` for the full numbers and the suggested
  narrower follow-up sweep (around 1.5x-2.5x).
- Config still carries a few small OBSOLETE-tagged leftovers, intentionally not
  deleted (tracked so you can remove them yourself): `config.z_slm_ccd`,
  `config.norm_momentum` (see the loss.py rewrite above), and a stale comment
  in `recompute_derived()` about `config.C` (the attribute itself is already
  gone; `model.py`'s `getattr(config, 'C', 1)` fallback makes this harmless).
- `dataloader.py` still imports `os`/`random`/`Path`/`F`/`np`/`torchvision`/
  `transforms`, all now unused (leftover from the deleted image-dataset code).

## Provenance

The original nine files in `code/` were copied from
[`ary-portes/single-pixel-qpi-paper`](https://github.com/ary-portes/single-pixel-qpi-paper)
(commit `f5b388c1d8191181444970c7c397b7249aa60fd9`, 29 Aug 2026), then pivoted to
image classification (Sep 2026), then pivoted again to this parallel nonlinear
function approximation design (17 Sep 2026) -- `config.py`, `model.py`,
`loss.py`, `dataloader.py`, `train.py`, `test.py` rewritten, `target_functions.py`
newly added, `wave_prop.py`/`paths.py` unchanged throughout both pivots.
